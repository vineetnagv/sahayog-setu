"""
Sahayog Setu - Job Allocation Service
Enhanced allocation layer combining location + job necessity scoring.

ALLOCATION FORMULA:
    final_score = (geo_score × 0.3) + (need_score × 0.5) + (skill_score × 0.2)

FACTORS:
    - geo_score: Proximity to job (closer = higher score)
    - need_score: Drought/distress level from VEDAS satellite data
    - skill_score: Worker skill match with job requirements
    
JOB NECESSITY SCORING:
    - Urgency: Days until start date (sooner = higher priority)
    - Demand: Workers needed vs already allocated
    - Skill scarcity: Rare skills get priority matching
"""

from sqlalchemy.orm import Session
from sqlalchemy import text
from app.models.models import Job, Worker, Village, WorkStatus, JobStatus
from app.services.vedas_service import VedasService
from app.services.geospatial_service import GeospatialService
from app.schemas.job_allocation_schemas import (
    ScoredWorkerEnhanced,
    ScoredJobForWorker,
    JobAllocationResponse,
    WorkerJobsResponse,
    BatchAllocationResponse,
    BatchAllocationItem,
    JobPriorityScore
)
from typing import List, Optional, Dict, Tuple
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


class JobAllocationService:
    """
    Enhanced job allocation service with location + necessity-based scoring.
    
    Uses:
    - VEDAS geospatial data for drought/need scoring
    - Geodesic distance calculations for proximity
    - Skill matching for job-worker compatibility
    - Job necessity scoring for prioritization
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.vedas_service = VedasService()
        self.geo_service = GeospatialService()
        
        # Algorithm Weights (must sum to 1.0)
        self.WEIGHT_DISTANCE = 0.3   # Proximity matters (30%)
        self.WEIGHT_NEED = 0.5       # Need/Drought matters most (50%)
        self.WEIGHT_SKILLS = 0.2     # Skill match matters (20%)
        
        # Job Necessity Weights
        self.WEIGHT_URGENCY = 0.4    # Time-sensitive jobs (40%)
        self.WEIGHT_DEMAND = 0.35    # Worker demand ratio (35%)
        self.WEIGHT_SCARCITY = 0.25  # Skill scarcity (25%)
        
        # Maximum distance threshold (km) for workers
        self.MAX_DISTANCE_KM = 100
    
    # =========================================================================
    # CORE SCORING METHODS
    # =========================================================================
    
    def calculate_geo_score(self, distance_km: float) -> float:
        """
        Convert distance to a proximity score (0-100).
        
        Formula: geo_score = (1 / (1 + distance)) × 100
        
        Produces:
          - 0 km  -> 100 points
          - 1 km  -> 50 points
          - 9 km  -> 10 points
          - 99 km -> 1 point
        """
        if distance_km >= self.MAX_DISTANCE_KM:
            return 0.0
        return (1 / (1 + distance_km)) * 100
    
    def calculate_skill_score(self, worker_skills: str, job_requirements: str) -> float:
        """
        Calculate skill match score between worker and job (0-100).
        
        Parses comma-separated skill strings and calculates overlap percentage.
        
        Example:
            worker: "construction, masonry, welding"
            job: "construction, welding"
            -> skill_score = 100 (job fully covered)
            
            worker: "construction"
            job: "construction, welding, plumbing"
            -> skill_score = 33.33 (1/3 coverage)
        """
        if not worker_skills or not job_requirements:
            return 50.0  # Neutral score if no skill data
        
        # Parse skills (lowercase, stripped)
        worker_skill_set = {s.strip().lower() for s in worker_skills.split(",")}
        job_skill_set = {s.strip().lower() for s in job_requirements.split(",")}
        
        if not job_skill_set:
            return 100.0  # No requirements = perfect match
        
        # Calculate coverage percentage
        matched_skills = worker_skill_set.intersection(job_skill_set)
        coverage = len(matched_skills) / len(job_skill_set)
        
        # Bonus for having extra relevant skills
        extra_skills = len(worker_skill_set) - len(matched_skills)
        bonus = min(extra_skills * 2, 10)  # Max 10 bonus points
        
        return min(100.0, (coverage * 100) + bonus)
    
    def calculate_job_necessity_score(self, job: Job) -> Tuple[float, Dict]:
        """
        Calculate job necessity/priority score (0-100).
        
        Factors:
        1. Urgency: How soon does the job start?
        2. Demand: How many workers are still needed?
        3. Skill Scarcity: Are the required skills rare?
        
        Returns:
            Tuple of (score, breakdown_dict)
        """
        breakdown = {}
        
        # --- URGENCY SCORE (0-100) ---
        # Sooner start = higher urgency
        if job.start_date:
            days_until_start = (job.start_date.date() - date.today()).days
            if days_until_start <= 0:
                urgency_score = 100  # Already started or starting today
            elif days_until_start <= 3:
                urgency_score = 90
            elif days_until_start <= 7:
                urgency_score = 70
            elif days_until_start <= 14:
                urgency_score = 50
            elif days_until_start <= 30:
                urgency_score = 30
            else:
                urgency_score = 10
        else:
            urgency_score = 50  # Default if no start date
        
        breakdown["urgency"] = urgency_score
        
        # --- DEMAND SCORE (0-100) ---
        # More workers needed = higher demand
        # Using a simple scale based on open job status
        if job.status == JobStatus.OPEN:
            demand_score = 80  # Open jobs have high demand
        elif job.status == JobStatus.IN_PROGRESS:
            demand_score = 50  # In progress, moderate demand
        else:
            demand_score = 10  # Completed, low demand
        
        breakdown["demand"] = demand_score
        
        # --- SKILL SCARCITY SCORE (0-100) ---
        # Rare skill requirements = higher priority
        if job.required_skills:
            skills = [s.strip().lower() for s in job.required_skills.split(",")]
            # Check how many workers have these skills
            rare_skill_keywords = ["welding", "plumbing", "electrical", "driving", "masonry"]
            rare_count = sum(1 for s in skills if any(r in s for r in rare_skill_keywords))
            scarcity_score = min(100, 30 + (rare_count * 20))
        else:
            scarcity_score = 30  # No specific skills = lower scarcity
        
        breakdown["scarcity"] = scarcity_score
        
        # --- WEIGHTED FINAL SCORE ---
        final_score = (
            (urgency_score * self.WEIGHT_URGENCY) +
            (demand_score * self.WEIGHT_DEMAND) +
            (scarcity_score * self.WEIGHT_SCARCITY)
        )
        
        breakdown["final"] = round(final_score, 2)
        
        return final_score, breakdown
    
    # =========================================================================
    # MAIN ALLOCATION METHODS
    # =========================================================================
    
    async def allocate_workers_for_job(
        self, 
        job_id: int, 
        top_n: int = 10,
        min_skill_score: float = 0
    ) -> JobAllocationResponse:
        """
        Allocate best workers for a specific job.
        
        Enhanced algorithm with skill matching and location scoring.
        
        Args:
            job_id: ID of the job to allocate workers for
            top_n: Number of top workers to return
            min_skill_score: Minimum skill match required (0-100)
            
        Returns:
            JobAllocationResponse with ranked workers
        """
        # 1. Fetch Job and its location
        job = self.db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        job_village = job.village
        if not job_village:
            raise ValueError(f"Job {job_id} has no associated village")
            
        job_loc = (job_village.latitude, job_village.longitude)
        
        # 2. Calculate job necessity score
        job_necessity, necessity_breakdown = self.calculate_job_necessity_score(job)
        
        # 3. Fetch Available Workers
        candidates = self.db.query(Worker).join(Village).filter(
            Worker.status == WorkStatus.AVAILABLE
        ).all()
        
        scored_workers = []
        
        for worker in candidates:
            if not worker.village:
                continue
                
            worker_loc = (worker.village.latitude, worker.village.longitude)
            
            # --- FACTOR 1: DISTANCE ---
            distance_km = self.geo_service.calculate_distance(job_loc, worker_loc)
            
            # Skip workers too far away
            if distance_km > self.MAX_DISTANCE_KM:
                continue
            
            geo_score = self.calculate_geo_score(distance_km)
            
            # --- FACTOR 2: NEED (VEDAS DROUGHT SCORE) ---
            need_score = worker.village.drought_score or 50.0
            
            # --- FACTOR 3: SKILL MATCH ---
            skill_score = self.calculate_skill_score(
                worker.skills or "",
                job.required_skills or ""
            )
            
            # Skip if below minimum skill threshold
            if skill_score < min_skill_score:
                continue
            
            # --- FINAL WEIGHTED SCORE ---
            final_score = (
                (geo_score * self.WEIGHT_DISTANCE) +
                (need_score * self.WEIGHT_NEED) +
                (skill_score * self.WEIGHT_SKILLS)
            )
            
            scored_workers.append(ScoredWorkerEnhanced(
                worker_id=worker.id,
                worker_name=worker.name,
                village_name=worker.village.name,
                distance_km=round(distance_km, 2),
                geo_score=round(geo_score, 2),
                drought_score=round(need_score, 2),
                skill_score=round(skill_score, 2),
                final_score=round(final_score, 2),
                matched_skills=self._get_matched_skills(worker.skills, job.required_skills),
                match_reason=self._build_match_reason(distance_km, need_score, skill_score)
            ))
        
        # 4. Sort by Final Score (Descending)
        scored_workers.sort(key=lambda x: x.final_score, reverse=True)
        
        return JobAllocationResponse(
            job_id=job.id,
            job_title=job.title,
            job_location=job_village.name,
            job_necessity_score=round(job_necessity, 2),
            necessity_breakdown=necessity_breakdown,
            recommended_workers=scored_workers[:top_n],
            total_candidates=len(scored_workers)
        )
    
    async def allocate_jobs_for_worker(
        self,
        worker_id: int,
        top_n: int = 10,
        include_government: bool = True,
        include_private: bool = True
    ) -> WorkerJobsResponse:
        """
        Find best jobs for a specific worker.
        
        Reverse allocation - matches jobs to a worker based on:
        - Location (proximity to worker's village)
        - Skill match (worker's skills vs job requirements)
        - Job necessity (urgency and demand)
        
        Args:
            worker_id: ID of the worker
            top_n: Number of top jobs to return
            include_government: Include government jobs
            include_private: Include private/farmer jobs
            
        Returns:
            WorkerJobsResponse with ranked jobs
        """
        # 1. Fetch Worker
        worker = self.db.query(Worker).filter(Worker.id == worker_id).first()
        if not worker:
            raise ValueError(f"Worker {worker_id} not found")
        
        if not worker.village:
            raise ValueError(f"Worker {worker_id} has no associated village")
        
        worker_loc = (worker.village.latitude, worker.village.longitude)
        
        # 2. Fetch Open Jobs
        job_query = self.db.query(Job).join(Village).filter(
            Job.status == JobStatus.OPEN
        )
        
        jobs = job_query.all()
        scored_jobs = []
        
        for job in jobs:
            if not job.village:
                continue
            
            job_loc = (job.village.latitude, job.village.longitude)
            
            # --- FACTOR 1: DISTANCE ---
            distance_km = self.geo_service.calculate_distance(worker_loc, job_loc)
            
            if distance_km > self.MAX_DISTANCE_KM:
                continue
            
            geo_score = self.calculate_geo_score(distance_km)
            
            # --- FACTOR 2: SKILL MATCH ---
            skill_score = self.calculate_skill_score(
                worker.skills or "",
                job.required_skills or ""
            )
            
            # --- FACTOR 3: JOB NECESSITY ---
            necessity_score, necessity_breakdown = self.calculate_job_necessity_score(job)
            
            # --- COMBINED SCORE ---
            # For job matching, we weight differently:
            # - Distance still matters (worker wants close jobs)
            # - Skill match is important
            # - Job necessity shows which jobs need workers most
            combined_score = (
                (geo_score * 0.3) +
                (skill_score * 0.4) +
                (necessity_score * 0.3)
            )
            
            scored_jobs.append(ScoredJobForWorker(
                job_id=job.id,
                job_title=job.title,
                job_description=job.description,
                village_name=job.village.name,
                distance_km=round(distance_km, 2),
                geo_score=round(geo_score, 2),
                skill_score=round(skill_score, 2),
                necessity_score=round(necessity_score, 2),
                final_score=round(combined_score, 2),
                wage_per_day=job.wage_per_day,
                required_skills=job.required_skills,
                matched_skills=self._get_matched_skills(worker.skills, job.required_skills),
                match_reason=f"Distance: {distance_km:.1f}km, Skill match: {skill_score:.0f}%"
            ))
        
        # Sort by combined score
        scored_jobs.sort(key=lambda x: x.final_score, reverse=True)
        
        return WorkerJobsResponse(
            worker_id=worker.id,
            worker_name=worker.name,
            worker_village=worker.village.name,
            worker_skills=worker.skills,
            recommended_jobs=scored_jobs[:top_n],
            total_jobs_found=len(scored_jobs)
        )
    
    async def batch_allocate(
        self,
        job_ids: List[int],
        workers_per_job: int = 5
    ) -> BatchAllocationResponse:
        """
        Allocate workers across multiple jobs optimally.
        
        Uses a greedy algorithm to distribute workers:
        1. Score all worker-job combinations
        2. Assign best matches while avoiding double-booking workers
        3. Prioritize high-necessity jobs
        
        Args:
            job_ids: List of job IDs to allocate for
            workers_per_job: Max workers to allocate per job
            
        Returns:
            BatchAllocationResponse with allocations for each job
        """
        allocations: List[BatchAllocationItem] = []
        allocated_worker_ids: set = set()
        
        # First, get all jobs and sort by necessity
        jobs_with_scores = []
        for job_id in job_ids:
            job = self.db.query(Job).filter(Job.id == job_id).first()
            if job:
                necessity, breakdown = self.calculate_job_necessity_score(job)
                jobs_with_scores.append((job, necessity, breakdown))
        
        # Sort by necessity (highest first)
        jobs_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Allocate workers to each job
        for job, necessity, breakdown in jobs_with_scores:
            try:
                allocation_result = await self.allocate_workers_for_job(
                    job.id,
                    top_n=workers_per_job * 2  # Get extra candidates
                )
                
                # Filter out already-allocated workers
                available_workers = [
                    w for w in allocation_result.recommended_workers
                    if w.worker_id not in allocated_worker_ids
                ][:workers_per_job]
                
                # Mark these workers as allocated
                for w in available_workers:
                    allocated_worker_ids.add(w.worker_id)
                
                allocations.append(BatchAllocationItem(
                    job_id=job.id,
                    job_title=job.title,
                    job_location=allocation_result.job_location,
                    job_necessity_score=necessity,
                    allocated_workers=available_workers,
                    workers_requested=workers_per_job,
                    workers_allocated=len(available_workers)
                ))
                
            except Exception as e:
                logger.error(f"Error allocating for job {job.id}: {e}")
                allocations.append(BatchAllocationItem(
                    job_id=job.id,
                    job_title=job.title if job else "Unknown",
                    job_location="Unknown",
                    job_necessity_score=0,
                    allocated_workers=[],
                    workers_requested=workers_per_job,
                    workers_allocated=0
                ))
        
        total_allocated = sum(a.workers_allocated for a in allocations)
        
        return BatchAllocationResponse(
            allocations=allocations,
            total_jobs=len(allocations),
            total_workers_allocated=total_allocated,
            summary=f"Allocated {total_allocated} workers across {len(allocations)} jobs"
        )
    
    async def get_job_priorities(self, limit: int = 20) -> List[JobPriorityScore]:
        """
        Get a ranked list of open jobs by necessity/priority.
        
        Useful for dashboard views to see which jobs need workers most.
        
        Returns:
            List of JobPriorityScore sorted by priority (descending)
        """
        jobs = self.db.query(Job).filter(Job.status == JobStatus.OPEN).all()
        
        priorities = []
        for job in jobs:
            necessity, breakdown = self.calculate_job_necessity_score(job)
            
            priorities.append(JobPriorityScore(
                job_id=job.id,
                job_title=job.title,
                village_name=job.village.name if job.village else "Unknown",
                necessity_score=round(necessity, 2),
                urgency_score=breakdown.get("urgency", 0),
                demand_score=breakdown.get("demand", 0),
                scarcity_score=breakdown.get("scarcity", 0),
                required_skills=job.required_skills,
                status=job.status.value
            ))
        
        # Sort by necessity score
        priorities.sort(key=lambda x: x.necessity_score, reverse=True)
        
        return priorities[:limit]
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _get_matched_skills(self, worker_skills: str, job_skills: str) -> List[str]:
        """Get list of skills that match between worker and job."""
        if not worker_skills or not job_skills:
            return []
        
        worker_set = {s.strip().lower() for s in worker_skills.split(",")}
        job_set = {s.strip().lower() for s in job_skills.split(",")}
        
        return list(worker_set.intersection(job_set))
    
    def _build_match_reason(
        self, 
        distance_km: float, 
        need_score: float, 
        skill_score: float
    ) -> str:
        """Build human-readable match reason string."""
        reasons = []
        
        if distance_km < 10:
            reasons.append(f"Very close ({distance_km:.1f}km)")
        elif distance_km < 30:
            reasons.append(f"Nearby ({distance_km:.1f}km)")
        else:
            reasons.append(f"Distance: {distance_km:.1f}km")
        
        if need_score >= 70:
            reasons.append("High need area")
        elif need_score >= 40:
            reasons.append("Moderate need")
        
        if skill_score >= 80:
            reasons.append("Excellent skill match")
        elif skill_score >= 50:
            reasons.append("Good skill match")
        
        return " | ".join(reasons)
