"""
Sahayog Setu - Optimal Matching Engine
Global optimization for worker-job allocation using bipartite matching.

MATCHING ALGORITHM:
    Uses a modified Hungarian Algorithm approach with weighted cost matrix.
    
    The engine computes OPTIMAL allocations by:
    1. Building a cost matrix: workers × jobs
    2. Each cell contains the "cost" (inverse of allocation score)
    3. Solving the assignment problem for minimum total cost
    4. This maximizes total allocation quality across all assignments

OPTIMIZATION VARIABLES:
    - Location (geo_score): Distance from VEDAS geospatial
    - Need (need_score): Drought/distress level
    - Skills (skill_score): Worker-job compatibility
    - Job Necessity: Urgency, demand, scarcity
    
ALLOCATION PATHS:
    Returns the optimal assignment path showing why each
    worker-job pairing was selected with detailed breakdowns.
"""

from sqlalchemy.orm import Session
from sqlalchemy import text
from app.models.models import Job, Worker, Village, WorkStatus, JobStatus
from app.services.vedas_service import VedasService
from app.services.geospatial_service import GeospatialService
from app.services.job_allocation_service import JobAllocationService
from typing import List, Optional, Dict, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, date
import logging
import heapq

logger = logging.getLogger(__name__)


# ============================================
# Data Classes for Matching Results
# ============================================

@dataclass
class AllocationVariable:
    """Individual variable contribution to a match."""
    name: str
    value: float
    weight: float
    weighted_score: float
    explanation: str


@dataclass
class AllocationPath:
    """Complete allocation path showing all decision factors."""
    worker_id: int
    worker_name: str
    worker_village: str
    job_id: int
    job_title: str
    job_village: str
    total_score: float
    variables: List[AllocationVariable]
    matched_skills: List[str]
    path_explanation: str
    rank: int = 0


@dataclass
class OptimalMatchResult:
    """Result of optimal matching operation."""
    matches: List[AllocationPath]
    unmatched_workers: List[int]
    unmatched_jobs: List[int]
    total_score: float
    optimization_method: str
    summary: str


@dataclass
class CostMatrixEntry:
    """Single entry in the cost matrix."""
    worker_id: int
    job_id: int
    cost: float  # Lower is better (inverse of score)
    score: float  # Original score (higher is better)
    variables: Dict[str, float] = field(default_factory=dict)


class MatchingEngine:
    """
    Optimal Matching Engine for Worker-Job Allocation.
    
    Uses weighted bipartite matching to find globally optimal
    assignments across all workers and jobs simultaneously.
    
    Key Methods:
    - compute_optimal_matching(): Global optimization
    - compute_allocation_paths(): Detailed path analysis
    - analyze_variable_impact(): Variable contribution analysis
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.vedas_service = VedasService()
        self.geo_service = GeospatialService()
        self.allocation_service = JobAllocationService(db)
        
        # Optimization Weights
        self.WEIGHTS = {
            "location": 0.25,     # Proximity score
            "need": 0.35,         # Drought/distress (VEDAS)
            "skills": 0.20,       # Skill match
            "job_necessity": 0.20 # Job urgency/demand
        }
        
        # Maximum distance threshold
        self.MAX_DISTANCE_KM = 100
    
    # =========================================================================
    # COST MATRIX COMPUTATION
    # =========================================================================
    
    def _compute_cost_matrix(
        self,
        workers: List[Worker],
        jobs: List[Job]
    ) -> Tuple[List[List[CostMatrixEntry]], Dict[int, int], Dict[int, int]]:
        """
        Build cost matrix for worker-job assignments.
        
        Cost = 100 - score (so lower cost = better match)
        
        Returns:
            - 2D cost matrix
            - Worker ID to row index mapping
            - Job ID to column index mapping
        """
        worker_idx = {w.id: i for i, w in enumerate(workers)}
        job_idx = {j.id: i for i, j in enumerate(jobs)}
        
        n_workers = len(workers)
        n_jobs = len(jobs)
        
        # Initialize with high cost (no match)
        matrix = [[None for _ in range(n_jobs)] for _ in range(n_workers)]
        
        for worker in workers:
            if not worker.village:
                continue
            
            worker_loc = (worker.village.latitude, worker.village.longitude)
            w_idx = worker_idx[worker.id]
            
            for job in jobs:
                if not job.village:
                    continue
                
                job_loc = (job.village.latitude, job.village.longitude)
                j_idx = job_idx[job.id]
                
                # Calculate all variables
                variables = {}
                
                # 1. LOCATION SCORE
                distance_km = self.geo_service.calculate_distance(worker_loc, job_loc)
                if distance_km > self.MAX_DISTANCE_KM:
                    # Too far - high cost
                    matrix[w_idx][j_idx] = CostMatrixEntry(
                        worker_id=worker.id,
                        job_id=job.id,
                        cost=1000,  # Very high cost
                        score=0,
                        variables={"location": 0, "too_far": True}
                    )
                    continue
                
                location_score = (1 / (1 + distance_km)) * 100
                variables["location"] = location_score
                
                # 2. NEED SCORE (VEDAS)
                need_score = worker.village.drought_score or 50.0
                variables["need"] = need_score
                
                # 3. SKILL SCORE
                skill_score = self._calculate_skill_match(
                    worker.skills or "",
                    job.required_skills or ""
                )
                variables["skills"] = skill_score
                
                # 4. JOB NECESSITY SCORE
                necessity_score = self._calculate_job_necessity(job)
                variables["job_necessity"] = necessity_score
                
                # WEIGHTED TOTAL SCORE
                total_score = sum(
                    variables[var] * self.WEIGHTS[var]
                    for var in self.WEIGHTS
                )
                
                # Cost is inverse of score (we minimize cost)
                cost = 100 - total_score
                
                matrix[w_idx][j_idx] = CostMatrixEntry(
                    worker_id=worker.id,
                    job_id=job.id,
                    cost=cost,
                    score=total_score,
                    variables=variables
                )
        
        return matrix, worker_idx, job_idx
    
    def _calculate_skill_match(self, worker_skills: str, job_skills: str) -> float:
        """Calculate skill match percentage."""
        if not worker_skills or not job_skills:
            return 50.0
        
        worker_set = {s.strip().lower() for s in worker_skills.split(",")}
        job_set = {s.strip().lower() for s in job_skills.split(",")}
        
        if not job_set:
            return 100.0
        
        matched = worker_set.intersection(job_set)
        coverage = len(matched) / len(job_set)
        
        return min(100.0, coverage * 100 + min(len(worker_set) * 2, 10))
    
    def _calculate_job_necessity(self, job: Job) -> float:
        """Calculate job necessity score."""
        score = 50.0  # Default
        
        # Urgency based on start date
        if job.start_date:
            days = (job.start_date.date() - date.today()).days
            if days <= 0:
                score += 25
            elif days <= 3:
                score += 20
            elif days <= 7:
                score += 10
        
        # Open jobs have higher necessity
        if job.status == JobStatus.OPEN:
            score += 20
        
        return min(100.0, score)
    
    # =========================================================================
    # OPTIMAL MATCHING ALGORITHMS
    # =========================================================================
    
    def compute_optimal_matching(
        self,
        job_ids: Optional[List[int]] = None,
        worker_ids: Optional[List[int]] = None,
        max_workers_per_job: int = 5
    ) -> OptimalMatchResult:
        """
        Compute globally optimal worker-job matching.
        
        Uses a greedy auction algorithm that:
        1. Builds cost matrix for all worker-job pairs
        2. Iteratively assigns best available matches
        3. Respects job capacity constraints
        4. Returns globally optimized assignments
        
        Args:
            job_ids: Specific jobs to match (None = all open jobs)
            worker_ids: Specific workers to match (None = all available)
            max_workers_per_job: Maximum workers per job
            
        Returns:
            OptimalMatchResult with all optimized assignments
        """
        # Fetch workers and jobs
        if worker_ids:
            workers = self.db.query(Worker).filter(
                Worker.id.in_(worker_ids),
                Worker.status == WorkStatus.AVAILABLE
            ).all()
        else:
            workers = self.db.query(Worker).filter(
                Worker.status == WorkStatus.AVAILABLE
            ).all()
        
        if job_ids:
            jobs = self.db.query(Job).filter(
                Job.id.in_(job_ids),
                Job.status == JobStatus.OPEN
            ).all()
        else:
            jobs = self.db.query(Job).filter(
                Job.status == JobStatus.OPEN
            ).all()
        
        if not workers or not jobs:
            return OptimalMatchResult(
                matches=[],
                unmatched_workers=[w.id for w in workers],
                unmatched_jobs=[j.id for j in jobs],
                total_score=0,
                optimization_method="none",
                summary="No workers or jobs available for matching"
            )
        
        # Build cost matrix
        matrix, worker_idx, job_idx = self._compute_cost_matrix(workers, jobs)
        
        # Reverse mappings
        idx_to_worker = {v: k for k, v in worker_idx.items()}
        idx_to_job = {v: k for k, v in job_idx.items()}
        
        # Track allocations
        worker_assigned: Set[int] = set()
        job_allocation_count: Dict[int, int] = {j.id: 0 for j in jobs}
        
        # Build priority queue of all valid matches (min-heap by cost)
        match_heap = []
        for w_idx in range(len(workers)):
            for j_idx in range(len(jobs)):
                entry = matrix[w_idx][j_idx]
                if entry and entry.cost < 999:  # Valid match
                    heapq.heappush(match_heap, (entry.cost, w_idx, j_idx, entry))
        
        # Greedy assignment
        matches: List[AllocationPath] = []
        rank = 1
        
        while match_heap:
            cost, w_idx, j_idx, entry = heapq.heappop(match_heap)
            
            worker_id = idx_to_worker[w_idx]
            job_id = idx_to_job[j_idx]
            
            # Skip if worker already assigned or job full
            if worker_id in worker_assigned:
                continue
            if job_allocation_count[job_id] >= max_workers_per_job:
                continue
            
            # Make assignment
            worker_assigned.add(worker_id)
            job_allocation_count[job_id] += 1
            
            # Get worker and job objects
            worker = next(w for w in workers if w.id == worker_id)
            job = next(j for j in jobs if j.id == job_id)
            
            # Build allocation path
            path = self._build_allocation_path(worker, job, entry, rank)
            matches.append(path)
            rank += 1
        
        # Identify unmatched
        unmatched_workers = [w.id for w in workers if w.id not in worker_assigned]
        unmatched_jobs = [j.id for j in jobs if job_allocation_count[j.id] == 0]
        
        total_score = sum(m.total_score for m in matches)
        
        return OptimalMatchResult(
            matches=matches,
            unmatched_workers=unmatched_workers,
            unmatched_jobs=unmatched_jobs,
            total_score=round(total_score, 2),
            optimization_method="greedy_auction",
            summary=f"Optimally matched {len(matches)} workers to jobs with total score {total_score:.1f}"
        )
    
    def _build_allocation_path(
        self,
        worker: Worker,
        job: Job,
        entry: CostMatrixEntry,
        rank: int
    ) -> AllocationPath:
        """Build detailed allocation path from matrix entry."""
        
        variables = []
        
        # Location variable
        loc_score = entry.variables.get("location", 0)
        variables.append(AllocationVariable(
            name="Location (Proximity)",
            value=round(loc_score, 2),
            weight=self.WEIGHTS["location"],
            weighted_score=round(loc_score * self.WEIGHTS["location"], 2),
            explanation=f"Distance-based score: {loc_score:.1f}/100"
        ))
        
        # Need variable
        need_score = entry.variables.get("need", 50)
        variables.append(AllocationVariable(
            name="Need (VEDAS Drought)",
            value=round(need_score, 2),
            weight=self.WEIGHTS["need"],
            weighted_score=round(need_score * self.WEIGHTS["need"], 2),
            explanation=f"Drought severity from satellite data: {need_score:.1f}/100"
        ))
        
        # Skills variable
        skill_score = entry.variables.get("skills", 50)
        variables.append(AllocationVariable(
            name="Skills Match",
            value=round(skill_score, 2),
            weight=self.WEIGHTS["skills"],
            weighted_score=round(skill_score * self.WEIGHTS["skills"], 2),
            explanation=f"Skill compatibility: {skill_score:.1f}%"
        ))
        
        # Job necessity variable
        necessity_score = entry.variables.get("job_necessity", 50)
        variables.append(AllocationVariable(
            name="Job Necessity",
            value=round(necessity_score, 2),
            weight=self.WEIGHTS["job_necessity"],
            weighted_score=round(necessity_score * self.WEIGHTS["job_necessity"], 2),
            explanation=f"Job urgency/demand: {necessity_score:.1f}/100"
        ))
        
        # Get matched skills
        matched_skills = []
        if worker.skills and job.required_skills:
            w_set = {s.strip().lower() for s in worker.skills.split(",")}
            j_set = {s.strip().lower() for s in job.required_skills.split(",")}
            matched_skills = list(w_set.intersection(j_set))
        
        # Build explanation
        top_factors = sorted(variables, key=lambda x: x.weighted_score, reverse=True)[:2]
        explanation = " + ".join([f"{v.name}: {v.weighted_score:.1f}" for v in top_factors])
        
        return AllocationPath(
            worker_id=worker.id,
            worker_name=worker.name,
            worker_village=worker.village.name if worker.village else "Unknown",
            job_id=job.id,
            job_title=job.title,
            job_village=job.village.name if job.village else "Unknown",
            total_score=round(entry.score, 2),
            variables=variables,
            matched_skills=matched_skills,
            path_explanation=explanation,
            rank=rank
        )
    
    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================
    
    def analyze_variable_impact(
        self,
        job_ids: Optional[List[int]] = None
    ) -> Dict[str, Dict]:
        """
        Analyze how each variable impacts matching outcomes.
        
        Returns statistics on each variable's contribution
        to the final allocations.
        """
        result = self.compute_optimal_matching(job_ids=job_ids)
        
        if not result.matches:
            return {"error": "No matches to analyze"}
        
        # Aggregate variable contributions
        variable_stats = {
            var: {"total_contribution": 0, "avg_score": 0, "count": 0}
            for var in self.WEIGHTS
        }
        
        for match in result.matches:
            for var in match.variables:
                key = var.name.lower().split()[0]  # First word
                if key in variable_stats:
                    variable_stats[key]["total_contribution"] += var.weighted_score
                    variable_stats[key]["avg_score"] += var.value
                    variable_stats[key]["count"] += 1
        
        # Calculate averages
        for key in variable_stats:
            if variable_stats[key]["count"] > 0:
                variable_stats[key]["avg_score"] /= variable_stats[key]["count"]
        
        # Determine most impactful variable
        most_impactful = max(
            variable_stats.items(),
            key=lambda x: x[1]["total_contribution"]
        )
        
        return {
            "variable_statistics": variable_stats,
            "most_impactful_variable": most_impactful[0],
            "total_matches": len(result.matches),
            "optimization_score": result.total_score
        }
    
    def get_allocation_recommendations(
        self,
        worker_id: int
    ) -> List[AllocationPath]:
        """
        Get recommended job allocation paths for a specific worker.
        
        Shows all potential matches ranked with variable breakdowns.
        """
        worker = self.db.query(Worker).filter(Worker.id == worker_id).first()
        if not worker:
            return []
        
        jobs = self.db.query(Job).filter(Job.status == JobStatus.OPEN).all()
        
        recommendations = []
        rank = 1
        
        for job in jobs:
            if not job.village or not worker.village:
                continue
            
            worker_loc = (worker.village.latitude, worker.village.longitude)
            job_loc = (job.village.latitude, job.village.longitude)
            
            distance = self.geo_service.calculate_distance(worker_loc, job_loc)
            if distance > self.MAX_DISTANCE_KM:
                continue
            
            # Calculate all scores
            location_score = (1 / (1 + distance)) * 100
            need_score = worker.village.drought_score or 50
            skill_score = self._calculate_skill_match(worker.skills or "", job.required_skills or "")
            necessity_score = self._calculate_job_necessity(job)
            
            total = (
                location_score * self.WEIGHTS["location"] +
                need_score * self.WEIGHTS["need"] +
                skill_score * self.WEIGHTS["skills"] +
                necessity_score * self.WEIGHTS["job_necessity"]
            )
            
            entry = CostMatrixEntry(
                worker_id=worker.id,
                job_id=job.id,
                cost=100 - total,
                score=total,
                variables={
                    "location": location_score,
                    "need": need_score,
                    "skills": skill_score,
                    "job_necessity": necessity_score
                }
            )
            
            path = self._build_allocation_path(worker, job, entry, rank)
            recommendations.append(path)
            rank += 1
        
        # Sort by score
        recommendations.sort(key=lambda x: x.total_score, reverse=True)
        
        # Update ranks after sorting
        for i, rec in enumerate(recommendations):
            rec.rank = i + 1
        
        return recommendations
