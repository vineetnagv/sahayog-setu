"""
Sahayog Setu - Job Allocation Schemas
Pydantic models for the enhanced job allocation layer.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict


# ============================================
# Enhanced Worker Scoring
# ============================================

class ScoredWorkerEnhanced(BaseModel):
    """Worker with enhanced allocation scoring including skill match."""
    worker_id: int
    worker_name: str
    village_name: str
    distance_km: float
    geo_score: float          # 0-100, proximity score
    drought_score: float      # 0-100, need score from VEDAS
    skill_score: float        # 0-100, skill match percentage
    final_score: float        # Weighted combination
    matched_skills: List[str] # Skills that matched with job
    match_reason: str         # Human-readable explanation


class JobAllocationResponse(BaseModel):
    """Response for allocating workers to a job."""
    job_id: int
    job_title: str
    job_location: str
    job_necessity_score: float
    necessity_breakdown: Dict[str, float]
    recommended_workers: List[ScoredWorkerEnhanced]
    total_candidates: int


# ============================================
# Job Scoring for Workers
# ============================================

class ScoredJobForWorker(BaseModel):
    """Job scored for a specific worker."""
    job_id: int
    job_title: str
    job_description: Optional[str]
    village_name: str
    distance_km: float
    geo_score: float          # Proximity score (0-100)
    skill_score: float        # Skill match (0-100)
    necessity_score: float    # Job's need for workers (0-100)
    final_score: float        # Combined score
    wage_per_day: Optional[float]
    required_skills: Optional[str]
    matched_skills: List[str]
    match_reason: str


class WorkerJobsResponse(BaseModel):
    """Response for finding jobs for a worker."""
    worker_id: int
    worker_name: str
    worker_village: str
    worker_skills: Optional[str]
    recommended_jobs: List[ScoredJobForWorker]
    total_jobs_found: int


# ============================================
# Batch Allocation
# ============================================

class BatchAllocationItem(BaseModel):
    """Single job allocation within a batch."""
    job_id: int
    job_title: str
    job_location: str
    job_necessity_score: float
    allocated_workers: List[ScoredWorkerEnhanced]
    workers_requested: int
    workers_allocated: int


class BatchAllocationRequest(BaseModel):
    """Request for batch allocation."""
    job_ids: List[int]
    workers_per_job: int = 5


class BatchAllocationResponse(BaseModel):
    """Response for batch allocation across multiple jobs."""
    allocations: List[BatchAllocationItem]
    total_jobs: int
    total_workers_allocated: int
    summary: str


# ============================================
# Job Priority / Necessity
# ============================================

class JobPriorityScore(BaseModel):
    """Job priority score breakdown."""
    job_id: int
    job_title: str
    village_name: str
    necessity_score: float     # Final necessity score (0-100)
    urgency_score: float       # Time-based urgency
    demand_score: float        # Worker demand
    scarcity_score: float      # Skill scarcity 
    required_skills: Optional[str]
    status: str
