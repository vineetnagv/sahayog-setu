"""
Sahayog Setu - Matching Engine Schemas
Pydantic models for the optimal matching engine API.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict


class AllocationVariableSchema(BaseModel):
    """Individual variable contribution to a match."""
    name: str
    value: float
    weight: float
    weighted_score: float
    explanation: str


class AllocationPathSchema(BaseModel):
    """Complete allocation path showing all decision factors."""
    worker_id: int
    worker_name: str
    worker_village: str
    job_id: int
    job_title: str
    job_village: str
    total_score: float
    variables: List[AllocationVariableSchema]
    matched_skills: List[str]
    path_explanation: str
    rank: int


class OptimalMatchRequest(BaseModel):
    """Request for optimal matching."""
    job_ids: Optional[List[int]] = None
    worker_ids: Optional[List[int]] = None
    max_workers_per_job: int = 5


class OptimalMatchResponse(BaseModel):
    """Response from optimal matching engine."""
    matches: List[AllocationPathSchema]
    unmatched_workers: List[int]
    unmatched_jobs: List[int]
    total_score: float
    optimization_method: str
    summary: str


class VariableImpactResponse(BaseModel):
    """Response from variable impact analysis."""
    variable_statistics: Dict[str, Dict]
    most_impactful_variable: str
    total_matches: int
    optimization_score: float


class WorkerRecommendationsResponse(BaseModel):
    """Job recommendations for a worker."""
    worker_id: int
    worker_name: str
    recommendations: List[AllocationPathSchema]
    total_recommendations: int
