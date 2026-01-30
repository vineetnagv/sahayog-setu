"""
Sahayog Setu - Job Allocation Router
Enhanced allocation API endpoints with location + necessity-based scoring.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from app.database import get_db
from app.services.job_allocation_service import JobAllocationService
from app.schemas.job_allocation_schemas import (
    JobAllocationResponse,
    WorkerJobsResponse,
    BatchAllocationRequest,
    BatchAllocationResponse,
    JobPriorityScore
)

router = APIRouter()


# ============================================
# Worker Allocation for Jobs
# ============================================

@router.post("/allocate/job/{job_id}", response_model=JobAllocationResponse)
async def allocate_workers_for_job(
    job_id: int,
    top_n: int = Query(10, ge=1, le=50, description="Number of workers to return"),
    min_skill_score: float = Query(0, ge=0, le=100, description="Minimum skill match required"),
    db: Session = Depends(get_db)
):
    """
    **Enhanced Worker Allocation** for a specific job.
    
    This endpoint uses the enhanced allocation algorithm:
    
    **ALLOCATION FORMULA:**
    ```
    final_score = (geo_score × 0.3) + (need_score × 0.5) + (skill_score × 0.2)
    ```
    
    **Factors:**
    - **geo_score**: Proximity to job (closer = higher score)
    - **need_score**: Drought/distress level from VEDAS satellite data  
    - **skill_score**: Worker skill match with job requirements
    
    **Returns:**
    - Ranked list of workers with detailed scoring breakdown
    - Job necessity score indicating urgency
    - Skill match details for each worker
    """
    service = JobAllocationService(db)
    try:
        return await service.allocate_workers_for_job(
            job_id=job_id,
            top_n=top_n,
            min_skill_score=min_skill_score
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Allocation error: {str(e)}")


# ============================================
# Job Allocation for Workers
# ============================================

@router.post("/allocate/worker/{worker_id}", response_model=WorkerJobsResponse)
async def allocate_jobs_for_worker(
    worker_id: int,
    top_n: int = Query(10, ge=1, le=50, description="Number of jobs to return"),
    db: Session = Depends(get_db)
):
    """
    **Find Best Jobs for a Worker** (Reverse Allocation).
    
    This endpoint finds the most suitable jobs for a specific worker based on:
    
    **Scoring Formula:**
    ```
    final_score = (proximity × 0.3) + (skill_match × 0.4) + (job_necessity × 0.3)
    ```
    
    **Factors:**
    - **Proximity**: Distance from worker's village to job location
    - **Skill Match**: How well worker's skills match job requirements
    - **Job Necessity**: How urgently the job needs workers
    
    **Use Case:**
    - Mazdoor Mitra voice interface: "What jobs are near me?"
    - Worker dashboard: Show personalized job recommendations
    """
    service = JobAllocationService(db)
    try:
        return await service.allocate_jobs_for_worker(
            worker_id=worker_id,
            top_n=top_n
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Allocation error: {str(e)}")


# ============================================
# Batch Allocation
# ============================================

@router.post("/allocate/batch", response_model=BatchAllocationResponse)
async def batch_allocate_workers(
    request: BatchAllocationRequest,
    db: Session = Depends(get_db)
):
    """
    **Batch Allocation** - Allocate workers across multiple jobs optimally.
    
    This endpoint uses a greedy algorithm to distribute workers:
    
    1. **Sort jobs** by necessity (most urgent first)
    2. **For each job**: Find and assign best available workers
    3. **Avoid double-booking**: Workers assigned to one job are removed from pool
    
    **Use Case:**
    - Gram Sahayak dashboard: "Allocate workers for all open projects"
    - Morning allocation run: Distribute workers for the day
    
    **Request Body:**
    ```json
    {
        "job_ids": [1, 2, 3],
        "workers_per_job": 5
    }
    ```
    """
    service = JobAllocationService(db)
    try:
        return await service.batch_allocate(
            job_ids=request.job_ids,
            workers_per_job=request.workers_per_job
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch allocation error: {str(e)}")


# ============================================
# Job Priorities
# ============================================

@router.get("/allocate/job-priorities", response_model=List[JobPriorityScore])
async def get_job_priorities(
    limit: int = Query(20, ge=1, le=100, description="Number of jobs to return"),
    db: Session = Depends(get_db)
):
    """
    **Get Job Priority Rankings** - Which jobs need workers most?
    
    Returns a ranked list of open jobs scored by necessity:
    
    **Necessity Score Formula:**
    ```
    necessity = (urgency × 0.4) + (demand × 0.35) + (scarcity × 0.25)
    ```
    
    **Factors:**
    - **Urgency**: How soon does the job start? (sooner = higher)
    - **Demand**: How many workers are still needed?
    - **Scarcity**: Does the job require rare skills?
    
    **Use Case:**
    - Drishti Dashboard: Prioritize which jobs to focus on
    - Resource planning: Identify staffing gaps
    """
    service = JobAllocationService(db)
    try:
        return await service.get_job_priorities(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching priorities: {str(e)}")


# ============================================
# Legacy Compatibility (from original allocation.py)
# ============================================

@router.post("/allocate/{job_id}")
async def allocate_workers_legacy(
    job_id: int,
    db: Session = Depends(get_db)
):
    """
    **Legacy Allocation Endpoint** - Backward compatible.
    
    This wraps the new enhanced allocation for backward compatibility.
    Consider migrating to `/allocate/job/{job_id}` for enhanced features.
    """
    from app.services.allocation_service import AllocationService
    from app.schemas.schemas import AllocationResponse
    
    service = AllocationService(db)
    try:
        response = await service.allocate_workers_for_job(job_id)
        return response
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
