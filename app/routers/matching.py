"""
Sahayog Setu - Matching Engine Router
API endpoints for optimal worker-job matching.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from app.database import get_db
from app.services.matching_engine import MatchingEngine, AllocationPath, AllocationVariable
from app.schemas.matching_schemas import (
    OptimalMatchRequest,
    OptimalMatchResponse,
    AllocationPathSchema,
    AllocationVariableSchema,
    VariableImpactResponse,
    WorkerRecommendationsResponse
)

router = APIRouter()


def _convert_path_to_schema(path: AllocationPath) -> AllocationPathSchema:
    """Convert internal AllocationPath to Pydantic schema."""
    return AllocationPathSchema(
        worker_id=path.worker_id,
        worker_name=path.worker_name,
        worker_village=path.worker_village,
        job_id=path.job_id,
        job_title=path.job_title,
        job_village=path.job_village,
        total_score=path.total_score,
        variables=[
            AllocationVariableSchema(
                name=v.name,
                value=v.value,
                weight=v.weight,
                weighted_score=v.weighted_score,
                explanation=v.explanation
            ) for v in path.variables
        ],
        matched_skills=path.matched_skills,
        path_explanation=path.path_explanation,
        rank=path.rank
    )


# ============================================
# Optimal Matching Endpoints
# ============================================

@router.post("/match/optimal", response_model=OptimalMatchResponse)
async def compute_optimal_matching(
    request: OptimalMatchRequest,
    db: Session = Depends(get_db)
):
    """
    **Compute Optimal Worker-Job Matching**
    
    Uses a greedy auction algorithm with weighted cost matrix to find
    the globally optimal assignment of workers to jobs.
    
    ## Algorithm
    
    1. **Build Cost Matrix**: Scores all worker-job pairs using:
       - Location (25%): Proximity from VEDAS geospatial
       - Need (35%): Drought/distress level
       - Skills (20%): Worker-job skill compatibility
       - Job Necessity (20%): Urgency and demand
    
    2. **Optimal Assignment**: Uses greedy auction to minimize total cost
       while respecting job capacity constraints.
    
    3. **Returns**: Complete allocation paths showing why each pairing
       was selected with detailed variable breakdowns.
    
    ## Request Body
    
    ```json
    {
        "job_ids": [1, 2, 3],      // Optional: specific jobs (null = all open)
        "worker_ids": [1, 2, 3],   // Optional: specific workers (null = all available)
        "max_workers_per_job": 5   // Max workers per job
    }
    ```
    """
    engine = MatchingEngine(db)
    try:
        result = engine.compute_optimal_matching(
            job_ids=request.job_ids,
            worker_ids=request.worker_ids,
            max_workers_per_job=request.max_workers_per_job
        )
        
        return OptimalMatchResponse(
            matches=[_convert_path_to_schema(m) for m in result.matches],
            unmatched_workers=result.unmatched_workers,
            unmatched_jobs=result.unmatched_jobs,
            total_score=result.total_score,
            optimization_method=result.optimization_method,
            summary=result.summary
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Matching error: {str(e)}")


@router.get("/match/worker/{worker_id}/recommendations", response_model=WorkerRecommendationsResponse)
async def get_worker_recommendations(
    worker_id: int,
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """
    **Get Optimal Job Recommendations for a Worker**
    
    Returns all potential job matches for a worker, ranked by score,
    with complete allocation path breakdowns showing each variable's
    contribution.
    
    ## Response includes:
    
    - **Ranked job recommendations**
    - **Variable breakdown** for each job:
      - Location score (proximity)
      - Need score (VEDAS drought data)
      - Skill match percentage
      - Job necessity (urgency)
    - **Matched skills** for each job
    - **Path explanation** summarizing top factors
    
    ## Use Cases:
    
    - Mazdoor Mitra: "What's the best job for me?"
    - Worker dashboard: Show personalized recommendations with explanations
    """
    engine = MatchingEngine(db)
    try:
        recommendations = engine.get_allocation_recommendations(worker_id)
        
        if not recommendations:
            from app.models.models import Worker
            worker = db.query(Worker).filter(Worker.id == worker_id).first()
            if not worker:
                raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")
            
            return WorkerRecommendationsResponse(
                worker_id=worker_id,
                worker_name=worker.name,
                recommendations=[],
                total_recommendations=0
            )
        
        return WorkerRecommendationsResponse(
            worker_id=worker_id,
            worker_name=recommendations[0].worker_name if recommendations else "Unknown",
            recommendations=[_convert_path_to_schema(r) for r in recommendations[:limit]],
            total_recommendations=len(recommendations)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")


@router.get("/match/analyze-variables", response_model=VariableImpactResponse)
async def analyze_variable_impact(
    job_ids: Optional[str] = Query(None, description="Comma-separated job IDs"),
    db: Session = Depends(get_db)
):
    """
    **Analyze Variable Impact on Matching**
    
    Runs the optimal matching algorithm and analyzes how each variable
    (location, need, skills, job necessity) contributed to the final
    allocations.
    
    ## Returns:
    
    - **Variable statistics**: Total contribution and average scores
    - **Most impactful variable**: Which factor drove most matches
    - **Optimization score**: Total quality of all matches
    
    ## Use Cases:
    
    - Drishti Dashboard: Understand what drives allocation decisions
    - Policy tuning: See if weights need adjustment
    - Fairness monitoring: Ensure need-based prioritization works
    """
    engine = MatchingEngine(db)
    try:
        parsed_job_ids = None
        if job_ids:
            parsed_job_ids = [int(x.strip()) for x in job_ids.split(",")]
        
        result = engine.analyze_variable_impact(job_ids=parsed_job_ids)
        
        if "error" in result:
            return VariableImpactResponse(
                variable_statistics={},
                most_impactful_variable="none",
                total_matches=0,
                optimization_score=0
            )
        
        return VariableImpactResponse(
            variable_statistics=result.get("variable_statistics", {}),
            most_impactful_variable=result.get("most_impactful_variable", "none"),
            total_matches=result.get("total_matches", 0),
            optimization_score=result.get("optimization_score", 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@router.get("/match/allocation-path/{worker_id}/{job_id}", response_model=AllocationPathSchema)
async def get_allocation_path(
    worker_id: int,
    job_id: int,
    db: Session = Depends(get_db)
):
    """
    **Get Detailed Allocation Path for Worker-Job Pair**
    
    Shows exactly how a specific worker-job match would be scored,
    with complete breakdown of all variables.
    
    ## Returns:
    
    - **Total score**: Combined weighted score
    - **Variable breakdown**: Each factor's contribution
    - **Matched skills**: Skills that overlap
    - **Path explanation**: Human-readable summary
    
    ## Use Case:
    
    - Explain to worker/supervisor why a match was made
    - Audit trail for fairness monitoring
    """
    engine = MatchingEngine(db)
    try:
        recommendations = engine.get_allocation_recommendations(worker_id)
        
        for rec in recommendations:
            if rec.job_id == job_id:
                return _convert_path_to_schema(rec)
        
        raise HTTPException(
            status_code=404,
            detail=f"No valid allocation path between worker {worker_id} and job {job_id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing path: {str(e)}")
