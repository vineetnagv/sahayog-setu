"""
Sahayog Setu - Matching Router
API endpoints to trigger job matching and allocation.
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.services.matching_engine import MatchingEngine
from app.services.allocation_service import AllocationService

router = APIRouter()

class AllocationRequest(BaseModel):
    worker_id: str
    job_type: str
    job_id: str

@router.get("/matching/worker/{worker_id}")
async def find_match(worker_id: str, db: Session = Depends(get_db)):
    """
    Find the best job match for a worker.
    Triggers 'Harvest Hero' logic if govt work is paused.
    """
    engine = MatchingEngine(db)
    return await engine.find_matches_for_worker(worker_id)

@router.post("/matching/allocate")
async def allocate_job(
    request: AllocationRequest,
    db: Session = Depends(get_db)
):
    """
    Accept and allocate a job.
    Records transaction on the Fairness Ledger.
    """
    service = AllocationService(db)
    try:
        result = service.allocate_job(
            worker_id=request.worker_id,
            job_type=request.job_type,
            job_id=request.job_id
        )
        
        # Trigger immediate bias check for demo purposes
        # In production, this might be async/background task
        from app.services.bias_checker import BiasChecker
        checker = BiasChecker(db)
        alerts = checker.check_for_bias(result["allocation_id"])
        
        if alerts:
            result["bias_alerts"] = alerts
            result["fairness_warning"] = "Potential bias detected in this allocation."
            
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
