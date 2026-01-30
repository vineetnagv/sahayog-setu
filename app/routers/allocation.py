
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.models import Job
from app.schemas.schemas import AllocationResponse
from app.services.allocation_service import AllocationService

router = APIRouter()

@router.post("/allocate/{job_id}", response_model=AllocationResponse)
async def allocate_workers(job_id: int, db: Session = Depends(get_db)):
    """
    **Run the Allocator Algorithm** for a specific job.
    
    This endpoint:
    1.  Calculates distance between Job and all Workers.
    2.  Fetches/Uses VEDAS Drought Score (NDVI/MNDWI) for Worker villages.
    3.  Scores workers based on `(0.4 * Distance) + (0.6 * Need)`.
    4.  Returns a ranked list of workers.
    """
    service = AllocationService(db)
    try:
        response = await service.allocate_workers_for_job(job_id)
        return response
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
