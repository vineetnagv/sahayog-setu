
from sqlalchemy.orm import Session
from app.models.models import Job, Worker, Village, WorkStatus
from app.services.vedas_service import VedasService
from app.services.geospatial_service import GeospatialService
from app.schemas.schemas import ScoredWorker, AllocationResponse
import logging

logger = logging.getLogger(__name__)

class AllocationService:
    def __init__(self, db: Session):
        self.db = db
        self.vedas_service = VedasService()
        self.geo_service = GeospatialService()
        
        # Algorithm Weights
        self.WEIGHT_DISTANCE = 0.4  # Distance matters (40%)
        self.WEIGHT_NEED = 0.6      # Need/Poverty matters more (60%)

    async def allocate_workers_for_job(self, job_id: int) -> AllocationResponse:
        """
        The Master Algorithm for Work Allocation.
        """
        # 1. Fetch Job and its location
        job = self.db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise ValueError(f"Job {job_id} not found")
            
        job_village = job.village
        job_loc = (job_village.latitude, job_village.longitude)

        # 2. Fetch Available Workers
        start_q = self.db.query(Worker).join(Village).filter(Worker.status == WorkStatus.AVAILABLE)
        
        # Optimization: Don't fetch workers from > 50km away (Pre-filter if DB supports geospatial, here we do python side)
        candidates = start_q.all()
        
        scored_workers = []

        for worker in candidates:
            worker_loc = (worker.village.latitude, worker.village.longitude)
            
            # --- FACTOR 1: DISTANCE ---
            distance_km = self.geo_service.calculate_distance(activity_loc=job_loc, worker_loc=worker_loc)
            
            # Normalize Distance Score (Closer is better)
            # 0km = 100 score, 100km = 0 score.
            # Formula: max(0, 100 - distance) ? Too linear.
            # Using Inverse: 1 / (1 + distance) * 100
            geo_score = (1 / (1 + distance_km)) * 100
            
            # --- FACTOR 2: NEED / PRIORITY (VEDAS) ---
            # We use the cached drought score from the Village model.
            # (Assuming a background job or seeder updates this via `vedas_service`)
            # If we want real-time, we await self.vedas_service.calculate_drought_score(...)
            # For performance, we use the stored value.
            need_score = worker.village.drought_score
            
            # --- FINAL WEIGHTED SCORE ---
            final_score = (geo_score * self.WEIGHT_DISTANCE) + (need_score * self.WEIGHT_NEED)
            
            scored_workers.append(ScoredWorker(
                worker_id=worker.id,
                worker_name=worker.name,
                village_name=worker.village.name,
                distance_km=round(distance_km, 2),
                drought_score=round(need_score, 2),
                final_score=round(final_score, 2),
                match_reason=f"Dist: {round(distance_km,1)}km, Need: {round(need_score,1)}/100"
            ))

        # 3. Sort by Final Score (Descending)
        scored_workers.sort(key=lambda x: x.final_score, reverse=True)
        
        return AllocationResponse(
            job_id=job.id,
            job_location=job_village.name,
            recommended_workers=scored_workers
        )
