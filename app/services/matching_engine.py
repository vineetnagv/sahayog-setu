"""
Sahayog Setu - Matching Engine
The core logic for "Harvest Hero" - finding work when government projects are paused.
"""

from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Dict, Any, List, Optional
from uuid import UUID

from app.services.need_score import calculate_need_score

class MatchingEngine:
    def __init__(self, db: Session):
        self.db = db

    async def find_matches_for_worker(self, worker_id: str) -> Dict[str, Any]:
        """
        Find best work opportunities for a worker.
        Logic:
        1. Check Government Work Status in worker's village.
        2. IF 'ACTIVE' -> Return Government Project.
        3. IF 'PAUSED' -> Search Private Demands (Harvest Hero).
        4. IF 'NO_PROJECTS' -> Search Private Demands.
        """
        # 1. Get Worker Details & Village
        worker = self.db.execute(
            text("SELECT * FROM workers WHERE id = :id"),
            {"id": worker_id}
        ).fetchone()
        
        if not worker:
            return {"status": "error", "message": "Worker not found"}
        
        worker_dict = dict(worker._mapping)
        village_code = worker_dict["village_code"]
        
        # 2. Check Government Work Status
        govt_status = self.db.execute(
            text("SELECT * FROM village_work_status WHERE village_code = :code"),
            {"code": village_code}
        ).fetchone()
        
        # Default to checking private if no status found or paused
        check_private = True
        govt_project = None
        
        if govt_status and govt_status.overall_status == 'ACTIVE':
            check_private = False
            # Find the specific active project
            govt_project = self.db.execute(
                text("""
                    SELECT * FROM government_jobs 
                    WHERE village_code = :code AND status = 'ACTIVE'
                    LIMIT 1
                """),
                {"code": village_code}
            ).fetchone()
            
        # 3. Return Government Job if available
        if govt_project:
            return {
                "match_type": "GOVERNMENT",
                "status": "MATCH_FOUND",
                "job": dict(govt_project._mapping),
                "message": "Government work is available under MGNREGA."
            }
            
        # 4. Harvest Hero: Search Private Demands
        if check_private:
            # Find matches in same village (view already contains farmer info)
            matches = self.db.execute(
                text("""
                    SELECT *
                    FROM open_private_demands
                    WHERE village_code = :code
                    ORDER BY daily_wage DESC
                """),
                {"code": village_code}
            ).fetchall()
            
            # If no local matches, expand search (mocked for now)
            # In Phase 4, we use GIS to find nearby villages
            
            if matches:
                 # Helper to serialze the result
                best_match = dict(matches[0]._mapping)
                return {
                    "match_type": "PRIVATE",
                    "status": "MATCH_FOUND",
                    "marketing_label": "Harvest Hero Opportunity",
                    "job": best_match,
                    "message": f"Government work paused. Found {len(matches)} private jobs. Best: ₹{best_match['daily_wage']}/day with {best_match['farmer_name']}."
                }
                
        return {
            "status": "NO_MATCH",
            "message": "No work available at this time."
        }
