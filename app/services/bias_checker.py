"""
Sahayog Setu - Bias Checker Service
Monitors job allocations to detect potential fairness violations.
"""

from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Dict, Any
import uuid

class BiasChecker:
    def __init__(self, db: Session):
        self.db = db

    def check_for_bias(self, allocation_id: str) -> List[Dict[str, Any]]:
        """
        Analyze a specific allocation for potential bias.
        
        Logic:
        1. Get the allocated worker's Need Score.
        2. Find any *available* workers in the same village who:
           - Were NOT allocated a job
           - Have a significantly HIGHER Need Score (> 10 points higher)
        3. If found, flag as a BIAS ALERT.
        """
        
        # 1. Get Allocation Details
        allocation = self.db.execute(
            text("""
                SELECT ja.*, w.village_code, w.need_score as allocated_score
                FROM job_allocations ja
                JOIN workers w ON ja.worker_id = w.id
                WHERE ja.id = :id
            """),
            {"id": allocation_id}
        ).fetchone()
        
        if not allocation:
            return []
            
        allocated_score = allocation.allocated_score
        village_code = allocation.village_code
        
        # 2. Find Skipped Workers with Higher Need Score
        # "Significantly higher" = 10+ points difference
        threshold_score = allocated_score + 10
        
        skipped_workers = self.db.execute(
            text("""
                SELECT * FROM workers
                WHERE village_code = :village
                AND is_available = TRUE -- They were available but skipped
                AND need_score >= :threshold
                ORDER BY need_score DESC
            """),
            {"village": village_code, "threshold": threshold_score}
        ).fetchall()
        
        alerts = []
        
        # 3. Generate Alerts
        for skipped in skipped_workers:
            # Check if alert already exists to prevent duplicates
            existing = self.db.execute(
                text("""
                    SELECT id FROM bias_alerts 
                    WHERE allocation_id = :alloc_id AND skipped_worker_id = :skipped_id
                """),
                {"alloc_id": allocation_id, "skipped_id": skipped.id}
            ).fetchone()
            
            if existing:
                continue
                
            # Calculate score difference
            score_diff = skipped.need_score - allocated_score
            
            # Insert Alert
            alert_id = str(uuid.uuid4())
            reason = f"Worker {skipped.name} (Score: {skipped.need_score}) was skipped for Worker ID {allocation.worker_id} (Score: {allocated_score}). Diff: {score_diff}"
            
            self.db.execute(
                text("""
                    INSERT INTO bias_alerts (
                        id, allocation_id, skipped_worker_id, allocated_worker_id,
                        skipped_need_score, allocated_need_score, score_difference,
                        alert_reason, status
                    ) VALUES (
                        :id, :alloc_id, :skipped_id, :allocated_id,
                        :skipped_score, :allocated_score, :diff,
                        :reason, 'PENDING'
                    )
                """),
                {
                    "id": alert_id,
                    "alloc_id": allocation_id,
                    "skipped_id": skipped.id,
                    "allocated_id": allocation.worker_id,
                    "skipped_score": skipped.need_score,
                    "allocated_score": allocated_score,
                    "diff": score_diff,
                    "reason": reason
                }
            )
            
            alerts.append({
                "alert_id": alert_id,
                "reason": reason,
                "score_difference": score_diff
            })
            
        self.db.commit()
        return alerts

    def run_daily_audit(self) -> Dict[str, Any]:
        """
        Run a batch check on all allocations from the last 24 hours.
        Useful for a nightly cron job.
        """
        recent_allocations = self.db.execute(
            text("""
                SELECT id FROM job_allocations
                WHERE allocated_at >= NOW() - INTERVAL '1 day'
            """)
        ).fetchall()
        
        total_alerts = 0
        
        for row in recent_allocations:
            alerts = self.check_for_bias(row.id)
            total_alerts += len(alerts)
            
        return {
            "allocations_checked": len(recent_allocations),
            "new_alerts_generated": total_alerts
        }
