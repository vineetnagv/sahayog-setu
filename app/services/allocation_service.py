"""
Sahayog Setu - Allocation Service
Manages job acceptance, ledger recording, and transaction integrity.
"""

from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Dict, Any, Optional
import json
import uuid

from app.utils.hash_chain import create_ledger_entry

class AllocationService:
    def __init__(self, db: Session):
        self.db = db

    def allocate_job(
        self,
        worker_id: str,
        job_type: str, # 'GOVERNMENT' or 'PRIVATE'
        job_id: str
    ) -> Dict[str, Any]:
        """
        Execute a job allocation transaction.
        1. Verify availability
        2. Create Allocation Record
        3. Generate Hash Chain Entry
        4. Log to Audit Table
        5. Update Worker Status & Job Counts
        """
        
        # 1. Get latest hash from audit log for chaining
        last_entry = self.db.execute(
            text("SELECT hash FROM audit_log ORDER BY created_at DESC LIMIT 1")
        ).fetchone()
        
        prev_hash = last_entry.hash if last_entry else "0" * 64
        
        # 2. Get Worker Need Score (Snapshotted)
        worker = self.db.execute(
            text("SELECT need_score FROM workers WHERE id = :id"), 
            {"id": worker_id}
        ).fetchone() # Simplification for now
        
        worker_score = 0.0 # Default
        if worker and hasattr(worker, 'need_score'):
             worker_score = worker.need_score
             
        # 3. Create Allocation ID
        allocation_id = str(uuid.uuid4())
        
        # 4. Prepare Payload for Ledger
        payload = {
            "allocation_id": allocation_id,
            "worker_id": worker_id,
            "job_id": job_id,
            "job_type": job_type,
            "need_score": float(worker_score)
        }
        
        # 5. Generate Ledger Entry
        # Use a constant System UUID for the actor
        system_actor_id = "00000000-0000-0000-0000-000000000001"
        
        ledger_entry = create_ledger_entry(
            payload=payload,
            prev_hash=prev_hash,
            action="JOB_ALLOCATION",
            actor_id=system_actor_id
        )
        
        # 6. Insert into Audit Log
        self.db.execute(
            text("""
                INSERT INTO audit_log (
                    id, action, entity_type, entity_id, actor_type, actor_id,
                    payload, hash, prev_hash
                ) VALUES (
                    uuid_generate_v4(), :action, 'ALLOCATION', :entity_id, 'SYSTEM', :actor_id,
                    :payload, :hash, :prev_hash
                )
            """),
            {
                "action": ledger_entry['action'],
                "entity_id": allocation_id,
                "actor_id": ledger_entry['actor_id'],
                "payload": json.dumps(payload),
                "hash": ledger_entry['hash'],
                "prev_hash": ledger_entry['prev_hash']
            }
        )
        
        # 7. Insert into Job Allocations Table
        col_name = "government_job_id" if job_type == "GOVERNMENT" else "private_demand_id"
        
        self.db.execute(
            text(f"""
                INSERT INTO job_allocations (
                    id, worker_id, {col_name}, job_type, 
                    worker_need_score_at_allocation, daily_wage, work_date,
                    hash, prev_hash
                ) VALUES (
                    :id, :worker_id, :job_id, :job_type,
                    :need_score, 350.00, CURRENT_DATE,
                    :hash, :prev_hash
                )
            """),
            {
                "id": allocation_id,
                "worker_id": worker_id,
                "job_id": job_id,
                "job_type": job_type,
                "need_score": worker_score,
                "hash": ledger_entry['hash'],
                "prev_hash": ledger_entry['prev_hash']
            }
        )
        
        # 8. Update Worker Availability
        self.db.execute(
            text("UPDATE workers SET is_available = FALSE WHERE id = :id"),
            {"id": worker_id}
        )
        
        # 9. Update Job/Demand Counts & Fund Flow
        if job_type == "GOVERNMENT":
            self.db.execute(
                text("UPDATE government_jobs SET workers_allocated = workers_allocated + 1 WHERE id = :id"),
                {"id": job_id}
            )
            
            # Record Spending in Fund Flow (Phase 3)
            # Upsert into daily_spending
            self.db.execute(
                text("""
                    INSERT INTO daily_spending (
                        id, project_id, date, amount_spent, workers_count
                    ) VALUES (
                        uuid_generate_v4(), :project_id, CURRENT_DATE, :wage, 1
                    )
                    ON CONFLICT (project_id, date) 
                    DO UPDATE SET 
                        amount_spent = daily_spending.amount_spent + :wage,
                        workers_count = daily_spending.workers_count + 1,
                        updated_at = NOW()
                """),
                {
                    "project_id": job_id,
                    "wage": 350.00 # Standard wage for now
                }
            )
            
        else:
             self.db.execute(
                text("UPDATE private_demands SET workers_allocated = workers_allocated + 1 WHERE id = :id"),
                {"id": job_id}
            )
            
        self.db.commit()
        
        return {
            "status": "ALLOCATED",
            "allocation_id": allocation_id,
            "ledger_hash": ledger_entry['hash']
        }
