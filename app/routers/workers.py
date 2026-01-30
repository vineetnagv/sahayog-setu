"""
Sahayog Setu - Workers Router (Mazdoor Mitra Module)
API endpoints for worker registration, lookup, and management.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
import hashlib

from app.database import get_db

router = APIRouter()


# ============================================
# Pydantic Schemas
# ============================================

class WorkerCreate(BaseModel):
    """Schema for creating a new worker."""
    aadhaar_number: str = Field(..., min_length=12, max_length=12, description="12-digit Aadhaar number")
    name: str = Field(..., min_length=2, max_length=100)
    phone: str = Field(..., min_length=10, max_length=15)
    village_code: str = Field(..., min_length=1, max_length=20)
    dialect: str = Field(default="hindi", max_length=50)
    family_size: int = Field(default=1, ge=1)
    land_owned_acres: float = Field(default=0.0, ge=0)


class WorkerResponse(BaseModel):
    """Schema for worker response."""
    id: str
    name: str
    phone: str
    village_code: str
    dialect: str
    need_score: float
    is_available: bool
    family_size: int
    land_owned_acres: float
    days_since_last_work: int
    created_at: datetime


class WorkerUpdate(BaseModel):
    """Schema for updating worker details."""
    name: Optional[str] = None
    phone: Optional[str] = None
    dialect: Optional[str] = None
    family_size: Optional[int] = None
    is_available: Optional[bool] = None


# ============================================
# Helper Functions
# ============================================

def hash_aadhaar(aadhaar: str) -> str:
    """Hash Aadhaar number for privacy."""
    return hashlib.sha256(aadhaar.encode()).hexdigest()


def calculate_need_score(
    days_since_last_work: int,
    family_size: int,
    land_owned_acres: float
) -> float:
    """
    Calculate worker's need score (0-100).
    Higher score = more priority for job allocation.
    
    Factors:
    - Days since last work (max 40 points)
    - Family size (max 30 points)
    - Land ownership - less land = higher score (max 30 points)
    """
    # Days since last work: 0 days = 0 points, 30+ days = 40 points
    days_score = min(days_since_last_work / 30 * 40, 40)
    
    # Family size: 1 person = 5 points, 5+ people = 30 points
    family_score = min((family_size / 5) * 30, 30)
    
    # Land ownership: 0 acres = 30 points, 5+ acres = 0 points
    land_score = max(30 - (land_owned_acres / 5 * 30), 0)
    
    return round(days_score + family_score + land_score, 2)


# ============================================
# API Endpoints
# ============================================

@router.post("/workers", status_code=201)
async def create_worker(worker: WorkerCreate, db: Session = Depends(get_db)):
    """
    Register a new worker in the system.
    
    - Aadhaar is hashed for privacy
    - Initial need score is calculated
    - Worker is marked as available by default
    """
    # Hash the Aadhaar number
    aadhaar_hash = hash_aadhaar(worker.aadhaar_number)
    
    # Check if worker already exists
    existing = db.execute(
        text("SELECT id FROM workers WHERE aadhaar_hash = :hash"),
        {"hash": aadhaar_hash}
    ).fetchone()
    
    if existing:
        raise HTTPException(
            status_code=400,
            detail="Worker with this Aadhaar already registered"
        )
    
    # Calculate initial need score
    try:
        need_score = calculate_need_score(
            days_since_last_work=0,
            family_size=worker.family_size,
            land_owned_acres=worker.land_owned_acres
        )
        
        # Insert new worker
        worker_id = str(uuid.uuid4())
        db.execute(
            text("""
                INSERT INTO workers (
                    id, aadhaar_hash, name, phone, village_code, dialect,
                    need_score, is_available, family_size, land_owned_acres,
                    days_since_last_work
                ) VALUES (
                    :id, :aadhaar_hash, :name, :phone, :village_code, :dialect,
                    :need_score, TRUE, :family_size, :land_owned_acres, 0
                )
            """),
            {
                "id": worker_id,
                "aadhaar_hash": aadhaar_hash,
                "name": worker.name,
                "phone": worker.phone,
                "village_code": worker.village_code,
                "dialect": worker.dialect,
                "need_score": need_score,
                "family_size": worker.family_size,
                "land_owned_acres": worker.land_owned_acres
            }
        )
        db.commit()
        
        return {
            "message": "Worker registered successfully",
            "worker_id": worker_id,
            "need_score": need_score
        }
    except Exception as e:
        import traceback
        with open("error_log.txt", "w") as f:
            f.write(str(e))
            f.write("\n")
            f.write(traceback.format_exc())
            
        print(f"Error creating worker: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workers")
async def list_workers(
    village_code: Optional[str] = Query(None, description="Filter by village code"),
    available_only: bool = Query(True, description="Only show available workers"),
    min_need_score: float = Query(0, ge=0, le=100, description="Minimum need score"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    List workers with optional filters.
    
    - Filter by village for local job matching
    - Filter by availability
    - Sort by need score (highest first) for fair allocation
    """
    query = """
        SELECT id, name, phone, village_code, dialect, need_score,
               is_available, family_size, land_owned_acres,
               days_since_last_work, created_at
        FROM workers
        WHERE need_score >= :min_score
    """
    params = {"min_score": min_need_score}
    
    if village_code:
        query += " AND village_code = :village_code"
        params["village_code"] = village_code
    
    if available_only:
        query += " AND is_available = TRUE"
    
    query += " ORDER BY need_score DESC LIMIT :limit OFFSET :offset"
    params["limit"] = limit
    params["offset"] = offset
    
    result = db.execute(text(query), params)
    workers = [dict(row._mapping) for row in result]
    
    # Get total count
    count_query = "SELECT COUNT(*) FROM workers WHERE need_score >= :min_score"
    if village_code:
        count_query += " AND village_code = :village_code"
    if available_only:
        count_query += " AND is_available = TRUE"
    
    total = db.execute(text(count_query), params).scalar()
    
    return {
        "workers": workers,
        "count": len(workers),
        "total": total,
        "limit": limit,
        "offset": offset
    }


@router.get("/workers/{worker_id}")
async def get_worker(worker_id: str, db: Session = Depends(get_db)):
    """
    Get detailed information about a specific worker.
    """
    result = db.execute(
        text("""
            SELECT id, name, phone, village_code, dialect, need_score,
                   is_available, family_size, land_owned_acres,
                   days_since_last_work, created_at, updated_at
            FROM workers WHERE id = :id
        """),
        {"id": worker_id}
    )
    worker = result.fetchone()
    
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    
    return dict(worker._mapping)


@router.patch("/workers/{worker_id}")
async def update_worker(
    worker_id: str,
    update: WorkerUpdate,
    db: Session = Depends(get_db)
):
    """
    Update worker details.
    Only provided fields will be updated.
    """
    # Check worker exists
    existing = db.execute(
        text("SELECT id FROM workers WHERE id = :id"),
        {"id": worker_id}
    ).fetchone()
    
    if not existing:
        raise HTTPException(status_code=404, detail="Worker not found")
    
    # Build update query dynamically
    updates = []
    params = {"id": worker_id}
    
    if update.name is not None:
        updates.append("name = :name")
        params["name"] = update.name
    if update.phone is not None:
        updates.append("phone = :phone")
        params["phone"] = update.phone
    if update.dialect is not None:
        updates.append("dialect = :dialect")
        params["dialect"] = update.dialect
    if update.family_size is not None:
        updates.append("family_size = :family_size")
        params["family_size"] = update.family_size
    if update.is_available is not None:
        updates.append("is_available = :is_available")
        params["is_available"] = update.is_available
    
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    query = f"UPDATE workers SET {', '.join(updates)} WHERE id = :id"
    db.execute(text(query), params)
    db.commit()
    
    return {"message": "Worker updated successfully", "worker_id": worker_id}


@router.put("/workers/{worker_id}/availability")
async def toggle_availability(
    worker_id: str,
    available: bool,
    db: Session = Depends(get_db)
):
    """
    Toggle worker availability status.
    Used when worker starts/finishes a job.
    """
    result = db.execute(
        text("UPDATE workers SET is_available = :available WHERE id = :id RETURNING id"),
        {"id": worker_id, "available": available}
    )
    
    if not result.fetchone():
        raise HTTPException(status_code=404, detail="Worker not found")
    
    db.commit()
    
    status = "available" if available else "unavailable"
    return {"message": f"Worker marked as {status}", "worker_id": worker_id}


@router.get("/workers/village/{village_code}/available")
async def get_available_workers_in_village(
    village_code: str,
    db: Session = Depends(get_db)
):
    """
    Get all available workers in a village, sorted by need score.
    Used by the Matching Engine for job allocation.
    """
    result = db.execute(
        text("""
            SELECT * FROM available_workers_by_need
            WHERE village_code = :code
        """),
        {"code": village_code}
    )
    workers = [dict(row._mapping) for row in result]
    
    return {
        "village_code": village_code,
        "available_workers": workers,
        "count": len(workers)
    }
