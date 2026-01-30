"""
Sahayog Setu - Jobs Router
API endpoints for government jobs and work status tracking.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import date, datetime
from enum import Enum
import uuid

from app.database import get_db

router = APIRouter()


# ============================================
# Enums and Schemas
# ============================================

class JobStatus(str, Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"


class GovernmentJobCreate(BaseModel):
    """Schema for creating a government job/project."""
    project_code: str = Field(..., min_length=1, max_length=50)
    village_code: str = Field(..., min_length=1, max_length=20)
    panchayat_code: str = Field(..., min_length=1, max_length=20)
    title: str = Field(..., min_length=5, max_length=200)
    description: Optional[str] = None
    daily_wage: float = Field(default=350.00, ge=100)
    workers_needed: int = Field(default=10, ge=1)
    budget_allocated: float = Field(..., ge=0)
    start_date: date


class GovernmentJobUpdate(BaseModel):
    """Schema for updating a government job."""
    status: Optional[JobStatus] = None
    pause_reason: Optional[str] = None
    workers_needed: Optional[int] = None
    end_date: Optional[date] = None


# ============================================
# API Endpoints - Government Jobs
# ============================================

@router.post("/government/projects", status_code=201)
async def create_government_project(
    job: GovernmentJobCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new government work project (MGNREGA / VB-G RAM G).
    """
    # Check if project code already exists
    existing = db.execute(
        text("SELECT id FROM government_jobs WHERE project_code = :code"),
        {"code": job.project_code}
    ).fetchone()
    
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Project with code {job.project_code} already exists"
        )
    
    job_id = str(uuid.uuid4())
    db.execute(
        text("""
            INSERT INTO government_jobs (
                id, project_code, village_code, panchayat_code, title,
                description, daily_wage, workers_needed, budget_allocated,
                start_date, status
            ) VALUES (
                :id, :project_code, :village_code, :panchayat_code, :title,
                :description, :daily_wage, :workers_needed, :budget_allocated,
                :start_date, 'ACTIVE'
            )
        """),
        {
            "id": job_id,
            "project_code": job.project_code,
            "village_code": job.village_code,
            "panchayat_code": job.panchayat_code,
            "title": job.title,
            "description": job.description,
            "daily_wage": job.daily_wage,
            "workers_needed": job.workers_needed,
            "budget_allocated": job.budget_allocated,
            "start_date": job.start_date
        }
    )
    db.commit()
    
    return {
        "message": "Government project created successfully",
        "project_id": job_id,
        "project_code": job.project_code
    }


@router.get("/government/projects")
async def list_government_projects(
    village_code: Optional[str] = None,
    panchayat_code: Optional[str] = None,
    status: Optional[JobStatus] = None,
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """
    List government projects with optional filters.
    """
    query = """
        SELECT id, project_code, village_code, panchayat_code, title,
               status, daily_wage, workers_needed, workers_allocated,
               budget_allocated, budget_spent, start_date, end_date,
               pause_reason, created_at
        FROM government_jobs
        WHERE 1=1
    """
    params = {}
    
    if village_code:
        query += " AND village_code = :village_code"
        params["village_code"] = village_code
    
    if panchayat_code:
        query += " AND panchayat_code = :panchayat_code"
        params["panchayat_code"] = panchayat_code
    
    if status:
        query += " AND status = :status"
        params["status"] = status.value
    
    query += " ORDER BY created_at DESC LIMIT :limit"
    params["limit"] = limit
    
    result = db.execute(text(query), params)
    projects = [dict(row._mapping) for row in result]
    
    return {"projects": projects, "count": len(projects)}


@router.get("/government/projects/{project_id}")
async def get_government_project(project_id: str, db: Session = Depends(get_db)):
    """
    Get details of a specific government project.
    """
    result = db.execute(
        text("SELECT * FROM government_jobs WHERE id = :id"),
        {"id": project_id}
    )
    project = result.fetchone()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return dict(project._mapping)


@router.patch("/government/projects/{project_id}")
async def update_government_project(
    project_id: str,
    update: GovernmentJobUpdate,
    db: Session = Depends(get_db)
):
    """
    Update a government project (e.g., pause for harvest season).
    """
    # Check project exists
    existing = db.execute(
        text("SELECT id, status FROM government_jobs WHERE id = :id"),
        {"id": project_id}
    ).fetchone()
    
    if not existing:
        raise HTTPException(status_code=404, detail="Project not found")
    
    updates = []
    params = {"id": project_id}
    
    if update.status is not None:
        updates.append("status = :status")
        params["status"] = update.status.value
    
    if update.pause_reason is not None:
        updates.append("pause_reason = :pause_reason")
        params["pause_reason"] = update.pause_reason
    
    if update.workers_needed is not None:
        updates.append("workers_needed = :workers_needed")
        params["workers_needed"] = update.workers_needed
    
    if update.end_date is not None:
        updates.append("end_date = :end_date")
        params["end_date"] = update.end_date
    
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    query = f"UPDATE government_jobs SET {', '.join(updates)} WHERE id = :id"
    db.execute(text(query), params)
    db.commit()
    
    return {"message": "Project updated successfully", "project_id": project_id}


@router.put("/government/projects/{project_id}/status")
async def update_project_status(
    project_id: str,
    status: JobStatus,
    pause_reason: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Update project status (ACTIVE/PAUSED/COMPLETED).
    
    This is crucial for the "Harvest Hero" feature:
    - When status = PAUSED, workers are routed to private jobs
    """
    result = db.execute(
        text("""
            UPDATE government_jobs 
            SET status = :status, pause_reason = :reason
            WHERE id = :id
            RETURNING id, project_code
        """),
        {"id": project_id, "status": status.value, "reason": pause_reason}
    )
    project = result.fetchone()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    db.commit()
    
    return {
        "message": f"Project status updated to {status.value}",
        "project_id": project_id,
        "project_code": project.project_code
    }


# ============================================
# API Endpoints - Village Work Status
# ============================================

@router.get("/government/status/{village_code}")
async def get_village_work_status(village_code: str, db: Session = Depends(get_db)):
    """
    Check if government work is ACTIVE or PAUSED for a village.
    
    This is the KEY endpoint for the "Harvest Hero" feature:
    - If PAUSED → System should match workers with private jobs
    - If ACTIVE → Workers can get government work
    
    Used by the voice IVR system to respond appropriately.
    """
    # Use the view we created in the schema
    result = db.execute(
        text("SELECT * FROM village_work_status WHERE village_code = :code"),
        {"code": village_code}
    )
    status = result.fetchone()
    
    if not status:
        return {
            "village_code": village_code,
            "overall_status": "NO_PROJECTS",
            "active_projects": 0,
            "paused_projects": 0,
            "workers_needed": 0,
            "message": "No government projects found for this village",
            "harvest_hero_active": True  # Route to private jobs
        }
    
    status_dict = dict(status._mapping)
    
    # Add Harvest Hero flag
    status_dict["harvest_hero_active"] = status_dict["overall_status"] == "PAUSED"
    
    if status_dict["overall_status"] == "PAUSED":
        status_dict["message"] = "Government work is paused. Redirecting to private opportunities."
    else:
        status_dict["message"] = f"Government work is active. {status_dict['workers_needed']} positions available."
    
    return status_dict


@router.get("/government/paused-villages")
async def get_paused_villages(db: Session = Depends(get_db)):
    """
    Get all villages where government work is currently paused.
    Useful for bulk operations and dashboard views.
    """
    result = db.execute(
        text("""
            SELECT * FROM village_work_status 
            WHERE overall_status = 'PAUSED'
        """)
    )
    villages = [dict(row._mapping) for row in result]
    
    return {
        "paused_villages": villages,
        "count": len(villages),
        "message": f"{len(villages)} villages have paused government work"
    }
