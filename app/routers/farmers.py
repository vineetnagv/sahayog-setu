"""
Sahayog Setu - Farmers & Private Demand Router
API endpoints for farmers and private job postings (Harvest Hero feature).
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional
from pydantic import BaseModel, Field
from datetime import date
from enum import Enum
import uuid

from app.database import get_db

router = APIRouter()


# ============================================
# Enums and Schemas
# ============================================

class DemandStatus(str, Enum):
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


class FarmerCreate(BaseModel):
    """Schema for registering a farmer."""
    name: str = Field(..., min_length=2, max_length=100)
    phone: str = Field(..., min_length=10, max_length=15)
    village_code: str = Field(..., min_length=1, max_length=20)
    land_area_acres: float = Field(..., ge=0)


class PrivateDemandCreate(BaseModel):
    """Schema for creating a private work demand."""
    farmer_id: str
    description: str = Field(..., min_length=10, max_length=500)
    work_type: str = Field(..., min_length=3, max_length=100)  # e.g., "harvesting", "sowing"
    workers_needed: int = Field(..., ge=1, le=100)
    daily_wage: float = Field(..., ge=100)
    work_date: date
    work_duration_days: int = Field(default=1, ge=1, le=30)
    location_gps: Optional[str] = None  # "lat,lng" format
    location_description: Optional[str] = None


class PrivateDemandUpdate(BaseModel):
    """Schema for updating a private demand."""
    workers_needed: Optional[int] = None
    daily_wage: Optional[float] = None
    status: Optional[DemandStatus] = None


# ============================================
# Farmer Endpoints
# ============================================

@router.post("/farmers", status_code=201)
async def create_farmer(farmer: FarmerCreate, db: Session = Depends(get_db)):
    """
    Register a new farmer who can post private work demands.
    """
    farmer_id = str(uuid.uuid4())
    
    db.execute(
        text("""
            INSERT INTO farmers (id, name, phone, village_code, land_area_acres)
            VALUES (:id, :name, :phone, :village_code, :land_area_acres)
        """),
        {
            "id": farmer_id,
            "name": farmer.name,
            "phone": farmer.phone,
            "village_code": farmer.village_code,
            "land_area_acres": farmer.land_area_acres
        }
    )
    db.commit()
    
    return {
        "message": "Farmer registered successfully",
        "farmer_id": farmer_id
    }


@router.get("/farmers")
async def list_farmers(
    village_code: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """
    List registered farmers.
    """
    query = "SELECT * FROM farmers WHERE 1=1"
    params = {}
    
    if village_code:
        query += " AND village_code = :village_code"
        params["village_code"] = village_code
    
    query += " ORDER BY created_at DESC LIMIT :limit"
    params["limit"] = limit
    
    result = db.execute(text(query), params)
    farmers = [dict(row._mapping) for row in result]
    
    return {"farmers": farmers, "count": len(farmers)}


@router.get("/farmers/{farmer_id}")
async def get_farmer(farmer_id: str, db: Session = Depends(get_db)):
    """
    Get farmer details including their active demands.
    """
    result = db.execute(
        text("SELECT * FROM farmers WHERE id = :id"),
        {"id": farmer_id}
    )
    farmer = result.fetchone()
    
    if not farmer:
        raise HTTPException(status_code=404, detail="Farmer not found")
    
    # Get active demands
    demands = db.execute(
        text("""
            SELECT * FROM private_demands 
            WHERE farmer_id = :id AND status = 'OPEN'
            ORDER BY work_date ASC
        """),
        {"id": farmer_id}
    )
    
    farmer_dict = dict(farmer._mapping)
    farmer_dict["active_demands"] = [dict(row._mapping) for row in demands]
    
    return farmer_dict


# ============================================
# Private Demand Endpoints (Harvest Hero)
# ============================================

@router.post("/private-demand", status_code=201)
async def create_private_demand(
    demand: PrivateDemandCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new private work demand (Harvest Hero posting).
    
    Farmers use this to post work requirements when government
    projects are paused and they need harvest help.
    """
    # Verify farmer exists
    farmer = db.execute(
        text("SELECT id, name, village_code FROM farmers WHERE id = :id"),
        {"id": demand.farmer_id}
    ).fetchone()
    
    if not farmer:
        raise HTTPException(status_code=404, detail="Farmer not found")
    
    demand_id = str(uuid.uuid4())
    
    db.execute(
        text("""
            INSERT INTO private_demands (
                id, farmer_id, description, work_type, workers_needed,
                daily_wage, work_date, work_duration_days,
                location_gps, location_description, status
            ) VALUES (
                :id, :farmer_id, :description, :work_type, :workers_needed,
                :daily_wage, :work_date, :work_duration_days,
                :location_gps, :location_description, 'OPEN'
            )
        """),
        {
            "id": demand_id,
            "farmer_id": demand.farmer_id,
            "description": demand.description,
            "work_type": demand.work_type,
            "workers_needed": demand.workers_needed,
            "daily_wage": demand.daily_wage,
            "work_date": demand.work_date,
            "work_duration_days": demand.work_duration_days,
            "location_gps": demand.location_gps,
            "location_description": demand.location_description
        }
    )
    db.commit()
    
    return {
        "message": "Private work demand created successfully",
        "demand_id": demand_id,
        "farmer_name": farmer.name,
        "village_code": farmer.village_code
    }


@router.get("/private-demand")
async def list_private_demands(
    village_code: Optional[str] = None,
    work_type: Optional[str] = None,
    status: DemandStatus = DemandStatus.OPEN,
    from_date: Optional[date] = None,
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """
    List private work demands (Harvest Hero opportunities).
    
    Default shows only OPEN demands for matching.
    """
    # Use the view for open demands
    if status == DemandStatus.OPEN and not work_type and not from_date:
        query = "SELECT * FROM open_private_demands"
        params = {}
        
        if village_code:
            query += " WHERE village_code = :village_code"
            params["village_code"] = village_code
        
        query += " LIMIT :limit"
        params["limit"] = limit
    else:
        # Custom query for other filters
        query = """
            SELECT pd.*, f.name AS farmer_name, f.phone AS farmer_phone, f.village_code
            FROM private_demands pd
            JOIN farmers f ON pd.farmer_id = f.id
            WHERE pd.status = :status
        """
        params = {"status": status.value}
        
        if village_code:
            query += " AND f.village_code = :village_code"
            params["village_code"] = village_code
        
        if work_type:
            query += " AND pd.work_type ILIKE :work_type"
            params["work_type"] = f"%{work_type}%"
        
        if from_date:
            query += " AND pd.work_date >= :from_date"
            params["from_date"] = from_date
        
        query += " ORDER BY pd.work_date ASC LIMIT :limit"
        params["limit"] = limit
    
    result = db.execute(text(query), params)
    demands = [dict(row._mapping) for row in result]
    
    return {
        "demands": demands,
        "count": len(demands),
        "status_filter": status.value
    }


@router.get("/private-demand/{demand_id}")
async def get_private_demand(demand_id: str, db: Session = Depends(get_db)):
    """
    Get details of a specific private demand.
    """
    result = db.execute(
        text("""
            SELECT pd.*, f.name AS farmer_name, f.phone AS farmer_phone, f.village_code
            FROM private_demands pd
            JOIN farmers f ON pd.farmer_id = f.id
            WHERE pd.id = :id
        """),
        {"id": demand_id}
    )
    demand = result.fetchone()
    
    if not demand:
        raise HTTPException(status_code=404, detail="Demand not found")
    
    demand_dict = dict(demand._mapping)
    demand_dict["workers_remaining"] = demand_dict["workers_needed"] - demand_dict["workers_allocated"]
    
    return demand_dict


@router.patch("/private-demand/{demand_id}")
async def update_private_demand(
    demand_id: str,
    update: PrivateDemandUpdate,
    db: Session = Depends(get_db)
):
    """
    Update a private demand.
    """
    existing = db.execute(
        text("SELECT id FROM private_demands WHERE id = :id"),
        {"id": demand_id}
    ).fetchone()
    
    if not existing:
        raise HTTPException(status_code=404, detail="Demand not found")
    
    updates = []
    params = {"id": demand_id}
    
    if update.workers_needed is not None:
        updates.append("workers_needed = :workers_needed")
        params["workers_needed"] = update.workers_needed
    
    if update.daily_wage is not None:
        updates.append("daily_wage = :daily_wage")
        params["daily_wage"] = update.daily_wage
    
    if update.status is not None:
        updates.append("status = :status")
        params["status"] = update.status.value
    
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    query = f"UPDATE private_demands SET {', '.join(updates)} WHERE id = :id"
    db.execute(text(query), params)
    db.commit()
    
    return {"message": "Demand updated successfully", "demand_id": demand_id}


@router.get("/private-demand/village/{village_code}/opportunities")
async def get_village_opportunities(village_code: str, db: Session = Depends(get_db)):
    """
    Get all open work opportunities in a village.
    
    This is used by the Matching Engine to find private jobs
    when government work is paused (Harvest Hero feature).
    """
    result = db.execute(
        text("""
            SELECT * FROM open_private_demands
            WHERE village_code = :code
            ORDER BY daily_wage DESC, work_date ASC
        """),
        {"code": village_code}
    )
    opportunities = [dict(row._mapping) for row in result]
    
    total_workers_needed = sum(
        (opp["workers_needed"] - opp["workers_allocated"]) 
        for opp in opportunities
    )
    
    return {
        "village_code": village_code,
        "opportunities": opportunities,
        "count": len(opportunities),
        "total_workers_needed": total_workers_needed,
        "message": f"{len(opportunities)} private work opportunities available"
    }
