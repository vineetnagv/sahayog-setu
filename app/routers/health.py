"""
Sahayog Setu - Health Check Router
Provides API and database health status endpoints.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime

from app.database import get_db
from app.config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """
    Comprehensive health check for API and database.
    
    Returns:
        - API status
        - Database connection status
        - Current timestamp
        - Environment info
    """
    # Check database connection
    try:
        result = db.execute(text("SELECT 1"))
        result.fetchone()
        db_status = "healthy"
        db_message = "Connected to Supabase PostgreSQL"
    except Exception as e:
        db_status = "unhealthy"
        db_message = str(e)
    
    return {
        "status": "ok" if db_status == "healthy" else "degraded",
        "service": "sahayog-setu-api",
        "version": settings.app_version,
        "environment": settings.app_env,
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {
            "database": {
                "status": db_status,
                "message": db_message
            }
        }
    }


@router.get("/health/db")
async def database_health(db: Session = Depends(get_db)):
    """
    Detailed database health check with table counts.
    """
    try:
        # Get table row counts
        tables = ["workers", "farmers", "government_jobs", "private_demands", "job_allocations"]
        counts = {}
        
        for table in tables:
            result = db.execute(text(f"SELECT COUNT(*) FROM {table}"))
            counts[table] = result.scalar()
        
        return {
            "status": "healthy",
            "database": "supabase-postgresql",
            "tables": counts,
            "total_records": sum(counts.values())
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
