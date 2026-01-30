"""
Sahayog Setu - Database Connection Module
Manages SQLAlchemy engine and session for Supabase PostgreSQL.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from app.config import get_settings

settings = get_settings()

# Create SQLAlchemy engine with connection pooling
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,      # Verify connections before using
    pool_size=5,             # Number of connections to keep open
    max_overflow=10,         # Max additional connections when pool is full
    pool_recycle=3600,       # Recycle connections after 1 hour
    echo=settings.debug      # Log SQL queries in debug mode
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for all ORM models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session.
    Automatically closes the session when the request is complete.
    
    Usage:
        @router.get("/items")
        async def get_items(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
