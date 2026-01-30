
from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import enum

class WorkStatus(str, enum.Enum):
    AVAILABLE = "AVAILABLE"
    BUSY = "BUSY"

class JobStatus(str, enum.Enum):
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"

class Village(Base):
    __tablename__ = "villages"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    
    # Cached VEDAS Data (Simulating a "Live" view for the algorithm)
    # We store the latest calculated need score here for fast querying
    # 0 = Low Need (Lush Green), 100 = High Need (Drought)
    drought_score = Column(Float, default=0.0) 
    
    # Metadata
    district = Column(String, nullable=True)
    state = Column(String, nullable=True)

    workers = relationship("Worker", back_populates="village")
    jobs = relationship("Job", back_populates="village")

class Worker(Base):
    __tablename__ = "workers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    phone_number = Column(String, unique=True, index=True)
    skills = Column(String) # Comma-separated or JSON
    
    village_id = Column(Integer, ForeignKey("villages.id"))
    village = relationship("Village", back_populates="workers")
    
    status = Column(Enum(WorkStatus), default=WorkStatus.AVAILABLE)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String)
    required_skills = Column(String)
    
    # Job Location
    village_id = Column(Integer, ForeignKey("villages.id"))
    village = relationship("Village", back_populates="jobs")
    
    status = Column(Enum(JobStatus), default=JobStatus.OPEN)
    
    wage_per_day = Column(Float, nullable=True)
    start_date = Column(DateTime(timezone=True), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
