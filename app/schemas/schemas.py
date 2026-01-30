
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from datetime import datetime
from enum import Enum

# --- Enums ---
class WorkStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    BUSY = "BUSY"

class JobStatus(str, Enum):
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"

# --- Village Schemas ---
class VillageBase(BaseModel):
    name: str
    latitude: float
    longitude: float
    district: Optional[str] = None
    state: Optional[str] = None

class VillageCreate(VillageBase):
    pass

class Village(VillageBase):
    id: int
    drought_score: float
    
    model_config = ConfigDict(from_attributes=True)

# --- Worker Schemas ---
class WorkerBase(BaseModel):
    name: str
    phone_number: str
    skills: str
    village_id: int

class WorkerCreate(WorkerBase):
    pass

class Worker(WorkerBase):
    id: int
    status: WorkStatus
    created_at: datetime
    
    # Enriched data (optional)
    village: Optional[Village] = None

    model_config = ConfigDict(from_attributes=True)

# --- Job Schemas ---
class JobBase(BaseModel):
    title: str
    description: Optional[str] = None
    required_skills: Optional[str] = None
    village_id: int
    wage_per_day: Optional[float] = None

class JobCreate(JobBase):
    pass

class Job(JobBase):
    id: int
    status: JobStatus
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

# --- Allocation Algorithm Schemas ---

class ScoredWorker(BaseModel):
    worker_id: int
    worker_name: str
    village_name: str
    distance_km: float
    drought_score: float
    final_score: float
    match_reason: str

class AllocationResponse(BaseModel):
    job_id: int
    job_location: str
    recommended_workers: List[ScoredWorker]
