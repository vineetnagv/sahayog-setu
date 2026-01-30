"""
Sahayog Setu - FastAPI Application Entry Point
360° Livelihood Grid - Bridging rural employment gaps
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import get_settings
from app.routers import health, workers, jobs, farmers, matching, ledger

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Runs startup and shutdown logic.
    """
    # Startup
    print(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    print(f"📍 Environment: {settings.app_env}")
    yield
    # Shutdown
    print("👋 Shutting down Sahayog Setu API")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="""
## Sahayog Setu API

**360° Livelihood Grid** - Bridging the gap from rights to assets in the VB-G RAM G era.

### Modules

- **Mazdoor Mitra** (Worker Facing) - Voice-first job matching
- **Gram Sahayak** (Leader Facing) - Smart plan generation
- **Drishti Dashboard** (System Facing) - Budget forecasting & fairness monitoring

### Key Features

- 🎤 Voice-first interface for feature phones
- 🌾 "Harvest Hero" - Private job matching during government work pause
- 📊 Fund Flow Forecaster - Budget exhaustion prediction
- ⚖️ Fairness Ledger - Immutable audit trail with bias detection
    """,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(health.router, tags=["Health"])
app.include_router(
    workers.router,
    prefix=settings.api_prefix,
    tags=["Workers (Mazdoor Mitra)"]
)
app.include_router(
    jobs.router,
    prefix=settings.api_prefix,
    tags=["Jobs & Government Work"]
)
app.include_router(
    farmers.router,
    prefix=settings.api_prefix,
    tags=["Farmers & Private Demand"]
)
app.include_router(
    matching.router,
    prefix=settings.api_prefix,
    tags=["Matching Engine"]
)
app.include_router(
    ledger.router,
    prefix=settings.api_prefix,
    tags=["Fairness Ledger & Audit"]
)


@app.get("/", tags=["Root"])
async def root():
    """
    Welcome endpoint with API information.
    """
    return {
        "message": "🙏 Welcome to Sahayog Setu API",
        "tagline": "360° Livelihood Grid - No worker left behind",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }
