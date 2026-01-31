"""
HMM Risk Model Schemas for VBG RAM G Scheme
Pydantic models for API request/response handling
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class RiskRegime(str, Enum):
    """Three-state risk regime for HMM"""
    NORMAL = "NORMAL"
    STRESS = "STRESS"
    CRISIS = "CRISIS"


class StateType(str, Enum):
    """Indian state/UT classification for funding patterns"""
    GENERAL = "GENERAL"                 # 60% Central, 40% State
    NE_HIMALAYAN = "NE_HIMALAYAN"       # 90% Central, 10% State
    UNION_TERRITORY = "UNION_TERRITORY" # 100% Central


class MitigationPriority(int, Enum):
    """Priority levels for mitigation actions"""
    URGENT = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class TimeSeriesDataPoint(BaseModel):
    """Single observation in the time series"""
    date: datetime = Field(..., description="Date of observation")
    budget_utilization_rate: float = Field(
        ..., ge=0, le=100, 
        description="Percentage of allocated budget spent (0-100)"
    )
    worker_demand_index: float = Field(
        ..., ge=0, le=100,
        description="Ratio of work requests to available jobs (0-100)"
    )
    agricultural_pause_active: bool = Field(
        default=False,
        description="Whether 60-day agricultural pause is active"
    )
    drought_stress_score: float = Field(
        ..., ge=0, le=100,
        description="Drought stress from NDVI/NDWI data (0-100)"
    )
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "date": "2025-09-15T00:00:00",
            "budget_utilization_rate": 45.5,
            "worker_demand_index": 62.0,
            "agricultural_pause_active": False,
            "drought_stress_score": 38.0
        }
    })


class RiskAnalysisRequest(BaseModel):
    """Request for regime detection and risk analysis"""
    state_name: str = Field(..., description="Name of the state/UT")
    state_type: StateType = Field(
        default=StateType.GENERAL,
        description="State category for funding pattern"
    )
    observations: List[TimeSeriesDataPoint] = Field(
        ..., min_length=2,
        description="Time series observations (minimum 2 required)"
    )
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "state_name": "Maharashtra",
            "state_type": "GENERAL",
            "observations": [
                {
                    "date": "2025-09-01T00:00:00",
                    "budget_utilization_rate": 40.0,
                    "worker_demand_index": 55.0,
                    "agricultural_pause_active": False,
                    "drought_stress_score": 30.0
                },
                {
                    "date": "2025-09-02T00:00:00",
                    "budget_utilization_rate": 42.0,
                    "worker_demand_index": 58.0,
                    "agricultural_pause_active": False,
                    "drought_stress_score": 32.0
                }
            ]
        }
    })


class ForecastRequest(BaseModel):
    """Request for regime forecasting"""
    current_regime: RiskRegime = Field(..., description="Current detected regime")
    forecast_days: int = Field(
        default=30, ge=7, le=90,
        description="Number of days to forecast (7-90)"
    )


class SyntheticDataRequest(BaseModel):
    """Request for synthetic scenario data generation"""
    scenario: str = Field(
        ...,
        description="Scenario name: NORMAL_STABLE, BUDGET_STRESS, BUDGET_CRISIS, AG_PAUSE_STRESS, DROUGHT_CRISIS, MONSOON_RECOVERY, YEAR_END_CRUNCH, MIXED_VOLATILITY"
    )
    start_date: datetime = Field(..., description="Start date for scenario")
    days: int = Field(default=90, ge=7, le=365, description="Number of days to generate")


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class TransitionProbabilities(BaseModel):
    """Probability of transitioning to each regime"""
    to_normal: float = Field(..., ge=0, le=1)
    to_stress: float = Field(..., ge=0, le=1)
    to_crisis: float = Field(..., ge=0, le=1)


class WeeklyForecast(BaseModel):
    """Weekly regime forecast"""
    week: int = Field(..., description="Week number (1-4)")
    dominant_regime: RiskRegime
    probability: float = Field(..., ge=0, le=1)


class RegimeDetectionResult(BaseModel):
    """Result of regime detection"""
    current_regime: RiskRegime
    regime_probability: float = Field(..., ge=0, le=1)
    transition_probabilities: TransitionProbabilities
    days_in_current_regime: int
    regime_sequence: List[RiskRegime] = Field(
        ..., description="Detected regime sequence for all observations"
    )


class MitigationRecommendation(BaseModel):
    """Single mitigation recommendation"""
    priority: MitigationPriority
    action: str
    description: str
    estimated_impact: str
    responsible_entity: str


class FundingStructure(BaseModel):
    """Central-State funding split"""
    central_share: float
    state_share: float
    state_type: StateType


class RiskAnalysisResponse(BaseModel):
    """Complete risk analysis output"""
    analysis_timestamp: datetime
    state_name: str
    regime_detection: RegimeDetectionResult
    mitigations: List[MitigationRecommendation]
    funding_structure: FundingStructure
    weekly_forecast: List[WeeklyForecast]
    executive_summary: str


class ForecastResponse(BaseModel):
    """Regime forecast response"""
    current_regime: RiskRegime
    forecast_days: int
    weekly_forecast: List[WeeklyForecast]
    overall_risk_trend: str = Field(
        ..., description="IMPROVING, STABLE, or DETERIORATING"
    )


class MitigationsResponse(BaseModel):
    """Mitigations for a specific regime"""
    regime: RiskRegime
    mitigations: List[MitigationRecommendation]
    funding_alert: Optional[str] = None


class SyntheticDataResponse(BaseModel):
    """Generated synthetic scenario data"""
    scenario: str
    state_name: str
    observations: List[TimeSeriesDataPoint]
    expected_regime: RiskRegime
