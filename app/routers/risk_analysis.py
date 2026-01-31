"""
Risk Analysis Router for VBG RAM G Scheme
HMM-based regime detection and mitigation recommendations
"""

from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from typing import Optional

from app.schemas.hmm_risk_schemas import (
    RiskRegime, StateType,
    RiskAnalysisRequest, RiskAnalysisResponse,
    ForecastRequest, ForecastResponse,
    MitigationsResponse,
    SyntheticDataRequest, SyntheticDataResponse
)
from app.services.hmm_risk_service import HMMRiskService

router = APIRouter(prefix="/risk")

# Initialize service
risk_service = HMMRiskService()


@router.post("/analyze", response_model=RiskAnalysisResponse)
async def analyze_risk(request: RiskAnalysisRequest):
    """
    Analyze time series data to detect current risk regime.
    
    Uses a 3-state Hidden Markov Model to classify the current state as:
    - **NORMAL**: Stable operations, budget on track
    - **STRESS**: Warning signs, elevated risk indicators
    - **CRISIS**: Emergency conditions, immediate intervention needed
    
    Returns regime detection results, transition probabilities,
    30-day forecast, and VBG RAM G policy-aligned mitigation recommendations.
    """
    try:
        result = risk_service.analyze_risk(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/forecast", response_model=ForecastResponse)
async def forecast_regimes(
    current_regime: RiskRegime = Query(..., description="Current detected regime"),
    forecast_days: int = Query(30, ge=7, le=90, description="Days to forecast")
):
    """
    Forecast regime probabilities for the next N days.
    
    Uses Monte Carlo simulation based on HMM transition probabilities
    to predict weekly regime distributions.
    """
    try:
        result = risk_service.forecast(current_regime, forecast_days)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")


@router.get("/mitigations/{regime}", response_model=MitigationsResponse)
async def get_regime_mitigations(
    regime: RiskRegime,
    state_type: StateType = Query(StateType.GENERAL, description="State category for funding")
):
    """
    Get mitigation recommendations for a specific regime.
    
    Returns VBG RAM G policy-aligned actions including:
    - Budget rationing protocols
    - Priority allocation rules
    - Harvest Hero activation triggers
    - State/Central escalation procedures
    - Funding alerts based on state category
    """
    try:
        result = risk_service.get_regime_mitigations(regime, state_type)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get mitigations: {str(e)}")


@router.post("/generate-scenario", response_model=SyntheticDataResponse)
async def generate_synthetic_scenario(request: SyntheticDataRequest):
    """
    Generate synthetic time series data for testing scenarios.
    
    Available scenarios:
    - **NORMAL_STABLE**: Stable operations (Maharashtra)
    - **BUDGET_STRESS**: Accelerating budget consumption (Assam)
    - **BUDGET_CRISIS**: Near budget exhaustion (Andaman & Nicobar)
    - **AG_PAUSE_STRESS**: 60-day agricultural pause period (Bihar)
    - **DROUGHT_CRISIS**: Severe drought conditions (Rajasthan)
    - **MONSOON_RECOVERY**: Post-monsoon recovery (Odisha)
    - **YEAR_END_CRUNCH**: Fiscal year-end panic (Uttar Pradesh)
    - **MIXED_VOLATILITY**: High volatility regime switching (Jharkhand)
    """
    valid_scenarios = [
        "NORMAL_STABLE", "BUDGET_STRESS", "BUDGET_CRISIS",
        "AG_PAUSE_STRESS", "DROUGHT_CRISIS", "MONSOON_RECOVERY",
        "YEAR_END_CRUNCH", "MIXED_VOLATILITY"
    ]
    
    if request.scenario not in valid_scenarios:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scenario. Choose from: {', '.join(valid_scenarios)}"
        )
    
    try:
        result = risk_service.generate_synthetic_data(
            request.scenario, request.start_date, request.days
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.get("/scenarios")
async def list_scenarios():
    """
    List all available synthetic scenarios with descriptions.
    """
    return {
        "scenarios": [
            {
                "name": "NORMAL_STABLE",
                "description": "Normal stable operations",
                "expected_regime": "NORMAL",
                "example_state": "Maharashtra"
            },
            {
                "name": "BUDGET_STRESS",
                "description": "Accelerating budget consumption leading to stress",
                "expected_regime": "STRESS",
                "example_state": "Assam (NE State - 90/10 split)"
            },
            {
                "name": "BUDGET_CRISIS",
                "description": "Budget near exhaustion - crisis conditions",
                "expected_regime": "CRISIS",
                "example_state": "Andaman & Nicobar (UT - 100% Central)"
            },
            {
                "name": "AG_PAUSE_STRESS",
                "description": "60-day agricultural pause with high worker displacement",
                "expected_regime": "STRESS",
                "example_state": "Bihar"
            },
            {
                "name": "DROUGHT_CRISIS",
                "description": "Severe drought driving demand spike",
                "expected_regime": "CRISIS",
                "example_state": "Rajasthan"
            },
            {
                "name": "MONSOON_RECOVERY",
                "description": "Post-monsoon recovery period",
                "expected_regime": "STRESS -> NORMAL",
                "example_state": "Odisha"
            },
            {
                "name": "YEAR_END_CRUNCH",
                "description": "Fiscal year-end panic spending / budget exhaustion",
                "expected_regime": "CRISIS",
                "example_state": "Uttar Pradesh"
            },
            {
                "name": "MIXED_VOLATILITY",
                "description": "High volatility with frequent regime shifts",
                "expected_regime": "MIXED",
                "example_state": "Jharkhand"
            }
        ],
        "state_types": [
            {"type": "GENERAL", "funding": "60% Central, 40% State"},
            {"type": "NE_HIMALAYAN", "funding": "90% Central, 10% State"},
            {"type": "UNION_TERRITORY", "funding": "100% Central"}
        ]
    }


@router.get("/model-info")
async def get_model_info():
    """
    Get information about the HMM risk model architecture.
    """
    return {
        "model_type": "3-State Gaussian Hidden Markov Model",
        "states": {
            "NORMAL": "Low risk, stable operations, budget on track",
            "STRESS": "Elevated risk, warning signs, needs attention",
            "CRISIS": "High risk, emergency measures needed"
        },
        "input_features": [
            {
                "name": "budget_utilization_rate",
                "range": "0-100%",
                "description": "Percentage of allocated budget spent"
            },
            {
                "name": "worker_demand_index",
                "range": "0-100",
                "description": "Ratio of work requests to available jobs"
            },
            {
                "name": "agricultural_pause_active",
                "range": "boolean",
                "description": "Whether 60-day statutory pause is active"
            },
            {
                "name": "drought_stress_score",
                "range": "0-100",
                "description": "Drought severity from NDVI/NDWI satellite data"
            }
        ],
        "transition_matrix": {
            "from_NORMAL": {"to_NORMAL": 0.85, "to_STRESS": 0.12, "to_CRISIS": 0.03},
            "from_STRESS": {"to_NORMAL": 0.25, "to_STRESS": 0.55, "to_CRISIS": 0.20},
            "from_CRISIS": {"to_NORMAL": 0.10, "to_STRESS": 0.35, "to_CRISIS": 0.55}
        },
        "algorithms_used": ["Forward Algorithm", "Viterbi Algorithm", "Monte Carlo Simulation"],
        "policy_alignment": "VBG RAM G (Viksit Bharat - Guarantee for Rozgar and Ajeevika Mission Gramin)"
    }
