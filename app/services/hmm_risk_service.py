"""
HMM Risk Service for VBG RAM G Scheme
Hidden Markov Model-based risk detection and mitigation engine
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging

from app.schemas.hmm_risk_schemas import (
    RiskRegime, StateType, MitigationPriority,
    TimeSeriesDataPoint, RiskAnalysisRequest,
    RegimeDetectionResult, TransitionProbabilities, WeeklyForecast,
    MitigationRecommendation, FundingStructure, RiskAnalysisResponse,
    ForecastResponse, MitigationsResponse, SyntheticDataResponse
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# GAUSSIAN HIDDEN MARKOV MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class GaussianHMMRiskModel:
    """
    3-State Gaussian Hidden Markov Model for Risk Regime Detection
    
    States:
        0 = NORMAL  (Low risk, stable operations)
        1 = STRESS  (Elevated risk, warning signs)
        2 = CRISIS  (High risk, emergency measures needed)
    """
    
    def __init__(self):
        self.n_states = 3
        self.n_features = 4
        
        self.state_to_regime = {
            0: RiskRegime.NORMAL,
            1: RiskRegime.STRESS,
            2: RiskRegime.CRISIS
        }
        
        self.regime_to_state = {v: k for k, v in self.state_to_regime.items()}
        
        # Initial state probabilities
        self.initial_probs = np.array([0.70, 0.25, 0.05])
        
        # Transition probability matrix
        self.transition_matrix = np.array([
            [0.85, 0.12, 0.03],  # From NORMAL
            [0.25, 0.55, 0.20],  # From STRESS
            [0.10, 0.35, 0.55]   # From CRISIS
        ])
        
        # Emission parameters (means)
        self.emission_means = np.array([
            [30.0, 40.0, 0.0, 25.0],    # NORMAL
            [60.0, 70.0, 30.0, 55.0],   # STRESS
            [85.0, 90.0, 70.0, 80.0]    # CRISIS
        ])
        
        # Emission standard deviations
        self.emission_stds = np.array([
            [15.0, 15.0, 20.0, 15.0],
            [12.0, 12.0, 35.0, 15.0],
            [10.0, 10.0, 30.0, 12.0]
        ])
    
    def _observation_to_vector(self, obs: TimeSeriesDataPoint) -> np.ndarray:
        """Convert TimeSeriesDataPoint to numpy array"""
        return np.array([
            obs.budget_utilization_rate,
            obs.worker_demand_index,
            100.0 if obs.agricultural_pause_active else 0.0,
            obs.drought_stress_score
        ])
    
    def _gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
        """Multivariate Gaussian probability density"""
        z = (x - mean) / std
        return np.exp(-0.5 * np.sum(z ** 2)) / np.prod(std * np.sqrt(2 * np.pi))
    
    def _compute_emission_probs(self, observation: np.ndarray) -> np.ndarray:
        """Compute emission probability for each state"""
        probs = np.zeros(self.n_states)
        for state in range(self.n_states):
            probs[state] = self._gaussian_pdf(
                observation,
                self.emission_means[state],
                self.emission_stds[state]
            )
        probs = probs / (probs.sum() + 1e-10)
        return probs
    
    def forward_algorithm(self, observations: List[np.ndarray]) -> Tuple[np.ndarray, float]:
        """Forward algorithm for state probabilities"""
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        
        emission_probs = self._compute_emission_probs(observations[0])
        alpha[0] = self.initial_probs * emission_probs
        alpha[0] /= (alpha[0].sum() + 1e-10)
        
        for t in range(1, T):
            emission_probs = self._compute_emission_probs(observations[t])
            for j in range(self.n_states):
                alpha[t, j] = emission_probs[j] * np.sum(
                    alpha[t-1] * self.transition_matrix[:, j]
                )
            alpha[t] /= (alpha[t].sum() + 1e-10)
        
        log_likelihood = np.log(alpha[-1].sum() + 1e-10)
        return alpha, log_likelihood
    
    def viterbi_algorithm(self, observations: List[np.ndarray]) -> List[int]:
        """Viterbi algorithm for most likely state sequence"""
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        emission_probs = self._compute_emission_probs(observations[0])
        delta[0] = np.log(self.initial_probs + 1e-10) + np.log(emission_probs + 1e-10)
        
        for t in range(1, T):
            emission_probs = self._compute_emission_probs(observations[t])
            for j in range(self.n_states):
                trans_probs = delta[t-1] + np.log(self.transition_matrix[:, j] + 1e-10)
                psi[t, j] = np.argmax(trans_probs)
                delta[t, j] = trans_probs[psi[t, j]] + np.log(emission_probs[j] + 1e-10)
        
        states = [np.argmax(delta[-1])]
        for t in range(T - 2, -1, -1):
            states.insert(0, psi[t + 1, states[0]])
        
        return states
    
    def detect_regime(self, observations: List[TimeSeriesDataPoint]) -> RegimeDetectionResult:
        """Detect current risk regime from observations"""
        if len(observations) < 2:
            raise ValueError("Need at least 2 observations")
        
        obs_vectors = [self._observation_to_vector(obs) for obs in observations]
        
        alpha, _ = self.forward_algorithm(obs_vectors)
        state_sequence = self.viterbi_algorithm(obs_vectors)
        
        current_state = state_sequence[-1]
        current_regime = self.state_to_regime[current_state]
        regime_probability = float(alpha[-1, current_state])
        
        # Count days in current regime
        days_in_regime = 1
        for i in range(len(state_sequence) - 2, -1, -1):
            if state_sequence[i] == current_state:
                days_in_regime += 1
            else:
                break
        
        # Build regime sequence
        regime_sequence = [self.state_to_regime[s] for s in state_sequence]
        
        return RegimeDetectionResult(
            current_regime=current_regime,
            regime_probability=regime_probability,
            transition_probabilities=TransitionProbabilities(
                to_normal=float(self.transition_matrix[current_state, 0]),
                to_stress=float(self.transition_matrix[current_state, 1]),
                to_crisis=float(self.transition_matrix[current_state, 2])
            ),
            days_in_current_regime=days_in_regime,
            regime_sequence=regime_sequence
        )
    
    def forecast_regimes(self, current_regime: RiskRegime, days: int = 30) -> List[WeeklyForecast]:
        """Monte Carlo forecast of regime probabilities"""
        n_simulations = 1000
        current_state = self.regime_to_state[current_regime]
        regime_counts = {regime: np.zeros(days) for regime in RiskRegime}
        
        for _ in range(n_simulations):
            state = current_state
            for day in range(days):
                state = np.random.choice(
                    self.n_states,
                    p=self.transition_matrix[state]
                )
                regime_counts[self.state_to_regime[state]][day] += 1
        
        weekly_forecast = []
        num_weeks = min(4, days // 7)
        
        for week in range(num_weeks):
            start_day = week * 7
            end_day = min((week + 1) * 7, days)
            probs = {}
            for regime in RiskRegime:
                avg_prob = regime_counts[regime][start_day:end_day].mean() / n_simulations
                probs[regime] = avg_prob
            dominant_regime = max(probs.items(), key=lambda x: x[1])
            weekly_forecast.append(WeeklyForecast(
                week=week + 1,
                dominant_regime=dominant_regime[0],
                probability=round(dominant_regime[1], 3)
            ))
        
        return weekly_forecast


# ═══════════════════════════════════════════════════════════════════════════════
# RISK MITIGATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class RiskMitigationEngine:
    """VBG RAM G policy-aligned mitigation recommendations"""
    
    def __init__(self, state_type: StateType):
        self.state_type = state_type
        self.funding = self._get_funding_structure()
    
    def _get_funding_structure(self) -> FundingStructure:
        """Get Central-State funding split"""
        if self.state_type == StateType.GENERAL:
            return FundingStructure(
                central_share=60.0, state_share=40.0, state_type=self.state_type
            )
        elif self.state_type == StateType.NE_HIMALAYAN:
            return FundingStructure(
                central_share=90.0, state_share=10.0, state_type=self.state_type
            )
        else:
            return FundingStructure(
                central_share=100.0, state_share=0.0, state_type=self.state_type
            )
    
    def get_mitigations(
        self,
        regime: RiskRegime,
        budget_utilization: float,
        ag_pause_active: bool,
        drought_score: float
    ) -> List[MitigationRecommendation]:
        """Generate mitigation recommendations"""
        
        if regime == RiskRegime.NORMAL:
            return self._normal_mitigations(budget_utilization)
        elif regime == RiskRegime.STRESS:
            return self._stress_mitigations(budget_utilization, ag_pause_active, drought_score)
        else:
            return self._crisis_mitigations(budget_utilization, ag_pause_active, drought_score)
    
    def _normal_mitigations(self, budget_util: float) -> List[MitigationRecommendation]:
        return [
            MitigationRecommendation(
                priority=MitigationPriority.MEDIUM,
                action="Proactive Budget Planning",
                description=f"Utilization at {budget_util:.1f}%. Build buffer reserves for seasonal spikes.",
                estimated_impact="Prevent future fund shortages",
                responsible_entity="District Programme Coordinator"
            ),
            MitigationRecommendation(
                priority=MitigationPriority.MEDIUM,
                action="Skill Development Training",
                description="Upskill workers in priority verticals during stable period.",
                estimated_impact="Improve asset quality",
                responsible_entity="Block Development Officer"
            ),
            MitigationRecommendation(
                priority=MitigationPriority.LOW,
                action="VGPP Pipeline Development",
                description="Pre-approve Viksit Gram Panchayat Plans for upcoming quarters.",
                estimated_impact="Reduce approval delays",
                responsible_entity="Gram Panchayat"
            )
        ]
    
    def _stress_mitigations(
        self, budget_util: float, ag_pause: bool, drought: float
    ) -> List[MitigationRecommendation]:
        mitigations = [
            MitigationRecommendation(
                priority=MitigationPriority.HIGH,
                action="Budget Rationing Protocol",
                description=f"Utilization at {budget_util:.1f}% - implement daily spending caps.",
                estimated_impact="Extend budget runway by 2-3 months",
                responsible_entity="District Collector"
            ),
            MitigationRecommendation(
                priority=MitigationPriority.HIGH,
                action="Priority Queue Activation",
                description="Prioritize workers from high-drought villages using Need Score.",
                estimated_impact=f"Direct relief to {drought:.0f}% drought-affected",
                responsible_entity="Block Development Officer"
            ),
            MitigationRecommendation(
                priority=MitigationPriority.HIGH,
                action="State Government Alert",
                description=f"Notify treasury of potential supplemental funding need ({self.funding.state_share:.0f}% state share).",
                estimated_impact="Prepare contingency funds",
                responsible_entity="State Rural Development Department"
            )
        ]
        
        if ag_pause:
            mitigations.insert(0, MitigationRecommendation(
                priority=MitigationPriority.URGENT,
                action="Harvest Hero Activation",
                description="Agricultural pause active - activate private labor marketplace.",
                estimated_impact="Income continuity for registered workers",
                responsible_entity="Sahayog Setu Platform"
            ))
        
        return mitigations
    
    def _crisis_mitigations(
        self, budget_util: float, ag_pause: bool, drought: float
    ) -> List[MitigationRecommendation]:
        mitigations = [
            MitigationRecommendation(
                priority=MitigationPriority.URGENT,
                action="EMERGENCY: Budget Freeze",
                description=f"Utilization at {budget_util:.1f}%! Freeze new project approvals.",
                estimated_impact="Prevent mid-project abandonment",
                responsible_entity="District Collector (URGENT)"
            ),
            MitigationRecommendation(
                priority=MitigationPriority.URGENT,
                action="State Supplemental Funding",
                description="State must bear 100% of excess demand per VBG RAM G policy.",
                estimated_impact="Cover excess worker demand",
                responsible_entity="State Finance Department"
            ),
            MitigationRecommendation(
                priority=MitigationPriority.URGENT,
                action="Crisis Priority Allocation",
                description="Exclusively serve workers with Need Score > 80.",
                estimated_impact="Protect most vulnerable workers",
                responsible_entity="Block Development Officer"
            ),
            MitigationRecommendation(
                priority=MitigationPriority.HIGH,
                action="Central Government Escalation",
                description="Escalate to Ministry of Rural Development for allocation review.",
                estimated_impact="Unlock emergency central funds",
                responsible_entity="State Government"
            )
        ]
        
        if ag_pause:
            mitigations.insert(0, MitigationRecommendation(
                priority=MitigationPriority.URGENT,
                action="CRITICAL: Private Marketplace Pivot",
                description="Ag-pause + crisis - redirect ALL workers to Harvest Hero.",
                estimated_impact="100% income via private sector",
                responsible_entity="Sahayog Setu Platform + Agriculture Dept"
            ))
        
        if drought > 70:
            mitigations.insert(2, MitigationRecommendation(
                priority=MitigationPriority.URGENT,
                action="Drought Relief Coordination",
                description=f"Extreme drought ({drought:.0f}/100) - coordinate with SDRF.",
                estimated_impact="Activate disaster mitigation vertical",
                responsible_entity="District Disaster Management Authority"
            ))
        
        return mitigations


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class SyntheticDataGenerator:
    """Generate synthetic VBG RAM G scenario data"""
    
    SCENARIOS = {
        "NORMAL_STABLE": {"regime": RiskRegime.NORMAL, "state": "Maharashtra"},
        "BUDGET_STRESS": {"regime": RiskRegime.STRESS, "state": "Assam"},
        "BUDGET_CRISIS": {"regime": RiskRegime.CRISIS, "state": "Andaman & Nicobar"},
        "AG_PAUSE_STRESS": {"regime": RiskRegime.STRESS, "state": "Bihar"},
        "DROUGHT_CRISIS": {"regime": RiskRegime.CRISIS, "state": "Rajasthan"},
        "MONSOON_RECOVERY": {"regime": RiskRegime.STRESS, "state": "Odisha"},
        "YEAR_END_CRUNCH": {"regime": RiskRegime.CRISIS, "state": "Uttar Pradesh"},
        "MIXED_VOLATILITY": {"regime": RiskRegime.STRESS, "state": "Jharkhand"}
    }
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
    
    def generate(self, scenario: str, start_date: datetime, days: int) -> SyntheticDataResponse:
        """Generate synthetic data for a scenario"""
        scenario_info = self.SCENARIOS.get(scenario, self.SCENARIOS["NORMAL_STABLE"])
        observations = []
        
        for d in range(days):
            date = start_date + timedelta(days=d)
            
            if scenario == "NORMAL_STABLE":
                obs = self._normal_obs(date)
            elif scenario == "BUDGET_STRESS":
                obs = self._stress_obs(date, d, days)
            elif scenario == "BUDGET_CRISIS":
                obs = self._crisis_obs(date)
            elif scenario == "AG_PAUSE_STRESS":
                obs = self._ag_pause_obs(date, d)
            elif scenario == "DROUGHT_CRISIS":
                obs = self._drought_obs(date)
            elif scenario == "MONSOON_RECOVERY":
                obs = self._recovery_obs(date, d, days)
            elif scenario == "YEAR_END_CRUNCH":
                obs = self._year_end_obs(date, d, days)
            else:
                obs = self._volatile_obs(date, d)
            
            observations.append(obs)
        
        return SyntheticDataResponse(
            scenario=scenario,
            state_name=scenario_info["state"],
            observations=observations,
            expected_regime=scenario_info["regime"]
        )
    
    def _normal_obs(self, date: datetime) -> TimeSeriesDataPoint:
        return TimeSeriesDataPoint(
            date=date,
            budget_utilization_rate=max(0, min(100, 25 + self.rng.normal(0, 8))),
            worker_demand_index=max(0, min(100, 40 + self.rng.normal(0, 10))),
            agricultural_pause_active=False,
            drought_stress_score=max(0, min(100, 20 + self.rng.normal(0, 5)))
        )
    
    def _stress_obs(self, date: datetime, d: int, days: int) -> TimeSeriesDataPoint:
        base_util = 30 + (d / days) * 40
        return TimeSeriesDataPoint(
            date=date,
            budget_utilization_rate=max(0, min(100, base_util + self.rng.normal(0, 5))),
            worker_demand_index=max(0, min(100, 50 + (d / days) * 25 + self.rng.normal(0, 8))),
            agricultural_pause_active=False,
            drought_stress_score=max(0, min(100, 40 + self.rng.normal(0, 10)))
        )
    
    def _crisis_obs(self, date: datetime) -> TimeSeriesDataPoint:
        return TimeSeriesDataPoint(
            date=date,
            budget_utilization_rate=max(0, min(100, 85 + self.rng.normal(0, 5))),
            worker_demand_index=max(0, min(100, 90 + self.rng.normal(0, 5))),
            agricultural_pause_active=False,
            drought_stress_score=max(0, min(100, 70 + self.rng.normal(0, 8)))
        )
    
    def _ag_pause_obs(self, date: datetime, d: int) -> TimeSeriesDataPoint:
        is_pause = d < 60
        return TimeSeriesDataPoint(
            date=date,
            budget_utilization_rate=max(0, min(100, 20 if is_pause else 50 + self.rng.normal(0, 8))),
            worker_demand_index=max(0, min(100, 90 if is_pause else 60 + self.rng.normal(0, 10))),
            agricultural_pause_active=is_pause,
            drought_stress_score=max(0, min(100, 35 + self.rng.normal(0, 10)))
        )
    
    def _drought_obs(self, date: datetime) -> TimeSeriesDataPoint:
        return TimeSeriesDataPoint(
            date=date,
            budget_utilization_rate=max(0, min(100, 60 + self.rng.normal(0, 10))),
            worker_demand_index=max(0, min(100, 88 + self.rng.normal(0, 5))),
            agricultural_pause_active=False,
            drought_stress_score=max(0, min(100, 85 + self.rng.normal(0, 5)))
        )
    
    def _recovery_obs(self, date: datetime, d: int, days: int) -> TimeSeriesDataPoint:
        recovery = min(1.0, d / 45)
        return TimeSeriesDataPoint(
            date=date,
            budget_utilization_rate=max(0, min(100, 70 - recovery * 35 + self.rng.normal(0, 8))),
            worker_demand_index=max(0, min(100, 80 - recovery * 35 + self.rng.normal(0, 10))),
            agricultural_pause_active=False,
            drought_stress_score=max(0, min(100, 75 - recovery * 50 + self.rng.normal(0, 8)))
        )
    
    def _year_end_obs(self, date: datetime, d: int, days: int) -> TimeSeriesDataPoint:
        if d < 60:
            util = 50 + d * 0.5
            demand = 55 + d * 0.3
        else:
            util = min(98, 80 + (d - 60) * 0.6)
            demand = 85
        return TimeSeriesDataPoint(
            date=date,
            budget_utilization_rate=max(0, min(100, util + self.rng.normal(0, 5))),
            worker_demand_index=max(0, min(100, demand + self.rng.normal(0, 8))),
            agricultural_pause_active=False,
            drought_stress_score=max(0, min(100, 40 + self.rng.normal(0, 10)))
        )
    
    def _volatile_obs(self, date: datetime, d: int) -> TimeSeriesDataPoint:
        regime = self.rng.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
        if regime == 0:
            util, demand, drought = 30, 45, 25
        elif regime == 1:
            util, demand, drought = 60, 70, 55
        else:
            util, demand, drought = 85, 90, 75
        return TimeSeriesDataPoint(
            date=date,
            budget_utilization_rate=max(0, min(100, util + self.rng.normal(0, 10))),
            worker_demand_index=max(0, min(100, demand + self.rng.normal(0, 8))),
            agricultural_pause_active=False,
            drought_stress_score=max(0, min(100, drought + self.rng.normal(0, 8)))
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SERVICE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class HMMRiskService:
    """Main service for HMM-based risk analysis"""
    
    def __init__(self):
        self.hmm_model = GaussianHMMRiskModel()
        self.data_generator = SyntheticDataGenerator()
    
    def analyze_risk(self, request: RiskAnalysisRequest) -> RiskAnalysisResponse:
        """Perform complete risk analysis"""
        
        # Detect regime
        regime_result = self.hmm_model.detect_regime(request.observations)
        
        # Get mitigations
        mitigation_engine = RiskMitigationEngine(request.state_type)
        last_obs = request.observations[-1]
        mitigations = mitigation_engine.get_mitigations(
            regime=regime_result.current_regime,
            budget_utilization=last_obs.budget_utilization_rate,
            ag_pause_active=last_obs.agricultural_pause_active,
            drought_score=last_obs.drought_stress_score
        )
        
        # Forecast
        weekly_forecast = self.hmm_model.forecast_regimes(regime_result.current_regime)
        
        # Executive summary
        summary = self._generate_summary(
            request.state_name, regime_result, last_obs, mitigation_engine.funding
        )
        
        return RiskAnalysisResponse(
            analysis_timestamp=datetime.now(),
            state_name=request.state_name,
            regime_detection=regime_result,
            mitigations=mitigations,
            funding_structure=mitigation_engine.funding,
            weekly_forecast=weekly_forecast,
            executive_summary=summary
        )
    
    def forecast(self, current_regime: RiskRegime, days: int) -> ForecastResponse:
        """Generate regime forecast"""
        weekly_forecast = self.hmm_model.forecast_regimes(current_regime, days)
        
        # Determine trend
        if len(weekly_forecast) >= 2:
            first_prob = weekly_forecast[0].probability
            last_prob = weekly_forecast[-1].probability
            first_regime = weekly_forecast[0].dominant_regime
            last_regime = weekly_forecast[-1].dominant_regime
            
            if last_regime == RiskRegime.NORMAL and first_regime != RiskRegime.NORMAL:
                trend = "IMPROVING"
            elif last_regime == RiskRegime.CRISIS and first_regime != RiskRegime.CRISIS:
                trend = "DETERIORATING"
            else:
                trend = "STABLE"
        else:
            trend = "STABLE"
        
        return ForecastResponse(
            current_regime=current_regime,
            forecast_days=days,
            weekly_forecast=weekly_forecast,
            overall_risk_trend=trend
        )
    
    def get_regime_mitigations(
        self, regime: RiskRegime, state_type: StateType
    ) -> MitigationsResponse:
        """Get mitigations for a specific regime"""
        engine = RiskMitigationEngine(state_type)
        
        # Use typical values for the regime
        if regime == RiskRegime.NORMAL:
            budget, drought = 30.0, 25.0
        elif regime == RiskRegime.STRESS:
            budget, drought = 60.0, 55.0
        else:
            budget, drought = 85.0, 75.0
        
        mitigations = engine.get_mitigations(
            regime=regime,
            budget_utilization=budget,
            ag_pause_active=False,
            drought_score=drought
        )
        
        funding_alert = None
        if regime == RiskRegime.CRISIS:
            funding_alert = f"State must prepare for 100% liability on excess demand (current state share: {engine.funding.state_share}%)"
        
        return MitigationsResponse(
            regime=regime,
            mitigations=mitigations,
            funding_alert=funding_alert
        )
    
    def generate_synthetic_data(
        self, scenario: str, start_date: datetime, days: int
    ) -> SyntheticDataResponse:
        """Generate synthetic scenario data"""
        return self.data_generator.generate(scenario, start_date, days)
    
    def _generate_summary(
        self,
        state_name: str,
        regime_result: RegimeDetectionResult,
        last_obs: TimeSeriesDataPoint,
        funding: FundingStructure
    ) -> str:
        """Generate executive summary"""
        regime = regime_result.current_regime
        
        if regime == RiskRegime.NORMAL:
            status = "stable with healthy indicators"
            action = "Continue proactive budget management"
        elif regime == RiskRegime.STRESS:
            status = "showing warning signs requiring attention"
            action = "Implement rationing protocols and prepare contingency"
        else:
            status = "in CRISIS requiring immediate intervention"
            action = "Freeze new projects and escalate to state/central authorities"
        
        return (
            f"{state_name} is currently {status}. "
            f"Budget utilization: {last_obs.budget_utilization_rate:.1f}%, "
            f"Worker demand: {last_obs.worker_demand_index:.1f}, "
            f"Drought score: {last_obs.drought_stress_score:.1f}. "
            f"Funding: {funding.central_share:.0f}% Central / {funding.state_share:.0f}% State. "
            f"Recommended: {action}."
        )
