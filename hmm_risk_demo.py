"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              VBG RAM G SCHEME - HMM RISK MITIGATION MODEL DEMO               ║
║         Hidden Markov Model for Regime Switching & Risk Detection            ║
╚══════════════════════════════════════════════════════════════════════════════╝

STANDALONE DEMO SCRIPT
======================
This script demonstrates a Hidden Markov Model (HMM) approach for detecting
risk regimes and providing mitigation recommendations for the VBG RAM G scheme.

KEY CONCEPTS:
- VBG RAM G (Viksit Bharat – Guarantee for Rozgar and Ajeevika Mission Gramin)
- Replaced MGNREGA with "normative allocation" (capped budgets)
- 60/40 Central-State funding split (90/10 for NE states)
- 60-day Agricultural Pause during harvest seasons
- Four priority verticals: Water, Infrastructure, Livelihood, Disaster

THE MODEL:
- 3-State HMM: NORMAL → STRESS → CRISIS
- Multivariate observations: Budget velocity, Worker demand, Ag-pause, Drought
- Regime-specific mitigation strategies aligned with VBG RAM G policy

Author: Sahayog Setu Team
Date: January 2026
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass, field
import random

# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS & DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class RiskRegime(Enum):
    """Three-state risk regime for HMM"""
    NORMAL = "NORMAL"       # Business as usual
    STRESS = "STRESS"       # Warning signs, needs attention
    CRISIS = "CRISIS"       # Emergency, immediate action required


class StateType(Enum):
    """Indian state/UT classification for funding patterns"""
    GENERAL = "GENERAL"                 # 60% Central, 40% State
    NE_HIMALAYAN = "NE_HIMALAYAN"       # 90% Central, 10% State
    UNION_TERRITORY = "UNION_TERRITORY" # 100% Central


class PriorityVertical(Enum):
    """Four priority verticals under VBG RAM G"""
    WATER_SECURITY = "Water Security"
    RURAL_INFRASTRUCTURE = "Core Rural Infrastructure"
    LIVELIHOOD_ASSETS = "Livelihood Assets"
    DISASTER_MITIGATION = "Disaster Mitigation"


@dataclass
class TimeSeriesObservation:
    """Single observation in the time series"""
    date: datetime
    budget_utilization_rate: float   # 0-100 (% of allocated budget spent)
    worker_demand_index: float       # 0-100 (ratio of requests to jobs)
    agricultural_pause_active: int   # 0 or 1
    drought_stress_score: float      # 0-100 (from NDVI/NDWI data)
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy array for HMM processing"""
        return np.array([
            self.budget_utilization_rate,
            self.worker_demand_index,
            self.agricultural_pause_active * 100,  # Scale to 0-100
            self.drought_stress_score
        ])


@dataclass
class RegimeDetectionResult:
    """Result of regime detection"""
    current_regime: RiskRegime
    regime_probability: float
    transition_probabilities: Dict[str, float]
    days_in_current_regime: int
    forecast_next_30_days: List[Tuple[str, float]]


@dataclass
class MitigationRecommendation:
    """Single mitigation recommendation"""
    priority: int  # 1=highest
    action: str
    description: str
    estimated_impact: str
    responsible_entity: str


@dataclass
class RiskAnalysisResult:
    """Complete risk analysis output"""
    analysis_date: datetime
    state_name: str
    state_type: StateType
    regime_result: RegimeDetectionResult
    mitigations: List[MitigationRecommendation]
    budget_status: Dict[str, float]
    executive_summary: str


# ═══════════════════════════════════════════════════════════════════════════════
# HIDDEN MARKOV MODEL IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

class GaussianHMMRiskModel:
    """
    3-State Gaussian Hidden Markov Model for Risk Regime Detection
    
    States:
        0 = NORMAL  (Low risk, stable operations)
        1 = STRESS  (Elevated risk, warning signs)
        2 = CRISIS  (High risk, emergency measures needed)
    
    Observations (4 features):
        - Budget utilization rate (0-100%)
        - Worker demand index (0-100)
        - Agricultural pause indicator (0 or 100)
        - Drought stress score (0-100)
    """
    
    def __init__(self):
        self.n_states = 3  # NORMAL, STRESS, CRISIS
        self.n_features = 4
        
        # State-to-regime mapping
        self.state_to_regime = {
            0: RiskRegime.NORMAL,
            1: RiskRegime.STRESS,
            2: RiskRegime.CRISIS
        }
        
        # Initial state probabilities (π)
        # Most likely to start in NORMAL state
        self.initial_probs = np.array([0.70, 0.25, 0.05])
        
        # Transition probability matrix (A)
        # Rows = from state, Cols = to state
        self.transition_matrix = np.array([
            # To:    NORMAL  STRESS  CRISIS
            [0.85,   0.12,   0.03],   # From NORMAL
            [0.25,   0.55,   0.20],   # From STRESS
            [0.10,   0.35,   0.55]    # From CRISIS
        ])
        
        # Emission parameters (Gaussian means for each state)
        # [budget_util, worker_demand, ag_pause, drought]
        self.emission_means = np.array([
            [30.0, 40.0, 0.0, 25.0],    # NORMAL: low utilization, balanced demand
            [60.0, 70.0, 30.0, 55.0],   # STRESS: moderate-high utilization
            [85.0, 90.0, 70.0, 80.0]    # CRISIS: high utilization, high demand
        ])
        
        # Emission covariances (diagonal for simplicity)
        self.emission_stds = np.array([
            [15.0, 15.0, 20.0, 15.0],   # NORMAL: lower variance
            [12.0, 12.0, 35.0, 15.0],   # STRESS: moderate variance
            [10.0, 10.0, 30.0, 12.0]    # CRISIS: focused high values
        ])
    
    def _gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
        """Multivariate Gaussian probability density (diagonal covariance)"""
        z = (x - mean) / std
        return np.exp(-0.5 * np.sum(z ** 2)) / np.prod(std * np.sqrt(2 * np.pi))
    
    def _compute_emission_probs(self, observation: np.ndarray) -> np.ndarray:
        """Compute emission probability for each state given observation"""
        probs = np.zeros(self.n_states)
        for state in range(self.n_states):
            probs[state] = self._gaussian_pdf(
                observation,
                self.emission_means[state],
                self.emission_stds[state]
            )
        # Normalize
        probs = probs / (probs.sum() + 1e-10)
        return probs
    
    def forward_algorithm(self, observations: List[np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Forward algorithm to compute state probabilities at each time step.
        Returns alpha matrix and log-likelihood.
        """
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        
        # Initialize with first observation
        emission_probs = self._compute_emission_probs(observations[0])
        alpha[0] = self.initial_probs * emission_probs
        alpha[0] /= (alpha[0].sum() + 1e-10)
        
        # Forward pass
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
        """
        Viterbi algorithm to find most likely state sequence.
        Returns list of state indices.
        """
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # Initialize
        emission_probs = self._compute_emission_probs(observations[0])
        delta[0] = np.log(self.initial_probs + 1e-10) + np.log(emission_probs + 1e-10)
        
        # Forward pass
        for t in range(1, T):
            emission_probs = self._compute_emission_probs(observations[t])
            for j in range(self.n_states):
                trans_probs = delta[t-1] + np.log(self.transition_matrix[:, j] + 1e-10)
                psi[t, j] = np.argmax(trans_probs)
                delta[t, j] = trans_probs[psi[t, j]] + np.log(emission_probs[j] + 1e-10)
        
        # Backtrack
        states = [np.argmax(delta[-1])]
        for t in range(T - 2, -1, -1):
            states.insert(0, psi[t + 1, states[0]])
        
        return states
    
    def detect_regime(self, observations: List[TimeSeriesObservation]) -> RegimeDetectionResult:
        """
        Detect current risk regime from time series observations.
        """
        if len(observations) < 2:
            raise ValueError("Need at least 2 observations for regime detection")
        
        # Convert to vectors
        obs_vectors = [obs.to_vector() for obs in observations]
        
        # Run forward algorithm for probabilities
        alpha, _ = self.forward_algorithm(obs_vectors)
        
        # Run Viterbi for most likely sequence
        state_sequence = self.viterbi_algorithm(obs_vectors)
        
        # Current regime
        current_state = state_sequence[-1]
        current_regime = self.state_to_regime[current_state]
        regime_probability = alpha[-1, current_state]
        
        # Count days in current regime
        days_in_regime = 1
        for i in range(len(state_sequence) - 2, -1, -1):
            if state_sequence[i] == current_state:
                days_in_regime += 1
            else:
                break
        
        # Transition probabilities from current state
        trans_probs = {
            regime.value: float(self.transition_matrix[current_state, state])
            for state, regime in self.state_to_regime.items()
        }
        
        # Forecast next 30 days (simple Markov chain projection)
        forecast = self._forecast_regimes(current_state, 30)
        
        return RegimeDetectionResult(
            current_regime=current_regime,
            regime_probability=float(regime_probability),
            transition_probabilities=trans_probs,
            days_in_current_regime=days_in_regime,
            forecast_next_30_days=forecast
        )
    
    def _forecast_regimes(self, current_state: int, days: int) -> List[Tuple[str, float]]:
        """Monte Carlo forecast of regime probabilities over next N days"""
        n_simulations = 1000
        regime_counts = {regime.value: np.zeros(days) for regime in RiskRegime}
        
        for _ in range(n_simulations):
            state = current_state
            for day in range(days):
                # Sample next state
                state = np.random.choice(
                    self.n_states,
                    p=self.transition_matrix[state]
                )
                regime_counts[self.state_to_regime[state].value][day] += 1
        
        # Aggregate into weekly buckets
        weekly_forecast = []
        for week in range(4):
            start_day = week * 7
            end_day = min((week + 1) * 7, days)
            probs = {}
            for regime in RiskRegime:
                avg_prob = regime_counts[regime.value][start_day:end_day].mean() / n_simulations
                probs[regime.value] = avg_prob
            dominant_regime = max(probs.items(), key=lambda x: x[1])
            weekly_forecast.append((f"Week {week+1}: {dominant_regime[0]}", dominant_regime[1]))
        
        return weekly_forecast


# ═══════════════════════════════════════════════════════════════════════════════
# RISK MITIGATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class RiskMitigationEngine:
    """
    Generates regime-specific mitigation recommendations
    aligned with VBG RAM G policy framework.
    """
    
    def __init__(self, state_type: StateType):
        self.state_type = state_type
        self.funding_ratio = self._get_funding_ratio()
    
    def _get_funding_ratio(self) -> Dict[str, float]:
        """Get Central-State funding split based on state type"""
        if self.state_type == StateType.GENERAL:
            return {"central": 60.0, "state": 40.0}
        elif self.state_type == StateType.NE_HIMALAYAN:
            return {"central": 90.0, "state": 10.0}
        else:  # UNION_TERRITORY
            return {"central": 100.0, "state": 0.0}
    
    def get_mitigations(
        self,
        regime: RiskRegime,
        budget_utilization: float,
        ag_pause_active: bool,
        drought_score: float
    ) -> List[MitigationRecommendation]:
        """Generate mitigation recommendations based on current state"""
        
        mitigations = []
        
        if regime == RiskRegime.NORMAL:
            mitigations = self._get_normal_mitigations(budget_utilization)
        elif regime == RiskRegime.STRESS:
            mitigations = self._get_stress_mitigations(
                budget_utilization, ag_pause_active, drought_score
            )
        else:  # CRISIS
            mitigations = self._get_crisis_mitigations(
                budget_utilization, ag_pause_active, drought_score
            )
        
        return mitigations
    
    def _get_normal_mitigations(self, budget_util: float) -> List[MitigationRecommendation]:
        """Proactive measures during normal operations"""
        return [
            MitigationRecommendation(
                priority=3,
                action="Proactive Budget Planning",
                description=f"Current utilization at {budget_util:.1f}%. Build buffer reserves for seasonal spikes.",
                estimated_impact="Prevent future fund shortages",
                responsible_entity="District Programme Coordinator"
            ),
            MitigationRecommendation(
                priority=3,
                action="Skill Development Training",
                description="Utilize stable period for upskilling workers in priority verticals.",
                estimated_impact="Improve asset quality, reduce rework",
                responsible_entity="Block Development Officer"
            ),
            MitigationRecommendation(
                priority=4,
                action="VGPP Pipeline Development",
                description="Pre-approve Viksit Gram Panchayat Plans for upcoming quarters.",
                estimated_impact="Reduce approval delays during high-demand periods",
                responsible_entity="Gram Panchayat"
            )
        ]
    
    def _get_stress_mitigations(
        self, budget_util: float, ag_pause: bool, drought: float
    ) -> List[MitigationRecommendation]:
        """Warning-level interventions"""
        mitigations = [
            MitigationRecommendation(
                priority=2,
                action="Budget Rationing Protocol",
                description=f"Utilization at {budget_util:.1f}% - implement daily spending caps to extend funds through fiscal year.",
                estimated_impact="Extend budget runway by 2-3 months",
                responsible_entity="District Collector"
            ),
            MitigationRecommendation(
                priority=2,
                action="Priority Queue Activation",
                description="Implement strict Need Score-based allocation. Prioritize workers from high-drought villages.",
                estimated_impact=f"Direct relief to {drought:.0f}% drought-affected population",
                responsible_entity="Block Development Officer"
            ),
            MitigationRecommendation(
                priority=2,
                action="State Government Alert",
                description=f"Notify state treasury of potential supplemental funding need ({self.funding_ratio['state']:.0f}% state share may increase).",
                estimated_impact="Prepare contingency funds",
                responsible_entity="State Rural Development Department"
            )
        ]
        
        if ag_pause:
            mitigations.insert(0, MitigationRecommendation(
                priority=1,
                action="Harvest Hero Activation",
                description="Agricultural pause active - activate private labor marketplace matching.",
                estimated_impact="Provide income continuity for 100% of registered workers",
                responsible_entity="Sahayog Setu Platform"
            ))
        
        return mitigations
    
    def _get_crisis_mitigations(
        self, budget_util: float, ag_pause: bool, drought: float
    ) -> List[MitigationRecommendation]:
        """Emergency measures for crisis regime"""
        mitigations = [
            MitigationRecommendation(
                priority=1,
                action="EMERGENCY: Budget Freeze Imminent",
                description=f"Budget utilization at {budget_util:.1f}%! Freeze new project approvals immediately.",
                estimated_impact="Prevent mid-project abandonment",
                responsible_entity="District Collector (URGENT)"
            ),
            MitigationRecommendation(
                priority=1,
                action="State Supplemental Funding Request",
                description=f"State must bear 100% of excess demand. Request ₹{budget_util * 0.5:.0f}L emergency allocation.",
                estimated_impact="Cover excess worker demand",
                responsible_entity="State Finance Department"
            ),
            MitigationRecommendation(
                priority=1,
                action="Crisis Priority Allocation",
                description="Exclusively serve workers with Need Score > 80. Suspend all other allocations.",
                estimated_impact=f"Protect most vulnerable {100 - drought:.0f}% of workers",
                responsible_entity="Block Development Officer"
            ),
            MitigationRecommendation(
                priority=2,
                action="Central Government Escalation",
                description="Escalate to Ministry of Rural Development for normative allocation review.",
                estimated_impact="Unlock emergency central funds",
                responsible_entity="State Government"
            )
        ]
        
        if ag_pause:
            mitigations.insert(0, MitigationRecommendation(
                priority=1,
                action="CRITICAL: Full Private Marketplace Pivot",
                description="Ag-pause + budget crisis - redirect ALL workers to Harvest Hero private matching.",
                estimated_impact="100% income continuity via private sector",
                responsible_entity="Sahayog Setu Platform + Agriculture Dept"
            ))
        
        if drought > 70:
            mitigations.insert(2, MitigationRecommendation(
                priority=1,
                action="Drought Relief Coordination",
                description=f"Extreme drought stress ({drought:.0f}/100) - coordinate with SDRF for relief works.",
                estimated_impact="Activate disaster mitigation vertical",
                responsible_entity="District Disaster Management Authority"
            ))
        
        return mitigations


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class VBGRAMGDataGenerator:
    """
    Generates synthetic time series data simulating various
    VBG RAM G scenarios throughout a fiscal year.
    """
    
    # Agricultural pause periods (sowing + harvesting)
    KHARIF_SOWING = (datetime(2025, 6, 15), datetime(2025, 8, 15))      # Monsoon sowing
    RABI_SOWING = (datetime(2025, 10, 15), datetime(2025, 12, 15))      # Winter sowing
    RABI_HARVEST = (datetime(2026, 3, 15), datetime(2026, 5, 15))       # Winter harvest
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
    
    def is_ag_pause(self, date: datetime) -> bool:
        """Check if agricultural pause is active on given date"""
        for start, end in [self.KHARIF_SOWING, self.RABI_SOWING, self.RABI_HARVEST]:
            # Adjust years for comparison
            check_date = date.replace(year=start.year)
            if start <= check_date <= end:
                return True
        return False
    
    def generate_scenario(
        self,
        scenario_name: str,
        start_date: datetime,
        days: int = 90
    ) -> List[TimeSeriesObservation]:
        """Generate time series for a specific scenario"""
        
        generators = {
            "NORMAL_STABLE": self._generate_normal_stable,
            "BUDGET_STRESS": self._generate_budget_stress,
            "BUDGET_CRISIS": self._generate_budget_crisis,
            "AG_PAUSE_STRESS": self._generate_ag_pause_stress,
            "DROUGHT_CRISIS": self._generate_drought_crisis,
            "MONSOON_RECOVERY": self._generate_monsoon_recovery,
            "YEAR_END_CRUNCH": self._generate_year_end_crunch,
            "MIXED_VOLATILITY": self._generate_mixed_volatility,
        }
        
        generator = generators.get(scenario_name, self._generate_normal_stable)
        return generator(start_date, days)
    
    def _generate_normal_stable(
        self, start: datetime, days: int
    ) -> List[TimeSeriesObservation]:
        """Normal operations with stable spending"""
        observations = []
        for d in range(days):
            date = start + timedelta(days=d)
            observations.append(TimeSeriesObservation(
                date=date,
                budget_utilization_rate=25 + self.rng.normal(0, 8),
                worker_demand_index=40 + self.rng.normal(0, 10),
                agricultural_pause_active=1 if self.is_ag_pause(date) else 0,
                drought_stress_score=20 + self.rng.normal(0, 5)
            ))
        return observations
    
    def _generate_budget_stress(
        self, start: datetime, days: int
    ) -> List[TimeSeriesObservation]:
        """Accelerating budget consumption leading to stress"""
        observations = []
        for d in range(days):
            date = start + timedelta(days=d)
            # Budget utilization ramps up over time
            base_util = 30 + (d / days) * 40  # 30% → 70%
            observations.append(TimeSeriesObservation(
                date=date,
                budget_utilization_rate=base_util + self.rng.normal(0, 5),
                worker_demand_index=50 + (d / days) * 25 + self.rng.normal(0, 8),
                agricultural_pause_active=1 if self.is_ag_pause(date) else 0,
                drought_stress_score=40 + self.rng.normal(0, 10)
            ))
        return observations
    
    def _generate_budget_crisis(
        self, start: datetime, days: int
    ) -> List[TimeSeriesObservation]:
        """Budget near exhaustion - crisis conditions"""
        observations = []
        for d in range(days):
            date = start + timedelta(days=d)
            observations.append(TimeSeriesObservation(
                date=date,
                budget_utilization_rate=min(95, 75 + d * 0.3 + self.rng.normal(0, 3)),
                worker_demand_index=85 + self.rng.normal(0, 5),
                agricultural_pause_active=1 if self.is_ag_pause(date) else 0,
                drought_stress_score=60 + self.rng.normal(0, 8)
            ))
        return observations
    
    def _generate_ag_pause_stress(
        self, start: datetime, days: int
    ) -> List[TimeSeriesObservation]:
        """Agricultural pause period with high worker displacement"""
        observations = []
        for d in range(days):
            date = start + timedelta(days=d)
            is_pause = d < 60  # First 60 days are pause period
            observations.append(TimeSeriesObservation(
                date=date,
                budget_utilization_rate=20 if is_pause else 50 + self.rng.normal(0, 8),
                worker_demand_index=90 if is_pause else 60 + self.rng.normal(0, 10),
                agricultural_pause_active=1 if is_pause else 0,
                drought_stress_score=35 + self.rng.normal(0, 10)
            ))
        return observations
    
    def _generate_drought_crisis(
        self, start: datetime, days: int
    ) -> List[TimeSeriesObservation]:
        """Severe drought conditions driving demand spike"""
        observations = []
        for d in range(days):
            date = start + timedelta(days=d)
            observations.append(TimeSeriesObservation(
                date=date,
                budget_utilization_rate=60 + self.rng.normal(0, 10),
                worker_demand_index=88 + self.rng.normal(0, 5),
                agricultural_pause_active=1 if self.is_ag_pause(date) else 0,
                drought_stress_score=85 + self.rng.normal(0, 5)  # Severe drought
            ))
        return observations
    
    def _generate_monsoon_recovery(
        self, start: datetime, days: int
    ) -> List[TimeSeriesObservation]:
        """Post-monsoon recovery period - stress to normal"""
        observations = []
        for d in range(days):
            date = start + timedelta(days=d)
            # Recovery trajectory
            recovery_factor = min(1.0, d / 45)  # Full recovery in 45 days
            observations.append(TimeSeriesObservation(
                date=date,
                budget_utilization_rate=70 - recovery_factor * 35 + self.rng.normal(0, 8),
                worker_demand_index=80 - recovery_factor * 35 + self.rng.normal(0, 10),
                agricultural_pause_active=0,
                drought_stress_score=75 - recovery_factor * 50 + self.rng.normal(0, 8)
            ))
        return observations
    
    def _generate_year_end_crunch(
        self, start: datetime, days: int
    ) -> List[TimeSeriesObservation]:
        """Fiscal year-end panic spending / budget exhaustion"""
        observations = []
        for d in range(days):
            date = start + timedelta(days=d)
            # Panic acceleration in last 30 days
            if d < 60:
                util = 50 + d * 0.5 + self.rng.normal(0, 5)
                demand = 55 + d * 0.3 + self.rng.normal(0, 8)
            else:
                util = min(98, 80 + (d - 60) * 0.6 + self.rng.normal(0, 3))
                demand = 85 + self.rng.normal(0, 5)
            
            observations.append(TimeSeriesObservation(
                date=date,
                budget_utilization_rate=util,
                worker_demand_index=demand,
                agricultural_pause_active=0,
                drought_stress_score=40 + self.rng.normal(0, 10)
            ))
        return observations
    
    def _generate_mixed_volatility(
        self, start: datetime, days: int
    ) -> List[TimeSeriesObservation]:
        """High volatility with regime shifts"""
        observations = []
        current_regime = 0  # Start normal
        regime_duration = 0
        
        for d in range(days):
            date = start + timedelta(days=d)
            regime_duration += 1
            
            # Random regime shifts
            if regime_duration > self.rng.randint(10, 25):
                current_regime = self.rng.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
                regime_duration = 0
            
            if current_regime == 0:  # Normal
                util, demand, drought = 30, 45, 25
            elif current_regime == 1:  # Stress
                util, demand, drought = 60, 70, 55
            else:  # Crisis
                util, demand, drought = 85, 90, 75
            
            observations.append(TimeSeriesObservation(
                date=date,
                budget_utilization_rate=util + self.rng.normal(0, 10),
                worker_demand_index=demand + self.rng.normal(0, 8),
                agricultural_pause_active=1 if self.is_ag_pause(date) else 0,
                drought_stress_score=drought + self.rng.normal(0, 8)
            ))
        return observations


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def print_header(title: str):
    """Print formatted section header"""
    print("\n" + "═" * 80)
    print(f"  {title}")
    print("═" * 80)


def print_subheader(title: str):
    """Print formatted subsection header"""
    print(f"\n  ╔{'═' * 60}╗")
    print(f"  ║  {title:<56}  ║")
    print(f"  ╚{'═' * 60}╝")


def run_scenario_analysis(
    scenario_name: str,
    state_name: str,
    state_type: StateType,
    start_date: datetime,
    days: int = 90
):
    """Run complete analysis for a scenario"""
    
    print_subheader(f"SCENARIO: {scenario_name}")
    print(f"\n  📍 State: {state_name} ({state_type.value})")
    print(f"  📅 Period: {start_date.strftime('%Y-%m-%d')} to {(start_date + timedelta(days=days)).strftime('%Y-%m-%d')}")
    
    # Generate data
    generator = VBGRAMGDataGenerator(seed=hash(scenario_name) % 10000)
    observations = generator.generate_scenario(scenario_name, start_date, days)
    
    # Print sample observations
    print(f"\n  📊 Sample Observations (first 5 days):")
    print(f"  {'Date':<12} {'Budget%':<10} {'Demand':<10} {'AgPause':<10} {'Drought':<10}")
    print(f"  {'-'*52}")
    for obs in observations[:5]:
        print(f"  {obs.date.strftime('%Y-%m-%d'):<12} {obs.budget_utilization_rate:>6.1f}%   "
              f"{obs.worker_demand_index:>6.1f}    {obs.agricultural_pause_active:^8}    {obs.drought_stress_score:>6.1f}")
    
    # Run HMM detection
    model = GaussianHMMRiskModel()
    result = model.detect_regime(observations)
    
    # Print regime detection results
    print(f"\n  🎯 REGIME DETECTION RESULTS:")
    print(f"  ┌{'─' * 50}┐")
    
    regime_emoji = {"NORMAL": "🟢", "STRESS": "🟡", "CRISIS": "🔴"}
    print(f"  │ Current Regime: {regime_emoji[result.current_regime.value]} {result.current_regime.value:<26} │")
    print(f"  │ Confidence: {result.regime_probability*100:>6.1f}%{' '*32}│")
    print(f"  │ Days in Regime: {result.days_in_current_regime:>3} days{' '*28}│")
    print(f"  └{'─' * 50}┘")
    
    print(f"\n  📈 Transition Probabilities from {result.current_regime.value}:")
    for regime, prob in result.transition_probabilities.items():
        bar = "█" * int(prob * 20)
        print(f"      → {regime:8}: {bar:<20} {prob*100:>5.1f}%")
    
    print(f"\n  🔮 30-Day Forecast:")
    for week_label, prob in result.forecast_next_30_days:
        print(f"      {week_label} (confidence: {prob*100:.0f}%)")
    
    # Generate mitigations
    engine = RiskMitigationEngine(state_type)
    last_obs = observations[-1]
    mitigations = engine.get_mitigations(
        regime=result.current_regime,
        budget_utilization=last_obs.budget_utilization_rate,
        ag_pause_active=bool(last_obs.agricultural_pause_active),
        drought_score=last_obs.drought_stress_score
    )
    
    print(f"\n  📋 MITIGATION RECOMMENDATIONS ({len(mitigations)} actions):")
    print(f"  {'─' * 70}")
    
    priority_emoji = {1: "🔴 URGENT", 2: "🟠 HIGH", 3: "🟡 MEDIUM", 4: "🟢 LOW"}
    for m in mitigations:
        print(f"\n  {priority_emoji.get(m.priority, '⚪')} | P{m.priority}")
        print(f"  Action: {m.action}")
        print(f"  → {m.description[:70]}...")
        print(f"  📌 Owner: {m.responsible_entity}")
    
    # Budget breakdown
    funding = engine.funding_ratio
    print(f"\n  💰 FUNDING STRUCTURE ({state_type.value}):")
    print(f"      Central Share: {funding['central']:.0f}%")
    print(f"      State Share:   {funding['state']:.0f}%")
    
    return result


def main():
    """Main demo execution"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║       VBG RAM G SCHEME - HIDDEN MARKOV MODEL RISK MITIGATION DEMO            ║
║                                                                              ║
║       Viksit Bharat – Guarantee for Rozgar and Ajeevika Mission (Gramin)     ║
║                                                                              ║
║       "From Rights to Assets - No Worker Left Behind"                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│                           MODEL OVERVIEW                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│  Hidden Markov Model with 3 Risk Regimes:                                    │
│                                                                              │
│    🟢 NORMAL  - Stable operations, budget on track                          │
│    🟡 STRESS  - Warning signs, elevated risk indicators                     │
│    🔴 CRISIS  - Emergency conditions, immediate intervention needed         │
│                                                                              │
│  Input Features:                                                             │
│    1. Budget Utilization Rate (0-100%)                                       │
│    2. Worker Demand Index (job requests / available work)                    │
│    3. Agricultural Pause Indicator (60-day statutory pause)                  │
│    4. Drought Stress Score (from NDVI/NDWI satellite data)                   │
│                                                                              │
│  Output:                                                                     │
│    • Current regime detection with confidence                                │
│    • Transition probability forecasts                                        │
│    • Regime-specific mitigation recommendations                              │
│    • Funding alerts based on state category                                  │
└──────────────────────────────────────────────────────────────────────────────┘
""")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SCENARIO 1: NORMAL STABLE - General State
    # ═══════════════════════════════════════════════════════════════════════════
    print_header("SCENARIO 1: NORMAL STABLE OPERATIONS - MAHARASHTRA (General State)")
    run_scenario_analysis(
        scenario_name="NORMAL_STABLE",
        state_name="Maharashtra",
        state_type=StateType.GENERAL,
        start_date=datetime(2025, 7, 1),
        days=90
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SCENARIO 2: BUDGET STRESS - NE State
    # ═══════════════════════════════════════════════════════════════════════════
    print_header("SCENARIO 2: BUDGET STRESS - ASSAM (NE/Himalayan State)")
    run_scenario_analysis(
        scenario_name="BUDGET_STRESS",
        state_name="Assam",
        state_type=StateType.NE_HIMALAYAN,
        start_date=datetime(2025, 9, 1),
        days=90
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SCENARIO 3: BUDGET CRISIS - Union Territory
    # ═══════════════════════════════════════════════════════════════════════════
    print_header("SCENARIO 3: BUDGET CRISIS - ANDAMAN & NICOBAR (Union Territory)")
    run_scenario_analysis(
        scenario_name="BUDGET_CRISIS",
        state_name="Andaman & Nicobar Islands",
        state_type=StateType.UNION_TERRITORY,
        start_date=datetime(2025, 12, 1),
        days=60
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SCENARIO 4: AGRICULTURAL PAUSE STRESS
    # ═══════════════════════════════════════════════════════════════════════════
    print_header("SCENARIO 4: AGRICULTURAL PAUSE PERIOD - BIHAR")
    run_scenario_analysis(
        scenario_name="AG_PAUSE_STRESS",
        state_name="Bihar",
        state_type=StateType.GENERAL,
        start_date=datetime(2025, 10, 15),  # Rabi sowing season
        days=90
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SCENARIO 5: DROUGHT CRISIS
    # ═══════════════════════════════════════════════════════════════════════════
    print_header("SCENARIO 5: SEVERE DROUGHT - RAJASTHAN")
    run_scenario_analysis(
        scenario_name="DROUGHT_CRISIS",
        state_name="Rajasthan",
        state_type=StateType.GENERAL,
        start_date=datetime(2025, 5, 1),
        days=90
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SCENARIO 6: POST-MONSOON RECOVERY
    # ═══════════════════════════════════════════════════════════════════════════
    print_header("SCENARIO 6: POST-MONSOON RECOVERY - ODISHA")
    run_scenario_analysis(
        scenario_name="MONSOON_RECOVERY",
        state_name="Odisha",
        state_type=StateType.GENERAL,
        start_date=datetime(2025, 9, 15),
        days=90
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SCENARIO 7: YEAR-END BUDGET CRUNCH
    # ═══════════════════════════════════════════════════════════════════════════
    print_header("SCENARIO 7: FISCAL YEAR-END CRUNCH - UTTAR PRADESH")
    run_scenario_analysis(
        scenario_name="YEAR_END_CRUNCH",
        state_name="Uttar Pradesh",
        state_type=StateType.GENERAL,
        start_date=datetime(2026, 1, 1),
        days=90
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SCENARIO 8: MIXED VOLATILITY
    # ═══════════════════════════════════════════════════════════════════════════
    print_header("SCENARIO 8: HIGH VOLATILITY / REGIME SWITCHING - JHARKHAND")
    run_scenario_analysis(
        scenario_name="MIXED_VOLATILITY",
        state_name="Jharkhand",
        state_type=StateType.GENERAL,
        start_date=datetime(2025, 4, 1),
        days=120
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    print_header("DEMO COMPLETE")
    print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│                           DEMO SUMMARY                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ✅ 8 Scenarios Analyzed:                                                    │
│     1. Normal Stable Operations (Maharashtra)                                │
│     2. Budget Stress (Assam - NE State)                                      │
│     3. Budget Crisis (Andaman & Nicobar - UT)                                │
│     4. Agricultural Pause Stress (Bihar)                                     │
│     5. Drought Crisis (Rajasthan)                                            │
│     6. Post-Monsoon Recovery (Odisha)                                        │
│     7. Fiscal Year-End Crunch (Uttar Pradesh)                                │
│     8. High Volatility / Regime Switching (Jharkhand)                        │
│                                                                              │
│  📊 3 State Categories Tested:                                               │
│     • General States (60/40 Central-State split)                             │
│     • NE/Himalayan States (90/10 split)                                      │
│     • Union Territories (100% Central)                                       │
│                                                                              │
│  🎯 Model Capabilities Demonstrated:                                         │
│     • Regime detection with confidence scores                                │
│     • Transition probability forecasting                                     │
│     • Regime-specific mitigation recommendations                             │
│     • VBG RAM G policy-aligned interventions                                 │
│     • Harvest Hero activation for agricultural pauses                        │
│     • Drought relief coordination triggers                                   │
│                                                                              │
│  💡 Key VBG RAM G Policy Features Incorporated:                              │
│     • Normative allocation budget caps                                       │
│     • 60-day agricultural pause (sowing/harvesting)                          │
│     • State liability for excess demand                                      │
│     • Four priority verticals alignment                                      │
│     • Digital governance (DBT, GPS verification)                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

🙏 Sahayog Setu - "360° Livelihood Grid - No Worker Left Behind"
""")


if __name__ == "__main__":
    main()
