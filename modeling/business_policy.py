"""Business assumptions for composite risk scoring and backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "modeling" / "artifacts"


AVG_CLAIM_COST = {
    "fall": 3500.0,
    "rth": 20000.0,
    "wound": 4000.0,
    "altercation": 2500.0,
    "med_error": 5000.0,
    "elopement": 2500.0,
}

PREDICTION_HORIZON_DAYS = {
    "fall": 7,
    "rth": 7,
    "wound": 14,
    "altercation": 7,
    "med_error": 7,
    "elopement": 30,
}

SUPPORTED_EVENT_TYPES = tuple(AVG_CLAIM_COST.keys())

# This is a scenario assumption, not a learned causal estimate. Keep one
# baseline value for the POC to avoid false precision; replace it with
# incident-specific measured effects after a prospective pilot.
DEFAULT_INTERVENTION_EFFECTIVENESS = 0.20
INTERVENTION_EFFECTIVENESS_SENSITIVITY = (0.10, 0.15, 0.20, 0.25)
INTERVENTION_EFFECTIVENESS = {
    event_type: DEFAULT_INTERVENTION_EFFECTIVENESS
    for event_type in SUPPORTED_EVENT_TYPES
}

DEFAULT_ALERT_REVIEW_COST = 100.0
DEFAULT_CAPACITY_RATES = (0.05, 0.10, 0.15, 0.20)
PRIMARY_CAPACITY_RATE = 0.10

# Resident-level altercation probabilities are propensity scores, not weekly
# probabilities. The scorer scales them to the observed pre-holdout weekly base
# rate while preserving resident ranking. This fallback is the current
# pre-holdout resident-window label rate: at least one altercation in 7 days.
ALTERCATION_7D_BASE_RATE_FALLBACK = 0.0024

# Fallbacks match the retrospective validation in modeling/artifacts.
RULE_PRECISION_FALLBACK = {
    "med_error": 0.09583333333333334,
    "elopement": 0.02242152466367713,
}

# Rare-event rule outputs are intentionally kept event-specific instead of
# pooled into an "other incident" score. The driver sets and interventions are
# different: medication errors are medication-process risks, and elopement is a
# cognitive/wandering risk.
# Pooling them would dilute the clinical signal and obscure the action to take.
RULE_EVENT_TYPES = {
    "med_error_flag": "med_error",
    "elopement_flag": "elopement",
}

EVENT_LABELS = {
    "fall": "Fall",
    "rth": "Return to hospital",
    "wound": "Wound / pressure injury",
    "altercation": "Altercation",
    "med_error": "Medication error",
    "elopement": "Elopement",
}


@dataclass(frozen=True)
class PolicyAssumptions:
    """Runtime assumptions used by a financial backtest."""

    alert_review_cost: float = DEFAULT_ALERT_REVIEW_COST
    capacity_rates: tuple[float, ...] = DEFAULT_CAPACITY_RATES
    primary_capacity_rate: float = PRIMARY_CAPACITY_RATE
    intervention_effectiveness: float = DEFAULT_INTERVENTION_EFFECTIVENESS
    effectiveness_sensitivity: tuple[float, ...] = INTERVENTION_EFFECTIVENESS_SENSITIVITY


def expected_cost_col(event_type: str) -> str:
    return f"{event_type}_expected_cost"


def probability_col(event_type: str) -> str:
    return f"{event_type}_probability"


def avoidable_cost_col(event_type: str) -> str:
    return f"{event_type}_expected_avoidable_cost"


def load_rule_precisions(
    validation_path: Path | None = None,
) -> dict[str, float]:
    """Load PPV estimates for rule flags, with stable fallbacks."""

    path = validation_path or ARTIFACTS_DIR / "business_rules_validation.csv"
    if not path.exists():
        return dict(RULE_PRECISION_FALLBACK)

    df = pd.read_csv(path)
    name_to_event = {
        "Medication Errors": "med_error",
        "Elopement": "elopement",
    }

    precisions = dict(RULE_PRECISION_FALLBACK)
    for _, row in df.iterrows():
        event_type = name_to_event.get(row["name"])
        if event_type:
            precisions[event_type] = float(row["precision"])
    return precisions
