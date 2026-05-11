"""Build composite resident risk scores.

The scorer can operate on the historical holdout matrix or on a target-free
production scoring feature file. It loads calibrated Tier 1 model artifacts,
scores point-in-time business rules, and writes one resident-window row with
expected claim dollars and recommended actions.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlflow
import mlflow.catboost
import mlflow.sklearn
import numpy as np
import polars as pl
from mlflow.tracking import MlflowClient

from modeling.business_policy import (
    ALTERCATION_7D_BASE_RATE_FALLBACK,
    AVG_CLAIM_COST,
    DEFAULT_ALERT_REVIEW_COST,
    DEFAULT_INTERVENTION_EFFECTIVENESS,
    load_rule_precisions,
)
from modeling.business_rules import score_rules_for_spine
from modeling.hyperparameter_tuning import load_hyperparameters


DATA_7D = ROOT / "data" / "processed" / "model_matrix_7d.parquet"
DATA_14D = ROOT / "data" / "processed" / "model_matrix_14d.parquet"
RAW = ROOT / "data" / "raw"
SCORED_DIR = ROOT / "data" / "scored"
ARTIFACTS_DIR = ROOT / "modeling" / "artifacts"
MLFLOW_TRACKING_DIR = ROOT / "mlruns"

DEFAULT_OUTPUT = SCORED_DIR / "composite_scores_holdout.parquet"
DEFAULT_TOP_OUTPUT = ARTIFACTS_DIR / "composite_scores_holdout_top.csv"
HOLDOUT_START = date(2025, 1, 1)
DEFAULT_MODEL_PHASE = "final_model"

META_COLS = [
    "resident_id",
    "facility_id",
    "window_start",
    "window_end",
    "feature_cutoff",
]

TARGETS_BY_PATH = {
    DATA_7D: ["fall_7d", "rth_7d"],
    DATA_14D: ["wound_14d"],
}

TIER1_TARGETS = {
    "fall": {"target": "fall_7d", "data_path": DATA_7D},
    "rth": {"target": "rth_7d", "data_path": DATA_7D},
    "wound": {"target": "wound_14d", "data_path": DATA_14D},
}

warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score composite resident risk.")
    parser.add_argument(
        "--holdout-start",
        default=HOLDOUT_START.isoformat(),
        help="First window_start date to score when using historical matrices.",
    )
    parser.add_argument(
        "--features-path",
        default=None,
        help="Target-free scoring features parquet. If omitted, scores the historical holdout.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Parquet output path for composite scores.",
    )
    parser.add_argument(
        "--top-output",
        default=None,
        help="CSV preview path for the top scored rows.",
    )
    parser.add_argument(
        "--model-source",
        choices=["mlflow", "train"],
        default="mlflow",
        help="Use MLflow final models or train fresh models if artifacts are unavailable.",
    )
    parser.add_argument(
        "--model-phase",
        default=DEFAULT_MODEL_PHASE,
        help="MLflow model phase to load, e.g. final_model or production_model.",
    )
    return parser.parse_args()


def configure_mlflow() -> None:
    MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_DIR.resolve().as_uri())
    mlflow.set_experiment("incident_prediction")


def _to_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _feature_cols(df: pl.DataFrame, data_path: Path) -> list[str]:
    excluded = META_COLS + TARGETS_BY_PATH[data_path]
    return sorted(c for c in df.columns if c not in excluded and c != "score_id")


def load_scoring_spine(
    holdout_start: date | None = None,
    features_path: Path | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    if features_path is None:
        df = pl.read_parquet(DATA_7D)
        score_df = df.filter(pl.col("window_start") >= holdout_start)
    else:
        df = pl.read_parquet(features_path)
        score_df = df

    if "score_id" not in score_df.columns:
        score_df = score_df.sort(["window_start", "facility_id", "resident_id"]).with_row_index(
            "score_id"
        )

    return score_df, _feature_cols(score_df, DATA_7D)


def _latest_mlflow_model(target: str, calibrated: bool = True, phase: str = DEFAULT_MODEL_PHASE):
    configure_mlflow()
    client = MlflowClient()
    experiment = client.get_experiment_by_name("incident_prediction")
    if experiment is None:
        raise FileNotFoundError(f"No MLflow experiment found for {target}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"tags.target = '{target}' and tags.phase = '{phase}' "
            f"and tags.calibrated = '{str(calibrated).lower()}'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise FileNotFoundError(f"No MLflow {phase} model found for {target}")

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    if calibrated:
        return mlflow.sklearn.load_model(model_uri), run_id
    return mlflow.catboost.load_model(model_uri), run_id


def _train_tier1_model(target: str):
    from modeling.train_models import load_data, train_final_model

    df, feature_cols = load_data(target)
    params, _ = load_hyperparameters(ROOT / "data" / "model_hyperparameters", target)
    res = train_final_model(df, target, feature_cols, params)
    return res["cal_model"], feature_cols


def score_tier1_models(
    score_df: pl.DataFrame,
    score_feature_cols: list[str],
    model_source: str,
    model_phase: str = DEFAULT_MODEL_PHASE,
) -> pl.DataFrame:
    target_cols = [c for c in ["fall_7d", "rth_7d", "wound_14d"] if c in score_df.columns]
    out = score_df.select(["score_id"] + META_COLS + target_cols)

    for event_type, cfg in TIER1_TARGETS.items():
        target = cfg["target"]
        train_df = pl.read_parquet(cfg["data_path"])
        model_feature_cols = _feature_cols(train_df, cfg["data_path"])

        missing = sorted(set(model_feature_cols) - set(score_feature_cols))
        if missing:
            raise ValueError(f"Scoring spine is missing features for {target}: {missing[:5]}")

        if model_source == "mlflow":
            try:
                model, run_id = _latest_mlflow_model(
                    target, calibrated=True, phase=model_phase
                )
                print(f"  Loaded MLflow {model_phase} model for {target}: {run_id}")
            except FileNotFoundError:
                if model_phase != DEFAULT_MODEL_PHASE:
                    raise FileNotFoundError(
                        f"No MLflow {model_phase} model found for {target}. "
                        "Run modeling/train_latest_models.py or use --model-phase final_model."
                    )
                print(f"  No MLflow {model_phase} model found for {target}; training fallback.")
                model, model_feature_cols = _train_tier1_model(target)
        else:
            model, model_feature_cols = _train_tier1_model(target)

        X_score = score_df.select(model_feature_cols)
        probabilities = model.predict_proba(X_score)[:, 1]
        out = out.join(
            pl.DataFrame(
                {
                    "score_id": score_df["score_id"],
                    f"{event_type}_probability": probabilities,
                }
            ),
            on="score_id",
            how="left",
        )

    return out


def _filter_pre_cutoff(raw: dict[str, pl.DataFrame], cutoff: datetime) -> dict[str, pl.DataFrame]:
    return {
        "residents": raw["residents"],
        "incidents": raw["incidents"].filter(pl.col("occurred_at") < cutoff),
        "diagnoses": raw["diagnoses"].filter(
            pl.col("onset_at").is_null() | (pl.col("onset_at") < cutoff)
        ),
        "needs": raw["needs"].filter(
            pl.col("initiated_at").is_null() | (pl.col("initiated_at") < cutoff)
        ),
        "document_tags": raw["document_tags"].filter(pl.col("created_at") < cutoff),
    }


def _altercation_weekly_base_rate(cutoff: datetime) -> float:
    matrix = pl.read_parquet(DATA_7D)
    windows = (
        matrix.filter(pl.col("window_end") <= cutoff)
        .select("resident_id", "window_start", "window_end")
        .with_row_index("row_id")
    )
    if windows.is_empty():
        return ALTERCATION_7D_BASE_RATE_FALLBACK

    incidents = (
        pl.read_parquet(RAW / "incidents.parquet")
        .filter(pl.col("strikeout") == False)
        .filter(pl.col("incident_type") == "Altercation")
        .filter(pl.col("occurred_at") < cutoff)
        .select("resident_id", "occurred_at")
    )
    if incidents.is_empty():
        return ALTERCATION_7D_BASE_RATE_FALLBACK

    positive_windows = (
        windows.join(incidents, on="resident_id", how="inner")
        .filter(
            (pl.col("occurred_at") >= pl.col("window_start"))
            & (pl.col("occurred_at") < pl.col("window_end"))
        )
        .select("row_id")
        .unique()
        .height
    )
    base_rate = positive_windows / windows.height
    return float(base_rate) if base_rate > 0 else ALTERCATION_7D_BASE_RATE_FALLBACK


def score_altercation_as_of(
    score_df: pl.DataFrame,
    cutoff: datetime,
    model_phase: str = DEFAULT_MODEL_PHASE,
) -> pl.DataFrame:
    """Train the resident-level H4 model on pre-cutoff data and score residents."""

    from modeling.train_altercation import build_altercation_dataset, make_catboost

    raw = {
        "residents": pl.read_parquet(RAW / "residents.parquet"),
        "incidents": pl.read_parquet(RAW / "incidents.parquet").filter(
            pl.col("strikeout") == False
        ),
        "diagnoses": pl.read_parquet(RAW / "diagnoses.parquet").filter(
            pl.col("strikeout") == False
        ),
        "needs": pl.read_parquet(RAW / "needs.parquet").filter(pl.col("strikeout") == False),
        "document_tags": pl.read_parquet(RAW / "document_tags.parquet").filter(
            pl.col("deleted_at").is_null()
        ),
    }
    pre = _filter_pre_cutoff(raw, cutoff)

    df, feature_cols, target_col = build_altercation_dataset(
        pre["residents"],
        pre["incidents"],
        pre["diagnoses"],
        pre["needs"],
        pre["document_tags"],
        signal_end=cutoff,
    )

    X = df.select(feature_cols)
    y = df[target_col].to_numpy().ravel()

    if model_phase == "production_model":
        try:
            model, run_id = _latest_mlflow_model(
                "has_altercation", calibrated=False, phase=model_phase
            )
            print(f"  Loaded MLflow {model_phase} model for has_altercation: {run_id}")
        except FileNotFoundError:
            print("  No production altercation model found; training fallback.")
            params, _ = load_hyperparameters(
                ROOT / "data" / "model_hyperparameters", "altercation"
            )
            model = make_catboost(params)
            model.fit(X, y)
    else:
        params, _ = load_hyperparameters(ROOT / "data" / "model_hyperparameters", "altercation")
        model = make_catboost(params)
        model.fit(X, y)

    resident_prob = model.predict_proba(X)[:, 1]
    mean_resident_prob = float(np.mean(resident_prob))
    weekly_base_rate = _altercation_weekly_base_rate(cutoff)
    scale = weekly_base_rate / mean_resident_prob if mean_resident_prob > 0 else 0.0
    weekly_prob = np.clip(resident_prob * scale, 0.0, 1.0)

    resident_scores = pl.DataFrame(
        {
            "resident_id": df["resident_id"],
            "altercation_resident_probability": resident_prob,
            "altercation_probability": weekly_prob,
        }
    )

    return (
        score_df.select("score_id", "resident_id")
        .join(resident_scores, on="resident_id", how="left")
        .with_columns(
            pl.col("altercation_resident_probability").fill_null(mean_resident_prob),
            pl.col("altercation_probability").fill_null(weekly_base_rate),
            pl.lit(weekly_base_rate).alias("altercation_weekly_base_rate"),
            pl.lit(scale).alias("altercation_probability_scale"),
        )
    )


def add_composite_costs(scored: pl.DataFrame) -> pl.DataFrame:
    out = scored.with_columns(
        fall_expected_cost=pl.col("fall_probability") * AVG_CLAIM_COST["fall"],
        rth_expected_cost=pl.col("rth_probability") * AVG_CLAIM_COST["rth"],
        wound_expected_cost=pl.col("wound_probability") * AVG_CLAIM_COST["wound"],
        altercation_expected_cost=pl.col("altercation_probability")
        * AVG_CLAIM_COST["altercation"],
        med_error_probability=pl.col("med_error_rule_probability"),
        elopement_probability=pl.col("elopement_rule_probability"),
        med_error_expected_cost=pl.col("med_error_rule_expected_cost"),
        elopement_expected_cost=pl.col("elopement_rule_expected_cost"),
    )

    expected_cols = [
        "fall_expected_cost",
        "rth_expected_cost",
        "wound_expected_cost",
        "altercation_expected_cost",
        "med_error_expected_cost",
        "elopement_expected_cost",
    ]

    avoidable_exprs = []
    for event_type in [
        "fall",
        "rth",
        "wound",
        "altercation",
        "med_error",
        "elopement",
    ]:
        avoidable_exprs.append(
            (
                pl.col(f"{event_type}_expected_cost")
                * DEFAULT_INTERVENTION_EFFECTIVENESS
            ).alias(f"{event_type}_expected_avoidable_cost")
        )

    out = out.with_columns(*avoidable_exprs)
    avoidable_cols = [
        f"{event_type}_expected_avoidable_cost"
        for event_type in [
            "fall",
            "rth",
            "wound",
            "altercation",
            "med_error",
            "elopement",
        ]
    ]

    return out.with_columns(
        composite_expected_cost=pl.sum_horizontal(expected_cols),
        expected_avoidable_cost=pl.sum_horizontal(avoidable_cols),
        economic_action=(pl.sum_horizontal(avoidable_cols) >= DEFAULT_ALERT_REVIEW_COST),
    )


def add_reason_codes(scored: pl.DataFrame) -> pl.DataFrame:
    contribution_cols = {
        "fall": "fall_expected_cost",
        "rth": "rth_expected_cost",
        "wound": "wound_expected_cost",
        "altercation": "altercation_expected_cost",
        "med_error": "med_error_expected_cost",
        "elopement": "elopement_expected_cost",
    }

    rule_cols = {
        "med_error": "med_error_flag",
        "elopement": "elopement_flag",
    }

    reason_codes = []
    top_reason = []

    def _positive_value(value) -> float:
        if value is None:
            return 0.0
        value = float(value)
        return value if np.isfinite(value) else 0.0

    for row in scored.iter_rows(named=True):
        contributions = [
            (event_type, _positive_value(row[col]))
            for event_type, col in contribution_cols.items()
            if _positive_value(row[col]) > 0
        ]
        contributions.sort(key=lambda item: item[1], reverse=True)

        reasons = [
            f"{'rule' if event in rule_cols else 'model'}:{event}"
            for event, _ in contributions[:3]
        ]
        for event, flag_col in rule_cols.items():
            if int(row.get(flag_col, 0)) == 1:
                reasons.append(f"rule:{event}")

        reason_codes.append("|".join(dict.fromkeys(reasons)))
        top_reason.append(contributions[0][0] if contributions else "")

    return scored.with_columns(
        pl.Series("reason_codes", reason_codes),
        pl.Series("top_reason", top_reason),
    )


def add_recommended_actions(scored: pl.DataFrame) -> pl.DataFrame:
    action_map = {
        "fall": "Fall-prevention review: reassess mobility, toileting plan, assistive devices, and recent vitals.",
        "rth": "Clinical escalation review: evaluate acute-change signs, recent transfers, abnormal labs, and provider follow-up.",
        "wound": "Skin-integrity review: inspect pressure areas, repositioning plan, nutrition/hydration, and wound-care orders.",
        "altercation": "Behavioral-care review: evaluate triggers, psychotropic monitoring, staffing plan, and resident interactions.",
        "med_error": "Medication reconciliation: review active medication list, recent missed/refused doses, and administration process.",
        "elopement": "Elopement precaution review: verify wander-risk care plan, supervision, exit controls, and recent admission adjustment.",
    }

    actions = []
    for row in scored.iter_rows(named=True):
        reasons = str(row.get("reason_codes", "")).split("|")
        ordered_events = []
        for reason in reasons:
            if ":" not in reason:
                continue
            _, event = reason.split(":", 1)
            if event in action_map and event not in ordered_events:
                ordered_events.append(event)

        if not ordered_events and row.get("top_reason") in action_map:
            ordered_events.append(row["top_reason"])

        actions.append(" | ".join(action_map[event] for event in ordered_events[:3]))

    return scored.with_columns(pl.Series("recommended_actions", actions))


def build_composite_scores(
    holdout_start: date = HOLDOUT_START,
    model_source: str = "mlflow",
    features_path: Path | None = None,
    model_phase: str = DEFAULT_MODEL_PHASE,
) -> pl.DataFrame:
    score_df, score_feature_cols = load_scoring_spine(holdout_start, features_path)
    if score_df.is_empty():
        raise ValueError("No scoring rows found.")

    score_start = score_df["window_start"].min()
    if hasattr(score_start, "date"):
        score_start_date = score_start.date()
    else:
        score_start_date = holdout_start

    print(f"Scoring {score_df.height:,} resident-windows from {score_start_date}...")
    scored = score_tier1_models(score_df, score_feature_cols, model_source, model_phase)

    cutoff = datetime.combine(score_start_date, datetime.min.time())
    print("  Training/scoring pre-score-date altercation propensity model...")
    altercation = score_altercation_as_of(score_df, cutoff, model_phase)
    scored = scored.join(altercation.drop("resident_id"), on="score_id", how="left")

    print("  Scoring point-in-time business rules...")
    rule_precisions = load_rule_precisions()
    rules = score_rules_for_spine(score_df.select(["score_id"] + META_COLS))
    scored = scored.join(
        rules.drop([c for c in META_COLS if c in rules.columns]),
        on="score_id",
        how="left",
    )

    scored = add_composite_costs(scored)
    scored = add_reason_codes(scored)
    scored = add_recommended_actions(scored)
    scored = scored.with_columns(
        pl.lit(rule_precisions["med_error"]).alias("med_error_rule_precision"),
        pl.lit(rule_precisions["elopement"]).alias("elopement_rule_precision"),
    )
    return scored.sort("composite_expected_cost", descending=True)


def main() -> None:
    args = parse_args()
    holdout_start = _to_date(args.holdout_start)
    features_path = Path(args.features_path) if args.features_path else None
    output = Path(args.output)
    top_output = Path(args.top_output) if args.top_output else output.with_name(f"{output.stem}_top.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    top_output.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    scored = build_composite_scores(
        holdout_start,
        args.model_source,
        features_path=features_path,
        model_phase=args.model_phase,
    )
    scored.write_parquet(output)

    top_cols = [
        "resident_id",
        "facility_id",
        "window_start",
        "composite_expected_cost",
        "expected_avoidable_cost",
        "top_reason",
        "reason_codes",
        "recommended_actions",
    ]
    scored.select(top_cols).head(50).write_csv(top_output)

    print(f"\nSaved composite scores to {output}")
    print(f"Saved top-score preview to {top_output}")


if __name__ == "__main__":
    main()
