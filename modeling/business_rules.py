"""
Tier 3 — Business Rules for Low-Frequency Incident Types
========================================================

Rule-based risk flags for incidents too rare for ML:
  - Medication Errors (46 events)
  - Elopement (10 events)

Each rule set produces a binary risk flag per resident.
Retrospective validation computes precision (PPV), recall, and coverage.

Usage:
    cd <project_root>
    python modeling/business_rules.py
"""

import sys
from datetime import datetime
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.business_policy import AVG_CLAIM_COST, load_rule_precisions

RAW = ROOT / "data" / "raw"
ARTIFACTS_DIR = ROOT / "modeling" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SIGNAL_END = datetime(2025, 2, 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════════


def load_raw():
    residents = pl.read_parquet(RAW / "residents.parquet")
    incidents = pl.read_parquet(RAW / "incidents.parquet").filter(pl.col("strikeout") == False)
    diagnoses = pl.read_parquet(RAW / "diagnoses.parquet").filter(pl.col("strikeout") == False)
    medications = pl.read_parquet(RAW / "medications.parquet")
    document_tags = pl.read_parquet(RAW / "document_tags.parquet").filter(pl.col("deleted_at").is_null())
    return residents, incidents, diagnoses, medications, document_tags


# ═══════════════════════════════════════════════════════════════════════════════
#  Medication Error Rules
# ═══════════════════════════════════════════════════════════════════════════════
#
# Risk factors for medication errors:
#   1. Polypharmacy: >= 9 distinct active medications
#   2. High missed/refused rate: > 10% of scheduled doses missed or refused
#   3. Cognitive impairment diagnosis (dementia, delirium)
#   4. Prior medication error history
#
# Flagged if >= 2 of the 4 rules trigger.


def medication_error_rules(residents, incidents, diagnoses, medications, doc_tags):
    base = residents.select("resident_id")

    # Rule 1: Polypharmacy — count distinct medication descriptions
    med_counts = (
        medications
        .group_by("resident_id")
        .agg(pl.col("description").n_unique().alias("n_distinct_meds"))
    )
    base = base.join(med_counts, on="resident_id", how="left").with_columns(
        pl.col("n_distinct_meds").fill_null(0)
    )
    base = base.with_columns(
        rule_polypharmacy=(pl.col("n_distinct_meds") >= 9).cast(pl.Int8)
    )

    # Rule 2: High missed/refused rate (>10%)
    med_status = (
        medications
        .group_by("resident_id")
        .agg(
            pl.len().alias("total_doses"),
            ((pl.col("status") == "Missed") | (pl.col("status") == "Refused"))
            .sum().cast(pl.Int32).alias("missed_refused"),
        )
        .with_columns(
            (pl.col("missed_refused") / pl.col("total_doses")).alias("missed_rate")
        )
    )
    base = base.join(
        med_status.select("resident_id", "missed_rate"), on="resident_id", how="left"
    ).with_columns(pl.col("missed_rate").fill_null(0.0))
    base = base.with_columns(
        rule_missed_rate=(pl.col("missed_rate") > 0.10).cast(pl.Int8)
    )

    # Rule 3: Cognitive impairment diagnosis
    cognitive_codes = ["F01", "F02", "F03", "F05", "G30", "G31"]
    cog_residents = (
        diagnoses
        .filter(
            pl.any_horizontal(
                [pl.col("icd_10_code").str.starts_with(code) for code in cognitive_codes]
            )
        )
        .select("resident_id")
        .unique()
        .with_columns(pl.lit(1).cast(pl.Int8).alias("rule_cognitive"))
    )
    base = base.join(cog_residents, on="resident_id", how="left").with_columns(
        pl.col("rule_cognitive").fill_null(0)
    )

    # Rule 4: Prior medication error history
    prior_med_err = (
        incidents
        .filter(pl.col("incident_type") == "Medication Error")
        .select("resident_id")
        .unique()
        .with_columns(pl.lit(1).cast(pl.Int8).alias("rule_prior_med_error"))
    )
    base = base.join(prior_med_err, on="resident_id", how="left").with_columns(
        pl.col("rule_prior_med_error").fill_null(0)
    )

    # Composite: flagged if >= 2 rules trigger
    rule_cols = ["rule_polypharmacy", "rule_missed_rate", "rule_cognitive", "rule_prior_med_error"]
    base = base.with_columns(
        rule_score=pl.sum_horizontal(rule_cols),
    ).with_columns(
        med_error_flag=(pl.col("rule_score") >= 2).cast(pl.Int8),
    )

    # Ground truth
    actual = (
        incidents
        .filter(pl.col("incident_type") == "Medication Error")
        .select("resident_id")
        .unique()
        .with_columns(pl.lit(1).cast(pl.Int8).alias("actual_med_error"))
    )
    base = base.join(actual, on="resident_id", how="left").with_columns(
        pl.col("actual_med_error").fill_null(0)
    )

    return base, rule_cols, "med_error_flag", "actual_med_error"


# ═══════════════════════════════════════════════════════════════════════════════
#  Elopement Rules
# ═══════════════════════════════════════════════════════════════════════════════
#
# Risk factors for elopement:
#   1. Dementia/Alzheimer's diagnosis (F01-F03, G30)
#   2. Wandering/elopement document tags
#   3. Prior elopement history
#   4. New admission (within 90 days)
#
# Flagged if >= 2 of the 4 rules trigger.


def elopement_rules(residents, incidents, diagnoses, doc_tags):
    base = residents.select("resident_id", "admission_date")

    # Rule 1: Dementia/Alzheimer's
    dementia_codes = ["F01", "F02", "F03", "G30"]
    dementia = (
        diagnoses
        .filter(
            pl.any_horizontal(
                [pl.col("icd_10_code").str.starts_with(code) for code in dementia_codes]
            )
        )
        .select("resident_id").unique()
        .with_columns(pl.lit(1).cast(pl.Int8).alias("rule_dementia"))
    )
    base = base.join(dementia, on="resident_id", how="left").with_columns(
        pl.col("rule_dementia").fill_null(0)
    )

    # Rule 2: Wandering/elopement tags
    wander_tags = ["wandering_risk_assessment", "elopement_incident", "elopement_risk"]
    wander = (
        doc_tags
        .filter(pl.col("tag_id").is_in(wander_tags))
        .select("resident_id").unique()
        .with_columns(pl.lit(1).cast(pl.Int8).alias("rule_wander_tag"))
    )
    base = base.join(wander, on="resident_id", how="left").with_columns(
        pl.col("rule_wander_tag").fill_null(0)
    )

    # Rule 3: Prior elopement
    prior_elop = (
        incidents
        .filter(pl.col("incident_type") == "Elopement")
        .select("resident_id").unique()
        .with_columns(pl.lit(1).cast(pl.Int8).alias("rule_prior_elopement"))
    )
    base = base.join(prior_elop, on="resident_id", how="left").with_columns(
        pl.col("rule_prior_elopement").fill_null(0)
    )

    # Rule 4: Recent admission (within 90 days of SIGNAL_END)
    base = base.with_columns(
        rule_new_admission=(
            (pl.lit(SIGNAL_END) - pl.col("admission_date")).dt.total_days() <= 90
        ).cast(pl.Int8)
    )

    # Composite
    rule_cols = ["rule_dementia", "rule_wander_tag", "rule_prior_elopement", "rule_new_admission"]
    base = base.with_columns(
        rule_score=pl.sum_horizontal(rule_cols),
    ).with_columns(
        elopement_flag=(pl.col("rule_score") >= 2).cast(pl.Int8),
    )

    # Ground truth
    actual = (
        incidents
        .filter(pl.col("incident_type") == "Elopement")
        .select("resident_id").unique()
        .with_columns(pl.lit(1).cast(pl.Int8).alias("actual_elopement"))
    )
    base = base.join(actual, on="resident_id", how="left").with_columns(
        pl.col("actual_elopement").fill_null(0)
    )

    return base.drop("admission_date"), rule_cols, "elopement_flag", "actual_elopement"


# ═══════════════════════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════════════════════


def validate_rule(df, rule_cols, flag_col, actual_col, name):
    """Compute precision, recall, and per-rule metrics for a rule set."""
    n_total = df.shape[0]
    n_flagged = df.filter(pl.col(flag_col) == 1).shape[0]
    n_actual = df.filter(pl.col(actual_col) == 1).shape[0]

    tp = df.filter((pl.col(flag_col) == 1) & (pl.col(actual_col) == 1)).shape[0]
    fp = df.filter((pl.col(flag_col) == 1) & (pl.col(actual_col) == 0)).shape[0]
    fn = df.filter((pl.col(flag_col) == 0) & (pl.col(actual_col) == 1)).shape[0]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    coverage = n_flagged / n_total if n_total > 0 else 0.0

    print(f"\n{'=' * 60}")
    print(f"  {name.upper()}")
    print(f"{'=' * 60}")
    print(f"  Total residents:  {n_total:,}")
    print(f"  Actual positives: {n_actual}")
    print(f"  Flagged:          {n_flagged} ({coverage:.1%} of all residents)")
    print(f"  True positives:   {tp}")
    print(f"  False positives:  {fp}")
    print(f"  False negatives:  {fn}")
    print(f"  Precision (PPV):  {precision:.2%}")
    print(f"  Recall:           {recall:.2%}")

    # Per-rule coverage
    print(f"\n  Per-rule breakdown:")
    for rule in rule_cols:
        n_rule = df.filter(pl.col(rule) == 1).shape[0]
        tp_rule = df.filter((pl.col(rule) == 1) & (pl.col(actual_col) == 1)).shape[0]
        pct = n_rule / n_total if n_total > 0 else 0.0
        ppv_rule = tp_rule / n_rule if n_rule > 0 else 0.0
        print(f"    {rule:30s}  flagged={n_rule:5d} ({pct:5.1%})  captured={tp_rule:3d}  PPV={ppv_rule:.2%}")

    return {
        "name": name,
        "n_total": n_total,
        "n_actual": n_actual,
        "n_flagged": n_flagged,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "coverage": coverage,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════


# ============================================================================
#  Point-in-Time Scoring
# ============================================================================

RULE_META_COLS = [
    "score_id",
    "resident_id",
    "facility_id",
    "window_start",
    "window_end",
    "feature_cutoff",
]


def _ensure_scoring_spine(spine: pl.DataFrame) -> pl.DataFrame:
    """Prepare a resident-window spine for point-in-time rule scoring."""

    missing = [c for c in RULE_META_COLS[1:] if c not in spine.columns]
    if missing:
        raise ValueError(f"Rule scoring spine is missing columns: {missing}")

    if "score_id" not in spine.columns:
        spine = spine.with_row_index("score_id")
    return spine.select(RULE_META_COLS)


def _prefix_expr(column: str, prefixes: list[str]) -> pl.Expr:
    return pl.any_horizontal(
        [pl.col(column).str.starts_with(prefix) for prefix in prefixes]
    )


def _active_diagnoses(spine: pl.DataFrame, diagnoses: pl.DataFrame) -> pl.DataFrame:
    return (
        spine.select("score_id", "resident_id", "feature_cutoff")
        .join(
            diagnoses.select("resident_id", "icd_10_code", "onset_at", "resolved_at"),
            on="resident_id",
            how="left",
        )
        .filter(pl.col("icd_10_code").is_not_null())
        .filter(pl.col("onset_at").is_null() | (pl.col("onset_at") <= pl.col("feature_cutoff")))
        .filter(pl.col("resolved_at").is_null() | (pl.col("resolved_at") > pl.col("feature_cutoff")))
    )


def _binary_group(df: pl.DataFrame, col_name: str) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame(schema={"score_id": pl.UInt32, col_name: pl.Int8})
    return df.group_by("score_id").agg(pl.lit(1).cast(pl.Int8).alias(col_name))


def score_point_in_time_rules(
    spine: pl.DataFrame,
    residents: pl.DataFrame,
    incidents: pl.DataFrame,
    diagnoses: pl.DataFrame,
    medications: pl.DataFrame,
    document_tags: pl.DataFrame,
    rule_precisions: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Score rare-event business rules at each row's feature cutoff.

    This scorer only uses records available at or before `feature_cutoff`,
    making it suitable for holdout scoring and production-style batch scoring.
    """

    base = _ensure_scoring_spine(spine)
    out = base.clone()
    precisions = rule_precisions or load_rule_precisions()

    med_window = (
        base.select("score_id", "resident_id", "feature_cutoff")
        .join(
            medications.select("resident_id", "description", "scheduled_at", "status"),
            on="resident_id",
            how="left",
        )
        .filter(pl.col("scheduled_at").is_not_null())
        .filter(pl.col("scheduled_at") <= pl.col("feature_cutoff"))
        .filter(pl.col("scheduled_at") >= pl.col("feature_cutoff") - pl.duration(days=14))
    )
    if med_window.is_empty():
        med_agg = pl.DataFrame(
            schema={
                "score_id": pl.UInt32,
                "n_distinct_meds_14d": pl.UInt32,
                "total_doses_14d": pl.UInt32,
                "missed_refused_14d": pl.UInt32,
            }
        )
    else:
        med_agg = med_window.group_by("score_id").agg(
            pl.col("description").n_unique().alias("n_distinct_meds_14d"),
            pl.len().alias("total_doses_14d"),
            ((pl.col("status") == "Missed") | (pl.col("status") == "Refused"))
            .sum()
            .alias("missed_refused_14d"),
        )

    out = out.join(med_agg, on="score_id", how="left").with_columns(
        pl.col("n_distinct_meds_14d").fill_null(0),
        pl.col("total_doses_14d").fill_null(0),
        pl.col("missed_refused_14d").fill_null(0),
    )
    out = out.with_columns(
        med_missed_refused_rate_14d=pl.when(pl.col("total_doses_14d") > 0)
        .then(pl.col("missed_refused_14d") / pl.col("total_doses_14d"))
        .otherwise(0.0)
    )
    out = out.with_columns(
        rule_polypharmacy=(pl.col("n_distinct_meds_14d") >= 9).cast(pl.Int8),
        rule_missed_rate=(pl.col("med_missed_refused_rate_14d") > 0.10).cast(pl.Int8),
    )

    dx_active = _active_diagnoses(base, diagnoses)
    cognitive_codes = ["F01", "F02", "F03", "F05", "G30", "G31"]
    cognitive = _binary_group(
        dx_active.filter(_prefix_expr("icd_10_code", cognitive_codes)),
        "rule_cognitive",
    )
    out = out.join(cognitive, on="score_id", how="left").with_columns(
        pl.col("rule_cognitive").fill_null(0)
    )

    prior_med_error = _binary_group(
        base.select("score_id", "resident_id", "feature_cutoff")
        .join(
            incidents.select("resident_id", "incident_type", "occurred_at"),
            on="resident_id",
            how="left",
        )
        .filter(pl.col("incident_type") == "Medication Error")
        .filter(pl.col("occurred_at") <= pl.col("feature_cutoff")),
        "rule_prior_med_error",
    )
    out = out.join(prior_med_error, on="score_id", how="left").with_columns(
        pl.col("rule_prior_med_error").fill_null(0)
    )

    med_rule_cols = [
        "rule_polypharmacy",
        "rule_missed_rate",
        "rule_cognitive",
        "rule_prior_med_error",
    ]
    out = out.with_columns(
        med_error_rule_score=pl.sum_horizontal(med_rule_cols),
    ).with_columns(
        med_error_flag=(pl.col("med_error_rule_score") >= 2).cast(pl.Int8)
    )

    tags_before_cutoff = (
        base.select("score_id", "resident_id", "feature_cutoff")
        .join(
            document_tags.select("resident_id", "tag_id", "created_at"),
            on="resident_id",
            how="left",
        )
        .filter(pl.col("tag_id").is_not_null())
        .filter(pl.col("created_at") <= pl.col("feature_cutoff"))
    )

    dementia_codes = ["F01", "F02", "F03", "G30"]
    dementia = _binary_group(
        dx_active.filter(_prefix_expr("icd_10_code", dementia_codes)),
        "rule_dementia",
    )
    wander_tags = ["wandering_risk_assessment", "elopement_incident", "elopement_risk"]
    wander = _binary_group(
        tags_before_cutoff.filter(pl.col("tag_id").is_in(wander_tags)),
        "rule_wander_tag",
    )
    prior_elopement = _binary_group(
        base.select("score_id", "resident_id", "feature_cutoff")
        .join(
            incidents.select("resident_id", "incident_type", "occurred_at"),
            on="resident_id",
            how="left",
        )
        .filter(pl.col("incident_type") == "Elopement")
        .filter(pl.col("occurred_at") <= pl.col("feature_cutoff")),
        "rule_prior_elopement",
    )
    out = (
        out.join(dementia, on="score_id", how="left")
        .join(wander, on="score_id", how="left")
        .join(prior_elopement, on="score_id", how="left")
        .join(residents.select("resident_id", "admission_date"), on="resident_id", how="left")
        .with_columns(
            pl.col("rule_dementia").fill_null(0),
            pl.col("rule_wander_tag").fill_null(0),
            pl.col("rule_prior_elopement").fill_null(0),
            rule_new_admission=(
                ((pl.col("window_start") - pl.col("admission_date")).dt.total_days() >= 0)
                & ((pl.col("window_start") - pl.col("admission_date")).dt.total_days() <= 90)
            )
            .fill_null(False)
            .cast(pl.Int8),
        )
        .drop("admission_date")
    )

    elopement_rule_cols = [
        "rule_dementia",
        "rule_wander_tag",
        "rule_prior_elopement",
        "rule_new_admission",
    ]
    out = out.with_columns(
        elopement_rule_score=pl.sum_horizontal(elopement_rule_cols),
    ).with_columns(
        elopement_flag=(pl.col("elopement_rule_score") >= 2).cast(pl.Int8)
    )

    out = out.with_columns(
        med_error_rule_probability=pl.when(pl.col("med_error_flag") == 1)
        .then(pl.lit(float(precisions["med_error"])))
        .otherwise(0.0),
        elopement_rule_probability=pl.when(pl.col("elopement_flag") == 1)
        .then(pl.lit(float(precisions["elopement"])))
        .otherwise(0.0),
    ).with_columns(
        med_error_rule_expected_cost=pl.col("med_error_rule_probability")
        * AVG_CLAIM_COST["med_error"],
        elopement_rule_expected_cost=pl.col("elopement_rule_probability")
        * AVG_CLAIM_COST["elopement"],
    )

    return out


def score_rules_for_spine(spine: pl.DataFrame) -> pl.DataFrame:
    """Load raw inputs and score point-in-time rules for a scoring spine."""

    residents, incidents, diagnoses, medications, doc_tags = load_raw()
    return score_point_in_time_rules(
        spine=spine,
        residents=residents,
        incidents=incidents,
        diagnoses=diagnoses,
        medications=medications,
        document_tags=doc_tags,
    )


def main():
    print("Loading raw tables...")
    residents, incidents, diagnoses, medications, doc_tags = load_raw()
    print(f"  {residents.shape[0]:,} residents")

    results = []

    # Medication Errors
    df_med, rules_med, flag_med, actual_med = medication_error_rules(
        residents, incidents, diagnoses, medications, doc_tags
    )
    results.append(validate_rule(df_med, rules_med, flag_med, actual_med, "Medication Errors"))

    # Elopement
    df_elop, rules_elop, flag_elop, actual_elop = elopement_rules(
        residents, incidents, diagnoses, doc_tags
    )
    results.append(validate_rule(df_elop, rules_elop, flag_elop, actual_elop, "Elopement"))

    # Summary table
    import pandas as pd
    summary = pd.DataFrame(results)
    summary_path = ARTIFACTS_DIR / "business_rules_validation.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n  Summary saved to {summary_path}")

    print("\n  SUMMARY")
    print(f"  {'Rule Set':<20s}  {'Flagged':>8s}  {'TP':>4s}  {'Precision':>10s}  {'Recall':>8s}  {'Coverage':>9s}")
    for r in results:
        print(
            f"  {r['name']:<20s}  {r['n_flagged']:8d}  {r['tp']:4d}  "
            f"{r['precision']:10.2%}  {r['recall']:8.2%}  {r['coverage']:9.1%}"
        )


if __name__ == "__main__":
    main()
