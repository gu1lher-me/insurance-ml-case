"""
Tier 3 — Business Rules for Low-Frequency Incident Types
========================================================

Rule-based risk flags for incidents too rare for ML:
  - Medication Errors (46 events)
  - Choking (9 events)
  - Elopement (10 events)

Each rule set produces a binary risk flag per resident.
Retrospective validation computes precision (PPV), recall, and coverage.

Usage:
    cd <project_root>
    python modeling/business_rules.py
"""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
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
    physician_orders = pl.read_parquet(RAW / "physician_orders.parquet")
    needs = pl.read_parquet(RAW / "needs.parquet").filter(pl.col("strikeout") == False)
    document_tags = pl.read_parquet(RAW / "document_tags.parquet").filter(pl.col("deleted_at").is_null())
    return residents, incidents, diagnoses, medications, physician_orders, needs, document_tags


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
#  Choking Rules
# ═══════════════════════════════════════════════════════════════════════════════
#
# Risk factors for choking:
#   1. Dysphagia diagnosis (R13.x)
#   2. Dietary texture modification order (Dietary - Diet category)
#   3. Speech therapy tag or order
#   4. Neurological diagnosis (G20 Parkinson's, G30 Alzheimer's, I63 stroke, G40 epilepsy)
#   5. Choking/aspiration-related document tags
#
# Flagged if >= 2 of the 5 rules trigger.


def choking_rules(residents, incidents, diagnoses, physician_orders, doc_tags):
    base = residents.select("resident_id")

    # Rule 1: Dysphagia diagnosis (R13.x)
    dysphagia = (
        diagnoses
        .filter(pl.col("icd_10_code").str.starts_with("R13"))
        .select("resident_id").unique()
        .with_columns(pl.lit(1).cast(pl.Int8).alias("rule_dysphagia"))
    )
    base = base.join(dysphagia, on="resident_id", how="left").with_columns(
        pl.col("rule_dysphagia").fill_null(0)
    )

    # Rule 2: Dietary modification order
    diet_orders = (
        physician_orders
        .filter(pl.col("category").str.contains("Dietary"))
        .select("resident_id").unique()
        .with_columns(pl.lit(1).cast(pl.Int8).alias("rule_diet_order"))
    )
    base = base.join(diet_orders, on="resident_id", how="left").with_columns(
        pl.col("rule_diet_order").fill_null(0)
    )

    # Rule 3: Speech therapy tag
    speech = (
        doc_tags
        .filter(pl.col("tag_id") == "speech_therapy")
        .select("resident_id").unique()
        .with_columns(pl.lit(1).cast(pl.Int8).alias("rule_speech_therapy"))
    )
    base = base.join(speech, on="resident_id", how="left").with_columns(
        pl.col("rule_speech_therapy").fill_null(0)
    )

    # Rule 4: Neurological diagnoses
    neuro_codes = ["G20", "G30", "I63", "G40", "G35", "F03"]
    neuro = (
        diagnoses
        .filter(
            pl.any_horizontal(
                [pl.col("icd_10_code").str.starts_with(code) for code in neuro_codes]
            )
        )
        .select("resident_id").unique()
        .with_columns(pl.lit(1).cast(pl.Int8).alias("rule_neuro_dx"))
    )
    base = base.join(neuro, on="resident_id", how="left").with_columns(
        pl.col("rule_neuro_dx").fill_null(0)
    )

    # Rule 5: Choking-related document tags
    choking_tags = ["choking", "choking_incident", "downgraded_diet"]
    choke_tag = (
        doc_tags
        .filter(pl.col("tag_id").is_in(choking_tags))
        .select("resident_id").unique()
        .with_columns(pl.lit(1).cast(pl.Int8).alias("rule_choking_tag"))
    )
    base = base.join(choke_tag, on="resident_id", how="left").with_columns(
        pl.col("rule_choking_tag").fill_null(0)
    )

    # Composite
    rule_cols = ["rule_dysphagia", "rule_diet_order", "rule_speech_therapy", "rule_neuro_dx", "rule_choking_tag"]
    base = base.with_columns(
        rule_score=pl.sum_horizontal(rule_cols),
    ).with_columns(
        choking_flag=(pl.col("rule_score") >= 3).cast(pl.Int8),
    )

    # Ground truth
    actual = (
        incidents
        .filter(pl.col("incident_type") == "Choking")
        .select("resident_id").unique()
        .with_columns(pl.lit(1).cast(pl.Int8).alias("actual_choking"))
    )
    base = base.join(actual, on="resident_id", how="left").with_columns(
        pl.col("actual_choking").fill_null(0)
    )

    return base, rule_cols, "choking_flag", "actual_choking"


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


def main():
    print("Loading raw tables...")
    residents, incidents, diagnoses, medications, orders, needs, doc_tags = load_raw()
    print(f"  {residents.shape[0]:,} residents")

    results = []

    # Medication Errors
    df_med, rules_med, flag_med, actual_med = medication_error_rules(
        residents, incidents, diagnoses, medications, doc_tags
    )
    results.append(validate_rule(df_med, rules_med, flag_med, actual_med, "Medication Errors"))

    # Choking
    df_choke, rules_choke, flag_choke, actual_choke = choking_rules(
        residents, incidents, diagnoses, orders, doc_tags
    )
    results.append(validate_rule(df_choke, rules_choke, flag_choke, actual_choke, "Choking"))

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
