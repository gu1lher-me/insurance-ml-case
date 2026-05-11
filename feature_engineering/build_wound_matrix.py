"""
Feature Matrix — 14-day Wound Model
====================================

Generates:
  - `data/feature_store/features_14d.parquet` with target-free features
  - `data/processed/model_matrix_14d.parquet` with model-ready targets + features

Artifacts use the same features as the 7-day matrix but:
  - stride = 14 days (non-overlapping)
  - target = wound_14d (any Wound incident in [t, t+14d))

Usage:
    cd <project_root>
    python feature_engineering/build_wound_matrix.py
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

try:
    from .admission_features import compute_admission_context
except ImportError:
    from admission_features import compute_admission_context

# ─── Configuration ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
FEATURE_OUT = ROOT / "data" / "feature_store" / "features_14d.parquet"
MODEL_OUT = ROOT / "data" / "processed" / "model_matrix_14d.parquet"

SIGNAL_START = datetime(2023, 7, 1)
SIGNAL_END = datetime(2025, 2, 1)
HORIZON = timedelta(days=14)
EMBARGO_DAYS = 1

VITALS_CLIP = {
    "BP - Systolic": (60, 250),
    "Pulse": (20, 200),
    "O2 sats": (50, 100),
    "Blood Sugar": (20, 600),
    "Temperature": (90, 108),
    "Respiration": (5, 60),
    "Pain Level": (0, 10),
    "Weight": (50, 500),
}
VITAL_LOOKBACKS = [3, 7, 14]
HISTORY_LOOKBACKS = {"7d": timedelta(days=7), "30d": timedelta(days=30), "90d": timedelta(days=90)}
INCIDENT_TYPES = ["Fall", "Wound", "Altercation"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Load raw tables
# ═══════════════════════════════════════════════════════════════════════════════


def load_tables():
    residents = pl.read_parquet(RAW / "residents.parquet")
    vitals = pl.read_parquet(RAW / "vitals.parquet").filter(pl.col("strikeout") == False)
    incidents = pl.read_parquet(RAW / "incidents.parquet").filter(pl.col("strikeout") == False)
    diagnoses = pl.read_parquet(RAW / "diagnoses.parquet").filter(pl.col("strikeout") == False)
    hospital_transfers = pl.read_parquet(RAW / "hospital_transfers.parquet")
    hospital_admissions = pl.read_parquet(RAW / "hospital_admissions.parquet")
    needs = pl.read_parquet(RAW / "needs.parquet").filter(pl.col("strikeout") == False)
    lab_reports = pl.read_parquet(RAW / "lab_reports.parquet")
    document_tags = pl.read_parquet(RAW / "document_tags.parquet").filter(pl.col("deleted_at").is_null())

    return (
        residents,
        vitals,
        incidents,
        diagnoses,
        hospital_transfers,
        hospital_admissions,
        needs,
        lab_reports,
        document_tags,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build 14-day historical features and labels for wounds."
    )
    parser.add_argument("--signal-start", default=SIGNAL_START.date().isoformat())
    parser.add_argument("--signal-end", default=SIGNAL_END.date().isoformat())
    parser.add_argument("--feature-out", default=str(FEATURE_OUT))
    parser.add_argument("--model-out", default=str(MODEL_OUT))
    return parser.parse_args()


def _parse_date(value):
    return datetime.strptime(value, "%Y-%m-%d")


# ═══════════════════════════════════════════════════════════════════════════════
#  Observation Spine
# ═══════════════════════════════════════════════════════════════════════════════


def build_resident_obs_bounds(residents):
    """Compute per-resident observation start/end dates."""
    return residents.select(
        "resident_id",
        "facility_id",
        pl.max_horizontal(
            pl.col("admission_date").cast(pl.Datetime("us")),
            pl.lit(SIGNAL_START),
        ).alias("obs_start"),
        pl.min_horizontal(
            *[
                pl.col(c).cast(pl.Datetime("us"))
                for c in ["discharge_date", "deceased_date"]
                if c in residents.columns
            ],
            pl.lit(SIGNAL_END),
        ).alias("obs_end"),
    ).filter(pl.col("obs_start") < pl.col("obs_end"))


def generate_observation_windows(residents_df):
    rows = []
    for row in residents_df.iter_rows(named=True):
        rid = row["resident_id"]
        fid = row["facility_id"]
        start = row["obs_start"]
        end = row["obs_end"]

        t = start
        while t + HORIZON <= end:
            rows.append({
                "resident_id": rid,
                "facility_id": fid,
                "window_start": t,
                "window_end": t + HORIZON,
                "feature_cutoff": t - timedelta(days=EMBARGO_DAYS),
            })
            t += HORIZON

    return pl.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  Target: wound_14d
# ═══════════════════════════════════════════════════════════════════════════════


def create_wound_labels(windows, incidents):
    wounds = incidents.filter(pl.col("incident_type") == "Wound")

    wound_labels = (
        windows
        .join(wounds.select("resident_id", "occurred_at"), on="resident_id", how="left")
        .filter(
            pl.col("occurred_at").is_null()
            | (
                (pl.col("occurred_at") >= pl.col("window_start"))
                & (pl.col("occurred_at") < pl.col("window_end"))
            )
        )
        .group_by("resident_id", "window_start")
        .agg(pl.col("occurred_at").is_not_null().any().cast(pl.Int8).alias("wound_14d"))
    )

    return (
        windows
        .join(wound_labels, on=["resident_id", "window_start"], how="left")
        .with_columns(pl.col("wound_14d").fill_null(0))
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Features (same logic as 7d matrix)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_demographics(obs, residents):
    return (
        obs
        .join(residents.select("resident_id", "date_of_birth", "admission_date"), on="resident_id", how="left")
        .with_columns(
            age_at_window=((pl.col("window_start") - pl.col("date_of_birth").cast(pl.Datetime("us"))).dt.total_days() / 365.25).round(1),
            los_days=(pl.col("window_start") - pl.col("admission_date").cast(pl.Datetime("us"))).dt.total_days(),
        )
        .select("resident_id", "window_start", "age_at_window", "los_days")
    )


def compute_vitals_features(obs, vitals):
    # Clip vitals
    clip_exprs = []
    for vtype, (lo, hi) in VITALS_CLIP.items():
        clip_exprs.append(
            pl.when(pl.col("vital_type") == vtype)
            .then(pl.col("value").clip(lo, hi))
            .otherwise(pl.col("value"))
        )
    vitals_clipped = vitals.with_columns(
        pl.coalesce(clip_exprs).alias("value")
    )

    result = obs.select("resident_id", "window_start", "feature_cutoff")

    for lb_days in VITAL_LOOKBACKS:
        lb = timedelta(days=lb_days)
        suffix = f"_{lb_days}d"

        joined = (
            obs.select("resident_id", "window_start", "feature_cutoff")
            .join(
                vitals_clipped.select("resident_id", "vital_type", "value", "measured_at"),
                on="resident_id",
                how="left",
            )
            .filter(
                pl.col("measured_at").is_not_null()
                & (pl.col("measured_at") <= pl.col("feature_cutoff"))
                & (pl.col("measured_at") > (pl.col("feature_cutoff") - lb))
            )
        )

        agg = (
            joined
            .group_by("resident_id", "window_start", "vital_type")
            .agg(
                pl.col("value").mean().alias("mean"),
                pl.col("value").std().alias("std"),
                pl.col("value").min().alias("min"),
                pl.col("value").max().alias("max"),
                pl.col("value").count().alias("count"),
            )
        )

        # Pivot stats
        pivoted = (
            agg.unpivot(
                index=["resident_id", "window_start", "vital_type"],
                on=["mean", "std", "min", "max", "count"],
                variable_name="stat",
                value_name="val",
            )
            .with_columns(
                (pl.col("vital_type").str.to_lowercase().str.replace_all(r"[\s\-]", "_")
                 + "_" + pl.col("stat") + pl.lit(suffix)).alias("col_name")
            )
            .pivot(on="col_name", index=["resident_id", "window_start"], values="val")
        )
        result = result.join(pivoted, on=["resident_id", "window_start"], how="left")

        # Measured flags
        measured = (
            agg.select("resident_id", "window_start", "vital_type")
            .with_columns(
                (pl.col("vital_type").str.to_lowercase().str.replace_all(r"[\s\-]", "_")
                 + pl.lit(f"_measured{suffix}")).alias("flag_name"),
                pl.lit(1).alias("measured"),
            )
            .pivot(on="flag_name", index=["resident_id", "window_start"], values="measured")
        )
        flag_cols = [c for c in measured.columns if c not in ("resident_id", "window_start")]
        result = result.join(measured, on=["resident_id", "window_start"], how="left")
        result = result.with_columns([pl.col(c).fill_null(0).cast(pl.Int8) for c in flag_cols])

    return result.drop("feature_cutoff")


def compute_diagnoses(obs, diagnoses):
    dx = diagnoses.select("resident_id", "icd_10_code", "onset_at", "resolved_at")

    active = (
        obs.select("resident_id", "window_start", "feature_cutoff")
        .join(dx, on="resident_id", how="left")
        .filter(
            pl.col("icd_10_code").is_not_null()
            & (pl.col("onset_at") <= pl.col("feature_cutoff"))
            & ((pl.col("resolved_at").is_null()) | (pl.col("resolved_at") > pl.col("feature_cutoff")))
        )
    )

    counts = (
        active
        .group_by("resident_id", "window_start")
        .agg(
            pl.col("icd_10_code").n_unique().alias("dx_active_count"),
            pl.col("icd_10_code").str.slice(0, 1).n_unique().alias("dx_chapter_count"),
        )
    )

    # Fall-risk flags
    fall_codes = {"R26": "dx_fall_R26", "R27": "dx_fall_R27", "M62.81": "dx_fall_M62_81",
                  "R29.6": "dx_fall_R29_6", "H81": "dx_fall_H81", "G40": "dx_fall_G40",
                  "F01": "dx_fall_F01", "F02": "dx_fall_F02", "F03": "dx_fall_F03", "G30": "dx_fall_G30"}
    rth_codes = {"I50": "dx_rth_I50", "J44": "dx_rth_J44", "I10": "dx_rth_I10",
                 "E11": "dx_rth_E11", "N18": "dx_rth_N18", "J18": "dx_rth_J18", "I63": "dx_rth_I63"}

    flag_exprs = []
    fall_flag_names = []
    rth_flag_names = []

    for prefix, col_name in fall_codes.items():
        flag_exprs.append(
            pl.col("icd_10_code").str.starts_with(prefix).any().cast(pl.Int8).alias(col_name)
        )
        fall_flag_names.append(col_name)

    for prefix, col_name in rth_codes.items():
        flag_exprs.append(
            pl.col("icd_10_code").str.starts_with(prefix).any().cast(pl.Int8).alias(col_name)
        )
        rth_flag_names.append(col_name)

    flags = active.group_by("resident_id", "window_start").agg(flag_exprs)

    result = obs.select("resident_id", "window_start").join(counts, on=["resident_id", "window_start"], how="left")
    result = result.join(flags, on=["resident_id", "window_start"], how="left")
    result = result.with_columns(pl.col("dx_active_count").fill_null(0))
    result = result.with_columns(
        pl.sum_horizontal([pl.col(c) for c in fall_flag_names]).fill_null(0).alias("dx_fall_risk_score"),
        pl.sum_horizontal([pl.col(c) for c in rth_flag_names]).fill_null(0).alias("dx_rth_risk_score"),
    )

    return result


def compute_incident_history(obs, incidents_df):
    base = obs.select("resident_id", "window_start", "feature_cutoff")
    joined = (
        base.join(
            incidents_df.select("resident_id", "incident_type", "occurred_at"),
            on="resident_id", how="left",
        )
        .filter(
            pl.col("occurred_at").is_not_null()
            & (pl.col("occurred_at") <= pl.col("feature_cutoff"))
        )
    )

    agg_exprs = []
    for itype in INCIDENT_TYPES:
        safe = itype.lower()
        agg_exprs.append(
            (pl.col("incident_type") == itype).sum().cast(pl.Int32).alias(f"hist_{safe}_all")
        )
        for lb_name, lb_delta in HISTORY_LOOKBACKS.items():
            agg_exprs.append(
                ((pl.col("incident_type") == itype) & (pl.col("occurred_at") > (pl.col("feature_cutoff") - lb_delta)))
                .sum().cast(pl.Int32).alias(f"hist_{safe}_{lb_name}")
            )

    agg_exprs.append(pl.len().cast(pl.Int32).alias("hist_incident_total"))
    history = joined.group_by("resident_id", "window_start").agg(agg_exprs)

    result = obs.select("resident_id", "window_start").join(history, on=["resident_id", "window_start"], how="left")
    hist_cols = [c for c in result.columns if c.startswith("hist_")]
    return result.with_columns([pl.col(c).fill_null(0) for c in hist_cols])


def compute_rth_history(obs, transfers):
    base = obs.select("resident_id", "window_start", "feature_cutoff")
    joined = (
        base.join(transfers.select("resident_id", "effective_date", "emergency_flag"), on="resident_id", how="left")
        .filter(pl.col("effective_date").is_not_null() & (pl.col("effective_date") <= pl.col("feature_cutoff")))
    )
    history = joined.group_by("resident_id", "window_start").agg(
        pl.len().cast(pl.Int32).alias("hist_rth_all"),
        (pl.col("effective_date") > (pl.col("feature_cutoff") - timedelta(days=30))).sum().cast(pl.Int32).alias("hist_rth_30d"),
        (pl.col("effective_date") > (pl.col("feature_cutoff") - timedelta(days=90))).sum().cast(pl.Int32).alias("hist_rth_90d"),
        (pl.col("emergency_flag") == True).sum().cast(pl.Int32).alias("hist_rth_emergency_all"),
    )
    result = obs.select("resident_id", "window_start").join(history, on=["resident_id", "window_start"], how="left")
    rth_cols = [c for c in result.columns if c.startswith("hist_rth")]
    return result.with_columns([pl.col(c).fill_null(0) for c in rth_cols])


def compute_needs(obs, needs):
    base = obs.select("resident_id", "window_start", "feature_cutoff")
    active_needs = (
        base.join(needs.select("resident_id", "need_type", "initiated_at", "resolved_at"), on="resident_id", how="left")
        .filter(
            pl.col("need_type").is_not_null()
            & (pl.col("initiated_at") <= pl.col("feature_cutoff"))
            & ((pl.col("resolved_at").is_null()) | (pl.col("resolved_at") > pl.col("feature_cutoff")))
        )
    )
    agg = active_needs.group_by("resident_id", "window_start").agg(
        pl.len().cast(pl.Int32).alias("needs_open_total"),
        (pl.col("need_type").str.to_lowercase().str.contains("fall")).sum().cast(pl.Int32).alias("needs_open_fall"),
        (pl.col("need_type").str.to_lowercase().str.contains("wound")).sum().cast(pl.Int32).alias("needs_open_wound"),
        (pl.col("need_type").str.to_lowercase().str.contains("nutrition")).sum().cast(pl.Int32).alias("needs_open_nutrition"),
    )
    result = obs.select("resident_id", "window_start").join(agg, on=["resident_id", "window_start"], how="left")
    need_cols = [c for c in result.columns if c.startswith("needs_")]
    result = result.with_columns([pl.col(c).fill_null(0) for c in need_cols])
    result = result.with_columns(
        (pl.col("needs_open_total") - pl.col("needs_open_fall") - pl.col("needs_open_wound") - pl.col("needs_open_nutrition"))
        .clip(0, None).alias("needs_open_other")
    )
    return result


def compute_labs(obs, lab_reports):
    base = obs.select("resident_id", "window_start", "feature_cutoff")
    joined = (
        base.join(lab_reports.select("resident_id", "reported_at", "severity_status"), on="resident_id", how="left")
        .filter(pl.col("reported_at").is_not_null() & (pl.col("reported_at") <= pl.col("feature_cutoff")))
    )

    feats_list = []
    for lb_days in [14, 30]:
        lb = timedelta(days=lb_days)
        suffix = f"_{lb_days}d"
        windowed = joined.filter(pl.col("reported_at") > (pl.col("feature_cutoff") - lb))
        agg = windowed.group_by("resident_id", "window_start").agg(
            pl.len().cast(pl.Int32).alias(f"labs_total{suffix}"),
            (pl.col("severity_status") == "Abnormal").sum().cast(pl.Int32).alias(f"labs_abnormal{suffix}"),
            (pl.col("severity_status") == "Critical").sum().cast(pl.Int32).alias(f"labs_critical{suffix}"),
        )
        feats_list.append(agg)

    result = obs.select("resident_id", "window_start")
    for f in feats_list:
        result = result.join(f, on=["resident_id", "window_start"], how="left")
    lab_cols = [c for c in result.columns if c.startswith("labs_")]
    return result.with_columns([pl.col(c).fill_null(0) for c in lab_cols])


def compute_document_tags(obs, document_tags):
    TAG_LIST = [
        "actual_fall", "fall_risk", "actual_wound", "wound_risk",
        "aggressive_behavior", "alzheimers_disease", "dementia", "depression",
        "anxiety", "pain", "infection", "pressure_injury",
        "skin_tear", "bruise", "wandering", "elopement",
    ]

    base = obs.select("resident_id", "window_start", "feature_cutoff")
    joined = (
        base.join(document_tags.select("resident_id", "tag_id", "created_at"), on="resident_id", how="left")
        .filter(pl.col("tag_id").is_not_null() & (pl.col("created_at") <= pl.col("feature_cutoff")))
    )

    agg_exprs = [
        (pl.col("tag_id") == tag).any().cast(pl.Int8).alias(f"tag_{tag}")
        for tag in TAG_LIST
    ]

    tags = joined.group_by("resident_id", "window_start").agg(agg_exprs)
    result = obs.select("resident_id", "window_start").join(tags, on=["resident_id", "window_start"], how="left")
    tag_cols = [c for c in result.columns if c.startswith("tag_")]
    return result.with_columns([pl.col(c).fill_null(0) for c in tag_cols])


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    global SIGNAL_START, SIGNAL_END, FEATURE_OUT, MODEL_OUT

    args = parse_args()
    SIGNAL_START = _parse_date(args.signal_start)
    SIGNAL_END = _parse_date(args.signal_end)
    FEATURE_OUT = Path(args.feature_out)
    MODEL_OUT = Path(args.model_out)

    print("Loading raw tables...")
    print(f"  Signal window: {SIGNAL_START.date()} -> {SIGNAL_END.date()}")
    (
        residents,
        vitals,
        incidents,
        diagnoses,
        transfers,
        admissions,
        needs,
        labs,
        doc_tags,
    ) = load_tables()

    print("Building resident observation bounds...")
    res_bounds = build_resident_obs_bounds(residents)

    print("Generating 14-day observation windows...")
    windows = generate_observation_windows(res_bounds)
    print(f"  {windows.shape[0]:,} windows, {windows['resident_id'].n_unique():,} residents")

    print("Creating wound_14d labels...")
    obs = create_wound_labels(windows, incidents)
    print(f"  wound_14d rate: {obs['wound_14d'].mean():.4%}")

    print("Computing features...")
    join_keys = ["resident_id", "window_start"]

    demo = compute_demographics(obs, residents)
    print("  Demographics done")

    vit = compute_vitals_features(obs, vitals)
    print("  Vitals done")

    dx = compute_diagnoses(obs, diagnoses)
    print("  Diagnoses done")

    hist = compute_incident_history(obs, incidents)
    print("  Incident history done")

    rth_hist = compute_rth_history(obs, transfers)
    print("  RTH history done")

    admission_feats = compute_admission_context(obs, admissions)
    print("  Admission context done")

    need_feats = compute_needs(obs, needs)
    print("  Care needs done")

    lab_feats = compute_labs(obs, labs)
    print("  Lab reports done")

    tag_feats = compute_document_tags(obs, doc_tags)
    print("  Document tags done")

    print("Assembling feature matrix...")
    feature_matrix = (
        obs
        .join(demo, on=join_keys, how="left")
        .join(vit, on=join_keys, how="left")
        .join(dx, on=join_keys, how="left")
        .join(hist, on=join_keys, how="left")
        .join(rth_hist, on=join_keys, how="left")
        .join(admission_feats, on=join_keys, how="left")
        .join(need_feats, on=join_keys, how="left")
        .join(lab_feats, on=join_keys, how="left")
        .join(tag_feats, on=join_keys, how="left")
    )

    meta = ["resident_id", "facility_id", "window_start", "window_end", "feature_cutoff"]
    targets = ["wound_14d"]
    feat_cols = [c for c in feature_matrix.columns if c not in meta + targets]

    print(f"\n  Shape: {feature_matrix.shape[0]:,} x {feature_matrix.shape[1]}")
    print(f"  Features: {len(feat_cols)}")
    print(f"  wound_14d rate: {feature_matrix['wound_14d'].mean():.4%}")

    features = feature_matrix.select(meta + feat_cols)

    FEATURE_OUT.parent.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(FEATURE_OUT)
    feature_matrix.write_parquet(MODEL_OUT)
    print(f"\n  Saved target-free features to {FEATURE_OUT}")
    print(f"  Saved model matrix to {MODEL_OUT}")


if __name__ == "__main__":
    main()
