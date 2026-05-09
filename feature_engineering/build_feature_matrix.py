"""
Feature Matrix — 7-day Falls & RTH Models
==========================================

Generates `data/feature_store/feature_matrix_7d.parquet` with:
  - stride = 7 days (non-overlapping)
  - target fall_7d: any Fall incident in [t, t+7d)
  - target rth_7d: any unplanned hospital transfer in [t, t+7d)

Feature groups (all point-in-time, data ≤ feature_cutoff = t − 1d):
  - Demographics: age, LOS
  - Vitals: rolling mean/std/min/max/count per type × 3d/7d/14d lookbacks
  - Diagnoses: active ICD-10 flags, fall-risk & RTH-risk code groups
  - Incident history: prior counts by type × 7d/30d/90d/all-time
  - RTH history: prior transfer counts × 30d/90d/all-time
  - Care needs: open need counts by category
  - Lab reports: abnormal/critical counts × 14d/30d
  - Document tags: binary flags for risk-relevant clinical tags

Usage:
    cd <project_root>
    python feature_engineering/build_feature_matrix.py
"""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

# ─── Configuration ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "feature_store" / "feature_matrix_7d.parquet"

SIGNAL_START = datetime(2023, 7, 1)
SIGNAL_END = datetime(2025, 2, 1)
HORIZON = timedelta(days=7)
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
VITAL_LOOKBACKS = [3, 7, 14]  # days
HISTORY_LOOKBACKS = {
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
    "90d": timedelta(days=90),
}
INCIDENT_TYPES = ["Fall", "Wound", "Altercation"]

FALL_RISK_CODES = {
    "R26": "dx_fall_R26",
    "R27": "dx_fall_R27",
    "M62.81": "dx_fall_M62_81",
    "R29.6": "dx_fall_R29_6",
    "H81": "dx_fall_H81",
    "G40": "dx_fall_G40",
    "F01": "dx_fall_F01",
    "F02": "dx_fall_F02",
    "F03": "dx_fall_F03",
    "G30": "dx_fall_G30",
}
RTH_RISK_CODES = {
    "I50": "dx_rth_I50",
    "J44": "dx_rth_J44",
    "I10": "dx_rth_I10",
    "E11": "dx_rth_E11",
    "N18": "dx_rth_N18",
    "J18": "dx_rth_J18",
    "I63": "dx_rth_I63",
}
RISK_TAGS = [
    "actual_fall", "fall_risk", "actual_wound", "wound_risk",
    "aggressive_behavior", "alzheimers_disease", "dementia",
    "depression", "anxiety", "pain", "infection",
    "pressure_injury", "skin_tear", "bruise",
    "wandering", "elopement",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Load raw tables
# ═══════════════════════════════════════════════════════════════════════════════


def load_tables():
    residents = pl.read_parquet(RAW / "residents.parquet")
    vitals = (
        pl.read_parquet(RAW / "vitals.parquet")
        .filter(pl.col("strikeout") == False)
        .filter(
            pl.col("measured_at") >= SIGNAL_START,
            pl.col("measured_at") < SIGNAL_END,
        )
    )
    incidents = (
        pl.read_parquet(RAW / "incidents.parquet")
        .filter(pl.col("strikeout") == False)
    )
    diagnoses = (
        pl.read_parquet(RAW / "diagnoses.parquet")
        .filter(pl.col("strikeout") == False)
    )
    hospital_transfers = (
        pl.read_parquet(RAW / "hospital_transfers.parquet")
        .filter(
            (pl.col("planned_flag") == False) | pl.col("planned_flag").is_null()
        )
    )
    needs = (
        pl.read_parquet(RAW / "needs.parquet")
        .filter(pl.col("strikeout") == False)
    )
    lab_reports = pl.read_parquet(RAW / "lab_reports.parquet")
    document_tags = (
        pl.read_parquet(RAW / "document_tags.parquet")
        .filter(pl.col("deleted_at").is_null())
    )

    return residents, vitals, incidents, diagnoses, hospital_transfers, needs, lab_reports, document_tags


# ═══════════════════════════════════════════════════════════════════════════════
#  Observation Spine
# ═══════════════════════════════════════════════════════════════════════════════


def build_resident_obs_bounds(residents):
    """Compute per-resident observation start/end clamped to signal window."""
    return (
        residents
        .filter(
            pl.col("admission_date") < SIGNAL_END,
            (pl.col("discharge_date") >= SIGNAL_START) | pl.col("discharge_date").is_null(),
            (pl.col("deceased_date") >= SIGNAL_START) | pl.col("deceased_date").is_null(),
        )
        .select(
            "resident_id",
            "facility_id",
            pl.max_horizontal(
                pl.col("admission_date").cast(pl.Datetime("us")),
                pl.lit(SIGNAL_START),
            ).alias("obs_start"),
            pl.min_horizontal(
                pl.col("discharge_date").fill_null(pl.lit(SIGNAL_END)).cast(pl.Datetime("us")),
                pl.col("deceased_date").fill_null(pl.lit(SIGNAL_END)).cast(pl.Datetime("us")),
                pl.lit(SIGNAL_END),
            ).alias("obs_end"),
        )
        .filter(pl.col("obs_start") < pl.col("obs_end"))
    )


def generate_observation_windows(residents_df):
    """Generate non-overlapping 7-day windows per resident."""
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
#  Target Labels
# ═══════════════════════════════════════════════════════════════════════════════


def create_labels(windows, incidents, hospital_transfers):
    """Create fall_7d and rth_7d binary labels for each observation window."""

    # fall_7d
    falls = incidents.filter(pl.col("incident_type") == "Fall")
    fall_labels = (
        windows
        .join(falls.select("resident_id", "occurred_at"), on="resident_id", how="left")
        .filter(
            pl.col("occurred_at").is_null()
            | (
                (pl.col("occurred_at") >= pl.col("window_start"))
                & (pl.col("occurred_at") < pl.col("window_end"))
            )
        )
        .group_by("resident_id", "window_start")
        .agg(pl.col("occurred_at").is_not_null().any().cast(pl.Int8).alias("fall_7d"))
    )

    # rth_7d
    rth_labels = (
        windows
        .join(hospital_transfers.select("resident_id", "effective_date"), on="resident_id", how="left")
        .filter(
            pl.col("effective_date").is_null()
            | (
                (pl.col("effective_date") >= pl.col("window_start"))
                & (pl.col("effective_date") < pl.col("window_end"))
            )
        )
        .group_by("resident_id", "window_start")
        .agg(pl.col("effective_date").is_not_null().any().cast(pl.Int8).alias("rth_7d"))
    )

    return (
        windows
        .join(fall_labels, on=["resident_id", "window_start"], how="left")
        .join(rth_labels, on=["resident_id", "window_start"], how="left")
        .with_columns(
            pl.col("fall_7d").fill_null(0),
            pl.col("rth_7d").fill_null(0),
        )
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Features
# ═══════════════════════════════════════════════════════════════════════════════


def compute_demographics(obs, residents):
    """Age at window start, length of stay in days."""
    return (
        obs.select("resident_id", "window_start")
        .join(
            residents.select("resident_id", "date_of_birth", "admission_date"),
            on="resident_id",
            how="left",
        )
        .with_columns(
            age_at_window=(
                (pl.col("window_start") - pl.col("date_of_birth").cast(pl.Datetime("us")))
                .dt.total_days() / 365.25
            ).round(1),
            los_days=(
                (pl.col("window_start") - pl.col("admission_date").cast(pl.Datetime("us")))
                .dt.total_days()
            ),
        )
        .select("resident_id", "window_start", "age_at_window", "los_days")
    )


def compute_vitals_features(obs, vitals):
    """
    Rolling vital statistics for each observation window.

    For each lookback window (3d, 7d, 14d) and each vital type, computes
    mean/std/min/max/count. Also adds binary "was measured" flags per
    vital type — absence of monitoring is itself a clinical signal.
    """
    # Apply per-type clipping
    clip_exprs = []
    for vtype, (lo, hi) in VITALS_CLIP.items():
        clip_exprs.append(
            pl.when(pl.col("vital_type") == vtype)
            .then(pl.col("value").clip(lo, hi))
            .otherwise(pl.col("value"))
        )
    vitals_clipped = vitals.with_columns(pl.coalesce(clip_exprs).alias("value"))

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

        # Pivot rolling stats into one column per (vital_type × stat × lookback)
        pivoted = (
            agg.unpivot(
                index=["resident_id", "window_start", "vital_type"],
                on=["mean", "std", "min", "max", "count"],
                variable_name="stat",
                value_name="val",
            )
            .with_columns(
                (
                    pl.col("vital_type").str.to_lowercase().str.replace_all(r"[\s\-]", "_")
                    + "_" + pl.col("stat") + pl.lit(suffix)
                ).alias("col_name")
            )
            .pivot(on="col_name", index=["resident_id", "window_start"], values="val")
        )
        result = result.join(pivoted, on=["resident_id", "window_start"], how="left")

        # Binary "was measured in window" flags — null stat columns mean zero readings
        measured = (
            agg.select("resident_id", "window_start", "vital_type")
            .with_columns(
                (
                    pl.col("vital_type").str.to_lowercase().str.replace_all(r"[\s\-]", "_")
                    + pl.lit(f"_measured{suffix}")
                ).alias("flag_name"),
                pl.lit(1).alias("measured"),
            )
            .pivot(on="flag_name", index=["resident_id", "window_start"], values="measured")
        )
        flag_cols = [c for c in measured.columns if c not in ("resident_id", "window_start")]
        result = result.join(measured, on=["resident_id", "window_start"], how="left")
        result = result.with_columns([pl.col(c).fill_null(0).cast(pl.Int8) for c in flag_cols])

    return result.drop("feature_cutoff")


def compute_diagnoses(obs, diagnoses):
    """
    Point-in-time active diagnosis features.

    Active = onset_at <= feature_cutoff AND (resolved_at IS NULL OR resolved_at > feature_cutoff).
    Produces: total active count, chapter count, per-code fall/RTH risk flags, composite risk scores.
    """
    active = (
        obs.select("resident_id", "window_start", "feature_cutoff")
        .join(
            diagnoses.select("resident_id", "icd_10_code", "onset_at", "resolved_at"),
            on="resident_id",
            how="left",
        )
        .filter(
            pl.col("icd_10_code").is_not_null()
            & (pl.col("onset_at") <= pl.col("feature_cutoff"))
            & (
                pl.col("resolved_at").is_null()
                | (pl.col("resolved_at") > pl.col("feature_cutoff"))
            )
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

    flag_exprs = []
    fall_flag_names = list(FALL_RISK_CODES.values())
    rth_flag_names = list(RTH_RISK_CODES.values())

    for prefix, col_name in FALL_RISK_CODES.items():
        flag_exprs.append(
            pl.col("icd_10_code").str.starts_with(prefix).any().cast(pl.Int8).alias(col_name)
        )
    for prefix, col_name in RTH_RISK_CODES.items():
        flag_exprs.append(
            pl.col("icd_10_code").str.starts_with(prefix).any().cast(pl.Int8).alias(col_name)
        )

    flags = active.group_by("resident_id", "window_start").agg(flag_exprs)

    result = (
        obs.select("resident_id", "window_start")
        .join(counts, on=["resident_id", "window_start"], how="left")
        .join(flags, on=["resident_id", "window_start"], how="left")
        .with_columns(pl.col("dx_active_count").fill_null(0))
        .with_columns(
            pl.sum_horizontal([pl.col(c) for c in fall_flag_names]).fill_null(0).alias("dx_fall_risk_score"),
            pl.sum_horizontal([pl.col(c) for c in rth_flag_names]).fill_null(0).alias("dx_rth_risk_score"),
        )
    )

    return result


def compute_incident_history(obs, incidents_df):
    """Prior incident counts by type and recency (7d/30d/90d/all-time)."""
    joined = (
        obs.select("resident_id", "window_start", "feature_cutoff")
        .join(
            incidents_df.select("resident_id", "incident_type", "occurred_at"),
            on="resident_id",
            how="left",
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
                (
                    (pl.col("incident_type") == itype)
                    & (pl.col("occurred_at") > (pl.col("feature_cutoff") - lb_delta))
                ).sum().cast(pl.Int32).alias(f"hist_{safe}_{lb_name}")
            )
    agg_exprs.append(pl.len().cast(pl.Int32).alias("hist_incident_total"))

    history = joined.group_by("resident_id", "window_start").agg(agg_exprs)

    result = (
        obs.select("resident_id", "window_start")
        .join(history, on=["resident_id", "window_start"], how="left")
    )
    hist_cols = [c for c in result.columns if c.startswith("hist_")]
    return result.with_columns([pl.col(c).fill_null(0) for c in hist_cols])


def compute_rth_history(obs, transfers):
    """Prior unplanned hospital transfer counts (30d/90d/all-time, plus emergency flag count)."""
    joined = (
        obs.select("resident_id", "window_start", "feature_cutoff")
        .join(
            transfers.select("resident_id", "effective_date", "emergency_flag"),
            on="resident_id",
            how="left",
        )
        .filter(
            pl.col("effective_date").is_not_null()
            & (pl.col("effective_date") <= pl.col("feature_cutoff"))
        )
    )

    history = joined.group_by("resident_id", "window_start").agg(
        pl.len().cast(pl.Int32).alias("hist_rth_all"),
        (pl.col("effective_date") > (pl.col("feature_cutoff") - timedelta(days=30)))
        .sum().cast(pl.Int32).alias("hist_rth_30d"),
        (pl.col("effective_date") > (pl.col("feature_cutoff") - timedelta(days=90)))
        .sum().cast(pl.Int32).alias("hist_rth_90d"),
        (pl.col("emergency_flag") == True).sum().cast(pl.Int32).alias("hist_rth_emergency_all"),
    )

    result = (
        obs.select("resident_id", "window_start")
        .join(history, on=["resident_id", "window_start"], how="left")
    )
    rth_cols = [c for c in result.columns if c.startswith("hist_rth")]
    return result.with_columns([pl.col(c).fill_null(0) for c in rth_cols])


def compute_needs(obs, needs):
    """Open care plan need counts by category at feature_cutoff."""
    active_needs = (
        obs.select("resident_id", "window_start", "feature_cutoff")
        .join(
            needs.select("resident_id", "need_category", "initiated_at", "resolved_at"),
            on="resident_id",
            how="left",
        )
        .filter(
            pl.col("need_category").is_not_null()
            & (pl.col("initiated_at") <= pl.col("feature_cutoff"))
            & (
                pl.col("resolved_at").is_null()
                | (pl.col("resolved_at") > pl.col("feature_cutoff"))
            )
        )
    )

    agg = active_needs.group_by("resident_id", "window_start").agg(
        pl.len().cast(pl.Int32).alias("needs_open_total"),
        (pl.col("need_category") == "Fall").sum().cast(pl.Int32).alias("needs_open_fall"),
        (pl.col("need_category") == "Wound").sum().cast(pl.Int32).alias("needs_open_wound"),
        (pl.col("need_category") == "Nutrition").sum().cast(pl.Int32).alias("needs_open_nutrition"),
        (pl.col("need_category") == "Other").sum().cast(pl.Int32).alias("needs_open_other"),
    )

    result = (
        obs.select("resident_id", "window_start")
        .join(agg, on=["resident_id", "window_start"], how="left")
    )
    need_cols = [c for c in result.columns if c.startswith("needs_")]
    return result.with_columns([pl.col(c).fill_null(0) for c in need_cols])


def compute_labs(obs, lab_reports):
    """Abnormal/critical lab counts in 14-day and 30-day lookback windows."""
    joined = (
        obs.select("resident_id", "window_start", "feature_cutoff")
        .join(
            lab_reports.select("resident_id", "severity_status", "reported_at"),
            on="resident_id",
            how="left",
        )
        .filter(
            pl.col("reported_at").is_not_null()
            & (pl.col("reported_at") <= pl.col("feature_cutoff"))
        )
    )

    result = obs.select("resident_id", "window_start")
    for lb_days in [14, 30]:
        lb = timedelta(days=lb_days)
        suffix = f"_{lb_days}d"
        recent = pl.col("reported_at") > (pl.col("feature_cutoff") - lb)

        agg = joined.group_by("resident_id", "window_start").agg(
            recent.sum().cast(pl.Int32).alias(f"labs_total{suffix}"),
            (recent & (pl.col("severity_status") == "Abnormal")).sum().cast(pl.Int32).alias(f"labs_abnormal{suffix}"),
            (recent & (pl.col("severity_status") == "Critical")).sum().cast(pl.Int32).alias(f"labs_critical{suffix}"),
        )
        result = result.join(agg, on=["resident_id", "window_start"], how="left")

    lab_cols = [c for c in result.columns if c.startswith("labs_")]
    return result.with_columns([pl.col(c).fill_null(0) for c in lab_cols])


def compute_document_tags(obs, document_tags):
    """Binary presence flags for risk-relevant clinical document tags (all-time up to feature_cutoff)."""
    joined = (
        obs.select("resident_id", "window_start", "feature_cutoff")
        .join(
            document_tags.filter(pl.col("tag_id").is_in(RISK_TAGS))
            .select("resident_id", "tag_id", "created_at"),
            on="resident_id",
            how="left",
        )
        .filter(
            pl.col("tag_id").is_not_null()
            & (pl.col("created_at") <= pl.col("feature_cutoff"))
        )
    )

    agg_exprs = [
        (pl.col("tag_id") == tag).any().cast(pl.Int8).alias(f"tag_{tag}")
        for tag in RISK_TAGS
    ]
    tags_agg = joined.group_by("resident_id", "window_start").agg(agg_exprs)

    result = (
        obs.select("resident_id", "window_start")
        .join(tags_agg, on=["resident_id", "window_start"], how="left")
    )
    tag_cols = [c for c in result.columns if c.startswith("tag_")]
    return result.with_columns([pl.col(c).fill_null(0) for c in tag_cols])


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    print("Loading raw tables...")
    residents, vitals, incidents, diagnoses, transfers, needs, labs, doc_tags = load_tables()

    print("Building resident observation bounds...")
    res_bounds = build_resident_obs_bounds(residents)
    print(f"  Active residents: {res_bounds.shape[0]:,}")

    print("Generating 7-day observation windows...")
    windows = generate_observation_windows(res_bounds)
    print(f"  {windows.shape[0]:,} windows, {windows['resident_id'].n_unique():,} residents")
    print(f"  Range: {windows['window_start'].min().date()} → {windows['window_end'].max().date()}")

    print("Creating labels...")
    obs = create_labels(windows, incidents, transfers)
    print(f"  fall_7d rate: {obs['fall_7d'].mean():.4%}")
    print(f"  rth_7d rate:  {obs['rth_7d'].mean():.4%}")

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
        .join(need_feats, on=join_keys, how="left")
        .join(lab_feats, on=join_keys, how="left")
        .join(tag_feats, on=join_keys, how="left")
    )

    meta = ["resident_id", "facility_id", "window_start", "window_end", "feature_cutoff"]
    targets = ["fall_7d", "rth_7d"]
    feat_cols = [c for c in feature_matrix.columns if c not in meta + targets]

    print(f"\n  Shape: {feature_matrix.shape[0]:,} × {feature_matrix.shape[1]}")
    print(f"  Features: {len(feat_cols)}")
    print(f"  fall_7d rate: {feature_matrix['fall_7d'].mean():.4%}")
    print(f"  rth_7d rate:  {feature_matrix['rth_7d'].mean():.4%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    feature_matrix.write_parquet(OUT)
    print(f"\n  Saved to {OUT}")


if __name__ == "__main__":
    main()
