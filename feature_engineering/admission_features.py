"""Point-in-time hospital admission context features."""

from __future__ import annotations

from datetime import timedelta

import polars as pl


ADMISSION_FEATURE_DTYPES = {
    "admission_active_count": pl.Int32,
    "admission_active_any": pl.Int8,
    "admission_active_post_acute": pl.Int8,
    "admission_active_chronic_long_term": pl.Int8,
    "admission_count_30d": pl.Int32,
    "admission_count_90d": pl.Int32,
    "admission_post_acute_30d": pl.Int32,
    "admission_post_acute_90d": pl.Int32,
    "admission_chronic_long_term_30d": pl.Int32,
    "admission_chronic_long_term_90d": pl.Int32,
    "admission_hospital_stay_to_30d": pl.Int32,
    "admission_hospital_stay_to_90d": pl.Int32,
}


def compute_admission_context(
    obs: pl.DataFrame,
    hospital_admissions: pl.DataFrame,
) -> pl.DataFrame:
    """Build leakage-safe SNF admission/status context features.

    A row is eligible only when the admission effective date and source row
    creation timestamp are both known by `feature_cutoff`. `hospital_stay_to`
    is counted only when that date is also on or before `feature_cutoff`.
    """
    base = obs.select("resident_id", "window_start", "feature_cutoff")
    admissions = hospital_admissions.select(
        "resident_id",
        "effective_date",
        "ineffective_date",
        "admission_status",
        "hospital_stay_to",
        "created_at",
    ).with_columns(
        (pl.col("admission_status") == "Post Acute").alias("is_post_acute"),
        (pl.col("admission_status") == "Chronic Long-Term").alias(
            "is_chronic_long_term"
        ),
    )

    known = (
        base.join(admissions, on="resident_id", how="left")
        .filter(
            pl.col("effective_date").is_not_null()
            & pl.col("created_at").is_not_null()
            & (pl.col("effective_date") <= pl.col("feature_cutoff"))
            & (pl.col("created_at") <= pl.col("feature_cutoff"))
        )
    )

    active = (
        known.filter(
            pl.col("ineffective_date").is_null()
            | (pl.col("ineffective_date") > pl.col("feature_cutoff"))
        )
        .group_by("resident_id", "window_start")
        .agg(
            pl.len().cast(pl.Int32).alias("admission_active_count"),
            pl.lit(1).cast(pl.Int8).alias("admission_active_any"),
            pl.col("is_post_acute")
            .any()
            .cast(pl.Int8)
            .alias("admission_active_post_acute"),
            pl.col("is_chronic_long_term")
            .any()
            .cast(pl.Int8)
            .alias("admission_active_chronic_long_term"),
        )
    )

    history = known.group_by("resident_id", "window_start").agg(
        (
            pl.col("effective_date")
            > (pl.col("feature_cutoff") - timedelta(days=30))
        )
        .sum()
        .cast(pl.Int32)
        .alias("admission_count_30d"),
        (
            pl.col("effective_date")
            > (pl.col("feature_cutoff") - timedelta(days=90))
        )
        .sum()
        .cast(pl.Int32)
        .alias("admission_count_90d"),
        (
            pl.col("is_post_acute")
            & (
                pl.col("effective_date")
                > (pl.col("feature_cutoff") - timedelta(days=30))
            )
        )
        .sum()
        .cast(pl.Int32)
        .alias("admission_post_acute_30d"),
        (
            pl.col("is_post_acute")
            & (
                pl.col("effective_date")
                > (pl.col("feature_cutoff") - timedelta(days=90))
            )
        )
        .sum()
        .cast(pl.Int32)
        .alias("admission_post_acute_90d"),
        (
            pl.col("is_chronic_long_term")
            & (
                pl.col("effective_date")
                > (pl.col("feature_cutoff") - timedelta(days=30))
            )
        )
        .sum()
        .cast(pl.Int32)
        .alias("admission_chronic_long_term_30d"),
        (
            pl.col("is_chronic_long_term")
            & (
                pl.col("effective_date")
                > (pl.col("feature_cutoff") - timedelta(days=90))
            )
        )
        .sum()
        .cast(pl.Int32)
        .alias("admission_chronic_long_term_90d"),
        (
            pl.col("hospital_stay_to").is_not_null()
            & (pl.col("hospital_stay_to") <= pl.col("feature_cutoff"))
            & (
                pl.col("hospital_stay_to")
                > (pl.col("feature_cutoff") - timedelta(days=30))
            )
        )
        .sum()
        .cast(pl.Int32)
        .alias("admission_hospital_stay_to_30d"),
        (
            pl.col("hospital_stay_to").is_not_null()
            & (pl.col("hospital_stay_to") <= pl.col("feature_cutoff"))
            & (
                pl.col("hospital_stay_to")
                > (pl.col("feature_cutoff") - timedelta(days=90))
            )
        )
        .sum()
        .cast(pl.Int32)
        .alias("admission_hospital_stay_to_90d"),
    )

    result = (
        base.select("resident_id", "window_start")
        .join(active, on=["resident_id", "window_start"], how="left")
        .join(history, on=["resident_id", "window_start"], how="left")
    )

    return result.with_columns(
        [
            pl.col(col).fill_null(0).cast(dtype).alias(col)
            for col, dtype in ADMISSION_FEATURE_DTYPES.items()
        ]
    )
