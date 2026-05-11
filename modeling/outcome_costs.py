"""Actual event and cost utilities for financial backtesting."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import polars as pl

from modeling.business_policy import AVG_CLAIM_COST, PREDICTION_HORIZON_DAYS


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

INCIDENT_TO_EVENT_TYPE = {
    "Fall": "fall",
    "Wound": "wound",
    "Altercation": "altercation",
    "Medication Error": "med_error",
    "Elopement": "elopement",
}


def _as_datetime(value: date | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.combine(value, datetime.min.time())


def build_actual_event_table(
    raw_dir: Path = RAW,
    start: date | datetime | None = None,
    end: date | datetime | None = None,
) -> pl.DataFrame:
    """Return one row per observed incident/transfer with modeled claim cost."""

    start_dt = _as_datetime(start)
    end_dt = _as_datetime(end)

    incidents = (
        pl.read_parquet(raw_dir / "incidents.parquet")
        .filter(pl.col("strikeout") == False)
        .filter(pl.col("incident_type").is_in(list(INCIDENT_TO_EVENT_TYPE)))
        .select(
            pl.col("incident_id").alias("event_id"),
            "resident_id",
            "facility_id",
            pl.col("occurred_at").alias("event_time"),
            pl.col("incident_type").replace(INCIDENT_TO_EVENT_TYPE).alias("event_type"),
        )
    )

    transfers = (
        pl.read_parquet(raw_dir / "hospital_transfers.parquet")
        .filter((pl.col("planned_flag") == False) | pl.col("planned_flag").is_null())
        .select(
            pl.col("transfer_id").alias("event_id"),
            "resident_id",
            "facility_id",
            pl.col("effective_date").alias("event_time"),
            pl.lit("rth").alias("event_type"),
        )
    )

    events = pl.concat([incidents, transfers], how="vertical").with_columns(
        pl.col("event_type").replace(AVG_CLAIM_COST).cast(pl.Float64).alias("claim_cost"),
        pl.col("event_type")
        .replace(PREDICTION_HORIZON_DAYS)
        .cast(pl.Int16)
        .alias("horizon_days"),
    )

    if start_dt is not None:
        events = events.filter(pl.col("event_time") >= start_dt)
    if end_dt is not None:
        events = events.filter(pl.col("event_time") < end_dt)

    return events.sort(["event_time", "resident_id", "event_type"])


def summarize_events(events: pl.DataFrame) -> pl.DataFrame:
    """Summarize actual event counts and costs by incident type."""

    if events.is_empty():
        return pl.DataFrame(
            {
                "event_type": [],
                "actual_events": [],
                "actual_claim_cost": [],
            }
        )

    return (
        events.group_by("event_type")
        .agg(
            pl.len().alias("actual_events"),
            pl.col("claim_cost").sum().alias("actual_claim_cost"),
        )
        .sort("event_type")
    )
