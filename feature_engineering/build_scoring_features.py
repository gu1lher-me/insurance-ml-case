"""Build target-free production scoring features for one as-of date.

The output uses the same 211 feature columns as the 7-day training feature
store. It creates one row per active resident:

    feature_cutoff = as_of_date - 1 day
    window_start   = as_of_date
    window_end     = as_of_date + 7 days

No outcome labels are created.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl

from feature_engineering import build_falls_rth_matrix as features_7d


RAW = ROOT / "data" / "raw"
DEFAULT_REFERENCE = ROOT / "data" / "feature_store" / "features_7d.parquet"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "feature_store" / "scoring"
META_COLS = ["resident_id", "facility_id", "window_start", "window_end", "feature_cutoff"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build target-free scoring features.")
    parser.add_argument(
        "--as-of-date",
        required=True,
        help="Scoring date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--reference-features",
        default=str(DEFAULT_REFERENCE),
        help="Historical target-free feature store used as schema reference.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output parquet path. Defaults to data/feature_store/scoring/features_YYYY-MM-DD.parquet.",
    )
    return parser.parse_args()


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def load_raw_tables():
    residents = pl.read_parquet(RAW / "residents.parquet")
    vitals = pl.read_parquet(RAW / "vitals.parquet").filter(pl.col("strikeout") == False)
    incidents = pl.read_parquet(RAW / "incidents.parquet").filter(pl.col("strikeout") == False)
    diagnoses = pl.read_parquet(RAW / "diagnoses.parquet").filter(pl.col("strikeout") == False)
    transfers = pl.read_parquet(RAW / "hospital_transfers.parquet").filter(
        (pl.col("planned_flag") == False) | pl.col("planned_flag").is_null()
    )
    needs = pl.read_parquet(RAW / "needs.parquet").filter(pl.col("strikeout") == False)
    labs = pl.read_parquet(RAW / "lab_reports.parquet")
    doc_tags = pl.read_parquet(RAW / "document_tags.parquet").filter(
        pl.col("deleted_at").is_null()
    )
    return residents, vitals, incidents, diagnoses, transfers, needs, labs, doc_tags


def build_scoring_spine(residents: pl.DataFrame, as_of_date: datetime) -> pl.DataFrame:
    active = (
        residents.filter(pl.col("admission_date") <= as_of_date)
        .filter(
            (pl.col("discharge_date").is_null()) | (pl.col("discharge_date") > as_of_date)
        )
        .filter(
            (pl.col("deceased_date").is_null()) | (pl.col("deceased_date") > as_of_date)
        )
        .select("resident_id", "facility_id")
        .unique()
    )

    return active.with_columns(
        pl.lit(as_of_date).cast(pl.Datetime("us")).alias("window_start"),
        pl.lit(as_of_date + timedelta(days=7)).cast(pl.Datetime("us")).alias("window_end"),
        pl.lit(as_of_date - timedelta(days=1)).cast(pl.Datetime("us")).alias("feature_cutoff"),
    ).select(META_COLS)


def assemble_features(
    obs: pl.DataFrame,
    residents: pl.DataFrame,
    vitals: pl.DataFrame,
    incidents: pl.DataFrame,
    diagnoses: pl.DataFrame,
    transfers: pl.DataFrame,
    needs: pl.DataFrame,
    labs: pl.DataFrame,
    doc_tags: pl.DataFrame,
) -> pl.DataFrame:
    join_keys = ["resident_id", "window_start"]

    feature_matrix = (
        obs.join(features_7d.compute_demographics(obs, residents), on=join_keys, how="left")
        .join(features_7d.compute_vitals_features(obs, vitals), on=join_keys, how="left")
        .join(features_7d.compute_diagnoses(obs, diagnoses), on=join_keys, how="left")
        .join(features_7d.compute_incident_history(obs, incidents), on=join_keys, how="left")
        .join(features_7d.compute_rth_history(obs, transfers), on=join_keys, how="left")
        .join(features_7d.compute_needs(obs, needs), on=join_keys, how="left")
        .join(features_7d.compute_labs(obs, labs), on=join_keys, how="left")
        .join(features_7d.compute_document_tags(obs, doc_tags), on=join_keys, how="left")
    )
    return feature_matrix


def align_to_reference(feature_matrix: pl.DataFrame, reference_path: Path) -> pl.DataFrame:
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference feature schema not found: {reference_path}. "
            "Build historical 7-day features first."
        )

    reference = pl.scan_parquet(reference_path).collect_schema()
    output = feature_matrix

    for col, dtype in reference.items():
        if col in output.columns:
            output = output.with_columns(pl.col(col).cast(dtype))
            continue

        if col in META_COLS:
            raise ValueError(f"Scoring features are missing required meta column {col}")

        if col.endswith("_measured_3d") or col.endswith("_measured_7d") or col.endswith("_measured_14d"):
            default_expr = pl.lit(0).cast(dtype)
        elif col.startswith(("hist_", "needs_", "labs_", "tag_", "dx_")):
            default_expr = pl.lit(0).cast(dtype)
        else:
            default_expr = pl.lit(None).cast(dtype)
        output = output.with_columns(default_expr.alias(col))

    return output.select(reference.names())


def build_scoring_features(as_of_date: datetime, reference_path: Path = DEFAULT_REFERENCE) -> pl.DataFrame:
    residents, vitals, incidents, diagnoses, transfers, needs, labs, doc_tags = load_raw_tables()
    spine = build_scoring_spine(residents, as_of_date)
    if spine.is_empty():
        raise ValueError(f"No active residents found as of {as_of_date.date()}")

    features = assemble_features(
        spine,
        residents,
        vitals,
        incidents,
        diagnoses,
        transfers,
        needs,
        labs,
        doc_tags,
    )
    return align_to_reference(features, reference_path)


def main() -> None:
    args = parse_args()
    as_of_date = _parse_date(args.as_of_date)
    output = (
        Path(args.output)
        if args.output
        else DEFAULT_OUTPUT_DIR / f"features_{args.as_of_date}.parquet"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building scoring features as of {as_of_date.date()}...")
    features = build_scoring_features(as_of_date, Path(args.reference_features))
    features.write_parquet(output)
    print(
        f"  Saved {features.height:,} residents x {features.width} columns to {output}"
    )


if __name__ == "__main__":
    main()
