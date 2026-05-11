"""Run an end-to-end daily scoring workflow for a new raw-data batch."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parent.parent
FEATURE_SCORING_DIR = ROOT / "data" / "feature_store" / "scoring"
SCORED_DIR = ROOT / "data" / "scored"
ACTION_QUEUE_DIR = ROOT / "data" / "action_queues"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily resident risk scoring.")
    parser.add_argument("--as-of-date", required=True, help="Scoring date, YYYY-MM-DD.")
    parser.add_argument(
        "--label-cutoff",
        default=None,
        help="Latest fully labeled date for training. Defaults to --as-of-date.",
    )
    parser.add_argument(
        "--signal-start",
        default="2023-07-01",
        help="Historical feature start date when rebuilding training matrices.",
    )
    parser.add_argument(
        "--rebuild-training-features",
        action="store_true",
        help="Rebuild historical 7d and 14d training matrices before training.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Train latest production MLflow models before scoring.",
    )
    parser.add_argument(
        "--model-phase",
        default=None,
        help="MLflow model phase to score. Defaults to production_model if --retrain else final_model.",
    )
    parser.add_argument(
        "--action-policy",
        choices=["economic", "capacity"],
        default="economic",
        help="How to select rows for the action queue.",
    )
    parser.add_argument(
        "--capacity-rate",
        type=float,
        default=0.10,
        help="Per-facility action rate when --action-policy capacity.",
    )
    return parser.parse_args()


def run_step(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def rebuild_training_features(as_of_date: str, signal_start: str) -> None:
    py = sys.executable
    run_step(
        [
            py,
            "feature_engineering/build_falls_rth_matrix.py",
            "--signal-start",
            signal_start,
            "--signal-end",
            as_of_date,
        ]
    )
    run_step(
        [
            py,
            "feature_engineering/build_wound_matrix.py",
            "--signal-start",
            signal_start,
            "--signal-end",
            as_of_date,
        ]
    )


def train_latest(label_cutoff: str) -> None:
    run_step(
        [
            sys.executable,
            "modeling/train_latest_models.py",
            "--label-cutoff",
            label_cutoff,
        ]
    )


def build_scoring_features(as_of_date: str, output: Path) -> None:
    run_step(
        [
            sys.executable,
            "feature_engineering/build_scoring_features.py",
            "--as-of-date",
            as_of_date,
            "--output",
            str(output),
        ]
    )


def score_composite(features_path: Path, scores_path: Path, model_phase: str) -> None:
    run_step(
        [
            sys.executable,
            "modeling/score_composite.py",
            "--features-path",
            str(features_path),
            "--output",
            str(scores_path),
            "--model-phase",
            model_phase,
        ]
    )


def select_capacity(df: pl.DataFrame, capacity_rate: float) -> pl.DataFrame:
    selected = []
    for group in df.partition_by("facility_id", maintain_order=True):
        n_select = max(1, int(math.ceil(group.height * capacity_rate)))
        selected.append(group.sort("composite_expected_cost", descending=True).head(n_select))
    return pl.concat(selected, how="vertical") if selected else df.head(0)


def export_action_queue(
    scores_path: Path,
    output_path: Path,
    action_policy: str,
    capacity_rate: float,
) -> None:
    df = pl.read_parquet(scores_path)
    if action_policy == "economic":
        queue = df.filter(pl.col("economic_action") == True)
    else:
        queue = select_capacity(df, capacity_rate)

    queue = queue.sort("composite_expected_cost", descending=True)
    cols = [
        "resident_id",
        "facility_id",
        "window_start",
        "composite_expected_cost",
        "expected_avoidable_cost",
        "economic_action",
        "top_reason",
        "reason_codes",
        "recommended_actions",
        "fall_probability",
        "rth_probability",
        "wound_probability",
        "altercation_probability",
        "med_error_flag",
        "elopement_flag",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    queue.select([c for c in cols if c in queue.columns]).write_csv(output_path)
    print(f"\nSaved action queue with {queue.height:,} rows to {output_path}")


def main() -> None:
    args = parse_args()
    _parse_date(args.as_of_date)
    label_cutoff = args.label_cutoff or args.as_of_date
    _parse_date(label_cutoff)

    model_phase = args.model_phase or ("production_model" if args.retrain else "final_model")
    FEATURE_SCORING_DIR.mkdir(parents=True, exist_ok=True)
    SCORED_DIR.mkdir(parents=True, exist_ok=True)
    ACTION_QUEUE_DIR.mkdir(parents=True, exist_ok=True)

    features_path = FEATURE_SCORING_DIR / f"features_{args.as_of_date}.parquet"
    scores_path = SCORED_DIR / f"composite_scores_{args.as_of_date}.parquet"
    action_queue_path = ACTION_QUEUE_DIR / f"action_queue_{args.as_of_date}.csv"

    print("Daily scoring workflow")
    print(f"  as_of_date: {args.as_of_date}")
    print(f"  label_cutoff: {label_cutoff}")
    print(f"  model_phase: {model_phase}")

    if args.rebuild_training_features:
        rebuild_training_features(label_cutoff, args.signal_start)

    if args.retrain:
        train_latest(label_cutoff)

    build_scoring_features(args.as_of_date, features_path)
    score_composite(features_path, scores_path, model_phase)
    export_action_queue(
        scores_path,
        action_queue_path,
        args.action_policy,
        args.capacity_rate,
    )

    print("\nDaily scoring workflow complete.")
    print(f"  Features: {features_path}")
    print(f"  Scores: {scores_path}")
    print(f"  Action queue: {action_queue_path}")


if __name__ == "__main__":
    main()
