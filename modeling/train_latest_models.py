"""Train latest production model artifacts from matured labeled data."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlflow
import mlflow.catboost
import mlflow.sklearn
import polars as pl

from modeling.hyperparameter_tuning import load_hyperparameters
from modeling.train_models import (
    ARTIFACTS_DIR,
    HYPERPARAMETERS_DIR,
    META_COLS,
    TARGET_CONFIG,
    configure_mlflow,
    load_data,
    make_catboost,
    train_temporal_isotonic_model,
)


RAW = ROOT / "data" / "raw"
PRODUCTION_PHASE = "production_model"
DEFAULT_TARGETS = ["fall_7d", "rth_7d", "wound_14d"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latest production models.")
    parser.add_argument(
        "--label-cutoff",
        required=True,
        help="Only train on rows with window_end <= this date, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_TARGETS,
        choices=DEFAULT_TARGETS,
        help="Tier 1 targets to train.",
    )
    parser.add_argument(
        "--skip-altercation",
        action="store_true",
        help="Skip the resident-level altercation model.",
    )
    parser.add_argument(
        "--only-altercation",
        action="store_true",
        help="Train only the resident-level altercation model.",
    )
    return parser.parse_args()


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _matured_rows(df: pl.DataFrame, label_cutoff: datetime) -> pl.DataFrame:
    return df.filter(pl.col("window_end") <= label_cutoff)


def train_tier1_target(target: str, label_cutoff: datetime) -> None:
    df, feature_cols = load_data(target)
    train_df = _matured_rows(df, label_cutoff)
    if train_df.is_empty():
        raise ValueError(f"No matured rows available for {target} by {label_cutoff.date()}")

    y = train_df[target].to_numpy().ravel()

    params, params_path = load_hyperparameters(HYPERPARAMETERS_DIR, target)
    print(
        f"  Training {target}: {len(y):,} rows, positive rate={y.mean():.4%}, "
        f"params={params_path.name if params_path else 'defaults'}"
    )
    calibrated = train_temporal_isotonic_model(
        train_df,
        target,
        feature_cols,
        TARGET_CONFIG[target]["horizon_days"],
        model_params=params,
        end_date=label_cutoff,
    )

    with mlflow.start_run(run_name=f"{target}_catboost_latest_calibrated"):
        mlflow.set_tag("target", target)
        mlflow.set_tag("phase", PRODUCTION_PHASE)
        mlflow.set_tag("calibrated", "true")
        mlflow.log_params(
            {
                "target": target,
                "algorithm": "catboost",
                "calibrated": True,
                "calibration_method": "isotonic",
                "calibration_strategy": "temporal_oof_expanding_window",
                "calibration_folds": calibrated["calibration_folds"].height,
                "calibration_size": len(calibrated["y_calibration"]),
                "calibration_positive_rate": round(
                    float(calibrated["y_calibration"].mean()), 6
                ),
                "n_features": len(feature_cols),
                "train_size": len(y),
                "train_positive_rate": round(float(y.mean()), 6),
                "label_cutoff": label_cutoff.date().isoformat(),
                "data_path": str(TARGET_CONFIG[target]["data_path"]),
                "tuned_hyperparameters": bool(params),
            }
        )
        if params:
            mlflow.log_params({f"catboost_{k}": v for k, v in params.items()})
        mlflow.sklearn.log_model(calibrated["model"], "model")

        calibration_fold_path = (
            ARTIFACTS_DIR / f"{target}_production_temporal_calibration_fold_metrics.csv"
        )
        calibrated["calibration_folds"].write_csv(calibration_fold_path)
        mlflow.log_artifact(str(calibration_fold_path))


def _load_raw_for_altercation(label_cutoff: datetime):
    residents = pl.read_parquet(RAW / "residents.parquet")
    incidents = (
        pl.read_parquet(RAW / "incidents.parquet")
        .filter(pl.col("strikeout") == False)
        .filter(pl.col("occurred_at") < label_cutoff)
    )
    diagnoses = (
        pl.read_parquet(RAW / "diagnoses.parquet")
        .filter(pl.col("strikeout") == False)
        .filter(pl.col("onset_at").is_null() | (pl.col("onset_at") < label_cutoff))
    )
    needs = (
        pl.read_parquet(RAW / "needs.parquet")
        .filter(pl.col("strikeout") == False)
        .filter(pl.col("initiated_at").is_null() | (pl.col("initiated_at") < label_cutoff))
    )
    doc_tags = (
        pl.read_parquet(RAW / "document_tags.parquet")
        .filter(pl.col("deleted_at").is_null())
        .filter(pl.col("created_at") < label_cutoff)
    )
    return residents, incidents, diagnoses, needs, doc_tags


def train_altercation(label_cutoff: datetime) -> None:
    from modeling.train_altercation import build_altercation_dataset, make_catboost

    residents, incidents, diagnoses, needs, doc_tags = _load_raw_for_altercation(label_cutoff)
    df, feature_cols, target_col = build_altercation_dataset(
        residents, incidents, diagnoses, needs, doc_tags, signal_end=label_cutoff
    )
    X = df.select(feature_cols)
    y = df[target_col].to_numpy().ravel()

    params, params_path = load_hyperparameters(HYPERPARAMETERS_DIR, "altercation")
    model = make_catboost(params)
    print(
        f"  Training altercation: {len(y):,} residents, positive rate={y.mean():.4%}, "
        f"params={params_path.name if params_path else 'defaults'}"
    )
    model.fit(X, y)

    with mlflow.start_run(run_name="altercation_catboost_latest"):
        mlflow.set_tag("target", "has_altercation")
        mlflow.set_tag("phase", PRODUCTION_PHASE)
        mlflow.set_tag("calibrated", "false")
        mlflow.log_params(
            {
                "target": "has_altercation",
                "algorithm": "catboost",
                "calibrated": False,
                "n_features": len(feature_cols),
                "n_residents": len(y),
                "positive_rate": round(float(y.mean()), 6),
                "label_cutoff": label_cutoff.date().isoformat(),
                "tuned_hyperparameters": bool(params),
            }
        )
        if params:
            mlflow.log_params({f"catboost_{k}": v for k, v in params.items()})
        mlflow.catboost.log_model(model, "model")


def main() -> None:
    args = parse_args()
    label_cutoff = _parse_date(args.label_cutoff)
    configure_mlflow()

    print(f"Training latest models with label cutoff {label_cutoff.date()}...")
    if not args.only_altercation:
        for target in args.targets:
            train_tier1_target(target, label_cutoff)

    if not args.skip_altercation:
        train_altercation(label_cutoff)

    print("Latest model training complete.")


if __name__ == "__main__":
    main()
