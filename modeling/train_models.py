"""
Incident Prediction Modeling — Falls, RTH & Wounds (CatBoost)
=============================================================

Expanding-window temporal cross-validation with CatBoost.
Each CV fold tests on a single observation window (matching the
prediction horizon) while training on all prior data.

Final model is trained on all pre-holdout data, evaluated on holdout
(Jan 2025), and compared with an isotonic-calibrated variant.

Targets:
  - fall_7d, rth_7d  → model_matrix_7d.parquet  (7-day horizon)
  - wound_14d        → model_matrix_14d.parquet (14-day horizon)

Usage:
    cd <project_root>
    python modeling/train_models.py
"""

import warnings
from datetime import date, timedelta
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.catboost
import mlflow.sklearn
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (auc,  # f1_score,; precision_score,; recall_score,
                             brier_score_loss, log_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ─── Configuration ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA_7D = ROOT / "data" / "processed" / "model_matrix_7d.parquet"
DATA_14D = ROOT / "data" / "processed" / "model_matrix_14d.parquet"
ARTIFACTS_DIR = ROOT / "modeling" / "artifacts"
MLFLOW_TRACKING_DIR = ROOT / "mlruns"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

META_COLS = [
    "resident_id",
    "facility_id",
    "window_start",
    "window_end",
    "feature_cutoff",
]

# Per-target configuration
TARGET_CONFIG = {
    "fall_7d": {
        "data_path": DATA_7D,
        "all_targets": ["fall_7d", "rth_7d"],
        "horizon_days": 7,
        "display": "Falls (7d)",
    },
    "rth_7d": {
        "data_path": DATA_7D,
        "all_targets": ["fall_7d", "rth_7d"],
        "horizon_days": 7,
        "display": "RTH (7d)",
    },
    "wound_14d": {
        "data_path": DATA_14D,
        "all_targets": ["wound_14d"],
        "horizon_days": 14,
        "display": "Wounds (14d)",
    },
}
TARGETS = list(TARGET_CONFIG.keys())
EXPERIMENT_NAME = "incident_prediction"

# Temporal CV
MIN_TRAIN_WEEKS = 26  # ~6 months minimum training data
FOLD_STEP = 4  # every Nth eligible window as a fold boundary (~monthly)
HOLDOUT_START = date(2025, 1, 1)


def configure_mlflow():
    """Configure a local MLflow file store in a Windows-safe format."""
    MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_DIR.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════════


def load_data(target):
    cfg = TARGET_CONFIG[target]
    df = pl.read_parquet(cfg["data_path"])
    all_targets = cfg["all_targets"]
    feature_cols = sorted(c for c in df.columns if c not in META_COLS + all_targets)
    return df, feature_cols


def get_fold_boundaries(df, horizon_days):
    """Generate a regular grid of fold boundaries.

    Each boundary defines a test window [boundary, boundary + horizon).
    Training uses all rows with window_end <= boundary (purging).
    Boundaries are spaced FOLD_STEP weeks apart.
    """
    ws_col = df.filter(pl.col("window_start") < HOLDOUT_START)["window_start"]
    data_start = ws_col.min()

    # Normalise to a plain date so arithmetic is clean
    if hasattr(data_start, "date"):
        data_start = data_start.date()

    first_boundary = data_start + timedelta(weeks=MIN_TRAIN_WEEKS)

    boundaries = []
    current = first_boundary
    while current < HOLDOUT_START:
        boundaries.append(current)
        current += timedelta(weeks=FOLD_STEP)

    return boundaries


# ═══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════════════


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)

    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "pr_auc": auc(rec_curve, prec_curve),
        # "f1": f1_score(y_true, y_pred, zero_division=0),
        # "precision": precision_score(y_true, y_pred, zero_division=0),
        # "recall": recall_score(y_true, y_pred, zero_division=0),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════════


def plot_roc(y_true, y_prob, title):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, lw=2, label=f"ROC AUC = {roc_auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pr(y_true, y_prob, title):
    prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_val = auc(rec_vals, prec_vals)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec_vals, prec_vals, lw=2, label=f"PR AUC = {pr_auc_val:.4f}")
    ax.axhline(baseline, color="grey", ls="--", lw=1, label=f"Baseline = {baseline:.4f}")
    ax.set(xlabel="Recall", ylabel="Precision", title=title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_feature_importance(importances, feature_names, title, top_n=15):
    idx = np.argsort(np.abs(importances))[::-1][:top_n]
    top_names = [feature_names[i] for i in idx][::-1]
    top_vals = np.abs(importances[idx])[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), top_vals, color="steelblue")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names)
    ax.set(xlabel="Importance (absolute)", title=title)
    fig.tight_layout()
    return fig


def plot_calibration_comparison(results, y_true, title):
    """Calibration curves for uncalibrated vs calibrated on the same axes."""
    fig, ax = plt.subplots(figsize=(10, 8))

    styles = {"uncalibrated": ("--", "#1f77b4"), "calibrated": ("-", "#d62728")}

    for label, y_prob in results.items():
        key = "calibrated" if "(calibrated)" in label else "uncalibrated"
        ls, color = styles[key]

        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=10, strategy="uniform"
        )
        ax.plot(prob_pred, prob_true, ls, marker="o", color=color, label=label)

    ax.plot([0, 1], [0, 1], "k:", lw=1, label="Perfectly calibrated")
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Fraction of positives",
        title=title,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_cv_metric_over_time(fold_df, metric, title):
    """Line chart of a CV metric across fold boundaries."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(fold_df)), fold_df[metric], "o-", lw=1.5, markersize=5)
    mean_val = fold_df[metric].mean()
    ax.axhline(mean_val, color="red", ls="--", lw=1, label=f"Mean = {mean_val:.4f}")
    ax.set(
        xlabel="Fold (chronological)",
        ylabel=metric.replace("_", " ").title(),
        title=title,
    )
    ax.set_xticks(range(len(fold_df)))
    ax.set_xticklabels(
        [str(b)[:10] for b in fold_df["boundary"]], rotation=45, ha="right"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════════════


def make_catboost():
    return CatBoostClassifier(
        verbose=0,
        random_seed=42,
        allow_writing_files=False,
    )


def _format_target(target):
    return TARGET_CONFIG[target]["display"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 1 — Expanding-Window Temporal CV
# ═══════════════════════════════════════════════════════════════════════════════


def run_expanding_cv(df, target, feature_cols, horizon_days):
    """
    Expanding-window temporal CV.

    Each fold: train on all data before `boundary`, test on the single
    observation window at `boundary`.  Returns per-fold metrics
    and pooled (concatenated) predictions.
    """
    boundaries = get_fold_boundaries(df, horizon_days)

    fold_records = []
    all_y_true = []
    all_y_prob = []

    for fold_idx, boundary in enumerate(boundaries):
        test_end = boundary + timedelta(days=horizon_days)

        # Purging: keep only training rows whose entire label window falls
        # strictly before the boundary. Rows with window_end > boundary
        # have labels that include events occurring after the scoring date,
        # which would not be available in production at the time of training.
        train_df = df.filter(pl.col("window_end") <= boundary)
        test_df = df.filter(
            (pl.col("window_start") >= boundary)
            & (pl.col("window_start") < test_end)
        )

        if test_df.shape[0] == 0:
            continue

        X_train = train_df.select(feature_cols).to_pandas()
        y_train = train_df[target].to_numpy().ravel()
        X_test = test_df.select(feature_cols).to_pandas()
        y_test = test_df[target].to_numpy().ravel()

        n_pos = int(y_test.sum())
        if n_pos < 2:
            print(
                f"    Fold {fold_idx + 1}/{len(boundaries)}: "
                f"skipped (only {n_pos} positives in test window)"
            )
            continue

        model = make_catboost()
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_prob)
        metrics["fold"] = fold_idx
        metrics["boundary"] = str(boundary)
        metrics["train_size"] = len(y_train)
        metrics["test_size"] = len(y_test)
        metrics["test_positives"] = n_pos
        fold_records.append(metrics)

        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(y_prob.tolist())

        print(
            f"    Fold {fold_idx + 1}/{len(boundaries)}: "
            f"train={len(y_train):,}  test={len(y_test):,}  "
            f"pos={n_pos}  ROC-AUC={metrics['roc_auc']:.4f}"
        )

    fold_df = pd.DataFrame(fold_records)
    pooled_metrics = compute_metrics(np.array(all_y_true), np.array(all_y_prob))

    return fold_df, pooled_metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 2 — Final Model + Holdout Evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def train_final_model(df, target, feature_cols):
    """Train on all pre-holdout data.  Return uncalibrated + calibrated models
    and their holdout predictions."""
    # Purging: same logic as CV — only keep training rows whose full label
    # window falls before HOLDOUT_START.
    train_df = df.filter(pl.col("window_end") <= HOLDOUT_START)
    holdout_df = df.filter(pl.col("window_start") >= HOLDOUT_START)

    X_train = train_df.select(feature_cols).to_pandas()
    y_train = train_df[target].to_numpy().ravel()
    X_holdout = holdout_df.select(feature_cols).to_pandas()
    y_holdout = holdout_df[target].to_numpy().ravel()

    # Uncalibrated
    print("  Training final CatBoost (uncalibrated)...")
    model = make_catboost()
    model.fit(X_train, y_train)
    y_prob_uncal = model.predict_proba(X_holdout)[:, 1]
    metrics_uncal = compute_metrics(y_holdout, y_prob_uncal)

    # Calibrated (isotonic, CV=5)
    print("  Training final CatBoost (calibrated, isotonic CV=5)...")
    cal_model = CalibratedClassifierCV(
        estimator=make_catboost(), cv=5, method="sigmoid"
    )
    cal_model.fit(X_train, y_train)
    y_prob_cal = cal_model.predict_proba(X_holdout)[:, 1]
    metrics_cal = compute_metrics(y_holdout, y_prob_cal)

    return {
        "model": model,
        "cal_model": cal_model,
        "y_holdout": y_holdout,
        "y_prob_uncal": y_prob_uncal,
        "y_prob_cal": y_prob_cal,
        "metrics_uncal": metrics_uncal,
        "metrics_cal": metrics_cal,
        "importances": model.feature_importances_,
        "train_size": len(y_train),
        "holdout_size": len(y_holdout),
        "train_pos_rate": float(y_train.mean()),
        "holdout_pos_rate": float(y_holdout.mean()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MLflow Logging
# ═══════════════════════════════════════════════════════════════════════════════

METRIC_NAMES = [
    "roc_auc",
    "log_loss",
    "brier_score",
    "pr_auc",
    # "f1",
    # "precision",
    # "recall",
]


def log_cv_run(target, fold_df, pooled_metrics, target_display, horizon_days):
    with mlflow.start_run(run_name=f"{target}_catboost_temporal_cv"):
        mlflow.set_tag("target", target)
        mlflow.set_tag("phase", "temporal_cv")

        mlflow.log_params(
            {
                "target": target,
                "algorithm": "catboost",
                "cv_type": "expanding_window",
                "test_window_days": horizon_days,
                "n_folds": len(fold_df),
                "fold_step": FOLD_STEP,
                "min_train_weeks": MIN_TRAIN_WEEKS,
            }
        )

        # Pooled metrics (computed on concatenated fold predictions)
        mlflow.log_metrics(
            {f"cv_pooled_{k}": v for k, v in pooled_metrics.items()}
        )

        # Per-fold mean ± std
        for m in METRIC_NAMES:
            mlflow.log_metric(f"cv_{m}_mean", fold_df[m].mean())
            mlflow.log_metric(f"cv_{m}_std", fold_df[m].std())

        # Fold detail CSV
        fold_csv_path = ARTIFACTS_DIR / f"{target}_cv_fold_metrics.csv"
        fold_df.to_csv(fold_csv_path, index=False)
        mlflow.log_artifact(str(fold_csv_path))

        # ROC-AUC over time
        fig = plot_cv_metric_over_time(
            fold_df, "roc_auc", f"ROC-AUC over CV Folds — {target_display}"
        )
        mlflow.log_figure(fig, "cv_roc_auc_over_time.png")
        plt.close(fig)


def log_final_uncalibrated(target, res, feature_cols, target_display):
    with mlflow.start_run(run_name=f"{target}_catboost_final"):
        mlflow.set_tag("target", target)
        mlflow.set_tag("phase", "final_model")
        mlflow.set_tag("calibrated", "false")

        mlflow.log_params(
            {
                "target": target,
                "algorithm": "catboost",
                "calibrated": False,
                "n_features": len(feature_cols),
                "train_size": res["train_size"],
                "holdout_size": res["holdout_size"],
                "train_positive_rate": round(res["train_pos_rate"], 5),
                "holdout_positive_rate": round(res["holdout_pos_rate"], 5),
            }
        )
        mlflow.log_metrics(res["metrics_uncal"])

        # ROC
        fig = plot_roc(
            res["y_holdout"],
            res["y_prob_uncal"],
            f"ROC — {target_display} — CatBoost (Holdout)",
        )
        mlflow.log_figure(fig, "roc_curve.png")
        plt.close(fig)

        # PR
        fig = plot_pr(
            res["y_holdout"],
            res["y_prob_uncal"],
            f"Precision-Recall — {target_display} — CatBoost (Holdout)",
        )
        mlflow.log_figure(fig, "pr_curve.png")
        plt.close(fig)

        # Feature importance (top 15 plot)
        fig = plot_feature_importance(
            res["importances"],
            feature_cols,
            f"Top 15 Features — {target_display} — CatBoost",
        )
        mlflow.log_figure(fig, "feature_importance_top15.png")
        plt.close(fig)

        # Feature importance (full CSV)
        fi_df = (
            pd.DataFrame(
                {"feature": feature_cols, "importance": res["importances"]}
            )
            .sort_values("importance", ascending=False, key=np.abs)
            .reset_index(drop=True)
        )
        fi_df.index = fi_df.index + 1
        fi_df.index.name = "rank"
        fi_path = ARTIFACTS_DIR / f"{target}_catboost_feature_importance.csv"
        fi_df.to_csv(fi_path)
        mlflow.log_artifact(str(fi_path))

        # Model artifact
        mlflow.catboost.log_model(res["model"], "model")


def log_final_calibrated(target, res, feature_cols, target_display):
    with mlflow.start_run(run_name=f"{target}_catboost_final_calibrated"):
        mlflow.set_tag("target", target)
        mlflow.set_tag("phase", "final_model")
        mlflow.set_tag("calibrated", "true")

        mlflow.log_params(
            {
                "target": target,
                "algorithm": "catboost",
                "calibrated": True,
                "calibration_method": "sigmoid",
                "calibration_cv": 5,
                "n_features": len(feature_cols),
                "train_size": res["train_size"],
                "holdout_size": res["holdout_size"],
                "train_positive_rate": round(res["train_pos_rate"], 5),
                "holdout_positive_rate": round(res["holdout_pos_rate"], 5),
            }
        )
        mlflow.log_metrics(res["metrics_cal"])

        # ROC
        fig = plot_roc(
            res["y_holdout"],
            res["y_prob_cal"],
            f"ROC — {target_display} — CatBoost Calibrated (Holdout)",
        )
        mlflow.log_figure(fig, "roc_curve.png")
        plt.close(fig)

        # PR
        fig = plot_pr(
            res["y_holdout"],
            res["y_prob_cal"],
            f"Precision-Recall — {target_display} — CatBoost Calibrated (Holdout)",
        )
        mlflow.log_figure(fig, "pr_curve.png")
        plt.close(fig)

        # Model artifact
        mlflow.sklearn.log_model(res["cal_model"], "model")


def log_calibration_comparison(target, res, target_display):
    curves = {
        "CatBoost (uncalibrated)": res["y_prob_uncal"],
        "CatBoost (calibrated)": res["y_prob_cal"],
    }
    fig = plot_calibration_comparison(
        curves,
        res["y_holdout"],
        f"Calibration Comparison — {target_display} (Holdout)",
    )

    with mlflow.start_run(run_name=f"{target}_calibration_comparison"):
        mlflow.set_tag("target", target)
        mlflow.set_tag("plot_type", "calibration_comparison")
        mlflow.log_figure(fig, "calibration_comparison.png")

    fig.savefig(
        ARTIFACTS_DIR / f"{target}_calibration_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    configure_mlflow()

    # Cache loaded data per file to avoid re-reading for targets sharing the same matrix
    _data_cache = {}

    for target in TARGETS:
        cfg = TARGET_CONFIG[target]
        target_display = cfg["display"]
        horizon_days = cfg["horizon_days"]
        data_path = str(cfg["data_path"])

        if data_path not in _data_cache:
            print(f"\nLoading {cfg['data_path'].name}...")
            df, feature_cols = load_data(target)
            _data_cache[data_path] = (df, feature_cols)
            print(f"  {df.shape[0]:,} rows, {len(feature_cols)} features")
        else:
            df, feature_cols = _data_cache[data_path]

        boundaries = get_fold_boundaries(df, horizon_days)

        print(f"\n{'=' * 60}")
        print(f"TARGET: {target} ({target_display})")
        print(f"  Horizon: {horizon_days}d, CV folds: {len(boundaries)}")
        print(f"{'=' * 60}")

        # ── Phase 1: Expanding-Window Temporal CV ─────────────────────────────
        print("\n  Phase 1: Expanding-Window Temporal CV")
        fold_df, pooled_metrics = run_expanding_cv(df, target, feature_cols, horizon_days)

        print(
            f"\n  CV pooled metrics:"
            f"  ROC-AUC={pooled_metrics['roc_auc']:.4f}"
            f"  Brier={pooled_metrics['brier_score']:.6f}"
            f"  PR-AUC={pooled_metrics['pr_auc']:.4f}"
        )
        print(
            f"  CV per-fold ROC-AUC: "
            f"{fold_df['roc_auc'].mean():.4f} ± {fold_df['roc_auc'].std():.4f}"
        )

        log_cv_run(target, fold_df, pooled_metrics, target_display, horizon_days)

        # ── Phase 2: Final Model + Holdout ────────────────────────────────────
        print("\n  Phase 2: Final Model + Holdout Evaluation")
        res = train_final_model(df, target, feature_cols)

        print(
            f"    Uncalibrated -> ROC-AUC: {res['metrics_uncal']['roc_auc']:.4f}  "
            f"Brier: {res['metrics_uncal']['brier_score']:.6f}"
        )
        print(
            f"    Calibrated   -> ROC-AUC: {res['metrics_cal']['roc_auc']:.4f}  "
            f"Brier: {res['metrics_cal']['brier_score']:.6f}"
        )

        log_final_uncalibrated(target, res, feature_cols, target_display)
        log_final_calibrated(target, res, feature_cols, target_display)
        log_calibration_comparison(target, res, target_display)

        print(f"  Calibration comparison saved.")

    print(
        "\nDone. Run `mlflow ui --backend-store-uri mlruns` to browse results."
    )


if __name__ == "__main__":
    main()
