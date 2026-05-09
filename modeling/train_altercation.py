"""
Altercation Risk Model — Resident-Level Binary Classifier (H4)
==============================================================

Predicts which residents are likely to have ANY altercation incident during
their stay. Resident-level (not temporal-window-level) because:
  - Only 257 altercation events across 127 residents (4.2% positive rate)
  - Too sparse for weekly/biweekly window prediction

Features are derived from the full history of each resident (up to SIGNAL_END).

Usage:
    cd <project_root>
    python modeling/train_altercation.py
"""

from datetime import datetime, timedelta
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
from sklearn.metrics import (
    auc,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

matplotlib.use("Agg")

# ─── Configuration ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
ARTIFACTS_DIR = ROOT / "modeling" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SIGNAL_END = datetime(2025, 2, 1)
EXPERIMENT_NAME = "incident_prediction"
RANDOM_SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════════


def load_raw():
    residents = pl.read_parquet(RAW / "residents.parquet")
    incidents = pl.read_parquet(RAW / "incidents.parquet").filter(pl.col("strikeout") == False)
    diagnoses = pl.read_parquet(RAW / "diagnoses.parquet").filter(pl.col("strikeout") == False)
    needs = pl.read_parquet(RAW / "needs.parquet").filter(pl.col("strikeout") == False)
    document_tags = pl.read_parquet(RAW / "document_tags.parquet").filter(pl.col("deleted_at").is_null())
    return residents, incidents, diagnoses, needs, document_tags


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature Engineering — Resident-Level
# ═══════════════════════════════════════════════════════════════════════════════


def build_altercation_dataset(residents, incidents, diagnoses, needs, doc_tags):
    """Build a resident-level dataset with altercation label + features."""

    # ── Target: has_altercation ───────────────────────────────────────────────
    alt_ids = (
        incidents
        .filter(pl.col("incident_type") == "Altercation")
        .select("resident_id")
        .unique()
        .with_columns(pl.lit(1).cast(pl.Int8).alias("has_altercation"))
    )

    base = residents.select(
        "resident_id", "facility_id", "date_of_birth", "admission_date",
        "discharge_date", "deceased_date",
    ).join(alt_ids, on="resident_id", how="left").with_columns(
        pl.col("has_altercation").fill_null(0)
    )

    # ── Demographics ─────────────────────────────────────────────────────────
    base = base.with_columns(
        age=((pl.lit(SIGNAL_END) - pl.col("date_of_birth")).dt.total_days() / 365.25).round(1),
        los_days=(
            pl.min_horizontal(
                pl.col("discharge_date").fill_null(pl.lit(SIGNAL_END)),
                pl.col("deceased_date").fill_null(pl.lit(SIGNAL_END)),
                pl.lit(SIGNAL_END),
            ) - pl.col("admission_date")
        ).dt.total_days(),
    )

    # ── Diagnosis flags (dementia, behavioral, cognitive) ────────────────────
    dx_codes = {
        "dx_dementia_F01": "F01", "dx_dementia_F02": "F02", "dx_dementia_F03": "F03",
        "dx_alzheimers_G30": "G30",
        "dx_alcohol_F10": "F10",
        "dx_schizophrenia_F20": "F20", "dx_schizoaffective_F25": "F25",
        "dx_bipolar_F31": "F31",
        "dx_depression_F32": "F32", "dx_depression_F33": "F33",
        "dx_anxiety_F41": "F41",
        "dx_intellectual_F70_F79": "F7",
    }

    dx_exprs = []
    for col_name, prefix in dx_codes.items():
        dx_exprs.append(
            pl.col("icd_10_code").str.starts_with(prefix).any().cast(pl.Int8).alias(col_name)
        )

    dx_flags = (
        base.select("resident_id")
        .join(diagnoses.select("resident_id", "icd_10_code"), on="resident_id", how="left")
        .filter(pl.col("icd_10_code").is_not_null())
        .group_by("resident_id")
        .agg(
            *dx_exprs,
            pl.col("icd_10_code").n_unique().alias("dx_total_count"),
        )
    )

    base = base.join(dx_flags, on="resident_id", how="left")
    for c in [col for col in dx_flags.columns if col != "resident_id"]:
        base = base.with_columns(pl.col(c).fill_null(0))

    # Composite: any dementia diagnosis
    base = base.with_columns(
        dx_any_dementia=pl.max_horizontal(
            "dx_dementia_F01", "dx_dementia_F02", "dx_dementia_F03", "dx_alzheimers_G30"
        ),
        dx_any_behavioral=pl.max_horizontal(
            "dx_schizophrenia_F20", "dx_schizoaffective_F25", "dx_bipolar_F31"
        ),
        dx_any_mood=pl.max_horizontal("dx_depression_F32", "dx_depression_F33", "dx_anxiety_F41"),
    )

    # ── Document tag flags ───────────────────────────────────────────────────
    tag_list = [
        "aggressive_behavior", "dementia",
        "alzheimers_disease", "psychotropic_medications",
        "wandering_risk_assessment", "hallucinations_delusions",
        "mental_status", "depression", "anxiety", "bipolar_disorder",
        "schizophrenia", "impaired_mobility",
    ]

    tag_exprs = [
        (pl.col("tag_id") == tag).any().cast(pl.Int8).alias(f"tag_{tag}")
        for tag in tag_list
    ]
    tag_count_expr = pl.col("tag_id").n_unique().alias("tag_total_unique")

    tag_flags = (
        base.select("resident_id")
        .join(doc_tags.select("resident_id", "tag_id"), on="resident_id", how="left")
        .filter(pl.col("tag_id").is_not_null())
        .group_by("resident_id")
        .agg(*tag_exprs, tag_count_expr)
    )

    base = base.join(tag_flags, on="resident_id", how="left")
    for c in [col for col in tag_flags.columns if col != "resident_id"]:
        base = base.with_columns(pl.col(c).fill_null(0))

    # ── Care plan needs ──────────────────────────────────────────────────────
    need_patterns = {
        "need_psychotropic": "psychotropic",
        "need_cognitive": "cognitive",
        "need_behavioral": "behav",
        "need_mood": "mood",
    }

    need_exprs = [
        pl.col("need_type").str.to_lowercase().str.contains(pattern).any().cast(pl.Int8).alias(col_name)
        for col_name, pattern in need_patterns.items()
    ]

    need_flags = (
        base.select("resident_id")
        .join(needs.select("resident_id", "need_type"), on="resident_id", how="left")
        .filter(pl.col("need_type").is_not_null())
        .group_by("resident_id")
        .agg(
            *need_exprs,
            pl.len().cast(pl.Int32).alias("need_total_count"),
        )
    )

    base = base.join(need_flags, on="resident_id", how="left")
    for c in [col for col in need_flags.columns if col != "resident_id"]:
        base = base.with_columns(pl.col(c).fill_null(0))

    # ── Incident history ─────────────────────────────────────────────────────
    inc_counts = (
        base.select("resident_id")
        .join(
            incidents.select("resident_id", "incident_type"),
            on="resident_id", how="left",
        )
        .filter(pl.col("incident_type").is_not_null())
        .filter(pl.col("incident_type") != "Altercation")  # exclude altercations to avoid leakage
        .group_by("resident_id")
        .agg(
            pl.len().cast(pl.Int32).alias("hist_incident_total"),
            (pl.col("incident_type") == "Fall").sum().cast(pl.Int32).alias("hist_fall_total"),
            (pl.col("incident_type") == "Wound").sum().cast(pl.Int32).alias("hist_wound_total"),
        )
    )

    base = base.join(inc_counts, on="resident_id", how="left")
    for c in [col for col in inc_counts.columns if col != "resident_id"]:
        base = base.with_columns(pl.col(c).fill_null(0))

    # ── Select features ──────────────────────────────────────────────────────
    meta_cols = ["resident_id", "facility_id", "date_of_birth", "admission_date",
                 "discharge_date", "deceased_date"]
    target_col = "has_altercation"
    feature_cols = sorted(c for c in base.columns if c not in meta_cols + [target_col])

    return base, feature_cols, target_col


# ═══════════════════════════════════════════════════════════════════════════════
#  Metrics & Plotting
# ═══════════════════════════════════════════════════════════════════════════════


def compute_metrics(y_true, y_prob):
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "pr_auc": auc(rec_curve, prec_curve),
    }


def plot_roc(y_true, y_prob, title):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, lw=2, label=f"ROC AUC = {roc_auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set(xlabel="FPR", ylabel="TPR", title=title)
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
    fig, ax = plt.subplots(figsize=(10, 8))
    styles = {"uncalibrated": ("--", "#1f77b4"), "calibrated": ("-", "#d62728")}
    for label, y_prob in results.items():
        key = "calibrated" if "(calibrated)" in label else "uncalibrated"
        ls, color = styles[key]
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, ls, marker="o", color=color, label=label)
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="Perfectly calibrated")
    ax.set(xlabel="Mean predicted probability", ylabel="Fraction of positives", title=title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  Training & Evaluation — Stratified K-Fold CV
# ═══════════════════════════════════════════════════════════════════════════════


def run_stratified_cv(X, y, feature_cols, n_splits=5):
    """Stratified K-Fold CV (resident-level, no temporal structure needed)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    fold_records = []
    all_y_true = []
    all_y_prob = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = CatBoostClassifier(
            verbose=0, random_seed=RANDOM_SEED, allow_writing_files=False,
            depth=4, l2_leaf_reg=5, iterations=300,
        )
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_prob)
        metrics["fold"] = fold_idx
        metrics["train_size"] = len(y_train)
        metrics["test_size"] = len(y_test)
        metrics["test_positives"] = int(y_test.sum())
        fold_records.append(metrics)

        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(y_prob.tolist())

        print(
            f"    Fold {fold_idx + 1}/{n_splits}: "
            f"train={len(y_train):,}  test={len(y_test):,}  "
            f"pos={int(y_test.sum())}  ROC-AUC={metrics['roc_auc']:.4f}"
        )

    fold_df = pd.DataFrame(fold_records)
    pooled_metrics = compute_metrics(np.array(all_y_true), np.array(all_y_prob))
    return fold_df, pooled_metrics


def train_final_model(X, y, feature_cols):
    """Train on full dataset: uncalibrated + calibrated with CV."""
    # Uncalibrated
    model = CatBoostClassifier(
        verbose=0, random_seed=RANDOM_SEED, allow_writing_files=False,
        depth=4, l2_leaf_reg=5, iterations=300,
    )
    model.fit(X, y)

    # Calibrated (internal CV)
    cal_model = CalibratedClassifierCV(
        estimator=CatBoostClassifier(
            verbose=0, random_seed=RANDOM_SEED, allow_writing_files=False,
            depth=4, l2_leaf_reg=5, iterations=300,
        ),
        cv=5, method="sigmoid",
    )
    cal_model.fit(X, y)

    return model, cal_model


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    print("Loading raw tables...")
    residents, incidents, diagnoses, needs, doc_tags = load_raw()

    print("Building altercation dataset...")
    df, feature_cols, target_col = build_altercation_dataset(
        residents, incidents, diagnoses, needs, doc_tags
    )

    print(f"  {df.shape[0]:,} residents, {len(feature_cols)} features")
    print(f"  Positive rate: {df[target_col].mean():.2%}")

    X = df.select(feature_cols).to_pandas()
    y = df[target_col].to_numpy().ravel()

    mlflow.set_tracking_uri(f"file://{ROOT / 'mlruns'}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ── Stratified 5-Fold CV ─────────────────────────────────────────────────
    print("\n  Phase 1: Stratified 5-Fold CV")
    fold_df, pooled_metrics = run_stratified_cv(X, y, feature_cols)

    print(
        f"\n  CV pooled: ROC-AUC={pooled_metrics['roc_auc']:.4f}"
        f"  Brier={pooled_metrics['brier_score']:.6f}"
        f"  PR-AUC={pooled_metrics['pr_auc']:.4f}"
    )
    print(
        f"  CV per-fold ROC-AUC: "
        f"{fold_df['roc_auc'].mean():.4f} ± {fold_df['roc_auc'].std():.4f}"
    )

    with mlflow.start_run(run_name="altercation_catboost_cv"):
        mlflow.set_tag("target", "has_altercation")
        mlflow.set_tag("phase", "stratified_cv")
        mlflow.log_params({
            "target": "has_altercation",
            "algorithm": "catboost",
            "cv_type": "stratified_kfold",
            "n_splits": 5,
            "n_features": len(feature_cols),
            "model_level": "resident",
        })
        mlflow.log_metrics({f"cv_pooled_{k}": v for k, v in pooled_metrics.items()})
        for m in ["roc_auc", "log_loss", "brier_score", "pr_auc"]:
            mlflow.log_metric(f"cv_{m}_mean", fold_df[m].mean())
            mlflow.log_metric(f"cv_{m}_std", fold_df[m].std())

        fold_csv = ARTIFACTS_DIR / "altercation_cv_fold_metrics.csv"
        fold_df.to_csv(fold_csv, index=False)
        mlflow.log_artifact(str(fold_csv))

    # ── Final model on full dataset ──────────────────────────────────────────
    print("\n  Phase 2: Final Model (full dataset)")
    model, cal_model = train_final_model(X, y, feature_cols)

    # Use CV pooled predictions for evaluation & calibration plots
    y_prob_uncal = np.array(pooled_metrics.pop("_y_prob", []))
    # Re-run CV to get pooled predictions for plots
    _, pooled = run_stratified_cv(X, y, feature_cols)
    # For final model artifacts, use OOF predictions from CV
    # Re-generate: collect OOF predictions
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof_prob_uncal = np.zeros(len(y))
    oof_prob_cal = np.zeros(len(y))
    for train_idx, test_idx in skf.split(X, y):
        m = CatBoostClassifier(
            verbose=0, random_seed=RANDOM_SEED, allow_writing_files=False,
            depth=4, l2_leaf_reg=5, iterations=300,
        )
        m.fit(X.iloc[train_idx], y[train_idx])
        oof_prob_uncal[test_idx] = m.predict_proba(X.iloc[test_idx])[:, 1]

        cm = CalibratedClassifierCV(
            estimator=CatBoostClassifier(
                verbose=0, random_seed=RANDOM_SEED, allow_writing_files=False,
                depth=4, l2_leaf_reg=5, iterations=300,
            ),
            cv=3, method="sigmoid",
        )
        cm.fit(X.iloc[train_idx], y[train_idx])
        oof_prob_cal[test_idx] = cm.predict_proba(X.iloc[test_idx])[:, 1]

    metrics_uncal = compute_metrics(y, oof_prob_uncal)
    metrics_cal = compute_metrics(y, oof_prob_cal)

    print(f"    Uncalibrated OOF -> ROC-AUC: {metrics_uncal['roc_auc']:.4f}  Brier: {metrics_uncal['brier_score']:.6f}")
    print(f"    Calibrated OOF   -> ROC-AUC: {metrics_cal['roc_auc']:.4f}  Brier: {metrics_cal['brier_score']:.6f}")

    # Log uncalibrated final
    with mlflow.start_run(run_name="altercation_catboost_final"):
        mlflow.set_tag("target", "has_altercation")
        mlflow.set_tag("phase", "final_model")
        mlflow.set_tag("calibrated", "false")
        mlflow.log_params({
            "target": "has_altercation",
            "algorithm": "catboost",
            "calibrated": False,
            "n_features": len(feature_cols),
            "n_residents": len(y),
            "positive_rate": round(float(y.mean()), 5),
            "model_level": "resident",
        })
        mlflow.log_metrics(metrics_uncal)

        fig = plot_roc(y, oof_prob_uncal, "ROC — Altercation — CatBoost (OOF)")
        mlflow.log_figure(fig, "roc_curve.png")
        plt.close(fig)

        fig = plot_pr(y, oof_prob_uncal, "PR — Altercation — CatBoost (OOF)")
        mlflow.log_figure(fig, "pr_curve.png")
        plt.close(fig)

        fig = plot_feature_importance(
            model.feature_importances_, feature_cols,
            "Top 15 Features — Altercation — CatBoost",
        )
        mlflow.log_figure(fig, "feature_importance_top15.png")
        plt.close(fig)

        fi_df = (
            pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
            .sort_values("importance", ascending=False, key=np.abs)
            .reset_index(drop=True)
        )
        fi_df.index = fi_df.index + 1
        fi_df.index.name = "rank"
        fi_path = ARTIFACTS_DIR / "altercation_catboost_feature_importance.csv"
        fi_df.to_csv(fi_path)
        mlflow.log_artifact(str(fi_path))

        mlflow.catboost.log_model(model, "model")

    # Log calibrated final
    with mlflow.start_run(run_name="altercation_catboost_final_calibrated"):
        mlflow.set_tag("target", "has_altercation")
        mlflow.set_tag("phase", "final_model")
        mlflow.set_tag("calibrated", "true")
        mlflow.log_params({
            "target": "has_altercation",
            "algorithm": "catboost",
            "calibrated": True,
            "calibration_method": "sigmoid",
            "n_features": len(feature_cols),
            "n_residents": len(y),
            "positive_rate": round(float(y.mean()), 5),
        })
        mlflow.log_metrics(metrics_cal)

        fig = plot_roc(y, oof_prob_cal, "ROC — Altercation — CatBoost Calibrated (OOF)")
        mlflow.log_figure(fig, "roc_curve.png")
        plt.close(fig)

        fig = plot_pr(y, oof_prob_cal, "PR — Altercation — CatBoost Calibrated (OOF)")
        mlflow.log_figure(fig, "pr_curve.png")
        plt.close(fig)

        mlflow.sklearn.log_model(cal_model, "model")

    # Calibration comparison
    fig = plot_calibration_comparison(
        {"CatBoost (uncalibrated)": oof_prob_uncal, "CatBoost (calibrated)": oof_prob_cal},
        y,
        "Calibration Comparison — Altercation (OOF)",
    )
    with mlflow.start_run(run_name="altercation_calibration_comparison"):
        mlflow.set_tag("target", "has_altercation")
        mlflow.set_tag("plot_type", "calibration_comparison")
        mlflow.log_figure(fig, "calibration_comparison.png")
    fig.savefig(ARTIFACTS_DIR / "altercation_calibration_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nDone. Run `mlflow ui --backend-store-uri mlruns` to browse results.")


if __name__ == "__main__":
    main()
