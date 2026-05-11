"""Shared Optuna helpers for CatBoost hyperparameter tuning."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import optuna

def suggest_catboost_params(trial):
    """Search space kept compact so the default 10 trials stay practical."""
    return {
        "iterations": 500,
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "depth": trial.suggest_int("depth", 2, 7),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1.0),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def save_hyperparameters(
    output_dir,
    model_name,
    best_params,
    best_value,
    n_trials,
    objective="minimize",
    objective_metric="log_loss",
    metadata=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{model_name}.json"

    payload = {
        "model": model_name,
        "objective": objective,
        "objective_metric": objective_metric,
        "best_value": float(best_value),
        "n_trials": int(n_trials),
        "params": _json_safe(best_params),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        payload["metadata"] = _json_safe(metadata)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    return path


def load_hyperparameters(input_dir, model_name):
    path = Path(input_dir) / f"{model_name}.json"
    if not path.exists():
        return None, path

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    params = payload.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"Saved hyperparameter file has no params object: {path}")

    return params, path
