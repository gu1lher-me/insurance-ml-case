"""Calibration helpers for production-style probability models."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError


class TemporalCalibratedClassifier(BaseEstimator, ClassifierMixin):
    """Wrap a fitted classifier with a fitted probability calibrator."""

    def __init__(self, base_model=None, calibrator=None):
        self.base_model = base_model
        self.calibrator = calibrator
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        if self.base_model is None or self.calibrator is None:
            raise NotFittedError("Both base_model and calibrator must be fitted.")

        raw_probability = np.asarray(self.base_model.predict_proba(X))[:, 1]
        calibrated_probability = np.asarray(
            self.calibrator.predict(raw_probability), dtype=float
        )
        calibrated_probability = np.clip(calibrated_probability, 0.0, 1.0)
        return np.column_stack(
            [1.0 - calibrated_probability, calibrated_probability]
        )

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
