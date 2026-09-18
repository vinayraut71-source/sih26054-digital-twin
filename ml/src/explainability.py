"""
explainability.py
=================
SHAP integration for post-hoc explanations of fault classifier predictions.

Returns the top-K feature names sorted by |SHAP value|.

TODO (ML2):
  - Compute background dataset from healthy telemetry for KernelExplainer.
  - Cache the explainer object (expensive to create per-request).
  - Add SHAP waterfall / bar chart generation for the PDF report feature.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "rpm", "egt", "cht", "oil_pressure",
    "oil_temp", "fuel_flow", "vibration",
    "altitude", "ambient_temp",
]


class SHAPExplainer:
    def __init__(self) -> None:
        self._explainer = None

    def setup(self, model, background_data: np.ndarray) -> None:
        """
        Initialise a TreeExplainer (XGBoost) or KernelExplainer.

        TODO (ML2): call this once at startup after the classifier is loaded.
        """
        try:
            import shap

            self._explainer = shap.TreeExplainer(model)
            logger.info("SHAP TreeExplainer initialised")
        except Exception as exc:  # noqa: BLE001
            logger.warning("SHAP setup failed: %s — explanations disabled", exc)

    def top_features(self, sample: dict, k: int = 3) -> list[str]:
        """
        Return the top-k feature names by absolute SHAP value.

        Falls back to a placeholder list if explainer is not set up.
        """
        if self._explainer is None:
            # TODO: remove stub after SHAP is wired up
            return FEATURE_COLS[:k]

        try:
            x = np.array([[sample.get(c, 0.0) for c in FEATURE_COLS]])
            values = self._explainer.shap_values(x)
            # Handle list, 3D array (samples, features, classes), or 2D array
            if isinstance(values, list):
                abs_vals = np.max([np.abs(v[0]) for v in values], axis=0)
            elif isinstance(values, np.ndarray) and values.ndim == 3:
                abs_vals = np.max(np.abs(values[0]), axis=-1)
            else:
                abs_vals = np.abs(values[0])
            top_idx = [int(i) for i in np.argsort(abs_vals)[::-1][:k]]
            return [FEATURE_COLS[i] for i in top_idx]
        except Exception as exc:  # noqa: BLE001
            logger.warning("SHAP inference failed: %s", exc)
            return FEATURE_COLS[:k]
