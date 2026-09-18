"""
fault_classifier.py
====================
Multi-class fault classifier: XGBoost (primary) with a Random Forest fallback.
Classifies engine sensor telemetry into 7 classes (6 fault modes + normal):
  0  normal
  1  misfire
  2  injector
  3  lubrication
  4  sensor_drift
  5  combustion
  6  overheating

Matches `engine/alerts` payload contract:
  - fault_type (str)
  - confidence (float)
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

FAULT_LABELS = [
    "normal",
    "misfire",
    "injector",
    "lubrication",
    "sensor_drift",
    "combustion",
    "overheating",
]

FEATURE_COLS = [
    "rpm", "egt", "cht", "oil_pressure",
    "oil_temp", "fuel_flow", "vibration",
    "altitude", "ambient_temp",
]


def get_default_model_path() -> Path:
    """Resolve model path supporting Docker (/app/models) and local host development."""
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return Path(env_path)
    env_dir = os.getenv("MODEL_DIR")
    if env_dir:
        return Path(env_dir) / "fault_classifier.pkl"
    app_models = Path("/app/models")
    if app_models.exists():
        return app_models / "fault_classifier.pkl"
    return Path(__file__).resolve().parent.parent / "models" / "fault_classifier.pkl"


def get_default_data_path() -> Path:
    """Resolve training dataset path supporting Docker and local development."""
    env_path = os.getenv("DATA_PATH")
    if env_path:
        return Path(env_path)
    app_data = Path("/app/data/labelled_telemetry.csv")
    if app_data.exists():
        return app_data
    return Path(__file__).resolve().parent.parent / "data" / "labelled_telemetry.csv"


class FaultClassifier:
    """
    Multi-class fault classifier with XGBoost as primary and Random Forest as fallback.
    Exposes train(), predict(), save(), load(), and get_model() for SHAP explainer.
    """

    def __init__(self, model_path: Optional[Path | str] = None) -> None:
        self.model_path = Path(model_path) if model_path else get_default_model_path()
        self._model = None
        self._classes: list[str] = list(FAULT_LABELS)
        self._features: list[str] = list(FEATURE_COLS)
        self._algorithm: str = "stub"
        self._metrics: dict = {}
        self.last_probabilities: dict[str, float] = {}

    @property
    def model(self):
        """Expose underlying tree estimator for SHAP TreeExplainer integration."""
        return self._model

    def get_model(self):
        """Getter for SHAP explainer interoperability."""
        return self._model

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    @property
    def algorithm(self) -> str:
        return self._algorithm

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self, path: Optional[Path | str] = None) -> bool:
        """Load trained model from disk."""
        target_path = Path(path) if path else self.model_path
        if target_path.exists():
            with target_path.open("rb") as f:
                data = pickle.load(f)

            if isinstance(data, dict) and "model" in data:
                self._model = data["model"]
                self._classes = data.get("classes", list(FAULT_LABELS))
                self._features = data.get("features", list(FEATURE_COLS))
                self._algorithm = data.get("algorithm", "unknown")
                self._metrics = data.get("metrics", {})
            else:
                # Legacy raw estimator pickle
                self._model = data
                self._classes = list(FAULT_LABELS)
                self._algorithm = type(data).__name__

            logger.info("Fault classifier (%s) loaded from %s", self._algorithm, target_path)
            return True

        logger.warning("No fault classifier at %s — returning stub predictions", target_path)
        return False

    def save(self, path: Optional[Path | str] = None) -> Path:
        """Persist model and metadata to disk."""
        target_path = Path(path) if path else self.model_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "classes": self._classes,
            "features": self._features,
            "algorithm": self._algorithm,
            "metrics": self._metrics,
        }
        with target_path.open("wb") as f:
            pickle.dump(payload, f)
        logger.info("Fault classifier saved to %s", target_path)
        return target_path

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: Optional[pd.DataFrame] = None,
        data_path: Optional[Path | str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        force_fallback: bool = False,
    ) -> dict:
        """
        Train multi-class fault classifier on labelled telemetry data.

        Tries XGBoost first; if unavailable or force_fallback is True,
        falls back to scikit-learn RandomForestClassifier.

        Args:
            df: Optional in-memory DataFrame containing FEATURE_COLS + 'fault_label'.
            data_path: Optional path to labelled telemetry CSV.
            test_size: Ratio of holdout test set.
            random_state: Seed for reproducibility.
            force_fallback: If True, forces Random Forest training.

        Returns:
            Dictionary containing evaluation metrics (accuracy, f1_macro, classification_report).
        """
        if df is None:
            resolved_path = Path(data_path) if data_path else get_default_data_path()
            if not resolved_path.exists():
                raise FileNotFoundError(
                    f"Labelled training dataset not found at {resolved_path}. "
                    "Run dataset generator or supply data_path/df."
                )
            logger.info("Loading training data from %s...", resolved_path)
            df = pd.read_csv(resolved_path)

        # Ensure all required features are present
        missing_features = [col for col in FEATURE_COLS if col not in df.columns]
        if missing_features:
            raise ValueError(f"Dataset missing required feature columns: {missing_features}")

        if "fault_label" not in df.columns:
            raise ValueError("Dataset missing 'fault_label' column")

        # Map labels to canonical class indices
        self._classes = list(FAULT_LABELS)
        label_to_idx = {label: idx for idx, label in enumerate(self._classes)}

        # Filter to known classes
        valid_mask = df["fault_label"].isin(label_to_idx)
        df_valid = df[valid_mask]
        if len(df_valid) == 0:
            raise ValueError("No rows with valid fault labels found in training dataset")

        X = df_valid[FEATURE_COLS].values.astype(np.float32)
        y = df_valid["fault_label"].map(label_to_idx).values.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        fitted_model = None
        algo_name = None

        if not force_fallback:
            try:
                import xgboost as xgb
                logger.info("Training primary XGBoost multi-class classifier...")
                model = xgb.XGBClassifier(
                    n_estimators=250,
                    max_depth=5,
                    learning_rate=0.08,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    eval_metric="mlogloss",
                    random_state=random_state,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train)
                fitted_model = model
                algo_name = "XGBoost"
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "XGBoost training failed or unavailable (%s). Falling back to Random Forest.",
                    exc,
                )

        if fitted_model is None:
            from sklearn.ensemble import RandomForestClassifier
            logger.info("Training fallback Random Forest multi-class classifier...")
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            fitted_model = model
            algo_name = "RandomForest"

        y_pred = fitted_model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1_macro = float(f1_score(y_test, y_pred, average="macro"))
        report = classification_report(
            y_test, y_pred, target_names=self._classes, output_dict=True, zero_division=0
        )

        self._model = fitted_model
        self._algorithm = algo_name
        self._metrics = {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "test_samples": int(len(y_test)),
            "algorithm": algo_name,
            "classification_report": report,
        }

        logger.info(
            "%s training complete: Accuracy=%.4f, Macro F1=%.4f (test set: %d samples)",
            algo_name,
            acc,
            f1_macro,
            len(y_test),
        )

        self.save()
        return self._metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self, sample: dict, return_probs: bool = False
    ) -> Union[tuple[str, float], tuple[str, float, dict[str, float]]]:
        """
        Predict fault mode and confidence for an incoming telemetry sample.

        Conforms strictly to architecture.md engine/alerts contract:
          - fault_type: str (one of the 6 fault labels or 'normal')
          - confidence: float [0.0, 1.0]

        Args:
            sample: Dict containing telemetry features (rpm, egt, cht, etc.)
            return_probs: If True, also returns full probability distribution mapping.

        Returns:
            (fault_type, confidence) by default.
            (fault_type, confidence, prob_map) when return_probs=True.
        """
        if self._model is None:
            stub_probs = {c: (1.0 if c == "normal" else 0.0) for c in self._classes}
            self.last_probabilities = stub_probs
            if return_probs:
                return "normal", 1.0, stub_probs
            return "normal", 1.0

        x = np.array([[sample.get(c, 0.0) for c in FEATURE_COLS]], dtype=np.float32)
        probs = self._model.predict_proba(x)[0]
        idx = int(np.argmax(probs))
        fault_type = self._classes[idx]
        confidence = round(float(probs[idx]), 4)

        prob_map = {self._classes[i]: round(float(probs[i]), 4) for i in range(len(probs))}
        self.last_probabilities = prob_map

        if return_probs:
            return fault_type, confidence, prob_map
        return fault_type, confidence

    def predict_proba(self, sample: dict) -> dict[str, float]:
        """Return the probability distribution over all fault classes."""
        if self._model is None:
            return {c: (1.0 if c == "normal" else 0.0) for c in self._classes}
        x = np.array([[sample.get(c, 0.0) for c in FEATURE_COLS]], dtype=np.float32)
        probs = self._model.predict_proba(x)[0]
        return {self._classes[i]: round(float(probs[i]), 4) for i in range(len(probs))}


# ---------------------------------------------------------------------------
# CLI entrypoint for offline training
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Digital Twin Fault Classifier")
    parser.add_argument("--data", type=str, default=None, help="Path to labelled telemetry CSV")
    parser.add_argument("--fallback", action="store_true", help="Force Random Forest fallback")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    classifier = FaultClassifier()
    metrics = classifier.train(data_path=args.data, force_fallback=args.fallback)

    print("\n" + "=" * 60)
    print(f"Algorithm: {metrics['algorithm']}")
    print(f"Overall Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1 Score:   {metrics['f1_macro']:.4f}")
    print("=" * 60)

    # Print summary per class
    rep = metrics["classification_report"]
    print(f"{'Class':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 52)
    for c in FAULT_LABELS:
        if c in rep:
            p = rep[c]["precision"]
            r = rep[c]["recall"]
            f = rep[c]["f1-score"]
            print(f"{c:<15} | {p:<10.2f} | {r:<10.2f} | {f:<10.2f}")
    print("=" * 60)
