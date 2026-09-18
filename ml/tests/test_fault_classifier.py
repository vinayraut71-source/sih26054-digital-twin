"""
Unit and Integration tests for FaultClassifier and ML pipeline integration.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from ml.src.fault_classifier import (
    FaultClassifier,
    FAULT_LABELS,
    FEATURE_COLS,
    get_default_model_path,
)
from ml.src.explainability import SHAPExplainer


class TestFaultClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = FaultClassifier()
        loaded = cls.classifier.load()
        if not loaded:
            # If not yet trained, train on a small synthetic dataframe
            data = []
            for fault in FAULT_LABELS:
                for _ in range(30):
                    row = {col: float(np.random.uniform(10, 100)) for col in FEATURE_COLS}
                    row["fault_label"] = fault
                    data.append(row)
            df = pd.DataFrame(data)
            cls.classifier.train(df=df)

    def test_model_loaded(self):
        self.assertIsNotNone(self.classifier.model)
        self.assertIn(self.classifier.algorithm, ["XGBoost", "RandomForest"])

    def test_predict_contract_architecture(self):
        """Verify predict output conforms strictly to docs/architecture.md schema."""
        sample = {
            "rpm": 2412.5,
            "egt": 731.2,
            "cht": 177.6,
            "oil_pressure": 4.51,
            "oil_temp": 91.3,
            "fuel_flow": 15.8,
            "vibration": 0.112,
            "altitude": 3000.0,
            "ambient_temp": -4.5,
        }

        # 1. Default 2-element tuple unpack (fault_type, confidence)
        result = self.classifier.predict(sample)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        fault_type, confidence = result

        self.assertIsInstance(fault_type, str)
        self.assertIn(fault_type, FAULT_LABELS)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

        # 2. Return probabilities mapping
        res_probs = self.classifier.predict(sample, return_probs=True)
        self.assertEqual(len(res_probs), 3)
        ft, conf, prob_map = res_probs
        self.assertEqual(ft, fault_type)
        self.assertEqual(conf, confidence)
        self.assertIsInstance(prob_map, dict)
        self.assertEqual(len(prob_map), len(FAULT_LABELS))
        total_prob = sum(prob_map.values())
        self.assertAlmostEqual(total_prob, 1.0, places=2)

    def test_random_forest_fallback(self):
        """Verify fallback to Random Forest works as expected."""
        data = []
        for fault in FAULT_LABELS:
            for _ in range(25):
                row = {col: float(np.random.uniform(10, 100)) for col in FEATURE_COLS}
                row["fault_label"] = fault
                data.append(row)
        df = pd.DataFrame(data)

        rf_classifier = FaultClassifier(model_path=Path("ml/models/test_rf.pkl"))
        metrics = rf_classifier.train(df=df, force_fallback=True)

        self.assertEqual(metrics["algorithm"], "RandomForest")
        self.assertEqual(rf_classifier.algorithm, "RandomForest")

        # Test predict with RF
        sample = {col: 50.0 for col in FEATURE_COLS}
        ft, conf = rf_classifier.predict(sample)
        self.assertIn(ft, FAULT_LABELS)
        self.assertIsInstance(conf, float)

        # Cleanup test artifact
        test_path = Path("ml/models/test_rf.pkl")
        if test_path.exists():
            test_path.unlink()

    def test_shap_compatibility(self):
        """Verify the trained classifier model is directly consumable by SHAP."""
        explainer = SHAPExplainer()
        raw_model = self.classifier.get_model()
        self.assertIsNotNone(raw_model)

        # Explainer setup should accept the classifier's tree booster without error
        explainer.setup(raw_model, None)

        sample = {
            "rpm": 2250.0,
            "egt": 820.0,
            "cht": 190.0,
            "oil_pressure": 2.5,
            "oil_temp": 115.0,
            "fuel_flow": 13.0,
            "vibration": 0.35,
            "altitude": 1000.0,
            "ambient_temp": 15.0,
        }
        top_feats = explainer.top_features(sample, k=3)
        self.assertIsInstance(top_feats, list)
        self.assertEqual(len(top_feats), 3)
        for f in top_feats:
            self.assertIn(f, FEATURE_COLS)


if __name__ == "__main__":
    unittest.main()
