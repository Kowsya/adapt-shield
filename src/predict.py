"""
STEP 5: PREDICTION (INFERENCE)
================================
Loads the trained model and makes predictions on new data.
This is used by the FastAPI app.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODELS_DIR = Path("models")


class AdaptShieldPredictor:
    """Loads model + preprocessors and makes predictions."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self._loaded = False

    def load(self):
        """Load model and all preprocessors from disk."""
        try:
            self.model = joblib.load(MODELS_DIR / "model.pkl")
            self.scaler = joblib.load(MODELS_DIR / "scaler.pkl")
            self.label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
            self.feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
            self._loaded = True
            print("✅ Model loaded successfully!")
            return True
        except FileNotFoundError as e:
            print(f"❌ Model file not found: {e}")
            print("   Run training first: python src/train.py")
            return False

    def predict(self, features: dict) -> dict:
        """
        Make a prediction from a dict of feature values.
        
        Args:
            features: dict like {"Flow Duration": 123, "Protocol": 6, ...}
        
        Returns:
            dict with prediction, confidence, and is_malicious flag
        """
        if not self._loaded:
            self.load()

        # Build a dataframe with the expected feature order
        row = pd.DataFrame([features])

        # Add missing columns with 0
        for col in self.feature_names:
            if col not in row.columns:
                row[col] = 0

        # Keep only expected features in correct order
        row = row[self.feature_names]

        # Scale
        row_scaled = self.scaler.transform(row)

        # Predict
        pred_int = self.model.predict(row_scaled)[0]
        pred_label = self.label_encoder.inverse_transform([pred_int])[0]

        # Confidence (probability)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(row_scaled)[0]
            confidence = float(np.max(proba))
        else:
            confidence = 1.0

        is_malicious = pred_label != "BENIGN"

        return {
            "prediction": pred_label,
            "is_malicious": is_malicious,
            "confidence": round(confidence, 4),
            "action": "BLOCK" if is_malicious else "ALLOW",
        }

    def predict_batch(self, df: pd.DataFrame) -> list:
        """Predict for multiple rows at once."""
        if not self._loaded:
            self.load()

        # Align columns
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names]

        # Scale and predict
        scaled = self.scaler.transform(df)
        preds_int = self.model.predict(scaled)
        preds_labels = self.label_encoder.inverse_transform(preds_int)

        results = []
        for label in preds_labels:
            results.append({
                "prediction": label,
                "is_malicious": label != "BENIGN",
                "action": "BLOCK" if label != "BENIGN" else "ALLOW",
            })
        return results


# Singleton predictor (reuse across API requests)
predictor = AdaptShieldPredictor()


if __name__ == "__main__":
    # Test with dummy data
    predictor.load()
    test_features = {col: 0 for col in predictor.feature_names}
    result = predictor.predict(test_features)
    print(f"\nTest prediction: {result}")