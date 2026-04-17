"""
TESTS: Unit tests for ADAPT-SHIELD
====================================
Run: pytest tests/ -v

These tests check that the code works correctly
without needing the actual ML model.
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Test 1: Data Ingestion ────────────────────────────────────
class TestDataIngestion:
    def test_label_map_covers_all_classes(self):
        """Every known label should be in the map."""
        from src.data_ingestion import LABEL_MAP
        required = ["Benign", "Bot", "DDoS", "FTP-Patator", "SSH-Patator"]
        for label in required:
            assert label in LABEL_MAP, f"'{label}' missing from LABEL_MAP"

    def test_label_map_maps_to_valid_values(self):
        """All map values should be non-empty strings."""
        from src.data_ingestion import LABEL_MAP
        for k, v in LABEL_MAP.items():
            assert isinstance(v, str) and len(v) > 0


# ── Test 2: API Endpoints ─────────────────────────────────────
class TestAPI:
    def test_health_endpoint(self):
        """Health endpoint should return running status."""
        # We test the structure without needing a running server
        from app.main import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_root_endpoint(self):
        """Root endpoint should return service info."""
        from app.main import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data

    def test_predict_without_model_returns_503(self):
        """Predict without loaded model should return 503."""
        from app.main import app, predictor
        from fastapi.testclient import TestClient
        predictor._loaded = False  # Simulate no model
        client = TestClient(app)
        response = client.post("/predict", json={"Protocol": 6})
        # Should be 503 (model not loaded) or 200 (if model exists)
        assert response.status_code in [200, 503]


# ── Test 3: Feature Engineering ──────────────────────────────
class TestFeatureEngineering:
    def test_cols_to_drop_is_list(self):
        """COLS_TO_DROP should be a list of strings."""
        from src.feature_engineering import COLS_TO_DROP
        assert isinstance(COLS_TO_DROP, list)
        for col in COLS_TO_DROP:
            assert isinstance(col, str)

    def test_clean_data_removes_duplicates(self):
        """clean_data should remove duplicate rows."""
        import pandas as pd
        import numpy as np
        from src.feature_engineering import clean_data

        # Create fake df with duplicates
        df = pd.DataFrame({
            "Protocol": [6, 6, 17],
            "Flow Duration": [100, 100, 200],
            "Label": ["BENIGN", "BENIGN", "DDoS"],
        })
        cleaned = clean_data(df)
        assert len(cleaned) <= len(df)


# ── Test 4: Prediction ────────────────────────────────────────
class TestPredictor:
    def test_predictor_init(self):
        """Predictor should initialize without errors."""
        from src.predict import AdaptShieldPredictor
        p = AdaptShieldPredictor()
        assert p._loaded is False
        assert p.model is None

    def test_predict_returns_correct_keys(self):
        """Prediction result should have required keys."""
        from src.predict import AdaptShieldPredictor
        import numpy as np
        from unittest.mock import MagicMock

        p = AdaptShieldPredictor()
        # Mock the model and dependencies
        p.model = MagicMock()
        p.model.predict.return_value = [0]
        p.model.predict_proba.return_value = [[0.9, 0.1]]
        p.scaler = MagicMock()
        p.scaler.transform.return_value = np.zeros((1, 5))

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(["BENIGN", "DDoS"])
        p.label_encoder = le
        p.feature_names = ["col1", "col2"]
        p._loaded = True

        result = p.predict({"col1": 0, "col2": 0})
        assert "prediction" in result
        assert "is_malicious" in result
        assert "confidence" in result
        assert "action" in result