import pytest
import pandas as pd
import numpy as np
from src.drift_monitor import DataDriftDetector
from fastapi.testclient import TestClient
from predict_app import app, PROMETHEUS_AVAILABLE

client = TestClient(app)

def test_drift_detector_no_drift():
    np.random.seed(42)
    ref_df = pd.DataFrame({
        "LotArea": np.random.normal(5000, 500, 200),
        "GrLivArea": np.random.normal(1500, 100, 200),
        "OverallQual": np.random.randint(5, 9, 200)
    })
    
    curr_df = pd.DataFrame({
        "LotArea": np.random.normal(5000, 500, 200),
        "GrLivArea": np.random.normal(1500, 100, 200),
        "OverallQual": np.random.randint(5, 9, 200)
    })

    detector = DataDriftDetector(reference_df=ref_df, features=["LotArea", "GrLivArea", "OverallQual"], alpha=0.01)
    result = detector.detect_drift(curr_df)

    assert "overall_drift_detected" in result
    assert "feature_metrics" in result
    assert result["total_features_evaluated"] == 3
    assert result["overall_drift_detected"] is False

def test_drift_detector_with_drift():
    np.random.seed(42)
    ref_df = pd.DataFrame({
        "LotArea": np.random.normal(5000, 100, 100),
        "GrLivArea": np.random.normal(1500, 100, 100)
    })
    
    curr_df = pd.DataFrame({
        "LotArea": np.random.normal(50000, 100, 100),
        "GrLivArea": np.random.normal(1500, 100, 100)
    })

    detector = DataDriftDetector(reference_df=ref_df, features=["LotArea", "GrLivArea"])
    result = detector.detect_drift(curr_df)

    assert result["overall_drift_detected"] is True
    assert result["feature_metrics"]["LotArea"]["drift_detected"] is True

def test_metrics_endpoint():
    response = client.get("/metrics")
    if PROMETHEUS_AVAILABLE:
        assert response.status_code == 200
        assert "house_price_prediction_requests_total" in response.text
    else:
        assert response.status_code == 501
