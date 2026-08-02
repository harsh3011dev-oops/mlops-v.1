import pytest
import pandas as pd
import numpy as np
import os
from src.data_loader import load_data, split_data
from src.features import FeatureEngineer, engineer_features
from src.model_pipeline import create_pipeline
from src.evaluate import evaluate_model

DATA_PATH = "data/train.csv"

def test_data_loader():
    df = load_data(DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'SalePrice' in df.columns

def test_train_test_split():
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) + len(X_test) == len(df)

def test_feature_engineer_transformer():
    sample_df = pd.DataFrame([{
        "LotArea": 5000,
        "OverallQual": 7,
        "OverallCond": 5,
        "YearBuilt": 2010,
        "YearRemodAdd": 2015,
        "TotalBsmtSF": 800,
        "1stFlrSF": 800,
        "2ndFlrSF": 700,
        "GrLivArea": 1500,
        "FullBath": 2,
        "HalfBath": 1,
        "BedroomAbvGr": 3,
        "TotRmsAbvGrd": 7,
        "GarageCars": 2,
        "GarageArea": 500
    }])

    fe = FeatureEngineer(ref_year=2026)
    transformed_df = fe.transform(sample_df)

    assert "TotalSqFt" in transformed_df.columns
    assert "TotalBath" in transformed_df.columns
    assert "HouseAge" in transformed_df.columns
    assert "IsRemodeled" in transformed_df.columns
    assert "QualityScore" in transformed_df.columns
    assert transformed_df["TotalSqFt"].iloc[0] == 2300
    assert transformed_df["TotalBath"].iloc[0] == 2.5
    assert transformed_df["HouseAge"].iloc[0] == 16
    assert transformed_df["IsRemodeled"].iloc[0] == 1

def test_pipeline_fit_predict():
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)

    pipeline = create_pipeline(n_estimators=10, max_depth=5)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    assert len(predictions) == len(X_test)

    metrics = evaluate_model(y_test, predictions)
    assert "R2" in metrics
    assert "RMSE" in metrics
    assert metrics["R2"] > 0.5  # Performance sanity check
