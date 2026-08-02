import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer for engineered domain features:
    - HouseAge: Current Year / Reference - YearBuilt
    - QualityScore: OverallQual * OverallCond
    - QualityPerSqFt: (OverallQual * OverallCond) / GrLivArea
    """
    def __init__(self, ref_year: int = 2026):
        self.ref_year = ref_year

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()

        # Engineered Feature 1: House Age
        if 'YearBuilt' in X_out.columns:
            X_out['HouseAge'] = np.maximum(0, self.ref_year - X_out['YearBuilt'])

        # Engineered Feature 2: Combined Quality Score
        if 'OverallQual' in X_out.columns and 'OverallCond' in X_out.columns:
            X_out['QualityScore'] = X_out['OverallQual'] * X_out['OverallCond']

        # Engineered Feature 3: Quality Score per SqFt
        if 'QualityScore' in X_out.columns and 'GrLivArea' in X_out.columns:
            X_out['QualityPerSqFt'] = X_out['QualityScore'] / (X_out['GrLivArea'] + 1e-5)

        return X_out

def engineer_features(df: pd.DataFrame, ref_year: int = 2026) -> pd.DataFrame:
    """
    Functional interface for feature engineering.
    """
    fe = FeatureEngineer(ref_year=ref_year)
    return fe.transform(df)
