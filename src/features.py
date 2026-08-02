import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer for engineered Ames domain features:
    - TotalSqFt: GrLivArea + TotalBsmtSF
    - TotalBath: FullBath + 0.5 * HalfBath
    - HouseAge: Current Year - YearBuilt
    - IsRemodeled: 1 if YearRemodAdd > YearBuilt else 0
    - QualityScore: OverallQual * OverallCond
    """
    def __init__(self, ref_year: int = 2026):
        self.ref_year = ref_year

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()

        # Engineered Feature 1: Total Living & Basement SqFt
        if 'GrLivArea' in X_out.columns and 'TotalBsmtSF' in X_out.columns:
            X_out['TotalSqFt'] = X_out['GrLivArea'] + X_out['TotalBsmtSF']

        # Engineered Feature 2: Total Bathrooms
        if 'FullBath' in X_out.columns:
            half_bath = X_out['HalfBath'] if 'HalfBath' in X_out.columns else 0
            X_out['TotalBath'] = X_out['FullBath'] + (0.5 * half_bath)

        # Engineered Feature 3: House Age
        if 'YearBuilt' in X_out.columns:
            X_out['HouseAge'] = np.maximum(0, self.ref_year - X_out['YearBuilt'])

        # Engineered Feature 4: Is Remodeled
        if 'YearRemodAdd' in X_out.columns and 'YearBuilt' in X_out.columns:
            X_out['IsRemodeled'] = (X_out['YearRemodAdd'] > X_out['YearBuilt']).astype(int)

        # Engineered Feature 5: Combined Quality Score
        if 'OverallQual' in X_out.columns and 'OverallCond' in X_out.columns:
            X_out['QualityScore'] = X_out['OverallQual'] * X_out['OverallCond']

        return X_out

def engineer_features(df: pd.DataFrame, ref_year: int = 2026) -> pd.DataFrame:
    fe = FeatureEngineer(ref_year=ref_year)
    return fe.transform(df)
