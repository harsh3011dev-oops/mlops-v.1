import pandas as pd
import numpy as np
import os
from typing import Tuple

REQUIRED_COLUMNS = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'GrLivArea', 'GarageCars', 'SalePrice']

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw CSV dataset and validates basic schema expectations.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at path: {filepath}")

    df = pd.read_csv(filepath)

    # Validate required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    # Drop missing target rows if any
    df = df.dropna(subset=['SalePrice']).reset_index(drop=True)
    return df

def split_data(df: pd.DataFrame, target_col: str = 'SalePrice', test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits features and target into train and test sets.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
