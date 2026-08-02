import pandas as pd
import numpy as np
import os
from typing import Tuple

SELECTED_FEATURES = [
    'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 
    'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 
    'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 
    'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'SalePrice'
]

def load_data(filepath: str = "data/train.csv") -> pd.DataFrame:
    """
    Loads Ames Housing CSV dataset (data/train.csv) and filters to selected feature columns.
    """
    if not os.path.exists(filepath):
        # Fallback check if house_prices.csv is passed
        if os.path.exists("data/house_prices.csv"):
            filepath = "data/house_prices.csv"
        else:
            raise FileNotFoundError(f"Dataset not found at path: {filepath}")

    df = pd.read_csv(filepath)

    # Filter to selected features if available in dataset
    available_cols = [col for col in SELECTED_FEATURES if col in df.columns]
    df = df[available_cols].copy()

    # Fill missing values with median for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Drop missing target rows if any
    if 'SalePrice' in df.columns:
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
