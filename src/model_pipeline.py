from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from src.features import FeatureEngineer

def create_pipeline(n_estimators: int = 200, max_depth: int = 20, random_state: int = 42, min_samples_split: int = 2) -> Pipeline:
    """
    Creates an end-to-end ML Pipeline combining Feature Engineering, Scaling, and RandomForestRegressor.
    """
    pipeline = Pipeline([
        ('feature_engineering', FeatureEngineer()),
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        ))
    ])
    return pipeline
