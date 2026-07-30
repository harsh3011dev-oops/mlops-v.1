import pandas as pd
import mlflow
import mlflow.sklearn
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------------------------------
# STEP 1: Set up MLflow Tracking URI
# -------------------------------------------------------
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("House Price Prediction - Traced")


# -------------------------------------------------------
# STEP 2: Define traced functions
# Each @mlflow.trace decorated function becomes a
# "span" (step) visible in the MLflow UI under Traces tab
# -------------------------------------------------------

@mlflow.trace(name="load_data")
def load_data(filepath: str):
    """Load and return the CSV dataset."""
    df = pd.read_csv(filepath)
    print(f"[load_data] Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


@mlflow.trace(name="preprocess_data")
def preprocess_data(df: pd.DataFrame):
    """Split features and target, then do train/test split."""
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"[preprocess_data] Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


@mlflow.trace(name="train_model")
def train_model(X_train, y_train, n_estimators=200, max_depth=20, random_state=50):
    """Train a RandomForest model and return it."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    print(f"[train_model] Model trained with {n_estimators} estimators, depth={max_depth}")
    return model


@mlflow.trace(name="evaluate_model")
def evaluate_model(model, X_test, y_test):
    """Run predictions and compute metrics."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2  = r2_score(y_test, predictions)
    print(f"[evaluate_model] MSE: {mse:.2f} | R2: {r2:.4f}")
    return mse, r2


@mlflow.trace(name="validate_model")
def validate_model(r2: float, threshold: float = 0.7):
    """
    Quality Gate: Check if model is good enough to register.
    Industry practice: Never deploy a model below a minimum score.
    """
    if r2 >= threshold:
        print(f"[validate_model] PASSED - R2={r2:.4f} >= threshold={threshold}")
        return True
    else:
        print(f"[validate_model] FAILED - R2={r2:.4f} < threshold={threshold}")
        return False


# -------------------------------------------------------
# STEP 3: Main MLflow Run with all traced steps
# -------------------------------------------------------
ESTIMATORS   = 200
DEPTH        = 20
RANDOM_STATE = 50

with mlflow.start_run(run_name="traced_pipeline_run"):

    # Log hyperparameters
    mlflow.log_param("n_estimators",  ESTIMATORS)
    mlflow.log_param("max_depth",     DEPTH)
    mlflow.log_param("random_state",  RANDOM_STATE)

    # Execute each traced step
    df                               = load_data("data/house_prices.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model                            = train_model(X_train, y_train, ESTIMATORS, DEPTH, RANDOM_STATE)
    mse, r2                          = evaluate_model(model, X_test, y_test)
    passed                           = validate_model(r2, threshold=0.7)

    # Log metrics
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2",  r2)
    mlflow.log_metric("model_passed_gate", int(passed))

    # Save model ONLY if it passes the quality gate
    if passed:
        mlflow.sklearn.log_model(
            sk_model=model,
            name="house_model"
        )
        print("\nModel saved to MLflow artifact store.")
    else:
        print("\nModel NOT saved - did not pass quality gate.")

    print("\n--- Training Summary ---")
    print(f"MSE : {mse:.2f}")
    print(f"R2  : {r2:.4f}")
    print(f"Gate: {'PASSED' if passed else 'FAILED'}")
