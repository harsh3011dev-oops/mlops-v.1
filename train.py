import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load CSV
df = pd.read_csv("data/house_prices.csv")

# Features
X = df.drop("SalePrice", axis=1)

# Target
y = df["SalePrice"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Experiment
mlflow.set_experiment("House Price Prediction")

with mlflow.start_run():

    ESTIMATORS = 200
    DEPTH = 20
    RANDOM_STATE = 50

    model = RandomForestRegressor(
        n_estimators=ESTIMATORS,
        max_depth=DEPTH,
        random_state=RANDOM_STATE
)

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    mse = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    # Log Parameters
    mlflow.log_param("n_estimators", ESTIMATORS)
    mlflow.log_param("max_depth", DEPTH)
    mlflow.log_param("random_state", RANDOM_STATE)

    # Log Metrics
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2", r2)

    # Save Model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="house_model"
    )

    print("Training Completed")
    print("MSE :", mse)
    print("R2  :", r2)
