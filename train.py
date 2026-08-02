import os
import joblib
import mlflow
import mlflow.sklearn
from src.data_loader import load_data, split_data
from src.model_pipeline import create_pipeline
from src.evaluate import evaluate_model, plot_residuals

def main():
    print("[+] Starting End-to-End Production Model Training...")
    
    # 1. Load & Split Data (Ames Housing Dataset: data/train.csv)
    data_path = "data/train.csv"
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)

    # 2. MLflow Tracking Setup — falls back to local SQLite if no server running
    # MLflow 3.x dropped file store; use sqlite:///mlflow.db as the local fallback
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("House Price Prediction - Production")
        print(f"[+] MLflow tracking server connected at {tracking_uri}")
    except Exception as e:
        print(f"[!] MLflow server unreachable ({e.__class__.__name__}). Falling back to sqlite:///mlflow.db")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("House Price Prediction - Production")

    # 3. Model Parameters
    ESTIMATORS = 200
    DEPTH = 20
    MIN_SAMPLES = 2
    RANDOM_STATE = 42

    with mlflow.start_run(run_name="Production_RandomForest_Pipeline") as run:
        # Build & Fit Pipeline
        pipeline = create_pipeline(
            n_estimators=ESTIMATORS,
            max_depth=DEPTH,
            min_samples_split=MIN_SAMPLES,
            random_state=RANDOM_STATE
        )
        pipeline.fit(X_train, y_train)

        # Predict & Evaluate
        predictions = pipeline.predict(X_test)
        metrics = evaluate_model(y_test, predictions)

        # Log Parameters & Metrics
        mlflow.log_param("n_estimators", ESTIMATORS)
        mlflow.log_param("max_depth", DEPTH)
        mlflow.log_param("min_samples_split", MIN_SAMPLES)
        mlflow.log_param("random_state", RANDOM_STATE)

        for metric_name, metric_val in metrics.items():
            mlflow.log_metric(metric_name, metric_val)

        # Plot Residuals & Log Artifact
        res_plot_path = plot_residuals(y_test, predictions)
        mlflow.log_artifact(res_plot_path, artifact_path="plots")

        # Save & Register Model in MLflow Registry
        try:
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="house_model",
                registered_model_name="HousePricePredictor",
                # MLflow 3.x requires explicitly trusting custom sklearn transformers
                skops_trusted_types=["src.features.FeatureEngineer"]
            )
        except Exception as reg_err:
            print(f"[!] Warning: Model registry skipped or unavailable ({reg_err})")

        # Export for Local FastAPI Serving
        os.makedirs("model", exist_ok=True)
        joblib.dump(pipeline, "model/model.pkl")
        print("[+] Production Model Pipeline successfully saved to model/model.pkl")

        print(f"[*] Training Metrics: R2 = {metrics['R2']:.4f} | RMSE = {metrics['RMSE']:.2f} | MAE = {metrics['MAE']:.2f}")
        print("[+] Training & Registration Complete!")

if __name__ == "__main__":
    main()
