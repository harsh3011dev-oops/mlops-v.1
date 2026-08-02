import optuna
import mlflow
import os
import joblib
import pandas as pd
from src.data_loader import load_data, split_data
from src.model_pipeline import create_pipeline
from src.evaluate import evaluate_model, plot_residuals

def objective(trial, X_train, X_test, y_train, y_test):
    n_estimators = trial.suggest_int("n_estimators", 50, 300, step=25)
    max_depth = trial.suggest_int("max_depth", 5, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

    with mlflow.start_run(nested=True):
        pipeline = create_pipeline(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        metrics = evaluate_model(y_test, preds)

        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split
        })
        mlflow.log_metrics(metrics)

    return metrics["R2"]

def run_tuning(n_trials: int = 10, data_path: str = "data/train.csv"):
    print(f"[+] Starting Optuna Hyperparameter Optimization ({n_trials} trials)...")
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("House Price - Hyperparameter Tuning")
        print(f"[+] MLflow tracking server connected at {tracking_uri}")
    except Exception as e:
        print(f"[!] MLflow server unreachable ({e.__class__.__name__}). Falling back to sqlite:///mlflow.db")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("House Price - Hyperparameter Tuning")

    with mlflow.start_run(run_name="Optuna_Study_Parent"):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test), n_trials=n_trials)

        print("\n[+] Optimization Complete!")
        print(f"[*] Best Trial R2 Score: {study.best_value:.4f}")
        print("[*] Best Hyperparameters:", study.best_params)

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_R2", study.best_value)

        # Train best model
        best_pipeline = create_pipeline(**study.best_params)
        best_pipeline.fit(X_train, y_train)
        best_preds = best_pipeline.predict(X_test)

        os.makedirs("model", exist_ok=True)
        joblib.dump(best_pipeline, "model/model.pkl")
        print("[+] Best tuned model pipeline saved to model/model.pkl")

if __name__ == "__main__":
    run_tuning(n_trials=10)
