import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes comprehensive regression evaluation metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "MAPE": float(mape)
    }

def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, output_path: str = "artifacts/residual_plot.png"):
    """
    Generates and saves a residual analysis scatter plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.6, color="#326CE5")
    plt.axhline(y=0, color="red", linestyle="--")
    plt.title("Residual Analysis Plot")
    plt.xlabel("Predicted Values (₹)")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path
