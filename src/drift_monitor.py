import os
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

DEFAULT_FEATURES = [
    "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
    "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea", "FullBath",
    "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd", "GarageCars", "GarageArea"
]

class DataDriftDetector:
    def __init__(self, reference_df: pd.DataFrame, features: list = None, alpha: float = 0.05):
        """
        Statistical Data Drift Detector comparing current inference/test batch against reference baseline.
        :param reference_df: Reference baseline pandas DataFrame (e.g. training set)
        :param features: List of feature column names to evaluate
        :param alpha: Significance threshold for Kolmogorov-Smirnov p-value (p < alpha indicates drift)
        """
        self.reference_df = reference_df
        self.features = features or [f for f in DEFAULT_FEATURES if f in reference_df.columns]
        self.alpha = alpha

    def detect_drift(self, current_df: pd.DataFrame) -> dict:
        """
        Computes KS-test p-values, statistics, and Wasserstein distances for each feature.
        Returns detailed summary dictionary with overall drift status.
        """
        feature_results = {}
        drift_count = 0

        for col in self.features:
            if col not in current_df.columns:
                continue

            ref_vals = self.reference_df[col].dropna().values
            curr_vals = current_df[col].dropna().values

            if len(ref_vals) == 0 or len(curr_vals) == 0:
                continue

            # Kolmogorov-Smirnov 2-sample test
            ks_stat, p_value = ks_2samp(ref_vals, curr_vals)
            # Wasserstein distance (Earth Mover's Distance)
            w_dist = wasserstein_distance(ref_vals, curr_vals)

            is_drifted = bool(p_value < self.alpha)
            if is_drifted:
                drift_count += 1

            feature_results[col] = {
                "ks_stat": round(float(ks_stat), 4),
                "p_value": round(float(p_value), 4),
                "wasserstein_distance": round(float(w_dist), 4),
                "drift_detected": is_drifted,
                "ref_mean": round(float(np.mean(ref_vals)), 2),
                "curr_mean": round(float(np.mean(curr_vals)), 2)
            }

        overall_drift = drift_count > 0

        return {
            "overall_drift_detected": overall_drift,
            "drifted_features_count": drift_count,
            "total_features_evaluated": len(feature_results),
            "significance_threshold_alpha": self.alpha,
            "feature_metrics": feature_results
        }

    def generate_html_report(self, current_df: pd.DataFrame, output_path: str = "artifacts/drift_report.html") -> str:
        """
        Generates a standalone HTML report displaying drift metrics and summary tables.
        """
        report_data = self.detect_drift(current_df)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        drift_status_badge = (
            '<span style="background-color: #ef4444; color: white; padding: 6px 16px; border-radius: 20px; font-weight: bold;">DRIFT DETECTED</span>'
            if report_data["overall_drift_detected"]
            else '<span style="background-color: #10b981; color: white; padding: 6px 16px; border-radius: 20px; font-weight: bold;">NO DRIFT DETECTED</span>'
        )

        rows = ""
        for feature, metrics in report_data["feature_metrics"].items():
            status_cell = (
                '<td style="color: #ef4444; font-weight: bold;">YES</td>'
                if metrics["drift_detected"]
                else '<td style="color: #10b981; font-weight: bold;">NO</td>'
            )
            rows += f"""
            <tr>
                <td style="font-weight: 600;">{feature}</td>
                {status_cell}
                <td>{metrics['ks_stat']}</td>
                <td>{metrics['p_value']}</td>
                <td>{metrics['wasserstein_distance']}</td>
                <td>{metrics['ref_mean']}</td>
                <td>{metrics['curr_mean']}</td>
            </tr>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Data Drift Report - House Price MLOps</title>
    <style>
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 40px; background: #0f172a; color: #f8fafc; }}
        .card {{ background: #1e293b; padding: 30px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.3); max-width: 1000px; margin: 0 auto; }}
        h1 {{ color: #38bdf8; margin-top: 0; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #0f172a; color: #94a3b8; text-transform: uppercase; font-size: 12px; letter-spacing: 0.05em; }}
        tr:hover {{ background: #334155; }}
        .summary {{ display: flex; gap: 20px; margin: 25px 0; }}
        .stat {{ background: #0f172a; padding: 16px 24px; border-radius: 8px; flex: 1; text-align: center; }}
        .stat-val {{ font-size: 24px; font-weight: bold; color: #38bdf8; }}
        .stat-label {{ font-size: 12px; color: #94a3b8; margin-top: 4px; }}
    </style>
</head>
<body>
    <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h1>📊 Data Drift Analysis Report</h1>
            {drift_status_badge}
        </div>
        <p style="color: #94a3b8;">Statistical comparison of production inference data against baseline training dataset.</p>
        
        <div class="summary">
            <div class="stat">
                <div class="stat-val">{report_data['total_features_evaluated']}</div>
                <div class="stat-label">Features Analyzed</div>
            </div>
            <div class="stat">
                <div class="stat-val" style="color: {'#ef4444' if report_data['drifted_features_count'] > 0 else '#10b981'};">{report_data['drifted_features_count']}</div>
                <div class="stat-label">Drifted Features</div>
            </div>
            <div class="stat">
                <div class="stat-val">{report_data['significance_threshold_alpha']}</div>
                <div class="stat-label">Alpha Threshold (p-val)</div>
            </div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Drift Alert</th>
                    <th>KS Stat</th>
                    <th>P-Value</th>
                    <th>Wasserstein Dist</th>
                    <th>Baseline Mean</th>
                    <th>Current Mean</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
</body>
</html>
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return output_path
