import os
import pandas as pd
from src.data_loader import load_data
from src.drift_monitor import DataDriftDetector

def run_drift_analysis(train_path: str = "data/train.csv", output_html: str = "artifacts/drift_report.html"):
    if not os.path.exists(train_path):
        print(f"Error: Training baseline file not found at {train_path}")
        return

    print("Loading baseline training dataset...")
    ref_df = load_data(train_path)

    # For demonstration/validation, sample/perturb data or split to compare
    test_df = ref_df.sample(frac=0.3, random_state=42).copy()
    
    print("Initializing Data Drift Detector...")
    detector = DataDriftDetector(reference_df=ref_df)

    print("Executing statistical Kolmogorov-Smirnov & Wasserstein drift tests...")
    results = detector.detect_drift(test_df)
    
    print(f"\n--- Drift Summary ---")
    print(f"Overall Drift Detected: {results['overall_drift_detected']}")
    print(f"Drifted Features: {results['drifted_features_count']} / {results['total_features_evaluated']}")

    report_path = detector.generate_html_report(test_df, output_path=output_html)
    print(f"HTML Drift Report successfully saved to: {report_path}")

if __name__ == "__main__":
    run_drift_analysis()
