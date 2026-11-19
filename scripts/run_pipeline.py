import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run

if __name__ == "__main__":
    result, selection, scores, failures = run()
    print("\nHMM Sentinel completed.")
    print("Selected states:", result["selected_states"])
    print("Threshold:", result["threshold"])
    print("Anomaly windows:", result["anomaly_windows"])
    print("Precision:", round(result["precision"], 4))
    print("Recall:", round(result["recall"], 4))
    print("F1:", round(result["f1"], 4))
