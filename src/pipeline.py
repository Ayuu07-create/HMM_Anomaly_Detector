import json
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    RAW_CSV,
    OUTPUT_DIR,
    FEATURES,
    TRAIN_END,
    HMM_STATES,
    RANDOM_STATE,
    WINDOW_SIZE,
    STRIDE,
    THRESHOLD_QUANTILE,
    VALIDATION_FRACTION,
    FAILURE_WINDOWS,
)

from src.data import (
    load_sensor_data,
    validate_features,
    chronological_split,
)

from src.model import (
    select_hmm_model,
    fit_hmm,
)

from src.detector import (
    window_scores,
    merge_anomaly_regions,
)

from src.evaluation import (
    pointwise_metrics,
    event_detection,
)


def run():

    print("\n========================================")
    print("       HMM SENTINEL ANOMALY DETECTOR")
    print("========================================\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading dataset...")
    df = load_sensor_data(RAW_CSV)

    print(f"      Rows: {len(df):,}")

    validate_features(df, FEATURES)

    print("[2/6] Creating chronological train/test split...")

    train_df, test_df = chronological_split(
        df,
        TRAIN_END
    )

    print(f"      Training rows: {len(train_df):,}")
    print(f"      Test rows:     {len(test_df):,}")

    print("[3/6] Preparing sensor features...")

    train_raw = train_df[FEATURES].apply(
        pd.to_numeric,
        errors="coerce"
    )

    test_raw = test_df[FEATURES].apply(
        pd.to_numeric,
        errors="coerce"
    )

    medians = train_raw.median()

    train_raw = train_raw.fillna(medians)
    test_raw = test_raw.fillna(medians)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(train_raw)
    X_test = scaler.transform(test_raw)

    split = int(
        len(X_train) *
        (1 - VALIDATION_FRACTION)
    )

    X_fit = X_train[:split]
    X_val = X_train[split:]

    print(f"      Model-training rows: {len(X_fit):,}")
    print(f"      Validation rows:     {len(X_val):,}")

    print("[4/6] Selecting HMM hidden-state count...")

    model, selection = select_hmm_model(
        X_fit,
        X_val,
        HMM_STATES,
        RANDOM_STATE
    )

    selected_states = model.n_components

    print(
        f"\n      Selected states: {selected_states}"
    )

    print(
        "\n      Training final HMM on complete "
        "normal baseline..."
    )

    final_model = fit_hmm(
        X_train,
        selected_states,
        RANDOM_STATE
    )

    print("[5/6] Learning anomaly threshold...")

    train_timestamps = (
        train_df["timestamp"]
        .reset_index(drop=True)
    )

    train_scores = window_scores(
        final_model,
        X_train,
        train_timestamps,
        WINDOW_SIZE,
        STRIDE
    )

    if train_scores.empty:
        raise ValueError(
            "No training windows were generated."
        )

    threshold = float(
        train_scores["score"].quantile(
            THRESHOLD_QUANTILE
        )
    )

    print(
        f"      Threshold: {threshold:.6f}"
    )

    print("[6/6] Scanning test stream...")

    test_timestamps = (
        test_df["timestamp"]
        .reset_index(drop=True)
    )

    test_scores = window_scores(
        final_model,
        X_test,
        test_timestamps,
        WINDOW_SIZE,
        STRIDE
    )

    test_scores["is_anomaly"] = (
        test_scores["score"] < threshold
    )

    anomaly_count = int(
        test_scores["is_anomaly"].sum()
    )

    print(
        f"      Anomaly windows: {anomaly_count:,}"
    )

    regions = merge_anomaly_regions(
        test_scores,
        threshold
    )

    metrics = pointwise_metrics(
        test_scores,
        threshold,
        FAILURE_WINDOWS
    )

    failures = event_detection(
        test_scores,
        threshold,
        FAILURE_WINDOWS
    )

    # Save results

    test_scores.to_csv(
        OUTPUT_DIR / "test_window_scores.csv",
        index=False
    )

    failures.to_csv(
        OUTPUT_DIR / "failure_detection.csv",
        index=False
    )

    pd.DataFrame(selection).to_csv(
        OUTPUT_DIR / "model_selection.csv",
        index=False
    )

    joblib.dump(
        {
            "model": final_model,
            "scaler": scaler,
            "features": FEATURES,
            "medians": medians.to_dict(),
            "threshold": threshold,
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
        },
        OUTPUT_DIR / "hmm_sentinel_model.joblib"
    )

    summary = {
        "selected_states": int(
            final_model.n_components
        ),
        "threshold": threshold,
        "anomaly_windows": anomaly_count,
        "anomaly_regions": len(regions),
        **metrics,
    }

    with open(
        OUTPUT_DIR / "summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    print("\n========================================")
    print("             PIPELINE COMPLETE")
    print("========================================")
    print(
        f"Hidden states : {summary['selected_states']}"
    )
    print(
        f"Threshold     : {summary['threshold']:.6f}"
    )
    print(
        f"Anomalies     : {summary['anomaly_windows']}"
    )
    print(
        f"Precision     : {summary['precision']:.4f}"
    )
    print(
        f"Recall        : {summary['recall']:.4f}"
    )
    print(
        f"F1 score      : {summary['f1']:.4f}"
    )
    print("\nResults saved to:")
    print(f"      {OUTPUT_DIR}")

    return (
        summary,
        pd.DataFrame(selection),
        test_scores,
        failures,
    )


if __name__ == "__main__":
    run()