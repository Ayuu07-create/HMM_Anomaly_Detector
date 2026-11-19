from pathlib import Path
import pandas as pd

def load_sensor_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run: python scripts/download_data.py")
    df = pd.read_csv(path)
    timestamp_col = next((c for c in df.columns if c.lower() in {"timestamp", "datetime", "date"}), None)
    if timestamp_col is None:
        raise ValueError("Could not identify a timestamp column.")
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.rename(columns={timestamp_col: "timestamp"})
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

def validate_features(df, features):
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required sensor columns: {missing}")

def chronological_split(df, train_end):
    train = df[df["timestamp"] < pd.Timestamp(train_end)].copy()
    test = df[df["timestamp"] >= pd.Timestamp(train_end)].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty train or test set.")
    return train, test
