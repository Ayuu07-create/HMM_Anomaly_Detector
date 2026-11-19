from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "data" / "raw" / "MetroPT3(AirCompressor).csv"
OUTPUT_DIR = ROOT / "outputs"

TRAIN_END = "2020-03-01"
FEATURES = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
HMM_STATES = range(2, 7)
RANDOM_STATE = 42
WINDOW_SIZE = 60
STRIDE = 10
THRESHOLD_QUANTILE = 0.01
VALIDATION_FRACTION = 0.20

FAILURE_WINDOWS = [
    ("F1", "2020-04-18 00:00:00", "2020-04-18 23:59:59"),
    ("F2", "2020-05-29 23:30:00", "2020-05-30 06:00:00"),
    ("F3", "2020-06-05 10:00:00", "2020-06-07 14:30:00"),
    ("F4", "2020-07-15 14:30:00", "2020-07-15 19:00:00"),
]
