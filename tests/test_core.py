import pandas as pd
from src.detector import merge_anomaly_regions

def test_merge_anomaly_regions():
    df = pd.DataFrame({
        "start": pd.to_datetime(["2020-01-01 00:00","2020-01-01 00:10","2020-01-01 01:00"]),
        "end": pd.to_datetime(["2020-01-01 00:09","2020-01-01 00:19","2020-01-01 01:09"]),
        "score": [-2.0,-3.0,-1.0]
    })
    assert len(merge_anomaly_regions(df, -1.5)) == 1
