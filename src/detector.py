import pandas as pd

def window_scores(model, X, timestamps, window_size=60, stride=10):
    records = []
    if len(X) < window_size:
        return pd.DataFrame(columns=["start", "end", "score"])
    for start in range(0, len(X)-window_size+1, stride):
        end = start + window_size
        records.append({"start": timestamps.iloc[start], "end": timestamps.iloc[end-1],
                        "score": float(model.score(X[start:end]) / window_size)})
    return pd.DataFrame(records)

def merge_anomaly_regions(scores, threshold):
    flagged = scores[scores["score"] < threshold]
    if flagged.empty:
        return []
    regions = []
    s, e = flagged.iloc[0]["start"], flagged.iloc[0]["end"]
    for _, row in flagged.iloc[1:].iterrows():
        if row["start"] <= e:
            e = max(e, row["end"])
        else:
            regions.append((s, e))
            s, e = row["start"], row["end"]
    regions.append((s, e))
    return regions
