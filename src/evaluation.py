import pandas as pd

def pointwise_metrics(scores, threshold, failure_windows):
    if scores.empty:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    s = scores.copy()
    s["pred"] = s["score"] < threshold
    s["actual"] = False
    for _, start, end in failure_windows:
        mask = (s["end"] >= pd.Timestamp(start)) & (s["start"] <= pd.Timestamp(end))
        s.loc[mask, "actual"] = True
    tp = int((s.pred & s.actual).sum()); fp = int((s.pred & ~s.actual).sum())
    fn = int((~s.pred & s.actual).sum()); tn = int((~s.pred & ~s.actual).sum())
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

def event_detection(scores, threshold, failure_windows):
    out = []
    for name, start, end in failure_windows:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        hit = scores[(scores.score < threshold) & (scores["end"] >= start_ts) & (scores["start"] <= end_ts)]
        first = hit["start"].min() if not hit.empty else pd.NaT
        lead = (start_ts-first).total_seconds()/3600 if pd.notna(first) else None
        out.append({"failure": name, "start": start_ts, "end": end_ts,
                    "detected": not hit.empty, "first_alarm": first, "lead_time_hours": lead})
    return pd.DataFrame(out)
