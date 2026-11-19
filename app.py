import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config import OUTPUT_DIR

st.set_page_config(page_title="HMM Sentinel", page_icon="◈", layout="wide")
st.title("◈ HMM Sentinel")
st.caption("Gaussian Hidden Markov Model • Industrial Sensor Anomaly Detection")

summary_path, scores_path, failures_path = OUTPUT_DIR/"summary.json", OUTPUT_DIR/"test_window_scores.csv", OUTPUT_DIR/"failure_detection.csv"
if not summary_path.exists() or not scores_path.exists():
    st.warning("No results found.")
    st.code("python scripts/download_data.py\npython scripts/run_pipeline.py")
    st.stop()

summary = json.loads(summary_path.read_text())
scores = pd.read_csv(scores_path, parse_dates=["start", "end"])
failures = pd.read_csv(failures_path, parse_dates=["start", "end", "first_alarm"])

a,b,c,d,e = st.columns(5)
a.metric("HMM states", summary["selected_states"])
b.metric("Threshold", f'{summary["threshold"]:.4f}')
c.metric("Anomaly windows", summary["anomaly_windows"])
d.metric("Precision", f'{summary["precision"]:.2%}')
e.metric("Recall", f'{summary["recall"]:.2%}')

st.subheader("Likelihood timeline")
fig = go.Figure()
fig.add_trace(go.Scatter(x=scores.start, y=scores.score, mode="lines", name="Normalized log-likelihood"))
fig.add_hline(y=summary["threshold"], line_dash="dash", annotation_text="Threshold")
an = scores[scores.is_anomaly]
fig.add_trace(go.Scatter(x=an.start, y=an.score, mode="markers", name="Anomaly"))
fig.update_layout(height=430, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Detected anomaly windows")
st.dataframe(scores[scores.is_anomaly], use_container_width=True)

st.subheader("Documented failure-event evaluation")
st.dataframe(failures, use_container_width=True)

st.download_button("Download anomaly scores CSV", scores.to_csv(index=False),
                   "test_window_scores.csv", "text/csv")
