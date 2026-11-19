# HMM Sentinel — Industrial Sensor Anomaly Detector

A complete Hidden Markov Model anomaly-detection project using the real MetroPT-3 Air Compressor dataset from the UCI Machine Learning Repository.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Download real data

```powershell
python scripts/download_data.py
```

The raw dataset is intentionally not bundled because it is large.

## Run

```powershell
python scripts/run_pipeline.py
streamlit run app.py
```

## Method

- Gaussian-emission HMM
- Candidate hidden states: 2–6
- Chronological validation
- Normalized sequence log-likelihood
- Threshold = 1st percentile of normal training-window scores
- Sliding-window monitoring
- Documented MetroPT-3 failure windows for evaluation

## Dataset

MetroPT-3 Air Compressor dataset, UCI Machine Learning Repository. DOI: 10.24432/C5VW3R. License: CC BY 4.0.

The project uses the first chronological month as the normal baseline and later observations as the monitoring period.

## Real-world application

The detector is designed as a predictive-maintenance analogue: unusual temporal sensor behavior can trigger inspection before a machine failure becomes operationally costly.

## Project structure

```text
HMM_Sentinel_Anomaly_Detector/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── src/
├── scripts/
├── tests/
├── data/
└── outputs/
```
