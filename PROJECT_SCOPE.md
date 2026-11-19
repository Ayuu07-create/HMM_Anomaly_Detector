# Project Scope

Primary dataset: real MetroPT-3 Air Compressor telemetry.

The HMM learns latent operating regimes from normal historical behavior. A sliding-window sequence receives a normalized log-likelihood score. Scores below a threshold learned from normal data are flagged as anomalous.

Evaluation uses documented failure periods as event-level ground truth rather than claiming every individual sensor row has a perfect binary label.
