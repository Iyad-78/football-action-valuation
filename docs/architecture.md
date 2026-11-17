# Architecture and experimental protocol

## Current vertical slice (v0.1)

```text
StatsBomb JSON files
        ↓ one match at a time
Leakage-safe shot table
        ↓ chronological match-level split
Logistic regression / LightGBM
        ↓ probability evaluation
xG artifacts + isolated 2022 final report
```

The final is tagged as `showcase` and is never used to fit or select the model.
StatsBomb's own xG value is retained as an external benchmark but explicitly
excluded from `MODEL_FEATURES`.

## Why start with xG?

xG provides a small, verifiable classification problem on which to establish:

- data provenance and validation;
- match-level splitting;
- probability metrics and calibration;
- feature-leakage safeguards;
- reproducible model artifacts.

The same infrastructure will later support action valuation. The next target is
to estimate the probability of scoring and conceding within a future event
window before and after each action.

## Planned model progression

1. Logistic-regression xG baseline.
2. LightGBM xG and action-value baseline.
3. MLP using the same tabular targets.
4. GRU over possession event sequences.
5. Compact Transformer with learned event embeddings.

Every advanced model must beat or complement a cheaper baseline on held-out
matches. Accuracy alone is insufficient: calibration, latency, memory usage,
position-specific errors and stability across competitions will also be tracked.

## Resource constraints

The ETL reads one match JSON file at a time. The prepared shot table is compact,
and model parallelism is intentionally limited. Full deep-learning experiments
may run on Colab, but preprocessing and debug runs must remain executable on a
CPU laptop with limited RAM.

