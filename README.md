# Maternal Health Monitoring — LSTM Risk Classification

Research implementation of an LSTM-based classifier for maternal health risk prediction (Low / Mid / High), using the [Maternal Health Risk dataset](https://data.mendeley.com/datasets/ckhgtgdg4f/1) (Hossain, 2025, DOI: 10.17632/ckhgtgdg4f.1). The source dataset contains 6,103 IoT-collected records from healthcare centres in Bangladesh.

## Model

- Eight clinical features (Age, Body Temperature, Heart Rate, Systolic BP, Diastolic BP, BMI, HbA1c, Fasting Glucose) serialised into an 8-step feature sequence: `(N, 8, 1)`. These are **static feature positions, not longitudinal time steps**.
- Single-layer LSTM (hidden size 50) → Dropout (0.2) → Linear classifier (3 classes).
- Inverse-frequency class weighting, Adam (lr 1e-3, weight decay 1e-4), early stopping (patience 15, max 180 epochs).
- Stratified 70:15:15 train/validation/test split; MinMaxScaler fit only on the training set.
- Seed 42; CPU execution.

## Run locally

Requires Python 3.10+ and an internet connection for the initial dataset download.

```bash
pip install -r requirements.txt
python maternal_risk_lstm.py
```

The program downloads the source dataset on first run and caches it as `maternal_health_mendeley.csv`. This file is excluded from Git because the source CSV includes patient names and identifiers. You may also supply your own authorised copy of the source CSV using that filename.

## Data validation

With the supplied 6,103-row CSV, the preprocessing function was checked to produce **4,272 training, 915 validation and 916 test** records; the model forward pass produces three class logits. This check is **not** a full training/evaluation run.

## Reference results reported with the supplied code

| Metric | Value |
|---|---|
| Accuracy | 76.7% |
| Macro Precision | 0.764 |
| Macro Recall | 0.769 |
| Macro F1 | 0.764 |
| Macro ROC-AUC (OvR) | 0.909 |
| Brier Score | 0.321 |

Per-class F1: Low risk 0.882 · Mid risk 0.659 · High risk 0.753. These metrics are **provided reference results** and have not been independently reproduced as part of this repository deployment.

## Important limitation

This project is for **research and educational use only**, not clinical diagnosis, patient triage or automated treatment decisions. An LSTM over reordered static features should not be interpreted as having learned temporal trends.

## Data citation

Hossain, M. M. (2025). *Maternal health risk data* (Version 1). Mendeley Data. https://doi.org/10.17632/ckhgtgdg4f.1
