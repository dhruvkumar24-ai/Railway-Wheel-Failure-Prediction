# 🚂 Predictive Maintenance for Railway Wheel Failures
### LightGBM · XGBoost · SMOTE · Sensor Data · Kaggle Competition

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Classifier-yellow?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-Benchmarked-orange?style=flat-square)
![Status](https://img.shields.io/badge/Kaggle-Rank_3_%7C_INFORMS_RAS_2025-brightgreen?style=flat-square)
![Competition](https://img.shields.io/badge/IIT_Kanpur-ML_Course_Project-red?style=flat-square)

---

## 🏆 Competition Result

> **Rank 3 — INFORMS RAS 2025 Kaggle Competition**
> Log Loss: **0.0283** | Best Iteration: 831

---

## 🧩 Problem Statement

Railway wheel failures are costly, dangerous, and difficult to predict. This project builds a **multi-class predictive maintenance classifier** on real-world railway sensor data to identify **5 failure types** before they occur:

| Class | Failure Type |
|---|---|
| 0 | High Flange |
| 1 | High Impact |
| 2 | Thin Flange |
| 3 | Other |
| 4 | Not Failed |

> *Given monthly sensor readings per wheel, predict the failure type probability for each wheel.*

---

## ⚙️ Pipeline Overview

```
Raw Sensor Data (20M+ readings)
        │
        ▼
Chunk-based Data Loading (memory-efficient)
        │
        ▼
Feature Engineering
  ├── Aggregated features (mean, std, lag across months)
  ├── Anomaly features (deviation from per-wheel baseline)
  └── Lag features (6-month rolling window)
        │
        ▼
Class Imbalance Handling → RUS + SMOTE
        │
        ▼
Model Benchmarking
  ├── Logistic Regression
  ├── Decision Tree
  ├── Random Forest
  ├── XGBoost
  ├── CatBoost
  └── LightGBM ✅ (Best)
        │
        ▼
LightGBM (Optuna-tuned) → Log Loss: 0.0283
        │
        ▼
Kaggle Submission → Rank 3
```

---

## 📊 Key Results

| Metric | Value |
|---|---|
| **Competition Rank** | **🥉 Rank 3 — INFORMS RAS 2025** |
| **Best Model** | LightGBM |
| **Validation Log Loss** | **0.0283** |
| **Best Iteration** | 831 (early stopping) |
| **Dataset Size** | 20M+ sensor readings |
| **Failure Classes** | 5 |

---

## 🔑 Key Engineering Decisions

**1. Memory-Efficient Chunked Loading**
Sensor data exceeds RAM limits — loaded in chunks using `pd.read_csv(chunksize=...)`, merged with failure labels per chunk.

**2. Per-Wheel Anomaly Features**
Baseline computed from "not failed" months per wheel — deviations flagged as anomaly signals. This captures deterioration patterns invisible to standard aggregations.

**3. Lag Features (6-month window)**
Rolling lag features capture temporal degradation trends leading up to failure.

**4. LightGBM Hyperparameters**
```python
LGBMClassifier(
    objective='multiclass', num_class=5,
    n_estimators=3000, learning_rate=0.01,
    num_leaves=64, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=5,
    reg_alpha=1.0, reg_lambda=1.0
)
```

---

## 📁 Repository Structure

```
📦 Railway-Wheel-Failure-Prediction/
├── Railway_Wheel_Failure_LightGBM.ipynb   # Full pipeline: features + model + submission
├── README.md
└── LICENSE
```

> **Note:** Raw sensor CSV files not included (competition data — proprietary). Notebook shows full pipeline with outputs.

---

## 🚀 Getting Started

```bash
pip install lightgbm xgboost catboost scikit-learn pandas numpy imbalanced-learn optuna
```

---

## 🧠 Concepts Covered

`Predictive Maintenance` · `Multi-class Classification` · `Sensor Data Engineering` · `Anomaly Detection` · `Lag Features` · `LightGBM` · `Class Imbalance (SMOTE/RUS)` · `Hyperparameter Tuning` · `Kaggle Competition`

---

## 🏅 Achievement

This solution achieved **Rank 3** in the **INFORMS RAS 2025 Problem Solving Competition** hosted on Kaggle — a railway operations research challenge open to participants globally.

---

## 👤 Author

**Dhruv Kumar Sahu**
M.Tech, Industrial & Management Engineering — IIT Kanpur
GATE 2024 AIR 33 | [LinkedIn](https://www.linkedin.com/in/dhruv-kumar-sahu-157ab9193/) · [GitHub](https://github.com/dhruvkumar24-ai)
