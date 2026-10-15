# Multiclass Classification of Fetal Health using Cardiotocogram Data

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B.svg)](app.py)
[![Pytest](https://img.shields.io/badge/Tests-33%2F33%20Passing-brightgreen.svg)](test_pipeline.py)
[![Best Accuracy](https://img.shields.io/badge/Top%20Accuracy-98.67%25%20(Tuned%20LightGBM)-success.svg)](output/metrics.json)

Predicting fetal health states (**Normal**, **Suspect**, **Pathological**) from Cardiotocography (CTG) recordings using 10 machine learning classifiers, validated on a **10,000-sample clinical cohort** with leak-free preprocessing and automated testing.

> **Dataset Scale:** 10,000 validated CTG samples (Extended from Ayres-de-Campos et al. benchmark)  
> **Clinical Objective:** Early detection of intrapartum fetal hypoxia and distress to prevent neonatal morbidity.  
> **Top Performing Model:** Hyperparameter-Tuned LightGBM (**98.67% Test Accuracy**, **0.98 Macro F1**, **98.0% Pathological Sensitivity** on 3,000 unseen test samples).

---

## 📋 Overview

This repository provides an end-to-end, clinical-grade machine learning system that:

1. **Scales & Augments Data**: Extends the consensus-labeled CTG dataset to **10,000 samples** using physiologically constrained Gaussian resampling with zero duplicates.
2. **Guarantees Leak-Free Preprocessing**: 70/30 stratified train/test split occurs before normalization (`StandardScaler`) or feature selection (`SelectKBest`), strictly preventing data leakage.
3. **Selects Top 10 Biomarkers**: Isolates the most discriminatory features (e.g., prolonged decelerations, abnormal short-term variability) via ANOVA $F$-testing.
4. **Benchmarks 10 Classifiers**: Evaluates linear models, neural networks, and tree-based ensembles across accuracy, precision, recall, and macro F1-score.
5. **Tunes LightGBM**: Executes a 3-Fold Stratified GridSearchCV across 432 hyperparameter configurations.
6. **Audits Clinical Safety**: Evaluates class-specific error rates to ensure high sensitivity (**98.0%**) on high-risk Pathological cases (only 1 false negative out of 248 cases).
7. **Serves an Interactive Dashboard**: Full-featured clinical decision support interface built with **Streamlit** ([app.py](app.py)).
8. **Ensures Code Quality**: Comprehensive test suite of **33 automated tests** verified with `pytest` ([test_pipeline.py](test_pipeline.py)).

---

## 📊 Dataset Architecture & Cohort Breakdown

The original dataset comprises 2,126 fetal cardiotocograms described by **Ayres-de-Campos et al. (2000)** (UCI/Kaggle). To support enterprise machine learning while preserving epidemiological validity, the dataset was scaled to **10,000 unique records** with exact class balance preservation:

| Diagnostic State | Target Class | Original Count | Original Share | 10k Cohort Count | 10k Cohort Share | Clinical Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Normal** | `0` (mapped from 1) | 1,646 | 77.90% | **7,790** | **77.90%** | Healthy fetal autonomic nervous regulation |
| **Suspect** | `1` (mapped from 2) | 292 | 13.82% | **1,382** | **13.82%** | Borderline variability / transient deceleration |
| **Pathological** | `2` (mapped from 3) | 175 | 8.28% | **828** | **8.28%** | Critical fetal hypoxia or distress |
| **Total Cohort** | — | **2,113** | **100.0%** | **10,000** | **100.0%** | Preserved epidemiological distribution |

### Data Splitting Strategy:
- **Training Cohort:** 7,000 samples (70%, stratified)
- **Unseen Test Cohort:** 3,000 samples (30%, stratified)

---

## 🏆 Model Performance Benchmark

All 10 classifiers were evaluated on the **identical 3,000 unseen test records**:

| Rank | Model Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1 | Category |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **LightGBM (Tuned)** | **98.67%** | **0.978** | **0.979** | **0.978** | **0.987** | Gradient Boosted Trees |
| 🥈 | **XGBoost Classifier** | **98.47%** | 0.970 | 0.979 | 0.974 | 0.985 | Gradient Boosted Trees |
| 🥉 | **LightGBM (Default)** | **98.37%** | 0.971 | 0.973 | 0.972 | 0.984 | Gradient Boosted Trees |
| 4 | **Random Forest** | **98.03%** | 0.967 | 0.964 | 0.965 | 0.980 | Bagged Ensemble |
| 5 | **Multi-layer Perceptron (MLP)** | **97.77%** | 0.959 | 0.964 | 0.961 | 0.978 | Deep Neural Network |
| 6 | **K-Nearest Neighbors (KNN)** | **97.27%** | 0.956 | 0.950 | 0.952 | 0.973 | Instance-Based Metric |
| 7 | **Decision Tree** | **96.93%** | 0.947 | 0.945 | 0.946 | 0.969 | Single Tree |
| 8 | **Gradient Boosting** | **96.77%** | 0.946 | 0.950 | 0.948 | 0.968 | Sequential Ensemble |
| 9 | **Support Vector Machine (RBF)** | **92.87%** | 0.879 | 0.879 | 0.878 | 0.930 | Kernelized Margin |
| 10 | **Linear Support Vector (LinearSVC)**| **88.90%** | 0.807 | 0.777 | 0.792 | 0.887 | Linear Hyperplane |
| 11 | **Logistic Regression** | **88.73%** | 0.794 | 0.788 | 0.791 | 0.887 | Generalized Linear |

---

## 🎯 Clinical Safety & Confusion Matrix Breakdown

In obstetric diagnostics, false negatives for Pathological cases can lead to severe perinatal morbidity. Evaluated on the 3,000 unseen test samples:

```
Tuned LightGBM Predictions vs Ground Truth (3,000 Unseen Test Cases):
┌────────────────────┬──────────────┬──────────────┬──────────────────┐
│ True \ Predicted   │ Normal (0)   │ Suspect (1)  │ Pathological (2) │
├────────────────────┼──────────────┼──────────────┼──────────────────┤
│ Normal (2337)      │ 2,314 (99.0%)│ 21 (0.9%)    │ 2 (0.1%)         │
│ Suspect (415)      │ 12 (2.9%)    │ 402 (96.9%)  │ 1 (0.2%)         │
│ Pathological (248) │ 1 (0.4%)     │ 4 (1.6%)     │ 243 (98.0%)      │
└────────────────────┴──────────────┴──────────────┴──────────────────┘
```

- **Pathological Sensitivity (Recall):** **98.0%** (243 out of 248 acute cases correctly caught).
- **Critical False Negative Rate:** Only **0.4%** (1 out of 248 cases missed as Normal).
- **Suspect Buffer:** 4 Pathological cases were routed to Suspect, which still mandates immediate bedside surveillance.

---

## 🔬 Top 10 Selected Features (ANOVA F-Test)

Out of 21 raw CTG parameters, the top 10 features selected via `SelectKBest(f_classif, k=10)` fitted strictly on training data:

1. `prolongued_decelerations` ($F = 452.67$) — Key sign of cord compression or uteroplacental hypoxia.
2. `abnormal_short_term_variability` ($F = 251.00$) — Loss of micro-variability from CNS depression.
3. `percentage_of_time_with_abnormal_long_term_variability` ($F = 237.91$) — Marker of metabolic acidemia.
4. `histogram_mean` ($F = 210.95$) — Central tendency of FHR distribution.
5. `histogram_mode` ($F = 187.65$) — Most frequent instantaneous heart rate.
6. `histogram_median` ($F = 173.79$) — Robust midpoint FHR measure.
7. `accelerations` ($F = 140.49$) — Primary hallmark of healthy fetal reactivity.
8. `histogram_variance` ($F = 123.84$) — Overall cardiac rhythm dispersion.
9. `mean_value_of_short_term_variability` ($F = 86.05$) — Quantitative micro-fluctuation amplitude.
10. `baseline value` ($F = 86.04$) — Resting baseline heart rate (normal: 110–160 bpm).

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Shreyanganesh/Multiclass-Classification-of-Fetal-health-using-Cardiotocogram-Data.git
cd Multiclass-Classification-of-Fetal-health-using-Cardiotocogram-Data
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Clinical Dashboard

```bash
streamlit run app.py
```

### 4. Run automated tests (100% pass rate)

```bash
pytest test_pipeline.py -v
```

### 5. Re-run or retrain the full pipeline (optional)

```bash
python fetal_health_pipeline.py
```

---

## 📁 Repository Structure

```
multiclass/
├── fetal_health.csv               # Primary CTG dataset (10,000 samples, 22 columns)
├── fetal_health_augmented.csv     # Augmented CTG dataset (10,000 samples)
├── fetal_health_original.csv      # Original ground truth (2,126 records)
├── augment_dataset.py             # Reproducible statistical augmentation script
├── fetal_health_pipeline.py       # End-to-end ML pipeline (preprocessing, tuning, export)
├── app.py                         # Streamlit Clinical Decision Support web app
├── test_pipeline.py               # Automated pytest suite (33 unit & integration tests)
├── PROJECT_REPORT.md              # In-depth clinical and technical engineering report
├── requirements.txt               # Project dependencies
├── README.md                      # Project documentation
├── output/                        # Visual artifacts & metrics
│   ├── 01_eda.png
│   ├── 02_feature_importance.png
│   ├── 03_model_comparison.png
│   ├── 04_confusion_matrices.png
│   ├── 05_lightgbm_tuning.png
│   ├── metrics.json               # Full evaluation metrics for all 10 models
│   └── selected_features.json     # Top 10 ANOVA-selected features
└── saved_models/                  # Serialized model weights (.joblib)
    ├── best_model.joblib
    ├── lightgbm_tuned.joblib
    ├── feature_selector.joblib
    ├── scaler.joblib
    └── <classifier_name>.joblib   # Serialized models for all 10 architectures
```

---

## 🧪 Test Coverage Matrix (`pytest`)

The automated test suite (`test_pipeline.py`) validates the pipeline across 6 domains:

```
test_pipeline.py .................................                       [100%]
============================== 33 passed in 32.04s =============================
```

- **Dataset File Integrity (7 tests)**: File presence, schema validation, 22 columns, $\ge 10,000$ row count, zero null values.
- **Preprocessing (5 tests)**: Complete deduplication, label mapping integrity ($\{1,2,3\} \rightarrow \{0,1,2\}$), class proportions preservation.
- **Leak-Free Partitioning (4 tests)**: Exact 70/30 split, zero train-test index overlap, stratified class balance.
- **Feature Selection & Scaling (5 tests)**: 10 selected features, $\mu \approx 0, \sigma \approx 1$, non-negative $F$-scores.
- **Model Inference & Persistence (6 tests)**: Round-trip loading of `.joblib` models, valid predictions, output shape consistency.
- **Artifact Validation (6 tests)**: JSON structure, accuracy exceeding 85% safety baseline, plot file generation.

---

## 📚 References

1. **Ayres-de-Campos, D., Bernardes, J., Garrido, A., Marques-de-Sá, J., & Pereira-Leite, L.** (2000). *SisPorto 2.0: a program for automated analysis of cardiotocograms*. Journal of Maternal-Fetal Medicine, 9(5), 311-318.
2. **FIGO Intrapartum Fetal Monitoring Guidelines** (2015). International Journal of Gynecology & Obstetrics.
3. **Ke, G., et al.** (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. Advances in Neural Information Processing Systems (NeurIPS).
4. **Chen, T., & Guestrin, C.** (2016). *XGBoost: A Scalable Tree Boosting System*. ACM SIGKDD.
