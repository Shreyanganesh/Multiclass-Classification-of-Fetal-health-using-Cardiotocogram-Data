# Multiclass Classification of Fetal Health using Cardiotocogram Data

Predicting fetal health outcomes (Normal, Suspect, Pathological) from cardiotocography (CTG) data using 10 machine learning classifiers.

> **Reference:** Ereyi & Huo, Stanford University  
> **Dataset:** Ayres-de-Campos et al. (SisPorto 2.0) — [UCI/Kaggle CTG Dataset](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification)

---

## 📋 Overview

This project implements a complete ML pipeline that:

1. **Preprocesses** 2,126 CTG records (removes 13 duplicates → 2,113 samples)
2. **Selects features** using SelectKBest (top 10 of 21 features)
3. **Trains 10 classifiers** and compares their performance
4. **Tunes hyperparameters** for the best-performing model (LightGBM)
5. **Saves trained models** and preprocessing artifacts for deployment
6. **Serves a Streamlit web interface** for interactive predictions

### Dataset Summary

| Class | Label | Samples |
|-------|-------|---------|
| 0 | Normal | 1,646 |
| 1 | Suspect | 292 |
| 2 | Pathological | 175 |

### Models Compared

| Model | Type |
|-------|------|
| Logistic Regression | Linear |
| Random Forest | Ensemble (Bagging) |
| Decision Tree | Tree |
| K-Nearest Neighbors | Instance-based |
| Support Vector (RBF) | Kernel |
| Linear SVC | Linear |
| Multi-layer Perceptron | Neural Network |
| Gradient Boosting | Ensemble (Boosting) |
| LightGBM | Gradient Boosting |
| XGBoost | Gradient Boosting |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the ML pipeline

```bash
python fetal_health_pipeline.py
```

This will:
- Preprocess the data and validate the dataset
- Generate EDA visualizations in `output/`
- Train all 10 classifiers and print results
- Tune LightGBM hyperparameters via 3-Fold GridSearchCV
- Save trained models to `saved_models/`
- Save evaluation metrics to `output/metrics.json`
- Verify the saved model can load and predict

### 3. Launch the Streamlit web interface

```bash
streamlit run app.py
```

### 4. Run automated tests

```bash
pytest test_pipeline.py -v
```

---

## 📁 Project Structure

```
multiclass/
├── fetal_health.csv           # CTG dataset (extended to 10,000 samples, 22 columns)
├── fetal_health_original.csv  # Original CTG dataset (2,126 records)
├── fetal_health_augmented.csv # Augmented CTG dataset (10,000 samples)
├── augment_dataset.py         # Dataset extension script (statistical resampling)
├── fetal_health_pipeline.py   # Full ML pipeline (main script)
├── app.py                     # Streamlit web interface
├── test_pipeline.py           # Automated test suite (pytest)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── output/                    # Generated plots & metrics
│   ├── 01_eda.png
│   ├── 02_feature_importance.png
│   ├── 03_model_comparison.png
│   ├── 04_confusion_matrices.png
│   ├── 05_lightgbm_tuning.png
│   ├── metrics.json
│   └── selected_features.json
└── saved_models/              # Trained model artifacts
    ├── best_model.joblib
    ├── lightgbm_tuned.joblib
    ├── feature_selector.joblib
    ├── scaler.joblib
    └── <model_name>.joblib    # All 10 classifiers
```

---

## 🛡️ Data Leakage Prevention

The pipeline is carefully designed to prevent data leakage:

1. **Train/test split happens FIRST** (before any feature engineering)
2. **SelectKBest** is fitted only on training data, then `transform()` is applied to test data
3. **StandardScaler** is fitted only on training data, then `transform()` is applied to test data
4. **GridSearchCV** performs cross-validation only on training data — the test set is never used for hyperparameter selection

---

## 🧪 Test Coverage

The test suite (`test_pipeline.py`) covers:

| Category | Tests |
|----------|-------|
| Dataset Integrity | Column validation, class verification, numeric types |
| Preprocessing | Null removal, duplicate removal, label mapping |
| Data Leakage | Split isolation, scaler/selector fit verification |
| Model Training | All 10 classifiers instantiate and beat 70% baseline |
| Model Persistence | Save/load round-trip, full inference pipeline |
| Streamlit App | File existence check |

---

## 📊 Results

Top-performing classifiers (typical results):

| Model | Accuracy |
|-------|----------|
| LightGBM | ~95.6% |
| Gradient Boosting | ~95.6% |
| Random Forest | ~94.9% |
| XGBoost | ~94.9% |
| K-Nearest Neighbors | ~94.0% |

> **Note:** Hyperparameter tuning of LightGBM via GridSearchCV may decrease test accuracy due to overfitting on the training data — a finding consistent with the original paper.

---

## ⚙️ Requirements

- Python 3.9+
- See `requirements.txt` for full dependency list

---

## 📚 References

- Ayres de Campos et al. (2000) SisPorto 2.0: A Program for Automated Analysis of Cardiotocograms
- Chen & Guestrin (2016) XGBoost: A Scalable Tree Boosting System
- Ke et al. (2017) LightGBM: A Highly Efficient Gradient Boosting Decision Tree
- Scikit-learn: Pedregosa et al., JMLR 12, 2011
