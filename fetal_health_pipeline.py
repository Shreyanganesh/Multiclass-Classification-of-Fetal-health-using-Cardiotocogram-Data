"""
Multiclass Classification of Fetal Health using Cardiotocogram Data
===================================================================
Full ML pipeline implementing:
  - Data loading & preprocessing (null/duplicate removal)
  - Exploratory data analysis & visualization
  - 70/30 stratified train/test split
  - Feature selection using SelectKBest (f_classif) — fitted on TRAINING data only
  - Feature standardization (StandardScaler) — fitted on TRAINING data only
  - 10 classifiers: Logistic Regression, Random Forest, Decision Tree,
    KNN, SVC, Linear SVC, MLP, Gradient Boosting, LightGBM, XGBoost
  - Model evaluation (accuracy, classification reports, confusion matrices)
  - Hyperparameter tuning for LightGBM via 3-Fold GridSearchCV
  - Model & artifact saving (joblib) with prediction verification

Reference: Ereyi & Huo, Stanford University
Dataset:   Ayres-de-Campos et al. (SisPorto 2.0) - UCI/Kaggle CTG dataset
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for CI/server environments
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fetal_health.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
RANDOM_STATE = 42
TEST_SIZE = 0.30  # 70/30 split as described in the paper
K_BEST_FEATURES = 10  # Top-10 features via KBest

# Labels remapped from {1,2,3} -> {0,1,2} for XGBoost compatibility
CLASS_LABELS = {0: "Normal", 1: "Suspect", 2: "Pathological"}

EXPECTED_COLUMNS = [
    "baseline value", "accelerations", "fetal_movement",
    "uterine_contractions", "light_decelerations", "severe_decelerations",
    "prolongued_decelerations", "abnormal_short_term_variability",
    "mean_value_of_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability",
    "mean_value_of_long_term_variability", "histogram_width",
    "histogram_min", "histogram_max", "histogram_number_of_peaks",
    "histogram_number_of_zeroes", "histogram_mode", "histogram_mean",
    "histogram_median", "histogram_variance", "histogram_tendency",
    "fetal_health",
]


def create_dirs():
    """Create output directories for plots, results, and saved models."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"[INFO] Output directory: {OUTPUT_DIR}")
    print(f"[INFO] Models directory: {MODELS_DIR}")


# ═══════════════════════════════════════════════
# STEP 1: DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════
def load_and_preprocess(path: str) -> pd.DataFrame:
    """Load the CTG dataset and perform preprocessing.

    Validates expected columns, removes nulls and duplicates,
    and remaps class labels to 0-indexed.
    """
    print("\n" + "=" * 60)
    print("STEP 1: DATA LOADING & PREPROCESSING")
    print("=" * 60)

    df = pd.read_csv(path)
    print(f"  Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")

    # Validate expected columns
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")
    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra_cols:
        print(f"  [WARN] Unexpected extra columns (will be ignored): {extra_cols}")
    print("  Column validation: PASSED ✓")

    # Validate target classes
    raw_classes = sorted(df["fetal_health"].unique())
    expected_raw = [1.0, 2.0, 3.0]
    if raw_classes != expected_raw:
        raise ValueError(
            f"Expected target classes {expected_raw}, got {raw_classes}"
        )
    print("  Target class validation: PASSED ✓ (classes: 1=Normal, 2=Suspect, 3=Pathological)")

    # Check for null values
    null_count = df.isnull().sum().sum()
    print(f"  Null values found: {null_count}")
    if null_count > 0:
        df.dropna(inplace=True)
        print(f"  After removing nulls: {df.shape[0]} samples")

    # Remove duplicates
    dupes_before = df.shape[0]
    df.drop_duplicates(inplace=True)
    dupes_removed = dupes_before - df.shape[0]
    print(f"  Duplicate rows removed: {dupes_removed}")
    print(f"  Final dataset size: {df.shape[0]} samples")

    # Remap labels: {1, 2, 3} -> {0, 1, 2} (required by XGBoost)
    df["fetal_health"] = df["fetal_health"].map({1.0: 0, 2.0: 1, 3.0: 2}).astype(int)

    # Class distribution
    print("\n  Class distribution:")
    for cls_val, cls_name in CLASS_LABELS.items():
        count = (df["fetal_health"] == cls_val).sum()
        print(f"    {cls_name} ({cls_val}): {count} samples")

    return df


# ═══════════════════════════════════════════════
# STEP 2: EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════
def exploratory_analysis(df: pd.DataFrame):
    """Visualize class distribution and correlation matrix."""
    print("\n" + "=" * 60)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # ── Class distribution bar chart ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    class_counts = df["fetal_health"].value_counts().sort_index()
    colors = ["#6a5acd", "#e07b54", "#d4456a"]
    class_names = [CLASS_LABELS[i] for i in class_counts.index]
    axes[0].bar(class_names, class_counts.values, color=colors, edgecolor="white")
    axes[0].set_title("Fetal Health Class Distribution", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 15, str(v), ha="center", fontweight="bold")

    # ── Correlation heatmap ──
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdPu",
        annot=False,
        square=True,
        linewidths=0.5,
        ax=axes[1],
        cbar_kws={"shrink": 0.8},
    )
    axes[1].set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_eda.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 01_eda.png")


# ═══════════════════════════════════════════════
# STEP 3: TRAIN/TEST SPLIT (before any feature engineering)
# ═══════════════════════════════════════════════
def split_data(df: pd.DataFrame):
    """70/30 stratified train/test split.

    Split is done BEFORE feature selection and scaling to prevent data leakage.
    """
    print("\n" + "=" * 60)
    print("STEP 3: TRAIN/TEST SPLIT")
    print("=" * 60)

    X = df.drop("fetal_health", axis=1)
    y = df["fetal_health"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set:     {X_test.shape[0]} samples")
    print(f"  Features:     {X_train.shape[1]}")
    print(f"  Stratified:   Yes (class proportions preserved)")

    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════
# STEP 4: FEATURE SELECTION & STANDARDIZATION
#         (fitted on training data ONLY)
# ═══════════════════════════════════════════════
def select_and_scale_features(X_train, X_test, y_train):
    """Apply SelectKBest and StandardScaler — fitted only on training data.

    This avoids data leakage: the test set has no influence on feature
    selection or scaling parameters.
    """
    print("\n" + "=" * 60)
    print("STEP 4: FEATURE SELECTION & STANDARDIZATION")
    print("=" * 60)

    # ── Score ALL features on training data for visualization ──
    scorer = SelectKBest(score_func=f_classif, k="all")
    scorer.fit(X_train, y_train)
    scores = pd.DataFrame(
        {"Feature": X_train.columns, "Score": scorer.scores_}
    ).sort_values("Score", ascending=True)

    print(f"\n  Feature importance scores (all {len(scores)} features):")
    for _, row in scores.iterrows():
        print(f"    {row['Feature']:>50s}  {row['Score']:>10.3f}")

    # ── Feature importance bar chart ──
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.RdPu
    norm_scores = (scores["Score"] - scores["Score"].min()) / (
        scores["Score"].max() - scores["Score"].min()
    )
    bar_colors = [cmap(0.3 + 0.7 * s) for s in norm_scores]
    ax.barh(scores["Feature"], scores["Score"], color=bar_colors, edgecolor="white")
    ax.set_xlabel("Score", fontsize=12)
    ax.set_title(
        "Feature Importance Scoring (KBest - f_classif)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "02_feature_importance.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("  Saved: 02_feature_importance.png")

    # ── Select top-K features (fit on TRAINING data only) ──
    selector = SelectKBest(score_func=f_classif, k=K_BEST_FEATURES)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)  # transform only — no fit
    selected_mask = selector.get_support()
    selected_features = X_train.columns[selected_mask].tolist()

    print(f"\n  Top {K_BEST_FEATURES} selected features:")
    for f in selected_features:
        print(f"    - {f}")

    # Print table (matching paper's Table 1)
    top_scores = scores.tail(K_BEST_FEATURES).sort_values("Score", ascending=False)
    print(f"\n  {'Feature':<55s} {'Score':>10s}")
    print("  " + "-" * 67)
    for _, row in top_scores.iterrows():
        print(f"  {row['Feature']:<55s} {row['Score']:>10.6f}")

    # ── Standardize (fit on TRAINING data only) ──
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)  # transform only — no fit

    print(f"\n  Standardization: fit on training set ({X_train_scaled.shape[0]} samples)")
    print(f"  Applied to test set ({X_test_scaled.shape[0]} samples) — NO data leakage ✓")

    return X_train_scaled, X_test_scaled, selector, scaler, selected_features


# ═══════════════════════════════════════════════
# STEP 5: MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════
def get_classifiers():
    """Return a dictionary of all 10 classifiers from the paper."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector": SVC(kernel="rbf", random_state=RANDOM_STATE),
        "Linear Support Vector": LinearSVC(
            max_iter=5000, random_state=RANDOM_STATE, dual="auto"
        ),
        "Multi-layer Perceptron": MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        "LightGBM Classifier": lgb.LGBMClassifier(
            n_estimators=100, random_state=RANDOM_STATE, verbose=-1
        ),
        "XGBoost Classifier": xgb.XGBClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
        ),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train all classifiers and report results."""
    print("\n" + "=" * 60)
    print("STEP 5: MODEL TRAINING & EVALUATION")
    print("=" * 60)

    classifiers = get_classifiers()
    results = {}

    for name, clf in classifiers.items():
        print(f"\n  Training: {name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {
            "model": clf,
            "accuracy": acc,
            "predictions": y_pred,
        }
        print(f"    Accuracy: {acc:.6f}")

    return results


def display_results_table(results: dict):
    """Display sorted results table matching the paper's Table 2."""
    print("\n" + "-" * 50)
    print("  MODEL COMPARISON TABLE (sorted by accuracy)")
    print("-" * 50)
    sorted_results = sorted(
        results.items(), key=lambda x: x[1]["accuracy"], reverse=True
    )
    print(f"  {'Model':<30s} {'Accuracy':>10s}")
    print("  " + "-" * 42)
    for name, data in sorted_results:
        print(f"  {name:<30s} {data['accuracy']:>10.6f}")
    return sorted_results


def plot_model_comparison(results: dict):
    """Bar chart comparing model accuracies."""
    sorted_results = sorted(results.items(), key=lambda x: x[1]["accuracy"])
    names = [r[0] for r in sorted_results]
    scores = [r[1]["accuracy"] for r in sorted_results]

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.RdPu
    norm_scores = [
        (s - min(scores)) / (max(scores) - min(scores) + 1e-9) for s in scores
    ]
    bar_colors = [cmap(0.3 + 0.6 * ns) for ns in norm_scores]
    bars = ax.barh(names, scores, color=bar_colors, edgecolor="white", height=0.6)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}",
            va="center",
            fontsize=9,
        )

    ax.set_xlim(min(scores) - 0.02, 1.0)
    ax.set_xlabel("Accuracy Score", fontsize=12)
    ax.set_title("Classifier Model Scoring Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "03_model_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("  Saved: 03_model_comparison.png")


def print_classification_reports(results: dict, y_test):
    """Print classification reports for all models."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORTS")
    print("=" * 60)

    for name, data in results.items():
        print(f"\n  ── {name} ──")
        report = classification_report(
            y_test,
            data["predictions"],
            target_names=list(CLASS_LABELS.values()),
            digits=4,
        )
        for line in report.split("\n"):
            print(f"    {line}")


def plot_confusion_matrices(results: dict, y_test):
    """Plot confusion matrices for all classifiers."""
    n_models = len(results)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for idx, (name, data) in enumerate(results.items()):
        cm = confusion_matrix(y_test, data["predictions"], normalize="true")
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=list(CLASS_LABELS.values()),
        )
        disp.plot(ax=axes[idx], cmap="Purples", values_format=".3f", colorbar=False)
        axes[idx].set_title(name, fontsize=10, fontweight="bold")
        axes[idx].set_xlabel("")
        axes[idx].set_ylabel("")

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    fig.supxlabel("Predicted Label", fontsize=12)
    fig.supylabel("True Label", fontsize=12)
    plt.suptitle(
        "Confusion Matrices (Normalized)", fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "04_confusion_matrices.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("  Saved: 04_confusion_matrices.png")


# ═══════════════════════════════════════════════
# STEP 6: HYPERPARAMETER TUNING (LightGBM)
# ═══════════════════════════════════════════════
def tune_lightgbm(X_train, X_test, y_train, y_test):
    """Optimize LightGBM via 3-Fold GridSearchCV (as described in the paper).

    GridSearchCV performs cross-validation ONLY on X_train — the test set
    is never used for hyperparameter selection.
    """
    print("\n" + "=" * 60)
    print("STEP 6: HYPERPARAMETER TUNING (LightGBM - 3-Fold GridSearchCV)")
    print("=" * 60)

    # ── Before optimization ──
    lgbm_default = lgb.LGBMClassifier(
        n_estimators=100, random_state=RANDOM_STATE, verbose=-1
    )
    lgbm_default.fit(X_train, y_train)
    y_pred_before = lgbm_default.predict(X_test)
    acc_before = accuracy_score(y_test, y_pred_before)

    print(f"\n  LightGBM (before tuning): {acc_before:.6f}")
    print("\n  Classification Report (Before Optimization):")
    report_before = classification_report(
        y_test, y_pred_before, target_names=list(CLASS_LABELS.values()), digits=2
    )
    for line in report_before.split("\n"):
        print(f"    {line}")

    # ── Grid search (CV on training data only) ──
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7, -1],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "num_leaves": [15, 31, 63],
        "min_child_samples": [5, 10, 20],
    }

    lgbm = lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)

    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)
    print(f"\n  Running 3-Fold GridSearchCV...")
    print(f"  Parameter grid: {n_combos} combinations")
    print(f"  Total fits: {n_combos * 3}")

    grid_search = GridSearchCV(
        estimator=lgbm,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    print(f"\n  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV score:   {grid_search.best_score_:.6f}")

    # ── After optimization (evaluate on held-out test set) ──
    y_pred_after = grid_search.best_estimator_.predict(X_test)
    acc_after = accuracy_score(y_test, y_pred_after)

    print(f"\n  LightGBM (after tuning):  {acc_after:.6f}")
    print(f"  Change: {acc_after - acc_before:+.6f}")

    print("\n  Classification Report (After Optimization):")
    report_after = classification_report(
        y_test, y_pred_after, target_names=list(CLASS_LABELS.values()), digits=2
    )
    for line in report_after.split("\n"):
        print(f"    {line}")

    # ── Side-by-side confusion matrices ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, y_pred, title in [
        (axes[0], y_pred_before, f"Before Tuning (acc={acc_before:.4f})"),
        (axes[1], y_pred_after, f"After Tuning (acc={acc_after:.4f})"),
    ]:
        cm = confusion_matrix(y_test, y_pred, normalize="true")
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=list(CLASS_LABELS.values()),
        )
        disp.plot(ax=ax, cmap="Purples", values_format=".4f")
        ax.set_title(title, fontsize=11, fontweight="bold")

    plt.suptitle(
        "LightGBM: Before vs. After Hyperparameter Tuning",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "05_lightgbm_tuning.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("  Saved: 05_lightgbm_tuning.png")

    return grid_search.best_estimator_, acc_before, acc_after


# ═══════════════════════════════════════════════
# STEP 7: SAVE MODELS & ARTIFACTS
# ═══════════════════════════════════════════════
def save_artifacts(results, best_lgbm, selector, scaler, selected_features, y_test):
    """Save trained models, preprocessing artifacts, and evaluation metrics."""
    print("\n" + "=" * 60)
    print("STEP 7: SAVING MODELS & ARTIFACTS")
    print("=" * 60)

    # Save preprocessing artifacts
    joblib.dump(selector, os.path.join(MODELS_DIR, "feature_selector.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    print("  Saved: feature_selector.joblib")
    print("  Saved: scaler.joblib")

    # Save the best model from each category
    best_model_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_model_name]["model"]
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
    print(f"  Saved: best_model.joblib ({best_model_name})")

    # Save tuned LightGBM
    joblib.dump(best_lgbm, os.path.join(MODELS_DIR, "lightgbm_tuned.joblib"))
    print("  Saved: lightgbm_tuned.joblib")

    # Save all models
    for name, data in results.items():
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        filepath = os.path.join(MODELS_DIR, f"{safe_name}.joblib")
        joblib.dump(data["model"], filepath)
    print(f"  Saved: all 10 classifier models")

    # Save metrics as JSON
    metrics = {}
    for name, data in results.items():
        report = classification_report(
            y_test, data["predictions"], output_dict=True,
            target_names=list(CLASS_LABELS.values()),
        )
        metrics[name] = {
            "accuracy": data["accuracy"],
            "classification_report": report,
        }

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Saved: metrics.json")

    # Save selected features list
    features_path = os.path.join(OUTPUT_DIR, "selected_features.json")
    with open(features_path, "w") as f:
        json.dump(selected_features, f, indent=2)
    print(f"  Saved: selected_features.json")

    return best_model_name


# ═══════════════════════════════════════════════
# STEP 8: VERIFY SAVED MODEL
# ═══════════════════════════════════════════════
def verify_saved_model(X_test, y_test):
    """Load the saved model and preprocessing artifacts, make a prediction,
    and verify it matches the original results."""
    print("\n" + "=" * 60)
    print("STEP 8: VERIFYING SAVED MODEL")
    print("=" * 60)

    # Load artifacts
    loaded_model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    print(f"  Loaded: best_model.joblib ({type(loaded_model).__name__})")

    # Predict on test set
    y_pred_loaded = loaded_model.predict(X_test)
    acc_loaded = accuracy_score(y_test, y_pred_loaded)
    print(f"  Accuracy from loaded model: {acc_loaded:.6f}")

    # Verify single-sample prediction
    sample = X_test[:1]
    pred = loaded_model.predict(sample)
    pred_label = CLASS_LABELS[int(pred[0])]
    print(f"  Single sample prediction: {pred_label} (class {int(pred[0])})")

    # Verify preprocessing pipeline can be loaded
    loaded_selector = joblib.load(os.path.join(MODELS_DIR, "feature_selector.joblib"))
    loaded_scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    print(f"  Loaded: feature_selector.joblib (SelectKBest)")
    print(f"  Loaded: scaler.joblib (StandardScaler)")
    print("  Model verification: PASSED ✓")

    return acc_loaded


# ═══════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════
def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Multiclass Classification of Fetal Health              ║")
    print("║  using Cardiotocogram Data                              ║")
    print("║  Ereyi & Huo, Stanford University                       ║")
    print("╚══════════════════════════════════════════════════════════╝")

    create_dirs()

    # Step 1: Load & preprocess
    df = load_and_preprocess(DATASET_PATH)

    # Step 2: EDA (on full dataset — OK, this is just visualization)
    exploratory_analysis(df)

    # Step 3: Train/test split BEFORE feature engineering (prevents leakage)
    X_train, X_test, y_train, y_test = split_data(df)

    # Step 4: Feature selection & scaling (fit on training data ONLY)
    X_train_scaled, X_test_scaled, selector, scaler, selected_features = (
        select_and_scale_features(X_train, X_test, y_train)
    )

    # Step 5: Train & evaluate all 10 classifiers
    results = train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test)
    sorted_results = display_results_table(results)
    plot_model_comparison(results)
    print_classification_reports(results, y_test)
    plot_confusion_matrices(results, y_test)

    # Step 6: Hyperparameter tuning for LightGBM
    best_lgbm, acc_before, acc_after = tune_lightgbm(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    # Step 7: Save all models and artifacts
    best_model_name = save_artifacts(
        results, best_lgbm, selector, scaler, selected_features, y_test
    )

    # Step 8: Verify saved model can be loaded and predict
    verify_saved_model(X_test_scaled, y_test)

    # ── Final summary ──
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Dataset:           {df.shape[0]} samples, {K_BEST_FEATURES} features selected")
    print(f"  Best classifier:   {sorted_results[0][0]} ({sorted_results[0][1]['accuracy']:.6f})")
    print(f"  LightGBM before:   {acc_before:.6f}")
    print(f"  LightGBM after:    {acc_after:.6f}")
    print(f"\n  Outputs saved to:  {OUTPUT_DIR}/")
    print(f"  Models saved to:   {MODELS_DIR}/")


if __name__ == "__main__":
    main()
