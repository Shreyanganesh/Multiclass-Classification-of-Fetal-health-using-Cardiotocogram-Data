"""
Streamlit Web Interface for Fetal Health Classification
========================================================
Interactive dashboard for:
  - Exploring the CTG dataset
  - Visualizing feature importance and model comparisons
  - Running predictions with trained models
  - Comparing classifier performance
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from fetal_health_pipeline import (
    CLASS_LABELS,
    EXPECTED_COLUMNS,
    K_BEST_FEATURES,
    load_and_preprocess,
    split_data,
    select_and_scale_features,
    get_classifiers,
)

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
DATASET_PATH = os.path.join(PROJECT_DIR, "fetal_health.csv")
MODELS_DIR = os.path.join(PROJECT_DIR, "saved_models")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")


# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fetal Health Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# Custom CSS for premium look
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .main-header p {
        font-size: 1.05rem;
        opacity: 0.9;
        font-weight: 300;
    }

    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(0,0,0,0.05);
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
    }

    .metric-card h3 {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }

    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
    }

    .prediction-normal {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        color: #166534;
    }

    .prediction-suspect {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        color: #9a3412;
    }

    .prediction-pathological {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        color: #991b1b;
    }

    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .badge-success {
        background: #dcfce7;
        color: #166534;
    }

    .badge-warning {
        background: #fef3c7;
        color: #92400e;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🩺 Fetal Health Classification</h1>
    <p>Multiclass classification using Cardiotocogram (CTG) data — 10 ML models compared</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Dashboard", "🔍 Data Explorer", "🤖 Model Comparison", "🎯 Predict", "📄 About"],
)


# ──────────────────────────────────────────────
# Data loading (cached)
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load and preprocess the dataset."""
    df = load_and_preprocess(DATASET_PATH)
    return df


@st.cache_data
def get_pipeline_data():
    """Run the full pipeline and cache results."""
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_s, X_test_s, selector, scaler, features = select_and_scale_features(
        X_train, X_test, y_train
    )
    return df, X_train_s, X_test_s, y_train, y_test, selector, scaler, features, X_train, X_test


@st.cache_resource
def load_saved_models():
    """Load all saved models if they exist."""
    models = {}
    if os.path.exists(MODELS_DIR):
        for fname in os.listdir(MODELS_DIR):
            if fname.endswith(".joblib") and fname not in (
                "feature_selector.joblib", "scaler.joblib"
            ):
                name = fname.replace(".joblib", "").replace("_", " ").title()
                models[name] = joblib.load(os.path.join(MODELS_DIR, fname))
    return models


@st.cache_data
def load_metrics():
    """Load saved metrics if they exist."""
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return None


def extract_prediction_probabilities(model, sample_scaled, class_labels=CLASS_LABELS):
    """
    Safely extract and map probabilities from a model using model.classes_ and class_labels.
    Does not assume class order.

    Returns:
        prob_dict: dict mapping each class name ('Normal', 'Suspect', 'Pathological')
                   to float probability in [0.0, 1.0], or None if unsupported.
        err_msg: explanation if probabilities are unavailable, or None if successful.
    """
    if not hasattr(model, "predict_proba"):
        return None, "This model does not support probability estimation (`predict_proba` is not available)."

    try:
        raw_probs = model.predict_proba(sample_scaled)[0]
    except Exception as e:
        return None, f"Probability estimation failed: {e}"

    classes = getattr(model, "classes_", None)

    # Initialize all target classes to 0.0
    prob_dict = {label_name: 0.0 for label_name in class_labels.values()}

    if classes is not None and len(classes) == len(raw_probs):
        for idx, class_val in enumerate(classes):
            try:
                class_key = int(class_val)
            except (ValueError, TypeError):
                class_key = class_val
            label_name = class_labels.get(class_key)
            if label_name:
                prob_dict[label_name] = float(raw_probs[idx])
    else:
        for idx, (class_key, label_name) in enumerate(class_labels.items()):
            if idx < len(raw_probs):
                prob_dict[label_name] = float(raw_probs[idx])

    # Ensure probabilities sum to approximately 100% (1.0)
    total = sum(prob_dict.values())
    if total > 0 and abs(total - 1.0) > 1e-4:
        prob_dict = {k: v / total for k, v in prob_dict.items()}

    return prob_dict, None


def plot_prediction_probabilities(prob_dict):
    """
    Render a matplotlib bar chart with:
      - Exactly three labelled bars: Normal, Suspect, Pathological
      - Y-axis formatted as probability from 0% to 100% (no negative values)
      - Percentage displayed above each bar
      - Probabilities sum to approximately 100%
    """
    labels = ["Normal", "Suspect", "Pathological"]
    percentages = [prob_dict.get(lbl, 0.0) * 100.0 for lbl in labels]

    colors = ["#22c55e", "#f59e0b", "#ef4444"]  # Green (Normal), Amber (Suspect), Red (Pathological)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(
        labels,
        percentages,
        color=colors,
        width=0.52,
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
    )

    # Strictly non-negative y-axis from 0% to 100% (padded to 108% for labels, absolutely no negative values)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(100))
    ax.set_ylabel("Probability (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Prediction Confidence by Class", fontsize=13, fontweight="bold", pad=12)

    # Styling
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=10)

    # Percentage above each bar
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 2.0,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#1e293b",
        )

    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────
# PAGE: Dashboard
# ──────────────────────────────────────────────
if page == "📊 Dashboard":
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Samples</h3>
            <div class="value">{len(df):,}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Features</h3>
            <div class="value">{df.shape[1] - 1}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Classes</h3>
            <div class="value">3</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        normal_pct = (df["fetal_health"] == 0).mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>Normal %</h3>
            <div class="value">{normal_pct:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df["fetal_health"].value_counts().sort_index()
        colors = ["#667eea", "#f093fb", "#f5576c"]
        labels = [CLASS_LABELS[i] for i in counts.index]
        bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                    str(val), ha="center", fontweight="bold", fontsize=11)
        ax.set_ylabel("Count")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(7, 5))
        corr = df.drop("fetal_health", axis=1).corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap="RdPu", annot=False,
                    square=True, linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.7})
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=7)
        st.pyplot(fig)
        plt.close()

    # Feature importance
    st.subheader("Feature Importance (KBest Scores)")
    try:
        _, _, _, y_train, _, selector, _, features, X_train_raw, _ = get_pipeline_data()
        from sklearn.feature_selection import SelectKBest, f_classif
        scorer = SelectKBest(score_func=f_classif, k="all")
        scorer.fit(X_train_raw, y_train)
        scores_df = pd.DataFrame({
            "Feature": X_train_raw.columns,
            "Score": scorer.scores_
        }).sort_values("Score", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.cm.RdPu
        norm = (scores_df["Score"] - scores_df["Score"].min()) / (
            scores_df["Score"].max() - scores_df["Score"].min()
        )
        bar_colors = [cmap(0.3 + 0.7 * s) for s in norm]
        ax.barh(scores_df["Feature"], scores_df["Score"], color=bar_colors, edgecolor="white")
        ax.set_xlabel("Score")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"Could not compute feature importance: {e}")

    # Model results if available
    metrics = load_metrics()
    if metrics:
        st.subheader("Model Accuracy Summary")
        acc_data = {name: m["accuracy"] for name, m in metrics.items()}
        acc_df = pd.DataFrame(acc_data.items(), columns=["Model", "Accuracy"])
        acc_df = acc_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
        acc_df.index += 1
        st.dataframe(acc_df, use_container_width=True)


# ──────────────────────────────────────────────
# PAGE: Data Explorer
# ──────────────────────────────────────────────
elif page == "🔍 Data Explorer":
    st.subheader("Dataset Explorer")

    try:
        df = load_data()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        selected_class = st.multiselect(
            "Filter by class",
            options=list(CLASS_LABELS.values()),
            default=list(CLASS_LABELS.values()),
        )
    with col2:
        n_rows = st.slider("Rows to display", 10, 500, 50)

    # Map names back to codes
    name_to_code = {v: k for k, v in CLASS_LABELS.items()}
    codes = [name_to_code[n] for n in selected_class]
    filtered = df[df["fetal_health"].isin(codes)]

    st.write(f"Showing {min(n_rows, len(filtered))} of {len(filtered)} filtered samples")
    st.dataframe(filtered.head(n_rows), use_container_width=True)

    # Statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(filtered.describe().round(3), use_container_width=True)

    # Distribution of selected feature
    st.subheader("Feature Distribution")
    feature = st.selectbox("Select feature", df.columns.drop("fetal_health"))
    fig, ax = plt.subplots(figsize=(8, 4))
    for code, name in CLASS_LABELS.items():
        subset = df[df["fetal_health"] == code][feature]
        ax.hist(subset, bins=30, alpha=0.5, label=name)
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)
    plt.close()


# ──────────────────────────────────────────────
# PAGE: Model Comparison
# ──────────────────────────────────────────────
elif page == "🤖 Model Comparison":
    st.subheader("Classifier Performance Comparison")

    metrics = load_metrics()

    if metrics is None:
        st.warning(
            "No saved metrics found. Run the pipeline first:\n\n"
            "```bash\npython fetal_health_pipeline.py\n```"
        )

        if st.button("🚀 Train Models Now"):
            with st.spinner("Training all 10 classifiers..."):
                try:
                    df, X_train_s, X_test_s, y_train, y_test, *_ = get_pipeline_data()
                    results = {}
                    classifiers = get_classifiers()

                    progress = st.progress(0)
                    for i, (name, clf) in enumerate(classifiers.items()):
                        st.text(f"Training: {name}...")
                        clf.fit(X_train_s, y_train)
                        y_pred = clf.predict(X_test_s)
                        acc = accuracy_score(y_test, y_pred)
                        results[name] = {"accuracy": acc, "predictions": y_pred}
                        progress.progress((i + 1) / len(classifiers))

                    st.success("All models trained!")
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
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")
    else:
        # Accuracy comparison chart
        acc_data = sorted(metrics.items(), key=lambda x: x[1]["accuracy"])
        names = [x[0] for x in acc_data]
        accs = [x[1]["accuracy"] for x in acc_data]

        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.cm.RdPu
        norm = [(a - min(accs)) / (max(accs) - min(accs) + 1e-9) for a in accs]
        colors = [cmap(0.3 + 0.6 * n) for n in norm]
        bars = ax.barh(names, accs, color=colors, edgecolor="white", height=0.6)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{acc:.4f}", va="center", fontsize=9)
        ax.set_xlim(min(accs) - 0.02, 1.0)
        ax.set_xlabel("Accuracy")
        ax.set_title("Classifier Accuracy Comparison", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close()

        # Detailed view
        st.subheader("Detailed Classification Reports")
        selected_model = st.selectbox("Select model", sorted(metrics.keys()))
        if selected_model:
            m = metrics[selected_model]
            st.metric("Accuracy", f"{m['accuracy']:.4f}")
            report = m["classification_report"]
            report_df = pd.DataFrame(report).T
            st.dataframe(report_df.round(4), use_container_width=True)

        # Show saved plots if they exist
        st.subheader("Saved Visualizations")
        plots = ["04_confusion_matrices.png", "03_model_comparison.png", "05_lightgbm_tuning.png"]
        for plot_name in plots:
            plot_path = os.path.join(OUTPUT_DIR, plot_name)
            if os.path.exists(plot_path):
                st.image(plot_path, caption=plot_name.replace("_", " ").replace(".png", "").title())


# ──────────────────────────────────────────────
# PAGE: Predict
# ──────────────────────────────────────────────
elif page == "🎯 Predict":
    st.subheader("Make a Prediction")

    # Check if models exist
    has_models = os.path.exists(os.path.join(MODELS_DIR, "best_model.joblib"))

    if not has_models:
        st.warning(
            "No trained models found. Run the pipeline first:\n\n"
            "```bash\npython fetal_health_pipeline.py\n```"
        )
        st.stop()

    # Load artifacts
    try:
        model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
        selector = joblib.load(os.path.join(MODELS_DIR, "feature_selector.joblib"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
        st.success(f"Loaded model: {type(model).__name__}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    st.markdown("---")

    # Input mode
    input_mode = st.radio("Input mode", ["Manual Input", "Sample from Dataset"])

    if input_mode == "Sample from Dataset":
        df = load_data()
        sample_idx = st.number_input(
            "Sample index", min_value=0, max_value=len(df) - 1, value=0
        )
        sample = df.drop("fetal_health", axis=1).iloc[[sample_idx]]
        true_label = CLASS_LABELS[int(df.iloc[sample_idx]["fetal_health"])]

        st.write("**Selected sample features:**")
        st.dataframe(sample, use_container_width=True)
        st.info(f"True label: **{true_label}**")

    else:
        st.write("Enter CTG feature values:")
        feature_cols = [c for c in EXPECTED_COLUMNS if c != "fetal_health"]

        # Load dataset for default values
        df = load_data()
        defaults = df.drop("fetal_health", axis=1).mean()

        cols = st.columns(3)
        values = {}
        for i, feat in enumerate(feature_cols):
            with cols[i % 3]:
                values[feat] = st.number_input(
                    feat[:30] + ("..." if len(feat) > 30 else ""),
                    value=float(defaults[feat]),
                    format="%.4f",
                    help=feat,
                    key=feat,
                )
        sample = pd.DataFrame([values])

    # Predict button
    if st.button("🔮 Predict Fetal Health", type="primary", use_container_width=True):
        with st.spinner("Running prediction..."):
            try:
                # Apply preprocessing pipeline
                sample_selected = selector.transform(sample)
                sample_scaled = scaler.transform(sample_selected)
                prediction = model.predict(sample_scaled)[0]
                label = CLASS_LABELS[int(prediction)]

                css_class = {
                    "Normal": "prediction-normal",
                    "Suspect": "prediction-suspect",
                    "Pathological": "prediction-pathological",
                }[label]

                st.markdown(f"""
                <div class="{css_class}">
                    Prediction: {label}
                </div>
                """, unsafe_allow_html=True)

                # Extract and map probabilities safely without assuming class order
                prob_dict, err_msg = extract_prediction_probabilities(
                    model, sample_scaled, CLASS_LABELS
                )

                if prob_dict is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.subheader("Prediction Confidence")

                    fig = plot_prediction_probabilities(prob_dict)
                    st.pyplot(fig)
                    plt.close(fig)

                    # Display numerical probability metrics below chart
                    col_norm, col_susp, col_path = st.columns(3)
                    with col_norm:
                        st.metric("Normal", f"{prob_dict.get('Normal', 0.0) * 100:.1f}%")
                    with col_susp:
                        st.metric("Suspect", f"{prob_dict.get('Suspect', 0.0) * 100:.1f}%")
                    with col_path:
                        st.metric("Pathological", f"{prob_dict.get('Pathological', 0.0) * 100:.1f}%")

                    total_pct = sum(prob_dict.values()) * 100.0
                    st.caption(f"Total probability sum: {total_pct:.1f}%")
                else:
                    st.info(f"ℹ️ {err_msg}")

            except Exception as e:
                st.error(f"Prediction failed: {e}")


# ──────────────────────────────────────────────
# PAGE: About
# ──────────────────────────────────────────────
elif page == "📄 About":
    st.subheader("About This Project")

    st.markdown("""
    ### Multiclass Classification of Fetal Health using Cardiotocogram Data

    **Authors:** JR Ereyi & Marc Huo, Stanford University

    #### Overview
    This project applies 10 machine learning classifiers to cardiotocography (CTG) data
    to predict fetal health outcomes: **Normal**, **Suspect**, or **Pathological**.

    #### Dataset
    - **Source:** Ayres-de-Campos et al. (SisPorto 2.0) — UCI/Kaggle
    - **Samples:** 2,113 (after deduplication)
    - **Features:** 21 CTG metrics
    - **Classes:** Normal (1646), Suspect (292), Pathological (175)

    #### Models
    | # | Model | Type |
    |---|-------|------|
    | 1 | Logistic Regression | Linear |
    | 2 | Random Forest | Ensemble |
    | 3 | Decision Tree | Tree |
    | 4 | K-Nearest Neighbors | Instance-based |
    | 5 | Support Vector (RBF) | Kernel |
    | 6 | Linear SVC | Linear |
    | 7 | Multi-layer Perceptron | Neural Network |
    | 8 | Gradient Boosting | Ensemble |
    | 9 | LightGBM | Gradient Boosting |
    | 10 | XGBoost | Gradient Boosting |

    #### Key Findings
    - **Best performers:** Gradient Boosting & LightGBM (~95.6% accuracy)
    - **Hyperparameter tuning** of LightGBM via 3-Fold GridSearchCV showed
      that tuning can sometimes decrease performance due to overfitting

    #### Data Leakage Prevention
    - Feature selection (`SelectKBest`) is fitted only on training data
    - Standardization (`StandardScaler`) is fitted only on training data
    - GridSearchCV uses only training data for cross-validation
    """)

    # Show pipeline status
    st.subheader("Pipeline Status")
    checks = {
        "Dataset": os.path.exists(DATASET_PATH),
        "Saved Models": os.path.exists(MODELS_DIR) and len(os.listdir(MODELS_DIR)) > 0,
        "Evaluation Metrics": os.path.exists(os.path.join(OUTPUT_DIR, "metrics.json")),
        "Plots": os.path.exists(os.path.join(OUTPUT_DIR, "01_eda.png")),
    }
    for item, status in checks.items():
        icon = "✅" if status else "❌"
        st.write(f"{icon} {item}")
