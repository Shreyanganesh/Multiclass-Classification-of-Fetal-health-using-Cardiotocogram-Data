"""
Automated tests for the Fetal Health Classification pipeline.

Tests cover:
  - Dataset integrity (columns, classes, shape)
  - Preprocessing correctness (nulls, duplicates, label mapping)
  - Data leakage prevention (feature selection & scaling)
  - Model training and accuracy baselines
  - Model saving, loading, and prediction verification
  - Streamlit app importability
"""

import json
import os
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetal_health_pipeline import (
    DATASET_PATH,
    EXPECTED_COLUMNS,
    CLASS_LABELS,
    K_BEST_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
    load_and_preprocess,
    split_data,
    select_and_scale_features,
    get_classifiers,
    train_and_evaluate,
)


# ───────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────
@pytest.fixture(scope="session")
def raw_df():
    """Load raw CSV for validation tests."""
    return pd.read_csv(DATASET_PATH)


@pytest.fixture(scope="session")
def preprocessed_df():
    """Preprocessed dataframe (nulls/dupes removed, labels remapped)."""
    return load_and_preprocess(DATASET_PATH)


@pytest.fixture(scope="session")
def split_datasets(preprocessed_df):
    """Train/test split."""
    return split_data(preprocessed_df)


@pytest.fixture(scope="session")
def scaled_datasets(split_datasets):
    """Feature-selected and scaled train/test sets."""
    X_train, X_test, y_train, y_test = split_datasets
    X_train_s, X_test_s, selector, scaler, features = select_and_scale_features(
        X_train, X_test, y_train
    )
    return X_train_s, X_test_s, y_train, y_test, selector, scaler, features


# ───────────────────────────────────────────────
# 1. DATASET INTEGRITY TESTS
# ───────────────────────────────────────────────
class TestDatasetIntegrity:
    """Verify the CSV file has expected structure."""

    def test_file_exists(self):
        assert os.path.exists(DATASET_PATH), f"Dataset not found: {DATASET_PATH}"

    def test_expected_columns(self, raw_df):
        for col in EXPECTED_COLUMNS:
            assert col in raw_df.columns, f"Missing column: {col}"

    def test_column_count(self, raw_df):
        assert raw_df.shape[1] == 22, f"Expected 22 columns, got {raw_df.shape[1]}"

    def test_raw_target_classes(self, raw_df):
        classes = sorted(raw_df["fetal_health"].unique())
        assert classes == [1.0, 2.0, 3.0], f"Expected [1.0, 2.0, 3.0], got {classes}"

    def test_minimum_sample_count(self, raw_df):
        assert raw_df.shape[0] >= 10000, f"Too few samples: {raw_df.shape[0]}"

    def test_no_all_null_columns(self, raw_df):
        all_null = raw_df.columns[raw_df.isnull().all()].tolist()
        assert len(all_null) == 0, f"All-null columns: {all_null}"

    def test_numeric_features(self, raw_df):
        for col in raw_df.columns:
            assert pd.api.types.is_numeric_dtype(raw_df[col]), (
                f"Column '{col}' is not numeric"
            )


# ───────────────────────────────────────────────
# 2. PREPROCESSING TESTS
# ───────────────────────────────────────────────
class TestPreprocessing:
    """Verify preprocessing correctness."""

    def test_no_nulls_after_preprocessing(self, preprocessed_df):
        assert preprocessed_df.isnull().sum().sum() == 0

    def test_no_duplicates_after_preprocessing(self, preprocessed_df):
        assert preprocessed_df.duplicated().sum() == 0

    def test_label_remapping(self, preprocessed_df):
        classes = sorted(preprocessed_df["fetal_health"].unique())
        assert classes == [0, 1, 2], f"Expected [0, 1, 2], got {classes}"

    def test_expected_sample_count(self, preprocessed_df):
        # Augmented dataset: ~10,000 samples (minus any duplicates removed)
        assert preprocessed_df.shape[0] >= 9500, (
            f"Expected ~10000 samples, got {preprocessed_df.shape[0]}"
        )

    def test_class_distribution(self, preprocessed_df):
        counts = preprocessed_df["fetal_health"].value_counts().sort_index()
        # Augmented dataset preserves ~78% / ~14% / ~8% class proportions
        assert counts[0] >= 7000, f"Normal count too low: {counts[0]}"
        assert counts[1] >= 1200, f"Suspect count too low: {counts[1]}"
        assert counts[2] >= 700, f"Pathological count too low: {counts[2]}"
        assert counts[0] + counts[1] + counts[2] >= 9500, "Total count too low"


# ───────────────────────────────────────────────
# 3. DATA LEAKAGE PREVENTION TESTS
# ───────────────────────────────────────────────
class TestNoDataLeakage:
    """Verify that no test data leaks into training."""

    def test_split_sizes(self, split_datasets):
        X_train, X_test, y_train, y_test = split_datasets
        total = len(y_train) + len(y_test)
        assert total >= 9500, f"Total samples too low: {total}"
        # 70/30 split
        assert abs(len(y_test) / total - 0.30) < 0.02

    def test_no_overlapping_indices(self, split_datasets):
        X_train, X_test, y_train, y_test = split_datasets
        train_idx = set(y_train.index)
        test_idx = set(y_test.index)
        overlap = train_idx & test_idx
        assert len(overlap) == 0, f"Overlapping indices: {overlap}"

    def test_scaler_fitted_on_train_only(self, scaled_datasets):
        """Verify scaler mean/var come from training data, not full dataset."""
        X_train_s, X_test_s, y_train, y_test, selector, scaler, features = scaled_datasets
        # Training data should have mean ≈ 0 and std ≈ 1 after scaling
        train_means = np.abs(X_train_s.mean(axis=0))
        assert np.all(train_means < 1e-10), (
            f"Training data not zero-centered: max mean = {train_means.max()}"
        )
        # Test data should NOT necessarily have mean ≈ 0
        # (if it does, that could indicate data leakage, but it's not definitive)

    def test_feature_selector_fitted_on_train_only(self, scaled_datasets):
        """Verify feature selector was fitted on training data."""
        X_train_s, X_test_s, y_train, y_test, selector, scaler, features = scaled_datasets
        # Selector should have scores_ attribute from fitting
        assert hasattr(selector, "scores_"), "Selector not fitted"
        assert len(selector.scores_) == 21  # All original features
        assert sum(selector.get_support()) == K_BEST_FEATURES

    def test_selected_feature_count(self, scaled_datasets):
        X_train_s, X_test_s, y_train, y_test, selector, scaler, features = scaled_datasets
        assert X_train_s.shape[1] == K_BEST_FEATURES
        assert X_test_s.shape[1] == K_BEST_FEATURES


# ───────────────────────────────────────────────
# 4. MODEL TRAINING TESTS
# ───────────────────────────────────────────────
class TestModelTraining:
    """Verify models train and achieve baseline accuracy."""

    def test_all_classifiers_instantiate(self):
        classifiers = get_classifiers()
        assert len(classifiers) == 10

    def test_models_achieve_baseline(self, scaled_datasets):
        X_train_s, X_test_s, y_train, y_test, *_ = scaled_datasets
        results = train_and_evaluate(X_train_s, X_test_s, y_train, y_test)
        for name, data in results.items():
            # All models should beat random chance (33% for 3 classes)
            assert data["accuracy"] > 0.70, (
                f"{name} accuracy {data['accuracy']:.4f} below baseline"
            )

    def test_predictions_are_valid_classes(self, scaled_datasets):
        X_train_s, X_test_s, y_train, y_test, *_ = scaled_datasets
        results = train_and_evaluate(X_train_s, X_test_s, y_train, y_test)
        valid_classes = {0, 1, 2}
        for name, data in results.items():
            pred_classes = set(data["predictions"])
            assert pred_classes.issubset(valid_classes), (
                f"{name} predicted invalid classes: {pred_classes - valid_classes}"
            )


# ───────────────────────────────────────────────
# 5. MODEL PERSISTENCE TESTS
# ───────────────────────────────────────────────
class TestModelPersistence:
    """Verify models can be saved, loaded, and produce same predictions."""

    def test_save_and_load_model(self, scaled_datasets):
        X_train_s, X_test_s, y_train, y_test, selector, scaler, features = scaled_datasets

        # Train a simple model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=RANDOM_STATE)
        model.fit(X_train_s, y_train)
        y_pred_original = model.predict(X_test_s)

        # Save and reload
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            tmp_path = f.name
        try:
            joblib.dump(model, tmp_path)
            loaded_model = joblib.load(tmp_path)
            y_pred_loaded = loaded_model.predict(X_test_s)
            np.testing.assert_array_equal(y_pred_original, y_pred_loaded)
        finally:
            os.unlink(tmp_path)

    def test_save_and_load_scaler(self, scaled_datasets):
        _, _, _, _, _, scaler, _ = scaled_datasets

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            tmp_path = f.name
        try:
            joblib.dump(scaler, tmp_path)
            loaded_scaler = joblib.load(tmp_path)
            np.testing.assert_array_almost_equal(scaler.mean_, loaded_scaler.mean_)
            np.testing.assert_array_almost_equal(scaler.scale_, loaded_scaler.scale_)
        finally:
            os.unlink(tmp_path)

    def test_save_and_load_selector(self, scaled_datasets):
        _, _, _, _, selector, _, _ = scaled_datasets

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            tmp_path = f.name
        try:
            joblib.dump(selector, tmp_path)
            loaded_selector = joblib.load(tmp_path)
            np.testing.assert_array_equal(
                selector.get_support(), loaded_selector.get_support()
            )
        finally:
            os.unlink(tmp_path)

    def test_full_inference_pipeline(self, scaled_datasets, preprocessed_df):
        """Test loading all artifacts and running inference on a raw sample."""
        X_train_s, X_test_s, y_train, y_test, selector, scaler, features = scaled_datasets

        # Train and save
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=RANDOM_STATE)
        model.fit(X_train_s, y_train)

        # Simulate raw input: take a sample from the original dataframe
        raw_sample = preprocessed_df.drop("fetal_health", axis=1).iloc[:1]

        # Apply pipeline: select -> scale -> predict
        sample_selected = selector.transform(raw_sample)
        sample_scaled = scaler.transform(sample_selected)
        prediction = model.predict(sample_scaled)

        assert prediction[0] in {0, 1, 2}


# ───────────────────────────────────────────────
# 6. SAVED ARTIFACTS & INFERENCE TESTS
# ───────────────────────────────────────────────
class TestSavedArtifacts:
    """Verify saved models, artifacts, and output metrics on disk."""

    def test_artifacts_exist(self):
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
        assert os.path.exists(os.path.join(models_dir, "best_model.joblib"))
        assert os.path.exists(os.path.join(models_dir, "feature_selector.joblib"))
        assert os.path.exists(os.path.join(models_dir, "scaler.joblib"))

    def test_metrics_json_valid(self):
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        metrics_path = os.path.join(output_dir, "metrics.json")
        assert os.path.exists(metrics_path)
        with open(metrics_path) as f:
            metrics = json.load(f)
        assert len(metrics) == 10
        for name, data in metrics.items():
            assert "accuracy" in data
            assert data["accuracy"] > 0.70

    def test_loaded_best_model_inference(self, split_datasets):
        X_train, X_test, y_train, y_test = split_datasets
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
        best_model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
        selector = joblib.load(os.path.join(models_dir, "feature_selector.joblib"))
        scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))

        X_test_sel = selector.transform(X_test)
        X_test_scaled = scaler.transform(X_test_sel)
        y_pred = best_model.predict(X_test_scaled)

        acc = (y_pred == y_test).mean()
        assert acc > 0.90, f"Loaded model accuracy {acc} is below expected threshold"


# ───────────────────────────────────────────────
# 7. STREAMLIT APP TESTS
# ───────────────────────────────────────────────
class TestStreamlitApp:
    """Basic importability and syntax check for the Streamlit app."""

    def test_streamlit_app_file_exists(self):
        app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
        assert os.path.exists(app_path), "Streamlit app (app.py) not found"

    def test_streamlit_app_valid_syntax(self):
        app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            code = f.read()
        compile(code, app_path, "exec")

    def test_extract_prediction_probabilities(self, scaled_datasets):
        from app import extract_prediction_probabilities
        X_train_s, X_test_s, *_ = scaled_datasets
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
        model = joblib.load(os.path.join(models_dir, "best_model.joblib"))

        prob_dict, err_msg = extract_prediction_probabilities(model, X_test_s[:1], CLASS_LABELS)
        assert err_msg is None
        assert prob_dict is not None
        assert set(prob_dict.keys()) == {"Normal", "Suspect", "Pathological"}
        assert abs(sum(prob_dict.values()) - 1.0) < 1e-4
        for val in prob_dict.values():
            assert 0.0 <= val <= 1.0

    def test_unsupported_model_probability_fallback(self, scaled_datasets):
        from app import extract_prediction_probabilities
        X_train_s, X_test_s, *_ = scaled_datasets

        class MockUnsupportedModel:
            pass

        prob_dict, err_msg = extract_prediction_probabilities(
            MockUnsupportedModel(), X_test_s[:1], CLASS_LABELS
        )
        assert prob_dict is None
        assert err_msg is not None
        assert "not support" in err_msg.lower() or "unavailable" in err_msg.lower()

    def test_plot_prediction_probabilities_structure(self):
        from app import plot_prediction_probabilities
        import matplotlib.pyplot as plt

        dummy_probs = {"Normal": 0.85, "Suspect": 0.10, "Pathological": 0.05}
        fig = plot_prediction_probabilities(dummy_probs)
        ax = fig.axes[0]

        # Exactly 3 bars
        bars = [p for p in ax.patches]
        assert len(bars) == 3

        # Exactly 3 labels in order
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert labels == ["Normal", "Suspect", "Pathological"]

        # Strictly non-negative y-axis
        assert ax.get_ylim()[0] == 0

        # Percentage text above each bar
        assert len(ax.texts) == 3
        plt.close(fig)

    def test_prediction_probabilities_update_with_inputs(self, split_datasets):
        from app import extract_prediction_probabilities
        X_train, X_test, y_train, y_test = split_datasets
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
        model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
        selector = joblib.load(os.path.join(models_dir, "feature_selector.joblib"))
        scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))

        # Find a normal sample and a suspect/pathological sample
        norm_idx = y_test[y_test == 0].index[0]
        susp_or_path_idx = y_test[y_test != 0].index[0]

        sample_norm = scaler.transform(selector.transform(X_test.loc[[norm_idx]]))
        sample_alt = scaler.transform(selector.transform(X_test.loc[[susp_or_path_idx]]))

        probs_norm, _ = extract_prediction_probabilities(model, sample_norm, CLASS_LABELS)
        probs_alt, _ = extract_prediction_probabilities(model, sample_alt, CLASS_LABELS)

        assert probs_norm != probs_alt, "Probabilities should update when input sample changes"
        assert probs_norm["Normal"] > probs_alt["Normal"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


