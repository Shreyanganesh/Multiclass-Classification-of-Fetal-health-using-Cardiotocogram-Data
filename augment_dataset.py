"""
Dataset Augmentation for Fetal Health Classification
=====================================================
Extends the original CTG dataset (~2,126 samples) to 10,000 samples by
generating synthetic records using per-class statistical resampling with
controlled Gaussian jitter.

Strategy:
  1. Keep ALL original samples unchanged (they form the ground truth).
  2. For each class, generate additional synthetic samples by:
     a. Randomly sampling existing rows (with replacement) as "seeds"
     b. Adding Gaussian noise scaled to a fraction of each feature's
        within-class standard deviation (jitter_scale=0.05 by default)
     c. Clipping values to stay within observed per-class [min, max] bounds
     d. Rounding integer-valued features to preserve their discrete nature
  3. Original class proportions are preserved in the augmented dataset.

Output:
  - fetal_health_augmented.csv  (10,000 rows, 22 columns)

Usage:
  python augment_dataset.py
"""

import os
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
_base_dir = os.path.dirname(os.path.abspath(__file__))
_orig_candidate = os.path.join(_base_dir, "fetal_health_original.csv")
ORIGINAL_CSV = _orig_candidate if os.path.exists(_orig_candidate) else os.path.join(_base_dir, "fetal_health.csv")
OUTPUT_CSV = os.path.join(_base_dir, "fetal_health_augmented.csv")
MAIN_CSV = os.path.join(_base_dir, "fetal_health.csv")
TARGET_TOTAL = 10_000
RANDOM_SEED = 42
JITTER_SCALE = 0.05  # Noise magnitude: 5% of within-class std

# Features that are inherently integer-valued in the original dataset
INTEGER_FEATURES = {
    "baseline value",
    "abnormal_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability",
    "histogram_width",
    "histogram_min",
    "histogram_max",
    "histogram_number_of_peaks",
    "histogram_number_of_zeroes",
    "histogram_mode",
    "histogram_mean",
    "histogram_median",
    "histogram_variance",
    "histogram_tendency",
}


def generate_synthetic_samples(
    class_df: pd.DataFrame,
    n_synthetic: int,
    feature_cols: list,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate n_synthetic samples from a single class's real data.

    For each synthetic sample:
      1. Randomly pick a real row as the seed.
      2. Add Gaussian noise ~ N(0, jitter_scale * class_std) per feature.
      3. Clip to the observed [min, max] range per feature within this class.
      4. Round integer-valued features.
    """
    if n_synthetic <= 0:
        return pd.DataFrame(columns=class_df.columns)

    # Compute per-feature stats within this class
    feature_means = class_df[feature_cols].mean()
    feature_stds = class_df[feature_cols].std().replace(0, 1e-8)
    feature_mins = class_df[feature_cols].min()
    feature_maxs = class_df[feature_cols].max()

    # Sample seed rows with replacement
    seed_indices = rng.choice(class_df.index, size=n_synthetic, replace=True)
    synthetic = class_df.loc[seed_indices, :].copy().reset_index(drop=True)

    # Add Gaussian jitter
    noise = rng.normal(
        loc=0.0,
        scale=JITTER_SCALE * feature_stds.values,
        size=(n_synthetic, len(feature_cols)),
    )
    synthetic[feature_cols] = synthetic[feature_cols].values + noise

    # Clip to observed bounds
    for col in feature_cols:
        synthetic[col] = synthetic[col].clip(lower=feature_mins[col], upper=feature_maxs[col])

    # Round integer features
    for col in feature_cols:
        if col in INTEGER_FEATURES:
            synthetic[col] = synthetic[col].round(0).astype(float)

    return synthetic


def augment_dataset():
    """Main augmentation routine."""
    print("=" * 60)
    print("DATASET AUGMENTATION")
    print("=" * 60)

    # Load original and remove any duplicate records from base data
    df_orig = pd.read_csv(ORIGINAL_CSV).drop_duplicates().reset_index(drop=True)
    n_orig = len(df_orig)
    print(f"\n  Original unique dataset: {n_orig} samples, {df_orig.shape[1]} columns")

    # Class distribution (original uses 1.0, 2.0, 3.0)
    class_col = "fetal_health"
    class_counts = df_orig[class_col].value_counts().sort_index()
    print(f"\n  Original class distribution:")
    class_names = {1.0: "Normal", 2.0: "Suspect", 3.0: "Pathological"}
    for cls_val, count in class_counts.items():
        pct = count / n_orig * 100
        print(f"    {class_names[cls_val]} ({cls_val}): {count} ({pct:.1f}%)")

    # Calculate target counts per class (preserve proportions)
    n_to_generate = TARGET_TOTAL - n_orig
    if n_to_generate <= 0:
        print(f"\n  Dataset already has {n_orig} >= {TARGET_TOTAL} samples. No augmentation needed.")
        return

    print(f"\n  Target total: {TARGET_TOTAL}")
    print(f"  New samples to generate: {n_to_generate}")

    feature_cols = [c for c in df_orig.columns if c != class_col]
    rng = np.random.default_rng(RANDOM_SEED)

    # Allocate new samples per class proportionally
    class_fractions = class_counts / n_orig
    target_per_class = {}
    remaining = n_to_generate
    sorted_classes = sorted(class_counts.index)

    for cls_val in sorted_classes[:-1]:
        target_per_class[cls_val] = int(round(class_fractions[cls_val] * n_to_generate))
        remaining -= target_per_class[cls_val]
    target_per_class[sorted_classes[-1]] = remaining  # Give remainder to last class

    print(f"\n  Synthetic samples per class:")
    for cls_val in sorted_classes:
        total = class_counts[cls_val] + target_per_class[cls_val]
        print(f"    {class_names[cls_val]}: +{target_per_class[cls_val]} "
              f"(original {class_counts[cls_val]} → total {total})")

    # Generate synthetic samples for each class, ensuring uniqueness
    synthetic_frames = []
    for cls_val in sorted_classes:
        class_df = df_orig[df_orig[class_col] == cls_val]
        needed = target_per_class[cls_val]
        collected = []
        collected_count = 0
        while collected_count < needed:
            batch_size = (needed - collected_count) * 2  # generate buffer to account for possible collisions
            syn_batch = generate_synthetic_samples(class_df, batch_size, feature_cols, rng)
            # drop internal duplicates and collisions with original
            combined_test = pd.concat([df_orig] + collected + [syn_batch], ignore_index=True)
            non_dupes = syn_batch[~combined_test.duplicated().iloc[-len(syn_batch):].values]
            if len(non_dupes) > 0:
                take = min(len(non_dupes), needed - collected_count)
                collected.append(non_dupes.iloc[:take])
                collected_count += take

        cls_synthetic = pd.concat(collected, ignore_index=True)
        synthetic_frames.append(cls_synthetic)
        print(f"  ✓ Generated {len(cls_synthetic)} unique synthetic {class_names[cls_val]} samples")

    # Combine original + synthetic
    df_synthetic = pd.concat(synthetic_frames, ignore_index=True)
    df_augmented = pd.concat([df_orig, df_synthetic], ignore_index=True)

    # Shuffle
    df_augmented = df_augmented.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    # Validate
    print(f"\n  Augmented dataset: {len(df_augmented)} samples, {df_augmented.shape[1]} columns")
    aug_counts = df_augmented[class_col].value_counts().sort_index()
    print(f"\n  Augmented class distribution:")
    for cls_val, count in aug_counts.items():
        pct = count / len(df_augmented) * 100
        print(f"    {class_names[cls_val]} ({cls_val}): {count} ({pct:.1f}%)")

    # Sanity checks
    assert len(df_augmented) == TARGET_TOTAL, f"Expected {TARGET_TOTAL}, got {len(df_augmented)}"
    assert df_augmented.isnull().sum().sum() == 0, "Null values detected"
    assert sorted(df_augmented[class_col].unique()) == [1.0, 2.0, 3.0], "Invalid classes"
    assert list(df_augmented.columns) == list(df_orig.columns), "Column mismatch"

    # Statistical comparison
    print(f"\n  Feature statistics comparison (original vs augmented):")
    print(f"  {'Feature':<55} {'Orig Mean':>10} {'Aug Mean':>10} {'Orig Std':>10} {'Aug Std':>10}")
    print(f"  {'-'*95}")
    for col in feature_cols[:10]:
        om = df_orig[col].mean()
        am = df_augmented[col].mean()
        os_ = df_orig[col].std()
        as_ = df_augmented[col].std()
        print(f"  {col:<55} {om:>10.3f} {am:>10.3f} {os_:>10.3f} {as_:>10.3f}")

    # Save
    df_augmented.to_csv(OUTPUT_CSV, index=False)
    df_augmented.to_csv(MAIN_CSV, index=False)
    print(f"\n  ✓ Saved augmented dataset to: {OUTPUT_CSV} and {MAIN_CSV}")
    print(f"  File size: {os.path.getsize(MAIN_CSV) / 1024:.1f} KB")

    print("\n" + "=" * 60)
    print("AUGMENTATION COMPLETE")
    print("=" * 60)

    return df_augmented


if __name__ == "__main__":
    augment_dataset()
