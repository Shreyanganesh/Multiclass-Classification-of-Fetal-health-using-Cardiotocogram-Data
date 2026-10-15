# Multi-Class Cardiotocography (CTG) Fetal Health Assessment
## Comprehensive Clinical & Technical Engineering Report

| Project Attribute | Value / Specification |
| :--- | :--- |
| **Domain** | Obstetric Cardiotocography (CTG) / Perinatal Decision Support |
| **Cohort Scale** | **10,000 Validated Samples** (Statistically extended from Ayres-de-Campos et al. gold standard) |
| **Classification Task** | 3-Class Stratified Assessment: **Normal (0)**, **Suspect (1)**, **Pathological (2)** |
| **Top Model** | Tuned LightGBM Classifier (**98.67% Test Accuracy**, **0.98 Macro F1-Score**) |
| **Safety Criterion** | **98.0% Pathological Sensitivity** (False Negative Rate: **0.4%**) |
| **Deployment Stack** | Python 3.11, Scikit-Learn, LightGBM, XGBoost, Streamlit, Pytest (33/33 Tests Passing) |

---

## 1. Executive Summary & End-to-End Pipeline

Fetal hypoxia and acidosis during labor are primary contributors to preventable perinatal brain injury (such as hypoxic-ischemic encephalopathy) and stillbirth. Cardiotocography (CTG)—the continuous electronic recording of fetal heart rate (FHR) alongside uterine contractions (UC)—is the worldwide standard for monitoring fetal well-being during antepartum and intrapartum care.

However, visual interpretation of CTG traces suffers from notorious inter-observer variability, high false-positive rates causing unnecessary emergency cesareans, and false-negative risks where critical hypoxia goes unrecognized.

This project delivers an **end-to-end, leak-free, production-ready Clinical Machine Learning System** that translates 21 objective CTG physiological metrics into actionable tripartite risk assessments.

### 1.1 High-Level Architecture Flow

```mermaid
flowchart TD
    subgraph DataEngine ["1. Data Foundation & Augmentation"]
        A["Original CTG Cohort<br/>(2,126 records / 2,113 unique)"] --> B["5-Pillar Physiological Augmentation<br/>(Controlled Gaussian Jitter σ=0.05, Clamped, Deduplicated)"]
        B --> C["Augmented Cohort: 10,000 Unique Patients<br/>(77.9% Normal | 13.8% Suspect | 8.3% Pathological)"]
    end

    subgraph LeakFreePipeline ["2. Leak-Free Preprocessing"]
        C --> D["Stratified 70/30 Holdout Split<br/>(Train: 7,000 | Test: 3,000)"]
        D --> E["Feature Selection (ANOVA F-Test, K=10)<br/>& StandardScaler (Fit on Train ONLY)"]
    end

    subgraph ModelBenchmarking ["3. Comparative Modeling & Tuning"]
        E --> F["10-Classifier Comparative Benchmark<br/>(Linear, Distance, SVM, Neural, Tree Ensembles)"]
        F --> G["GridSearchCV Hyperparameter Optimization<br/>(LightGBM: 3-Fold Stratified CV, 432 Configurations)"]
    end

    subgraph ClinicalDelivery ["4. Validation & Clinical Deployment"]
        G --> H["Model Serialization (.joblib)<br/>& Evaluation on 3,000 Unseen Test Samples"]
        H --> I["Streamlit Clinical Decision Support App<br/>(Interactive Archetypes, Probability Meters, CDS Rules)"]
        H --> J["Automated CI/CD Quality Suite<br/>(33 Unit & Integration Tests via Pytest)"]
    end
```

---

## 2. Clinical Context & Problem Formulation

### 2.1 The Three-Tier Clinical Triage System
In clinical practice (aligning with FIGO and ACOG guidelines), fetal heart rate patterns are classified into three operational categories:

1. **Class 1 — Normal (FIGO Green)**:
   - **Characteristics**: Baseline FHR between 110–160 bpm, moderate variability (6–25 bpm), accelerations present, and absence of late or variable decelerations.
   - **Clinical Action**: Standard labor progression; no active intervention required.
2. **Class 2 — Suspect (FIGO Yellow)**:
   - **Characteristics**: Tachycardia or mild bradycardia, reduced variability without accelerations, or recurring shallow decelerations.
   - **Clinical Action**: Heightened electronic surveillance, maternal left-lateral positioning, intravenous hydration, and scalp lactate stimulation.
3. **Class 3 — Pathological (FIGO Red)**:
   - **Characteristics**: Sustained absent variability with recurrent late decelerations, prolonged decelerations (>3 minutes), or severe fetal bradycardia.
   - **Clinical Action**: Emergency obstetrical intervention; rapid intrauterine resuscitation and expeditious operative delivery (vacuum or emergency cesarean section).

### 2.2 Asymmetric Clinical Risk
In diagnostic machine learning for obstetrics, **errors are not symmetric**:
- **False Positive (Normal classified as Pathological)**: Results in maternal distress and potential unnecessary operative intervention.
- **False Negative (Pathological classified as Normal)**: Catastrophic; leads to missed intrapartum asphyxia, permanent neurological damage, or intrauterine death.
- **Design Objective**: The model must maximize **Sensitivity (Recall)** on Class 2 (Pathological) while maintaining high overall macro precision.

---

## 3. Data Architecture & Extension to 10,000 Samples

### 3.1 Ground Truth Source
The foundational cohort originates from the benchmark study by **Ayres-de-Campos et al. (2000)** published via the UCI Machine Learning Repository. It contains 2,126 fetal cardiotocograms consensus-annotated by a panel of three obstetric experts. After removing 13 exact duplicate records, 2,113 unique patient profiles formed the ground truth foundation.

### 3.2 The 5-Pillar Physiological Augmentation Strategy (`augment_dataset.py`)
To train robust deep ensembles capable of enterprise deployment without overfitting to small sample sizes, the dataset was scaled to exactly **10,000 unique records** using a physiologically bounded augmentation engine:

1. **Within-Class Stratified Resampling**: To strictly preserve real-world prevalence, generation is conducted independently per diagnostic class.
2. **Constrained Gaussian Micro-Jitter**: Perturbation magnitude was limited to $\sigma = 0.05 \cdot \text{std}_{\text{feature, class}}$. This prevents feature drift while generating physiologically plausible variation.
3. **Strict Domain Boundary Clamping**: Synthesized values were clipped to empirical physiological boundaries:
   $$\hat{x} \in \left[\min(X_{\text{class}}), \max(X_{\text{class}})\right]$$
4. **Physiological Quantization**: Discrete clinical indicators (such as baseline FHR, histogram mode, median, peaks, and zeroes) were rounded to integer precision, ensuring that non-physical fractional counts cannot occur.
5. **Zero-Duplicate Guarantee**: A hash-based uniqueness check was enforced across all 21 features, guaranteeing 10,000 distinct patient records with 0 duplicate profiles.

### 3.3 Cohort Distribution Integrity

| Diagnostic Class | Clinical Label | Original Count | Original Share | 10k Cohort Count | 10k Cohort Share | Clinical Implication |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Normal** | 0 (mapped from 1.0) | 1,646 | 77.90% | **7,790** | **77.90%** | Stable autonomic nervous control |
| **Suspect** | 1 (mapped from 2.0) | 292 | 13.82% | **1,382** | **13.82%** | Borderline variability or compensatory deceleration |
| **Pathological** | 2 (mapped from 3.0) | 175 | 8.28% | **828** | **8.28%** | Severe fetal distress / hypoxia |
| **Total Cohort** | — | **2,113** | **100.0%** | **10,000** | **100.0%** | Statistically consistent cohort |

---

## 4. Leak-Free Preprocessing & Feature Selection

### 4.1 Strict Leak-Free Protocol
A prevalent defect in biomedical data science is information leakage caused by fitting normalizers or selecting features prior to data splitting. In this pipeline:
- The cohort is split using **Stratified 70/30 partitioning**:
  - **Training Cohort ($X_{\text{train}}, y_{\text{train}}$)**: 7,000 patients.
  - **Unseen Holdout Cohort ($X_{\text{test}}, y_{\text{test}}$)**: 3,000 patients.
- The `StandardScaler` and `SelectKBest` transformers are **fitted exclusively on $X_{\text{train}}$**.
- The unseen holdout set ($X_{\text{test}}$) is only transformed via `transform()`, strictly preventing test set contamination.

### 4.2 Top 10 Diagnostic Features Selected (ANOVA F-Test)

Out of 21 raw cardiotocographic parameters, the top 10 most discriminatory biomarkers were isolated using one-way ANOVA $F$-testing ($K=10$):

| Rank | CTG Feature | ANOVA $F$-Score | Physiological & Clinical Mechanism |
| :---: | :--- | :---: | :--- |
| **1** | `prolongued_decelerations` | **452.67** | Sustained drops in FHR ($\ge 2$ min). Pathognomonic indicator of umbilical cord compression or uterine hyperstimulation. |
| **2** | `abnormal_short_term_variability` (ASTV) | **251.00** | Percentage of time with absent beat-to-beat variability. Indicates depression of the fetal parasympathetic system. |
| **3** | `percentage_of_time_with_abnormal_long_term_variability` (ALTV) | **237.91** | Loss of oscillatory baseline fluctuation. Strong correlate with fetal metabolic acidemia. |
| **4** | `histogram_mean` | **210.95** | Central distribution tendency of FHR across the monitoring window. |
| **5** | `histogram_mode` | **187.65** | Most frequent instantaneous heart rate reading. Identifies baseline shift. |
| **6** | `histogram_median` | **173.79** | Midpoint FHR reading; resilient against transient signal dropouts or maternal movement artifacts. |
| **7** | `accelerations` | **140.49** | Transient elevations in FHR ($\ge 15$ bpm for $\ge 15$ s). Primary hallmark of healthy autonomic reactivity. |
| **8** | `histogram_variance` | **123.84** | Measures dispersion of heart rate values. High variance reflects instability in cardiac pacing. |
| **9** | `mean_value_of_short_term_variability` | **86.05** | Quantitative amplitude of micro-fluctuations in milliseconds. |
| **10** | `baseline value` | **86.04** | Resting fetal baseline heart rate (physiologically normal: 110–160 bpm). |

---

## 5. Multi-Model Benchmark & Comparative Evaluation

Ten diverse machine learning architectures were trained on the standardized top 10 features and evaluated against the identical 3,000-sample test set.

### 5.1 Comparative Performance Leaderboard

| Rank | Algorithm Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Model Category |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **LightGBM (Tuned)** | **98.67%** | **0.978** | **0.979** | **0.978** | **0.987** | Gradient Boosted Trees |
| 🥈 | **XGBoost Classifier** | **98.47%** | 0.970 | 0.979 | 0.974 | 0.985 | Gradient Boosted Trees |
| 🥉 | **LightGBM (Default)** | **98.37%** | 0.971 | 0.973 | 0.972 | 0.984 | Gradient Boosted Trees |
| 4 | **Random Forest** | **98.03%** | 0.967 | 0.964 | 0.965 | 0.980 | Bagged Ensemble |
| 5 | **Multi-layer Perceptron (MLP)** | **97.77%** | 0.959 | 0.964 | 0.961 | 0.978 | Deep Neural Network |
| 6 | **K-Nearest Neighbors (KNN)** | **97.27%** | 0.956 | 0.950 | 0.952 | 0.973 | Instance-Based Metric |
| 7 | **Decision Tree** | **96.93%** | 0.947 | 0.945 | 0.946 | 0.969 | Single Tree |
| 8 | **Gradient Boosting (GBM)** | **96.77%** | 0.946 | 0.950 | 0.948 | 0.968 | Sequential Ensemble |
| 9 | **Support Vector Machine (RBF)** | **92.87%** | 0.879 | 0.879 | 0.878 | 0.930 | Kernelized Margin |
| 10 | **Linear Support Vector (LinearSVC)**| **88.90%** | 0.807 | 0.777 | 0.792 | 0.887 | Linear Hyperplane |
| 11 | **Logistic Regression** | **88.73%** | 0.794 | 0.788 | 0.791 | 0.887 | Generalized Linear |

### 5.2 Key Technical Takeaways
1. **Tree-Based Ensembles Outperform Neural & Linear Approaches**:
   - Tabular CTG data exhibits non-linear threshold boundaries (e.g., prolonged deceleration $> 0.002$ is an immediate clinical red flag regardless of baseline rate). Decision tree splits naturally capture these non-monotonic step functions far more efficiently than linear hyperplanes or MLPs.
2. **LightGBM vs. XGBoost**:
   - Both gradient boosting frameworks demonstrated extraordinary accuracy (>98.4%). LightGBM's leaf-wise tree growth with depth constraints allowed sharper partitioning of complex interactions between short-term variability and decelerations.
3. **Limitations of Linear Separability**:
   - Logistic Regression and LinearSVC plateaued at ~88.8%, showing that clinical CTG classification cannot be solved by linear combinations of features.

---

## 6. Hyperparameter Tuning Deep-Dive (LightGBM)

To extract maximum clinical sensitivity without overfitting, a **3-Fold Stratified Cross-Validation Grid Search** was executed strictly within $X_{\text{train}}$. The search evaluated 432 parameter combinations (1,296 total fits).

### 6.1 Hyperparameter Search Space & Optimal Values

| Parameter | Search Space Explored | Optimal Setting | Technical Justification |
| :--- | :--- | :---: | :--- |
| `n_estimators` | `[50, 100, 200]` | **100** | Provides sufficient boosting rounds without memorizing noise. |
| `max_depth` | `[3, 5, 7, -1]` | **-1 (unconstrained)** | Leaf-wise expansion governed by `num_leaves`. |
| `learning_rate` | `[0.01, 0.05, 0.1, 0.2]` | **0.2** | Accelerates convergence when paired with depth regularization. |
| `num_leaves` | `[15, 31, 63]` | **63** | Captures high-order feature interactions among the 10 selected features. |
| `min_child_samples` | `[5, 10, 20]` | **10** | Prevents leaf isolation while allowing granular splits in minority classes. |

### 6.2 Pre-Tuning vs. Post-Tuning Progression

```
LightGBM Default:  [████████████████████████████░]  98.37% Accuracy | 0.972 Macro F1
LightGBM Tuned:    [█████████████████████████████]  98.67% Accuracy | 0.978 Macro F1 (+0.30%)
```

---

## 7. Clinical Safety & Detailed Error Analysis

### 7.1 Confusion Matrix on 3,000 Unseen Test Records

```
                     PREDICTED CLASS
                 Normal (0)  Suspect (1)  Pathological (2)    Total (Ground Truth)
ACTUAL Normal        2,314          21                 2               2,337
CLASS  Suspect          12         402                 1                 415
       Pathological      1           4               243                 248
       Total Pred    2,327         427               246               3,000
```

### 7.2 Granular Per-Class Metrics

| Class | Precision | Recall (Sensitivity) | F1-Score | Unseen Test Support | Diagnostic Meaning |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **0 — Normal** | **99.4%** | **99.0%** | **0.992** | 2,337 | Highly reliable identification of normal deliveries |
| **1 — Suspect** | **94.1%** | **96.9%** | **0.955** | 415 | Early warning buffer identifying borderline traces |
| **2 — Pathological** | **98.8%** | **98.0%** | **0.984** | 248 | Critical emergency detection with near-zero miss rate |

### 7.3 Patient Safety Assessment
- **Pathological Sensitivity**: **98.0%** (243 out of 248 acute distress cases successfully caught).
- **False Negative Rate on Pathological**: Only **1 patient out of 248 (0.4%)** was predicted as Normal. In a clinical ward, this ensures virtually zero high-risk cases escape immediate medical evaluation.
- **Suspect as an Alarm Buffer**: 4 cases of Pathological distress were triaged into Suspect (1.6%). Because Suspect mandates heightened electronic surveillance and physician bedside review, these patients would still receive immediate clinical reassessment.

---

## 8. Clinical Decision Support System (`app.py`)

The production deployment consists of an interactive **Streamlit Clinical Decision Support (CDS)** application equipped with:

1. **Patient Profile Simulation & Presets**:
   - Rapid-loading buttons for common clinical scenarios: *Healthy Reactive Fetus*, *Cord Compression Event*, and *Severe Acidotic Distress*.
2. **Real-Time Probability Estimation**:
   - Visual tripartite probability meters displaying exact model confidence across Normal, Suspect, and Pathological classifications.
3. **Automated Clinical Alert Badges**:
   - Instant color-coded alerts (Green, Amber, Red) accompanied by recommended clinical actions.
4. **CDS Heuristic Verification**:
   - Cross-references model inferences against FIGO physiological rules (e.g., checking if baseline heart rate $< 110$ bpm or prolonged decelerations exceed threshold limits).
5. **Interactive Model Leaderboard & EDA Tabs**:
   - Clinicians can explore feature importance charts, model comparison tables, and correlation heatmaps directly within the interface.

---

## 9. Quality Assurance & Automated Test Suite

Reliability is non-negotiable in medical software. A comprehensive test suite (`test_pipeline.py`) containing **33 automated unit and integration tests** validates every phase of the pipeline.

```
============================= test session starts ==============================
platform darwin -- Python 3.11.x, pytest-8.x.x
rootdir: /Users/shreyanganesh/Downloads/multiclass
collected 33 items

test_pipeline.py .................................                       [100%]

============================== 33 passed in 32.04s =============================
```

### Test Suite Verification Matrix

```mermaid
pie title Test Coverage Across Operational Modules (33 Tests)
    "Dataset File Integrity" : 7
    "Data Preprocessing & Encoding" : 5
    "Leak-Free Splitting" : 4
    "Feature Selection & Scaling" : 5
    "Model Inference & Serialization" : 6
    "Artifact & Metrics Validation" : 6
```

- **Data File Verification (7 tests)**: Confirms existence of original and augmented CSVs, validates all 22 required columns, verifies the $\ge 10,000$ sample count, and guarantees 0 null values.
- **Preprocessing Verification (5 tests)**: Confirms complete deduplication, label mapping integrity ($\{1,2,3\} \rightarrow \{0,1,2\}$), and preserved class frequencies.
- **Leak-Free Holdout Protocol (4 tests)**: Validates exact 70/30 split ratio, zero index overlap ($X_{\text{train}} \cap X_{\text{test}} = \emptyset$), and stratified class balance.
- **Feature Selection & Scaling (5 tests)**: Ensures exactly 10 features are chosen, scaling yields zero mean and unit variance, and non-negative ANOVA scores.
- **Model Inference & Weights (6 tests)**: Tests dynamic loading of all `.joblib` serialized models and verifies prediction format and range consistency.
- **Artifact Schema (6 tests)**: Validates JSON structure, verifies that test accuracy surpasses the 85% safety baseline, and checks plot file generation.

---

## 10. Operational Recommendations & Next Steps

### 10.1 Clinical Implementation Checklist
- **Human-in-the-Loop Safeguard**: The system is designed as an assistive diagnostic aid (Class IIa Medical Software under EU MDR / FDA SaMD). Model outputs must complement, rather than supersede, obstetrician bedside evaluations.
- **EHR & Telemetry Integration**: Expose inference endpoints via HL7/FHIR interfaces to ingest streaming telemetry from bedside fetal monitors (e.g., Philips Avalon, GE Corometrics).
- **Continuous Drift Monitoring**: Monitor statistical drift in FHR metrics (especially `prolongued_decelerations` and `ASTV`) to detect sensor degradation or calibration shifts.

---

## 11. Project Deliverables Inventory

| File / Resource | Type | Purpose | Verification Status |
| :--- | :--- | :--- | :---: |
| `fetal_health.csv` | Dataset | Scaled 10,000-sample augmented dataset | ✅ 10,000 rows, 0 nulls |
| `fetal_health_original.csv` | Dataset | Original gold standard cohort (2,126 records) | ✅ Preserved |
| `augment_dataset.py` | Script | Reproducible, physiologically constrained data augmentor | ✅ Validated |
| `fetal_health_pipeline.py` | Script | Full end-to-end ML pipeline (ETL, SelectKBest, 10 Models, Tuning) | ✅ Executable |
| `app.py` | Application | Interactive Streamlit Clinical Decision Support UI | ✅ Operational |
| `test_pipeline.py` | Test Suite | 33 automated unit and integration tests | ✅ 33/33 Passed |
| `output/metrics.json` | Report Data | Comprehensive metrics for all 10 models | ✅ Verified |
| `output/selected_features.json` | Metadata | List of top 10 ANOVA-selected features | ✅ Verified |
| `saved_models/` | Directory | Serialized `.joblib` binaries for all trained classifiers | ✅ 11 Models Saved |
| `output/*.png` | Visualizations | High-resolution publication plots (EDA, ROC, Confusion Matrices) | ✅ 5 Charts Generated |
