# project_physionet_hgb
Machine Learning project for sepsis prediction using clinical data and a HistGradientBoosting model. Includes preprocessing, missing-value handling, feature selection, class-imbalance treatment, model comparison, evaluation and threshold tuning. A Streamlit web app lets users enter measurements and view prediction results. For educational use only.
# physionet
ML-based Sepsis Prediction using HistGradientBoosting & Streamlit


## 1. Project Overview

Sepsis is a serious medical condition that can develop when the body's response to an infection causes tissue and organ damage. Early identification of sepsis can support timely medical intervention.

This project applies Machine Learning techniques to clinical data to build a binary classification model for sepsis prediction. The workflow covers data exploration, data cleaning, patient-level splitting, missing-value handling, correlation analysis, outlier treatment, class-imbalance handling, model training, validation, threshold tuning, final evaluation, and model deployment using Streamlit.

> **Note:** This project is developed for educational purposes only and is not intended for medical diagnosis, treatment, or clinical decision-making.

---

## 2. Project Objective

The main objective is to develop a Machine Learning model that can identify records associated with sepsis from clinical measurements.

The project focuses on:

- Exploring and cleaning a large clinical dataset.
- Preventing patient-level data leakage.
- Handling missing and invalid values.
- Removing features with extremely high missing rates.
- Reducing highly correlated features.
- Removing leakage-related features from the updated preprocessing.
- Handling extreme values using outlier clipping.
- Addressing severe class imbalance in the training data.
- Comparing different classification models.
- Selecting classification thresholds using validation data.
- Evaluating the models using metrics suitable for imbalanced classification.
- Analyzing influential features.
- Saving the trained model and preprocessing components for deployment.

---

## 3. Dataset

The project uses the PhysioNet Sepsis dataset available through OpenML.

### Dataset Information

| Property | Value |
|---|---:|
| OpenML Dataset ID | 46817 |
| Records | 1,552,210 |
| Original Columns | 44 |
| Target | `SepsisLabel` |
| Negative Class | 98.20% |
| Positive Class | 1.80% |

The target is binary:

| Value | Meaning |
|---|---|
| `0` | No Sepsis |
| `1` | Sepsis |

The dataset contains vital signs, laboratory measurements, patient information, ICU-related variables, and other clinical measurements.

Examples include:

- Heart Rate
- Pulse Oximetry
- Temperature
- Blood Pressure
- Respiration Rate
- pH
- Glucose
- Creatinine
- Hemoglobin
- Platelet Count
- Age
- Gender
- ICU-related information

A major challenge is the large amount of missing data, particularly in laboratory measurements.

---

## 4. Data Exploration

The dataset was first inspected to understand its structure and data quality.

The following checks were performed:

- Dataset shape
- Column names
- Data types
- Descriptive statistics
- Missing values
- Duplicate records
- Target distribution
- Invalid values

The initial dataset contained **1,552,210 rows** and **44 columns**.

No complete duplicate rows were found.

The target distribution confirmed severe class imbalance, with only about **1.8% positive records**.

---

## 5. Data Cleaning and Preprocessing

The preprocessing workflow was designed to avoid using information from the validation or test sets when fitting preprocessing steps.

### 5.1 Removing Identifier and Redundant Columns

The following columns were removed:

| Column | Reason |
|---|---|
| `Patient_ID` | Identifier, not a clinical predictor |
| `Unnamed: 0` | Redundant index-like column |
| `Hour` | Removed from the current modeling pipeline to avoid relying on the time index |
| `ICU_length_of_stay` | Removed to reduce potential information leakage |
| `Time_between_hospital_and_ICU_admission` | Removed to reduce potential information leakage |

The last two features were removed from the updated dataset before the final modeling workflow.

### 5.2 Handling Invalid Values

Clearly invalid values were converted to missing values before imputation.

Measurements such as the following were checked for invalid low values:

- Heart Rate
- Blood Pressure
- Temperature
- Respiration Rate
- Age

Binary variables were also checked to ensure that their values were restricted to `0` and `1`.

---

## 6. Patient-Level Train, Validation, and Test Split

Because the dataset contains multiple records for the same patient, a normal row-based split could cause data leakage.

Instead, patients were divided first and their records were assigned to the corresponding datasets.

### Split Strategy

| Stage | Proportion |
|---|---:|
| Train + Validation | 80% |
| Test | 20% |
| Training from Train + Validation | 80% |
| Validation from Train + Validation | 20% |

### Final Dataset Sizes

| Dataset | Records | Positive Records |
|---|---:|---:|
| Training | 993,338 | 17,605 |
| Validation | 248,879 | 4,439 |
| Test | 309,993 | 5,872 |

Patient IDs were checked to ensure there was no overlap between the three sets.

### Patient ID Overlap Check

| Comparison | Overlap |
|---|---:|
| Train ↔ Validation | 0 |
| Train ↔ Test | 0 |
| Validation ↔ Test | 0 |

This reduces the risk of patient-level data leakage.

---

## 7. Handling Missing Values

Missing-value analysis was performed using the training set.

Features with more than **95% missing values** were removed because they contained too little observed information to be reliably imputed.

### Features Removed

| Feature |
|---|
| Direct Bilirubin |
| Fibrinogen |
| Troponin I |
| Total Bilirubin |
| Alkaline Phosphatase |
| AST |
| Lactic Acid |
| Partial Thromboplastin Time |
| Arterial Oxygen Saturation |
| End-tidal CO₂ |
| Phosphate |
| Bicarbonate |
| Chloride |

This removed **13 features**, leaving **27 features** before correlation analysis.

The remaining missing values were handled using:

```python
SimpleImputer(
    strategy="median",
    add_indicator=True
)
```

The imputer was fitted only on the training data and then applied to validation and test data.

Missing-value indicators were also added to preserve information about whether a measurement was originally missing.

> **Note:** The exact number of final model features depends on the updated feature-selection and retraining workflow after removing the two leakage-related features.

---

## 8. Correlation Analysis

Correlation analysis was performed using temporary median imputation on the training data.

A threshold of:

```text
|correlation| > 0.95
```

was used to identify highly correlated features.

### Feature Removed

| Feature | Reason |
|---|---|
| `Administrative_identifier_for_ICU_unit_SICU_false_0_or_true` | Highly correlated with another ICU-related feature |

The feature selection procedure was then applied consistently to the validation and test data.

---

## 9. Outlier Handling

Instead of deleting complete rows containing extreme values, continuous numerical features were clipped using training-set quantiles.

### Clipping Parameters

| Parameter | Value |
|---|---:|
| Lower percentile | 0.5% |
| Upper percentile | 99.5% |

The clipping limits were calculated using the training data only and then applied to validation and test data.

Binary variables were excluded from outlier clipping.

This approach reduces the effect of extreme values while preserving clinical records.

---

## 10. Handling Class Imbalance

The original dataset contains approximately **98% negative** and **2% positive** records.

Training directly on this distribution can cause a model to favor the majority class.

Therefore, oversampling was applied **only to the training set**.

The positive class was randomly oversampled with replacement until it reached approximately one quarter of the majority-class size.

### Training Distribution

| Class | Before Oversampling | After Oversampling |
|---|---:|---:|
| No Sepsis | 975,733 | 975,733 |
| Sepsis | 17,605 | 243,933 |

The validation and test sets were not oversampled, so they retained the original class distribution.

---

## 11. Feature Scaling

The preprocessing strategy was kept model-specific.

Tree-based models such as Decision Tree, Random Forest, and HistGradientBoosting do not require standardization in the same way as linear models.

| Model | Scaling |
|---|---|
| Logistic Regression | Yes |
| Decision Tree | No |
| Random Forest | No |
| HistGradientBoosting | No |

The final deployed HGB pipeline uses its saved preprocessing artifacts rather than a separate StandardScaler artifact.

---

## 12. Machine Learning Models

Four classification models were compared.

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline linear model |
| Decision Tree | Non-linear decision rules |
| Random Forest | Ensemble of decision trees |
| HistGradientBoosting | Efficient gradient-boosting model for non-linear patterns |

### Random Forest Configuration

```python
RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
```

### HistGradientBoosting Configuration

```python
HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=42
)
```

HistGradientBoosting was selected as the final model.

---

## 13. Validation Strategy

Model selection was based on the validation set, not the test set.

The validation set was used to:

- Compare model performance.
- Evaluate Precision, Recall, and F1 Score.
- Compare ROC-AUC and PR-AUC.
- Select the classification threshold.

The test set remained untouched until the final evaluation.

This prevents the test results from influencing model selection.

---

## 14. Threshold Tuning

The default classification threshold is usually `0.50`.

Because this is a highly imbalanced classification problem, the default threshold may not provide a suitable balance between Precision and Recall.

Therefore, thresholds between `0.05` and `0.95` were evaluated using the validation set.

The threshold producing the highest validation F1 Score was selected for each model.

### Selected Thresholds

| Model | Validation Threshold | Validation F1 |
|---|---:|---:|
| Logistic Regression | 0.40 | 15.28% |
| Decision Tree | 0.55 | 15.60% |
| Random Forest | 0.10 | 15.32% |
| HistGradientBoosting | 0.55 | 18.03% |

> **Current supplied HGB artifact:** the saved threshold is approximately **0.5247474747**. If the HGB was retrained after the two-feature removal, the newly saved threshold should replace this value and the evaluation tables should be regenerated.

---

## 15. Evaluation Metrics

Because of the severe class imbalance, accuracy alone is not sufficient.

| Metric | Purpose |
|---|---|
| Precision | How many predicted positive records are actually positive |
| Recall | How many actual positive records are detected |
| F1 Score | Balance between Precision and Recall |
| ROC-AUC | Overall class-separation ability across thresholds |
| PR-AUC | Precision-Recall performance, especially useful for imbalanced data |
| Accuracy | Overall proportion of correct predictions |

---

## 16. Validation Results

The models were first compared using the validation set.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 96.10% | 11.90% | 18.52% | 14.49% | 73.58% | 7.82% |
| Decision Tree | 93.85% | 9.79% | 29.83% | 14.74% | 70.16% | 6.45% |
| Random Forest | 98.18% | 12.41% | 0.38% | 0.74% | 74.39% | 7.15% |
| HistGradientBoosting | 94.76% | 12.46% | 32.19% | 17.97% | 77.92% | 9.55% |

HistGradientBoosting provided the strongest overall validation performance, particularly in F1, ROC-AUC, and PR-AUC.

> **Important:** These metrics belong to the previously evaluated HGB training run. If the model was retrained after dropping the two leakage-related features, these values should be replaced with the new validation results.

---

## 17. Final Test Results

After selecting the thresholds using validation data, the models were evaluated on the untouched test set.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 94.32% | 11.30% | 28.20% | 16.14% | 74.06% | 7.82% |
| Decision Tree | 94.36% | 11.18% | 27.47% | 15.90% | 71.68% | 6.51% |
| Random Forest | 93.82% | 10.43% | 30.82% | 15.59% | 74.83% | 7.21% |
| HistGradientBoosting | 95.35% | 14.17% | 27.74% | 18.76% | 79.55% | 9.68% |

### Final Model

HistGradientBoosting was selected as the final model because it achieved the best overall balance across the main evaluation metrics.

| Metric | Result |
|---|---:|
| Accuracy | 95.35% |
| Precision | 14.17% |
| Recall | 27.74% |
| F1 Score | 18.76% |
| ROC-AUC | 79.55% |
| PR-AUC | 9.68% |

Random Forest achieved slightly higher Recall, but HistGradientBoosting achieved better Precision, F1, ROC-AUC, and PR-AUC.

> **Important:** The results above are the previously evaluated results. They should only be presented as final updated results if the same model was evaluated after the feature-removal change.

---

## 18. Overfitting Analysis

A Train vs Validation comparison was performed to identify potential overfitting.

Random Forest showed the largest performance gap.

### Random Forest: Training vs Validation

| Metric | Training | Validation |
|---|---:|---:|
| Precision | 99.93% | 10.16% |
| Recall | 100.00% | 31.13% |
| F1 Score | 99.96% | 15.32% |

The large gap indicates substantial overfitting to the oversampled training data.

This is one of the reasons Random Forest was not selected as the final model despite its relatively good recall.

---

## 19. Feature Importance

Feature importance for the final HistGradientBoosting model was analyzed using Permutation Importance on the validation set.

Permutation Importance measures how model performance changes when a feature is randomly shuffled.

It helps identify which variables the trained model relies on most.

> **Note:** Feature importance describes model behavior and does not establish clinical causation.

---

## 20. Exploratory Data Analysis

Several visualizations were created during the analysis:

- Missing-value percentage
- Class distribution
- Heart Rate distribution
- Heart Rate vs Respiration Rate
- Correlation heatmap
- Average Heart Rate over ICU time
- Precision-Recall curves
- Confusion Matrix
- Model F1 comparison

These visualizations were used to understand the dataset, identify preprocessing issues, and interpret model performance.

---

## 21. Final Model and Deployment

The final HistGradientBoosting model and required preprocessing components were saved for deployment.

### Saved Artifacts

| File | Description |
|---|---|
| `hgb.pkl` | Trained HistGradientBoosting model |
| `Imputer_hgb.pkl` | Fitted median imputer with missing-value indicators |
| `hgb_threshold.pkl` | Selected classification threshold for the HGB model |
| `feature_names_hgb.pkl` | Feature names used by the final HGB model |

These files allow the Streamlit application to load the trained model and apply the same preprocessing and feature structure used during training.

### Prediction Pipeline

```text
User Clinical Input
        ↓
Feature Preparation
        ↓
Saved Imputer
        ↓
HistGradientBoosting Model
        ↓
Predicted Probability
        ↓
Selected HGB Threshold
        ↓
Sepsis / No Sepsis
```

---

## 22. Streamlit Application

The trained model is integrated into a Streamlit application for interactive demonstration.

The application is organized into:

| Section | Purpose |
|---|---|
| Dashboard | Project overview and navigation |
| Patient Assessment | Clinical input fields |
| Risk Assessment | Model prediction and risk result |
| Prediction Records | Previous prediction records |
| Model Information | Model, dataset, preprocessing, and evaluation information |

The interface uses the supplied `doctor_patient.png` visual asset as part of the application design.

The application is intended only as a demonstration of Machine Learning deployment and should not be used for clinical decision-making.

---

## 23. Project Structure

The current project structure is:

```text
physionet/
│
├── app.py
├── doctor_patient.png
├── feature_names_hgb.pkl
├── hgb_threshold.pkl
├── hgb.pkl
├── Imputer_hgb.pkl
└── README.md
```

### File Description

| File | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `doctor_patient.png` | Image used in the application |
| `feature_names_hgb.pkl` | Saved feature names for the HGB model |
| `hgb_threshold.pkl` | Saved HGB classification threshold |
| `hgb.pkl` | Trained HistGradientBoosting model |
| `Imputer_hgb.pkl` | Saved imputer used for preprocessing |
| `README.md` | Project documentation |

---

## 24. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing and Machine Learning |
| OpenML | Dataset access |
| Joblib | Model and artifact saving |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |
| Streamlit | Model deployment |
| Google Colab | Development environment |

---

## 25. Limitations

- The dataset is highly imbalanced.
- Positive-class Precision and F1 Score remain relatively low.
- The dataset contains substantial missing data.
- Oversampling can increase the risk of overfitting.
- Model performance depends on the dataset and preprocessing strategy.
- Feature importance does not establish clinical causation.
- The test results should not be interpreted as clinical performance.
- The model is intended for educational purposes only.
- The Streamlit application should not be used for real clinical decisions.

---

## 26. Conclusion

This project demonstrates an end-to-end Machine Learning workflow for predicting sepsis from a large and highly imbalanced clinical dataset.

The workflow included:

```text
Data Exploration
        ↓
Data Cleaning
        ↓
Patient-Level Split
        ↓
Missing-Value Handling
        ↓
Correlation Filtering
        ↓
Leakage-Related Feature Removal
        ↓
Outlier Clipping
        ↓
Training-Only Oversampling
        ↓
Model Training
        ↓
Validation
        ↓
Threshold Tuning
        ↓
Final Test Evaluation
        ↓
Deployment
```

Four models were compared:

- Logistic Regression
- Decision Tree
- Random Forest
- HistGradientBoosting

The validation set was used for model comparison and threshold selection, while the test set remained separate for final evaluation.

HistGradientBoosting achieved the best overall performance in the previously evaluated experiment.

The project also demonstrates why accuracy alone can be misleading in highly imbalanced medical classification problems. Metrics such as Precision, Recall, F1 Score, ROC-AUC, and PR-AUC provide a more meaningful view of model performance.

Overall, the project highlights the practical challenges of applying Machine Learning to clinical data, including missing values, class imbalance, patient-level leakage, outliers, threshold selection, and overfitting.

> **Educational Use Disclaimer:** This project is intended for educational and demonstration purposes only. It is not a medical diagnostic system and should not be used to make real clinical decisions.
