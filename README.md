# Postpartum Depression Prediction

## Overview
This project explores machine learning approaches to predict **postpartum depression** outcomes using maternal and infant health data.  

The target variable is the **Hamilton Depression Rating Scale at 6 months postpartum (`hamd_6m`)**, a widely used clinical measure of depression severity.  
- **Scale**: Higher scores indicate more severe depressive symptoms.  
- **Importance**: Predicting this score early enables timely intervention, improving maternal mental health and child development outcomes.  
- **Task Type**: Since `hamd_6m` is continuous, the task is framed as a **regression problem**.  

The goal is to build a reliable model while preventing data leakage, handling missing values, and selecting strong predictors.

---

## SheCode Africa ML/AI Challenge

### Challenge Theme
The competition focuses on **Postpartum Depression Prediction** using supervised learning (regression) techniques.  
- Dataset is provided by the organizers.  
- No external data is allowed unless explicitly approved.  
- Challenge duration: **Thursday, 21st August 2025 – Monday, 15th September 2025**.  

### Dataset
- **Dataset**: Postpartum depression prediction from demographic data, birth complications, social support scores, and medical history.  
- **Dependent Variable**: Hamilton Depression Rating (`hamd_6m`).  
- **Dataset Link**: [Download here](https://drive.google.com/file/d/1b9479YMBAOlU-lIP0wL2wqouuxt1oYi6/view?usp=sharing)  
- **Dataset Schema**: [View schema](https://drive.google.com/file/d/1qGD13ErsDcvzwE4IFsWo30SaagZNBYFn/view?usp=sharing)  

---

## Methodology

### 1. Data Preprocessing
- **Target Definition**: Regression task with continuous outcome (`hamd_6m`).  
- **Missing Value Handling**:
  - Dropped columns with >50% missing values.  
  - Dropped rows with >60% missing values.  
  - Numeric columns imputed with **median**; categorical columns imputed with **mode**.  
  - Added **missingness indicator flags** to capture patterns in missing data.  

- **Leakage Prevention**:
  - Removed post-target variables (`HAZ_6`, `WAZ_12`) to avoid leakage.  
  - Retained strong predictors (`var611`, `HAZ_12`, `infantdev_f`).  

### 2. Feature Engineering
- Applied **One-Hot Encoding** for categorical variables.  
- Assessed feature importance using **Chi-Square tests** for categorical variables and model-based importance for continuous features.  

---

## Models Evaluated
1. ElasticNet (baseline linear model)  
2. LightGBM (gradient boosting)  
3. CatBoost (boosting with categorical support)  
4. RandomForest (bagging ensemble)  
5. Stacked Ensemble (LightGBM + RandomForest with ElasticNet as meta-learner)  

---

## Results

| Model                | MAE (↓) | RMSE (↓) | R² (↑) |
|-----------------------|---------|----------|--------|
| ElasticNet (Baseline) | 2.87 ± 0.18 | 3.99 ± 1.15 | 0.53 ± 0.03 |
| LightGBM             | 1.99 ± 0.14 | 3.10 ± 0.91 | 0.72 ± 0.03 |
| CatBoost             | 2.12 ± 0.09 | 3.20 ± 0.56 | 0.70 ± 0.01 |
| RandomForest         | 1.99 ± 0.09 | 3.09 ± 0.97 | 0.72 ± 0.03 |
| Stacked Ensemble     | 1.95 ± 0.11 | 3.04 ± 0.96 | 0.73 ± 0.03 |

---

## Key Insights
- ElasticNet underperformed, confirming linear models are insufficient for this task.  
- Tree-based models (LightGBM, RandomForest, CatBoost) achieved strong predictive accuracy and explained variance.  
- Stacking LightGBM and RandomForest improved performance further, offering the best trade-off between error reduction and stability.  
- Incremental improvement from stacking (MAE ↓ 0.04, R² ↑ 0.01) is modest but meaningful.  

---

## Conclusion
- **Recommended model for deployment**: Stacked Ensemble (LightGBM + RandomForest).  
- If interpretability is important: RandomForest may be preferred due to easier feature importance extraction.  
- ElasticNet should only serve as a lightweight baseline.  

---
