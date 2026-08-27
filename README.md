# Explainable Scholarship Eligibility & Recommendation System

An ML-powered system that predicts scholarship eligibility, explains *why* a decision was made using SHAP, cross-checks predictions against explicit scholarship rules, and recommends alternative scholarships when a user doesn't qualify.

## Overview

Government and institutional scholarship schemes often lack transparency in how eligibility decisions are made. This project addresses that gap by combining:
- A trained **XGBoost classifier** to predict eligibility from applicant profile data
- **SHAP (SHapley Additive exPlanations)** to show which features drove each individual prediction
- A **rule-verification layer** that checks predictions against each scholarship's explicit criteria
- An **alternative recommendation engine** that suggests other scholarships the applicant may qualify for

## Tech Stack

- **Modeling:** XGBoost
- **Explainability:** SHAP
- **Data processing:** Pandas
- **Interface:** Gradio

## Dataset

Trained on a balanced dataset of ~200,000 applicant records, cleaned down to ~74,000 valid rows after handling missing values in fields like education qualification, religion, and annual percentage. Features include education level, gender, community, religion, ex-servicemen status, disability status, sports participation, academic percentage bracket, income bracket, and study location.

## Model Performance

| Metric | Score |
|---|---|
| Accuracy | 78.65% |
| Precision | 74.87% |
| Recall | 86.18% |

## How It Works

1. User selects a scholarship and fills in their profile via the Gradio interface
2. The XGBoost model predicts eligibility
3. SHAP explains the top contributing features behind that prediction
4. A rule-verification step checks the prediction against the scholarship's explicit requirements
5. If ineligible (by rule or by model), the system suggests alternative scholarships the user may qualify for

## Running It

```bash
pip install pandas gradio xgboost scikit-learn shap
python app.py
```

Place your dataset CSV in the working directory before running (update the file path in the script if needed).

## Notes

This project was built as part of ongoing research into transparency and interpretability in state scholarship recommendation systems.
