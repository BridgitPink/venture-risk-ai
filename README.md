# Venture Risk AI — Startup Failure Prediction

Machine learning pipeline that predicts whether a startup will **fail within 24 months** using venture signals, enriched startup datasets, and interpretable models.

This project demonstrates how **structured venture data + external startup datasets + explainable machine learning** can be used to estimate startup survival risk.

⚠️ This project is intended for **research and educational purposes only** and does not provide investment advice.

---

# Project Overview

Startup failure prediction is difficult due to limited data, noisy signals, and survivorship bias.

This project builds a reproducible ML pipeline that:

1. Cleans and labels startup outcome data
2. Merges external venture datasets (~125k startups)
3. Enriches the labeled startup dataset
4. Trains a failure prediction model
5. Generates interpretable explanations for predictions

The model predicts:

```
failed_within_24_months
```

for each startup.

---

# Example Output

Example prediction output from `predict.py`:

```
Startup: SignalForge AI

Failure probability: 0.31
Survival probability: 0.69
Risk band: Medium

Top risk signals
- Limited funding rounds
- Early stage company

Positive signals
- Strong investor participation
- High milestone completion
```

---

# Machine Learning Pipeline

```
Raw Startup Dataset
        ↓
Data Cleaning
        ↓
External Dataset Merge
        ↓
Feature Engineering
        ↓
Logistic Regression Model
        ↓
Failure Probability
        ↓
Explainability (SHAP)
```

---

# Model Performance

Final model performance on the test set:

| Metric | Score |
|------|------|
Accuracy | **0.735**
Precision | **0.589**
Recall | **0.815**
F1 Score | **0.684**
ROC-AUC | **0.810**

The model intentionally prioritizes **recall for failing startups**, which is valuable in venture-risk analysis.

Confusion matrix:

```
[[83, 37],
 [12, 53]]
```

---

# Features Used

### Numeric Signals

- funding_total_usd
- funding_rounds
- milestones
- relationships
- avg_participants
- funding_per_round_usd
- has_vc
- has_angel
- has_roundA
- has_roundB
- has_roundC
- has_roundD
- is_top500
- latitude
- longitude
- external_founded_year
- external_funding_total_usd
- external_funding_rounds
- external_startup_age_years
- has_external_match

### Categorical Signals

- sector
- stage
- region
- business_model
- external_sector
- external_region

---

# Explainable AI (SHAP)

The project integrates **SHAP (SHapley Additive exPlanations)** to interpret model predictions.

SHAP helps explain:

- which features increased failure risk
- which features reduced failure risk
- how much each signal contributed to the prediction

Example insights:

- strong investor relationships reduce failure risk
- higher milestone completion correlates with survival
- certain regions or sectors may have higher failure rates

Generated explanation artifacts are saved to:

```
outputs/explanations/
```

---

# External Data Enrichment

The labeled startup dataset is enriched using external venture datasets containing **125k+ startup records**.

External features include:

- founding year
- funding totals
- funding rounds
- startup age
- sector
- region

These signals significantly improve model performance compared to using the raw dataset alone.

---

# Repository Structure

```
venture-risk-ai

src/
  prepare_real_data.py
  merge_external_datasets.py
  enrich_labeled_data.py
  train.py
  predict.py
  explain.py
  run_pipeline.py

data/
  processed/

models/
  startup_risk_metrics.json
  startup_risk_feature_importance.csv
  startup_risk_test_predictions.csv

outputs/
  explanations/

README.md
requirements.txt
```

---

# Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/venture-risk-ai.git
cd venture-risk-ai
```

Create a virtual environment:

```
python -m venv .venv
```

Activate the environment:

Windows:
```
.venv\Scripts\activate
```

Mac/Linux:
```
source .venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Running the Pipeline

Run the full pipeline:

```
python src/run_pipeline.py
```

Pipeline steps:

```
prepare_real_data.py
merge_external_datasets.py
enrich_labeled_data.py
train.py
```

Outputs:

```
models/startup_risk_metrics.json
models/startup_risk_feature_importance.csv
models/startup_risk_test_predictions.csv
```

---

# Making Predictions

```
python src/predict.py
```

Prediction output is saved to:

```
outputs/prediction.json
```

---

# Why Logistic Regression?

The project uses logistic regression because it is:

- interpretable
- stable on tabular venture data
- easy to explain with SHAP
- a strong baseline for structured features

Future models may include:

- XGBoost
- LightGBM
- Gradient Boosting
- Hybrid ML architectures

---

# Future Improvements

Possible extensions:

### Interactive Dashboard
Build a **Streamlit interface** where users can:

- enter startup metrics
- see predictions
- view feature explanations

### Feature Expansion
Add additional venture signals such as:

- founder background features
- funding stage timelines
- investor network graphs

### Advanced Models
Evaluate more advanced models:

- XGBoost
- LightGBM
- ensemble models

---
# Key Insights from the Model

Beyond predicting startup failure, the model reveals several patterns commonly associated with startup survival and failure.

These insights come from the model’s **feature coefficients and SHAP explanations**.

### Strong Signals of Startup Survival

Several features were consistently associated with **lower failure risk**:

- **Investor relationships**
  
  Startups with stronger investor participation and network relationships were less likely to fail.

- **Milestone completion**

  Companies that consistently achieved operational milestones showed significantly better survival rates.

- **Higher funding per round**

  Larger average funding rounds often indicate stronger investor confidence and better capital efficiency.

These features likely reflect **organizational maturity and market validation**.

---

### Signals Associated with Higher Failure Risk

The model identified several patterns correlated with increased failure probability:

- **Early-stage startups**

  Companies at earlier stages have higher uncertainty and failure rates.

- **Limited funding rounds**

  Fewer funding rounds often correlate with limited investor confidence or stalled growth.

- **Certain regional startup ecosystems**

  Regional startup ecosystems can affect survival probability due to differences in funding availability and network effects.

---

### Venture Interpretation

These patterns align with observations from venture capital research:

- **Investor network strength** improves access to capital and strategic guidance.
- **Milestone execution** signals operational discipline.
- **Capital efficiency** improves runway and growth sustainability.

While the model is purely statistical, its signals reflect **real venture ecosystem dynamics**.

---

### Example Feature Importance

Top signals identified by the model include:

```
relationships
milestones
funding_per_round_usd
region
sector
```

These features had the strongest influence on predicted startup outcomes.

---

### Why This Matters

Startup prediction models are rarely perfect due to the complexity of innovation and markets.

However, models like this can help:

- identify **early warning signals**
- prioritize **startup due diligence**
- support **venture portfolio risk analysis**

This project demonstrates how **machine learning and explainable AI can assist venture decision workflows**.

---

# Disclaimer

This project is for **educational and portfolio purposes only**.

The model should not be used for real investment decisions.

---

# Author

Machine learning portfolio project exploring:

- startup analytics
- venture risk modeling
- explainable AI

---

# License

MIT License