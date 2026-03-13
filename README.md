# Startup Failure Predictor AI

AI system that predicts the probability a startup will fail within 24 months using **venture signals, machine learning, semantic embeddings, and explainable AI (SHAP)**.

This project demonstrates how **structured business data + NLP embeddings + interpretable ML** can be combined to estimate startup survival risk.

The goal of the project is to showcase **practical ML system design and explainability**, not to provide real investment advice.

---

# Project Overview

Startups fail for many reasons:

- weak product–market fit
- limited runway
- poor retention
- inefficient capital usage
- inexperienced founding teams

This project builds a model that analyzes a startup profile and produces:

- Failure probability
- Survival probability
- Risk band
- Top risk factors
- Positive signals
- Strategic recommendations

The system mimics the reasoning process a **venture analyst or investment associate** might use when screening startups.

---

# Example Output

```
Startup: SignalForge AI
Model: embedding
Predicted outcome: survive_24_months
Failure probability: 0.31
Risk band: medium

Top risk factors
- Runway is moderate and may become a constraint if growth stalls

Positive signals
- Presence of a technical founder
- Strong customer growth

Recommendations
- Improve capital efficiency
- Monitor churn and retention metrics
```

---

# Architecture

```
Startup Profile
      ↓
Feature Engineering
      ↓
Embedding Generation (Sentence Transformers)
      ↓
Machine Learning Model
      ↓
Failure Probability
      ↓
Reasoning Engine
      ↓
Human‑Readable Risk Report
      ↓
SHAP Explainability
```

Two model pipelines are implemented.

### Baseline Model

Uses **structured startup signals only**.

Purpose:

- establish a benchmark
- demonstrate experiment design

### Embedding Model

Uses:

- structured signals
- semantic embeddings from startup text

This allows the model to learn patterns in:

- startup descriptions
- founder bios
- investor updates

---

# Explainable AI (SHAP)

The project integrates **SHAP (SHapley Additive exPlanations)** to make model predictions interpretable.

SHAP explains:

- which features increased failure risk
- which features reduced failure risk
- how much each feature contributed to the final prediction

Outputs include:

- global feature importance
- beeswarm distribution plots
- individual startup explanations (waterfall plots)

Example explanation insights:

- limited runway increased failure probability
- high churn increased risk
- strong revenue growth reduced risk
- presence of a technical founder reduced risk

Generated explanation artifacts:

```
outputs/explanations/

baseline_shap_global_bar.png
baseline_shap_beeswarm.png
baseline_shap_waterfall_row_0.png

embedding_shap_global_bar.png
embedding_shap_beeswarm.png
embedding_shap_waterfall_row_0.png
```

---

# Features Used

## Numeric Signals

- funding_total_usd
- monthly_burn_usd
- runway_months
- revenue_growth_pct
- customer_growth_pct
- churn_pct
- burn_multiple
- annual_revenue_run_rate
- founder_experience_years
- team_size

## Categorical Signals

- sector
- stage
- region
- business_model

## Text Signals

Embedded using **Sentence Transformers**.

- startup description
- founder bios
- investor updates

Embedding model:

```
all-MiniLM-L6-v2
```

---

# Dataset

Real startup failure datasets are difficult to obtain and often incomplete.

This project uses a **synthetic but realistic startup dataset generator**.

The generator creates startup profiles containing:

- sector
- stage
- founder experience
- team size
- funding
- burn
- runway
- revenue growth
- churn
- founder bios
- startup description

A synthetic risk scoring function produces the label:

```
failed_within_24_months
```

Generated datasets:

```
data/processed/synthetic_startups.csv
```

Embedding dataset:

```
data/processed/synthetic_startups_with_embeddings.parquet
```

---

# Repository Structure

```
startup-failure-predictor-ai

src/
  data_gen.py
  embedder.py
  train.py
  train_with_embeddings.py
  predict.py
  explain.py


data/
  processed/

models/

outputs/
  explanations/

README.md
requirements.txt
```

---

# Installation

Clone the repository

```
git clone https://github.com/yourusername/startup-failure-predictor-ai.git
cd startup-failure-predictor-ai
```

Create virtual environment

```
python -m venv venv
source venv/bin/activate
```

Install dependencies

```
pip install -r requirements.txt
```

---

# Usage

## Step 1 — Generate Synthetic Dataset

```
python src/data_gen.py
```

Output:

```
data/processed/synthetic_startups.csv
```

---

## Step 2 — Generate Text Embeddings

```
python src/embedder.py
```

Outputs:

```
data/processed/synthetic_startups_with_embeddings.parquet
```

```
data/processed/text_embeddings.npy
```

---

## Step 3 — Train Baseline Model

```
python src/train.py
```

Artifacts:

```
models/baseline_logreg.joblib
models/baseline_metrics.json
```

---

## Step 4 — Train Embedding Model

```
python src/train_with_embeddings.py
```

Artifacts:

```
models/embedding_logreg.joblib
models/embedding_metrics.json
models/model_comparison.json
```

---

## Step 5 — Generate Model Explanations

```
python src/explain.py --mode embedding
```

Example:

```
python src/explain.py --mode baseline --row-index 0
```

Explanation artifacts saved to:

```
outputs/explanations/
```

---

## Step 6 — Make Predictions

Default (embedding model):

```
python src/predict.py
```

Explicit modes:

Baseline model

```
python src/predict.py --mode baseline
```

Embedding model

```
python src/predict.py --mode embedding
```

Prediction output:

```
outputs/prediction.json
```

---

# Model Choice

Baseline model:

```
Logistic Regression
```

Reasons:

- interpretable
- stable baseline
- strong for tabular signals

Embedding model combines:

- structured venture signals
- semantic startup descriptions

Future models may include:

- XGBoost
- LightGBM
- neural architectures

---

# Future Improvements

Planned upgrades:

### Interactive Dashboard

Build a **Streamlit interface** where users can:

- enter startup metrics
- see predictions
- view feature explanations

### Vector Similarity Search

Compare startups with similar historical profiles using embedding similarity.

### Advanced Models

- XGBoost
- LightGBM
- hybrid neural models

---

# Disclaimer

This project is for **educational and portfolio purposes only**.

Predictions are generated using **synthetic data** and should not be used for investment decisions.

---

# Author

Machine learning portfolio project exploring:

- startup analytics
- venture risk modeling
- explainable AI

---

# License

MIT License
