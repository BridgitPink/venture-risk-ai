from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "failed_within_24_months"
DROP_COLUMNS = ["startup_name", "risk_score"]
MODEL_DIR = Path("models")
DATA_PATH = Path("data/processed/real_startups_model_ready.csv")
METRICS_PATH = MODEL_DIR / "baseline_metrics.json"
MODEL_PATH = MODEL_DIR / "baseline_logreg.joblib"


NUMERIC_FEATURES = [
    "founded_year",
    "team_size",
    "founder_count",
    "founder_experience_years",
    "has_technical_founder",
    "funding_total_usd",
    "monthly_burn_usd",
    "runway_months",
    "revenue_growth_pct",
    "customer_growth_pct",
    "churn_pct",
    "burn_multiple",
    "annual_revenue_run_rate",
    "funding_rounds",
    "milestones",
    "relationships",
    "avg_participants",
    "company_age_years",
    "funding_per_round_usd",
    "milestones_per_year",
    "years_to_first_funding",
    "years_to_last_funding",
    "has_vc",
    "has_angel",
    "has_roundA",
    "has_roundB",
    "has_roundC",
    "has_roundD",
    "is_top500",
    "latitude",
    "longitude",
]

CATEGORICAL_FEATURES = [
    "sector",
    "stage",
    "region",
    "business_model",
]

TEXT_FEATURES = [
    "description",
    "founder_bios",
    "recent_update",
]


def load_dataset(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Run `python src/data_gen.py` first."
        )
    return pd.read_csv(csv_path)


def combine_text_columns(df: pd.DataFrame, text_columns: List[str]) -> pd.Series:
    return (
        df[text_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    working_df = df.copy()
    working_df["combined_text"] = combine_text_columns(working_df, TEXT_FEATURES)

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["combined_text"]
    X = working_df[feature_columns]
    y = working_df[TARGET_COLUMN]
    return X, y


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Phase 1 baseline:
    # We intentionally ignore raw text here so the first model is a clean
    # structured-data baseline. Embeddings can be added in the next phase.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
    }
    return metrics


def save_artifacts(model: Pipeline, metrics: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def print_summary(metrics: dict) -> None:
    print("\nBaseline Logistic Regression Results")
    print("-" * 42)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])


def main() -> None:
    df = load_dataset()

    # Safety check: drop columns that should never be used directly as features.
    df = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns])

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(pipeline, X_test, y_test)
    save_artifacts(pipeline, metrics)
    print_summary(metrics)

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
