# src/train_embedding_logreg.py

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "real_startups_model_ready.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "failed_within_24_months"
RANDOM_STATE = 42
TEST_SIZE = 0.20

NUMERIC_CANDIDATES = [
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

CATEGORICAL_CANDIDATES = [
    "sector",
    "stage",
    "region",
    "business_model",
]

TEXT_CANDIDATES = [
    "description",
    "founder_bios",
    "recent_update",
]

LEAKAGE_TERMS_REGEX = r"acquired|closed|failed|bankrupt|ipo|shutdown|shut down|dead|inactive"


class TextCombiner(BaseEstimator, TransformerMixin):
    def __init__(self, text_columns: list[str]):
        self.text_columns = text_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            data = X.copy()
        else:
            data = pd.DataFrame(X, columns=self.text_columns)

        combined = pd.Series("", index=data.index, dtype="object")
        for col in self.text_columns:
            if col in data.columns:
                combined = combined + " " + data[col].fillna("").astype(str)

        return combined.str.strip().to_numpy()


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def print_metrics(title: str, metrics: dict) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))


def run_leakage_audit(df: pd.DataFrame, cols: list[str], label: str) -> None:
    print(f"\nLeakage audit: {label}")
    found_any = False

    for col in cols:
        if col not in df.columns:
            continue

        series = df[col].fillna("").astype(str)
        match_count = int(series.str.contains(LEAKAGE_TERMS_REGEX, case=False, regex=True, na=False).sum())

        if match_count > 0:
            found_any = True
            print(f"  - {col}: {match_count} potential matches")

    if not found_any:
        print("  None")


def print_dataset_summary(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str], text_cols: list[str]) -> None:
    print("\nDataset summary")
    print("---------------")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nTarget distribution:")
    print(df[TARGET_COL].value_counts(dropna=False))

    print("\nNumeric columns considered:")
    print(numeric_cols)

    print("\nCategorical columns considered:")
    print(categorical_cols)

    print("\nText columns considered:")
    print(text_cols)


def save_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL].isin([0, 1])].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    if len(df) == 0:
        raise ValueError(
            "Training dataset is empty after filtering valid target rows."
        )

    numeric_cols = [c for c in NUMERIC_CANDIDATES if c in df.columns]
    categorical_cols = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]
    text_cols = [c for c in TEXT_CANDIDATES if c in df.columns]

    print_dataset_summary(df, numeric_cols, categorical_cols, text_cols)
    run_leakage_audit(df, categorical_cols, "categorical columns")
    run_leakage_audit(df, text_cols, "text columns")

    X = df[numeric_cols + categorical_cols + text_cols].copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTrain/Test split")
    print("----------------")
    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")
    print("y_train distribution:")
    print(y_train.value_counts(dropna=False))
    print("y_test distribution:")
    print(y_test.value_counts(dropna=False))

    all_null_numeric = [c for c in numeric_cols if X_train[c].notna().sum() == 0]
    all_null_categorical = [c for c in categorical_cols if X_train[c].notna().sum() == 0]
    all_blank_text = [
        c for c in text_cols
        if X_train[c].fillna("").astype(str).str.strip().eq("").all()
    ]

    numeric_cols = [c for c in numeric_cols if c not in all_null_numeric]
    categorical_cols = [c for c in categorical_cols if c not in all_null_categorical]
    text_cols = [c for c in text_cols if c not in all_blank_text]

    print("\nDropped all-null numeric columns:")
    print(all_null_numeric if all_null_numeric else "None")

    print("\nDropped all-null categorical columns:")
    print(all_null_categorical if all_null_categorical else "None")

    print("\nDropped all-blank text columns:")
    print(all_blank_text if all_blank_text else "None")

    print("\nUsing numeric columns:")
    print(numeric_cols)

    print("\nUsing categorical columns:")
    print(categorical_cols)

    print("\nUsing text columns:")
    print(text_cols)

    if not numeric_cols and not categorical_cols:
        raise ValueError("No structured features available for baseline model.")

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    baseline_transformers = []
    if numeric_cols:
        baseline_transformers.append(("num", num_pipe, numeric_cols))
    if categorical_cols:
        baseline_transformers.append(("cat", cat_pipe, categorical_cols))

    baseline_preprocessor = ColumnTransformer(
        transformers=baseline_transformers,
        remainder="drop",
    )

    baseline_model = Pipeline([
        ("prep", baseline_preprocessor),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])

    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    baseline_proba = baseline_model.predict_proba(X_test)[:, 1]
    baseline_metrics = compute_metrics(y_test, baseline_pred, baseline_proba)
    print_metrics("Structured Baseline Logistic Regression Results", baseline_metrics)

    text_metrics = None
    text_model = None

    if text_cols:
        text_pipe = Pipeline([
            ("combine", TextCombiner(text_cols)),
            ("tfidf", TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                stop_words="english",
                min_df=2,
            )),
        ])

        text_transformers = []
        if numeric_cols:
            text_transformers.append(("num", num_pipe, numeric_cols))
        if categorical_cols:
            text_transformers.append(("cat", cat_pipe, categorical_cols))
        text_transformers.append(("text", text_pipe, text_cols))

        text_preprocessor = ColumnTransformer(
            transformers=text_transformers,
            remainder="drop",
        )

        text_model = Pipeline([
            ("prep", text_preprocessor),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ])

        text_model.fit(X_train, y_train)
        text_pred = text_model.predict(X_test)
        text_proba = text_model.predict_proba(X_test)[:, 1]
        text_metrics = compute_metrics(y_test, text_pred, text_proba)
        print_metrics("Text-Enhanced Logistic Regression Results", text_metrics)

        delta = {
            "accuracy": text_metrics["accuracy"] - baseline_metrics["accuracy"],
            "precision": text_metrics["precision"] - baseline_metrics["precision"],
            "recall": text_metrics["recall"] - baseline_metrics["recall"],
            "f1_score": text_metrics["f1_score"] - baseline_metrics["f1_score"],
            "roc_auc": text_metrics["roc_auc"] - baseline_metrics["roc_auc"],
        }

        print("\nDelta vs Baseline:")
        print(f"Accuracy : {delta['accuracy']:+.4f}")
        print(f"Precision: {delta['precision']:+.4f}")
        print(f"Recall   : {delta['recall']:+.4f}")
        print(f"F1 Score : {delta['f1_score']:+.4f}")
        print(f"ROC-AUC  : {delta['roc_auc']:+.4f}")
    else:
        delta = None
        print("\nSkipping text-enhanced model because no usable text columns were available.")

    joblib.dump(baseline_model, MODELS_DIR / "baseline_logreg.joblib")
    print(f"\nSaved baseline model to: {MODELS_DIR / 'baseline_logreg.joblib'}")

    if text_model is not None:
        joblib.dump(text_model, MODELS_DIR / "embedding_logreg.joblib")
        print(f"Saved text-enhanced model to: {MODELS_DIR / 'embedding_logreg.joblib'}")

    save_json(MODELS_DIR / "baseline_metrics.json", baseline_metrics)
    print(f"Saved baseline metrics to: {MODELS_DIR / 'baseline_metrics.json'}")

    if text_metrics is not None:
        save_json(MODELS_DIR / "embedding_metrics.json", text_metrics)
        print(f"Saved embedding metrics to: {MODELS_DIR / 'embedding_metrics.json'}")

    comparison_payload = {
        "baseline": baseline_metrics,
        "embedding_model": text_metrics,
        "delta_vs_baseline": delta,
        "used_numeric_columns": numeric_cols,
        "used_categorical_columns": categorical_cols,
        "used_text_columns": text_cols,
        "dropped_all_null_numeric_columns": all_null_numeric,
        "dropped_all_null_categorical_columns": all_null_categorical,
        "dropped_all_blank_text_columns": all_blank_text,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }
    save_json(MODELS_DIR / "model_comparison.json", comparison_payload)
    print(f"Saved comparison file to: {MODELS_DIR / 'model_comparison.json'}")


if __name__ == "__main__":
    main()