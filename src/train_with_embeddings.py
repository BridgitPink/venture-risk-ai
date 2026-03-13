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
DATA_PATH = Path("data/processed/real_startups_with_embeddings.parquet")

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "embedding_logreg.joblib"
METRICS_PATH = MODEL_DIR / "embedding_metrics.json"

DROP_COLUMNS = ["startup_name", "risk_score"]

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
]

CATEGORICAL_FEATURES = [
    "sector",
    "stage",
    "region",
    "business_model",
]

EMBEDDING_PREFIX = "emb_"


def load_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Embedding dataset not found at {data_path}. "
            "Run `python src/embedder.py` first."
        )
    return pd.read_parquet(data_path)


def get_embedding_columns(df: pd.DataFrame, prefix: str = EMBEDDING_PREFIX) -> List[str]:
    embedding_cols = [col for col in df.columns if col.startswith(prefix)]
    if not embedding_cols:
        raise ValueError(
            f"No embedding columns found with prefix '{prefix}'. "
            "Check that embedder.py completed successfully."
        )
    return sorted(embedding_cols, key=lambda x: int(x.replace(prefix, "")))


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    working_df = df.copy()

    # Drop columns that should not be used directly.
    for col in DROP_COLUMNS:
        if col in working_df.columns:
            working_df = working_df.drop(columns=col)

    embedding_columns = get_embedding_columns(working_df)
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + embedding_columns

    missing = [col for col in feature_columns + [TARGET_COLUMN] if col not in working_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = working_df[feature_columns]
    y = working_df[TARGET_COLUMN]
    return X, y, embedding_columns


def build_pipeline(embedding_columns: List[str]) -> Pipeline:
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

    embedding_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("emb", embedding_transformer, embedding_columns),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        max_iter=3000,
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

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            zero_division=0,
            output_dict=True,
        ),
    }


def try_load_baseline_metrics() -> dict | None:
    baseline_path = MODEL_DIR / "baseline_metrics.json"
    if baseline_path.exists():
        return json.loads(baseline_path.read_text(encoding="utf-8"))
    return None


def build_comparison_block(current_metrics: dict, baseline_metrics: dict | None) -> dict:
    comparison = {
        "embedding_model": {
            "accuracy": current_metrics["accuracy"],
            "precision": current_metrics["precision"],
            "recall": current_metrics["recall"],
            "f1": current_metrics["f1"],
            "roc_auc": current_metrics["roc_auc"],
        }
    }

    if baseline_metrics is not None:
        comparison["baseline_model"] = {
            "accuracy": baseline_metrics.get("accuracy"),
            "precision": baseline_metrics.get("precision"),
            "recall": baseline_metrics.get("recall"),
            "f1": baseline_metrics.get("f1"),
            "roc_auc": baseline_metrics.get("roc_auc"),
        }
        comparison["delta_vs_baseline"] = {
            "accuracy": round(current_metrics["accuracy"] - baseline_metrics.get("accuracy", 0.0), 6),
            "precision": round(current_metrics["precision"] - baseline_metrics.get("precision", 0.0), 6),
            "recall": round(current_metrics["recall"] - baseline_metrics.get("recall", 0.0), 6),
            "f1": round(current_metrics["f1"] - baseline_metrics.get("f1", 0.0), 6),
            "roc_auc": round(current_metrics["roc_auc"] - baseline_metrics.get("roc_auc", 0.0), 6),
        }

    return comparison


def save_artifacts(model: Pipeline, metrics: dict, comparison: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    comparison_path = MODEL_DIR / "model_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")


def print_summary(metrics: dict, comparison: dict) -> None:
    print("\nEmbedding-Enhanced Logistic Regression Results")
    print("-" * 48)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    if "delta_vs_baseline" in comparison:
        print("\nDelta vs Baseline:")
        print(
            f"Accuracy : {comparison['delta_vs_baseline']['accuracy']:+.4f}\n"
            f"Precision: {comparison['delta_vs_baseline']['precision']:+.4f}\n"
            f"Recall   : {comparison['delta_vs_baseline']['recall']:+.4f}\n"
            f"F1 Score : {comparison['delta_vs_baseline']['f1']:+.4f}\n"
            f"ROC-AUC  : {comparison['delta_vs_baseline']['roc_auc']:+.4f}"
        )


def main() -> None:
    df = load_dataset()
    X, y, embedding_columns = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    pipeline = build_pipeline(embedding_columns)
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(pipeline, X_test, y_test)
    baseline_metrics = try_load_baseline_metrics()
    comparison = build_comparison_block(metrics, baseline_metrics)

    save_artifacts(pipeline, metrics, comparison)
    print_summary(metrics, comparison)

    print(f"\nSaved embedding model to: {MODEL_PATH}")
    print(f"Saved embedding metrics to: {METRICS_PATH}")
    print(f"Saved comparison file to: {MODEL_DIR / 'model_comparison.json'}")


if __name__ == "__main__":
    main()