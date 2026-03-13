from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap


BASELINE_MODEL_PATH = Path("models/baseline_logreg.joblib")
EMBEDDING_MODEL_PATH = Path("models/embedding_logreg.joblib")

BASELINE_DATA_PATH = Path("data/processed/synthetic_startups.csv")
EMBEDDING_DATA_PATH = Path("data/processed/synthetic_startups_with_embeddings.parquet")

OUTPUT_DIR = Path("outputs/explanations")

TARGET_COLUMN = "failed_within_24_months"
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

TEXT_FEATURES = [
    "description",
    "founder_bios",
    "recent_update",
]


def load_dataset(mode: str) -> pd.DataFrame:
    data_path = EMBEDDING_DATA_PATH if mode == "embedding" else BASELINE_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    if data_path.suffix == ".parquet":
        return pd.read_parquet(data_path)
    return pd.read_csv(data_path)


def load_model(mode: str):
    model_path = EMBEDDING_MODEL_PATH if mode == "embedding" else BASELINE_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


def combine_text_columns(df: pd.DataFrame, text_columns: List[str]) -> pd.Series:
    return (
        df[text_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def get_embedding_columns(df: pd.DataFrame) -> List[str]:
    cols = [col for col in df.columns if col.startswith("emb_")]
    return sorted(cols, key=lambda x: int(x.replace("emb_", "")))


def prepare_features(df: pd.DataFrame, mode: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    working_df = df.copy()

    for col in DROP_COLUMNS:
        if col in working_df.columns:
            working_df = working_df.drop(columns=col)

    if mode == "baseline":
        working_df["combined_text"] = combine_text_columns(working_df, TEXT_FEATURES)
        feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["combined_text"]
    else:
        embedding_columns = get_embedding_columns(working_df)
        feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + embedding_columns

    missing = [col for col in feature_columns + [TARGET_COLUMN] if col not in working_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = working_df[feature_columns]
    y = working_df[TARGET_COLUMN]
    return X, y, feature_columns


def transform_features(pipeline, X: pd.DataFrame):
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X)

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
    return X_transformed_df


def build_explainer(model_pipeline, X_transformed: pd.DataFrame):
    classifier = model_pipeline.named_steps["model"]
    explainer = shap.Explainer(classifier, X_transformed)
    shap_values = explainer(X_transformed)
    return explainer, shap_values


def save_global_importance_plot(shap_values, X_transformed: pd.DataFrame, output_path: Path) -> None:
    plt.figure()
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()


def save_beeswarm_plot(shap_values, output_path: Path) -> None:
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()


def save_waterfall_plot(shap_values, row_index: int, output_path: Path) -> None:
    plt.figure()
    shap.plots.waterfall(shap_values[row_index], max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()


def extract_local_explanation(
    shap_values,
    X_transformed: pd.DataFrame,
    original_df: pd.DataFrame,
    row_index: int,
    top_k: int = 10,
) -> dict:
    row_values = shap_values[row_index].values
    row_feature_names = list(X_transformed.columns)

    contributions = [
        {
            "feature": feature,
            "shap_value": float(value),
            "abs_shap_value": float(abs(value)),
        }
        for feature, value in zip(row_feature_names, row_values)
    ]
    contributions.sort(key=lambda item: item["abs_shap_value"], reverse=True)

    startup_name = (
        original_df.iloc[row_index]["startup_name"]
        if "startup_name" in original_df.columns
        else f"row_{row_index}"
    )

    predicted_base_value = float(shap_values[row_index].base_values)
    predicted_log_odds = predicted_base_value + float(row_values.sum())

    return {
        "row_index": int(row_index),
        "startup_name": startup_name,
        "base_value": predicted_base_value,
        "approx_model_output": predicted_log_odds,
        "top_contributors": contributions[:top_k],
    }


def save_summary_json(summary: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SHAP explanations for startup failure models.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "embedding"],
        default="embedding",
        help="Which trained model to explain.",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=0,
        help="Row index for the local startup explanation.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=250,
        help="Number of rows to use for SHAP explanation generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.mode)
    model_pipeline = load_model(args.mode)

    X, y, feature_columns = prepare_features(df, args.mode)

    sample_size = min(args.sample_size, len(X))
    X_sample = X.sample(n=sample_size, random_state=42).sort_index()

    row_index = max(0, min(args.row_index, len(X_sample) - 1))

    X_transformed = transform_features(model_pipeline, X_sample)
    explainer, shap_values = build_explainer(model_pipeline, X_transformed)

    global_bar_path = OUTPUT_DIR / f"{args.mode}_shap_global_bar.png"
    beeswarm_path = OUTPUT_DIR / f"{args.mode}_shap_beeswarm.png"
    waterfall_path = OUTPUT_DIR / f"{args.mode}_shap_waterfall_row_{row_index}.png"
    summary_path = OUTPUT_DIR / f"{args.mode}_shap_summary_row_{row_index}.json"

    save_global_importance_plot(shap_values, X_transformed, global_bar_path)
    save_beeswarm_plot(shap_values, beeswarm_path)
    save_waterfall_plot(shap_values, row_index, waterfall_path)

    original_sample = df.loc[X_sample.index].reset_index(drop=True)
    local_summary = extract_local_explanation(
        shap_values=shap_values,
        X_transformed=X_transformed.reset_index(drop=True),
        original_df=original_sample,
        row_index=row_index,
    )
    save_summary_json(local_summary, summary_path)

    print("\nSHAP explanation artifacts created")
    print("-" * 36)
    print(f"Mode                 : {args.mode}")
    print(f"Rows explained       : {sample_size}")
    print(f"Local row index      : {row_index}")
    print(f"Global bar plot      : {global_bar_path}")
    print(f"Beeswarm plot        : {beeswarm_path}")
    print(f"Waterfall plot       : {waterfall_path}")
    print(f"Local summary JSON   : {summary_path}")


if __name__ == "__main__":
    main()