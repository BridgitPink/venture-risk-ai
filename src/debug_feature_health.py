# src/debug_feature_health.py

from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/processed/real_startups_model_ready.csv")
TARGET_COL = "failed_within_24_months"

TEXT_CANDIDATES = ["description", "founder_bios", "recent_update"]
ID_CANDIDATES = ["startup_name"]

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


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"\nLoaded: {DATA_PATH}")
    print(f"Shape: {df.shape}\n")

    print("Columns:")
    print(df.columns.tolist())
    print()

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    print("Target distribution:")
    print(df[TARGET_COL].value_counts(dropna=False))
    print()

    numeric_cols = [c for c in NUMERIC_CANDIDATES if c in df.columns]
    categorical_cols = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]
    text_cols = [c for c in TEXT_CANDIDATES if c in df.columns]
    id_cols = [c for c in ID_CANDIDATES if c in df.columns]

    print(f"Numeric cols found ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical cols found ({len(categorical_cols)}): {categorical_cols}")
    print(f"Text cols found ({len(text_cols)}): {text_cols}")
    print(f"ID cols found ({len(id_cols)}): {id_cols}")
    print()

    if numeric_cols:
        non_null_counts = df[numeric_cols].notna().sum().sort_values()
        missing_rates = df[numeric_cols].isna().mean().sort_values(ascending=False)

        print("Numeric columns with fewest observed values:")
        print(non_null_counts.head(20))
        print()

        all_null_numeric = non_null_counts[non_null_counts == 0].index.tolist()
        print("All-null numeric columns:")
        print(all_null_numeric if all_null_numeric else "None")
        print()

        print("Highest missing-rate numeric columns:")
        print(missing_rates.head(20))
        print()

    if categorical_cols:
        cat_non_null = df[categorical_cols].notna().sum().sort_values()
        all_null_cat = cat_non_null[cat_non_null == 0].index.tolist()

        print("All-null categorical columns:")
        print(all_null_cat if all_null_cat else "None")
        print()

    if "startup_name" in df.columns:
        dupes = df["startup_name"].duplicated().sum()
        print(f"Duplicate startup_name rows: {dupes}")
        print()

    print("Sample rows:")
    sample_cols = [c for c in (id_cols + categorical_cols + numeric_cols[:8] + text_cols) if c in df.columns]
    print(df[sample_cols].head(5).to_string(index=False))
    print()

    # Helpful merge sanity check
    expected_problem_cols = [
        "team_size",
        "founder_count",
        "founder_experience_years",
        "has_technical_founder",
        "monthly_burn_usd",
        "runway_months",
        "revenue_growth_pct",
        "customer_growth_pct",
        "churn_pct",
        "burn_multiple",
        "annual_revenue_run_rate",
    ]
    expected_problem_cols = [c for c in expected_problem_cols if c in df.columns]

    if expected_problem_cols:
        print("Problem-column non-null counts:")
        print(df[expected_problem_cols].notna().sum())
        print()


if __name__ == "__main__":
    main()