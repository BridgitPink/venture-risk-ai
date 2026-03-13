# src/prepare_real_data.py

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_CANDIDATES = [
    PROJECT_ROOT / "data" / "raw" / "startup_success_prediction" / "startup_success_prediction.csv"
]

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "real_startups_model_ready.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

TARGET_COL = "failed_within_24_months"


def find_input_file() -> Path:
    print("Searching for raw dataset in:")
    for path in RAW_CANDIDATES:
        print(f"  - {path} | exists={path.exists()}")
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find a raw startup dataset. Checked:\n"
        + "\n".join(str(p) for p in RAW_CANDIDATES)
    )


def clean_column_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_column_name(c) for c in df.columns]
    return df


def ensure_column(df: pd.DataFrame, col: str, default=np.nan) -> pd.DataFrame:
    if col not in df.columns:
        df[col] = default
    return df


def to_numeric_safe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def to_string_safe(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def map_region_from_text(value: str) -> str:
    if not value or value == "unknown":
        return "unknown"

    x = value.lower().strip()

    northeast = {"ny", "ma", "nj", "pa", "ct", "ri", "vt", "nh", "me"}
    southeast = {"fl", "ga", "nc", "sc", "va", "wv", "md", "de", "dc", "al", "ms", "tn", "ky", "la", "ar"}
    midwest = {"il", "in", "oh", "mi", "wi", "mn", "ia", "mo", "nd", "sd", "ne", "ks"}
    southwest = {"tx", "az", "nm", "ok"}
    west = {"ca", "wa", "or", "co", "ut", "nv", "id", "mt", "wy", "ak", "hi"}

    token = x.split(",")[-1].strip() if "," in x else x.split()[-1].strip()

    if token in northeast:
        return "northeast"
    if token in southeast:
        return "southeast"
    if token in midwest:
        return "midwest"
    if token in southwest:
        return "southwest"
    if token in west:
        return "west"
    return x


def build_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build neutral text fields only from non-leaky attributes.
    This intentionally overwrites any raw text fields so status/outcome
    words like 'acquired' or 'closed' do not leak into embeddings.
    """
    df = df.copy()

    startup_name = to_string_safe(df["startup_name"]) if "startup_name" in df.columns else pd.Series(["unknown"] * len(df))
    sector = to_string_safe(df["sector"]) if "sector" in df.columns else pd.Series(["unknown"] * len(df))
    location = to_string_safe(df["location"]) if "location" in df.columns else pd.Series(["unknown"] * len(df))
    region = to_string_safe(df["region"]) if "region" in df.columns else pd.Series(["unknown"] * len(df))
    founded_year = df["founded_year"] if "founded_year" in df.columns else pd.Series([np.nan] * len(df))
    relationships = df["relationships"] if "relationships" in df.columns else pd.Series([np.nan] * len(df))
    milestones = df["milestones"] if "milestones" in df.columns else pd.Series([np.nan] * len(df))
    funding_total_usd = df["funding_total_usd"] if "funding_total_usd" in df.columns else pd.Series([np.nan] * len(df))
    funding_rounds = df["funding_rounds"] if "funding_rounds" in df.columns else pd.Series([np.nan] * len(df))
    company_age_years = df["company_age_years"] if "company_age_years" in df.columns else pd.Series([np.nan] * len(df))

    def fmt_num(v):
        if pd.isna(v):
            return "unknown"
        try:
            return str(int(v))
        except Exception:
            return str(v)

    descriptions = []
    bios = []
    updates = []

    for i in range(len(df)):
        place = location.iloc[i] if location.iloc[i] else region.iloc[i]
        descriptions.append(
            f"Startup {startup_name.iloc[i] or 'unknown'} operates in {sector.iloc[i] or 'unknown'} "
            f"and is based in {place or 'unknown'}."
        )
        bios.append(
            f"The company has {fmt_num(relationships.iloc[i])} relationships, "
            f"{fmt_num(milestones.iloc[i])} milestones, and "
            f"{fmt_num(funding_rounds.iloc[i])} funding rounds."
        )
        updates.append(
            f"Founded in {fmt_num(founded_year.iloc[i])}, with total funding of "
            f"{fmt_num(funding_total_usd.iloc[i])} USD and company age of "
            f"{fmt_num(company_age_years.iloc[i])} years."
        )

    df["description"] = descriptions
    df["founder_bios"] = bios
    df["recent_update"] = updates

    return df


def derive_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if TARGET_COL in df.columns and df[TARGET_COL].notna().any():
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
        print(f"Using existing target column: {TARGET_COL}")
        return df

    source_col = None
    for candidate in ["status", "outcome", "company_status", "operating_status", "is_closed", "closed"]:
        if candidate in df.columns:
            source_col = candidate
            break

    if source_col is None:
        raise ValueError(
            f"Could not derive '{TARGET_COL}'. None of these columns were found: "
            "status, outcome, company_status, operating_status, is_closed, closed"
        )

    if source_col in ["is_closed", "closed"]:
        raw = pd.to_numeric(df[source_col], errors="coerce")
        df[TARGET_COL] = np.where(raw == 1, 1, np.where(raw == 0, 0, np.nan))
        print(f"Derived {TARGET_COL} from numeric column: {source_col}")
        return df

    status = df[source_col].fillna("").astype(str).str.strip().str.lower()

    failed_values = {
        "closed", "failed", "shutdown", "shut down", "dead", "inactive", "bankrupt"
    }
    success_or_alive_values = {
        "operating", "active", "acquired", "ipo", "public", "running", "live"
    }

    df[TARGET_COL] = np.where(
        status.isin(failed_values),
        1,
        np.where(status.isin(success_or_alive_values), 0, np.nan)
    )

    print(f"Derived {TARGET_COL} from text column: {source_col}")
    print("Raw source value counts:")
    print(df[source_col].value_counts(dropna=False).head(20))

    return df


def main() -> None:
    input_path = find_input_file()
    df = pd.read_csv(input_path)
    df = normalize_columns(df)

    print(f"\nLoaded raw dataset: {input_path}")
    print(f"Raw shape: {df.shape}")

    # Canonical columns
    # IMPORTANT: do NOT map status -> stage, because status is used to derive the label
    # and would cause direct leakage into the model.
    rename_map = {
        "name": "startup_name",
        "company_name": "startup_name",
        "category_code": "sector",
        "market": "sector",
        "state": "region",
        "state_code": "region",
        "funding_round_code": "stage",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # Ensure required columns exist
    required = [
        "startup_name",
        "sector",
        "stage",
        "region",
        "business_model",
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
        "has_vc",
        "has_angel",
        "has_rounda",
        "has_roundb",
        "has_roundc",
        "has_roundd",
        "is_top500",
        "latitude",
        "longitude",
    ]
    for c in required:
        df = ensure_column(df, c)

    # Normalize string columns
    for c in ["startup_name", "sector", "stage", "region", "business_model"]:
        df[c] = to_string_safe(df[c]).replace("", "unknown")

    # Derive target BEFORE dropping leaky columns
    df = derive_target(df)

    print("\nBefore target filtering:")
    print(df[TARGET_COL].value_counts(dropna=False))

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL].isin([0, 1])].copy()

    print("\nAfter target filtering:")
    print(df[TARGET_COL].value_counts(dropna=False))
    print(f"Remaining rows after target filter: {len(df)}")

    if len(df) == 0:
        raise ValueError(
            f"All rows were removed after deriving/filtering '{TARGET_COL}'. "
            "Check the raw label column values and expand the mapping."
        )

    # Drop columns that directly define or reveal the label
    leaky_cols = [
        "status",
        "outcome",
        "company_status",
        "operating_status",
        "closed",
        "is_closed",
    ]
    existing_leaky = [c for c in leaky_cols if c in df.columns]
    if existing_leaky:
        print("\nDropping leaky columns:")
        print(existing_leaky)
        df = df.drop(columns=existing_leaky)

    # Drop stage too if it contains outcome-like values
    if "stage" in df.columns:
        stage_vals = df["stage"].fillna("").astype(str).str.strip().str.lower()
        if stage_vals.isin(["acquired", "closed", "failed", "ipo", "operating"]).any():
            print("Dropping leaky column: stage")
            df = df.drop(columns=["stage"])

    # Numeric conversions
    numeric_cols = [
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
        "has_vc",
        "has_angel",
        "has_rounda",
        "has_roundb",
        "has_roundc",
        "has_roundd",
        "is_top500",
        "latitude",
        "longitude",
        TARGET_COL,
    ]
    df = to_numeric_safe(df, numeric_cols)

    # Basic feature engineering
    current_year = 2026

    df["company_age_years"] = np.where(
        df["founded_year"].notna(),
        current_year - df["founded_year"],
        np.nan,
    )

    df["funding_per_round_usd"] = np.where(
        (df["funding_rounds"].fillna(0) > 0) & df["funding_total_usd"].notna(),
        df["funding_total_usd"] / df["funding_rounds"].replace(0, np.nan),
        np.nan,
    )

    df["milestones_per_year"] = np.where(
        (df["company_age_years"].fillna(0) > 0) & df["milestones"].notna(),
        df["milestones"] / df["company_age_years"].replace(0, np.nan),
        np.nan,
    )

    df["years_to_first_funding"] = np.nan
    df["years_to_last_funding"] = np.nan

    if "first_funding_at" in df.columns:
        first_funding_year = pd.to_datetime(df["first_funding_at"], errors="coerce").dt.year
        df["years_to_first_funding"] = first_funding_year - df["founded_year"]

    if "last_funding_at" in df.columns:
        last_funding_year = pd.to_datetime(df["last_funding_at"], errors="coerce").dt.year
        df["years_to_last_funding"] = last_funding_year - df["founded_year"]

    # Preserve old expected capitalized columns for downstream scripts
    for lower, original in [
        ("has_rounda", "has_roundA"),
        ("has_roundb", "has_roundB"),
        ("has_roundc", "has_roundC"),
        ("has_roundd", "has_roundD"),
    ]:
        if lower in df.columns and original not in df.columns:
            df[original] = df[lower]

    # Region handling
    if "location" in df.columns:
        df["region"] = np.where(
            df["region"].eq("unknown"),
            df["location"].fillna("").astype(str).apply(map_region_from_text),
            df["region"],
        )

    # Build safe text columns AFTER leakage cleanup
    df = build_text_fields(df)

    # Remove duplicate startup names, keep first
    if "startup_name" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["startup_name"], keep="first").copy()
        print(f"Removed duplicates by startup_name: {before - len(df)}")

    # Final schema in the order the model expects
    final_cols = [
        "startup_name",
        "sector",
        "stage",
        "region",
        "business_model",
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
        "description",
        "founder_bios",
        "recent_update",
        TARGET_COL,
    ]

    for c in final_cols:
        df = ensure_column(df, c)

    df = df[final_cols].copy()

    # Final leakage audit
    print("\nLeakage audit on text columns:")
    for c in ["description", "founder_bios", "recent_update"]:
        if c in df.columns:
            has_leak = df[c].astype(str).str.contains(
                r"acquired|closed|failed|bankrupt|ipo|shutdown",
                case=False,
                regex=True,
                na=False,
            ).any()
            print(f"  - {c}: {'POTENTIAL LEAK' if has_leak else 'clean'}")

    # Report all-null numeric features
    final_numeric = [
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

    all_null_numeric = [c for c in final_numeric if df[c].notna().sum() == 0]
    print("\nAll-null numeric columns in final dataset:")
    print(all_null_numeric if all_null_numeric else "None")

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved model-ready dataset to: {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print("\nLabel distribution:")
    print(df[TARGET_COL].value_counts(dropna=False))


if __name__ == "__main__":
    main()