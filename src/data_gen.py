from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


SECTORS = [
    "fintech",
    "healthtech",
    "edtech",
    "saas",
    "climate",
    "cybersecurity",
    "ecommerce",
    "biotech",
    "logistics",
    "ai_tools",
]

STAGES = ["pre_seed", "seed", "series_a", "series_b"]
REGIONS = ["us", "canada", "europe", "latam", "africa", "asia"]
BUSINESS_MODELS = ["b2b", "b2c", "b2b2c"]

SECTOR_PHRASES: Dict[str, List[str]] = {
    "fintech": [
        "digital payments platform for underserved consumers",
        "embedded finance tools for small businesses",
        "compliance-first banking infrastructure",
    ],
    "healthtech": [
        "patient engagement platform for chronic care",
        "clinical workflow automation for providers",
        "remote monitoring tools for outpatient care",
    ],
    "edtech": [
        "adaptive learning platform for workforce upskilling",
        "AI tutoring assistant for students",
        "credentialing and assessment tools for schools",
    ],
    "saas": [
        "workflow automation software for operations teams",
        "analytics dashboard for mid-market companies",
        "collaboration software for distributed teams",
    ],
    "climate": [
        "carbon measurement and reporting platform",
        "grid optimization software for utilities",
        "sustainability intelligence tools for enterprises",
    ],
    "cybersecurity": [
        "threat detection platform for cloud infrastructure",
        "identity security tools for modern enterprises",
        "security automation for lean IT teams",
    ],
    "ecommerce": [
        "conversion optimization tools for online brands",
        "creator-led commerce infrastructure",
        "inventory and fulfillment software for merchants",
    ],
    "biotech": [
        "AI-assisted drug discovery workflows",
        "diagnostics platform for early disease detection",
        "lab automation tools for biotech teams",
    ],
    "logistics": [
        "route optimization platform for fleet operators",
        "supply chain visibility tools for shippers",
        "warehouse automation orchestration software",
    ],
    "ai_tools": [
        "LLM workflow automation for knowledge workers",
        "AI copilots for enterprise productivity",
        "multimodal search and reasoning tools",
    ],
}

SUCCESS_WORDS = [
    "strong retention",
    "capital efficient",
    "experienced founders",
    "clear product-market fit",
    "growing revenue base",
    "repeatable acquisition engine",
]

RISK_WORDS = [
    "crowded market",
    "high burn",
    "weak retention",
    "unclear differentiation",
    "limited runway",
    "go-to-market challenges",
]


@dataclass
class StartupRecord:
    startup_name: str
    sector: str
    stage: str
    region: str
    business_model: str
    founded_year: int
    team_size: int
    founder_count: int
    founder_experience_years: int
    has_technical_founder: int
    funding_total_usd: int
    monthly_burn_usd: int
    runway_months: float
    revenue_growth_pct: float
    customer_growth_pct: float
    churn_pct: float
    burn_multiple: float
    annual_revenue_run_rate: int
    description: str
    founder_bios: str
    recent_update: str
    failed_within_24_months: int
    risk_score: float


class StartupDataGenerator:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def generate(self, n_samples: int = 1000) -> pd.DataFrame:
        rows = [asdict(self._generate_one(i)) for i in range(n_samples)]
        return pd.DataFrame(rows)

    def _generate_one(self, idx: int) -> StartupRecord:
        sector = random.choice(SECTORS)
        stage = random.choices(STAGES, weights=[0.35, 0.35, 0.2, 0.1], k=1)[0]
        region = random.choice(REGIONS)
        business_model = random.choices(BUSINESS_MODELS, weights=[0.55, 0.3, 0.15], k=1)[0]

        founded_year = random.randint(2018, 2025)
        founder_count = random.randint(1, 4)
        founder_experience_years = random.randint(1, 20)
        has_technical_founder = random.choices([0, 1], weights=[0.3, 0.7], k=1)[0]

        stage_team_ranges = {
            "pre_seed": (2, 12),
            "seed": (5, 30),
            "series_a": (15, 80),
            "series_b": (40, 200),
        }
        team_low, team_high = stage_team_ranges[stage]
        team_size = random.randint(team_low, team_high)

        funding_ranges = {
            "pre_seed": (100_000, 1_500_000),
            "seed": (750_000, 5_000_000),
            "series_a": (3_000_000, 18_000_000),
            "series_b": (12_000_000, 60_000_000),
        }
        funding_total_usd = random.randint(*funding_ranges[stage])

        burn_ratio = random.uniform(0.01, 0.08)
        monthly_burn_usd = max(15_000, int(funding_total_usd * burn_ratio / 12))
        runway_months = round(random.uniform(3, 30), 1)
        revenue_growth_pct = round(random.uniform(-20, 250), 1)
        customer_growth_pct = round(random.uniform(-15, 300), 1)
        churn_pct = round(random.uniform(1, 35), 1)
        burn_multiple = round(random.uniform(0.5, 8.0), 2)
        annual_revenue_run_rate = max(0, int(random.uniform(0, funding_total_usd * 1.5)))

        description = self._make_description(sector, stage, business_model)
        founder_bios = self._make_founder_bios(founder_count, founder_experience_years, has_technical_founder)
        recent_update = self._make_update(
            runway_months=runway_months,
            revenue_growth_pct=revenue_growth_pct,
            churn_pct=churn_pct,
            sector=sector,
        )

        risk_score = self._compute_risk_score(
            stage=stage,
            runway_months=runway_months,
            revenue_growth_pct=revenue_growth_pct,
            customer_growth_pct=customer_growth_pct,
            churn_pct=churn_pct,
            burn_multiple=burn_multiple,
            founder_experience_years=founder_experience_years,
            has_technical_founder=has_technical_founder,
            annual_revenue_run_rate=annual_revenue_run_rate,
            funding_total_usd=funding_total_usd,
            sector=sector,
        )

        failed_within_24_months = 1 if risk_score >= 0.55 else 0
        startup_name = f"{sector.title().replace('_', '')}Labs{idx:04d}"

        return StartupRecord(
            startup_name=startup_name,
            sector=sector,
            stage=stage,
            region=region,
            business_model=business_model,
            founded_year=founded_year,
            team_size=team_size,
            founder_count=founder_count,
            founder_experience_years=founder_experience_years,
            has_technical_founder=has_technical_founder,
            funding_total_usd=funding_total_usd,
            monthly_burn_usd=monthly_burn_usd,
            runway_months=runway_months,
            revenue_growth_pct=revenue_growth_pct,
            customer_growth_pct=customer_growth_pct,
            churn_pct=churn_pct,
            burn_multiple=burn_multiple,
            annual_revenue_run_rate=annual_revenue_run_rate,
            description=description,
            founder_bios=founder_bios,
            recent_update=recent_update,
            failed_within_24_months=failed_within_24_months,
            risk_score=round(risk_score, 3),
        )

    def _compute_risk_score(
        self,
        *,
        stage: str,
        runway_months: float,
        revenue_growth_pct: float,
        customer_growth_pct: float,
        churn_pct: float,
        burn_multiple: float,
        founder_experience_years: int,
        has_technical_founder: int,
        annual_revenue_run_rate: int,
        funding_total_usd: int,
        sector: str,
    ) -> float:
        score = 0.35

        if runway_months < 6:
            score += 0.22
        elif runway_months < 12:
            score += 0.10
        else:
            score -= 0.08

        if revenue_growth_pct < 10:
            score += 0.16
        elif revenue_growth_pct < 40:
            score += 0.06
        elif revenue_growth_pct > 120:
            score -= 0.08

        if customer_growth_pct < 15:
            score += 0.10
        elif customer_growth_pct > 80:
            score -= 0.06

        if churn_pct > 20:
            score += 0.18
        elif churn_pct > 12:
            score += 0.08
        elif churn_pct < 5:
            score -= 0.05

        if burn_multiple > 4:
            score += 0.15
        elif burn_multiple < 1.5:
            score -= 0.05

        if founder_experience_years < 4:
            score += 0.08
        elif founder_experience_years > 10:
            score -= 0.06

        if not has_technical_founder and sector in {"ai_tools", "cybersecurity", "fintech", "biotech"}:
            score += 0.08

        revenue_to_funding = annual_revenue_run_rate / max(funding_total_usd, 1)
        if revenue_to_funding < 0.1:
            score += 0.08
        elif revenue_to_funding > 0.6:
            score -= 0.07

        if stage in {"pre_seed", "seed"}:
            score += 0.03

        return float(min(max(score, 0.01), 0.99))

    def _make_description(self, sector: str, stage: str, business_model: str) -> str:
        phrase = random.choice(SECTOR_PHRASES[sector])
        strengths = random.sample(SUCCESS_WORDS, k=2)
        risks = random.sample(RISK_WORDS, k=2)
        return (
            f"{stage.replace('_', ' ')} {business_model.upper()} startup building a {phrase}. "
            f"Team messaging emphasizes {strengths[0]} and {strengths[1]}, while operating in a market facing "
            f"{risks[0]} and {risks[1]}."
        )

    def _make_founder_bios(self, founder_count: int, experience_years: int, has_technical_founder: int) -> str:
        technical_text = "includes a technical co-founder with product and engineering depth" if has_technical_founder else "lacks a dedicated technical co-founder"
        return (
            f"Founding team of {founder_count}. Combined founder experience averages about {experience_years} years. "
            f"The team has prior startup and operator exposure and {technical_text}."
        )

    def _make_update(self, *, runway_months: float, revenue_growth_pct: float, churn_pct: float, sector: str) -> str:
        outlook = "strong" if revenue_growth_pct > 60 and churn_pct < 8 else "mixed" if revenue_growth_pct > 20 else "fragile"
        return (
            f"Recent investor update: management reports {revenue_growth_pct}% revenue growth, {churn_pct}% churn, "
            f"and approximately {runway_months} months of runway. Leadership says demand in {sector} remains {outlook}."
        )


def save_dataset(df: pd.DataFrame, output_csv: str | Path, metadata_json: str | Path | None = None) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    if metadata_json is not None:
        metadata = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "label_distribution": df["failed_within_24_months"].value_counts().to_dict(),
        }
        metadata_path = Path(metadata_json)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    generator = StartupDataGenerator(seed=42)
    df = generator.generate(n_samples=1000)

    save_dataset(
        df,
        output_csv="data/processed/synthetic_startups.csv",
        metadata_json="data/processed/synthetic_startups_metadata.json",
    )

    print("Saved dataset to data/processed/synthetic_startups.csv")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
