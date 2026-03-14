from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd


BASELINE_MODEL_PATH = Path("models/baseline_logreg.joblib")
EMBEDDING_MODEL_PATH = Path("models/embedding_logreg.joblib")
INPUT_PATH = Path("data/processed/sample_startup_input.json")
OUTPUT_PATH = Path("outputs/prediction.json")
EMBEDDER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

NUMERIC_FEATURES = [
    "funding_total_usd",
    "funding_rounds",
    "milestones",
    "relationships",
    "avg_participants",
    "funding_per_round_usd",
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

REQUIRED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES


class StartupPredictor:
    def __init__(
        self,
        *,
        model_mode: str = "embedding",
        baseline_model_path: Path = BASELINE_MODEL_PATH,
        embedding_model_path: Path = EMBEDDING_MODEL_PATH,
        embedder_model_name: str = EMBEDDER_MODEL_NAME,
    ) -> None:
        self.model_mode = model_mode
        self.baseline_model_path = baseline_model_path
        self.embedding_model_path = embedding_model_path
        self.embedder_model_name = embedder_model_name

        self.model = self._load_model()
        self.embedder = None

        if self.model_mode == "embedding":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for embedding mode. "
                    "Install it with: pip install sentence-transformers"
                ) from exc

            self.embedder = SentenceTransformer(self.embedder_model_name)

    def _load_model(self):
        model_path = (
            self.embedding_model_path
            if self.model_mode == "embedding"
            else self.baseline_model_path
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run the appropriate training script first."
            )

        return joblib.load(model_path)

    def predict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        X_input = self._build_input_frame(record)

        failure_probability = float(self.model.predict_proba(X_input)[0][1])
        predicted_label = int(self.model.predict(X_input)[0])
        reasoning = generate_reasoning(record, failure_probability)

        return {
            "startup_name": record.get("startup_name", "Unknown Startup"),
            "model_mode": self.model_mode,
            "predicted_label": predicted_label,
            "predicted_outcome": (
                "fail_within_24_months"
                if predicted_label == 1
                else "survive_24_months"
            ),
            "failure_probability": round(failure_probability, 4),
            "survival_probability": round(1.0 - failure_probability, 4),
            "reasoning": reasoning,
            "input_summary": {
                "sector": record.get("sector"),
                "stage": record.get("stage"),
                "region": record.get("region"),
                "funding_total_usd": record.get("funding_total_usd"),
                "funding_rounds": record.get("funding_rounds"),
                "milestones": record.get("milestones"),
                "relationships": record.get("relationships"),
                "avg_participants": record.get("avg_participants"),
                "company_age_years": record.get("company_age_years"),
                "has_vc": record.get("has_vc"),
                "has_angel": record.get("has_angel"),
                "is_top500": record.get("is_top500"),
            },
        }

    def _build_input_frame(self, record: Dict[str, Any]) -> pd.DataFrame:
        normalized = normalize_record(record)

        if self.model_mode == "baseline":
            feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
            row = {col: normalized[col] for col in feature_columns}
            return pd.DataFrame([row])

        text_embedding = self._embed_text(normalized["combined_text"])
        embedding_columns = {
            f"emb_{i}": float(value) for i, value in enumerate(text_embedding)
        }

        row = {
            **{col: normalized[col] for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES},
            **embedding_columns,
        }
        return pd.DataFrame([row])

    def _embed_text(self, text: str) -> np.ndarray:
        if self.embedder is None:
            raise RuntimeError("Embedder is not initialized. Use model_mode='embedding'.")

        embedding = self.embedder.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        return embedding.astype(np.float32)


def load_input_json(input_path: Path = INPUT_PATH) -> Dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found at {input_path}. Create it first or pass a custom path."
        )

    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def combine_text_fields(record: Dict[str, Any]) -> str:
    parts = []
    for col in TEXT_FEATURES:
        value = record.get(col, "")
        parts.append("" if value is None else str(value))
    return " ".join(parts).strip()


def coerce_numeric(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}

    for feature in REQUIRED_FEATURES:
        value = record.get(feature)

        if feature in NUMERIC_FEATURES:
            normalized[feature] = coerce_numeric(value, 0.0)
        else:
            normalized[feature] = "" if value is None else str(value)

    normalized["combined_text"] = combine_text_fields(normalized)
    return normalized


def generate_reasoning(record: Dict[str, Any], failure_probability: float) -> Dict[str, Any]:
    reasons = []
    strengths = []
    recommendations = []

    funding_total = coerce_numeric(record.get("funding_total_usd"), 0.0)
    funding_rounds = coerce_numeric(record.get("funding_rounds"), 0.0)
    milestones = coerce_numeric(record.get("milestones"), 0.0)
    relationships = coerce_numeric(record.get("relationships"), 0.0)
    avg_participants = coerce_numeric(record.get("avg_participants"), 0.0)
    company_age_years = coerce_numeric(record.get("company_age_years"), 0.0)
    milestones_per_year = coerce_numeric(record.get("milestones_per_year"), 0.0)
    years_to_first_funding = coerce_numeric(record.get("years_to_first_funding"), 0.0)

    has_vc = int(coerce_numeric(record.get("has_vc"), 0.0))
    has_angel = int(coerce_numeric(record.get("has_angel"), 0.0))
    has_roundA = int(coerce_numeric(record.get("has_roundA"), 0.0))
    has_roundB = int(coerce_numeric(record.get("has_roundB"), 0.0))
    has_roundC = int(coerce_numeric(record.get("has_roundC"), 0.0))
    has_roundD = int(coerce_numeric(record.get("has_roundD"), 0.0))
    is_top500 = int(coerce_numeric(record.get("is_top500"), 0.0))

    sector = str(record.get("sector", "") or "").lower()
    stage = str(record.get("stage", "") or "").lower()

    if funding_total <= 0:
        reasons.append("The company shows no recorded funding, which may indicate limited traction or support.")
        recommendations.append("Strengthen fundraising readiness and validate investor interest.")
    elif funding_total >= 10_000_000:
        strengths.append("The company has raised substantial capital.")
    elif funding_total < 500_000:
        reasons.append("Total funding is relatively low compared with many surviving startups.")

    if funding_rounds <= 0:
        reasons.append("No funding rounds are recorded.")
    elif funding_rounds >= 3:
        strengths.append("Multiple funding rounds suggest continued investor support.")

    if milestones <= 0:
        reasons.append("The company has no recorded milestones, suggesting limited visible progress.")
        recommendations.append("Focus on achieving clearer traction milestones.")
    elif milestones >= 4:
        strengths.append("The company has a strong milestone record.")

    if milestones_per_year < 0.5 and company_age_years >= 3:
        reasons.append("Milestone velocity appears low relative to company age.")
    elif milestones_per_year >= 1.0:
        strengths.append("Milestone velocity is relatively strong.")

    if relationships < 3:
        reasons.append("The company has a limited relationship network in the dataset.")
    elif relationships >= 10:
        strengths.append("The company has a strong relationship network signal.")

    if avg_participants < 1.5 and funding_rounds > 0:
        reasons.append("Funding rounds appear to involve relatively few participants.")
    elif avg_participants >= 3:
        strengths.append("Funding rounds show broader participant support.")

    if years_to_first_funding > 3:
        reasons.append("The company took a long time to reach first funding.")
    elif funding_rounds > 0 and years_to_first_funding <= 1:
        strengths.append("The company reached first funding quickly.")

    if company_age_years >= 8 and milestones <= 1:
        reasons.append("The company is older but shows limited milestone progress.")
        recommendations.append("Reassess strategy, market position, and execution priorities.")

    if has_vc == 1:
        strengths.append("VC backing is a positive institutional signal.")
    else:
        reasons.append("The company does not show VC backing in the dataset.")

    if has_angel == 1:
        strengths.append("Angel support provides an early validation signal.")

    if has_roundA == 1:
        strengths.append("The company progressed through Round A.")
    if has_roundB == 1:
        strengths.append("The company progressed through Round B.")
    if has_roundC == 1 or has_roundD == 1:
        strengths.append("Later-stage rounds suggest strong historical momentum.")

    if is_top500 == 1:
        strengths.append("Top-500 status is a strong prestige and visibility signal.")

    if stage == "pre_seed" and funding_total < 250_000:
        reasons.append("The startup appears very early with limited capital.")
    elif stage in {"series_a", "series_b"} and milestones < 2:
        reasons.append("Later stage positioning is not matched by strong milestone count.")

    if sector == "biotech" and funding_total < 1_000_000:
        reasons.append("Biotech startups often require more capital than currently recorded.")
    if sector in {"software", "web", "mobile"} and milestones == 0:
        reasons.append("The startup operates in a fast-moving sector but shows no recorded milestones.")

    if not reasons:
        reasons.append("The startup shows a relatively balanced operating and funding profile based on the available inputs.")

    if not recommendations:
        recommendations.append("Improve traction signals, investor support, and milestone progression over time.")

    risk_band = (
        "high"
        if failure_probability >= 0.70
        else "medium"
        if failure_probability >= 0.40
        else "low"
    )

    return {
        "risk_band": risk_band,
        "top_risk_factors": reasons[:4],
        "positive_signals": strengths[:4],
        "recommendations": recommendations[:4],
    }


def save_prediction(result: Dict[str, Any], output_path: Path = OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def print_prediction(result: Dict[str, Any]) -> None:
    print("\nStartup Failure Prediction")
    print("-" * 32)
    print(f"Startup              : {result['startup_name']}")
    print(f"Model mode           : {result['model_mode']}")
    print(f"Predicted outcome    : {result['predicted_outcome']}")
    print(f"Failure probability  : {result['failure_probability']:.4f}")
    print(f"Survival probability : {result['survival_probability']:.4f}")
    print(f"Risk band            : {result['reasoning']['risk_band']}")

    print("\nTop risk factors:")
    for item in result["reasoning"]["top_risk_factors"]:
        print(f"- {item}")

    if result["reasoning"]["positive_signals"]:
        print("\nPositive signals:")
        for item in result["reasoning"]["positive_signals"]:
            print(f"- {item}")

    print("\nRecommendations:")
    for item in result["reasoning"]["recommendations"]:
        print(f"- {item}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict startup failure risk.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(INPUT_PATH),
        help="Path to input startup JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_PATH),
        help="Path to save prediction JSON.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "embedding"],
        default="baseline",
        help="Prediction mode: baseline or embedding.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    record = load_input_json(Path(args.input))
    predictor = StartupPredictor(model_mode=args.mode)
    result = predictor.predict(record)

    save_prediction(result, Path(args.output))
    print_prediction(result)
    print(f"\nSaved prediction to: {args.output}")


if __name__ == "__main__":
    main()