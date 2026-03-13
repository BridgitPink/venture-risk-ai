from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DATA_PATH = Path("data/processed/real_startups_model_ready.csv")
OUTPUT_DATA_PATH = Path("data/processed/real_startups_with_embeddings.parquet")
OUTPUT_MATRIX_PATH = Path("data/processed/real_text_embeddings.npy")
OUTPUT_METADATA_PATH = Path("data/processed/real_text_embeddings_metadata.json")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_COLUMNS = ["description", "founder_bios", "recent_update"]
BATCH_SIZE = 32


def load_dataset(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Run `python src/data_gen.py` first."
        )
    return pd.read_csv(csv_path)


def combine_text_columns(df: pd.DataFrame, text_columns: Iterable[str]) -> List[str]:
    combined = (
        df[list(text_columns)]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return combined.tolist()


def load_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def generate_embeddings(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def build_embedding_dataframe(embeddings: np.ndarray, prefix: str = "emb_") -> pd.DataFrame:
    return pd.DataFrame(
        embeddings,
        columns=[f"{prefix}{i}" for i in range(embeddings.shape[1])],
    )


def save_outputs(df_with_embeddings: pd.DataFrame, embeddings: np.ndarray) -> None:
    OUTPUT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_with_embeddings.to_parquet(OUTPUT_DATA_PATH, index=False)
    np.save(OUTPUT_MATRIX_PATH, embeddings)

    metadata = {
        "model_name": MODEL_NAME,
        "rows": int(df_with_embeddings.shape[0]),
        "embedding_dimensions": int(embeddings.shape[1]),
        "text_columns": TEXT_COLUMNS,
        "output_data_path": str(OUTPUT_DATA_PATH),
        "output_matrix_path": str(OUTPUT_MATRIX_PATH),
    }
    OUTPUT_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    df = load_dataset()
    texts = combine_text_columns(df, TEXT_COLUMNS)

    print(f"Loaded {len(texts)} startup profiles")
    print(f"Loading embedding model: {MODEL_NAME}")

    model = load_embedding_model()
    embeddings = generate_embeddings(texts, model=model)

    embedding_df = build_embedding_dataframe(embeddings)
    df_with_embeddings = pd.concat([df.reset_index(drop=True), embedding_df], axis=1)

    save_outputs(df_with_embeddings, embeddings)

    print("\nEmbedding generation complete")
    print(f"Saved parquet dataset to: {OUTPUT_DATA_PATH}")
    print(f"Saved raw embedding matrix to: {OUTPUT_MATRIX_PATH}")
    print(f"Saved metadata to: {OUTPUT_METADATA_PATH}")
    print(f"Embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
