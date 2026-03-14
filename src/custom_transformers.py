from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class TextCombiner(BaseEstimator, TransformerMixin):
    def __init__(self, text_columns=None):
        self.text_columns = text_columns or [
            "description",
            "founder_bios",
            "recent_update",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        def combine_row(row):
            parts = []
            for col in self.text_columns:
                value = row[col] if col in row else ""
                if pd.notna(value):
                    parts.append(str(value))
            return " ".join(parts).strip()

        return X.apply(combine_row, axis=1)