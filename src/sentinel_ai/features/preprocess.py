from __future__ import annotations

from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from sentinel_ai.data.synthetic import FEATURE_COLUMNS

BOOLEAN_COLUMNS = ["is_cross_border", "is_new_device"]
NUMERIC_COLUMNS = [column for column in FEATURE_COLUMNS if column not in BOOLEAN_COLUMNS]


def _bool_to_int(values: Any) -> Any:
    return values.astype(int)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
            (
                "boolean",
                FunctionTransformer(_bool_to_int, feature_names_out="one-to-one"),
                BOOLEAN_COLUMNS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_feature_pipeline(model: ClassifierMixin) -> Pipeline:
    return Pipeline([("features", build_preprocessor()), ("model", model)])
