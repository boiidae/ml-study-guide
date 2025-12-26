from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


IRIS_FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


def load_iris_df(add_species_name: bool = True) -> pd.DataFrame:
    """
    Load Iris from scikit-learn and return a clean DataFrame with:
      - 4 numeric feature columns
      - species (0/1/2)
      - species_name (optional)
    """
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target

    if add_species_name:
        mapping = dict(enumerate(iris.target_names))
        df["species_name"] = df["species"].map(mapping)

    return df


def describe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Describe only the numeric measurement columns (exclude label columns)."""
    return df[IRIS_FEATURES].describe()


def mean_sepal_length(df: pd.DataFrame) -> float:
    """Compute the mean of sepal length (cm)."""
    return float(df["sepal length (cm)"].mean())


def grouped_feature_means(df: pd.DataFrame, by: str = "species_name") -> pd.DataFrame:
    """
    Group by species or species_name and compute mean of 4 features.
    by = 'species' or 'species_name'
    """
    if by not in df.columns:
        raise KeyError(f"Column '{by}' not found in df. Available: {list(df.columns)}")
    return df.groupby(by)[IRIS_FEATURES].mean()


def first_flower_vector(df: pd.DataFrame) -> np.ndarray:
    """Return the first row's 4 features as a NumPy vector."""
    return df.loc[0, IRIS_FEATURES].to_numpy(dtype=float)


def dot_score(feature_vector: np.ndarray, weights: np.ndarray) -> float:
    """Compute dot product score."""
    feature_vector = np.asarray(feature_vector, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if feature_vector.shape != weights.shape:
        raise ValueError(f"Shape mismatch: x{feature_vector.shape} vs w{weights.shape}")
    return float(np.dot(feature_vector, weights))


def export_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Export DataFrame to CSV (GitHub-friendly artifact)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
