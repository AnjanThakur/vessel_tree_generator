"""Population statistics, circular statistics, 126-D joint deviation PCA, and validation thresholds."""

from __future__ import annotations

import math
from typing import Any
import numpy as np
from scipy.stats import circmean, circstd


def compute_linear_stats(data: Iterable[float]) -> dict[str, float]:
    """Compute standard linear statistics (mean, std, min, max, P2.5, P97.5, median, count)."""
    arr = np.asarray([x for x in data if x is not None and np.isfinite(x)], dtype=float)
    if len(arr) == 0:
        return {
            "count": 0,
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "max": math.nan,
            "median": math.nan,
            "p2_5": math.nan,
            "p97_5": math.nan,
        }
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "p2_5": float(np.percentile(arr, 2.5)),
        "p97_5": float(np.percentile(arr, 97.5)),
    }


def compute_circular_stats(angles_rad: Iterable[float]) -> dict[str, float]:
    """Compute circular statistics for angular quantities in radians."""
    arr = np.asarray([x for x in angles_rad if x is not None and np.isfinite(x)], dtype=float)
    if len(arr) == 0:
        return {
            "count": 0,
            "circular_mean_rad": math.nan,
            "circular_std_rad": math.nan,
            "circular_mean_deg": math.nan,
            "circular_std_deg": math.nan,
        }
    c_mean = float(circmean(arr, low=-np.pi, high=np.pi))
    c_std = float(circstd(arr, low=-np.pi, high=np.pi))
    return {
        "count": int(len(arr)),
        "circular_mean_rad": c_mean,
        "circular_std_rad": c_std,
        "circular_mean_deg": float(math.degrees(c_mean)),
        "circular_std_deg": float(math.degrees(c_std)),
    }


def fit_deviation_pca(shape_vectors_matrix: np.ndarray, variance_cutoff: float = 0.95) -> dict[str, Any]:
    """Fit 126-D joint deviation PCA across complete patient shape vectors.

    Design Doc §5.3.3:
    - Input shape vectors: (N_complete, 126)
    - Center shape vectors by population mean
    - SVD: U, S, Vt
    - Retain k components explaining >= 95% variance
    """
    X = np.asarray(shape_vectors_matrix, dtype=float)
    n_samples, n_features = X.shape
    if n_samples < 2 or n_features != 126:
        raise ValueError(f"PCA requires at least 2 complete cases and 126 dimensions; got shape {X.shape}")

    mean_vector = np.mean(X, axis=0)
    centered = X - mean_vector

    # SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    total_var = np.sum(S**2)
    if total_var <= 1.0e-12:
        explained_ratio = np.ones(len(S)) / len(S)
    else:
        explained_ratio = (S**2) / total_var

    cum_explained_variance = np.cumsum(explained_ratio)
    k_retained = int(np.searchsorted(cum_explained_variance, variance_cutoff) + 1)
    k_retained = min(k_retained, len(S))

    components_retained = Vt[:k_retained]
    singular_values_retained = S[:k_retained]
    eigenvalues = (S**2) / max(n_samples - 1, 1)

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "mean_vector": mean_vector,
        "components_all": Vt,
        "components_retained": components_retained,
        "singular_values": S,
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": explained_ratio,
        "cumulative_explained_variance": cum_explained_variance,
        "k_retained": k_retained,
        "variance_cutoff": variance_cutoff,
    }


def build_validation_thresholds(
    scaffold_params: list[dict[str, float]],
    branch_lengths: dict[str, list[float]],
    bifurcation_angles: list[float],
    pca_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute empirical P2.5 - P97.5 population validation thresholds."""
    thresholds = {}

    # Scaffold semi-axes thresholds
    for key in ["a", "b", "c"]:
        vals = [p[key] for p in scaffold_params if key in p]
        stats = compute_linear_stats(vals)
        thresholds[f"ellipsoid_{key}_mm"] = {
            "unit": "mm",
            "p2_5": stats["p2_5"],
            "p97_5": stats["p97_5"],
            "mean": stats["mean"],
            "std": stats["std"],
        }

    # Branch lengths thresholds
    for vessel, lengths in branch_lengths.items():
        stats = compute_linear_stats(lengths)
        thresholds[f"branch_length_{vessel}_mm"] = {
            "unit": "mm",
            "p2_5": stats["p2_5"],
            "p97_5": stats["p97_5"],
            "mean": stats["mean"],
            "std": stats["std"],
        }

    # Bifurcation angle thresholds
    angle_stats = compute_linear_stats(bifurcation_angles)
    thresholds["bifurcation_angle_deg"] = {
        "unit": "deg",
        "p2_5": angle_stats["p2_5"],
        "p97_5": angle_stats["p97_5"],
        "mean": angle_stats["mean"],
        "std": angle_stats["std"],
    }

    if pca_results is not None:
        thresholds["pca"] = {
            "n_samples": pca_results["n_samples"],
            "k_retained": pca_results["k_retained"],
            "cumulative_variance_explained": float(pca_results["cumulative_explained_variance"][pca_results["k_retained"] - 1]),
        }

    return thresholds
