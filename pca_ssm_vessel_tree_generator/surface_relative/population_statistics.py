"""Population statistics and joint fixed-branch deviation PCA.

Implements Part 5 (§5.1 – §5.6) of the Technical Design Document.
"""

from __future__ import annotations

import math
from typing import Any, Iterable
import numpy as np
from scipy.stats import circmean, circstd

EPS = 1.0e-12


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
    """Fit joint deviation PCA across a fixed branch representation.

    Design Doc §5.3.3:
    - Input shape vectors X: (N, 126)
    - Mean shape vector mu: (126,)
    - Center shape vectors: X_centered = X - mu
    - SVD: U, S, Vt = SVD(X_centered)
    - Eigenvalues: lambda_j = S_j^2 / (N - 1)
    - Explained variance ratio: r_j = S_j^2 / sum(S^2)
    - Retain k components explaining >= 95% cumulative variance
    """
    X = np.asarray(shape_vectors_matrix, dtype=float)
    n_samples, n_features = X.shape
    if X.ndim != 2 or n_samples < 2 or n_features < 3 or n_features % 3 != 0:
        raise ValueError(
            "PCA requires at least 2 complete cases and a positive multiple-of-3 "
            f"feature dimension; got shape {X.shape}"
        )

    mean_vector = np.mean(X, axis=0)
    centered = X - mean_vector

    # SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    total_var = float(np.sum(S**2))
    if total_var <= EPS:
        explained_ratio = np.ones(len(S)) / len(S)
    else:
        explained_ratio = (S**2) / total_var

    cum_explained_variance = np.cumsum(explained_ratio)
    k_retained = int(np.searchsorted(cum_explained_variance, variance_cutoff) + 1)
    k_retained = min(k_retained, len(S))

    components_retained = Vt[:k_retained]  # (k, 126)
    singular_values_retained = S[:k_retained]
    eigenvalues = (S**2) / max(n_samples - 1, 1)  # lambda_j = S_j^2 / (N - 1)
    eigenvalues_retained = eigenvalues[:k_retained]

    training_scores = centered @ components_retained.T  # (N, k)
    retained_scales = np.sqrt(np.maximum(eigenvalues_retained, 0.0))
    standardized_training_scores = np.divide(
        training_scores,
        retained_scales,
        out=np.zeros_like(training_scores),
        where=retained_scales > EPS,
    )

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "mean_vector": mean_vector,
        "components_all": Vt,
        "components_retained": components_retained,
        "training_scores": training_scores,
        "standardized_training_scores": standardized_training_scores,
        "singular_values": S,
        "eigenvalues": eigenvalues,
        "eigenvalues_retained": eigenvalues_retained,
        "explained_variance_ratio": explained_ratio,
        "cumulative_explained_variance": cum_explained_variance,
        "k_retained": k_retained,
        "variance_cutoff": variance_cutoff,
    }


def compute_landmark_statistics(landmarks_matrix: np.ndarray) -> dict[str, Any]:
    """Compute Level 2 Landmark Statistics (18-D vector) across population (Design Doc §5.2).

    Vector ordering (18-D):
    [lca_ost_u, lca_ost_v, lca_ost_off,
     rca_ost_u, rca_ost_v, rca_ost_off,
     bif_u, bif_v, bif_off,
     lad_end_u, lad_end_v, lad_end_off,
     lcx_end_u, lcx_end_v, lcx_end_off,
     rca_end_u, rca_end_v, rca_end_off]
    """
    L = np.asarray(landmarks_matrix, dtype=float)
    n_samples, n_dims = L.shape
    if n_dims != 18:
        raise ValueError(f"Landmark matrix must have shape (N, 18); got {L.shape}")

    mean_vec = np.mean(L, axis=0)
    std_vec = np.std(L, axis=0, ddof=1) if n_samples > 1 else np.zeros(18)

    centered = L - mean_vec
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    total_var = float(np.sum(S**2))
    explained_ratio = (S**2) / total_var if total_var > EPS else np.ones(18) / 18
    cum_var = np.cumsum(explained_ratio)

    return {
        "n_samples": n_samples,
        "mean_18d": mean_vec.tolist(),
        "std_18d": std_vec.tolist(),
        "singular_values": S.tolist(),
        "explained_variance_ratio": explained_ratio.tolist(),
        "cumulative_explained_variance": cum_var.tolist(),
        "components_all": Vt.tolist(),
    }


def compute_side_branch_statistics(patient_branches_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute Level 5 Side Branch Statistics across population (Design Doc §5.5).

    Statistics per parent vessel (RCA, LMCA, LAD, LCX):
    - count: number of side branches per parent vessel
    - attachment_t: parametric attachment position t in [0, 1]
    - branch_length_mm: side branch length in mm
    - branch_angle_deg: relative branch angle in degrees
    """
    results = {}
    for vessel in ["RCA", "LMCA", "LAD", "LCX"]:
        counts = []
        attachments_t = []
        lengths_mm = []
        angles_deg = []

        for p_data in patient_branches_data:
            v_data = p_data.get(vessel, {})
            counts.append(v_data.get("count", 0))
            attachments_t.extend(v_data.get("attachment_t", []))
            lengths_mm.extend(v_data.get("length_mm", []))
            angles_deg.extend(v_data.get("angle_deg", []))

        results[vessel] = {
            "branch_count": compute_linear_stats(counts),
            "attachment_t": compute_linear_stats(attachments_t),
            "branch_length_mm": compute_linear_stats(lengths_mm),
            "branch_angle_deg": compute_linear_stats(angles_deg),
        }
    return results


def measure_tortuosity(u_path: np.ndarray, v_path: np.ndarray) -> float:
    """Measure vessel tortuosity in parameter space (Design Doc §5.4.1).

    Computes standard deviation of deviation from a straight line in (u, v) parameter space:
    T = sqrt(std(u - u_linear)^2 + std(v - v_linear)^2)
    """
    u_p = np.asarray(u_path, dtype=float)
    v_p = np.asarray(v_path, dtype=float)
    n_pts = len(u_p)
    if n_pts < 2:
        return 0.0

    t = np.linspace(0.0, 1.0, n_pts)
    u_linear = u_p[0] + (u_p[-1] - u_p[0]) * t
    v_linear = v_p[0] + (v_p[-1] - v_p[0]) * t

    u_dev = float(np.std(u_p - u_linear))
    v_dev = float(np.std(v_p - v_linear))
    return float(math.sqrt(u_dev**2 + v_dev**2))


def measure_obliquity(u_path: np.ndarray) -> float:
    """Measure vessel obliquity (Design Doc §5.4.2).

    Obliquity = total drift in u from start to end: Omega = u_end - u_start
    """
    u_p = np.asarray(u_path, dtype=float)
    if len(u_p) < 2:
        return 0.0
    return float(u_p[-1] - u_p[0])


def build_validation_thresholds(
    scaffold_params: list[dict[str, float]],
    branch_lengths: dict[str, list[float]],
    bifurcation_angles: list[float],
    max_out_of_plane_devs: dict[str, list[float]] | None = None,
    tortuosities: dict[str, list[float]] | None = None,
    pca_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute empirical P2.5 - P97.5 population validation thresholds (Design Doc §5.6)."""
    thresholds = {}

    # Scaffold semi-axes thresholds
    for key in ["a", "b", "c"]:
        vals = [p[key] for p in scaffold_params if key in p]
        stats = compute_linear_stats(vals)
        thresholds[f"ellipsoid_{key}_mm"] = {
            "unit": "mm",
            "min": stats["min"],
            "max": stats["max"],
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
            "min": stats["min"],
            "max": stats["max"],
            "p2_5": stats["p2_5"],
            "p97_5": stats["p97_5"],
            "mean": stats["mean"],
            "std": stats["std"],
        }

    # Bifurcation angle thresholds
    angle_stats = compute_linear_stats(bifurcation_angles)
    thresholds["bifurcation_angle_deg"] = {
        "unit": "deg",
        "min": angle_stats["min"],
        "max": angle_stats["max"],
        "p2_5": angle_stats["p2_5"],
        "p97_5": angle_stats["p97_5"],
        "mean": angle_stats["mean"],
        "std": angle_stats["std"],
    }

    # Max out-of-plane deviation thresholds
    if max_out_of_plane_devs is not None:
        for vessel, devs in max_out_of_plane_devs.items():
            stats = compute_linear_stats(devs)
            thresholds[f"max_out_of_plane_{vessel}_mm"] = {
                "unit": "mm",
                "min": stats["min"],
                "max": stats["max"],
                "p2_5": stats["p2_5"],
                "p97_5": stats["p97_5"],
                "mean": stats["mean"],
                "std": stats["std"],
            }

    # Tortuosity thresholds
    if tortuosities is not None:
        for vessel, torts in tortuosities.items():
            stats = compute_linear_stats(torts)
            thresholds[f"tortuosity_{vessel}"] = {
                "unit": "dimensionless",
                "min": stats["min"],
                "max": stats["max"],
                "p2_5": stats["p2_5"],
                "p97_5": stats["p97_5"],
                "mean": stats["mean"],
                "std": stats["std"],
            }

    if pca_results is not None:
        thresholds["pca"] = {
            "n_samples": pca_results["n_samples"],
            "k_retained": pca_results["k_retained"],
            "cumulative_variance_explained": float(pca_results["cumulative_explained_variance"][pca_results["k_retained"] - 1]),
        }

    return thresholds
