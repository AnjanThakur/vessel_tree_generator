"""Coronary and Interventricular Plane Fitting Module for Batch 2.

Implements Part 2.1 & 2.2 of the Technical Design Document:
- fit_coronary_plane(): Least-squares SVD plane fit on RCA + LCX centerlines.
  Extracts raw least-variance vector from SVD (bypassing canonicalize_direction)
  and explicitly reorients normal via LAD descent vector (v_LAD . n_cor < 0).
- compute_iv_normal(): Derives Interventricular (IV) plane normal from
  cross product unit(n_cor x v_LAD), checking orthogonality error.
"""

from __future__ import annotations

from typing import Any
import numpy as np

EPS = 1.0e-12


def fit_coronary_plane(
    rca_points: np.ndarray,
    lcx_points: np.ndarray,
    lad_points: np.ndarray,
    min_points: int = 6,
) -> dict[str, Any]:
    """Fit Coronary (AV-groove) plane from RCA and LCX centerline points using SVD.

    Design Doc §2.1 & §5:
    1. Combine RCA and LCX points to form coronary ring.
    2. Compute centroid_cor = mean(ring_points).
    3. Fit plane via SVD: least-variance direction = vh[-1].
    4. Orient coronary_normal using LAD descent vector so dot(lad_dir, coronary_normal) < 0.
    """
    rca_pts = np.asarray(rca_points, dtype=float)
    lcx_pts = np.asarray(lcx_points, dtype=float)
    lad_pts = np.asarray(lad_points, dtype=float)

    if rca_pts.ndim != 2 or rca_pts.shape[1] != 3 or len(rca_pts) < 2:
        raise ValueError("rca_points must have shape (N, 3) with N >= 2")
    if lcx_pts.ndim != 2 or lcx_pts.shape[1] != 3 or len(lcx_pts) < 2:
        raise ValueError("lcx_points must have shape (N, 3) with N >= 2")
    if lad_pts.ndim != 2 or lad_pts.shape[1] != 3 or len(lad_pts) < 2:
        raise ValueError("lad_points must have shape (N, 3) with N >= 2")

    ring_points = np.vstack([rca_pts, lcx_pts])
    if len(ring_points) < min_points:
        raise ValueError(f"ring points require at least {min_points} total points; got {len(ring_points)}")

    if not np.all(np.isfinite(ring_points)) or not np.all(np.isfinite(lad_pts)):
        raise ValueError("non-finite (NaN or Inf) coordinates in vessel point cloud")

    centroid = ring_points.mean(axis=0)
    centered = ring_points - centroid

    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    scale = max(float(singular_values[0]), EPS)
    rank = int(np.sum(singular_values > scale * 1.0e-8))

    if rank < 2:
        raise ValueError("ring point cloud is collinear; a stable coronary plane cannot be fitted")

    raw_svd_normal = vh[-1]
    norm_val = np.linalg.norm(raw_svd_normal)
    if norm_val <= EPS:
        raise ValueError("SVD returned zero-length normal vector")
    raw_svd_normal = raw_svd_normal / norm_val

    # LAD descent vector
    lad_dir = lad_pts[-1] - lad_pts[0]
    lad_dir_norm = np.linalg.norm(lad_dir)
    if lad_dir_norm <= EPS:
        raise ValueError("LAD direction vector has zero length")

    dot_before = float(np.dot(lad_dir, raw_svd_normal))

    # Orient coronary_normal so LAD descends in negative Z (v_LAD . n_cor < 0)
    if dot_before > 0.0:
        coronary_normal = -raw_svd_normal
        dot_after = -dot_before
    else:
        coronary_normal = raw_svd_normal
        dot_after = dot_before

    signed_distances = centered @ coronary_normal
    residuals = np.abs(signed_distances)

    return {
        "coronary_centroid": centroid,
        "raw_svd_normal": raw_svd_normal,
        "coronary_normal": coronary_normal,
        "lad_direction": lad_dir,
        "dot_before_correction": dot_before,
        "dot_after_correction": dot_after,
        "rms_residual_mm": float(np.sqrt(np.mean(signed_distances**2))),
        "mean_residual_mm": float(np.mean(residuals)),
        "max_residual_mm": float(np.max(residuals)),
        "singular_values": singular_values.tolist(),
        "rank": rank,
        "num_rca_points": len(rca_pts),
        "num_lcx_points": len(lcx_pts),
    }


def compute_iv_normal(
    coronary_normal: np.ndarray,
    lad_points: np.ndarray,
) -> dict[str, Any]:
    """Derive Interventricular (IV) plane normal from coronary plane normal and LAD direction.

    Design Doc §2.2 & §7-8:
    IV normal = unit(coronary_normal x LAD_direction)
    """
    cor_norm = np.asarray(coronary_normal, dtype=float).reshape(3)
    cor_norm_len = np.linalg.norm(cor_norm)
    if cor_norm_len <= EPS:
        raise ValueError("coronary_normal has zero length")
    cor_norm = cor_norm / cor_norm_len

    lad_pts = np.asarray(lad_points, dtype=float)
    if lad_pts.ndim != 2 or lad_pts.shape[1] != 3 or len(lad_pts) < 2:
        raise ValueError("lad_points must have shape (N, 3) with N >= 2")

    lad_vec = lad_pts[-1] - lad_pts[0]
    lad_len = np.linalg.norm(lad_vec)
    if lad_len <= EPS:
        raise ValueError("LAD direction vector has zero length")
    lad_unit = lad_vec / lad_len

    iv_cross = np.cross(cor_norm, lad_unit)
    cross_len = np.linalg.norm(iv_cross)
    if cross_len <= EPS:
        raise ValueError("coronary_normal and LAD direction vector are collinear")

    iv_normal = iv_cross / cross_len
    ortho_error = float(abs(np.dot(iv_normal, cor_norm)))

    return {
        "iv_normal": iv_normal,
        "lad_unit_vector": lad_unit,
        "cross_product_norm": float(cross_len),
        "orthogonality_error": ortho_error,
    }
