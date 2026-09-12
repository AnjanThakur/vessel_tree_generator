"""Batch 3 Pipeline Coordinator for Ellipsoid Fitting & Parametric Surface Projection."""

from __future__ import annotations

from typing import Any
import numpy as np

from surface_relative.ellipsoid_model import fit_patient_ellipsoid
from surface_relative.surface_projection import (
    ellipsoid_point,
    project_centerline_to_surface,
)
from surface_relative.fixed_representation import (
    build_patient_fixed_representation,
)


def process_patient_batch3(
    centerlines_cardiac: dict[str, np.ndarray],
    landmarks_cardiac: dict[str, np.ndarray],
    patient_id: str,
) -> dict[str, Any]:
    """Execute Batch 3 pipeline per patient:

    1. Fit 2D Coronary Ellipse & IV Ellipse -> Derive axis-aligned triaxial ellipsoid (a, b, c).
    2. Calculate b-axis consistency relative error.
    3. Project centerlines (RCA, LMCA, LAD, LCX) onto ellipsoid -> (u, v, offset, local_deviations).
    4. Construct fixed 42-point representation & 126-D shape vector.
    5. Perform strict Batch-3 validation checks.
    """
    # Normalize dictionary keys so both upper and lower case keys exist
    branches_norm = {}
    for k, v in centerlines_cardiac.items():
        if v is not None:
            branches_norm[k.lower()] = v
            branches_norm[k.upper()] = v

    # 1. Fit Ellipsoid
    ellipsoid, ellipsoid_meta = fit_patient_ellipsoid(patient_id, branches_norm)

    if not ellipsoid.is_valid:
        return {
            "patient_id": patient_id,
            "batch3_status": "ellipsoid_fit_failed",
            "batch3_passed": False,
            "rejection_reasons": [f"Ellipsoid fit failed: {ellipsoid.exclusion_flags}"],
        }

    a, b, c = ellipsoid.a, ellipsoid.b, ellipsoid.c

    # 2. Project full centerlines onto ellipsoid
    projections = {}
    for name in ("RCA", "LMCA", "LAD", "LCX"):
        pts = branches_norm.get(name.lower())
        if pts is not None and len(pts) > 0:
            proj_data = project_centerline_to_surface(pts, a, b, c)

            # Evaluate surface points & surface equation residuals
            n_pts = len(pts)
            surf_pts = np.zeros((n_pts, 3), dtype=float)
            quad_residuals = np.zeros(n_pts, dtype=float)
            for idx in range(n_pts):
                u_val = proj_data["u"][idx]
                v_val = proj_data["v"][idx]
                sp = ellipsoid_point(u_val, v_val, a, b, c)
                surf_pts[idx] = sp
                quad_residuals[idx] = (sp[0] / a) ** 2 + (sp[1] / b) ** 2 + (sp[2] / c) ** 2

            projections[name] = {
                "u": proj_data["u"],
                "v": proj_data["v"],
                "offset": proj_data["offset"],
                "deviations": proj_data["deviations"],
                "surface_points": surf_pts,
                "quad_residuals": quad_residuals,
            }
        else:
            projections[name] = None

    # 3. Construct Fixed 42-Point Representation & 126-D Shape Vector
    fixed_res = build_patient_fixed_representation(branches_norm, a, b, c)

    # 4. Perform Batch-3 Invariant & Geometric Validation
    validation_failures = []

    if not (a > 0.0 and b > 0.0 and c > 0.0):
        validation_failures.append(f"Invalid ellipsoid semi-axes: a={a:.2f}, b={b:.2f}, c={c:.2f}")

    if not fixed_res["is_complete"] or fixed_res["shape_vector"] is None:
        validation_failures.append("Incomplete fixed representation: one or more vessels missing")

    if fixed_res["shape_vector"] is not None:
        shape_vector = fixed_res["shape_vector"]
        if shape_vector.shape != (126,):
            validation_failures.append(f"Invalid shape vector shape: {shape_vector.shape} (expected (126,))")
        if not np.all(np.isfinite(shape_vector)):
            validation_failures.append("Non-finite (NaN or Inf) values in 126-D shape vector")

    # Check surface equation residual (x/a)^2 + (y/b)^2 + (z/c)^2 ≈ 1 for evaluated surface points
    max_quad_err = 0.0
    for p_name, p_data in projections.items():
        if p_data is not None:
            err = float(np.max(np.abs(p_data["quad_residuals"] - 1.0)))
            max_quad_err = max(max_quad_err, err)

    if max_quad_err > 1.0e-5:
        validation_failures.append(f"Surface point quadratic residual error ({max_quad_err:.2e}) exceeds 1e-5 threshold")

    batch3_passed = (len(validation_failures) == 0)

    return {
        "patient_id": patient_id,
        "batch3_status": "surface_projection_succeeded" if batch3_passed else "surface_projection_failed",
        "batch3_passed": batch3_passed,
        "rejection_reasons": validation_failures,
        "ellipsoid": ellipsoid,
        "ellipsoid_metadata": ellipsoid_meta,
        "projections": projections,
        "fixed_representation": fixed_res,
        "validation": {
            "overall_pass": batch3_passed,
            "max_surface_equation_error": max_quad_err,
            "b_consistency_relative_error": ellipsoid.b_consistency_relative_error,
            "shape_vector_dim": fixed_res["shape_vector"].shape[0] if fixed_res["shape_vector"] is not None else 0,
            "failures": validation_failures,
        },
    }
