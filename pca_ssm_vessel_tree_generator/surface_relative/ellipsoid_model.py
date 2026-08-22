"""Parametric triaxial support ellipsoid model layer for Batch 3.

Provides PatientEllipsoid dataclass and fit_patient_ellipsoid() coordinator
delegating 2D EllipseModel fitting to surface_relative.ellipse_fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from surface_relative.ellipse_fitting import (
    fit_coronary_ellipse,
    fit_iv_ellipse,
)

EPS = 1.0e-12


@dataclass
class PatientEllipsoid:
    """Parametric triaxial support ellipsoid centered at canonical cardiac frame origin (0,0,0).

    Equation: (x/a)^2 + (y/b)^2 + (z/c)^2 = 1
    a: Coronary ellipse semi-axis extent along X (mm)
    b: Coronary ellipse semi-axis extent along Y (mm) -> b_effective = b_cor
    c: IV ellipse semi-axis extent along Z (mm, base-to-apex)
    """

    case_id: str
    a: float
    b: float
    c: float
    center: np.ndarray  # (3,) float array in cardiac frame, strictly (0,0,0)
    b_cor: float
    b_iv: float
    b_consistency_relative_error: float
    quality_flags: list[str]
    exclusion_flags: list[str]
    raw_parameters: dict[str, Any]
    ellipse_center_separation: float = 0.0
    c_source: str = "direct_fit"
    raw_axes: dict[str, float] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return len(self.exclusion_flags) == 0 and min(self.a, self.b, self.c) > EPS


def fit_patient_ellipsoid(
    case_id: str,
    branches_cardiac: dict[str, np.ndarray],
) -> tuple[PatientEllipsoid, dict[str, Any]]:
    """Derive axis-aligned triaxial support ellipsoid parameters (a, b, c) directly from Batch-2 cardiac frame centerlines.

    Design Doc §4.1 - §4.2 & Batch 3 Instructions:
    1. Fit Coronary Ellipse on RCA+LCX cardiac centerlines -> a (X-extent), b_cor (Y-extent).
    2. Fit IV Ellipse on LAD cardiac centerline -> b_IV (Y-extent), c (Z-extent).
    3. Effective b = b_cor (as RCA+LCX forms complete 360-deg ring around AV groove).
    4. Compute b-consistency relative error = abs(b_cor - b_iv) / b_cor.
    """
    quality_flags: list[str] = []
    exclusion_flags: list[str] = []

    rca_c = branches_cardiac.get("rca", branches_cardiac.get("RCA"))
    lcx_c = branches_cardiac.get("lcx", branches_cardiac.get("LCX"))
    lad_c = branches_cardiac.get("lad", branches_cardiac.get("LAD"))

    if rca_c is None or len(rca_c) < 2 or lcx_c is None or len(lcx_c) < 2:
        exclusion_flags.append("missing_or_insufficient_coronary_ring_points")
        cor_fit = None
    else:
        try:
            cor_fit = fit_coronary_ellipse(rca_c, lcx_c)
        except Exception as exc:
            exclusion_flags.append(f"coronary_ellipse_fit_failed: {exc}")
            cor_fit = None

    if lad_c is None or len(lad_c) < 2:
        exclusion_flags.append("missing_or_insufficient_lad_points")
        iv_fit = None
    else:
        try:
            iv_fit = fit_iv_ellipse(lad_c)
        except Exception as exc:
            exclusion_flags.append(f"iv_ellipse_fit_failed: {exc}")
            iv_fit = None

    if cor_fit is None or iv_fit is None:
        return PatientEllipsoid(
            case_id=case_id,
            a=0.0,
            b=0.0,
            c=0.0,
            center=np.zeros(3, dtype=float),
            b_cor=0.0,
            b_iv=0.0,
            b_consistency_relative_error=999.0,
            quality_flags=quality_flags,
            exclusion_flags=sorted(set(exclusion_flags)),
            raw_parameters={},
        ), {}

    a = cor_fit["aligned_axes"]["a_mm"]
    b_cor = cor_fit["aligned_axes"]["b_cor_mm"]
    b_iv = iv_fit["aligned_axes"]["b_iv_mm"]
    c = iv_fit["aligned_axes"]["c_mm"]

    # Rule: Effective b = b_cor (coronary ellipse covers full AV groove)
    b_effective = b_cor
    b_error = abs(b_cor - b_iv) / max(b_cor, EPS)

    if b_error > 0.30:
        quality_flags.append("large_b_axis_discrepancy_gt_30_percent")

    if not (np.isfinite(a) and a > EPS):
        exclusion_flags.append("invalid_a_semi_axis")
    if not (np.isfinite(b_effective) and b_effective > EPS):
        exclusion_flags.append("invalid_b_semi_axis")
    if not (np.isfinite(c) and c > EPS):
        exclusion_flags.append("invalid_c_semi_axis")

    ellipsoid = PatientEllipsoid(
        case_id=case_id,
        a=a,
        b=b_effective,
        c=c,
        center=np.zeros(3, dtype=float),
        b_cor=b_cor,
        b_iv=b_iv,
        b_consistency_relative_error=b_error,
        quality_flags=quality_flags,
        exclusion_flags=sorted(set(exclusion_flags)),
        raw_parameters={
            "coronary_ellipse": cor_fit["raw_ellipse"],
            "iv_ellipse": iv_fit["raw_ellipse"],
        },
    )

    metadata = {
        "patient_id": case_id,
        "raw_coronary_ellipse": cor_fit["raw_ellipse"],
        "raw_iv_ellipse": iv_fit["raw_ellipse"],
        "cardiac_axis_aligned_ellipsoid": {
            "a_mm": a,
            "b_cor_mm": b_cor,
            "b_iv_mm": b_iv,
            "b_effective_mm": b_effective,
            "c_mm": c,
            "b_consistency_relative_error": b_error,
        },
        "quality_flags": quality_flags,
        "exclusion_flags": sorted(set(exclusion_flags)),
    }

    return ellipsoid, metadata


def derive_patient_ellipsoid(
    ppt_row: dict[str, Any],
    *,
    lad_major_z_alignment: float | None = None,
    lad_minor_z_alignment: float | None = None,
) -> PatientEllipsoid:
    """Legacy PPT CSV row adapter preserved for backward compatibility."""
    case_id = str(ppt_row.get("case_id", "unknown"))
    quality_flags: list[str] = []
    exclusion_flags: list[str] = []

    try:
        crown_a = float(ppt_row["crown_a"])
        crown_b = float(ppt_row["crown_b"])
        lad_a = float(ppt_row["lad_a"])
        lad_b = float(ppt_row["lad_b"])

        crown_center_3d = np.array(
            [
                float(ppt_row["crown_ellipse_center_3d_x"]),
                float(ppt_row["crown_ellipse_center_3d_y"]),
                float(ppt_row["crown_ellipse_center_3d_z"]),
            ],
            dtype=float,
        )
        lad_center_3d = np.array(
            [
                float(ppt_row["lad_ellipse_center_3d_x"]),
                float(ppt_row["lad_ellipse_center_3d_y"]),
                float(ppt_row["lad_ellipse_center_3d_z"]),
            ],
            dtype=float,
        )
        separation = float(np.linalg.norm(crown_center_3d - lad_center_3d))
    except (KeyError, ValueError, TypeError) as err:
        quality_flags.append(f"missing_or_invalid_field: {err}")
        return PatientEllipsoid(
            case_id=case_id,
            a=0.0,
            b=0.0,
            c=0.0,
            center=np.zeros(3),
            b_cor=0.0,
            b_iv=0.0,
            b_consistency_relative_error=999.0,
            quality_flags=quality_flags,
            exclusion_flags=["missing_or_invalid_ellipse_fields"],
            raw_parameters={},
            ellipse_center_separation=999.0,
            c_source="unavailable",
            raw_axes={},
        )

    if not (np.isfinite(crown_a) and crown_a > EPS):
        exclusion_flags.append("invalid_crown_a")
    if not (np.isfinite(crown_b) and crown_b > EPS):
        exclusion_flags.append("invalid_crown_b")

    alignments_available = lad_major_z_alignment is not None and lad_minor_z_alignment is not None
    if alignments_available and float(lad_major_z_alignment) >= float(lad_minor_z_alignment):
        aligned_c = lad_a
        aligned_name = "lad_a_axis_most_aligned_with_cardiac_z"
    elif alignments_available:
        aligned_c = lad_b
        aligned_name = "lad_b_axis_most_aligned_with_cardiac_z"
    else:
        aligned_c = max(lad_a, lad_b)
        aligned_name = "larger_lad_axis_without_orientation_metadata"
    if np.isfinite(aligned_c) and 10.0 <= aligned_c <= 200.0:
        c_candidate = aligned_c
        c_source = aligned_name
    else:
        c_candidate = min(lad_a, lad_b)
        c_source = "smaller_saved_lad_axis_stability_proxy"
        quality_flags.append("aligned_lad_axis_underconstrained_used_saved_axis_proxy")

    if not (np.isfinite(c_candidate) and c_candidate > EPS):
        exclusion_flags.append("invalid_c_axis")
    if np.isfinite(crown_a) and crown_a > 250.0:
        exclusion_flags.append("underconstrained_crown_major_axis_gt_250mm")
    if np.isfinite(crown_a) and np.isfinite(crown_b) and crown_b > EPS and crown_a / crown_b > 4.0:
        exclusion_flags.append("underconstrained_crown_axis_ratio_gt_4")
    if np.isfinite(c_candidate) and not 10.0 <= c_candidate <= 220.0:
        exclusion_flags.append("long_axis_proxy_outside_10_to_220mm")
    if separation > 150.0:
        quality_flags.append("large_ellipse_center_separation")

    return PatientEllipsoid(
        case_id=case_id,
        a=crown_a,
        b=crown_b,
        c=c_candidate,
        center=np.zeros(3, dtype=float),
        b_cor=crown_b,
        b_iv=min(lad_a, lad_b),
        b_consistency_relative_error=abs(crown_b - min(lad_a, lad_b)) / max(crown_b, EPS),
        quality_flags=quality_flags,
        exclusion_flags=sorted(set(exclusion_flags)),
        raw_parameters={"crown_a": crown_a, "crown_b": crown_b, "lad_a": lad_a, "lad_b": lad_b},
        ellipse_center_separation=separation,
        c_source=c_source,
        raw_axes={"crown_a": crown_a, "crown_b": crown_b, "lad_a": lad_a, "lad_b": lad_b},
    )
