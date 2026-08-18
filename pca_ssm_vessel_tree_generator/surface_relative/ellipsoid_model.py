"""Parametric triaxial support ellipsoid derivation from PPT two-ellipse outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

EPS = 1.0e-12


@dataclass
class PatientEllipsoid:
    """Parametric triaxial support ellipsoid centered at the global cardiac frame origin (0,0,0).

    Equation: (x/a)^2 + (y/b)^2 + (z/c)^2 = 1
    a: crown ellipse semi-axis along X
    b: crown ellipse semi-axis along Y
    c: IV ellipse semi-axis along Z (base-to-apex)
    """

    case_id: str
    a: float
    b: float
    c: float
    center: np.ndarray  # (3,) float array in cardiac frame, expected near (0,0,0)
    ellipse_center_separation: float
    quality_flags: list[str]
    exclusion_flags: list[str]
    c_source: str
    raw_axes: dict[str, float]

    @property
    def is_valid(self) -> bool:
        return len(self.exclusion_flags) == 0 and min(self.a, self.b, self.c) > EPS


def derive_patient_ellipsoid(
    ppt_row: dict[str, Any],
    *,
    lad_major_z_alignment: float | None = None,
    lad_minor_z_alignment: float | None = None,
) -> PatientEllipsoid:
    """Derive 3D triaxial support ellipsoid parameters (a, b, c) from a PPT parameter CSV row.

    Design Doc §4.1 & §4.2:
    - a: crown ellipse semi-axis 1 (x direction) -> crown_a
    - b: crown ellipse semi-axis 2 (y direction) -> crown_b
    - c: IV ellipse semi-axis 2 (z direction, base-to-apex) -> lad_b (or lad_a depending on axial orientation)
    """
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
            ellipse_center_separation=999.0,
            quality_flags=quality_flags,
            exclusion_flags=["missing_or_invalid_ellipse_fields"],
            c_source="unavailable",
            raw_axes={},
        )

    # Validate semi-axes
    if not (np.isfinite(crown_a) and crown_a > EPS):
        exclusion_flags.append("invalid_crown_a")
    if not (np.isfinite(crown_b) and crown_b > EPS):
        exclusion_flags.append("invalid_crown_b")

    # Prefer the fitted LAD ellipse axis most aligned with cardiac Z. A partial
    # arc can make that major axis numerically unbounded, so the other *saved*
    # fitted semi-axis is used as a declared stability proxy when the aligned
    # axis is outside a broad anatomical display range. No ellipse is refitted
    # and no saved PPT parameter is overwritten.
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
        ellipse_center_separation=separation,
        quality_flags=quality_flags,
        exclusion_flags=sorted(set(exclusion_flags)),
        c_source=c_source,
        raw_axes={"crown_a": crown_a, "crown_b": crown_b, "lad_a": lad_a, "lad_b": lad_b},
    )
