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

    @property
    def is_valid(self) -> bool:
        return len(self.quality_flags) == 0 and min(self.a, self.b, self.c) > EPS


def derive_patient_ellipsoid(ppt_row: dict[str, Any]) -> PatientEllipsoid:
    """Derive 3D triaxial support ellipsoid parameters (a, b, c) from a PPT parameter CSV row.

    Design Doc §4.1 & §4.2:
    - a: crown ellipse semi-axis 1 (x direction) -> crown_a
    - b: crown ellipse semi-axis 2 (y direction) -> crown_b
    - c: IV ellipse semi-axis 2 (z direction, base-to-apex) -> lad_b (or lad_a depending on axial orientation)
    """
    case_id = str(ppt_row.get("case_id", "unknown"))
    quality_flags = []

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
        )

    # Validate semi-axes
    if not (np.isfinite(crown_a) and crown_a > EPS):
        quality_flags.append("invalid_crown_a")
    if not (np.isfinite(crown_b) and crown_b > EPS):
        quality_flags.append("invalid_crown_b")

    # c represents base-to-apex semi-axis from IV plane ellipse.
    # Take the smaller non-degenerate semi-axis of IV fit if one is hugely unconstrained (e.g. > 500mm), or lad_b.
    c_candidate = min(lad_a, lad_b) if min(lad_a, lad_b) > 5.0 else max(lad_a, lad_b)
    if c_candidate > 300.0 or c_candidate < 10.0:
        # Fallback to mean crown radius if IV plane fit is ill-conditioned
        quality_flags.append("unreasonable_lad_ellipse_c_axis")
        c_candidate = 0.5 * (crown_a + crown_b)

    if not (np.isfinite(c_candidate) and c_candidate > EPS):
        quality_flags.append("invalid_c_axis")

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
    )
