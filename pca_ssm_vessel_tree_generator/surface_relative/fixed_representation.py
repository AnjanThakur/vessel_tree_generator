"""Fixed-point surface representations for LCA and optional RCA models.

The primary statistical contract is LCA-only: LMCA 5 + LAD 12 + LCX 10
points, yielding 81 dimensions.  A legacy 126-D four-vessel vector is still
created when an independently resolved RCA is available.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from surface_relative.surface_projection import project_point_to_surface

FIXED_COUNTS = {
    "LMCA": 5,
    "LAD": 12,
    "LCX": 10,
    "RCA": 15,
}
TOTAL_FIXED_POINTS = sum(FIXED_COUNTS.values())  # 42
TOTAL_SHAPE_DIMENSIONS = TOTAL_FIXED_POINTS * 3  # 126
LCA_BRANCH_ORDER = ("LMCA", "LAD", "LCX")
LCA_FIXED_COUNTS = {name: FIXED_COUNTS[name] for name in LCA_BRANCH_ORDER}
LCA_TOTAL_FIXED_POINTS = sum(LCA_FIXED_COUNTS.values())  # 27
LCA_TOTAL_SHAPE_DIMENSIONS = LCA_TOTAL_FIXED_POINTS * 3  # 81


def resample_centerline_arc_length(points: np.ndarray, num_target_points: int) -> np.ndarray:
    """Arc-length resample a 3D centerline to a fixed number of control points."""
    points = np.asarray(points, dtype=float)
    n_pts = len(points)
    if n_pts == 0:
        raise ValueError("cannot resample empty points array")
    if n_pts == 1:
        return np.tile(points, (num_target_points, 1))

    # Compute cumulative arc lengths
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_length = np.zeros(n_pts, dtype=float)
    cum_length[1:] = np.cumsum(seg_lengths)
    total_length = cum_length[-1]

    if total_length <= 1.0e-12:
        return np.tile(points[0], (num_target_points, 1))

    normalized_cum = cum_length / total_length
    target_s = np.linspace(0.0, 1.0, num_target_points)

    resampled = np.zeros((num_target_points, 3), dtype=float)
    for dim in range(3):
        resampled[:, dim] = np.interp(target_s, normalized_cum, points[:, dim])

    return resampled


def build_patient_fixed_representation(
    branches_cardiac: dict[str, np.ndarray],
    a: float,
    b: float,
    c: float,
) -> dict[str, Any]:
    """Construct the primary LCA and optional legacy four-vessel representations.

    Primary LCA contract:
    - LMCA: 5 points
    - LAD: 12 points
    - LCX: 10 points
    - 27 points x 3 local deviations = 81 dimensions

    If an independently resolved RCA is supplied, the function also emits the
    legacy RCA/LMCA/LAD/LCX 42-point, 126-dimensional representation. Missing
    RCA data never prevents construction of the primary LCA model.
    """
    fixed_data = {}
    deviations_dict = {}

    has_rca = "rca" in branches_cardiac and branches_cardiac["rca"] is not None and len(branches_cardiac["rca"]) >= 5

    for vessel_name, count in FIXED_COUNTS.items():
        key = vessel_name.lower()
        if key in branches_cardiac and branches_cardiac[key] is not None and len(branches_cardiac[key]) >= 2:
            pts_cardiac = resample_centerline_arc_length(branches_cardiac[key], count)
        else:
            pts_cardiac = None

        if pts_cardiac is not None:
            u_arr = np.zeros(count, dtype=float)
            v_arr = np.zeros(count, dtype=float)
            off_arr = np.zeros(count, dtype=float)
            dev_arr = np.zeros((count, 3), dtype=float)
            for idx in range(count):
                u, v, off, dev = project_point_to_surface(pts_cardiac[idx], a, b, c)
                u_arr[idx] = u
                v_arr[idx] = v
                off_arr[idx] = off
                dev_arr[idx] = dev

            fixed_data[vessel_name] = {
                "cardiac_points": pts_cardiac,
                "u": u_arr,
                "v": v_arr,
                "offset": off_arr,
                "uvo": np.column_stack([u_arr, v_arr, off_arr]),
            }
            deviations_dict[vessel_name] = dev_arr
        else:
            fixed_data[vessel_name] = None
            deviations_dict[vessel_name] = None

    # The LCA-only model is the primary contract and must never depend on the
    # inferred/disconnected RCA candidate.
    is_lca_complete = all(deviations_dict[v] is not None for v in LCA_BRANCH_ORDER)
    if is_lca_complete:
        lca_shape_vector = np.concatenate(
            [deviations_dict[name].flatten() for name in LCA_BRANCH_ORDER]
        )
        lca_uvo_matrix = np.vstack([fixed_data[name]["uvo"] for name in LCA_BRANCH_ORDER])
    else:
        lca_shape_vector = None
        lca_uvo_matrix = None

    # Construct the legacy 126-D shape vector only if all four vessels exist.
    is_complete = all(deviations_dict[v] is not None for v in ["RCA", "LMCA", "LAD", "LCX"])
    if is_complete:
        shape_vector = np.concatenate(
            [
                deviations_dict["RCA"].flatten(),  # 15 * 3 = 45
                deviations_dict["LMCA"].flatten(), # 5 * 3  = 15
                deviations_dict["LAD"].flatten(),  # 12 * 3 = 36
                deviations_dict["LCX"].flatten(),  # 10 * 3 = 30
            ]
        )
    else:
        shape_vector = None

    # Matrix representation of 42 x 3 (u, v, offset) for this patient if complete
    if is_complete:
        matrix_42_3 = np.vstack(
            [
                fixed_data["RCA"]["uvo"],
                fixed_data["LMCA"]["uvo"],
                fixed_data["LAD"]["uvo"],
                fixed_data["LCX"]["uvo"],
            ]
        )
    else:
        matrix_42_3 = None

    return {
        "fixed_branches": fixed_data,
        "deviations": deviations_dict,
        "is_complete": is_complete,
        "is_lca_complete": is_lca_complete,
        "has_rca": has_rca,
        "shape_vector": shape_vector,
        "uvo_matrix_42_3": matrix_42_3,
        "lca_shape_vector": lca_shape_vector,
        "lca_uvo_matrix_27_3": lca_uvo_matrix,
        "primary_branch_order": list(LCA_BRANCH_ORDER),
        "primary_branch_counts": LCA_FIXED_COUNTS.copy(),
    }
