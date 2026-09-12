"""Geometry Resampling and .npy Exporter for Batch 7."""

from __future__ import annotations

from typing import Any
import numpy as np

EPS = 1.0e-12


def resample_curve_3d(pts: np.ndarray, num_points: int = 50) -> np.ndarray:
    """Deterministically resample 3D centerline curve along cumulative arc-length to num_points with exact endpoint preservation."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) == 0:
        return np.zeros((num_points, 3), dtype=float)
    if len(pts) == 1:
        return np.tile(pts[0], (num_points, 1))

    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_lens = np.insert(np.cumsum(seg_lens), 0, 0.0)
    total_len = float(cum_lens[-1])

    if total_len < EPS:
        res = np.tile(pts[0], (num_points, 1))
        res[-1] = pts[-1]
        return res

    s_target = np.linspace(0.0, total_len, num_points)
    resampled = np.zeros((num_points, 3), dtype=float)

    for dim in range(3):
        resampled[:, dim] = np.interp(s_target, cum_lens, pts[:, dim])

    # Preserve exact endpoints
    resampled[0] = pts[0]
    resampled[-1] = pts[-1]

    return resampled


def compute_vessel_radius(vessel_name: str, num_points: int = 50) -> np.ndarray:
    """Compute anatomical vessel radius r(t) along normalized arc-length t in [0, 1] (Design Doc §8.1 & §8.5).

    - LMCA: 2.0 mm (constant)
    - Major Vessels (RCA, LAD, LCX): 1.8 -> 0.8 mm (linear tapering)
    - Side Branches (SB_*): 0.6 -> 0.4 mm (linear tapering)
    """
    t = np.linspace(0.0, 1.0, num_points)

    if vessel_name == "LMCA":
        return np.full(num_points, 2.0, dtype=float)
    elif vessel_name in ("RCA", "LAD", "LCX"):
        return 1.8 - 1.0 * t
    else:  # Side branches
        return 0.6 - 0.2 * t


def get_ordered_tree_branches(
    vessels_3d: dict[str, np.ndarray], side_branches: list[dict[str, Any]]
) -> list[tuple[str, np.ndarray]]:
    """Return deterministic ordered list of vessel branches [(name, points_3d), ...].

    Ordering:
    0 = RCA
    1 = LMCA
    2 = LAD
    3 = LCX
    4+ = side branches sorted by parent (LAD -> LCX -> RCA) and attachment_index
    """
    ordered_branches = []

    # Major vessels in deterministic order. Optional RCA is omitted instead
    # of being represented as a zero-length placeholder branch.
    for vname in ("RCA", "LMCA", "LAD", "LCX"):
        pts = vessels_3d.get(vname)
        if pts is not None:
            ordered_branches.append((vname, np.asarray(pts, dtype=float)))

    # Sort side branches deterministically
    parent_order = {"LAD": 0, "LCX": 1, "RCA": 2}
    sorted_sbs = sorted(
        side_branches,
        key=lambda sb: (parent_order.get(sb.get("parent", "LAD"), 99), sb.get("attachment_index", 0)),
    )

    for idx, sb in enumerate(sorted_sbs):
        sb_name = f"SB_{sb.get('parent', 'LAD')}_{idx:02d}"
        pts = sb.get("points", np.zeros((2, 3), dtype=float))
        ordered_branches.append((sb_name, np.asarray(pts, dtype=float)))

    return ordered_branches


def construct_geometry_static_array(tree_4d: dict[str, Any], num_points: int = 50) -> np.ndarray:
    """Construct static geometry array of shape (M_branches x num_points x 4) containing [x, y, z, r] at phi=0.0."""
    ref_frame = tree_4d["frames"][0]
    ordered_branches = get_ordered_tree_branches(ref_frame["vessels_3d"], ref_frame["side_branches"])

    m_branches = len(ordered_branches)
    geom_static = np.zeros((m_branches, num_points, 4), dtype=float)

    for b_idx, (b_name, b_pts) in enumerate(ordered_branches):
        res_3d = resample_curve_3d(b_pts, num_points=num_points)
        radii = compute_vessel_radius(b_name, num_points=num_points)

        geom_static[b_idx, :, 0:3] = res_3d
        geom_static[b_idx, :, 3] = radii

    return geom_static


def construct_geometry_cine_array(tree_4d: dict[str, Any], num_points: int = 50) -> np.ndarray:
    """Construct 4D cine geometry array of shape (T_phases x M_branches x num_points x 4) containing [x, y, z, r].

    Radii r remain 100% constant across all 10 cine phases per TDD §7.5.
    """
    frames = tree_4d["frames"]
    t_phases = len(frames)

    ref_frame = frames[0]
    ordered_ref_branches = get_ordered_tree_branches(ref_frame["vessels_3d"], ref_frame["side_branches"])
    m_branches = len(ordered_ref_branches)

    geom_cine = np.zeros((t_phases, m_branches, num_points, 4), dtype=float)

    for f_idx, frame in enumerate(frames):
        ordered_branches = get_ordered_tree_branches(frame["vessels_3d"], frame["side_branches"])

        for b_idx, (b_name, b_pts) in enumerate(ordered_branches):
            res_3d = resample_curve_3d(b_pts, num_points=num_points)
            radii = compute_vessel_radius(b_name, num_points=num_points)  # Constant across phases

            geom_cine[f_idx, b_idx, :, 0:3] = res_3d
            geom_cine[f_idx, b_idx, :, 3] = radii

    return geom_cine
