# LCA_topology_generator/export.py

from pathlib import Path
from typing import Dict, List, Any
import json
import numpy as np
from geomdl import BSpline, utilities

from .parameters import SIDE_BRANCH_PARAMETRIC_POSITIONS
from .topology import LCA_TOPOLOGY


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def interpolate_branch(control_points: np.ndarray, num_points: int, degree: int = 3) -> np.ndarray:
    """
    Interpolate branch control points into a smooth centerline using Cubic B-spline.
    Dynamic degree calculation prevents errors for short/low-ctrl-point branches.
    """
    curve = BSpline.Curve()
    curve.degree = min(degree, len(control_points) - 1)
    curve.ctrlpts = control_points.tolist()
    curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
    curve.sample_size = num_points
    return np.array(curve.evalpts)


def export_lca_population(
    selected_branches: List[Dict[str, np.ndarray]],
    selected_metadata: List[Dict[str, Any]],
    validation_reports: List[Dict[str, Any]],
    output_dir: str,
    lmca_points: int = 150,
    lad_points: int = 300,
    lcx_points: int = 250,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    branch_names = ["LMCA", "LAD", "LCX"]

    # 1. Export collective control points (for backward compatibility)
    for branch in branch_names:
        arr = np.stack([tree[branch] for tree in selected_branches], axis=0)
        np.save(out / f"{branch}_ctrl_points.npy", arr)

    save_json(out / "topology.json", LCA_TOPOLOGY)
    save_json(out / "parameter_log.json", selected_metadata)
    save_json(out / "validation_report.json", validation_reports)

    # 2. Export individual smooth patient centerlines
    point_counts = {
        "LMCA": lmca_points,
        "LAD": lad_points,
        "LCX": lcx_points,
    }

    for idx, (tree, metadata) in enumerate(zip(selected_branches, selected_metadata)):
        tree_id = metadata.get("tree_id", idx)
        patient_dir = out / f"patient_{tree_id:04d}"
        patient_dir.mkdir(parents=True, exist_ok=True)

        for branch in branch_names:
            cp = tree[branch]
            n_pts = point_counts[branch]
            centerline = interpolate_branch(cp, n_pts)
            np.save(patient_dir / f"{branch.lower()}_centerline.npy", centerline)

        save_json(patient_dir / "patient_info.json", metadata)


def export_side_branch_parametric_positions(output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(
        out / "side_branch_parametric_positions.json",
        SIDE_BRANCH_PARAMETRIC_POSITIONS,
    )

