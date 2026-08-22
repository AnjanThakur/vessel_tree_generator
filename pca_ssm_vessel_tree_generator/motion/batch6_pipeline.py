"""Batch 6 Pipeline Coordinator for 4D Cardiac Phase Motion Deformation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
import numpy as np

try:
    from generation.validator import minimum_interbranch_distance, minimum_nonlocal_distance
except ImportError:
    from pca_ssm_vessel_tree_generator.generation.validator import (
        minimum_interbranch_distance,
        minimum_nonlocal_distance,
    )

from motion.cardiac_motion import apply_cardiac_motion_to_tree

EPS = 1.0e-12


def _load_reference_trees(source_dir: Path) -> list[dict[str, Any]]:
    """Load either legacy Batch-5 JSON or the validated cohort layout."""
    legacy = source_dir / "synthetic_trees.json"
    if legacy.is_file():
        return json.loads(legacy.read_text(encoding="utf-8"))

    trees: list[dict[str, Any]] = []
    for tree_dir in sorted(path for path in source_dir.glob("tree_*") if path.is_dir()):
        parameter_path = tree_dir / "parameters.json"
        if not parameter_path.is_file():
            continue
        parameters = json.loads(parameter_path.read_text(encoding="utf-8"))
        ellipsoid = parameters["ellipsoid"]
        vessels = {
            name: np.load(tree_dir / f"{name}.npy", allow_pickle=False)
            for name in ("LMCA", "LAD", "LCX", "RCA")
            if (tree_dir / f"{name}.npy").is_file()
        }
        if not {"LMCA", "LAD", "LCX"}.issubset(vessels):
            raise ValueError(f"{tree_dir} is missing a mandatory LCA branch")
        trees.append({
            "tree_id": tree_dir.name,
            "ellipsoid_params": {
                "a_mm": float(ellipsoid["a"]),
                "b_mm": float(ellipsoid["b"]),
                "c_mm": float(ellipsoid["c"]),
            },
            "vessels_3d": vessels,
            "side_branches": [],
            "source_metadata": {
                "seed": parameters.get("seed"),
                "landmarks": parameters.get("landmarks", {}),
                "generation": parameters.get("generation", {}),
                "generation_mode": parameters.get("generation_mode"),
            },
        })
    if not trees:
        raise FileNotFoundError(
            f"No synthetic_trees.json or validated tree_* directories found in {source_dir}"
        )
    return trees


def _frame_collision_free(vessels: dict[str, np.ndarray], tolerance_mm: float = 0.75) -> bool:
    """Check physical nonlocal clearance without dense-sampling false positives."""
    if any(
        np.isfinite(distance := minimum_nonlocal_distance(points)) and distance < tolerance_mm
        for points in vessels.values()
    ):
        return False
    trims = {name: max(3, int(round(0.08 * len(points)))) for name, points in vessels.items()}
    pairs = [
        minimum_interbranch_distance(
            vessels["LMCA"], vessels["LAD"],
            trim_first_end=trims["LMCA"], trim_second_start=trims["LAD"],
        ),
        minimum_interbranch_distance(
            vessels["LMCA"], vessels["LCX"],
            trim_first_end=trims["LMCA"], trim_second_start=trims["LCX"],
        ),
        minimum_interbranch_distance(
            vessels["LAD"], vessels["LCX"],
            trim_first_start=trims["LAD"], trim_second_start=trims["LCX"],
        ),
    ]
    if "RCA" in vessels:
        pairs.extend(
            minimum_interbranch_distance(vessels[name], vessels["RCA"])
            for name in ("LMCA", "LAD", "LCX")
        )
    return all(not np.isfinite(distance) or distance >= tolerance_mm for distance in pairs)


def process_batch6_motion(
    batch5_dir: Path,
    output_dir: Path,
    num_phases: int = 10,
    radial_amplitude: float = 0.15,
    longitudinal_amplitude: float = 0.10,
    torsion_amplitude_deg: float = 10.0,
    peak_phase: float = 0.35,
) -> dict[str, Any]:
    """Execute Batch 6 4D cardiac phase motion pipeline across all Batch 5 accepted synthetic trees."""
    trees_batch5 = _load_reference_trees(batch5_dir)

    n_trees = len(trees_batch5)
    trees_4d = []

    bifurcation_snapped_all = True
    self_intersection_free_all = True
    bif_errors = []
    phase_metrics = []

    start_time = time.time()

    for t_idx, tree in enumerate(trees_batch5):
        t_4d = apply_cardiac_motion_to_tree(
            tree_data=tree,
            num_phases=num_phases,
            radial_amplitude=radial_amplitude,
            longitudinal_amplitude=longitudinal_amplitude,
            torsion_amplitude_deg=torsion_amplitude_deg,
            peak_phase=peak_phase,
        )
        trees_4d.append(t_4d)

        # Validate 4D invariants across all num_phases frames
        for frame in t_4d["frames"]:
            v3d = frame["vessels_3d"]
            lmca_end = v3d["LMCA"][-1]
            lad_start = v3d["LAD"][0]
            lcx_start = v3d["LCX"][0]

            d1 = float(np.linalg.norm(lad_start - lmca_end))
            d2 = float(np.linalg.norm(lcx_start - lmca_end))
            if d1 > 1.0e-6 or d2 > 1.0e-6:
                bifurcation_snapped_all = False
                bif_errors.append(f"Tree {t_4d['tree_id']} phase {frame['phase']:.2f}: d1={d1:.2e}, d2={d2:.2e}")

            if not _frame_collision_free(v3d, tolerance_mm=0.75):
                self_intersection_free_all = False

    elapsed_time = time.time() - start_time

    # Collect phase-by-phase vessel statistics
    for p_idx in range(num_phases):
        p_val = trees_4d[0]["frames"][p_idx]["phase"]
        s_val = trees_4d[0]["frames"][p_idx]["contraction_scale_s"]

        lad_lens = [float(np.sum(np.linalg.norm(np.diff(t["frames"][p_idx]["vessels_3d"]["LAD"], axis=0), axis=1))) for t in trees_4d]
        rca_lens = [
            float(np.sum(np.linalg.norm(np.diff(t["frames"][p_idx]["vessels_3d"]["RCA"], axis=0), axis=1)))
            for t in trees_4d
            if "RCA" in t["frames"][p_idx]["vessels_3d"]
        ]
        a_vals = [t["frames"][p_idx]["ellipsoid_params"]["a_mm"] for t in trees_4d]
        c_vals = [t["frames"][p_idx]["ellipsoid_params"]["c_mm"] for t in trees_4d]

        phase_metrics.append({
            "phase_index": p_idx,
            "phase": p_val,
            "contraction_scale_s": s_val,
            "mean_scaffold_a_mm": float(np.mean(a_vals)),
            "mean_scaffold_c_mm": float(np.mean(c_vals)),
            "mean_RCA_length_mm": float(np.mean(rca_lens)) if rca_lens else None,
            "mean_LAD_length_mm": float(np.mean(lad_lens)),
        })

    overall_pass = (bifurcation_snapped_all and self_intersection_free_all and (len(trees_4d) == n_trees))

    return {
        "batch": "Batch 6 — 4D Cardiac Phase Motion Deformation",
        "motion_summary": {
            "num_trees_processed": n_trees,
            "num_phases": num_phases,
            "phase_values": [f["phase"] for f in trees_4d[0]["frames"]],
            "reference_phase": 0.0,
            "peak_systole_phase": peak_phase,
            "motion_parameters": {
                "radial_amplitude": radial_amplitude,
                "longitudinal_amplitude": longitudinal_amplitude,
                "torsion_amplitude_deg": torsion_amplitude_deg,
            },
            "total_execution_time_seconds": elapsed_time,
            "mean_time_per_4d_tree_seconds": float(elapsed_time / max(n_trees, 1)),
        },
        "phase_metrics": phase_metrics,
        "trees_4d": trees_4d,
        "verification": {
            "overall_pass": overall_pass,
            "bifurcation_snapped_all_phases": bifurcation_snapped_all,
            "self_intersection_free_all_phases": self_intersection_free_all,
            "bifurcation_errors": bif_errors,
        },
    }
