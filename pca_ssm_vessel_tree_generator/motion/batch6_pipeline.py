"""Batch 6 Pipeline Coordinator for 4D Cardiac Phase Motion Deformation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
import numpy as np

try:
    from generation.validator import validate_synthetic_tree, has_self_intersection
except ImportError:
    from pca_ssm_vessel_tree_generator.generation.validator import validate_synthetic_tree, has_self_intersection

from motion.cardiac_motion import apply_cardiac_motion_to_tree

EPS = 1.0e-12


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
    b5_trees_json = batch5_dir / "synthetic_trees.json"
    thresh_json = batch5_dir.parent / "batch4_pca_ssm" / "validation_thresholds.json"

    if not b5_trees_json.exists():
        raise FileNotFoundError(f"Missing Batch 5 synthetic trees payload at {b5_trees_json}")

    with open(b5_trees_json, "r", encoding="utf-8") as f:
        trees_batch5 = json.load(f)

    thresholds = {}
    if thresh_json.exists():
        with open(thresh_json, "r", encoding="utf-8") as tf:
            thresholds = json.load(tf)

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

            has_self_int, _ = has_self_intersection(v3d, frame["side_branches"], min_dist_threshold_mm=0.0005)
            if has_self_int:
                self_intersection_free_all = False

    elapsed_time = time.time() - start_time

    # Collect phase-by-phase vessel statistics
    for p_idx in range(num_phases):
        p_val = trees_4d[0]["frames"][p_idx]["phase"]
        s_val = trees_4d[0]["frames"][p_idx]["contraction_scale_s"]

        rca_lens = [float(np.sum(np.linalg.norm(np.diff(t["frames"][p_idx]["vessels_3d"]["RCA"], axis=0), axis=1))) for t in trees_4d]
        lad_lens = [float(np.sum(np.linalg.norm(np.diff(t["frames"][p_idx]["vessels_3d"]["LAD"], axis=0), axis=1))) for t in trees_4d]
        a_vals = [t["frames"][p_idx]["ellipsoid_params"]["a_mm"] for t in trees_4d]
        c_vals = [t["frames"][p_idx]["ellipsoid_params"]["c_mm"] for t in trees_4d]

        phase_metrics.append({
            "phase_index": p_idx,
            "phase": p_val,
            "contraction_scale_s": s_val,
            "mean_scaffold_a_mm": float(np.mean(a_vals)),
            "mean_scaffold_c_mm": float(np.mean(c_vals)),
            "mean_RCA_length_mm": float(np.mean(rca_lens)),
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
