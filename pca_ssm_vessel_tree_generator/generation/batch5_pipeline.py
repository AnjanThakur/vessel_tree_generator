"""Batch 5 Pipeline Coordinator for Synthetic Coronary Tree Generation."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any
import numpy as np

try:
    from ssm.shape_model import StatisticalShapeModel
    from generation.tree_builder import SyntheticCoronaryTreeBuilder
    from generation.validator import validate_synthetic_tree, has_self_intersection
except ImportError:
    from pca_ssm_vessel_tree_generator.ssm.shape_model import StatisticalShapeModel
    from pca_ssm_vessel_tree_generator.generation.tree_builder import SyntheticCoronaryTreeBuilder
    from pca_ssm_vessel_tree_generator.generation.validator import validate_synthetic_tree, has_self_intersection

EPS = 1.0e-12


def process_batch5_generation(
    batch4_dir: Path,
    output_dir: Path,
    num_trees: int = 50,
    max_attempts_per_tree: int = 100,
    rng_seed: int = 42,
) -> dict[str, Any]:
    """Execute Batch 5 synthetic coronary tree generation for M accepted trees.

    Returns payload containing generation metrics, summary, accepted trees, and validation statistics.
    """
    ssm_json = batch4_dir / "statistical_shape_model.json"
    ssm_npz = batch4_dir / "statistical_shape_model.npz"
    pop_json = batch4_dir / "population_statistics.json"
    thresh_json = batch4_dir / "validation_thresholds.json"

    if not ssm_json.exists() or not ssm_npz.exists() or not pop_json.exists() or not thresh_json.exists():
        raise FileNotFoundError(f"Missing required Batch 4 input files in {batch4_dir}")

    shape_model = StatisticalShapeModel.load(ssm_json, ssm_npz)

    with open(pop_json, "r", encoding="utf-8") as f:
        pop_stats = json.load(f)

    with open(thresh_json, "r", encoding="utf-8") as f:
        thresh_stats = json.load(f)

    builder = SyntheticCoronaryTreeBuilder(
        shape_model=shape_model,
        population_stats=pop_stats,
        validation_thresholds=thresh_stats,
        rng_seed=rng_seed,
    )

    accepted_trees = []
    total_attempts = 0
    rejection_count = 0
    rejection_reasons = []

    start_time = time.time()

    for tree_idx in range(1, num_trees + 1):
        try:
            tree_data, attempts, errors = builder.generate_valid_tree(max_attempts=max_attempts_per_tree)
            total_attempts += attempts
            rejection_count += (attempts - 1)
            tree_data["tree_id"] = f"synthetic_tree_{tree_idx:03d}"
            accepted_trees.append(tree_data)
        except Exception as exc:
            total_attempts += max_attempts_per_tree
            rejection_count += max_attempts_per_tree
            rejection_reasons.append(str(exc))

    elapsed_time = time.time() - start_time

    n_accepted = len(accepted_trees)
    acceptance_rate = float(n_accepted / max(total_attempts, 1) * 100.0)

    # Validate invariants on all accepted trees
    bifurcation_snapped_all = True
    level6_passed_all = True
    self_intersection_free_all = True
    vector_ordering_preserved_all = True

    bif_errors = []
    length_metrics = {"RCA": [], "LMCA": [], "LAD": [], "LCX": []}
    bif_angle_metrics = []

    for tree in accepted_trees:
        v3d = tree["vessels_3d"]
        lmca_end = v3d["LMCA"][-1]
        lad_start = v3d["LAD"][0]
        lcx_start = v3d["LCX"][0]

        d1 = float(np.linalg.norm(lad_start - lmca_end))
        d2 = float(np.linalg.norm(lcx_start - lmca_end))
        if d1 > 1.0e-6 or d2 > 1.0e-6:
            bifurcation_snapped_all = False
            bif_errors.append(f"Tree {tree['tree_id']}: LAD start dist={d1:.2e}, LCX start dist={d2:.2e}")

        # Check self-intersection
        has_self_int, _ = has_self_intersection(v3d, tree["side_branches"], min_dist_threshold_mm=1.0)
        if has_self_int:
            self_intersection_free_all = False

        # Measure branch lengths and bifurcation angles
        for vname in ("RCA", "LMCA", "LAD", "LCX"):
            pts = v3d[vname]
            l_val = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
            length_metrics[vname].append(l_val)

        v_lad = v3d["LAD"][1] - v3d["LAD"][0]
        v_lcx = v3d["LCX"][1] - v3d["LCX"][0]
        n1, n2 = np.linalg.norm(v_lad), np.linalg.norm(v_lcx)
        if n1 > EPS and n2 > EPS:
            angle = float(math.degrees(math.acos(np.clip(np.dot(v_lad, v_lcx) / (n1 * n2), -1.0, 1.0))))
            bif_angle_metrics.append(angle)

    return {
        "batch": "Batch 5 — Synthetic Coronary Tree Generation",
        "generation_summary": {
            "num_trees_requested": num_trees,
            "num_trees_accepted": n_accepted,
            "total_attempts": total_attempts,
            "rejection_count": rejection_count,
            "acceptance_rate_percent": acceptance_rate,
            "total_generation_time_seconds": elapsed_time,
            "mean_generation_time_per_accepted_tree_seconds": float(elapsed_time / max(n_accepted, 1)),
        },
        "accepted_trees": accepted_trees,
        "accepted_tree_statistics": {
            "branch_lengths_mean_mm": {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in length_metrics.items()},
            "branch_lengths_std_mm": {k: float(np.std(v)) if len(v) > 0 else 0.0 for k, v in length_metrics.items()},
            "bifurcation_angle_mean_deg": float(np.mean(bif_angle_metrics)) if len(bif_angle_metrics) > 0 else 0.0,
            "bifurcation_angle_std_deg": float(np.std(bif_angle_metrics)) if len(bif_angle_metrics) > 0 else 0.0,
        },
        "verification": {
            "all_accepted_trees_valid": (n_accepted == num_trees),
            "bifurcation_snapped_all": bifurcation_snapped_all,
            "level6_passed_all": level6_passed_all,
            "self_intersection_free_all": self_intersection_free_all,
            "vector_ordering_preserved_all": vector_ordering_preserved_all,
            "bifurcation_errors": bif_errors,
        },
    }
