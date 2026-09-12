"""Batch 7 Pipeline Coordinator for Output Formatting and Packaging."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
import numpy as np

from output.save_geometry import (
    construct_geometry_static_array,
    construct_geometry_cine_array,
    get_ordered_tree_branches,
)
from output.save_metadata import (
    build_tree_metadata,
    build_ellipsoid_params_metadata,
    build_global_pipeline_report,
    jsonable,
)
from output.save_visualization import (
    save_tree_visualization,
    save_pipeline_qc_report,
)

EPS = 1.0e-12


def process_batch7_output(
    batch6_dir: Path,
    output_dir: Path,
    num_points: int = 50,
) -> dict[str, Any]:
    """Execute Batch 7 output formatting and packaging pipeline across all 50 Batch 6 synthetic 4D trees."""
    b6_4d_json = batch6_dir / "synthetic_trees_4d.json"
    b5_trees_json = batch6_dir.parent / "batch5_synthetic_trees" / "synthetic_trees.json"

    if not b6_4d_json.exists():
        raise FileNotFoundError(f"Missing Batch 6 4D payload at {b6_4d_json}")

    with open(b6_4d_json, "r", encoding="utf-8") as f:
        trees_4d = json.load(f)

    b5_lookup = {}
    if b5_trees_json.exists():
        with open(b5_trees_json, "r", encoding="utf-8") as b5_f:
            b5_data = json.load(b5_f)
            b5_lookup = {t["tree_id"]: t for t in b5_data}

    output_dir.mkdir(parents=True, exist_ok=True)
    n_trees = len(trees_4d)
    start_time = time.time()

    static_equal_cine0_all = True
    max_identity_error = 0.0
    exported_records = []

    for t_idx, tree_4d in enumerate(trees_4d):
        t_id = tree_4d["tree_id"]
        t_dir = output_dir / t_id
        t_dir.mkdir(parents=True, exist_ok=True)

        # 1. Construct static and cine geometry arrays
        geom_static = construct_geometry_static_array(tree_4d, num_points=num_points)
        geom_cine = construct_geometry_cine_array(tree_4d, num_points=num_points)

        # 2. Verify invariant: geometry_static == geometry_cine[0] at ph=0.0
        diff = np.abs(geom_static - geom_cine[0])
        max_diff = float(np.max(diff))
        if max_diff > max_identity_error:
            max_identity_error = max_diff

        if max_diff > 1.0e-12:
            static_equal_cine0_all = False

        # 3. Save .npy geometry arrays
        np.save(t_dir / "geometry_static.npy", geom_static)
        np.save(t_dir / "geometry_cine.npy", geom_cine)

        # 4. Save metadata JSON files
        b5_data = b5_lookup.get(t_id)
        meta = build_tree_metadata(tree_4d, b5_data)
        ell_params = build_ellipsoid_params_metadata(tree_4d)

        with open(t_dir / "metadata.json", "w", encoding="utf-8") as mf:
            json.dump(jsonable(meta), mf, indent=2)

        with open(t_dir / "ellipsoid_params.json", "w", encoding="utf-8") as ef:
            json.dump(jsonable(ell_params), ef, indent=2)

        # 5. Save 3D static rendering plot
        reference = tree_4d["frames"][0]
        branch_names = [
            name
            for name, _ in get_ordered_tree_branches(
                reference["vessels_3d"], reference["side_branches"]
            )
        ]
        save_tree_visualization(t_dir, geom_static, branch_names=branch_names)

        exported_records.append({
            "tree_id": t_id,
            "static_shape": list(geom_static.shape),
            "cine_shape": list(geom_cine.shape),
            "max_identity_error_mm": max_diff,
            "directory": str(t_dir.resolve()),
            "branch_order": branch_names,
        })

    elapsed_time = time.time() - start_time

    # 6. Dynamically build global pipeline report across Batches 1 to 7
    pipeline_report = build_global_pipeline_report(output_dir.parent)
    with open(output_dir / "pipeline_report.json", "w", encoding="utf-8") as pr_f:
        json.dump(jsonable(pipeline_report), pr_f, indent=2)

    # 7. Generate global pipeline QC report plot
    save_pipeline_qc_report(output_dir, pipeline_report)

    overall_pass = static_equal_cine0_all and (len(exported_records) == n_trees)

    return {
        "batch": "Batch 7 — Standardized Output Formatting and Packaging",
        "output_summary": {
            "num_trees_exported": n_trees,
            "num_points_per_vessel": num_points,
            "static_array_shape": list(exported_records[0]["static_shape"]),
            "cine_array_shape": list(exported_records[0]["cine_shape"]),
            "branch_order": exported_records[0]["branch_order"],
            "radius_model_provenance": "prototype taper defaults; not population learned",
            "total_execution_time_seconds": elapsed_time,
        },
        "verification": {
            "overall_pass": overall_pass,
            "geometry_static_equals_geometry_cine0": static_equal_cine0_all,
            "max_identity_error_mm": max_identity_error,
        },
        "exported_records": exported_records,
        "pipeline_report": pipeline_report,
    }
