#!/usr/bin/env python
"""Generate a deterministic cohort of accepted population-derived coronary trees."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import xml.etree.ElementTree as ET
from argparse import Namespace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from run_person2_week1_demo import run as run_single_tree


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATS = REPO_ROOT / "outputs/lca_ssm/lca_population_model/generator_statistics"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/lca_ssm/lca_population_cohort"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_tree_record(index: int, tree_directory: Path, seed: int) -> dict[str, Any]:
    parameters = json.loads((tree_directory / "parameters.json").read_text(encoding="utf-8"))
    validation = json.loads((tree_directory / "validation.json").read_text(encoding="utf-8"))
    attempts = json.loads((tree_directory / "sampling_attempts.json").read_text(encoding="utf-8"))
    metrics = validation["metrics"]
    direction = metrics["anatomical_direction"]
    record: dict[str, Any] = {
        "tree_id": f"tree_{index:04d}",
        "seed": seed,
        "accepted": validation["accepted"],
        "accepted_attempt": attempts["accepted_attempt"],
        "source_case_id": parameters["ellipsoid"]["source_case_id"],
        "trajectory_case_id": parameters["generation"]["empirical_trajectory_source_case_id"],
        "pca_baseline_case_id": parameters["generation"]["deviation_sample"]["baseline_case_id"],
        "pca_innovation_scale": parameters["generation"]["deviation_sample"]["variation_scale"],
        "pca_innovation_l2": float(np.linalg.norm(
            parameters["generation"]["deviation_sample"]["innovation_coefficients"]
        )),
        "ellipsoid_a_mm": metrics["ellipsoid_axes_mm"]["a"],
        "ellipsoid_b_mm": metrics["ellipsoid_axes_mm"]["b"],
        "ellipsoid_c_mm": metrics["ellipsoid_axes_mm"]["c"],
        "bifurcation_angle_deg": metrics["bifurcation_angle_deg"],
        "lad_inferior_displacement_mm": direction["LAD_inferior_displacement_mm"],
        "lcx_inferior_displacement_mm": direction["LCX_inferior_displacement_mm"],
        "lcx_to_lad_inferior_displacement_ratio": direction["LCX_to_LAD_inferior_displacement_ratio"],
        "lad_descending_segment_fraction": direction["LAD_descending_segment_fraction"],
        "topology_lmca_lad_mm": metrics["topology_errors"]["LMCA_to_LAD_mm"],
        "topology_lmca_lcx_mm": metrics["topology_errors"]["LMCA_to_LCX_mm"],
        "tree_directory": tree_directory.name,
        "vtk_path": f"{tree_directory.name}/vtk/synthetic_tree.vtm",
    }
    for branch, value in metrics["branch_lengths_mm"].items():
        record[f"{branch.lower()}_length_mm"] = value
    for branch, value in metrics["branch_tortuosity"].items():
        record[f"{branch.lower()}_tortuosity"] = value
    for branch, value in metrics["branch_obliquity_rad"].items():
        record[f"{branch.lower()}_obliquity_rad"] = value
    for branch, value in metrics["minimum_nonlocal_distance_mm"].items():
        record[f"{branch.lower()}_self_clearance_mm"] = value
    for branch, values in metrics["branch_progression"].items():
        for metric_name, value in values.items():
            record[f"{branch.lower()}_{metric_name}"] = value
    for pair, value in metrics["minimum_interbranch_distance_mm"].items():
        record[f"pair_{pair.lower()}"] = value
    return record


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_cohort_vtm(path: Path, tree_directories: list[Path]) -> None:
    root = ET.Element("VTKFile", type="vtkMultiBlockDataSet", version="1.0", byte_order="LittleEndian")
    multiblock = ET.SubElement(root, "vtkMultiBlockDataSet")
    for index, tree_directory in enumerate(tree_directories, start=1):
        block = ET.SubElement(multiblock, "Block", index=str(index - 1), name=f"TREE_{index:04d}")
        vtk_directory = tree_directory / "vtk"
        datasets = [
            ("SYNTHETIC_ELLIPSOID", vtk_directory / "synthetic_ellipsoid.vtp"),
            ("SYNTHETIC_LMCA", vtk_directory / "LMCA.vtp"),
            ("SYNTHETIC_LAD", vtk_directory / "LAD.vtp"),
            ("SYNTHETIC_LCX", vtk_directory / "LCX.vtp"),
        ]
        if (vtk_directory / "RCA.vtp").is_file():
            datasets.append(("SYNTHETIC_RCA", vtk_directory / "RCA.vtp"))
        datasets.append(("SYNTHETIC_LANDMARKS", vtk_directory / "landmarks.vtp"))
        for dataset_index, (name, target) in enumerate(datasets):
            relative = Path(os.path.relpath(target, path.parent)).as_posix()
            ET.SubElement(
                block,
                "DataSet",
                index=str(dataset_index),
                name=name,
                file=relative,
            )
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    reopened = pv.read(path)
    if len(reopened) != len(tree_directories):
        raise RuntimeError(f"cohort VTM read-back returned {len(reopened)} blocks")
    expected_children = 6 if (tree_directories[0] / "vtk/RCA.vtp").is_file() else 5
    if any(block is None or len(block) != expected_children for block in reopened):
        raise RuntimeError("cohort VTM read-back contains an incomplete tree block")


def write_montage(path: Path, tree_directories: list[Path]) -> None:
    columns = 5
    rows = int(np.ceil(len(tree_directories) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(16, 3.25 * rows))
    axes_array = np.atleast_1d(axes).reshape(rows, columns)
    for index, axis in enumerate(axes_array.flat):
        axis.axis("off")
        if index < len(tree_directories):
            axis.imshow(plt.imread(tree_directories[index] / "preview.png"))
            axis.set_title(tree_directories[index].name, fontsize=10)
    figure.suptitle(
        "Population-derived synthetic coronary cohort — fixed cardiac X-Z front views",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    stats = args.stats_dir.resolve()
    required = (
        "generator_statistics_manifest.json",
        "population_surface_statistics.json",
        "population_ellipsoid_parameters.csv",
        "landmark_stats.json",
        "surface_deviation_pca.npz",
        "population_validation_thresholds.json",
        "fixed_branch_surface_coordinates.npz",
        "branch_assignment_gate.json",
    )
    missing = [name for name in required if not (stats / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete frozen LCA statistics package: {missing}")
    if args.count < 1:
        raise ValueError("--count must be positive")
    with (stats / "population_ellipsoid_parameters.csv").open(newline="", encoding="utf-8") as handle:
        source_case_ids = sorted(
            (
                row["case_id"] for row in csv.DictReader(handle)
                if row.get("is_valid", "").strip().lower() == "true"
            ),
            key=lambda value: int(value.split(".", 1)[0]),
        )
    if not source_case_ids:
        raise ValueError("frozen LCA statistics package has no anatomy-eligible empirical source cases")
    if output.exists():
        if not args.clean:
            raise FileExistsError(f"refusing to overwrite {output}; pass --clean")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    records: list[dict[str, Any]] = []
    tree_directories: list[Path] = []
    for index in range(1, args.count + 1):
        tree_directory = output / f"tree_{index:04d}"
        seed = args.base_seed + (index - 1) * args.seed_stride
        single_args = Namespace(
            output_dir=tree_directory,
            seed=seed,
            clean=False,
            include_rca=args.include_rca,
            stats_dir=stats,
            landmark_stats=None,
            pca=None,
            thresholds=None,
            max_attempts=args.max_attempts,
            pca_scale=args.pca_scale,
            source_case_id=source_case_ids[(index - 1) % len(source_case_ids)],
        )
        run_single_tree(single_args)
        records.append(flatten_tree_record(index, tree_directory, seed))
        tree_directories.append(tree_directory)
        print(f"accepted {tree_directory.name} ({index}/{args.count})", flush=True)

    write_csv(output / "cohort_metrics.csv", records)
    write_montage(output / "cohort_preview_montage.png", tree_directories)
    write_cohort_vtm(output / "synthetic_cohort.vtm", tree_directories)
    package_manifest = stats / "generator_statistics_manifest.json"
    manifest = {
        "schema_version": 2,
        "status": "PASS",
        "generation_mode": "frozen_lca_statistics_case_matched_bootstrap_plus_pca_innovation",
        "tree_count": len(records),
        "all_trees_accepted": all(row["accepted"] for row in records),
        "include_rca": args.include_rca,
        "base_seed": args.base_seed,
        "seed_stride": args.seed_stride,
        "pca_innovation_scale": args.pca_scale,
        "maximum_attempts_per_tree": args.max_attempts,
        "total_sampling_attempts": int(sum(row["accepted_attempt"] for row in records)),
        "unique_source_case_count": len({row["source_case_id"] for row in records}),
        "eligible_source_case_count": len(source_case_ids),
        "eligible_source_cases": source_case_ids,
        "balanced_source_case_schedule": True,
        "exact_lca_topology_for_all_trees": all(
            row["topology_lmca_lad_mm"] == 0.0 and row["topology_lmca_lcx_mm"] == 0.0
            for row in records
        ),
        "visual_anatomy_preview": "fixed cardiac X-Z front view; apex is negative Z",
        "unresolved_source_assignments_used": False,
        "generator_statistics_manifest": str(package_manifest.relative_to(REPO_ROOT).as_posix()),
        "generator_statistics_manifest_sha256": file_sha256(package_manifest),
        "cohort_metrics": "cohort_metrics.csv",
        "cohort_vtm": "synthetic_cohort.vtm",
        "preview_montage": "cohort_preview_montage.png",
    }
    write_json(output / "cohort_manifest.json", manifest)
    (output / "README.md").write_text(
        "# Population-derived synthetic coronary cohort\n\n"
        "Open `synthetic_cohort.vtm` in ParaView to inspect every accepted tree. "
        "Each `tree_XXXX` directory contains parameters, validation, NumPy arrays, surface coordinates, "
        "a fixed front preview, a fixed three-view anatomy preview, and modular VTK. "
        "The montage uses the common cardiac X-Z front view so LAD descent and LCX crown behaviour "
        "can be compared without camera-angle ambiguity. `cohort_metrics.csv` contains one quantitative row per tree.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=52)
    parser.add_argument("--base-seed", type=int, default=20260817)
    parser.add_argument("--seed-stride", type=int, default=1009)
    parser.add_argument("--max-attempts", type=int, default=250)
    parser.add_argument("--pca-scale", type=float, default=0.04)
    parser.add_argument("--include-rca", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    manifest = run(args)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
