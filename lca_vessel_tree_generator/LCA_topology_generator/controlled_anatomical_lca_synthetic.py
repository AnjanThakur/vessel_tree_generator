import argparse
import math
import shutil
from pathlib import Path

import numpy as np

from .anatomical_lca_pipeline import load_anatomical_records, run_pipeline_records, _write_json
from .anatomical_lca_reorientation import (
    _branch_lengths,
    _lad_template,
    _lcx_template,
    _lmca_template,
    save_anatomical_tree_plot,
    save_anatomical_vtk_files,
    validate_anatomical_lca_tree,
)
from .bspline import interpolate_lca_tree
from .disease_model import DEFAULT_DISEASE_SETTINGS
from .lca_validation import validate_lca_tree
from .paths import lca_generated_control_points_dir, output_path
from .tortuosity import calculate_lca_tortuosity, calculate_tortuosity


BRANCH_ORDER = ["LMCA", "LAD", "LCX"]
BRANCH_SLICES = {
    "LMCA": slice(0, 5),
    "LAD": slice(5, 17),
    "LCX": slice(17, 27),
}


def _rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm <= 1e-12:
        return np.eye(3)
    axis = axis / norm
    angle = math.radians(angle_deg)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


def _rotate_branch(points: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    rotation = _rotation_matrix(axis, angle_deg)
    rotated = origin + (points - origin) @ rotation.T
    rotated[0] = origin
    return rotated


def _random_unit(rng: np.random.Generator, lateral_only: bool = False) -> np.ndarray:
    if lateral_only:
        vector = np.array([rng.normal(), rng.normal(), 0.0], dtype=float)
    else:
        vector = np.array([rng.normal(), rng.normal(), rng.normal()], dtype=float)
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return vector / norm


def _smooth_offset(points: np.ndarray, rng: np.random.Generator, max_noise_mm: float, lateral_only: bool = False) -> np.ndarray:
    if max_noise_mm <= 0:
        return points.copy()
    result = np.asarray(points, dtype=float).copy()
    direction = _random_unit(rng, lateral_only=lateral_only)
    amplitude = rng.uniform(0.0, max_noise_mm)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    t = np.linspace(0.0, 1.0, len(result))
    weights = np.sin(math.pi * t) * np.sin(2.0 * math.pi * t + phase)
    result += amplitude * weights[:, None] * direction[None, :]
    result[0] = points[0]
    result[-1] = points[-1]
    return result


def _branch_tortuosity(tree: np.ndarray) -> dict:
    return {
        branch_name: calculate_tortuosity(tree[branch_slice])["tortuosity"]
        for branch_name, branch_slice in BRANCH_SLICES.items()
    }


def _synthetic_validation(
    candidate: np.ndarray,
    base_tree: np.ndarray,
    base_lengths: dict,
    args,
) -> dict:
    validation = validate_anatomical_lca_tree(candidate, reference_lengths=base_lengths)
    candidate_lengths = _branch_lengths(candidate)
    base_tortuosity = _branch_tortuosity(base_tree)
    candidate_tortuosity = _branch_tortuosity(candidate)

    length_tolerance = args.max_length_scale_variation + 0.03
    length_ratios = {
        branch_name: candidate_lengths[branch_name] / max(base_lengths[branch_name], 1e-9)
        for branch_name in BRANCH_ORDER
    }
    length_preservation_ok = all(
        (1.0 - length_tolerance) <= ratio <= (1.0 + length_tolerance)
        for ratio in length_ratios.values()
    )
    tortuosity_delta = {}
    tortuosity_ok = True
    for branch_name in BRANCH_ORDER:
        base_value = base_tortuosity[branch_name]
        candidate_value = candidate_tortuosity[branch_name]
        if base_value is None or candidate_value is None:
            tortuosity_delta[branch_name] = None
            tortuosity_ok = False
            continue
        delta = abs(candidate_value - base_value)
        tortuosity_delta[branch_name] = float(delta)
        tortuosity_ok = tortuosity_ok and delta <= args.max_tortuosity_change

    duplicate_ok = validation["anatomical_checks"]["no_unexpected_duplicate_points"]
    centerlines = interpolate_lca_tree(candidate)
    centerline_metrics = calculate_lca_tortuosity(centerlines)
    centerline_validation = validate_lca_tree(candidate, centerlines, centerline_metrics)
    validation["synthetic_checks"] = {
        "length_preservation_within_config": bool(length_preservation_ok),
        "tortuosity_change_within_config": bool(tortuosity_ok),
        "duplicate_point_check": bool(duplicate_ok),
        "bifurcation_consistency": validation["anatomical_checks"]["branch_topology_lmca_to_lad_lcx"],
        "lad_not_upward": validation["anatomical_checks"]["lad_dominant_downward"],
        "lcx_not_second_lad": validation["anatomical_checks"]["lcx_not_second_lad"],
        "existing_centerline_validation": bool(centerline_validation["is_valid"]),
    }
    validation["synthetic_metrics"] = {
        "branch_length_preservation_ratio": length_ratios,
        "length_tolerance": float(length_tolerance),
        "base_tortuosity": base_tortuosity,
        "candidate_tortuosity": candidate_tortuosity,
        "tortuosity_delta": tortuosity_delta,
        "max_tortuosity_change": float(args.max_tortuosity_change),
    }
    validation["centerline_validation"] = centerline_validation
    if not length_preservation_ok:
        validation["errors"].append("Synthetic branch length preservation is outside configured tolerance")
    if not tortuosity_ok:
        validation["errors"].append("Synthetic tortuosity change is outside configured tolerance")
    if not centerline_validation["is_valid"]:
        validation["errors"].append("Synthetic candidate fails existing centerline validation")
    validation["is_valid"] = len(validation["errors"]) == 0
    return validation


def _generate_candidate(base_tree: np.ndarray, rng: np.random.Generator, args) -> tuple:
    base_lengths = _branch_lengths(base_tree)
    scale = {
        branch_name: rng.uniform(
            1.0 - args.max_length_scale_variation,
            1.0 + args.max_length_scale_variation,
        )
        for branch_name in BRANCH_ORDER
    }
    lengths = {
        branch_name: base_lengths[branch_name] * scale[branch_name]
        for branch_name in BRANCH_ORDER
    }

    lmca = _lmca_template(lengths["LMCA"], 5)
    bifurcation = lmca[-1].copy()
    lad = _lad_template(bifurcation, lengths["LAD"], 12)
    lcx = _lcx_template(bifurcation, lengths["LCX"], 10)

    lad_angle = rng.uniform(-args.max_angle_variation_deg, args.max_angle_variation_deg)
    lcx_angle = rng.uniform(-args.max_angle_variation_deg, args.max_angle_variation_deg)
    lad = _rotate_branch(lad, bifurcation, np.array([0.0, 1.0, 0.0]), lad_angle)
    lcx = _rotate_branch(lcx, bifurcation, np.array([0.0, 0.0, 1.0]), lcx_angle)

    lmca = _smooth_offset(lmca, rng, args.max_control_point_noise_mm * 0.35, lateral_only=True)
    lmca[-1] = bifurcation
    lad = _smooth_offset(lad, rng, args.max_control_point_noise_mm, lateral_only=False)
    lcx = _smooth_offset(lcx, rng, args.max_control_point_noise_mm, lateral_only=True)
    lad[0] = bifurcation
    lcx[0] = bifurcation

    candidate = np.vstack([lmca, lad, lcx])
    perturbation = {
        "length_scale": scale,
        "lad_descent_angle_delta_deg": float(lad_angle),
        "lcx_crown_arc_angle_delta_deg": float(lcx_angle),
        "max_control_point_noise_mm": float(args.max_control_point_noise_mm),
    }
    return candidate, perturbation


def _make_synthetic_variant(base_record: dict, variant_index: int, rng: np.random.Generator, args) -> tuple:
    base_tree = np.asarray(base_record["control_tree"], dtype=float)
    base_lengths = _branch_lengths(base_tree)
    last_validation = None
    rejected = 0

    for _ in range(args.validation_retry_count):
        candidate, perturbation = _generate_candidate(base_tree, rng, args)
        validation = _synthetic_validation(candidate, base_tree, base_lengths, args)
        last_validation = validation
        if validation["is_valid"]:
            return candidate, validation, perturbation, rejected
        rejected += 1

    raise ValueError(f"Could not generate valid synthetic variant after {args.validation_retry_count} retries: {last_validation['errors']}")


def _write_synthetic_markdown(path: Path, summary: dict):
    lines = [
        "# Synthetic LCA Generation Summary",
        "",
        "## Counts",
        "",
        f"- Base anatomical patients used: {summary['base_anatomical_patients_used']}",
        f"- Variants per patient: {summary['variants_per_patient']}",
        f"- Total synthetic trees generated: {summary['total_synthetic_trees_generated']}",
        f"- Total rejected candidates: {summary['total_rejected_candidates']}",
        f"- Validation pass count: {summary['validation_pass_count']}",
        f"- Validation fail count: {summary['validation_fail_count']}",
        "",
        "## Average Scores",
        "",
        f"- Average anatomical score: {summary['average_anatomical_score']}",
        f"- Average LAD downward score: {summary['average_lad_downward_score']}",
        f"- Average LCX lateral score: {summary['average_lcx_lateral_score']}",
        "",
        "## Pipeline Results",
        "",
        f"- Pipeline valid radius outputs: {summary['pipeline']['valid_radius_outputs']}",
        f"- Pipeline disease cases generated: {summary['pipeline']['disease_cases_generated']}",
        f"- Pipeline valid diseased cases: {summary['pipeline']['valid_diseased_cases']}",
        f"- Pipeline valid diseased tight mesh cases: {summary['pipeline']['valid_diseased_tight_mesh_cases']}",
        "",
        "## Limitations",
        "",
        "- Controlled synthetic augmentation, not clinical patient-specific reconstruction.",
        "- Synthetic samples are patient-inspired and anatomically constrained.",
        "- Variants stay close to base patient scale and branch layout.",
        "- Disease, mesh, and hub connector logic are reused unchanged.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _average(records: list, metric: str):
    values = []
    for record in records:
        validation = record["validation"]
        metrics = validation.get("anatomical_metrics", {})
        if metric in metrics:
            values.append(float(metrics[metric]))
    if not values:
        return None
    return float(np.mean(values))


def generate_synthetic_dataset(args) -> dict:
    input_records = load_anatomical_records(args.input_dir)
    if len(input_records) == 0:
        raise FileNotFoundError(f"No anatomical patient records found in {args.input_dir}")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.no_clean:
        shutil.rmtree(output_dir)
    tree_dir = output_dir / "trees"
    tree_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.random_seed)
    synthetic_records = []
    validation_records = []
    rejected_total = 0
    failed_records = []

    for base_record in input_records:
        base_patient_id = base_record["patient_id"]
        for variant_index in range(args.variants_per_patient):
            synthetic_id = f"{base_patient_id}_synth_{variant_index:03d}"
            try:
                candidate, validation, perturbation, rejected = _make_synthetic_variant(
                    base_record,
                    variant_index,
                    rng,
                    args,
                )
                rejected_total += rejected
            except Exception as exc:
                failed_records.append({
                    "base_patient_id": base_patient_id,
                    "variant_index": variant_index,
                    "error": str(exc),
                })
                continue

            variant_dir = tree_dir / synthetic_id
            variant_dir.mkdir(parents=True, exist_ok=True)
            np.save(variant_dir / "control_points_27x3_synthetic.npy", candidate)
            np.save(variant_dir / "control_points_27x3_anatomical.npy", candidate)
            np.save(variant_dir / "control_points_27x3_base_anatomical.npy", base_record["control_tree"])
            _write_json(variant_dir / "synthetic_validation.json", validation)
            _write_json(
                variant_dir / "synthetic_summary.json",
                {
                    "synthetic_id": synthetic_id,
                    "base_patient_id": base_patient_id,
                    "variant_index": variant_index,
                    "perturbation": perturbation,
                    "validation": validation,
                },
            )
            save_anatomical_tree_plot(
                variant_dir / "synthetic_control_points_3d.png",
                candidate,
                f"{synthetic_id} controlled synthetic LCA",
            )
            save_anatomical_vtk_files(variant_dir, candidate, synthetic_id)

            synthetic_records.append({
                "patient_id": synthetic_id,
                "metadata_patient_id": base_record.get("metadata_patient_id", base_patient_id),
                "tree_index": base_record.get("tree_index"),
                "control_tree": candidate,
                "source_dir": str(variant_dir),
                "reference_lengths": _branch_lengths(base_record["control_tree"]),
                "base_patient_id": base_patient_id,
            })
            validation_records.append({
                "synthetic_id": synthetic_id,
                "base_patient_id": base_patient_id,
                "validation": validation,
                "perturbation": perturbation,
            })

    if synthetic_records:
        np.save(
            output_dir / "LCA_tree_ctrl_points_synthetic.npy",
            np.asarray([record["control_tree"] for record in synthetic_records], dtype=float),
        )

    pipeline_args = argparse.Namespace(
        input_dir=output_dir,
        output_dir=output_dir / "pipeline",
        metadata_dir=args.metadata_dir,
        no_clean=False,
        lmca_points=args.lmca_points,
        lad_points=args.lad_points,
        lcx_points=args.lcx_points,
        radius_adjacent_jump_threshold=args.radius_adjacent_jump_threshold,
        tube_circle_points=args.tube_circle_points,
        minimum_radius_mm=args.minimum_radius_mm,
        disease_adjacent_jump_threshold=args.disease_adjacent_jump_threshold,
    )
    pipeline_summary = run_pipeline_records(synthetic_records, pipeline_args.output_dir, pipeline_args)

    summary = {
        "mode": "controlled_anatomical_lca_synthetic_mvp",
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "base_anatomical_patients_used": int(len(input_records)),
        "variants_per_patient": int(args.variants_per_patient),
        "total_synthetic_trees_generated": int(len(synthetic_records)),
        "total_rejected_candidates": int(rejected_total),
        "validation_pass_count": int(sum(1 for record in validation_records if record["validation"]["is_valid"])),
        "validation_fail_count": int(len(failed_records)),
        "average_anatomical_score": _average(validation_records, "anatomical_score"),
        "average_lad_downward_score": _average(validation_records, "lad_downward_score"),
        "average_lcx_lateral_score": _average(validation_records, "lcx_lateral_score"),
        "config": {
            "variants_per_patient": args.variants_per_patient,
            "random_seed": args.random_seed,
            "max_length_scale_variation": args.max_length_scale_variation,
            "max_angle_variation_deg": args.max_angle_variation_deg,
            "max_control_point_noise_mm": args.max_control_point_noise_mm,
            "max_tortuosity_change": args.max_tortuosity_change,
            "validation_retry_count": args.validation_retry_count,
        },
        "failed_records": failed_records,
        "synthetic_records": validation_records,
        "pipeline": pipeline_summary,
        "limitations": [
            "Controlled synthetic augmentation, not clinical patient-specific reconstruction.",
            "Synthetic samples are patient-inspired and anatomically constrained.",
            "Small branch-level perturbations are used instead of random unrealistic trees.",
        ],
    }
    _write_json(output_dir / "synthetic_lca_generation_summary.json", summary)
    _write_synthetic_markdown(output_dir / "SYNTHETIC_LCA_GENERATION_SUMMARY.md", summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Generate controlled synthetic anatomical LCA variants and run the existing pipeline.")
    parser.add_argument("--input-dir", type=Path, default=output_path("dataset_lca_anatomical"))
    parser.add_argument("--output-dir", type=Path, default=output_path("dataset_lca_anatomical_synthetic"))
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=lca_generated_control_points_dir(),
    )
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--variants-per-patient", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-length-scale-variation", type=float, default=0.05)
    parser.add_argument("--max-angle-variation-deg", type=float, default=8.0)
    parser.add_argument("--max-control-point-noise-mm", type=float, default=0.5)
    parser.add_argument("--max-tortuosity-change", type=float, default=0.12)
    parser.add_argument("--validation-retry-count", type=int, default=30)
    parser.add_argument("--lmca-points", type=int, default=150)
    parser.add_argument("--lad-points", type=int, default=300)
    parser.add_argument("--lcx-points", type=int, default=250)
    parser.add_argument("--radius-adjacent-jump-threshold", type=float, default=0.2)
    parser.add_argument("--tube-circle-points", type=int, default=24)
    parser.add_argument("--minimum-radius-mm", type=float, default=DEFAULT_DISEASE_SETTINGS["minimum_radius_mm"])
    parser.add_argument("--disease-adjacent-jump-threshold", type=float, default=DEFAULT_DISEASE_SETTINGS["adjacent_jump_threshold_mm"])
    return parser.parse_args()


def main():
    args = parse_args()
    summary = generate_synthetic_dataset(args)
    print(f"Base anatomical patients used: {summary['base_anatomical_patients_used']}")
    print(f"Synthetic trees generated: {summary['total_synthetic_trees_generated']}")
    print(f"Rejected candidates: {summary['total_rejected_candidates']}")
    print(f"Pipeline disease cases generated: {summary['pipeline']['disease_cases_generated']}")
    print(f"Summary: {args.output_dir / 'SYNTHETIC_LCA_GENERATION_SUMMARY.md'}")


if __name__ == "__main__":
    main()
