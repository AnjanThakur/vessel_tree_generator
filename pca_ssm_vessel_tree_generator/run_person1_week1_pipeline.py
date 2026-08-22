#!/usr/bin/env python
"""Build generator-ready statistics from immutable real LCA cases.

The pipeline consumes the validated PPT two-plane/two-ellipse measurements and
their matching ``raw_cases`` archives. It never writes to those inputs and it
does not refit an ellipse, plane, or source centerline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.interpolate import PchipInterpolator

from surface_relative.cardiac_frame import compute_cardiac_frame, transform_to_cardiac_frame, validate_cardiac_frame
from surface_relative.anatomy import anatomical_role_acceptance, coronary_course_metrics
from surface_relative.ellipsoid_model import PatientEllipsoid, derive_patient_ellipsoid
from surface_relative.fixed_representation import (
    FIXED_COUNTS,
    LCA_BRANCH_ORDER,
    LCA_FIXED_COUNTS,
    LCA_TOTAL_FIXED_POINTS,
    build_patient_fixed_representation,
)
from surface_relative.population_statistics import (
    build_validation_thresholds,
    compute_circular_stats,
    compute_linear_stats,
    fit_deviation_pca,
)
from surface_relative.surface_projection import project_point_to_surface


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
PPT_OUTPUT_DIR = REPO_ROOT / "outputs/lca_ssm/ppt_priority_completion"
RAW_CASES_DIR = REPO_ROOT / "outputs/lca_ssm/raw_cases"
ASSIGNMENT_MAP_PATH = (
    REPO_ROOT
    / "outputs/lca_ssm/stage1_final_anatomical_model/branch_resolution/resolved_branch_assignments.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/lca_ssm/lca_population_model"
SMOKE_TEST_CASES = ["37.label", "89.label", "133.label"]
RESOLVED_ASSIGNMENT_STATUSES = {
    "resolved_existing_assignment",
    "resolved_swapped_assignment",
}
MIN_ASSIGNMENT_SCORE_MARGIN = 0.15
MIN_ASSIGNMENT_WINNER_SCORE = 0.45
CORE_ANATOMICAL_ROLE_CHECKS = {
    "lad_is_dominant_descending_branch",
    "lcx_is_not_apex_descending_branch",
    "lcx_has_horizontal_crown_course",
    "lcx_is_more_lateral_than_lad",
    "lmca_is_shorter_than_both_daughters",
}
VALIDATION_SCAFFOLD_SAMPLE_COUNTS = {"lmca": 60, "lad": 180, "lcx": 160, "rca": 190}


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    if not root.is_dir():
        return {}
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def protected_snapshot() -> dict[str, Any]:
    input_files = [
        PPT_OUTPUT_DIR / "population_two_plane_two_ellipse_parameters.csv",
        PPT_OUTPUT_DIR / "population_pointwise_ellipse_residuals.csv",
        PPT_OUTPUT_DIR / "population_case_status.csv",
        PPT_OUTPUT_DIR / "population_statistics.json",
        PPT_OUTPUT_DIR / "protected_stage1_integrity.json",
    ]
    raw_archives = sorted(RAW_CASES_DIR.glob("*/original_centerlines.npz"))
    return {
        "ppt_input_files": {path.relative_to(REPO_ROOT).as_posix(): file_sha256(path) for path in input_files},
        "branch_assignment_map": {
            ASSIGNMENT_MAP_PATH.relative_to(REPO_ROOT).as_posix(): file_sha256(ASSIGNMENT_MAP_PATH)
        },
        "raw_case_archives": {path.parent.name: file_sha256(path) for path in raw_archives},
        "protected_stage1_directories": {
            "stage1_heart_scaffold_vtk": tree_hashes(REPO_ROOT / "outputs/lca_ssm/stage1_heart_scaffold_vtk"),
            "stage1_parametric_heart_surface": tree_hashes(REPO_ROOT / "outputs/lca_ssm/stage1_parametric_heart_surface"),
        },
    }


def read_ppt_parameters(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["case_id"]: row for row in csv.DictReader(handle)}


def numeric_case_key(case_id: str) -> tuple[int, str]:
    token = case_id.split(".", 1)[0]
    return (int(token) if token.isdigit() else 10**9, case_id)


def load_case(case_id: str) -> tuple[dict[str, np.ndarray], Path]:
    path = RAW_CASES_DIR / case_id / "original_centerlines.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing immutable raw-case archive: {path}")
    with np.load(path, allow_pickle=False) as archive:
        branches = {name: np.asarray(archive[name]).copy() for name in archive.files}
    mandatory = {"lmca", "lad", "lcx"}
    if not mandatory.issubset(branches):
        raise ValueError(f"{case_id} is missing mandatory arrays: {sorted(mandatory - set(branches))}")
    for name, points in branches.items():
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.all(np.isfinite(points)):
            raise ValueError(f"{case_id}/{name} is not a finite Nx3 centerline")
    return branches, path


def load_assignment_records(path: Path) -> dict[str, dict[str, Any]]:
    """Resolve roles from each immutable case's multi-signal RAS evidence.

    The former final map covered only a subset of cases and deliberately left
    most assignments unresolved. Every raw case already records both candidate
    scores, the score margin, the adapter mapping, and the selected roles.
    Low-confidence cases remain manual review and are excluded from PCA.
    """
    del path  # retained in the public signature for backward compatibility
    records: dict[str, dict[str, Any]] = {}
    for metadata_path in sorted(RAW_CASES_DIR.glob("*/source_metadata.json")):
        case_id = metadata_path.parent.name
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        evidence = payload.get("assignment", {})
        adapter_lad = evidence.get("adapter_lad_source")
        selected_lad = evidence.get("selected_lad_source")
        selected_lcx = evidence.get("selected_lcx_source")
        margin = float(evidence.get("margin", 0.0))
        winner_score = float(evidence.get("score", -math.inf))
        selected_candidate = evidence.get("candidate_scores", {}).get(
            f"{selected_lad}_as_lad", {}
        )
        lcx_apex_dominant = bool(
            selected_candidate.get("lcx_behaves_as_main_apex_branch", True)
        )
        mapping_valid = {adapter_lad, selected_lad, selected_lcx}.issubset(
            {"branch_a", "branch_b"}
        )
        confident = bool(
            mapping_valid
            and selected_lad != selected_lcx
            and margin >= MIN_ASSIGNMENT_SCORE_MARGIN
            and winner_score >= MIN_ASSIGNMENT_WINNER_SCORE
            and not lcx_apex_dominant
        )
        status = "unresolved_manual_review"
        if confident:
            status = (
                "resolved_existing_assignment"
                if selected_lad == adapter_lad
                else "resolved_swapped_assignment"
            )
        adapter_lcx = "branch_b" if adapter_lad == "branch_a" else "branch_a"
        records[case_id] = {
            "case": case_id,
            "resolution_status": status,
            "new_resolved_assignment": {
                "lad_source": selected_lad,
                "lcx_source": selected_lcx,
            },
            "neutral_reconstruction": {
                f"{adapter_lad}_source_array": "lad",
                f"{adapter_lcx}_source_array": "lcx",
            },
            "assignment_scoring": {
                "method": evidence.get("method", "multi_signal_ras_anatomy_score"),
                "score_margin": margin,
                "winning_score": winner_score,
                "candidate_scores": evidence.get("candidate_scores", {}),
                "selected_lcx_behaves_as_main_apex_branch": lcx_apex_dominant,
                "root_radius_used_for_assignment": bool(
                    evidence.get("root_radius_used_for_assignment", False)
                ),
                "single_direction_vector_can_select_assignment": bool(
                    evidence.get("single_direction_vector_can_select_assignment", False)
                ),
                "confidence_rule": {
                    "minimum_score_margin": MIN_ASSIGNMENT_SCORE_MARGIN,
                    "minimum_winner_score": MIN_ASSIGNMENT_WINNER_SCORE,
                    "lcx_must_not_be_main_apex_descending_branch": True,
                },
            },
            "source_metadata": str(metadata_path.relative_to(REPO_ROOT)),
        }
    if not records:
        raise ValueError(f"no assignment evidence found below {RAW_CASES_DIR}")
    return records


def apply_resolved_roles(
    case_id: str,
    raw_branches: dict[str, np.ndarray],
    assignment_records: dict[str, dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return named branches only when their neutral daughter roles are resolved.

    Raw archives remain untouched.  The assignment map records which saved
    raw array represented neutral Branch A/Branch B, so this adapter also
    remains correct if a future resolved case requires a swap.
    """
    record = assignment_records.get(case_id)
    status = None if record is None else record.get("resolution_status")
    resolved = status in RESOLVED_ASSIGNMENT_STATUSES
    result = {name: points.copy() for name, points in raw_branches.items()}
    metadata: dict[str, Any] = {
        "case_id": case_id,
        "resolution_status": "absent_from_assignment_map" if record is None else status,
        "resolved_for_statistics": bool(resolved),
        "lad_source_array": None,
        "lcx_source_array": None,
        "assignment_score_margin": None,
        "assignment_winner_score": None,
    }
    if not resolved:
        return result, metadata
    neutral_mapping = record.get("neutral_reconstruction", {})
    resolved_mapping = record.get("new_resolved_assignment", {})
    branch_a_array = neutral_mapping.get("branch_a_source_array")
    branch_b_array = neutral_mapping.get("branch_b_source_array")
    neutral = {
        "branch_a": raw_branches.get(str(branch_a_array)),
        "branch_b": raw_branches.get(str(branch_b_array)),
    }
    lad_source = resolved_mapping.get("lad_source")
    lcx_source = resolved_mapping.get("lcx_source")
    if {lad_source, lcx_source} != {"branch_a", "branch_b"}:
        raise ValueError(f"{case_id}: malformed resolved daughter assignment")
    if neutral[lad_source] is None or neutral[lcx_source] is None:
        raise ValueError(f"{case_id}: resolved assignment references an unavailable raw array")
    result["lad"] = np.asarray(neutral[lad_source]).copy()
    result["lcx"] = np.asarray(neutral[lcx_source]).copy()
    score = record.get("assignment_scoring", {}).get("score_margin")
    winner_score = record.get("assignment_scoring", {}).get("winning_score")
    metadata.update({
        "lad_source_array": neutral_mapping[f"{lad_source}_source_array"],
        "lcx_source_array": neutral_mapping[f"{lcx_source}_source_array"],
        "assignment_score_margin": score,
        "assignment_winner_score": winner_score,
    })
    return result, metadata


def ellipse_z_alignments(patient: dict[str, Any], rotation: np.ndarray) -> tuple[float, float]:
    plane = patient["planes"]["interventricular_LAD"]
    ellipse = patient["ellipses"]["interventricular_LAD"]
    u = np.asarray(plane["basis_u_ras"], dtype=float)
    v = np.asarray(plane["basis_v_ras"], dtype=float)
    tilt = float(ellipse["tilt_rad"])
    major = math.cos(tilt) * u + math.sin(tilt) * v
    minor = -math.sin(tilt) * u + math.cos(tilt) * v
    return abs(float((rotation @ major)[2])), abs(float((rotation @ minor)[2]))


def normalized_arc(points: np.ndarray) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    return cumulative / max(float(cumulative[-1]), 1.0e-12)


def branch_length(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def branch_tortuosity(points: np.ndarray) -> float:
    return branch_length(points) / max(float(np.linalg.norm(points[-1] - points[0])), 1.0e-12)


def interpolate_fixed_scaffold(points: np.ndarray, sample_count: int = 180) -> np.ndarray:
    """Shape-preservingly smooth fixed real controls in cardiac XYZ."""
    points = np.asarray(points, dtype=float)
    source = np.linspace(0.0, 1.0, len(points))
    target = np.linspace(0.0, 1.0, sample_count)
    result = np.column_stack([
        PchipInterpolator(source, points[:, axis])(target) for axis in range(3)
    ])
    result[0] = points[0]
    result[-1] = points[-1]
    return result


def path_progression_metrics(points: np.ndarray, sample_count: int = 60) -> dict[str, float]:
    """Measure loop/backtracking behaviour on uniform arc-length samples."""
    target = np.linspace(0.0, 1.0, sample_count)
    resampled = np.column_stack([
        np.interp(target, normalized_arc(points), points[:, dimension]) for dimension in range(3)
    ])
    segments = np.diff(resampled, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    unit_segments = segments / np.maximum(segment_lengths[:, None], 1.0e-12)
    turns = np.arccos(np.clip(np.sum(unit_segments[:-1] * unit_segments[1:], axis=1), -1.0, 1.0))
    chord = resampled[-1] - resampled[0]
    chord_length = max(float(np.linalg.norm(chord)), 1.0e-12)
    chord_progress = segments @ (chord / chord_length)
    terminal_distance = np.linalg.norm(resampled - resampled[-1], axis=1)
    return {
        "backward_progress_ratio": float(-np.sum(np.minimum(chord_progress, 0.0)) / chord_length),
        "terminal_progress_fraction": float(np.mean(np.diff(terminal_distance) <= 0.0)),
        "max_resampled_turn_angle_deg": float(np.degrees(np.max(turns))) if len(turns) else 0.0,
    }


def daughter_angle_deg(lad: np.ndarray, lcx: np.ndarray) -> float:
    lad_index = min(max(int(round(0.04 * (len(lad) - 1))), 1), len(lad) - 1)
    lcx_index = min(max(int(round(0.04 * (len(lcx) - 1))), 1), len(lcx) - 1)
    lad_direction = lad[lad_index] - lad[0]
    lcx_direction = lcx[lcx_index] - lcx[0]
    denominator = float(np.linalg.norm(lad_direction) * np.linalg.norm(lcx_direction))
    if denominator <= 1.0e-12:
        return math.nan
    cosine = float(np.clip(np.dot(lad_direction, lcx_direction) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def landmark_source_points(branches: dict[str, np.ndarray]) -> dict[str, tuple[str, int, np.ndarray]]:
    result = {
        "lca_ostium": ("lmca", 0, branches["lmca"][0]),
        "bifurcation": ("lmca", len(branches["lmca"]) - 1, branches["lmca"][-1]),
        "lad_endpoint": ("lad", len(branches["lad"]) - 1, branches["lad"][-1]),
        "lcx_endpoint": ("lcx", len(branches["lcx"]) - 1, branches["lcx"][-1]),
    }
    if "rca" in branches:
        result["rca_ostium"] = ("rca", 0, branches["rca"][0])
        result["rca_endpoint"] = ("rca", len(branches["rca"]) - 1, branches["rca"][-1])
    return result


def trajectory_statistics(records: dict[str, list[np.ndarray]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for branch, values in records.items():
        if not values:
            continue
        array = np.stack(values)
        unwrapped_u = np.unwrap(array[:, :, 0], axis=1)
        result[branch] = {
            "case_count": len(array),
            "point_count": array.shape[1],
            "u_unwrapped_mean_rad": np.mean(unwrapped_u, axis=0).tolist(),
            "u_unwrapped_std_rad": np.std(unwrapped_u, axis=0, ddof=1).tolist() if len(array) > 1 else np.zeros(array.shape[1]).tolist(),
            "v_mean_rad": np.mean(array[:, :, 1], axis=0).tolist(),
            "v_std_rad": np.std(array[:, :, 1], axis=0, ddof=1).tolist() if len(array) > 1 else np.zeros(array.shape[1]).tolist(),
            "offset_mean_mm": np.mean(array[:, :, 2], axis=0).tolist(),
            "offset_std_mm": np.std(array[:, :, 2], axis=0, ddof=1).tolist() if len(array) > 1 else np.zeros(array.shape[1]).tolist(),
        }
    return result


def landmark_statistics(rows: list[dict[str, Any]], eligible_cases: set[str]) -> dict[str, Any]:
    names = sorted({row["landmark_name"] for row in rows if row["case_id"] in eligible_cases})
    summaries: dict[str, Any] = {}
    for name in names:
        selected = [row for row in rows if row["case_id"] in eligible_cases and row["landmark_name"] == name]
        u_stats = compute_circular_stats([row["u"] for row in selected])
        v_stats = compute_linear_stats([row["v"] for row in selected])
        offset_stats = compute_linear_stats([row["offset"] for row in selected])
        summaries[name] = {
            "count": len(selected),
            "u_mean": u_stats["circular_mean_rad"], "u_std": u_stats["circular_std_rad"],
            "v_mean": v_stats["mean"], "v_std": v_stats["std"],
            "offset_mean": offset_stats["mean"], "offset_std": offset_stats["std"],
            "v_p2_5": v_stats["p2_5"], "v_p97_5": v_stats["p97_5"],
            "offset_p2_5": offset_stats["p2_5"], "offset_p97_5": offset_stats["p97_5"],
        }
    by_case: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["case_id"] in eligible_cases:
            by_case.setdefault(row["case_id"], {})[row["landmark_name"]] = {
                "u": row["u"], "v": row["v"], "offset": row["offset"]
            }
    complete = [
        {"case_id": case_id, "landmarks": values}
        for case_id, values in sorted(by_case.items(), key=lambda item: numeric_case_key(item[0]))
        if all(name in values for name in ("lca_ostium", "bifurcation", "lad_endpoint", "lcx_endpoint"))
    ]
    return {
        "schema_version": 1,
        "coordinate_system": "common cardiac frame surface coordinates",
        "angular_units": "radians", "linear_units": "millimetres",
        "recommended_sampling": "empirical_joint_case_bootstrap_to_preserve_landmark_dependence",
        "landmarks": summaries, "empirical_cases": complete,
    }


def freeze_generator_package(output: Path, filenames: Iterable[str], metadata: dict[str, Any]) -> dict[str, Any]:
    package = output / "generator_statistics"
    package.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in filenames:
        source = output / name
        if source.is_file():
            shutil.copy2(source, package / name)
            copied.append(name)
    manifest = {
        **metadata, "frozen": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": {name: file_sha256(package / name) for name in copied},
    }
    write_json(package / "generator_statistics_manifest.json", manifest)
    return manifest


def run_pipeline(*, smoke_test: bool, output: Path, clean: bool) -> dict[str, Any]:
    if output.exists():
        if not clean:
            raise FileExistsError(f"refusing to overwrite {output}; pass --clean explicitly")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    protected_before = protected_snapshot()
    parameters = read_ppt_parameters(PPT_OUTPUT_DIR / "population_two_plane_two_ellipse_parameters.csv")
    assignment_records = load_assignment_records(ASSIGNMENT_MAP_PATH)
    case_ids = SMOKE_TEST_CASES if smoke_test else sorted(parameters, key=numeric_case_key)

    frame_rows: list[dict[str, Any]] = []
    frame_validation_rows: list[dict[str, Any]] = []
    ellipsoid_rows: list[dict[str, Any]] = []
    surface_rows: list[dict[str, Any]] = []
    landmark_rows: list[dict[str, Any]] = []
    case_qc_rows: list[dict[str, Any]] = []
    assignment_gate_rows: list[dict[str, Any]] = []
    trajectory_records: dict[str, list[np.ndarray]] = {name: [] for name in ("LMCA", "LAD", "LCX", "RCA")}
    trajectory_deviation_records: dict[str, list[np.ndarray]] = {
        name: [] for name in trajectory_records
    }
    trajectory_cardiac_records: dict[str, list[np.ndarray]] = {
        name: [] for name in trajectory_records
    }
    trajectory_case_ids: dict[str, list[str]] = {name: [] for name in trajectory_records}
    shape_vectors: list[np.ndarray] = []
    complete_case_ids: list[str] = []
    fixed_complete: list[np.ndarray] = []
    valid_scaffolds: list[dict[str, float]] = []
    branch_lengths: dict[str, list[float]] = {name: [] for name in ("lmca", "lad", "lcx", "rca")}
    branch_tortuosities: dict[str, list[float]] = {name: [] for name in branch_lengths}
    branch_obliquities: dict[str, list[float]] = {name: [] for name in branch_lengths}
    branch_backward_ratios: dict[str, list[float]] = {name: [] for name in branch_lengths}
    branch_terminal_progress: dict[str, list[float]] = {name: [] for name in branch_lengths}
    branch_max_turn_angles: dict[str, list[float]] = {name: [] for name in branch_lengths}
    bifurcation_angles: list[float] = []
    scaffold_branch_lengths: dict[str, list[float]] = {name: [] for name in branch_lengths}
    scaffold_branch_tortuosities: dict[str, list[float]] = {name: [] for name in branch_lengths}
    scaffold_branch_obliquities: dict[str, list[float]] = {name: [] for name in branch_lengths}
    scaffold_branch_backward_ratios: dict[str, list[float]] = {name: [] for name in branch_lengths}
    scaffold_branch_terminal_progress: dict[str, list[float]] = {name: [] for name in branch_lengths}
    scaffold_branch_max_turn_angles: dict[str, list[float]] = {name: [] for name in branch_lengths}
    scaffold_bifurcation_angles: list[float] = []
    scaffold_validation_rows: list[dict[str, Any]] = []

    for case_id in case_ids:
        raw_branches, archive_path = load_case(case_id)
        original = {name: points.copy() for name, points in raw_branches.items()}
        patient_path = PPT_OUTPUT_DIR / "patients" / case_id / "two_plane_two_ellipse.json"
        patient = json.loads(patient_path.read_text(encoding="utf-8"))
        archive_hash_match = file_sha256(archive_path) == patient["source_integrity"]["raw_archive_sha256"]
        source_hash_match = all(
            array_sha256(points) == patient["source_integrity"]["source_hashes"][name]
            for name, points in raw_branches.items()
        )
        if not archive_hash_match or not source_hash_match:
            raise RuntimeError(f"immutable source integrity failed for {case_id}")
        branches, assignment_metadata = apply_resolved_roles(
            case_id, raw_branches, assignment_records
        )

        coronary_plane = patient["planes"]["coronary_AV_groove"]
        lad_plane = patient["planes"]["interventricular_LAD"]
        origin, rotation, frame_metadata = compute_cardiac_frame(
            coronary_normal=coronary_plane["normal_ras"], iv_normal=lad_plane["normal_ras"],
            coronary_centroid=coronary_plane["centroid_ras_mm"], lad_centerline=branches["lad"],
            lca_ostium=branches["lmca"][0],
        )
        frame_validation = validate_cardiac_frame(origin, rotation, branches)
        frame_rows.append({
            "case_id": case_id,
            **{f"origin_{axis}": origin[index] for index, axis in enumerate("xyz")},
            **{f"R_{row}{column}": rotation[row, column] for row in range(3) for column in range(3)},
            "delta_theta_rad": frame_metadata["delta_theta_rad"], "determinant": frame_validation["determinant"],
        })
        frame_validation_rows.append({"case_id": case_id, **frame_validation})
        cardiac = {name: transform_to_cardiac_frame(points, origin, rotation) for name, points in branches.items()}
        anatomy_metrics = coronary_course_metrics(cardiac["lad"], cardiac["lcx"])
        anatomy_acceptance = anatomical_role_acceptance(anatomy_metrics)
        major_alignment, minor_alignment = ellipse_z_alignments(patient, rotation)
        ellipsoid: PatientEllipsoid = derive_patient_ellipsoid(
            parameters[case_id], lad_major_z_alignment=major_alignment, lad_minor_z_alignment=minor_alignment,
        )
        lad_required_z = float(np.max(np.abs(cardiac["lad"][:, 2])))
        if ellipsoid.c < 0.45 * lad_required_z:
            ellipsoid.exclusion_flags.append("c_axis_below_45pct_of_observed_lad_z_extent")
        if ellipsoid.c > 3.0 * max(lad_required_z, 1.0):
            ellipsoid.exclusion_flags.append("c_axis_above_3x_observed_lad_z_extent")
        ellipsoid.exclusion_flags = sorted(set(ellipsoid.exclusion_flags))
        fixed = build_patient_fixed_representation(
            cardiac, ellipsoid.a, ellipsoid.b, ellipsoid.c
        )
        scaffold_paths = {
            name.lower(): interpolate_fixed_scaffold(
                values["cardiac_points"], VALIDATION_SCAFFOLD_SAMPLE_COUNTS[name.lower()]
            )
            for name, values in fixed["fixed_branches"].items()
            if values is not None
        }
        scaffold_anatomy_metrics = coronary_course_metrics(
            scaffold_paths["lad"], scaffold_paths["lcx"]
        )
        scaffold_anatomy_acceptance = anatomical_role_acceptance(
            scaffold_anatomy_metrics
        )
        core_anatomy_failures = sorted(
            CORE_ANATOMICAL_ROLE_CHECKS.intersection(anatomy_acceptance["failed_checks"])
        )
        scaffold_core_anatomy_failures = sorted(
            CORE_ANATOMICAL_ROLE_CHECKS.intersection(
                scaffold_anatomy_acceptance["failed_checks"]
            )
        )
        # This is a topology-relative anatomical rule, not a manually tuned
        # millimetre cutoff: the left-main trunk must not be longer than either
        # of the two major daughter courses it supplies.
        if branch_length(cardiac["lmca"]) >= min(
            branch_length(cardiac["lad"]), branch_length(cardiac["lcx"])
        ):
            core_anatomy_failures.append("lmca_is_shorter_than_both_daughters")
        if branch_length(scaffold_paths["lmca"]) >= min(
            branch_length(scaffold_paths["lad"]), branch_length(scaffold_paths["lcx"])
        ):
            scaffold_core_anatomy_failures.append("lmca_is_shorter_than_both_daughters")
        core_anatomy_failures = sorted(set(core_anatomy_failures))
        scaffold_core_anatomy_failures = sorted(set(scaffold_core_anatomy_failures))
        # Assignment confidence, immutable-source integrity, rigid-frame
        # validity and ellipsoid validity are hard inclusion criteria. The
        # absolute-size anatomy thresholds remain descriptive because they are
        # manually selected geometric limits, not clinical truth. Four
        # relationship checks remain hard: LMCA must remain the short trunk,
        # LAD the dominant descending branch, and LCX the more lateral,
        # horizontal crown branch.
        eligible = bool(
            assignment_metadata["resolved_for_statistics"]
            and fixed["is_lca_complete"]
            and ellipsoid.is_valid
            and frame_validation["overall_pass"]
            and archive_hash_match
            and source_hash_match
            and not core_anatomy_failures
            and not scaffold_core_anatomy_failures
        )
        ellipsoid_rows.append({
            "case_id": case_id, "a": ellipsoid.a, "b": ellipsoid.b, "c": ellipsoid.c,
            "ellipsoid_center_x": 0.0, "ellipsoid_center_y": 0.0, "ellipsoid_center_z": 0.0,
            "ellipse_center_separation": ellipsoid.ellipse_center_separation, "c_source": ellipsoid.c_source,
            "lad_major_z_alignment": major_alignment, "lad_minor_z_alignment": minor_alignment,
            "observed_lad_max_abs_z": lad_required_z, "quality_flags": ";".join(ellipsoid.quality_flags),
            "exclusion_flags": ";".join(ellipsoid.exclusion_flags),
            "intrinsic_ellipsoid_valid": ellipsoid.is_valid, "is_valid": eligible,
            "assignment_resolution_status": assignment_metadata["resolution_status"],
            "anatomical_role_gate_pass": anatomy_acceptance["accepted"],
            "scaffold_anatomical_role_gate_pass": scaffold_anatomy_acceptance["accepted"],
            "core_anatomical_role_gate_pass": not core_anatomy_failures,
            "core_anatomical_role_gate_failures": ";".join(core_anatomy_failures),
            "scaffold_core_anatomical_role_gate_pass": not scaffold_core_anatomy_failures,
            "scaffold_core_anatomical_role_gate_failures": ";".join(scaffold_core_anatomy_failures),
            **{f"raw_{name}": value for name, value in ellipsoid.raw_axes.items()},
        })

        max_coordinate_change = max(
            float(np.max(np.abs(raw_branches[name] - original[name]))) for name in raw_branches
        )
        max_segment_change = max(
            float(np.max(np.abs(
                np.linalg.norm(np.diff(raw_branches[name], axis=0), axis=1)
                - np.linalg.norm(np.diff(original[name], axis=0), axis=1)
            ))) for name in raw_branches
        )
        gate_failures = []
        if not assignment_metadata["resolved_for_statistics"]:
            gate_failures.append(f"assignment:{assignment_metadata['resolution_status']}")
        gate_failures.extend(f"anatomy:{name}" for name in anatomy_acceptance["failed_checks"])
        gate_failures.extend(
            f"scaffold_anatomy:{name}"
            for name in scaffold_anatomy_acceptance["failed_checks"]
        )
        gate_failures.extend(f"ellipsoid:{name}" for name in ellipsoid.exclusion_flags)
        if not frame_validation["overall_pass"]:
            gate_failures.append("cardiac_frame:failed")
        case_qc_rows.append({
            "case_id": case_id, "statistics_eligible": eligible,
            "frame_pass": frame_validation["overall_pass"], "ellipsoid_valid": ellipsoid.is_valid,
            "assignment_resolution_status": assignment_metadata["resolution_status"],
            "assignment_resolved": assignment_metadata["resolved_for_statistics"],
            "anatomical_role_gate_pass": anatomy_acceptance["accepted"],
            "anatomical_role_gate_failures": ";".join(anatomy_acceptance["failed_checks"]),
            "scaffold_anatomical_role_gate_pass": scaffold_anatomy_acceptance["accepted"],
            "scaffold_anatomical_role_gate_failures": ";".join(
                scaffold_anatomy_acceptance["failed_checks"]
            ),
            "has_rca": "rca" in branches, "source_archive_hash_match": archive_hash_match,
            "source_array_hashes_match": source_hash_match,
            "max_source_coordinate_change_mm": max_coordinate_change,
            "max_source_segment_length_change_mm": max_segment_change,
            "LMCA_to_LAD_source_continuity_mm": float(np.linalg.norm(branches["lmca"][-1] - branches["lad"][0])),
            "LMCA_to_LCX_source_continuity_mm": float(np.linalg.norm(branches["lmca"][-1] - branches["lcx"][0])),
            "exclusion_flags": ";".join(gate_failures),
            **anatomy_metrics,
        })
        assignment_gate_rows.append({
            **assignment_metadata,
            "anatomical_role_gate_pass": anatomy_acceptance["accepted"],
            "anatomical_role_gate_failures": anatomy_acceptance["failed_checks"],
            "scaffold_anatomical_role_gate_pass": scaffold_anatomy_acceptance["accepted"],
            "scaffold_anatomical_role_gate_failures": scaffold_anatomy_acceptance["failed_checks"],
            "core_anatomical_role_gate_pass": not core_anatomy_failures,
            "core_anatomical_role_gate_failures": core_anatomy_failures,
            "scaffold_core_anatomical_role_gate_pass": not scaffold_core_anatomy_failures,
            "scaffold_core_anatomical_role_gate_failures": scaffold_core_anatomy_failures,
            "statistics_eligible": eligible,
            "metrics": anatomy_metrics,
            "scaffold_metrics": scaffold_anatomy_metrics,
        })

        for name, points in branches.items():
            cardiac_points = cardiac[name]
            s_values = normalized_arc(points)
            progression = path_progression_metrics(points)
            if eligible:
                branch_lengths[name].append(branch_length(points))
                branch_tortuosities[name].append(branch_tortuosity(points))
                branch_backward_ratios[name].append(progression["backward_progress_ratio"])
                branch_terminal_progress[name].append(progression["terminal_progress_fraction"])
                branch_max_turn_angles[name].append(progression["max_resampled_turn_angle_deg"])
            projected_u: list[float] = []
            for index, cardiac_point in enumerate(cardiac_points):
                u, v, offset_value, deviation = project_point_to_surface(
                    cardiac_point, ellipsoid.a, ellipsoid.b, ellipsoid.c
                )
                projected_u.append(u)
                surface_rows.append({
                    "case_id": case_id, "branch": name.upper(), "source_point_index": index,
                    "normalized_arc_position": s_values[index],
                    "source_x": points[index, 0], "source_y": points[index, 1], "source_z": points[index, 2],
                    "cardiac_x": cardiac_point[0], "cardiac_y": cardiac_point[1], "cardiac_z": cardiac_point[2],
                    "u": u, "v": v, "offset": offset_value,
                    "deviation_tangent_u": deviation[0], "deviation_tangent_v": deviation[1],
                    "deviation_normal": deviation[2], "statistics_eligible": eligible,
                })
            unwrapped_u = np.unwrap(np.asarray(projected_u))
            if eligible:
                branch_obliquities[name].append(float(unwrapped_u[-1] - unwrapped_u[0]))

        angle = daughter_angle_deg(cardiac["lad"], cardiac["lcx"])
        if eligible and np.isfinite(angle):
            bifurcation_angles.append(angle)
        for landmark_name, (branch_name, point_index, source_point) in landmark_source_points(branches).items():
            cardiac_point = cardiac[branch_name][point_index]
            u, v, offset_value, _ = project_point_to_surface(cardiac_point, ellipsoid.a, ellipsoid.b, ellipsoid.c)
            landmark_rows.append({
                "case_id": case_id, "landmark_name": landmark_name, "source_branch": branch_name.upper(),
                "source_point_index": point_index,
                "source_x": source_point[0], "source_y": source_point[1], "source_z": source_point[2],
                "cardiac_x": cardiac_point[0], "cardiac_y": cardiac_point[1], "cardiac_z": cardiac_point[2],
                "u": u, "v": v, "offset": offset_value, "statistics_eligible": eligible,
            })

        if eligible:
            valid_scaffolds.append({"a": ellipsoid.a, "b": ellipsoid.b, "c": ellipsoid.c})
            scaffold_row: dict[str, Any] = {
                "case_id": case_id,
                **scaffold_anatomy_metrics,
            }
            for name, scaffold_points in scaffold_paths.items():
                length_value = branch_length(scaffold_points)
                tortuosity_value = branch_tortuosity(scaffold_points)
                progression = path_progression_metrics(scaffold_points)
                projected_u = np.unwrap(np.asarray([
                    project_point_to_surface(point, ellipsoid.a, ellipsoid.b, ellipsoid.c)[0]
                    for point in scaffold_points
                ]))
                obliquity_value = float(projected_u[-1] - projected_u[0])
                scaffold_branch_lengths[name].append(length_value)
                scaffold_branch_tortuosities[name].append(tortuosity_value)
                scaffold_branch_obliquities[name].append(obliquity_value)
                scaffold_branch_backward_ratios[name].append(progression["backward_progress_ratio"])
                scaffold_branch_terminal_progress[name].append(progression["terminal_progress_fraction"])
                scaffold_branch_max_turn_angles[name].append(progression["max_resampled_turn_angle_deg"])
                scaffold_row[f"{name}_length_mm"] = length_value
                scaffold_row[f"{name}_tortuosity"] = tortuosity_value
                scaffold_row[f"{name}_obliquity_rad"] = obliquity_value
                for metric_name, value in progression.items():
                    scaffold_row[f"{name}_{metric_name}"] = value
            scaffold_angle = daughter_angle_deg(scaffold_paths["lad"], scaffold_paths["lcx"])
            if np.isfinite(scaffold_angle):
                scaffold_bifurcation_angles.append(scaffold_angle)
            scaffold_row["bifurcation_angle_deg"] = scaffold_angle
            scaffold_validation_rows.append(scaffold_row)
            for branch_name, fixed_values in fixed["fixed_branches"].items():
                if fixed_values is not None:
                    trajectory_records[branch_name].append(fixed_values["uvo"])
                    trajectory_deviation_records[branch_name].append(
                        fixed["deviations"][branch_name]
                    )
                    trajectory_cardiac_records[branch_name].append(
                        fixed_values["cardiac_points"]
                    )
                    trajectory_case_ids[branch_name].append(case_id)
            if fixed["is_lca_complete"]:
                fixed_complete.append(fixed["lca_uvo_matrix_27_3"])
                shape_vectors.append(fixed["lca_shape_vector"])
                complete_case_ids.append(case_id)

    if not valid_scaffolds:
        raise RuntimeError("no cases passed frame and ellipsoid quality control")
    write_csv(output / "population_cardiac_frames.csv", frame_rows)
    write_csv(output / "cardiac_frame_validation.csv", frame_validation_rows)
    write_csv(output / "population_ellipsoid_parameters.csv", ellipsoid_rows)
    write_csv(output / "population_surface_coordinates.csv", surface_rows)
    write_csv(output / "population_landmarks.csv", landmark_rows)
    write_csv(output / "population_case_qc.csv", case_qc_rows)
    write_csv(output / "validation_scaffold_case_metrics.csv", scaffold_validation_rows)
    write_json(output / "branch_assignment_gate.json", {
        "assignment_source": "immutable raw_cases/*/source_metadata.json",
        "assignment_method": "per-case multi-signal RAS anatomy score",
        "minimum_assignment_score_margin": MIN_ASSIGNMENT_SCORE_MARGIN,
        "minimum_assignment_winner_score": MIN_ASSIGNMENT_WINNER_SCORE,
        "accepted_resolution_statuses": sorted(RESOLVED_ASSIGNMENT_STATUSES),
        "input_case_count": len(case_ids),
        "resolved_assignment_count": sum(
            bool(row["resolved_for_statistics"]) for row in assignment_gate_rows
        ),
        "anatomy_gate_pass_count": sum(
            bool(row["resolved_for_statistics"] and row["anatomical_role_gate_pass"])
            for row in assignment_gate_rows
        ),
        "core_anatomy_gate_pass_count": sum(
            bool(
                row["resolved_for_statistics"]
                and row["core_anatomical_role_gate_pass"]
                and row["scaffold_core_anatomical_role_gate_pass"]
            )
            for row in assignment_gate_rows
        ),
        "core_anatomical_role_checks": sorted(CORE_ANATOMICAL_ROLE_CHECKS),
        "statistics_eligible_count": sum(bool(row["statistics_eligible"]) for row in assignment_gate_rows),
        "unresolved_cases_used_for_statistics": [],
        "cases": assignment_gate_rows,
    })

    fixed_array = (
        np.stack(fixed_complete)
        if fixed_complete
        else np.zeros((0, LCA_TOTAL_FIXED_POINTS, 3), dtype=float)
    )
    np.savez_compressed(
        output / "fixed_surface_representation.npz", fixed_surface_representation=fixed_array,
        case_ids=np.asarray(complete_case_ids), branch_order=np.asarray(LCA_BRANCH_ORDER),
        branch_counts=np.asarray([LCA_FIXED_COUNTS[name] for name in LCA_BRANCH_ORDER]),
    )
    branch_npz: dict[str, np.ndarray] = {}
    for name, values in trajectory_records.items():
        branch_npz[f"{name}_uvo"] = np.stack(values) if values else np.zeros((0, FIXED_COUNTS[name], 3))
        deviations = trajectory_deviation_records[name]
        cardiac_points = trajectory_cardiac_records[name]
        branch_npz[f"{name}_local_deviation"] = (
            np.stack(deviations) if deviations else np.zeros((0, FIXED_COUNTS[name], 3))
        )
        branch_npz[f"{name}_cardiac_points"] = (
            np.stack(cardiac_points) if cardiac_points else np.zeros((0, FIXED_COUNTS[name], 3))
        )
        branch_npz[f"{name}_case_ids"] = np.asarray(trajectory_case_ids[name])
    np.savez_compressed(output / "fixed_branch_surface_coordinates.npz", **branch_npz)

    pca_results = None
    if len(shape_vectors) >= 2:
        pca_results = fit_deviation_pca(np.stack(shape_vectors), variance_cutoff=0.95)
        np.savez_compressed(
            output / "surface_deviation_pca.npz", mean_vector=pca_results["mean_vector"],
            components=pca_results["components_retained"], all_components=pca_results["components_all"],
            singular_values=pca_results["singular_values"], eigenvalues=pca_results["eigenvalues"],
            explained_variance_ratio=pca_results["explained_variance_ratio"],
            training_scores=pca_results["training_scores"],
            standardized_training_scores=pca_results["standardized_training_scores"],
            complete_case_ids=np.asarray(complete_case_ids),
            branch_order=np.asarray(LCA_BRANCH_ORDER),
            branch_counts=np.asarray([LCA_FIXED_COUNTS[name] for name in LCA_BRANCH_ORDER]),
        )
        write_json(output / "surface_deviation_pca_summary.json", {
            "n_samples": pca_results["n_samples"], "n_features": pca_results["n_features"],
            "k_retained": pca_results["k_retained"], "variance_cutoff": pca_results["variance_cutoff"],
            "cumulative_variance_retained": pca_results["cumulative_explained_variance"][pca_results["k_retained"] - 1],
            "explained_variance_ratio": pca_results["explained_variance_ratio"],
            "eigenvalues": pca_results["eigenvalues"], "case_ids": complete_case_ids,
            "training_score_shape": list(pca_results["training_scores"].shape),
            "training_scores_are_case_matched": True,
            "model_scope": "LCA_only",
            "branch_order": list(LCA_BRANCH_ORDER),
            "branch_counts": LCA_FIXED_COUNTS,
        })

    eligible_cases = {row["case_id"] for row in case_qc_rows if row["statistics_eligible"]}
    landmark_stats_payload = landmark_statistics(landmark_rows, eligible_cases)
    write_json(output / "landmark_stats.json", landmark_stats_payload)
    population_stats = {
        "case_counts": {"input": len(case_ids), "statistics_eligible": len(eligible_cases),
                        "complete_lca_pca": len(complete_case_ids)},
        "sampling_policy": {
            "ellipsoid": "empirical_joint_bootstrap_from_valid_rows",
            "landmarks": "empirical_joint_case_bootstrap",
            "trajectory": "matched_resolved_case_bootstrap_with_exact_local_basis_coefficients",
            "deviation": "joint_81D_LCA_PCA_innovation_around_exact_matched_baseline",
            "assignment_gate": "per_case_multi_signal_RAS_score_with_margin_and_LCX_apex_exclusion",
            "unresolved_assignments_used": False,
            "independent_pointwise_noise": False,
        },
        "scaffold": {axis: compute_linear_stats([row[axis] for row in valid_scaffolds]) for axis in ("a", "b", "c")},
        "landmarks": landmark_stats_payload["landmarks"],
        "fixed_branch_trajectories": trajectory_statistics(trajectory_records),
        "branch_lengths_mm": {name: compute_linear_stats(values) for name, values in branch_lengths.items()},
        "branch_tortuosity": {name: compute_linear_stats(values) for name, values in branch_tortuosities.items()},
        "branch_obliquity_rad": {name: compute_linear_stats(values) for name, values in branch_obliquities.items()},
        "branch_backward_progress_ratio": {
            name: compute_linear_stats(values) for name, values in branch_backward_ratios.items()
        },
        "branch_terminal_progress_fraction": {
            name: compute_linear_stats(values) for name, values in branch_terminal_progress.items()
        },
        "branch_max_resampled_turn_angle_deg": {
            name: compute_linear_stats(values) for name, values in branch_max_turn_angles.items()
        },
        "bifurcation_angle_deg": compute_linear_stats(bifurcation_angles),
        "validation_scaffold_reference": {
            "construction": "shape_preserving_interpolation_of_fixed_real_cardiac_controls",
            "case_count": len(scaffold_validation_rows),
            "branch_lengths_mm": {
                name: compute_linear_stats(values) for name, values in scaffold_branch_lengths.items()
            },
            "branch_tortuosity": {
                name: compute_linear_stats(values) for name, values in scaffold_branch_tortuosities.items()
            },
            "branch_obliquity_rad": {
                name: compute_linear_stats(values) for name, values in scaffold_branch_obliquities.items()
            },
            "bifurcation_angle_deg": compute_linear_stats(scaffold_bifurcation_angles),
        },
    }
    write_json(output / "population_surface_statistics.json", population_stats)
    thresholds = build_validation_thresholds(
        valid_scaffolds,
        scaffold_branch_lengths,
        scaffold_bifurcation_angles,
        pca_results=pca_results,
    )
    thresholds["validation_reference"] = {
        "geometry": "shape-preserving fixed-control scaffolds from confidence-resolved LCA cases",
        "raw_centerline_metrics_are_descriptive_only": True,
    }
    thresholds["branch_tortuosity"] = {
        name: {
            key: compute_linear_stats(values)[key]
            for key in ("min", "max", "p2_5", "p97_5")
        }
        for name, values in scaffold_branch_tortuosities.items()
    }
    thresholds["branch_obliquity_rad"] = {
        name: {
            key: compute_linear_stats(values)[key]
            for key in ("min", "max", "p2_5", "p97_5")
        }
        for name, values in scaffold_branch_obliquities.items()
    }
    thresholds["branch_backward_progress_ratio"] = {
        name: {
            "max": compute_linear_stats(values)["max"],
            "p97_5": compute_linear_stats(values)["p97_5"],
        }
        for name, values in scaffold_branch_backward_ratios.items()
    }
    thresholds["branch_terminal_progress_fraction"] = {
        name: {
            "min": compute_linear_stats(values)["min"],
            "p2_5": compute_linear_stats(values)["p2_5"],
        }
        for name, values in scaffold_branch_terminal_progress.items()
    }
    thresholds["branch_max_resampled_turn_angle_deg"] = {
        name: {
            "max": compute_linear_stats(values)["max"],
            "p97_5": compute_linear_stats(values)["p97_5"],
        }
        for name, values in scaffold_branch_max_turn_angles.items()
    }
    write_json(output / "population_validation_thresholds.json", thresholds)

    write_json(output / "fixed_surface_representation_summary.json", {
        "n_input_cases": len(case_ids), "n_statistics_eligible": len(eligible_cases),
        "n_complete_cases": len(complete_case_ids), "complete_case_ids": complete_case_ids,
        "model_scope": "LCA_only", "total_fixed_points": LCA_TOTAL_FIXED_POINTS,
        "branch_counts": LCA_FIXED_COUNTS, "fixed_matrix_shape": list(fixed_array.shape),
    })
    package_files = [
        "population_surface_statistics.json", "population_ellipsoid_parameters.csv", "landmark_stats.json",
        "surface_deviation_pca.npz", "surface_deviation_pca_summary.json",
        "population_validation_thresholds.json", "fixed_branch_surface_coordinates.npz",
        "branch_assignment_gate.json", "validation_scaffold_case_metrics.csv",
    ]
    package_manifest = freeze_generator_package(output, package_files, {
        "schema_version": 3, "model_scope": "LCA_only",
        "shape_vector_dimensions": 3 * LCA_TOTAL_FIXED_POINTS,
        "branch_order": list(LCA_BRANCH_ORDER),
        "branch_counts": LCA_FIXED_COUNTS,
        "source_population_case_count": len(case_ids),
        "statistics_eligible_case_count": len(eligible_cases), "pca_case_count": len(complete_case_ids),
        "resolved_assignment_count": sum(
            bool(row["resolved_for_statistics"]) for row in assignment_gate_rows
        ),
        "unresolved_assignments_used_for_statistics": False,
        "source_geometry_modified": False, "PPT_planes_or_ellipses_refitted": False,
    })
    protected_after = protected_snapshot()
    integrity = {"unchanged": protected_before == protected_after, "before": protected_before, "after": protected_after}
    write_json(output / "protected_source_integrity.json", integrity)
    if not integrity["unchanged"]:
        raise RuntimeError("protected PPT or raw-case input hashes changed during statistics build")
    manifest = {
        "implementation": "Complete real-LCA-data to frozen generator-statistics build",
        "created_utc": datetime.now(timezone.utc).isoformat(), "smoke_test": smoke_test, "status": "PASS",
        "input_case_count": len(case_ids),
        "frame_pass_count": sum(bool(row["overall_pass"]) for row in frame_validation_rows),
        "resolved_assignment_count": sum(
            bool(row["resolved_for_statistics"]) for row in assignment_gate_rows
        ),
        "unresolved_assignments_used_for_statistics": False,
        "statistics_eligible_count": len(eligible_cases), "complete_pca_case_count": len(complete_case_ids),
        "pca_retained_components": 0 if pca_results is None else pca_results["k_retained"],
        "protected_inputs_unchanged": integrity["unchanged"],
        "generator_statistics_manifest": package_manifest,
    }
    write_json(output / "week1_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke-test", action="store_true", help="Process the three historical smoke cases")
    mode.add_argument("--full", action="store_true", help="Process all 191 PPT cases")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    manifest = run_pipeline(smoke_test=not args.full, output=args.output_dir.resolve(), clean=args.clean)
    print(json.dumps({
        "status": manifest["status"], "input_case_count": manifest["input_case_count"],
        "statistics_eligible_count": manifest["statistics_eligible_count"],
        "complete_pca_case_count": manifest["complete_pca_case_count"],
        "output": str(args.output_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
