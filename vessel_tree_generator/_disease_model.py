import copy

import numpy as np

from .radius import arc_length_fraction


BRANCH_NAMES = ("LMCA", "LAD", "LCX")
SUPPORTED_LESION_TYPES = ("focal", "diffuse", "tandem")
SUPPORTED_PLAQUE_MODES = ("symmetric", "eccentric")


DEFAULT_DISEASE_SETTINGS = {
    "units": "mm",
    "minimum_radius_mm": 0.15,
    "adjacent_jump_threshold_mm": 0.35,
    "description": "MVP stenosis model; modifies radius only and keeps x, y, z unchanged.",
}


def example_disease_configs(patient_id: str) -> list:
    return [
        {
            "case_id": f"{patient_id}_lad_focal_70",
            "lesions": [
                {
                    "branch": "LAD",
                    "type": "focal",
                    "center": 0.45,
                    "length": 0.10,
                    "severity": 0.70,
                },
            ],
        },
        {
            "case_id": f"{patient_id}_lcx_diffuse_40",
            "lesions": [
                {
                    "branch": "LCX",
                    "type": "diffuse",
                    "start": 0.30,
                    "end": 0.70,
                    "severity": 0.40,
                },
            ],
        },
        {
            "case_id": f"{patient_id}_lad_tandem",
            "lesions": [
                {
                    "branch": "LAD",
                    "type": "focal",
                    "center": 0.35,
                    "length": 0.08,
                    "severity": 0.50,
                },
                {
                    "branch": "LAD",
                    "type": "focal",
                    "center": 0.65,
                    "length": 0.10,
                    "severity": 0.70,
                },
            ],
        },
    ]


def _smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def _as_float(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _validate_common_lesion_fields(lesion: dict) -> tuple:
    branch = str(lesion.get("branch", "")).upper()
    lesion_type = str(lesion.get("type", "")).lower()

    if branch not in BRANCH_NAMES:
        raise ValueError(f"Invalid branch '{branch}'. Expected one of {BRANCH_NAMES}.")
    if lesion_type not in SUPPORTED_LESION_TYPES:
        raise ValueError(f"Invalid lesion type '{lesion_type}'. Expected one of {SUPPORTED_LESION_TYPES}.")

    return branch, lesion_type


def _validate_severity(lesion: dict) -> float:
    severity = _as_float(lesion.get("severity"), "severity")
    if severity < 0.0 or severity >= 1.0:
        raise ValueError("severity must be >= 0 and < 1.")
    return severity


def _normalize_plaque_fields(lesion: dict) -> dict:
    plaque_mode = str(lesion.get("plaque_mode", "symmetric")).lower()
    if plaque_mode not in SUPPORTED_PLAQUE_MODES:
        raise ValueError(f"plaque_mode must be one of {SUPPORTED_PLAQUE_MODES}")

    plaque = {"plaque_mode": plaque_mode}
    if plaque_mode == "eccentric":
        plaque_angle = _as_float(lesion.get("plaque_angle", 0.0), "plaque_angle")
        eccentricity = _as_float(lesion.get("eccentricity", 0.75), "eccentricity")
        if eccentricity < 0.0 or eccentricity > 1.0:
            raise ValueError("eccentricity must be normalized from 0 to 1")
        plaque["plaque_angle"] = plaque_angle % 360.0
        plaque["eccentricity"] = eccentricity
    return plaque


def validate_disease_config(config: dict) -> dict:
    errors = []
    warnings = []
    normalized = copy.deepcopy(config)
    normalized.setdefault("case_id", "disease_case")
    normalized.setdefault("lesions", [])

    if not isinstance(normalized["lesions"], list) or len(normalized["lesions"]) == 0:
        errors.append("config must contain a non-empty lesions list")
        normalized["lesions"] = []

    flattened_lesions = []
    for lesion_index, lesion in enumerate(normalized["lesions"]):
        try:
            branch, lesion_type = _validate_common_lesion_fields(lesion)

            if lesion_type == "tandem":
                sublesions = lesion.get("lesions", [])
                if not isinstance(sublesions, list) or len(sublesions) == 0:
                    raise ValueError("tandem lesion must contain a non-empty lesions list")
                for sub_index, sublesion in enumerate(sublesions):
                    merged = copy.deepcopy(sublesion)
                    merged["branch"] = branch
                    merged.setdefault("type", "focal")
                    flattened_lesions.append(_validate_focal_lesion(merged))
                continue

            if lesion_type == "focal":
                flattened_lesions.append(_validate_focal_lesion(lesion))
            elif lesion_type == "diffuse":
                flattened_lesions.append(_validate_diffuse_lesion(lesion))
        except ValueError as exc:
            errors.append(f"lesion {lesion_index}: {exc}")

    normalized["lesions"] = flattened_lesions
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "normalized_config": normalized,
    }


def _validate_focal_lesion(lesion: dict) -> dict:
    branch, lesion_type = _validate_common_lesion_fields(lesion)
    if lesion_type != "focal":
        raise ValueError("nested tandem lesions currently support focal lesions only")

    center = _as_float(lesion.get("center"), "center")
    length = _as_float(lesion.get("length"), "length")
    severity = _validate_severity(lesion)
    if center < 0.0 or center > 1.0:
        raise ValueError("center must be normalized from 0 to 1")
    if length <= 0.0 or length > 1.0:
        raise ValueError("length must be > 0 and <= 1")
    half_length = 0.5 * length
    if center - half_length < 0.0 or center + half_length > 1.0:
        raise ValueError("focal lesion center +/- length/2 must stay inside the branch range")

    return {
        "branch": branch,
        "type": "focal",
        "center": center,
        "length": length,
        "severity": severity,
        **_normalize_plaque_fields(lesion),
    }


def _validate_diffuse_lesion(lesion: dict) -> dict:
    branch, lesion_type = _validate_common_lesion_fields(lesion)
    if lesion_type != "diffuse":
        raise ValueError("expected diffuse lesion")

    start = _as_float(lesion.get("start"), "start")
    end = _as_float(lesion.get("end"), "end")
    severity = _validate_severity(lesion)
    if start < 0.0 or start > 1.0 or end < 0.0 or end > 1.0:
        raise ValueError("start and end must be normalized from 0 to 1")
    if end <= start:
        raise ValueError("end must be greater than start")

    return {
        "branch": branch,
        "type": "diffuse",
        "start": start,
        "end": end,
        "severity": severity,
        **_normalize_plaque_fields(lesion),
    }


def focal_reduction_profile(s: np.ndarray, center: float, length: float, severity: float) -> np.ndarray:
    half_length = 0.5 * length
    distance = np.abs(s - center)
    profile = np.zeros_like(s, dtype=float)
    inside = distance <= half_length
    profile[inside] = 0.5 * (1.0 + np.cos(np.pi * distance[inside] / half_length))
    return severity * profile


def diffuse_reduction_profile(s: np.ndarray, start: float, end: float, severity: float) -> np.ndarray:
    span = end - start
    edge_width = min(0.15 * span, 0.08)
    if edge_width <= 1e-9:
        profile = ((s >= start) & (s <= end)).astype(float)
    else:
        left = _smoothstep((s - start) / edge_width)
        right = _smoothstep((end - s) / edge_width)
        profile = left * right
    return severity * profile


def lesion_reduction_profile(s: np.ndarray, lesion: dict) -> np.ndarray:
    if lesion["type"] == "focal":
        return focal_reduction_profile(
            s,
            center=lesion["center"],
            length=lesion["length"],
            severity=lesion["severity"],
        )
    if lesion["type"] == "diffuse":
        return diffuse_reduction_profile(
            s,
            start=lesion["start"],
            end=lesion["end"],
            severity=lesion["severity"],
        )
    raise ValueError(f"Unsupported lesion type: {lesion['type']}")


def apply_disease_to_branch(
    centerline_with_radius: np.ndarray,
    lesions: list,
    minimum_radius_mm: float = DEFAULT_DISEASE_SETTINGS["minimum_radius_mm"],
) -> tuple:
    points = np.asarray(centerline_with_radius, dtype=float)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("Expected radius-enabled centerline with shape (N, 4).")
    if not np.all(np.isfinite(points)):
        raise ValueError("Centerline contains NaN or Inf values.")
    if np.any(points[:, 3] <= 0):
        raise ValueError("Input radius must be positive.")
    if minimum_radius_mm <= 0:
        raise ValueError("minimum_radius_mm must be positive.")

    diseased = points.copy()
    baseline_radius = points[:, 3].copy()
    s = arc_length_fraction(points[:, :3])
    combined_multiplier = np.ones(len(points), dtype=float)
    lesion_reports = []

    for lesion in lesions:
        reduction = lesion_reduction_profile(s, lesion)
        multiplier = 1.0 - reduction
        combined_multiplier *= multiplier
        lesion_reports.append({
            **lesion,
            "max_radius_reduction_fraction": float(np.max(reduction)) if len(reduction) else 0.0,
            "affected_point_count": int(np.count_nonzero(reduction > 1e-9)),
        })

    unclamped_radius = baseline_radius * combined_multiplier
    diseased_radius = np.minimum(baseline_radius, np.maximum(unclamped_radius, minimum_radius_mm))
    diseased[:, 3] = diseased_radius

    metadata = {
        "minimum_radius_mm": float(minimum_radius_mm),
        "num_lesions": int(len(lesions)),
        "lesions": lesion_reports,
        "min_baseline_radius_mm": float(np.min(baseline_radius)),
        "min_unclamped_radius_mm": float(np.min(unclamped_radius)),
        "min_diseased_radius_mm": float(np.min(diseased_radius)),
        "max_radius_reduction_fraction": float(np.max(1.0 - diseased_radius / baseline_radius)),
        "clamped_point_count": int(np.count_nonzero(unclamped_radius < minimum_radius_mm)),
    }
    return diseased, metadata


def apply_disease_to_lca_tree(
    centerlines_with_radius: dict,
    config: dict,
    minimum_radius_mm: float = DEFAULT_DISEASE_SETTINGS["minimum_radius_mm"],
) -> tuple:
    config_validation = validate_disease_config(config)
    if not config_validation["is_valid"]:
        raise ValueError("; ".join(config_validation["errors"]))
    normalized_config = config_validation["normalized_config"]

    diseased_tree = {
        branch_name: np.asarray(centerlines_with_radius[branch_name], dtype=float).copy()
        for branch_name in BRANCH_NAMES
    }
    lesions_by_branch = {branch_name: [] for branch_name in BRANCH_NAMES}
    for lesion in normalized_config["lesions"]:
        lesions_by_branch[lesion["branch"]].append(lesion)

    branch_metadata = {}
    for branch_name in BRANCH_NAMES:
        diseased_branch, metadata = apply_disease_to_branch(
            centerlines_with_radius[branch_name],
            lesions_by_branch[branch_name],
            minimum_radius_mm=minimum_radius_mm,
        )
        diseased_tree[branch_name] = diseased_branch
        branch_metadata[branch_name] = metadata

    metadata = {
        "case_id": normalized_config["case_id"],
        "minimum_radius_mm": float(minimum_radius_mm),
        "input_format": ["x", "y", "z", "radius_mm"],
        "radius_only": True,
        "branches": branch_metadata,
        "config": normalized_config,
    }
    return diseased_tree, metadata


def validate_diseased_lca_tree(
    baseline_tree: dict,
    diseased_tree: dict,
    config: dict,
    minimum_radius_mm: float = DEFAULT_DISEASE_SETTINGS["minimum_radius_mm"],
    adjacent_jump_threshold_mm: float = DEFAULT_DISEASE_SETTINGS["adjacent_jump_threshold_mm"],
) -> dict:
    errors = []
    warnings = []
    branch_reports = {}
    config_validation = validate_disease_config(config)
    if not config_validation["is_valid"]:
        errors.extend(config_validation["errors"])

    for branch_name in BRANCH_NAMES:
        if branch_name not in baseline_tree:
            errors.append(f"{branch_name} baseline centerline is missing")
            continue
        if branch_name not in diseased_tree:
            errors.append(f"{branch_name} diseased centerline is missing")
            continue

        baseline = np.asarray(baseline_tree[branch_name], dtype=float)
        diseased = np.asarray(diseased_tree[branch_name], dtype=float)
        branch_errors = []
        branch_warnings = []

        if baseline.ndim != 2 or baseline.shape[1] != 4:
            branch_errors.append("baseline must have shape (N, 4)")
        if diseased.ndim != 2 or diseased.shape[1] != 4:
            branch_errors.append("diseased must have shape (N, 4)")
        if not branch_errors and baseline.shape != diseased.shape:
            branch_errors.append("baseline and diseased arrays must have the same shape")
        if branch_errors:
            errors.extend(f"{branch_name}: {message}" for message in branch_errors)
            branch_reports[branch_name] = {"errors": branch_errors, "warnings": branch_warnings}
            continue

        baseline_radius = baseline[:, 3]
        diseased_radius = diseased[:, 3]
        radius_steps = np.diff(diseased_radius)
        max_jump = float(np.max(np.abs(radius_steps))) if len(radius_steps) else 0.0
        max_reduction = float(np.max(1.0 - diseased_radius / baseline_radius))

        if not np.allclose(baseline[:, :3], diseased[:, :3], rtol=0.0, atol=1e-9):
            branch_errors.append("x, y, z coordinates changed")
        if not np.all(np.isfinite(diseased)):
            branch_errors.append("diseased centerline contains NaN or Inf values")
        if np.any(diseased_radius <= 0):
            branch_errors.append("diseased radius contains non-positive values")
        if np.any(diseased_radius < minimum_radius_mm - 1e-9):
            branch_errors.append("diseased radius is below minimum radius clamp")
        if np.any(diseased_radius - baseline_radius > 1e-9):
            branch_errors.append("disease increased radius above baseline")
        if max_jump > adjacent_jump_threshold_mm:
            branch_warnings.append(
                f"adjacent radius jump {max_jump:.3f} mm exceeds {adjacent_jump_threshold_mm:.3f} mm"
            )

        errors.extend(f"{branch_name}: {message}" for message in branch_errors)
        warnings.extend(f"{branch_name}: {message}" for message in branch_warnings)
        branch_reports[branch_name] = {
            "errors": branch_errors,
            "warnings": branch_warnings,
            "shape": list(diseased.shape),
            "xyz_unchanged": bool(np.allclose(baseline[:, :3], diseased[:, :3], rtol=0.0, atol=1e-9)),
            "baseline_min_radius_mm": float(np.min(baseline_radius)),
            "diseased_min_radius_mm": float(np.min(diseased_radius)),
            "diseased_max_radius_mm": float(np.max(diseased_radius)),
            "max_radius_reduction_fraction": max_reduction,
            "max_adjacent_jump_mm": max_jump,
            "passes_adjacent_jump_check": bool(max_jump <= adjacent_jump_threshold_mm),
        }

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "minimum_radius_mm": float(minimum_radius_mm),
        "adjacent_jump_threshold_mm": float(adjacent_jump_threshold_mm),
        "config_validation": config_validation,
        "branches": branch_reports,
    }
