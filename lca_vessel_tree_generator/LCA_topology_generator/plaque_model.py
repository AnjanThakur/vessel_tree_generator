import numpy as np

from .disease_model import BRANCH_NAMES, lesion_reduction_profile
from .radius_model import arc_length_fraction
from .tube_surface import build_lca_tube_surfaces


DEFAULT_PLAQUE_SETTINGS = {
    "minimum_radius_mm": 0.15,
    "description": "Optional MVP plaque layer; modifies tube rings after radius-level disease.",
}


def has_eccentric_plaque(config: dict) -> bool:
    return any(
        str(lesion.get("plaque_mode", "symmetric")).lower() == "eccentric"
        for lesion in config.get("lesions", [])
    )


def _angle_weights(num_circle_points: int, plaque_angle_degrees: float, eccentricity: float) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, num_circle_points, endpoint=False)
    target = np.deg2rad(plaque_angle_degrees)
    one_sided = 0.5 * (1.0 + np.cos(theta - target))
    return (1.0 - eccentricity) + eccentricity * one_sided


def _lesions_by_branch(config: dict) -> dict:
    result = {branch_name: [] for branch_name in BRANCH_NAMES}
    for lesion in config.get("lesions", []):
        result[lesion["branch"]].append(lesion)
    return result


def build_lca_plaque_surfaces(
    baseline_tree: dict,
    config: dict,
    num_circle_points: int = 24,
    minimum_radius_mm: float = DEFAULT_PLAQUE_SETTINGS["minimum_radius_mm"],
) -> tuple:
    """
    Builds optional plaque-deformed tube surfaces from baseline radius centerlines.

    Symmetric plaque applies the same lesion reduction to all ring points.
    Eccentric plaque applies the maximum reduction at `plaque_angle` and less
    reduction on the opposite wall according to `eccentricity`.
    """
    baseline_surfaces, base_metadata = build_lca_tube_surfaces(
        baseline_tree,
        num_circle_points=num_circle_points,
    )
    lesions_by_branch = _lesions_by_branch(config)
    plaque_surfaces = {}
    branch_metadata = {}

    for branch_name in BRANCH_NAMES:
        baseline = np.asarray(baseline_tree[branch_name], dtype=float)
        baseline_surface = np.asarray(baseline_surfaces[branch_name], dtype=float)
        centers = baseline[:, :3]
        baseline_radius = baseline[:, 3]
        s = arc_length_fraction(centers)
        ring_count = baseline_surface.shape[1]
        combined_multiplier = np.ones((len(baseline), ring_count), dtype=float)
        lesion_reports = []

        for lesion in lesions_by_branch[branch_name]:
            reduction = lesion_reduction_profile(s, lesion)
            plaque_mode = str(lesion.get("plaque_mode", "symmetric")).lower()
            if plaque_mode == "eccentric":
                angular_weights = _angle_weights(
                    ring_count,
                    plaque_angle_degrees=float(lesion.get("plaque_angle", 0.0)),
                    eccentricity=float(lesion.get("eccentricity", 0.75)),
                )
            else:
                angular_weights = np.ones(ring_count, dtype=float)

            lesion_multiplier = 1.0 - reduction[:, None] * angular_weights[None, :]
            combined_multiplier *= lesion_multiplier
            lesion_reports.append({
                **lesion,
                "max_surface_reduction_fraction": float(np.max(1.0 - lesion_multiplier)),
                "min_surface_reduction_fraction": float(np.min(1.0 - lesion_multiplier)),
            })

        target_radius = baseline_radius[:, None] * combined_multiplier
        target_radius = np.minimum(
            baseline_radius[:, None],
            np.maximum(target_radius, minimum_radius_mm),
        )
        scale = target_radius / np.maximum(baseline_radius[:, None], 1e-9)
        plaque_surfaces[branch_name] = centers[:, None, :] + (
            baseline_surface - centers[:, None, :]
        ) * scale[:, :, None]

        branch_metadata[branch_name] = {
            "surface_shape": list(plaque_surfaces[branch_name].shape),
            "num_lesions": int(len(lesion_reports)),
            "lesions": lesion_reports,
            "min_surface_radius_mm": float(np.min(target_radius)),
            "max_surface_radius_mm": float(np.max(target_radius)),
            "has_eccentric_plaque": any(
                str(lesion.get("plaque_mode", "symmetric")).lower() == "eccentric"
                for lesion in lesions_by_branch[branch_name]
            ),
        }

    metadata = {
        "method": "MVP optional plaque surface deformation from baseline tube rings",
        "input_layer": "radius-level disease config",
        "surface_shape": base_metadata["surface_shape"],
        "num_circle_points": int(num_circle_points),
        "minimum_radius_mm": float(minimum_radius_mm),
        "branches": branch_metadata,
        "limitations": [
            "Plaque modifies tube surface rings only.",
            "It does not change centerline coordinates.",
            "It does not model plaque material properties or blood flow.",
            "Eccentric plaque angle is defined in the generated tube ring frame.",
        ],
    }
    return plaque_surfaces, metadata


def validate_plaque_surfaces(plaque_surfaces: dict, baseline_tree: dict, metadata: dict) -> dict:
    errors = []
    warnings = []
    branch_reports = {}

    for branch_name in BRANCH_NAMES:
        if branch_name not in plaque_surfaces:
            errors.append(f"{branch_name} plaque surface is missing")
            continue
        if branch_name not in baseline_tree:
            errors.append(f"{branch_name} baseline centerline is missing")
            continue

        surface = np.asarray(plaque_surfaces[branch_name], dtype=float)
        baseline = np.asarray(baseline_tree[branch_name], dtype=float)
        branch_errors = []

        if surface.ndim != 3 or surface.shape[2] != 3:
            branch_errors.append("surface must have shape (N, ring_points, 3)")
        elif surface.shape[0] != len(baseline):
            branch_errors.append("surface length must match baseline centerline length")
        if not np.all(np.isfinite(surface)):
            branch_errors.append("surface contains NaN or Inf values")
        if surface.ndim == 3 and surface.shape[1] < 6:
            warnings.append(f"{branch_name}: plaque surface has very few ring points")

        errors.extend(f"{branch_name}: {message}" for message in branch_errors)
        branch_reports[branch_name] = {
            "shape": list(surface.shape),
            "errors": branch_errors,
            "is_finite": bool(np.all(np.isfinite(surface))),
            "matches_centerline_length": bool(surface.ndim == 3 and surface.shape[0] == len(baseline)),
            "has_eccentric_plaque": metadata.get("branches", {}).get(branch_name, {}).get(
                "has_eccentric_plaque",
                False,
            ),
        }

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "branches": branch_reports,
        "metadata": metadata,
    }
