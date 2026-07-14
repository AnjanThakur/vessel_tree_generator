import numpy as np

from .tortuosity import calculate_tortuosity


def resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    distances = np.zeros(len(points))
    distances[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    if distances[-1] <= 1e-9:
        return np.tile(points[0], (n, 1))

    target = np.linspace(0.0, distances[-1], n)
    result = np.zeros((n, points.shape[1]), dtype=float)
    for dim in range(points.shape[1]):
        result[:, dim] = np.interp(target, distances, points[:, dim])
    return result


def _branch_frame(points: np.ndarray):
    chord = points[-1] - points[0]
    chord_norm = np.linalg.norm(chord)
    if chord_norm <= 1e-9:
        return None, None, None

    tangent = chord / chord_norm
    centered = points - np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)

    normal = None
    for candidate in vh:
        candidate = candidate - np.dot(candidate, tangent) * tangent
        candidate_norm = np.linalg.norm(candidate)
        if candidate_norm > 1e-9:
            normal = candidate / candidate_norm
            break

    if normal is None:
        axis = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(axis, tangent)) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        normal = np.cross(tangent, axis)
        normal /= np.linalg.norm(normal)

    binormal = np.cross(tangent, normal)
    binormal /= np.linalg.norm(binormal)
    return tangent, normal, binormal


def add_wave_tortuosity(
    points: np.ndarray,
    amplitude: float,
    cycles: float = 2.0,
    phase: float = 0.0,
    secondary_scale: float = 0.35,
) -> np.ndarray:
    """
    Adds endpoint-preserving sinusoidal curvature to a branch.

    The original endpoints remain unchanged, while the interior points are
    displaced in a stable local normal/binormal frame.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 3 or amplitude <= 0:
        return points.copy()

    _, normal, binormal = _branch_frame(points)
    if normal is None:
        return points.copy()

    t = np.linspace(0.0, 1.0, len(points))
    envelope = np.sin(np.pi * t)
    primary_wave = np.sin(2.0 * np.pi * cycles * t + phase)
    secondary_wave = np.sin(2.0 * np.pi * (cycles * 0.5 + 0.5) * t + phase * 0.7)
    displacement = (
        amplitude * envelope * primary_wave
    )[:, None] * normal + (
        amplitude * secondary_scale * envelope * secondary_wave
    )[:, None] * binormal

    result = points + displacement
    result[0] = points[0]
    result[-1] = points[-1]
    return result


def make_target_tortuosity_variant(
    points: np.ndarray,
    target_tortuosity: float,
    cycles: float = 2.0,
    phase: float = 0.0,
    sample_points: int = 260,
    max_amplitude_fraction: float = 0.65,
) -> tuple:
    """
    Builds a branch variant that approaches a requested distance-ratio tortuosity.

    Returns `(variant_points, metadata)`. If the requested target is lower than
    the current tortuosity, the original branch is returned unchanged.
    """
    dense_points = resample_polyline(points, sample_points)
    base_metric = calculate_tortuosity(dense_points)
    base_tortuosity = base_metric["tortuosity"]
    chord_length = base_metric["chord_length"]

    if base_tortuosity is None or chord_length <= 1e-9:
        return dense_points, {
            "target_tortuosity": target_tortuosity,
            "actual_tortuosity": None,
            "amplitude_mm": 0.0,
            "status": "undefined_base_tortuosity",
        }

    if target_tortuosity <= base_tortuosity:
        return dense_points, {
            "target_tortuosity": target_tortuosity,
            "actual_tortuosity": base_tortuosity,
            "amplitude_mm": 0.0,
            "status": "target_not_above_base",
        }

    low = 0.0
    high = chord_length * max_amplitude_fraction
    best_points = dense_points.copy()
    best_metric = base_metric
    best_amplitude = 0.0

    for _ in range(32):
        mid = 0.5 * (low + high)
        candidate = add_wave_tortuosity(dense_points, mid, cycles=cycles, phase=phase)
        metric = calculate_tortuosity(candidate)
        actual = metric["tortuosity"]
        if actual is None:
            break

        best_points = candidate
        best_metric = metric
        best_amplitude = mid

        if actual < target_tortuosity:
            low = mid
        else:
            high = mid

    status = "target_reached"
    if best_metric["tortuosity"] is not None and best_metric["tortuosity"] < target_tortuosity * 0.98:
        status = "target_limited_by_amplitude_cap"

    return best_points, {
        "target_tortuosity": float(target_tortuosity),
        "actual_tortuosity": best_metric["tortuosity"],
        "amplitude_mm": float(best_amplitude),
        "cycles": float(cycles),
        "status": status,
    }


def make_tortuosity_ladder(points: np.ndarray, targets=(1.05, 1.25, 1.55, 1.9)) -> dict:
    """
    Creates original/low/medium/high/very-high branch variants.
    """
    labels = ["low", "medium", "high", "very_high"]
    variants = {
        "original": {
            "points": resample_polyline(points, 260),
            "metadata": {
                "target_tortuosity": None,
                "actual_tortuosity": calculate_tortuosity(resample_polyline(points, 260))["tortuosity"],
                "amplitude_mm": 0.0,
                "status": "original",
            },
        }
    }

    for index, (label, target) in enumerate(zip(labels, targets)):
        variant_points, metadata = make_target_tortuosity_variant(
            points,
            target_tortuosity=target,
            cycles=1.4 + 0.45 * index,
            phase=0.45 * index,
        )
        variants[label] = {
            "points": variant_points,
            "metadata": metadata,
        }

    return variants
