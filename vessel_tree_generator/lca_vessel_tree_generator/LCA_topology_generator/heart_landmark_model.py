import numpy as np

from .tortuosity import calculate_path_length


HEART_FRAME = {
    "units": "mm",
    "ostium": [0.0, 0.0, 0.0],
    "base_to_apex_axis": [0.0, 0.0, -1.0],
    "coronary_plane_normal": [0.0, 0.0, 1.0],
    "lmca_axis": [1.0, 0.0, 0.0],
    "lcx_crown_axis": [0.0, 1.0, 0.0],
    "description": "MVP heart landmark frame: Z is base-to-apex, X/Y is the coronary/crown plane.",
}


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        return np.zeros_like(vector)
    return vector / norm


def _rescale_path_from_anchor(points: np.ndarray, target_length: float, anchor_index: int = 0) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    current_length = calculate_path_length(points)
    if current_length <= 1e-9:
        return points.copy()
    anchor = points[anchor_index].copy()
    return anchor + (points - anchor) * (target_length / current_length)


def _polyline_distance(points: np.ndarray, guide_points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    guide_points = np.asarray(guide_points, dtype=float)
    if len(points) == 0 or len(guide_points) < 2:
        return np.zeros(len(points), dtype=float)

    distances = []
    for point in points:
        min_distance = np.inf
        for start, end in zip(guide_points[:-1], guide_points[1:]):
            segment = end - start
            length_squared = float(np.dot(segment, segment))
            if length_squared <= 1e-12:
                candidate = np.linalg.norm(point - start)
            else:
                t = np.clip(np.dot(point - start, segment) / length_squared, 0.0, 1.0)
                projection = start + t * segment
                candidate = np.linalg.norm(point - projection)
            min_distance = min(min_distance, candidate)
        distances.append(float(min_distance))
    return np.asarray(distances, dtype=float)


def build_heart_landmarks(lmca_length_mm: float, lad_length_mm: float, lcx_length_mm: float) -> dict:
    ostium = np.zeros(3, dtype=float)
    bifurcation = np.array([float(lmca_length_mm), 0.0, 0.0], dtype=float)
    apex = bifurcation + np.array([
        0.22 * float(lad_length_mm),
        -0.04 * float(lad_length_mm),
        -0.98 * float(lad_length_mm),
    ])
    coronary_groove_center = bifurcation + np.array([
        -0.42 * float(lcx_length_mm),
        0.45 * float(lcx_length_mm),
        0.0,
    ])
    base_plane_z = float(bifurcation[2])
    return {
        "frame": HEART_FRAME,
        "ostium": ostium,
        "lmca_bifurcation": bifurcation,
        "apex": apex,
        "base_plane_z": base_plane_z,
        "coronary_plane_normal": np.array(HEART_FRAME["coronary_plane_normal"], dtype=float),
        "coronary_groove_center": coronary_groove_center,
    }


def lmca_short_trunk_guide(length_mm: float, num_points: int = 5) -> np.ndarray:
    t = np.linspace(0.0, 1.0, num_points)
    points = np.column_stack([
        float(length_mm) * t,
        0.05 * float(length_mm) * np.sin(np.pi * t),
        0.018 * float(length_mm) * np.sin(2.0 * np.pi * t),
    ])
    return _rescale_path_from_anchor(points, float(length_mm), anchor_index=0)


def lad_interventricular_groove_guide(
    bifurcation: np.ndarray,
    length_mm: float,
    num_points: int = 12,
) -> np.ndarray:
    """
    MVP LAD guide: starts at bifurcation and follows an apex-directed
    anterior interventricular groove curve.
    """
    bifurcation = np.asarray(bifurcation, dtype=float)
    t = np.linspace(0.0, 1.0, num_points)
    anterior_sweep = 0.20 * float(length_mm) * t + 0.06 * float(length_mm) * np.sin(np.pi * t)
    septal_offset = -0.05 * float(length_mm) * np.sin(np.pi * t)
    apex_descent = -0.96 * float(length_mm) * t + 0.035 * float(length_mm) * np.sin(2.0 * np.pi * t)
    points = bifurcation + np.column_stack([anterior_sweep, septal_offset, apex_descent])
    return _rescale_path_from_anchor(points, float(length_mm), anchor_index=0)


def lcx_atrioventricular_groove_guide(
    bifurcation: np.ndarray,
    length_mm: float,
    num_points: int = 10,
) -> np.ndarray:
    """
    MVP LCX guide: starts at bifurcation and follows the left AV/coronary
    groove as a crown-plane arc with limited vertical excursion.
    """
    bifurcation = np.asarray(bifurcation, dtype=float)
    t = np.linspace(0.0, 1.0, num_points)
    theta = 0.92 * np.pi * t
    radius = 0.56 * float(length_mm)
    x = -0.28 * float(length_mm) * (1.0 - np.cos(theta))
    y = radius * np.sin(0.62 * theta) + 0.18 * float(length_mm) * t
    z = -0.10 * float(length_mm) * np.sin(np.pi * t)
    points = bifurcation + np.column_stack([x, y, z])
    return _rescale_path_from_anchor(points, float(length_mm), anchor_index=0)


def generate_heart_guided_lca_tree(lengths: dict) -> tuple:
    lmca = lmca_short_trunk_guide(lengths["LMCA"], 5)
    bifurcation = lmca[-1].copy()
    lad = lad_interventricular_groove_guide(bifurcation, lengths["LAD"], 12)
    lcx = lcx_atrioventricular_groove_guide(bifurcation, lengths["LCX"], 10)
    landmarks = build_heart_landmarks(lengths["LMCA"], lengths["LAD"], lengths["LCX"])
    landmarks["lmca_bifurcation"] = bifurcation
    landmarks["apex"] = lad[-1].copy()
    return np.vstack([lmca, lad, lcx]), landmarks


def _as_list_dict(data: dict) -> dict:
    result = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, dict):
            result[key] = _as_list_dict(value)
        else:
            result[key] = value
    return result


def serialize_landmarks(landmarks: dict) -> dict:
    return _as_list_dict(landmarks)


def evaluate_heart_guided_lca(control_tree: np.ndarray) -> dict:
    control_tree = np.asarray(control_tree, dtype=float)
    lmca = control_tree[0:5]
    lad = control_tree[5:17]
    lcx = control_tree[17:27]
    lengths = {
        "LMCA": float(calculate_path_length(lmca)),
        "LAD": float(calculate_path_length(lad)),
        "LCX": float(calculate_path_length(lcx)),
    }
    bifurcation = lmca[-1].copy()
    lad_guide = lad_interventricular_groove_guide(bifurcation, lengths["LAD"], 80)
    lcx_guide = lcx_atrioventricular_groove_guide(bifurcation, lengths["LCX"], 80)

    lad_distances = _polyline_distance(lad, lad_guide)
    lcx_distances = _polyline_distance(lcx, lcx_guide)
    apex_axis = _unit(np.asarray(HEART_FRAME["base_to_apex_axis"], dtype=float))
    coronary_normal = _unit(np.asarray(HEART_FRAME["coronary_plane_normal"], dtype=float))
    lad_vector = lad[-1] - lad[0]
    lcx_vector = lcx[-1] - lcx[0]
    lad_axis_alignment = float(max(np.dot(_unit(lad_vector), apex_axis), 0.0))
    lcx_plane_alignment = float(np.linalg.norm(lcx_vector - np.dot(lcx_vector, coronary_normal) * coronary_normal) / max(np.linalg.norm(lcx_vector), 1e-9))
    lcx_plane_offsets = np.abs((lcx - bifurcation) @ coronary_normal)

    return {
        "heart_frame": HEART_FRAME,
        "landmarks": serialize_landmarks(build_heart_landmarks(lengths["LMCA"], lengths["LAD"], lengths["LCX"])),
        "guide_lengths_mm": lengths,
        "lad_guide_distance_mean_mm": float(np.mean(lad_distances)),
        "lad_guide_distance_max_mm": float(np.max(lad_distances)),
        "lad_guide_distance_mean_ratio": float(np.mean(lad_distances) / max(lengths["LAD"], 1e-9)),
        "lad_apex_axis_alignment": lad_axis_alignment,
        "lcx_guide_distance_mean_mm": float(np.mean(lcx_distances)),
        "lcx_guide_distance_max_mm": float(np.max(lcx_distances)),
        "lcx_guide_distance_mean_ratio": float(np.mean(lcx_distances) / max(lengths["LCX"], 1e-9)),
        "lcx_coronary_plane_alignment": lcx_plane_alignment,
        "lcx_coronary_plane_offset_mean_ratio": float(np.mean(lcx_plane_offsets) / max(lengths["LCX"], 1e-9)),
        "lcx_coronary_plane_offset_max_ratio": float(np.max(lcx_plane_offsets) / max(lengths["LCX"], 1e-9)),
    }
