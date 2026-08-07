import numpy as np


def calculate_path_length(points: np.ndarray) -> float:
    """
    Returns the cumulative arc length of a 3D polyline.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0.0

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return float(np.sum(segment_lengths))


def calculate_chord_length(points: np.ndarray) -> float:
    """
    Returns the straight-line distance between the first and last point.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0.0

    return float(np.linalg.norm(points[-1] - points[0]))


def calculate_tortuosity(points: np.ndarray) -> dict:
    """
    Calculates distance-ratio tortuosity for a branch.

    A perfectly straight branch has tortuosity close to 1.0. Higher values
    indicate a longer path relative to the endpoint-to-endpoint distance.
    """
    path_length = calculate_path_length(points)
    chord_length = calculate_chord_length(points)
    ratio = path_length / chord_length if chord_length > 1e-9 else None

    return {
        "path_length": path_length,
        "chord_length": chord_length,
        "tortuosity": float(ratio) if ratio is not None else None,
    }


def calculate_lca_tortuosity(centerlines: dict) -> dict:
    """
    Calculates tortuosity metrics for each LCA branch in a centerline dict.
    """
    return {
        branch_name: calculate_tortuosity(points)
        for branch_name, points in centerlines.items()
    }


def summarize_lca_population_tortuosity(tree_ctrl_points: np.ndarray) -> dict:
    """
    Calculates per-patient control-point tortuosity summaries for LCA trees.

    Expected tree shape is (M, 27, 3), with fixed topology:
    LMCA 0:5, LAD 5:17, LCX 17:27.
    """
    tree_ctrl_points = np.asarray(tree_ctrl_points, dtype=float)
    if tree_ctrl_points.ndim == 2:
        tree_ctrl_points = tree_ctrl_points[np.newaxis, ...]

    branch_slices = {
        "LMCA": slice(0, 5),
        "LAD": slice(5, 17),
        "LCX": slice(17, 27),
    }

    summary = {}
    for branch_name, branch_slice in branch_slices.items():
        values = []
        undefined_count = 0
        for patient_tree in tree_ctrl_points:
            metric = calculate_tortuosity(patient_tree[branch_slice])
            if metric["tortuosity"] is not None:
                values.append(metric["tortuosity"])
            else:
                undefined_count += 1

        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            summary[branch_name] = {
                "valid_count": 0,
                "undefined_count": int(undefined_count),
                "mean_tortuosity": None,
                "std_tortuosity": None,
                "min_tortuosity": None,
                "max_tortuosity": None,
            }
            continue

        summary[branch_name] = {
            "valid_count": int(len(values)),
            "undefined_count": int(undefined_count),
            "mean_tortuosity": float(np.mean(values)),
            "std_tortuosity": float(np.std(values)),
            "min_tortuosity": float(np.min(values)),
            "max_tortuosity": float(np.max(values)),
        }

    return summary
