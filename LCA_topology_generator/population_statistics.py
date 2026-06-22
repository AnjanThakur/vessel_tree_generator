# LCA_topology_generator/population_statistics.py

from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import numpy as np

from .export import save_json


BRANCH_NAMES = ("LMCA", "LAD", "LCX")


def load_control_point_matrices(
    output_dir: Union[str, Path],
    branch_names: Iterable[str] = BRANCH_NAMES,
) -> Dict[str, np.ndarray]:
    out = Path(output_dir)
    return {
        branch: np.load(out / f"{branch}_ctrl_points.npy")
        for branch in branch_names
    }


def _validate_matrix(branch: str, matrix: np.ndarray) -> None:
    if matrix.ndim != 3:
        raise ValueError(
            f"{branch} matrix must be 3D with shape (K, P, 3); got {matrix.shape}"
        )
    if matrix.shape[2] != 3:
        raise ValueError(
            f"{branch} matrix coordinate dimension must be 3; got {matrix.shape}"
        )


def compute_population_statistics(
    matrices: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, np.ndarray]]:
    statistics = {}

    for branch, matrix in matrices.items():
        _validate_matrix(branch, matrix)
        statistics[branch] = {
            "mean": np.mean(matrix, axis=0),
            "std": np.std(matrix, axis=0),
        }

    return statistics


def export_population_statistics(
    output_dir: Union[str, Path],
    matrices: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if matrices is None:
        matrices = load_control_point_matrices(out)

    statistics = compute_population_statistics(matrices)

    branch_summaries = []
    for branch in BRANCH_NAMES:
        matrix = matrices[branch]
        mean = statistics[branch]["mean"]
        std = statistics[branch]["std"]

        np.save(out / f"{branch}_mean.npy", mean)
        np.save(out / f"{branch}_std.npy", std)

        branch_summaries.append({
            "branch_name": branch,
            "original_matrix_shape": list(matrix.shape),
            "mean_shape": list(mean.shape),
            "std_shape": list(std.shape),
            "coordinate_unit": "mm",
            "K_count": int(matrix.shape[0]),
            "P_count": int(matrix.shape[1]),
            "summary": (
                f"{branch} mean/std are computed coordinate-wise across "
                f"{matrix.shape[0]} validated templates."
            ),
        })

    save_json(
        out / "population_statistics.json",
        {
            "coordinate_unit": "mm",
            "branch_count": len(branch_summaries),
            "branches": branch_summaries,
        },
    )

    return statistics
