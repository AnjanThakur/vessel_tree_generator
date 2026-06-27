# LCA_topology_generator/angle_distribution.py

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .export import save_json
from .parameters import ANGLE_TOLERANCE_DEG


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = v1 / max(np.linalg.norm(v1), 1e-12)
    v2 = v2 / max(np.linalg.norm(v2), 1e-12)
    value = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(value)))


def compute_lad_lcx_angle(lad_cp: np.ndarray, lcx_cp: np.ndarray) -> float:
    lad_tangent = lad_cp[3] - lad_cp[0]
    lcx_tangent = lcx_cp[3] - lcx_cp[0]
    return angle_between_vectors(lad_tangent, lcx_tangent)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_metadata(output_dir: Path) -> List[Dict[str, Any]]:
    path = output_dir / "parameter_log.json"
    if not path.exists():
        return []
    return _load_json(path)


def _load_validation_reports(output_dir: Path) -> List[Dict[str, Any]]:
    path = output_dir / "validation_report.json"
    if not path.exists():
        return []
    return _load_json(path)


def _target_angles_from_metadata(
    metadata: Iterable[Dict[str, Any]],
) -> List[Optional[float]]:
    targets = []
    for item in metadata:
        value = item.get("lad_lcx_angle_target_deg")
        targets.append(None if value is None else float(value))
    return targets


def _validation_counts(
    validation_reports: Iterable[Dict[str, Any]],
) -> Dict[str, Optional[int]]:
    reports = list(validation_reports)
    if not reports:
        return {
            "pass_count": None,
            "fail_count": None,
        }

    pass_count = sum(1 for report in reports if report.get("passed") is True)
    fail_count = len(reports) - pass_count
    return {
        "pass_count": int(pass_count),
        "fail_count": int(fail_count),
    }


def compute_angle_distribution(
    lad_matrix: np.ndarray,
    lcx_matrix: np.ndarray,
    metadata: Optional[List[Dict[str, Any]]] = None,
    validation_reports: Optional[List[Dict[str, Any]]] = None,
    tolerance_deg: float = ANGLE_TOLERANCE_DEG,
) -> Dict[str, Any]:
    if lad_matrix.shape[0] != lcx_matrix.shape[0]:
        raise ValueError(
            "LAD and LCX matrices must have matching population count (K); "
            f"got {lad_matrix.shape[0]} and {lcx_matrix.shape[0]}"
        )
    if (
        lad_matrix.ndim != 3
        or lad_matrix.shape[2] != 3
        or lcx_matrix.ndim != 3
        or lcx_matrix.shape[2] != 3
    ):
        raise ValueError(
            "Expected LAD/LCX matrices with shape (K, P, 3); "
            f"got {lad_matrix.shape} and {lcx_matrix.shape}"
        )

    actual_angles = [
        compute_lad_lcx_angle(lad_matrix[i], lcx_matrix[i])
        for i in range(lad_matrix.shape[0])
    ]
    actual = np.array(actual_angles, dtype=float)

    target_angles = _target_angles_from_metadata(metadata or [])
    validation = _validation_counts(validation_reports or [])

    return {
        "angle_unit": "degree",
        "coordinate_unit": "mm",
        "count": int(len(actual_angles)),
        "tangent_definition": {
            "LAD": "LAD_cp[3] - LAD_cp[0]",
            "LCX": "LCX_cp[3] - LCX_cp[0]",
        },
        "target_angles_deg": target_angles,
        "actual_generated_angles_deg": [float(value) for value in actual_angles],
        "actual_mean_deg": float(np.mean(actual)),
        "actual_std_deg": float(np.std(actual)),
        "actual_min_deg": float(np.min(actual)),
        "actual_max_deg": float(np.max(actual)),
        "tolerance_used_deg": float(tolerance_deg),
        "validation": validation,
    }


def plot_angle_distribution(
    report: Dict[str, Any],
    output_path: Union[str, Path],
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    actual = np.array(report["actual_generated_angles_deg"], dtype=float)
    targets = [
        value
        for value in report.get("target_angles_deg", [])
        if value is not None
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = min(12, max(4, int(np.sqrt(len(actual)))))
    ax.hist(actual, bins=bins, alpha=0.75, edgecolor="black", label="Actual")
    ax.axvline(report["actual_mean_deg"], color="black", linestyle="--", label="Actual mean")

    if targets:
        target_mean = float(np.mean(np.array(targets, dtype=float)))
        ax.axvline(target_mean, color="tab:orange", linestyle=":", label="Target mean")

    ax.set_title("LAD-LCX Angle Distribution")
    ax.set_xlabel("Angle near bifurcation (degrees)")
    ax.set_ylabel("Template count")
    ax.grid(True, alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close(fig)


def export_angle_distribution(
    output_dir: Union[str, Path],
    lad_matrix: Optional[np.ndarray] = None,
    lcx_matrix: Optional[np.ndarray] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
    validation_reports: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if lad_matrix is None:
        lad_matrix = np.load(out / "LAD_ctrl_points.npy")
    if lcx_matrix is None:
        lcx_matrix = np.load(out / "LCX_ctrl_points.npy")
    if metadata is None:
        metadata = _load_metadata(out)
    if validation_reports is None:
        validation_reports = _load_validation_reports(out)

    report = compute_angle_distribution(
        lad_matrix=lad_matrix,
        lcx_matrix=lcx_matrix,
        metadata=metadata,
        validation_reports=validation_reports,
    )

    save_json(out / "angle_distribution.json", report)
    plot_angle_distribution(report, out / "angle_distribution_plot.png")

    return report
