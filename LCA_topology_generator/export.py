# LCA_topology_generator/export.py

from pathlib import Path
from typing import Dict, List, Any
import json
import numpy as np

from .topology import LCA_TOPOLOGY


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_lca_population(
    selected_branches: List[Dict[str, np.ndarray]],
    selected_metadata: List[Dict[str, Any]],
    validation_reports: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    branch_names = ["LMCA", "LAD", "LCX"]

    for branch in branch_names:
        arr = np.stack([tree[branch] for tree in selected_branches], axis=0)
        np.save(out / f"{branch}_ctrl_points.npy", arr)

    save_json(out / "topology.json", LCA_TOPOLOGY)
    save_json(out / "parameter_log.json", selected_metadata)
    save_json(out / "validation_report.json", validation_reports)