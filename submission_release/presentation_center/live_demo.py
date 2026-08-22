"""Safe live-generation adapter for the presentation center runtime directory."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from vessel_tree_generator import CoronaryTreeGenerator, stenosis_config
from vessel_tree_generator.api import GenerationConfig, MotionConfig
from vessel_tree_generator.disease import healthy_config
from vessel_tree_generator.export import export_case


CENTER = Path(__file__).resolve().parent
RUNTIME = CENTER / "runtime_demo"
OUTPUT = RUNTIME / "generated_case"


def _bounded(value: Any, low: float, high: float, name: str) -> float:
    number = float(value)
    if not low <= number <= high:
        raise ValueError(f"{name} must lie in [{low}, {high}]")
    return number


def generate_runtime_case(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 20260822))
    scale = _bounded(config.get("pca_scale", 0.04), 0.0, 0.20, "pca_scale")
    mode = str(config.get("disease_mode", "healthy")).lower()
    branch = str(config.get("disease_branch", "LAD")).upper()
    if mode not in {"healthy", "focal", "diffuse", "tandem"}:
        raise ValueError("unsupported disease mode")
    if branch not in {"LMCA", "LAD", "LCX"}:
        raise ValueError("unsupported disease branch")
    position = _bounded(config.get("position", 0.45), 0.05, 0.95, "position")
    length = _bounded(config.get("length", 0.12), 0.02, 0.80, "length")
    severity = _bounded(config.get("severity", 0.65), 0.0, 0.90, "severity")
    heart_rate = _bounded(config.get("heart_rate", 60.0), 35.0, 180.0, "heart_rate")

    disease = (
        healthy_config("presentation_runtime_healthy")
        if mode == "healthy"
        else stenosis_config(
            branch,
            position,
            length,
            severity,
            mode,
            case_id="presentation_runtime_case",
            tandem_positions=(max(0.08, position - 0.18), min(0.92, position + 0.18)) if mode == "tandem" else None,
        )
    )
    generation = GenerationConfig(
        seed=seed,
        pca_scale=scale,
        maximum_attempts=250,
        heart_rate_bpm=heart_rate,
    )
    motion = MotionConfig(number_of_phases=10)
    generator = CoronaryTreeGenerator()
    case = generator.generate_case(disease, generation=generation, motion=motion)

    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    manifest = export_case(case, OUTPUT, points_per_branch=50, clean=False)
    cine = np.load(OUTPUT / "geometry_cine.npy", mmap_mode="r", allow_pickle=False)
    graph = json.loads((OUTPUT / "graph.json").read_text(encoding="utf-8"))
    result = {
        "status": manifest["status"],
        "output": str(OUTPUT.resolve()),
        "configuration": {
            "generation": asdict(generation),
            "motion": asdict(motion),
            "disease": disease,
        },
        "shape": list(cine.shape),
        "branch_order": ["LMCA", "LAD", "LCX"],
        "topology": graph,
        "preview": "runtime_demo/generated_case/preview.png",
        "pvd": str((OUTPUT / "vtk" / "cine.pvd").resolve()),
    }
    (RUNTIME / "presentation_result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result
