"""Public API for reproducible, disease-aware 4D LCA tree generation."""

from __future__ import annotations

import json
import sys
import tempfile
from argparse import Namespace
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from lca_vessel_tree_generator.LCA_topology_generator.radius_model import (
    build_lca_radius_tree,
    validate_lca_radius_tree,
)
from pca_ssm_vessel_tree_generator.motion.cardiac_motion import apply_cardiac_motion_to_tree

from .disease import apply_disease_config, healthy_config
from .pulsatility import phase_radius, validate_pulsatility
from .validation import BRANCH_ORDER, validate_4d_case


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATISTICS_DIR = PROJECT_ROOT / "outputs/lca_ssm/lca_population_model/generator_statistics"


@dataclass(frozen=True)
class GenerationConfig:
    """Static statistical-generation settings."""

    seed: int = 20260822
    source_case_id: str | None = None
    pca_scale: float = 0.04
    maximum_attempts: int = 250
    heart_rate_bpm: float = 60.0

    def validate(self) -> None:
        if self.maximum_attempts < 1:
            raise ValueError("maximum_attempts must be at least one")
        if not np.isfinite(self.pca_scale) or not 0.0 <= self.pca_scale <= 1.0:
            raise ValueError("pca_scale must be finite and lie in [0, 1]")
        if not np.isfinite(self.heart_rate_bpm) or self.heart_rate_bpm <= 0.0:
            raise ValueError("heart_rate_bpm must be finite and positive")


@dataclass(frozen=True)
class MotionConfig:
    """Ellipsoid-relative cardiac motion settings."""

    number_of_phases: int = 10
    radial_amplitude: float = 0.14
    longitudinal_amplitude: float = 0.10
    torsion_amplitude_deg: float = 10.0
    peak_phase: float = 0.35

    def validate(self) -> None:
        if self.number_of_phases < 1:
            raise ValueError("number_of_phases must be at least one")
        if not 0.0 <= self.radial_amplitude <= 0.30:
            raise ValueError("radial_amplitude must lie in [0, 0.30]")
        if not 0.0 <= self.longitudinal_amplitude <= 0.30:
            raise ValueError("longitudinal_amplitude must lie in [0, 0.30]")
        if not 0.0 <= self.torsion_amplitude_deg <= 30.0:
            raise ValueError("torsion_amplitude_deg must lie in [0, 30]")
        if not 0.0 < self.peak_phase < 1.0:
            raise ValueError("peak_phase must lie in (0, 1)")


@dataclass(frozen=True)
class PulsatilityConfig:
    """Cyclic radius and lesion-compliance settings."""

    amplitude: float = 0.03
    stenosis_compliance_factor: float = 0.35

    def validate(self) -> None:
        validate_pulsatility(self.amplitude, self.stenosis_compliance_factor)


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _phase_array(number_of_phases: int, values: Sequence[float] | None) -> np.ndarray:
    phases = (
        # Include the repeated end-diastolic endpoint so exported animations
        # form an exactly closed cardiac cycle.
        np.linspace(0.0, 1.0, number_of_phases, endpoint=True, dtype=float)
        if values is None
        else np.asarray(tuple(values), dtype=float)
    )
    if phases.ndim != 1 or not len(phases):
        raise ValueError("phase_values must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(phases)) or np.any(phases < 0.0) or np.any(phases > 1.0):
        raise ValueError("phase_values must be finite and lie in [0, 1]")
    if np.any(np.diff(phases) < 0.0):
        raise ValueError("phase_values must be monotonically non-decreasing")
    return phases


class CoronaryTreeGenerator:
    """Generate validated 4D LCA trees with explicit, controlled disease.

    The population-derived statistical scope is LMCA/LAD/LCX. The class does
    not silently fabricate a population-derived RCA.
    """

    def __init__(self, statistics_dir: str | Path = DEFAULT_STATISTICS_DIR) -> None:
        self.statistics_dir = Path(statistics_dir).resolve()
        required = (
            "branch_assignment_gate.json",
            "fixed_branch_surface_coordinates.npz",
            "landmark_stats.json",
            "population_ellipsoid_parameters.csv",
            "population_surface_statistics.json",
            "population_validation_thresholds.json",
            "surface_deviation_pca.npz",
        )
        missing = [name for name in required if not (self.statistics_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"incomplete frozen LCA statistics package at {self.statistics_dir}: " + ", ".join(missing)
            )

    @staticmethod
    def _demo_runner():
        # The established generator keeps historical top-level imports such as
        # ``generation.*``. Add its own directory only for this lazy import.
        pca_root = PROJECT_ROOT / "pca_ssm_vessel_tree_generator"
        if str(pca_root) not in sys.path:
            sys.path.insert(0, str(pca_root))
        from pca_ssm_vessel_tree_generator.run_person2_week1_demo import run

        return run

    def sample_reference(self, config: GenerationConfig | None = None) -> dict[str, Any]:
        """Sample and validate one static population-derived LCA reference."""
        config = config or GenerationConfig()
        config.validate()
        runner = self._demo_runner()
        with tempfile.TemporaryDirectory(prefix="coronary4d_reference_") as temporary:
            output = Path(temporary) / "tree"
            runner(Namespace(
                output_dir=output,
                seed=int(config.seed),
                clean=True,
                include_rca=False,
                stats_dir=self.statistics_dir,
                landmark_stats=None,
                pca=None,
                thresholds=None,
                max_attempts=int(config.maximum_attempts),
                source_case_id=config.source_case_id,
                pca_scale=float(config.pca_scale),
            ))
            branches = {
                name: np.load(output / f"{name}.npy", allow_pickle=False)
                for name in BRANCH_ORDER
            }
            parameters = _json_load(output / "parameters.json")
            static_validation = _json_load(output / "validation.json")
            sampling = _json_load(output / "sampling_attempts.json")

        ellipsoid = parameters["ellipsoid"]
        return {
            "branches": branches,
            "ellipsoid_params": {
                "a_mm": float(ellipsoid["a"]),
                "b_mm": float(ellipsoid["b"]),
                "c_mm": float(ellipsoid["c"]),
            },
            "generation_parameters": parameters,
            "static_validation": static_validation,
            "sampling": {
                "attempt_count": sampling["attempt_count"],
                "accepted_attempt": sampling["accepted_attempt"],
            },
            "configuration": asdict(config),
        }

    def generate_case(
        self,
        disease_config: dict[str, Any] | None = None,
        *,
        generation: GenerationConfig | None = None,
        motion: MotionConfig | None = None,
        pulsatility: PulsatilityConfig | None = None,
        phase_values: Sequence[float] | None = None,
        reference: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a complete validated 4D case.

        Pass a value returned by :meth:`sample_reference` to ``reference`` to
        compare several disease configurations on exactly the same anatomy.
        """
        generation = generation or GenerationConfig()
        motion = motion or MotionConfig()
        pulsatility = pulsatility or PulsatilityConfig()
        generation.validate()
        motion.validate()
        pulsatility.validate()
        phases = _phase_array(motion.number_of_phases, phase_values)
        reference = reference or self.sample_reference(generation)

        centerlines = {
            name: np.asarray(reference["branches"][name], dtype=float).copy()
            for name in BRANCH_ORDER
        }
        healthy_tree, radius_metadata = build_lca_radius_tree(centerlines)
        radius_validation = validate_lca_radius_tree(healthy_tree)
        if not radius_validation["is_valid"]:
            raise RuntimeError("static radius validation failed: " + "; ".join(radius_validation["errors"]))
        diseased_tree, disease_metadata = apply_disease_config(
            healthy_tree,
            disease_config or healthy_config(),
        )

        motion_result = apply_cardiac_motion_to_tree(
            {
                "tree_id": disease_metadata.get("case_id", "coronary_4d_case"),
                "ellipsoid_params": reference["ellipsoid_params"],
                "vessels_3d": centerlines,
                "source_metadata": reference["generation_parameters"],
            },
            num_phases=len(phases),
            radial_amplitude=motion.radial_amplitude,
            longitudinal_amplitude=motion.longitudinal_amplitude,
            torsion_amplitude_deg=motion.torsion_amplitude_deg,
            peak_phase=motion.peak_phase,
            phase_values=phases,
        )

        period_seconds = 60.0 / generation.heart_rate_bpm
        time_seconds = phases * period_seconds
        frames: list[dict[str, Any]] = []
        for frame, time in zip(motion_result["frames"], time_seconds):
            phase = float(frame["phase"])
            branches: dict[str, np.ndarray] = {}
            radius_phase_metadata: dict[str, Any] = {}
            for name in BRANCH_ORDER:
                radius, radius_details = phase_radius(
                    healthy_tree[name][:, 3],
                    diseased_tree[name][:, 3],
                    phase,
                    amplitude=pulsatility.amplitude,
                    stenosis_compliance_factor=pulsatility.stenosis_compliance_factor,
                    peak_phase=motion.peak_phase,
                )
                branches[name] = np.column_stack((frame["vessels_3d"][name], radius))
                radius_phase_metadata[name] = radius_details
            frames.append({
                "phase_index": int(frame["phase_index"]),
                "phase": phase,
                "time_seconds": float(time),
                "contraction_scale": float(frame["contraction_scale_s"]),
                "ellipsoid_params": frame["ellipsoid_params"],
                "branches": branches,
                "radius_phase_metadata": radius_phase_metadata,
            })

        disease_reduction = {
            name: np.clip(1.0 - diseased_tree[name][:, 3] / healthy_tree[name][:, 3], 0.0, 1.0)
            for name in BRANCH_ORDER
        }
        result: dict[str, Any] = {
            "schema_version": "1.0.0",
            "case_id": disease_metadata.get("case_id", "healthy"),
            "model_scope": "population-derived LCA (LMCA, LAD, LCX); RCA not included",
            "branch_order": list(BRANCH_ORDER),
            "hierarchy": {
                "root": "LMCA",
                "nodes": list(BRANCH_ORDER),
                "directed_edges": [["LMCA", "LAD"], ["LMCA", "LCX"]],
                "shared_bifurcation": "LMCA[-1] == LAD[0] == LCX[0]",
            },
            "heart_rate_bpm": float(generation.heart_rate_bpm),
            "cardiac_period_seconds": float(period_seconds),
            "phase_values": phases.tolist(),
            "time_seconds": time_seconds.tolist(),
            "reference": {
                "branches": diseased_tree,
                "healthy_branches": healthy_tree,
                "disease_reduction_fraction": disease_reduction,
                "ellipsoid_params": reference["ellipsoid_params"],
            },
            "frames": frames,
            "disease": disease_metadata,
            "radius_model": {"metadata": radius_metadata, "validation": radius_validation},
            "motion": asdict(motion),
            "pulsatility": asdict(pulsatility),
            "provenance": {
                "generation": asdict(generation),
                "statistics_package": "outputs/lca_ssm/lca_population_model/generator_statistics",
                "generation_parameters": reference["generation_parameters"],
                "static_validation": reference["static_validation"],
                "sampling": reference["sampling"],
                "random_seed": int(generation.seed),
                "clinical_status": "research prototype; not clinically validated",
            },
        }
        result["validation"] = validate_4d_case(result)
        if not result["validation"]["is_valid"]:
            raise RuntimeError("4D case validation failed: " + "; ".join(result["validation"]["errors"]))
        return result

    def generate_tree(
        self,
        phase: float,
        disease_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate one exact requested cardiac phase and its provenance."""
        case = self.generate_case(disease_config, phase_values=[float(phase)], **kwargs)
        return {
            "case_id": case["case_id"],
            "phase": float(phase),
            "time_seconds": case["time_seconds"][0],
            "branches": case["frames"][0]["branches"],
            "hierarchy": case["hierarchy"],
            "validation": case["validation"],
            "case": case,
        }
