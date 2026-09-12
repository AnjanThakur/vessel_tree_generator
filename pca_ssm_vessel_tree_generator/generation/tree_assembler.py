"""Assemble major coronary branches and enforce exact LCA topology."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from generation.deviation_sampler import DeviationSample, ZeroDeviationSampler, interpolate_branch_deviation
from generation.landmark_sampler import SurfaceLandmark
from generation.parameter_sampler import EllipsoidParameters
from generation.surface_path_generator import SurfacePath, SurfacePathGenerator, surface_coordinate_to_point


@dataclass
class SyntheticTree:
    """Generated tree in the common cardiac coordinate frame."""

    ellipsoid: EllipsoidParameters
    landmarks: dict[str, SurfaceLandmark]
    branches: dict[str, np.ndarray]
    surface_paths: dict[str, SurfacePath]
    generation_metadata: dict[str, Any] = field(default_factory=dict)


class TreeAssembler:
    """Generate LMCA/LAD/LCX and optionally RCA with one joint deviation draw."""

    DEFAULT_PATH_CONFIG = {
        "LMCA": {"control_point_count": 4, "sample_count": 60, "tortuosity_strength_rad": 0.008},
        "LAD": {"control_point_count": 7, "sample_count": 180, "tortuosity_strength_rad": 0.018},
        "LCX": {"control_point_count": 7, "sample_count": 160, "tortuosity_strength_rad": 0.018},
        "RCA": {"control_point_count": 8, "sample_count": 190, "tortuosity_strength_rad": 0.018},
    }

    def __init__(
        self,
        ellipsoid: EllipsoidParameters,
        rng: np.random.Generator,
        deviation_sampler: Any | None = None,
        trajectory_sampler: Any | None = None,
        path_config: dict[str, dict[str, Any]] | None = None,
    ):
        self.ellipsoid = ellipsoid
        self.rng = rng
        self.deviation_sampler = deviation_sampler or ZeroDeviationSampler()
        self.trajectory_sampler = trajectory_sampler
        self.path_config = {name: values.copy() for name, values in self.DEFAULT_PATH_CONFIG.items()}
        if path_config:
            for name, values in path_config.items():
                if name not in self.path_config:
                    raise ValueError(f"unsupported branch in path_config: {name}")
                self.path_config[name].update(values)

    @staticmethod
    def _require_landmarks(landmarks: dict[str, SurfaceLandmark], include_rca: bool) -> None:
        required = {"lca_ostium", "bifurcation", "lad_endpoint", "lcx_endpoint"}
        if include_rca:
            required.update({"rca_ostium", "rca_endpoint"})
        missing = sorted(required - set(landmarks))
        if missing:
            raise ValueError(f"cannot assemble tree; missing landmarks: {missing}")

    def assemble(self, landmarks: dict[str, SurfaceLandmark], include_rca: bool = False) -> SyntheticTree:
        self._require_landmarks(landmarks, include_rca)
        path_generator = SurfacePathGenerator(self.ellipsoid, self.rng)
        deviation_sample: DeviationSample = self.deviation_sampler.sample(
            self.rng, source_case_id=self.ellipsoid.source_case_id
        )
        trajectory_sample = None if self.trajectory_sampler is None else self.trajectory_sampler.sample(
            self.rng,
            source_case_id=self.ellipsoid.source_case_id,
            include_rca=include_rca,
        )
        specifications = {
            "LMCA": (landmarks["lca_ostium"], landmarks["bifurcation"]),
            "LAD": (landmarks["bifurcation"], landmarks["lad_endpoint"]),
            "LCX": (landmarks["bifurcation"], landmarks["lcx_endpoint"]),
        }
        if include_rca:
            specifications["RCA"] = (landmarks["rca_ostium"], landmarks["rca_endpoint"])

        paths: dict[str, SurfacePath] = {}
        effective_tortuosity_strength: dict[str, float] = {}
        for name, (start, end) in specifications.items():
            config = self.path_config[name]
            # A matched empirical trajectory already contains the patient's
            # smooth bends. Do not add a second hand-tuned random perturbation
            # on top of that trajectory and its coordinated PCA residual.
            tortuosity_strength = (
                0.0
                if trajectory_sample is not None
                else float(config.get("tortuosity_strength_rad", 0.0))
            )
            effective_tortuosity_strength[name] = tortuosity_strength
            if trajectory_sample is not None:
                innovation_source = deviation_sample.innovation_branches.get(
                    name, np.zeros_like(trajectory_sample.local_deviations[name])
                )
                innovation = interpolate_branch_deviation(
                    innovation_source, int(config["sample_count"])
                )
                # Innovation changes only the interior course.  Exact matched
                # ostia, bifurcation and terminals remain the source baseline.
                t = np.linspace(0.0, 1.0, len(innovation))[:, None]
                endpoint_trend = (1.0 - t) * innovation[0] + t * innovation[-1]
                deviation = innovation - endpoint_trend
            else:
                deviation = interpolate_branch_deviation(
                    deviation_sample.branches[name], int(config["sample_count"])
                )
                if deviation_sample.pca_applied:
                    t = np.linspace(0.0, 1.0, len(deviation))[:, None]
                    endpoint_trend = (1.0 - t) * deviation[0] + t * deviation[-1]
                    deviation = deviation - endpoint_trend
            paths[name] = path_generator.generate(
                name,
                start,
                end,
                control_point_count=int(config["control_point_count"]),
                sample_count=int(config["sample_count"]),
                tortuosity_strength_rad=tortuosity_strength,
                obliquity_rad=float(config.get("obliquity_rad", 0.0)),
                local_deviation=deviation,
                control_uvo=(
                    trajectory_sample.branches[name]
                    if trajectory_sample is not None
                    and not trajectory_sample.exact_cardiac_points_available
                    else None
                ),
                control_points_xyz=(
                    trajectory_sample.cardiac_points[name]
                    if trajectory_sample is not None
                    and trajectory_sample.exact_cardiac_points_available
                    else None
                ),
            )

        # Use the named bifurcation landmark itself as the single source of
        # truth. PCA endpoint deviations are deliberately discarded here.
        bif = landmarks["bifurcation"]
        shared_bifurcation = (
            paths["LMCA"].points[-1].copy()
            if trajectory_sample is not None
            else surface_coordinate_to_point(bif.u, bif.v, bif.offset, self.ellipsoid)
        )
        topology_error_before = max(
            float(np.linalg.norm(paths["LMCA"].points[-1] - paths[name].points[0]))
            for name in ("LAD", "LCX")
        )
        paths["LMCA"].points[-1] = shared_bifurcation
        paths["LAD"].points[0] = shared_bifurcation
        paths["LCX"].points[0] = shared_bifurcation
        branches = {name: path.points.copy() for name, path in paths.items()}
        topology_error_after = max(
            float(np.linalg.norm(branches["LMCA"][-1] - branches[name][0]))
            for name in ("LAD", "LCX")
        )
        return SyntheticTree(
            ellipsoid=self.ellipsoid,
            landmarks=landmarks,
            branches=branches,
            surface_paths=paths,
            generation_metadata={
                "coordinate_frame": "common cardiac frame",
                "topology_constraint": "LMCA[-1] == LAD[0] == LCX[0]",
                "topology_error_before_snap_mm": topology_error_before,
                "topology_error_after_snap_mm": topology_error_after,
                "deviation_sample": deviation_sample.to_metadata(),
                "pointwise_independent_noise_added": False,
                "unlearned_tortuosity_perturbation_applied": any(
                    value > 0.0 for value in effective_tortuosity_strength.values()
                ),
                "effective_tortuosity_strength_rad": effective_tortuosity_strength,
                "PCA_endpoint_trend_removed_to_preserve_landmarks": deviation_sample.pca_applied,
                "baseline_geometry_source": (
                    "exact_matched_fixed_trajectory_and_local_basis_coefficients"
                    if trajectory_sample is not None
                    else "sampled_surface_landmarks"
                ),
                "exact_local_deviations_available": (
                    False
                    if trajectory_sample is None
                    else trajectory_sample.exact_local_deviations_available
                ),
                "exact_cardiac_controls_available": (
                    False
                    if trajectory_sample is None
                    else trajectory_sample.exact_cardiac_points_available
                ),
                "empirical_trajectory_source_case_id": (
                    None if trajectory_sample is None else trajectory_sample.source_case_id
                ),
                "path_interpolation": (
                    "normalized_arc_shape_preserving_cubic_bspline"
                    if trajectory_sample is not None
                    and trajectory_sample.exact_cardiac_points_available
                    else "normalized_parameter_cubic_bspline"
                ),
                "include_rca": include_rca,
                "path_config": self.path_config,
            },
        )
