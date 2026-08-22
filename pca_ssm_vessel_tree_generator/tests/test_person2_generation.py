"""Focused tests for the Person 2 Week-1 generation architecture."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyvista as pv

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from generation.deviation_sampler import DeviationSampler, ZeroDeviationSampler
from generation.landmark_sampler import LandmarkSampler
from generation.parameter_sampler import ParameterSampler
from generation.surface_path_generator import (
    SurfacePathGenerator,
    interpolate_bspline_samples,
    surface_coordinate_to_point,
)
from generation.trajectory_sampler import TrajectorySampler
from generation.tree_assembler import TreeAssembler
from generation.validator import TreeValidator, minimum_nonlocal_distance
from generation.vtk_export import export_tree_vtk
from surface_relative.anatomy import anatomical_role_acceptance, coronary_course_metrics
from surface_relative.surface_projection import (
    ellipsoid_normal,
    ellipsoid_point,
    ellipsoid_tangent_u,
    ellipsoid_tangent_v,
    project_point_to_surface,
)


class Person2GenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(12345)
        self.ellipsoid = ParameterSampler.controlled().sample(self.rng)
        self.landmarks = LandmarkSampler.controlled().sample(self.rng)

    def test_shared_surface_api_basis_is_consumed(self) -> None:
        u, v = 0.7, 1.8
        point = ellipsoid_point(u, v, self.ellipsoid.a, self.ellipsoid.b, self.ellipsoid.c)
        normal = ellipsoid_normal(u, v, self.ellipsoid.a, self.ellipsoid.b, self.ellipsoid.c)
        tangent_u = ellipsoid_tangent_u(u, v, self.ellipsoid.a, self.ellipsoid.b, self.ellipsoid.c)
        tangent_v = ellipsoid_tangent_v(u, v, self.ellipsoid.a, self.ellipsoid.b, self.ellipsoid.c)
        self.assertTrue(np.all(np.isfinite(point)))
        self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0, places=12)
        self.assertAlmostEqual(float(np.linalg.norm(tangent_u)), 1.0, places=12)
        self.assertAlmostEqual(float(np.linalg.norm(tangent_v)), 1.0, places=12)
        self.assertAlmostEqual(float(np.dot(normal, tangent_u)), 0.0, places=12)
        self.assertAlmostEqual(float(np.dot(normal, tangent_v)), 0.0, places=12)

    def test_triaxial_local_basis_reconstructs_source_point_exactly(self) -> None:
        point = np.asarray([31.2, -17.4, -82.0])
        u, v, offset, deviation = project_point_to_surface(
            point, self.ellipsoid.a, self.ellipsoid.b, self.ellipsoid.c
        )
        local = deviation.copy()
        local[2] = 0.0  # normal coefficient is already stored as offset
        reconstructed = surface_coordinate_to_point(
            u, v, offset, self.ellipsoid, local
        )
        np.testing.assert_allclose(reconstructed, point, atol=1.0e-10)

    def test_bspline_preserves_landmark_endpoints(self) -> None:
        generator = SurfacePathGenerator(self.ellipsoid, self.rng)
        path = generator.generate(
            "LAD",
            self.landmarks["bifurcation"],
            self.landmarks["lad_endpoint"],
            control_point_count=7,
            sample_count=100,
            tortuosity_strength_rad=0.02,
        )
        self.assertEqual(path.points.shape, (100, 3))
        self.assertAlmostEqual(path.u[0], self.landmarks["bifurcation"].u, places=12)
        self.assertAlmostEqual(path.v[0], self.landmarks["bifurcation"].v, places=12)
        self.assertAlmostEqual(path.u[-1], self.landmarks["lad_endpoint"].u, places=12)
        self.assertAlmostEqual(path.v[-1], self.landmarks["lad_endpoint"].v, places=12)

    def test_empirical_bspline_interpolates_fixed_trajectory_samples(self) -> None:
        samples = np.asarray([0.0, 1.25, -0.5, 2.0, 1.0])
        interpolated = interpolate_bspline_samples(samples, 101)
        np.testing.assert_allclose(interpolated[[0, 25, 50, 75, 100]], samples, atol=1.0e-12)

    def test_empirical_cartesian_controls_do_not_loop_at_surface_pole(self) -> None:
        controls = np.asarray([
            [35.0, 5.0, -20.0],
            [25.0, 8.0, -45.0],
            [12.0, 5.0, -65.0],
            [2.0, 1.0, -82.0],
            [-8.0, -2.0, -88.0],
        ])
        generator = SurfacePathGenerator(self.ellipsoid, self.rng)
        path = generator.generate(
            "LAD",
            self.landmarks["bifurcation"],
            self.landmarks["lad_endpoint"],
            control_point_count=len(controls),
            sample_count=180,
            local_deviation=np.zeros((180, 3)),
            control_points_xyz=controls,
        )
        control_length = float(np.linalg.norm(np.diff(controls, axis=0), axis=1).sum())
        generated_length = float(np.linalg.norm(np.diff(path.points, axis=0), axis=1).sum())
        np.testing.assert_allclose(path.points[[0, -1]], controls[[0, -1]], atol=1.0e-10)
        self.assertLess(generated_length, 1.10 * control_length)

    def test_joint_pca_sample_is_partitioned_without_point_noise(self) -> None:
        mean = np.zeros(126)
        components = np.zeros((2, 126))
        components[0, 0] = 1.0
        components[1, -1] = 1.0
        sampler = DeviationSampler(mean, components, np.array([4.0, 1.0]))
        sample = sampler.sample(self.rng)
        self.assertTrue(sample.pca_applied)
        self.assertEqual(sample.branches["RCA"].shape, (15, 3))
        self.assertEqual(sample.branches["LMCA"].shape, (5, 3))
        self.assertEqual(sample.branches["LAD"].shape, (12, 3))
        self.assertEqual(sample.branches["LCX"].shape, (10, 3))

    def test_joint_pca_uses_case_matched_training_score_baseline(self) -> None:
        mean = np.zeros(126)
        components = np.zeros((2, 126))
        components[0, 0] = 1.0
        components[1, 45] = 1.0
        sampler = DeviationSampler(
            mean,
            components,
            np.asarray([4.0, 1.0]),
            include_mean=True,
            variation_scale=0.0,
            training_scores=np.asarray([[2.5, -1.25]]),
            case_ids=np.asarray(["7.label"]),
        )
        sample = sampler.sample(self.rng, source_case_id="7.label")
        self.assertEqual(sample.baseline_case_id, "7.label")
        np.testing.assert_allclose(sample.coefficients, [2.5, -1.25])
        np.testing.assert_allclose(sample.innovation_coefficients, [0.0, 0.0])
        self.assertAlmostEqual(sample.branches["RCA"][0, 0], 2.5)
        self.assertAlmostEqual(sample.branches["LMCA"][0, 0], -1.25)

    def test_tree_topology_and_structural_validation(self) -> None:
        tree = TreeAssembler(self.ellipsoid, self.rng, ZeroDeviationSampler()).assemble(self.landmarks)
        self.assertTrue(np.array_equal(tree.branches["LMCA"][-1], tree.branches["LAD"][0]))
        self.assertTrue(np.array_equal(tree.branches["LMCA"][-1], tree.branches["LCX"][0]))
        report = TreeValidator().validate(tree)
        self.assertTrue(report["accepted"], report["errors"])
        self.assertEqual(report["validation_level"], "structural_only")
        anatomy = anatomical_role_acceptance(
            coronary_course_metrics(tree.branches["LAD"], tree.branches["LCX"])
        )
        self.assertTrue(anatomy["accepted"], anatomy["failed_checks"])

    def test_empirical_trajectory_is_matched_to_parameter_case(self) -> None:
        case_ids = np.asarray(["1.label", "2.label"])
        trajectories = {
            name: np.stack((np.full((4, 3), index), np.full((4, 3), index + 10.0)))
            for index, name in enumerate(("LMCA", "LAD", "LCX"), start=1)
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixed_branch_surface_coordinates.npz"
            np.savez(
                path,
                **{
                    key: value
                    for name, values in trajectories.items()
                    for key, value in ((f"{name}_uvo", values), (f"{name}_case_ids", case_ids))
                },
            )
            sample = TrajectorySampler.from_npz(path).sample(
                self.rng, source_case_id="2.label", include_rca=False
            )
        self.assertEqual(sample.source_case_id, "2.label")
        for name, values in trajectories.items():
            np.testing.assert_array_equal(sample.branches[name], values[1])

    def test_empirical_trajectory_disables_unlearned_tortuosity_noise(self) -> None:
        endpoints = {
            "LMCA": (self.landmarks["lca_ostium"], self.landmarks["bifurcation"]),
            "LAD": (self.landmarks["bifurcation"], self.landmarks["lad_endpoint"]),
            "LCX": (self.landmarks["bifurcation"], self.landmarks["lcx_endpoint"]),
        }
        trajectories = {}
        for name, (start, end) in endpoints.items():
            trajectories[name] = np.asarray([np.column_stack((
                np.linspace(start.u, end.u, 4),
                np.linspace(start.v, end.v, 4),
                np.linspace(start.offset, end.offset, 4),
            ))])
        cases = {name: np.asarray(["1.label"]) for name in trajectories}
        sampler = TrajectorySampler(trajectories, cases)
        tree = TreeAssembler(
            self.ellipsoid,
            self.rng,
            ZeroDeviationSampler(),
            trajectory_sampler=sampler,
        ).assemble(self.landmarks)
        self.assertFalse(tree.generation_metadata["unlearned_tortuosity_perturbation_applied"])
        self.assertEqual(
            tree.generation_metadata["effective_tortuosity_strength_rad"],
            {"LMCA": 0.0, "LAD": 0.0, "LCX": 0.0},
        )

    def test_validator_rejects_lcx_as_dominant_descending_branch(self) -> None:
        tree = TreeAssembler(self.ellipsoid, self.rng, ZeroDeviationSampler()).assemble(self.landmarks)
        bifurcation_z = float(tree.branches["LCX"][0, 2])
        lad_descent = bifurcation_z - float(tree.branches["LAD"][-1, 2])
        lcx = tree.branches["LCX"].copy()
        lcx[:, 2] = np.linspace(bifurcation_z, bifurcation_z - lad_descent - 10.0, len(lcx))
        tree.branches["LCX"] = lcx
        report = TreeValidator().validate(tree)
        self.assertFalse(report["accepted"])
        self.assertTrue(
            any("dominant descending branch" in error for error in report["errors"]),
            report["errors"],
        )

    def test_validator_rejects_lcx_with_excessive_apex_descent_ratio(self) -> None:
        tree = TreeAssembler(self.ellipsoid, self.rng, ZeroDeviationSampler()).assemble(self.landmarks)
        bifurcation_z = float(tree.branches["LCX"][0, 2])
        lad_descent = bifurcation_z - float(tree.branches["LAD"][-1, 2])
        lcx = tree.branches["LCX"].copy()
        lcx[:, 2] = np.linspace(bifurcation_z, bifurcation_z - 0.8 * lad_descent, len(lcx))
        tree.branches["LCX"] = lcx
        report = TreeValidator().validate(tree)
        self.assertFalse(report["accepted"])
        self.assertTrue(
            any("too apex-directed" in error for error in report["errors"]),
            report["errors"],
        )

    def test_validator_rejects_lmca_longer_than_a_major_daughter(self) -> None:
        tree = TreeAssembler(self.ellipsoid, self.rng, ZeroDeviationSampler()).assemble(self.landmarks)
        bifurcation = tree.branches["LMCA"][-1].copy()
        daughter_lengths = [
            float(np.linalg.norm(np.diff(tree.branches[name], axis=0), axis=1).sum())
            for name in ("LAD", "LCX")
        ]
        excessive_length = min(daughter_lengths) + 10.0
        tree.branches["LMCA"] = np.linspace(
            bifurcation + np.array([excessive_length, 0.0, 0.0]),
            bifurcation,
            len(tree.branches["LMCA"]),
        )

        report = TreeValidator().validate(tree)
        self.assertFalse(report["accepted"])
        self.assertIn(
            "LMCA is not shorter than both major daughter branches",
            report["errors"],
        )

    def test_validator_rejects_rca_lca_collision(self) -> None:
        landmarks = LandmarkSampler.controlled().sample(self.rng, include_rca=True)
        tree = TreeAssembler(self.ellipsoid, self.rng, ZeroDeviationSampler()).assemble(
            landmarks, include_rca=True
        )
        lad = tree.branches["LAD"]
        source_s = np.linspace(0.0, 1.0, len(lad))
        target_s = np.linspace(0.0, 1.0, len(tree.branches["RCA"]))
        tree.branches["RCA"] = np.column_stack([
            np.interp(target_s, source_s, lad[:, dimension]) for dimension in range(3)
        ])
        report = TreeValidator().validate(tree)
        self.assertFalse(report["accepted"])
        self.assertTrue(
            any("LAD and RCA" in error and "collision" in error for error in report["errors"]),
            report["errors"],
        )

    def test_physical_arc_separation_detects_a_true_loop(self) -> None:
        outbound = np.column_stack((np.linspace(0.0, 20.0, 80), np.zeros(80), np.zeros(80)))
        returning = np.column_stack((np.linspace(20.0, 0.2, 80), np.ones(80) * 0.2, np.zeros(80)))
        loop = np.vstack((outbound, returning[1:]))
        self.assertLess(minimum_nonlocal_distance(loop), 0.75)

    def test_physical_arc_separation_ignores_dense_straight_sampling(self) -> None:
        straight = np.column_stack((np.linspace(0.0, 6.0, 200), np.zeros(200), np.zeros(200)))
        self.assertGreaterEqual(minimum_nonlocal_distance(straight), 3.0)

    def test_vtk_export_reopens_named_blocks(self) -> None:
        tree = TreeAssembler(self.ellipsoid, self.rng, ZeroDeviationSampler()).assemble(self.landmarks)
        with tempfile.TemporaryDirectory() as temporary:
            paths = export_tree_vtk(tree, Path(temporary))
            multiblock = pv.read(paths["SYNTHETIC_TREE_VTM"])
            self.assertEqual(
                list(multiblock.keys()),
                ["SYNTHETIC_ELLIPSOID", "SYNTHETIC_LMCA", "SYNTHETIC_LAD", "SYNTHETIC_LCX", "SYNTHETIC_LANDMARKS"],
            )
            for name in ("LMCA", "LAD", "LCX"):
                mesh = pv.read(paths[f"SYNTHETIC_{name}"])
                self.assertEqual(mesh.n_lines, 1)
                self.assertEqual(mesh.n_verts, 0)
                self.assertTrue({
                    "surface_u_rad",
                    "surface_v_rad",
                    "surface_normal_offset_mm",
                    "deviation_tangent_u_mm",
                    "deviation_tangent_v_mm",
                    "deviation_normal_mm",
                }.issubset(mesh.point_data))


if __name__ == "__main__":
    unittest.main()
