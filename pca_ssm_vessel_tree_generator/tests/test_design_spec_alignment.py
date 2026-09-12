"""Regression contracts for the final design-specification alignment audit."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from generation.surface_path_generator import (  # noqa: E402
    interpolate_bspline_samples,
    surface_coordinate_to_point,
)
from generation.parameter_sampler import EllipsoidParameters  # noqa: E402
from motion.cardiac_motion import apply_cardiac_motion_to_tree  # noqa: E402
from surface_relative.surface_projection import (  # noqa: E402
    ellipsoid_normal,
    ellipsoid_tangent_u,
    ellipsoid_tangent_v,
    project_point_to_surface,
)


MODEL = PROJECT_ROOT / "outputs/lca_ssm/lca_population_model"
STATISTICS = MODEL / "generator_statistics"
COHORT = PROJECT_ROOT / "outputs/lca_ssm/lca_population_cohort"
AUDIT = PROJECT_ROOT / "submission_release/design_spec_alignment"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DesignSpecificationAlignmentTests(unittest.TestCase):
    def test_cardiac_frame_and_surface_role_evidence_covers_all_52_cases(self) -> None:
        rows = read_csv(AUDIT / "surface_role_audit.csv")
        self.assertEqual(len(rows), 52)
        lad_inferior = np.asarray([float(row["LAD_inferior_progression_mm"]) for row in rows])
        lcx_inferior = np.asarray([float(row["LCX_inferior_progression_mm"]) for row in rows])
        lad_v = np.asarray([float(row["LAD_v_net_rad"]) for row in rows])
        lcx_u = np.asarray([float(row["LCX_u_travel_rad"]) for row in rows])
        self.assertGreater(float(np.mean(lad_inferior)), float(np.mean(lcx_inferior)))
        self.assertGreater(float(np.mean(lad_v)), 0.0)
        self.assertGreater(float(np.mean(lcx_u)), 0.0)

    def test_wrapping_unwrapping_preserves_geometry_and_continuity(self) -> None:
        wrapped = np.asarray([3.0, 3.1, -3.1, -2.9])
        unwrapped = np.unwrap(wrapped)
        np.testing.assert_allclose(np.exp(1j * unwrapped), np.exp(1j * wrapped), atol=1.0e-12)
        self.assertLess(float(np.max(np.abs(np.diff(unwrapped)))), np.pi)

    def test_surface_reconstruction_and_local_basis(self) -> None:
        result = json.loads((AUDIT / "source_surface_reconstruction_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(result["case_count"], 52)
        self.assertLess(result["maximum_reconstruction_error_mm"], 1.0e-10)
        ellipsoid = EllipsoidParameters(45.0, 36.0, 58.0, "test")
        point = np.asarray([22.0, -11.0, -49.0])
        u, v, offset, deviation = project_point_to_surface(point, ellipsoid.a, ellipsoid.b, ellipsoid.c)
        deviation[2] = 0.0
        rebuilt = surface_coordinate_to_point(u, v, offset, ellipsoid, deviation)
        np.testing.assert_allclose(rebuilt, point, atol=1.0e-10)
        normal = ellipsoid_normal(u, v, ellipsoid.a, ellipsoid.b, ellipsoid.c)
        self.assertAlmostEqual(float(np.dot(normal, ellipsoid_tangent_u(u, v, ellipsoid.a, ellipsoid.b, ellipsoid.c))), 0.0, places=12)
        self.assertAlmostEqual(float(np.dot(normal, ellipsoid_tangent_v(u, v, ellipsoid.a, ellipsoid.b, ellipsoid.c))), 0.0, places=12)

    def test_ellipse_to_ellipsoid_provenance_and_positive_axes(self) -> None:
        rows = [row for row in read_csv(AUDIT / "ellipse_to_ellipsoid_trace.csv") if row["statistics_eligible"].lower() == "true"]
        self.assertEqual(len(rows), 52)
        for row in rows:
            self.assertAlmostEqual(float(row["coronary_ellipse_a"]), float(row["final_ellipsoid_a"]), places=10)
            self.assertAlmostEqual(float(row["coronary_ellipse_b"]), float(row["final_ellipsoid_b"]), places=10)
            self.assertGreater(min(float(row[name]) for name in ("final_ellipsoid_a", "final_ellipsoid_b", "final_ellipsoid_c")), 0.0)

    def test_lmca_lad_lcx_correspondence_and_deviation_pca_dimensions(self) -> None:
        with np.load(STATISTICS / "fixed_branch_surface_coordinates.npz", allow_pickle=False) as archive:
            case_ids = [archive[f"{branch}_case_ids"].astype(str) for branch in ("LMCA", "LAD", "LCX")]
            np.testing.assert_array_equal(case_ids[0], case_ids[1])
            np.testing.assert_array_equal(case_ids[0], case_ids[2])
            self.assertEqual(archive["LMCA_local_deviation"].shape, (52, 5, 3))
            self.assertEqual(archive["LAD_local_deviation"].shape, (52, 12, 3))
            self.assertEqual(archive["LCX_local_deviation"].shape, (52, 10, 3))
        with np.load(STATISTICS / "surface_deviation_pca.npz", allow_pickle=False) as pca:
            self.assertEqual(pca["mean_vector"].shape, (81,))
            self.assertEqual(pca["components"].shape[1], 81)

    def test_progression_obliquity_tortuosity_and_offset_statistics_are_finite(self) -> None:
        payload = json.loads((AUDIT / "real_surface_behavior_statistics.json").read_text(encoding="utf-8"))
        for name in (
            "LAD_v_net_rad", "LCX_u_travel_rad", "LCX_obliquity_rad",
            "LAD_surface_tortuosity_rad", "LCX_surface_tortuosity_rad",
            "LAD_offset_P95_abs_mm", "LCX_offset_P95_abs_mm",
        ):
            stats = payload["metrics"][name]
            self.assertEqual(stats["N"], 52)
            self.assertTrue(all(np.isfinite(float(stats[key])) for key in ("mean", "SD", "median", "P5", "P95")))
        self.assertGreater(payload["metrics"]["LAD_v_net_rad"]["mean"], 0.0)
        self.assertGreater(payload["metrics"]["LCX_u_travel_rad"]["mean"], 0.0)

    def test_uv_spline_candidate_preserves_endpoints_and_shared_bifurcation(self) -> None:
        result = json.loads((AUDIT / "uv_spline_equivalence_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(result["case_count"], 52)
        self.assertEqual(result["branch_path_count"], 156)
        self.assertLess(result["maximum_endpoint_error_mm"], 1.0e-10)
        self.assertLess(result["maximum_bifurcation_error_mm"], 1.0e-10)
        samples = np.asarray([3.0, 3.15, 3.30, 3.45])
        dense = interpolate_bspline_samples(samples, 101)
        np.testing.assert_allclose(dense[[0, -1]], samples[[0, -1]], atol=0.0)

    def test_source_and_frozen_package_are_immutable(self) -> None:
        result = json.loads((AUDIT / "design_alignment_integrity.json").read_text(encoding="utf-8"))
        self.assertEqual(result["source_coordinate_change_mm"], 0.0)
        self.assertEqual(result["source_segment_length_change_mm"], 0.0)
        self.assertTrue(result["protected_source_hash_verification"])
        manifest = json.loads((STATISTICS / "generator_statistics_manifest.json").read_text(encoding="utf-8"))
        for name, expected in manifest["files"].items():
            self.assertEqual(sha256(STATISTICS / name), expected)

    def test_surface_relative_tree_remains_motion_compatible(self) -> None:
        directory = COHORT / "tree_0001"
        parameters = json.loads((directory / "parameters.json").read_text(encoding="utf-8"))
        tree = {
            "tree_id": "tree_0001",
            "ellipsoid_params": {
                "a_mm": parameters["ellipsoid"]["a"],
                "b_mm": parameters["ellipsoid"]["b"],
                "c_mm": parameters["ellipsoid"]["c"],
            },
            "vessels_3d": {branch: np.load(directory / f"{branch}.npy") for branch in ("LMCA", "LAD", "LCX")},
        }
        motion = apply_cardiac_motion_to_tree(tree, phase_values=[0.0, 0.35, 1.0])
        for branch in ("LMCA", "LAD", "LCX"):
            np.testing.assert_allclose(motion["frames"][0]["vessels_3d"][branch], tree["vessels_3d"][branch], atol=1.0e-10)
            np.testing.assert_allclose(motion["frames"][0]["vessels_3d"][branch], motion["frames"][-1]["vessels_3d"][branch], atol=1.0e-10)
        for frame in motion["frames"]:
            bifurcation = frame["vessels_3d"]["LMCA"][-1]
            np.testing.assert_allclose(frame["vessels_3d"]["LAD"][0], bifurcation, atol=0.0)
            np.testing.assert_allclose(frame["vessels_3d"]["LCX"][0], bifurcation, atol=0.0)


if __name__ == "__main__":
    unittest.main()
