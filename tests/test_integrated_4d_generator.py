"""Contract tests for the public disease-aware 4D generator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from vessel_tree_generator import (
    CoronaryTreeGenerator,
    GenerationConfig,
    MotionConfig,
    PulsatilityConfig,
    stenosis_config,
)
from vessel_tree_generator.disease import healthy_config
from vessel_tree_generator.export import export_case, verify_export
from vessel_tree_generator.pulsatility import phase_radius
from vessel_tree_generator.visualization import branch_tortuosity_metrics, create_case_visualizations
from vessel_tree_generator.audit import audit_exported_case


class Integrated4DGeneratorTests(unittest.TestCase):
    def test_generation_defaults_use_validated_conservative_innovation(self) -> None:
        config = GenerationConfig()
        self.assertEqual(config.pca_scale, 0.04)
        self.assertEqual(MotionConfig().radial_amplitude, 0.14)
        with self.assertRaises(ValueError):
            GenerationConfig(pca_scale=1.01).validate()

    def test_tortuosity_metric_is_one_for_straight_line(self) -> None:
        points = np.column_stack((np.linspace(0.0, 10.0, 20), np.zeros(20), np.zeros(20)))
        metrics = branch_tortuosity_metrics(points)
        self.assertAlmostEqual(metrics["distance_metric_arc_over_chord"], 1.0)
        self.assertAlmostEqual(metrics["total_turning_angle_deg"], 0.0)

    def test_public_disease_configs_cover_required_modes(self) -> None:
        focal = stenosis_config("LAD", 0.45, 0.12, 0.65, "focal")
        diffuse = stenosis_config("LCX", 0.55, 0.40, 0.45, "diffuse")
        tandem = stenosis_config(
            "LAD", 0.45, 0.08, 0.55, "tandem", tandem_positions=(0.28, 0.66)
        )
        self.assertEqual(healthy_config()["lesions"], [])
        self.assertEqual(focal["lesions"][0]["type"], "focal")
        self.assertEqual(diffuse["lesions"][0]["type"], "diffuse")
        self.assertEqual(tandem["lesions"][0]["type"], "tandem")

    def test_stenotic_wall_has_reduced_pulsatility(self) -> None:
        baseline = np.ones(3)
        diseased = np.asarray([1.0, 0.5, 1.0])
        peak, metadata = phase_radius(
            baseline,
            diseased,
            0.35,
            amplitude=0.03,
            stenosis_compliance_factor=0.35,
        )
        healthy_relative_expansion = peak[0] / diseased[0] - 1.0
        lesion_relative_expansion = peak[1] / diseased[1] - 1.0
        self.assertTrue(np.isclose(healthy_relative_expansion, 0.03))
        self.assertTrue(np.isclose(lesion_relative_expansion, 0.03 * 0.35))
        self.assertEqual(metadata["peak_response"], "systolic_lumen_expansion")
        self.assertLess(metadata["minimum_local_amplitude"], metadata["maximum_local_amplitude"])

    def test_real_frozen_model_end_to_end_export(self) -> None:
        generator = CoronaryTreeGenerator()
        case = generator.generate_case(
            stenosis_config("LAD", 0.45, 0.12, 0.65, "focal", case_id="integration_focal"),
            generation=GenerationConfig(seed=20260822, maximum_attempts=250),
            motion=MotionConfig(number_of_phases=3),
            pulsatility=PulsatilityConfig(amplitude=0.03, stenosis_compliance_factor=0.35),
            phase_values=(0.0, 0.35, 1.0),
        )
        self.assertTrue(case["validation"]["is_valid"])
        for frame in case["frames"]:
            branches = frame["branches"]
            np.testing.assert_array_equal(branches["LMCA"][-1, :3], branches["LAD"][0, :3])
            np.testing.assert_array_equal(branches["LMCA"][-1, :3], branches["LCX"][0, :3])
        for name in ("LMCA", "LAD", "LCX"):
            np.testing.assert_allclose(
                case["frames"][0]["branches"][name][:, :3],
                case["frames"][-1]["branches"][name][:, :3],
                atol=1.0e-9,
            )

        with tempfile.TemporaryDirectory(prefix="coronary4d_test_") as temporary:
            output = Path(temporary) / "case"
            manifest = export_case(case, output, points_per_branch=50)
            self.assertEqual(manifest["status"], "PASS")
            self.assertTrue(manifest["vtk_readback_passed"])
            cine = np.load(output / "geometry_cine.npy", allow_pickle=False)
            self.assertEqual(cine.shape, (3, 3, 50, 4))
            self.assertTrue(np.all(np.isfinite(cine)))
            self.assertTrue(np.all(cine[..., 3] > 0.0))
            exported_radius_change = np.max(
                np.abs(cine[..., 3] / cine[0, ..., 3][None, ...] - 1.0)
            )
            self.assertLessEqual(exported_radius_change, 0.03 + 1.0e-9)
            self.assertEqual(verify_export(output)["status"], "PASS")
            visual_manifest = create_case_visualizations(
                output,
                output / "visualizations",
                create_gif=False,
            )
            self.assertEqual(visual_manifest["status"], "PASS")
            self.assertTrue((output / "visualizations/validation_dashboard.png").is_file())
            self.assertTrue((output / "visualizations/interactive_tree.html").is_file())
            audit = audit_exported_case(output)
            self.assertTrue(audit["engineering_validation_pass"])
            self.assertTrue(audit["major_vessel_LCA_anatomical_plausibility_pass"])
            self.assertEqual(audit["pulsatility"]["peak_change_sign"], "expansion")
            ratio = audit["pulsatility"]["branches"]["LAD"]["lesion_to_nonlesion_amplitude_ratio"]
            self.assertAlmostEqual(ratio, 0.35, delta=0.02)


if __name__ == "__main__":
    unittest.main()
