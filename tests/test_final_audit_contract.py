"""Regression tests for the independently audited final contracts."""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[1]
PCA_ROOT = ROOT / "pca_ssm_vessel_tree_generator"
if str(PCA_ROOT) not in sys.path:
    sys.path.insert(0, str(PCA_ROOT))

from final_validation_audit import load_fixed, local_training, source_grouped_folds
from generation.parameter_sampler import ParameterSampler
from generation.surface_path_generator import interpolate_bspline_points, surface_coordinate_to_point
from surface_relative.surface_projection import parameterize_point_angular_radial


def test_shape_preserving_bspline_matches_hermite_reference() -> None:
    points = np.asarray([
        [0.0, 0.0, 0.0], [2.0, 1.0, -3.0], [3.0, -1.5, -8.0], [8.0, 0.5, -12.0]
    ])
    target = np.linspace(0.0, 1.0, 101)
    expected = PchipInterpolator(np.linspace(0.0, 1.0, len(points)), points, axis=0)(target)
    actual = interpolate_bspline_points(points, len(target))
    np.testing.assert_allclose(actual, expected, atol=1.0e-12)
    np.testing.assert_array_equal(actual[[0, -1]], points[[0, -1]])


def test_angular_radial_parameterization_round_trip() -> None:
    ellipsoid = ParameterSampler.controlled().sample(np.random.default_rng(8))
    rng = np.random.default_rng(9)
    for point in rng.normal(size=(100, 3)) * np.asarray([50.0, 40.0, 70.0]):
        u, v, offset, local = parameterize_point_angular_radial(
            point, ellipsoid.a, ellipsoid.b, ellipsoid.c
        )
        local = local.copy()
        local[2] = 0.0
        rebuilt = surface_coordinate_to_point(u, v, offset, ellipsoid, local)
        np.testing.assert_allclose(rebuilt, point, atol=1.0e-10)


def test_source_grouped_folds_prevent_derivative_leakage() -> None:
    sources = ["1.label", "1.label", "2.label", "2.label", "3.label", "4.label", "5.label"]
    folds = source_grouped_folds(sources, fold_count=3, seed=22)
    seen_holdout = set()
    for train, holdout in folds:
        train_sources = {sources[index] for index in train}
        holdout_sources = {sources[index] for index in holdout}
        assert not train_sources & holdout_sources
        seen_holdout |= holdout_sources
    assert seen_holdout == set(sources)


def test_frozen_pca_is_independently_reproducible() -> None:
    matrix = local_training(load_fixed())
    mean = matrix.mean(axis=0)
    _, singular, vt = np.linalg.svd(matrix - mean, full_matrices=False)
    eigen = singular ** 2 / (len(matrix) - 1)
    ratio = eigen / eigen.sum()
    retained = int(np.searchsorted(np.cumsum(ratio), 0.95) + 1)
    with np.load(
        ROOT / "outputs/lca_ssm/lca_population_model/generator_statistics/surface_deviation_pca.npz",
        allow_pickle=False,
    ) as stored:
        np.testing.assert_allclose(mean, stored["mean_vector"], atol=1.0e-10)
        assert retained == len(stored["components"]) == 13
        assert min(abs(float(np.dot(vt[index], stored["components"][index]))) for index in range(13)) > 0.999999999


def test_frozen_statistics_manifest_hashes_match_files() -> None:
    statistics = ROOT / "outputs/lca_ssm/lca_population_model/generator_statistics"
    manifest = json.loads(
        (statistics / "generator_statistics_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model_scope"] == "LCA_only"
    assert manifest["pca_case_count"] == 52
    assert manifest["shape_vector_dimensions"] == 81
    assert manifest["source_geometry_modified"] is False
    for relative, expected in manifest["files"].items():
        path = statistics / relative
        assert path.is_file(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_final_52_tree_presentation_is_complete_and_readable() -> None:
    pack = ROOT / "submission_release/final_presentation_52"
    manifest = json.loads((pack / "ALL_52_PRESENTATION_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    assert manifest["tree_count_verified"] == manifest["tree_count_expected"] == 52
    assert manifest["entry_file_count_per_tree"] == 7
    assert manifest["total_cine_frames"] == 520
    assert manifest["source_integrity"]["canonical_cohort_unchanged"] is True
    assert manifest["source_integrity"]["production_export_unchanged"] is True
    assert manifest["maximum_errors"]["mesh_source_xyz_change_mm"] == 0.0
    tree = pack / "tree_0015"
    for relative in manifest["entry_file_names"]:
        assert (tree / relative).is_file(), relative
    mesh = pv.read(tree / "diseased_mesh_with_centerlines.vtm")
    assert list(mesh.keys()) == [
        "LMCA_MESH", "LAD_MESH", "LCX_MESH",
        "LMCA_CENTERLINE", "LAD_CENTERLINE", "LCX_CENTERLINE",
    ]
    assert max(mesh["LAD_MESH"].point_data["disease_reduction_fraction"]) > 0.60
