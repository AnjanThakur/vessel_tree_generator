"""Batch 4 Pipeline Coordinator for PCA / Statistical Shape Model (SSM) fitting."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
import numpy as np

from extraction.nifti_loader import load_centerline_archive
from extraction.skeleton_graph import build_graph_from_summary
from surface_relative.surface_projection import project_point_to_surface
from surface_relative.population_statistics import (
    fit_deviation_pca,
    compute_linear_stats,
    compute_landmark_statistics,
    compute_side_branch_statistics,
    measure_tortuosity,
    measure_obliquity,
    build_validation_thresholds,
)
from ssm.shape_model import StatisticalShapeModel, DIMENSION_MAPPING

EPS = 1.0e-12


def extract_patient_side_branches(
    centerlines_dir: Path, patient_id: str, centerlines_cardiac: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Extract per-patient side branch count, attachment t, length, and direction angle (Design Doc §5.5)."""
    p_archive_dir = centerlines_dir / patient_id
    if not p_archive_dir.exists():
        p_archive_dir = centerlines_dir / f"{patient_id}.label"

    side_branch_data = {
        vname: {"count": 0, "attachment_t": [], "length_mm": [], "angle_deg": []}
        for vname in ("RCA", "LMCA", "LAD", "LCX")
    }

    if not p_archive_dir.exists():
        return side_branch_data

    try:
        archive = load_centerline_archive(p_archive_dir)
        G, node_coords, _, _, junction_nodes = build_graph_from_summary(
            archive["summary_df"], archive["branches_dict"]
        )
    except Exception:
        return side_branch_data

    if len(junction_nodes) == 0:
        return side_branch_data

    for vname, card_pts in centerlines_cardiac.items():
        if card_pts is None or len(card_pts) < 2:
            continue

        seg_vectors = np.diff(card_pts, axis=0)
        seg_lens = np.linalg.norm(seg_vectors, axis=1)
        cum_lens = np.insert(np.cumsum(seg_lens), 0, 0.0)
        total_len = max(float(cum_lens[-1]), EPS)

        j_count = 0
        attachments_t = []
        lengths_mm = []
        angles_deg = []

        # Find junctions near this vessel centerline
        for j_node in junction_nodes:
            j_coord = node_coords[j_node]
            dists = np.linalg.norm(card_pts - j_coord, axis=1)
            min_idx = int(np.argmin(dists))
            min_dist = float(dists[min_idx])

            if min_dist <= 5.0:  # Junction within 5mm of main trunk
                j_count += 1
                t_val = float(np.clip(cum_lens[min_idx] / total_len, 0.0, 1.0))
                attachments_t.append(t_val)

                # Evaluate branch path starting at junction
                neighbors = [n for n in G.neighbors(j_node)]
                if len(neighbors) >= 3:
                    for nbr in neighbors:
                        nbr_coord = node_coords[nbr]
                        branch_vec = nbr_coord - j_coord
                        b_len = float(np.linalg.norm(branch_vec))
                        if b_len > EPS:
                            # Parent tangent vector at min_idx
                            if min_idx < len(card_pts) - 1:
                                parent_vec = card_pts[min_idx + 1] - card_pts[min_idx]
                            else:
                                parent_vec = card_pts[min_idx] - card_pts[min_idx - 1]

                            parent_norm = np.linalg.norm(parent_vec)
                            if parent_norm > EPS:
                                cos_angle = float(np.clip(np.dot(parent_vec, branch_vec) / (parent_norm * b_len), -1.0, 1.0))
                                angle_deg = float(math.degrees(math.acos(cos_angle)))
                                lengths_mm.append(b_len)
                                angles_deg.append(angle_deg)

        side_branch_data[vname] = {
            "count": j_count,
            "attachment_t": attachments_t,
            "length_mm": lengths_mm,
            "angle_deg": angles_deg,
        }

    return side_branch_data


def process_batch4_population(
    batch3_dir: Path,
    centerlines_dir: Path | None = None,
    variance_cutoff: float = 0.95,
) -> dict[str, Any]:
    """Execute Batch 4 PCA / Statistical Shape Model fitting across 174 Batch-3 accepted patient records."""
    b3_summary_path = batch3_dir / "batch3_summary.json"
    if not b3_summary_path.exists():
        raise FileNotFoundError(f"Batch 3 summary file missing at {b3_summary_path}")

    with open(b3_summary_path, "r", encoding="utf-8") as f:
        b3_summary = json.load(f)

    patient_results_b3 = b3_summary.get("patient_results", [])

    patient_ids = []
    shape_vectors_list = []
    scaffold_list = []
    landmarks_18d_list = []
    patient_branches_data = []

    branch_lengths = {"RCA": [], "LMCA": [], "LAD": [], "LCX": []}
    bifurcation_angles = []
    max_out_of_plane_devs = {"RCA": [], "LMCA": [], "LAD": [], "LCX": []}
    tortuosities_dict = {"RCA": [], "LMCA": [], "LAD": [], "LCX": []}
    obliquities_dict = {"RCA": [], "LMCA": [], "LAD": [], "LCX": []}

    project_root = batch3_dir.resolve().parent.parent
    if centerlines_dir is None:
        centerlines_dir = project_root / "lca_ssm" / "centerlines"

    for p_info in patient_results_b3:
        p_id = p_info["patient_id"]
        if not p_info.get("batch3_passed", False):
            continue

        p_dir = batch3_dir / p_id
        fixed_path = p_dir / "fixed_representation.json"
        ellipsoid_path = p_dir / "ellipsoid_params.json"
        proj_path = p_dir / "surface_projection.json"
        b2_dir = project_root / "outputs" / "batch2_cardiac_frame" / p_id
        lm_path = b2_dir / "landmarks_cardiac.json"
        cl_path = b2_dir / "centerlines_cardiac.npz"

        if not fixed_path.exists() or not ellipsoid_path.exists() or not proj_path.exists() or not lm_path.exists() or not cl_path.exists():
            continue

        with open(fixed_path, "r", encoding="utf-8") as ff:
            fixed_data = json.load(ff)

        shape_vec = fixed_data.get("shape_vector")
        if shape_vec is None or len(shape_vec) != 126:
            continue

        with open(ellipsoid_path, "r", encoding="utf-8") as ef:
            ell_data = json.load(ef)

        card_ell = ell_data.get("cardiac_axis_aligned_ellipsoid", {})
        a = card_ell.get("a_mm", 0.0)
        b = card_ell.get("b_effective_mm", 0.0)
        c = card_ell.get("c_mm", 0.0)

        with open(proj_path, "r", encoding="utf-8") as pf:
            proj_data = json.load(pf)

        with open(lm_path, "r", encoding="utf-8") as lmf:
            landmarks_cardiac = json.load(lmf)

        centerlines_data = np.load(cl_path)
        centerlines_cardiac = {k: centerlines_data[k] for k in centerlines_data.files}

        # 1. Level 2 Landmark Statistics: 18-D (u, v, offset) Vector Extraction (§5.2)
        lm_names = ["lca_ostium", "rca_ostium", "bifurcation", "lad_endpoint", "lcx_endpoint", "rca_endpoint"]
        lm_18d = []
        for lm_k in lm_names:
            lm_coord = landmarks_cardiac.get(lm_k)
            if lm_coord is not None and len(lm_coord) == 3:
                u_lm, v_lm, off_lm, _ = project_point_to_surface(np.asarray(lm_coord, dtype=float), a, b, c)
                lm_18d.extend([u_lm, v_lm, off_lm])
            else:
                lm_18d.extend([0.0, 0.0, 0.0])

        landmarks_18d_list.append(np.asarray(lm_18d, dtype=float))

        # 2. Level 6 Real Branch Lengths & Max Out-of-Plane Deviations (§5.6)
        for vname in ("RCA", "LMCA", "LAD", "LCX"):
            c_pts = centerlines_cardiac.get(vname)
            if c_pts is not None and len(c_pts) >= 2:
                c_pts = np.asarray(c_pts, dtype=float)
                lens = np.linalg.norm(np.diff(c_pts, axis=0), axis=1)
                branch_lengths[vname].append(float(np.sum(lens)))

            v_proj = proj_data.get(vname)
            if v_proj is not None and "offset" in v_proj:
                offset_arr = np.abs(np.asarray(v_proj["offset"], dtype=float))
                max_out_of_plane_devs[vname].append(float(np.max(offset_arr)))
            elif v_proj is not None and "deviations" in v_proj:
                devs_arr = np.asarray(v_proj["deviations"], dtype=float)
                offset_arr = np.abs(devs_arr[:, 2])
                max_out_of_plane_devs[vname].append(float(np.max(offset_arr)))

            if v_proj is not None and "u" in v_proj and "v" in v_proj:
                u_p = np.asarray(v_proj["u"], dtype=float)
                v_p = np.asarray(v_proj["v"], dtype=float)
                tortuosities_dict[vname].append(measure_tortuosity(u_p, v_p))
                obliquities_dict[vname].append(measure_obliquity(u_p))

        # 3. Level 6 Real LMCA Bifurcation Angle (§5.6)
        lad_pts = centerlines_cardiac.get("LAD")
        lcx_pts = centerlines_cardiac.get("LCX")
        if lad_pts is not None and lcx_pts is not None and len(lad_pts) >= 2 and len(lcx_pts) >= 2:
            v_lad = np.asarray(lad_pts[1] - lad_pts[0], dtype=float)
            v_lcx = np.asarray(lcx_pts[1] - lcx_pts[0], dtype=float)
            norm_lad = np.linalg.norm(v_lad)
            norm_lcx = np.linalg.norm(v_lcx)
            if norm_lad > EPS and norm_lcx > EPS:
                cos_bif = float(np.clip(np.dot(v_lad, v_lcx) / (norm_lad * norm_lcx), -1.0, 1.0))
                bif_angle_deg = float(math.degrees(math.acos(cos_bif)))
                bifurcation_angles.append(bif_angle_deg)

        # 4. Level 5 Side Branch Extraction (§5.5)
        sb_info = extract_patient_side_branches(centerlines_dir, p_id, centerlines_cardiac)
        patient_branches_data.append(sb_info)

        patient_ids.append(p_id)
        shape_vectors_list.append(np.asarray(shape_vec, dtype=float))
        scaffold_list.append({"a": a, "b": b, "c": c})

    n_patients = len(shape_vectors_list)
    if n_patients == 0:
        raise ValueError("No valid Batch-3 patient representations found for PCA model fitting")

    # 1. Build Population Shape Matrix X in R^(174 x 126)
    X = np.vstack(shape_vectors_list)  # (174, 126)

    # 2. Fit 126-D Deviation PCA
    pca_raw = fit_deviation_pca(X, variance_cutoff=variance_cutoff)

    # 3. Construct StatisticalShapeModel
    model = StatisticalShapeModel(
        mean_vector=pca_raw["mean_vector"],
        components_all=pca_raw["components_all"],
        components_retained=pca_raw["components_retained"],
        singular_values=pca_raw["singular_values"],
        eigenvalues=pca_raw["eigenvalues"],
        eigenvalues_retained=pca_raw["eigenvalues_retained"],
        explained_variance_ratio=pca_raw["explained_variance_ratio"],
        cumulative_explained_variance=pca_raw["cumulative_explained_variance"],
        k_retained=pca_raw["k_retained"],
        variance_cutoff=variance_cutoff,
        n_samples=n_patients,
        n_features=126,
    )

    # 4. Patient Reconstructions & RMSE
    patient_reconstructions = []
    rmse_list = []

    for idx, p_id in enumerate(patient_ids):
        x_orig = X[idx]
        b_vec = model.encode(x_orig)
        x_rec, rmse = model.reconstruct(x_orig)
        rmse_list.append(rmse)

        patient_reconstructions.append({
            "patient_id": p_id,
            "pca_coefficients_b": b_vec.tolist(),
            "reconstructed_shape_vector": x_rec.tolist(),
            "reconstruction_rmse": rmse,
        })

    # 5. Validation & Invariant Checks
    validation_failures = []

    # Check 1: Basis orthogonality P_k^T * P_k == I_k
    ortho_matrix = model.components_retained @ model.components_retained.T
    ortho_err = float(np.max(np.abs(ortho_matrix - np.eye(model.k_retained))))
    if ortho_err > 1.0e-10:
        validation_failures.append(f"PCA retained eigenvector orthogonality error ({ortho_err:.2e}) exceeds 1e-10")

    # Check 2: Exact reconstruction with all 126 modes
    full_model = StatisticalShapeModel(
        mean_vector=pca_raw["mean_vector"],
        components_all=pca_raw["components_all"],
        components_retained=pca_raw["components_all"],
        singular_values=pca_raw["singular_values"],
        eigenvalues=pca_raw["eigenvalues"],
        eigenvalues_retained=pca_raw["eigenvalues"],
        explained_variance_ratio=pca_raw["explained_variance_ratio"],
        cumulative_explained_variance=pca_raw["cumulative_explained_variance"],
        k_retained=126,
        variance_cutoff=1.0,
        n_samples=n_patients,
        n_features=126,
    )

    max_full_rec_err = 0.0
    for idx in range(n_patients):
        _, r_err = full_model.reconstruct(X[idx])
        max_full_rec_err = max(max_full_rec_err, r_err)

    if max_full_rec_err > 1.0e-10:
        validation_failures.append(f"Full-rank (126 modes) reconstruction RMSE ({max_full_rec_err:.2e}) exceeds 1e-10")

    # Check 3: Cumulative variance >= cutoff
    actual_cum_var = float(model.cumulative_explained_variance[model.k_retained - 1])
    if actual_cum_var < variance_cutoff - 1.0e-6:
        validation_failures.append(f"Retained variance ({actual_cum_var:.4f}) is below cutoff ({variance_cutoff})")

    overall_pass = (len(validation_failures) == 0)

    # 6. Compute Level 2 Landmark Statistics (§5.2)
    landmark_matrix_18d = np.vstack(landmarks_18d_list)  # (174, 18)
    landmark_stats = compute_landmark_statistics(landmark_matrix_18d)

    # 7. Compute Level 5 Side Branch Statistics (§5.5)
    side_branch_stats = compute_side_branch_statistics(patient_branches_data)

    # 8. Compute Level 6 Population Validation Thresholds with Real Measurements (§5.6)
    thresholds = build_validation_thresholds(
        scaffold_params=scaffold_list,
        branch_lengths=branch_lengths,
        bifurcation_angles=bifurcation_angles,
        max_out_of_plane_devs=max_out_of_plane_devs,
        tortuosities=tortuosities_dict,
        pca_results=pca_raw,
    )

    return {
        "overall_pass": overall_pass,
        "n_patients": n_patients,
        "population_shape_matrix_shape": list(X.shape),
        "shape_model": model,
        "pca_raw": pca_raw,
        "patient_reconstructions": patient_reconstructions,
        "population_statistics": {
            "scaffold_level_1": {
                "a_mean_mm": float(np.mean([s["a"] for s in scaffold_list])),
                "a_std_mm": float(np.std([s["a"] for s in scaffold_list])),
                "b_mean_mm": float(np.mean([s["b"] for s in scaffold_list])),
                "b_std_mm": float(np.std([s["b"] for s in scaffold_list])),
                "c_mean_mm": float(np.mean([s["c"] for s in scaffold_list])),
                "c_std_mm": float(np.std([s["c"] for s in scaffold_list])),
            },
            "landmarks_level_2": landmark_stats,
            "vessel_deviations_level_3": {
                "n_modes_retained": model.k_retained,
                "cumulative_variance_explained": actual_cum_var,
                "reconstruction_rmse": compute_linear_stats(rmse_list),
            },
            "tortuosity_obliquity_level_4": {
                "tortuosity": {v: compute_linear_stats(tortuosities_dict[v]) for v in ("RCA", "LMCA", "LAD", "LCX")},
                "obliquity": {v: compute_linear_stats(obliquities_dict[v]) for v in ("RCA", "LMCA", "LAD", "LCX")},
            },
            "side_branches_level_5": side_branch_stats,
        },
        "validation_thresholds": thresholds,
        "validation": {
            "overall_pass": overall_pass,
            "orthogonality_error": ortho_err,
            "max_full_rank_reconstruction_rmse": max_full_rec_err,
            "retained_cumulative_explained_variance": actual_cum_var,
            "failures": validation_failures,
        },
    }
