"""Synthetic Coronary Tree Builder implementing the exact 12-step Part 6 generation pipeline for Batch 5."""

from __future__ import annotations

import math
from typing import Any
import numpy as np

try:
    from surface_relative.surface_projection import (
        ellipsoid_point,
        ellipsoid_normal,
        ellipsoid_tangent_u,
        ellipsoid_tangent_v,
    )
except ImportError:
    from pca_ssm_vessel_tree_generator.surface_relative.surface_projection import (
        ellipsoid_point,
        ellipsoid_normal,
        ellipsoid_tangent_u,
        ellipsoid_tangent_v,
    )

try:
    from generation.bspline_surface import generate_vessel_uv, surface_spline
    from generation.validator import validate_synthetic_tree
    from ssm.shape_model import StatisticalShapeModel, DIMENSION_MAPPING
except ImportError:
    from pca_ssm_vessel_tree_generator.generation.bspline_surface import generate_vessel_uv, surface_spline
    from pca_ssm_vessel_tree_generator.generation.validator import validate_synthetic_tree
    from pca_ssm_vessel_tree_generator.ssm.shape_model import StatisticalShapeModel, DIMENSION_MAPPING

EPS = 1.0e-12

CONTROL_POINT_COUNTS = {
    "RCA": 15,
    "LMCA": 5,
    "LAD": 12,
    "LCX": 10,
}

LM_NAMES_ORDER = [
    "lca_ostium", "rca_ostium", "bifurcation",
    "lad_endpoint", "lcx_endpoint", "rca_endpoint"
]


class SyntheticCoronaryTreeBuilder:
    """Builder generating synthetic 3D coronary artery trees adhering strictly to Part 6 (Steps 1 to 12)."""

    def __init__(
        self,
        shape_model: StatisticalShapeModel,
        population_stats: dict[str, Any],
        validation_thresholds: dict[str, Any],
        rng_seed: int | None = None,
    ):
        self.shape_model = shape_model
        self.population_stats = population_stats
        self.validation_thresholds = validation_thresholds
        self.rng = np.random.default_rng(rng_seed)

    def sample_ellipsoid_scaffold(self) -> tuple[float, float, float]:
        """Step 1 — Sample Ellipsoid parameters (a, b, c) truncated at ±2σ (Design Doc §6.2)."""
        scaf_stats = self.population_stats.get("scaffold_level_1", {})
        mu_a = scaf_stats.get("a_mean_mm", 38.9)
        std_a = scaf_stats.get("a_std_mm", 11.8)
        mu_b = scaf_stats.get("b_mean_mm", 33.4)
        std_b = scaf_std = scaf_stats.get("b_std_mm", 14.0)
        mu_c = scaf_stats.get("c_mean_mm", 27.8)
        std_c = scaf_stats.get("c_std_mm", 9.1)

        a = float(self.rng.normal(mu_a, max(std_a, 0.1)))
        a = float(np.clip(a, mu_a - 2.0 * std_a, mu_a + 2.0 * std_a))

        b = float(self.rng.normal(mu_b, max(std_b, 0.1)))
        b = float(np.clip(b, mu_b - 2.0 * std_b, mu_b + 2.0 * std_b))

        c = float(self.rng.normal(mu_c, max(std_c, 0.1)))
        c = float(np.clip(c, mu_c - 2.0 * std_c, mu_c + 2.0 * std_c))

        return max(a, 10.0), max(b, 10.0), max(c, 10.0)

    def sample_landmarks(self) -> dict[str, dict[str, float]]:
        """Step 2 — Sample Landmark (u, v, offset) positions using 18-D SVD model (Design Doc §6.3)."""
        lm_stats_meta = self.population_stats.get("landmarks_level_2", {})
        mean_18d = np.asarray(lm_stats_meta.get("mean_18d", [0.0] * 18), dtype=float)
        sing_vals = np.asarray(lm_stats_meta.get("singular_values", []), dtype=float)
        components = np.asarray(lm_stats_meta.get("components", []), dtype=float)

        if len(sing_vals) > 0 and len(components) > 0:
            n_samples = max(self.shape_model.n_samples, 2)
            mode_stds = sing_vals / math.sqrt(n_samples - 1)
            c_coeffs = np.zeros(len(sing_vals), dtype=float)
            for k in range(len(sing_vals)):
                val = float(self.rng.normal(0.0, mode_stds[k]))
                c_coeffs[k] = float(np.clip(val, -1.25 * mode_stds[k], 1.25 * mode_stds[k]))

            syn_18d = mean_18d + c_coeffs @ components
        else:
            std_18d = lm_stats_meta.get("std_18d", [0.1] * 18)
            syn_18d = np.zeros(18, dtype=float)
            for i in range(18):
                syn_18d[i] = float(self.rng.normal(mean_18d[i], max(float(std_18d[i]) * 0.5, 0.01)))

        landmarks = {}
        for idx, lm_name in enumerate(LM_NAMES_ORDER):
            base_i = idx * 3
            u = float(np.clip(syn_18d[base_i], -np.pi, np.pi))
            v = float(np.clip(syn_18d[base_i + 1], 0.01, np.pi - 0.01))
            off = float(syn_18d[base_i + 2])

            landmarks[lm_name] = {"u": u, "v": v, "offset": off}

        return landmarks

    def sample_ssm_deviations(self) -> tuple[np.ndarray, np.ndarray]:
        """Step 7a — Sample 34 PCA mode coefficients b_j ~ N(0, sqrt(lambda_j)) clipped at ±2σ and synthesize 126-D deviation vector."""
        k_modes = self.shape_model.k_retained
        eigenvalues = self.shape_model.eigenvalues_retained

        b_coeffs = np.zeros(k_modes, dtype=float)
        for j in range(k_modes):
            std_j = math.sqrt(max(float(eigenvalues[j]), 0.0))
            val = float(self.rng.normal(0.0, std_j))
            b_coeffs[j] = float(np.clip(val, -2.0 * std_j, 2.0 * std_j))

        syn_126d = self.shape_model.decode(b_coeffs)
        return b_coeffs, syn_126d

    def generate_single_tree(self) -> dict[str, Any]:
        """Execute Steps 1 through 10 to construct a single candidate synthetic 3D coronary tree."""
        # Step 1: Sample Ellipsoid parameters
        a, b, c = self.sample_ellipsoid_scaffold()

        # Step 2: Sample Landmark (u, v, offset) positions
        landmarks = self.sample_landmarks()

        # Step 3, 4, 5, 6: Generate Vessel Paths in (u, v) + Tortuosity + Obliquity + B-splines
        tort_stats = self.population_stats.get("tortuosity_obliquity_level_4", {}).get("tortuosity", {})
        obliq_stats = self.population_stats.get("tortuosity_obliquity_level_4", {}).get("obliquity", {})

        vessel_uv_paths = {}
        vessel_ctrl_uv = {}

        # Vessel connections:
        # LMCA: lca_ostium -> bifurcation
        # LAD: bifurcation -> lad_endpoint
        # LCX: bifurcation -> lcx_endpoint
        # RCA: rca_ostium -> rca_endpoint
        conn_map = {
            "LMCA": (("lca_ostium", "u"), ("lca_ostium", "v"), ("bifurcation", "u"), ("bifurcation", "v")),
            "LAD": (("bifurcation", "u"), ("bifurcation", "v"), ("lad_endpoint", "u"), ("lad_endpoint", "v")),
            "LCX": (("bifurcation", "u"), ("bifurcation", "v"), ("lcx_endpoint", "u"), ("lcx_endpoint", "v")),
            "RCA": (("rca_ostium", "u"), ("rca_ostium", "v"), ("rca_endpoint", "u"), ("rca_endpoint", "v")),
        }

        for vname, (s_u_k, s_v_k, e_u_k, e_v_k) in conn_map.items():
            start_uv = (landmarks[s_u_k[0]][s_u_k[1]], landmarks[s_v_k[0]][s_v_k[1]])
            end_uv = (landmarks[e_u_k[0]][e_u_k[1]], landmarks[e_v_k[0]][e_v_k[1]])
            n_ctrl = CONTROL_POINT_COUNTS[vname]

            u_c, v_c, u_p, v_p = generate_vessel_uv(
                vessel_name=vname,
                start_uv=start_uv,
                end_uv=end_uv,
                num_ctrl_pts=n_ctrl,
                tort_stats=tort_stats,
                obliquity_stats=obliq_stats,
                rng=self.rng,
                num_eval=n_ctrl,
            )
            vessel_ctrl_uv[vname] = (u_c, v_c)
            vessel_uv_paths[vname] = (u_p, v_p)

        # Step 7: Add Deviation Noise from Joint PCA (126-D SSM vector)
        b_coeffs, syn_126d = self.sample_ssm_deviations()

        # De-concatenate 126-D vector into per-vessel control point deviations [dev_x, dev_y, dev_z]
        vessel_3d = {}
        for vname in ("RCA", "LMCA", "LAD", "LCX"):
            mapping = DIMENSION_MAPPING[vname]
            start_idx, end_idx = mapping["start_idx"], mapping["end_idx"]
            v_devs_1d = syn_126d[start_idx:end_idx]
            v_devs_3d = v_devs_1d.reshape(-1, 3)  # (N_ctrl, 3)

            u_p, v_p = vessel_uv_paths[vname]
            n_pts = len(u_p)
            pts_3d = np.zeros((n_pts, 3), dtype=float)

            for i in range(n_pts):
                u_i, v_i = u_p[i], v_p[i]
                surf = ellipsoid_point(u_i, v_i, a, b, c)
                tang_u = ellipsoid_tangent_u(u_i, v_i, a, b, c)
                tang_v = ellipsoid_tangent_v(u_i, v_i, a, b, c)
                normal = ellipsoid_normal(u_i, v_i, a, b, c)

                dx, dy, dz = v_devs_3d[i]
                pts_3d[i] = surf + dx * tang_u + dy * tang_v + dz * normal

            vessel_3d[vname] = pts_3d

        # Step 8: Apply Ostial Offsets (Decaying Off-Surface Offset, §6.6)
        for vname, ost_name in [("LMCA", "lca_ostium"), ("RCA", "rca_ostium")]:
            ost_offset = landmarks[ost_name]["offset"]
            pts = vessel_3d[vname]
            n_pts = len(pts)
            if n_pts >= 2:
                u_p, v_p = vessel_uv_paths[vname]
                t_arr = np.linspace(0.0, 1.0, n_pts)
                decay_offsets = ost_offset * (1.0 - t_arr)

                for i in range(n_pts):
                    normal = ellipsoid_normal(u_p[i], v_p[i], a, b, c)
                    pts[i] += decay_offsets[i] * normal

                vessel_3d[vname] = pts

        # Step 9: Snap LMCA Bifurcation (LAD[0] = LCX[0] = LMCA[-1], §6.7)
        bif_point = vessel_3d["LMCA"][-1].copy()
        vessel_3d["LAD"][0] = bif_point
        vessel_3d["LCX"][0] = bif_point

        # Step 10: Generate Side Branches as Surface Random Walks (§6.8)
        sb_stats = self.population_stats.get("side_branches_level_5", {})
        side_branches = []

        for parent_name in ["LAD", "LCX", "RCA"]:
            p_sb_info = sb_stats.get(parent_name, {})
            c_info = p_sb_info.get("branch_count", {})
            mean_c = c_info.get("mean", 1.0)
            if mean_c is None or math.isnan(mean_c):
                mean_c = 1.0

            n_sb = int(self.rng.poisson(max(mean_c, 0.1)))
            u_p, v_p = vessel_uv_paths[parent_name]
            n_parent_pts = len(u_p)

            for sb_i in range(n_sb):
                t_attach = float(self.rng.uniform(0.10, 0.85))
                idx_attach = int(t_attach * (n_parent_pts - 1))

                u0, v0 = u_p[idx_attach], v_p[idx_attach]

                # Sample branch length (mm)
                l_info = p_sb_info.get("branch_length_mm", {})
                l_mean = l_info.get("mean", 15.0)
                if l_mean is None or math.isnan(l_mean):
                    l_mean = 15.0
                branch_len_mm = max(float(self.rng.gamma(2.0, max(l_mean / 2.0, 1.0))), 5.0)

                # Departure angle and side
                angle_rad = float(self.rng.uniform(30.0, 90.0) * math.pi / 180.0)
                direction = float(self.rng.choice([-1.0, 1.0]))

                # Surface random walk
                n_sb_pts = max(5, int(branch_len_mm / 2.0))
                u_sb = [u0]
                v_sb = [v0]

                du = direction * math.sin(angle_rad) * 0.05
                dv = math.cos(angle_rad) * 0.05
                curvature = float(self.rng.uniform(-0.3, 0.3))

                for step_i in range(1, n_sb_pts):
                    turn = curvature * step_i / n_sb_pts
                    du_i = du * math.cos(turn) - dv * math.sin(turn)
                    dv_i = du * math.sin(turn) + dv * math.cos(turn)
                    u_new = u_sb[-1] + du_i
                    v_new = float(np.clip(v_sb[-1] + dv_i, 0.01, math.pi - 0.01))
                    u_sb.append(u_new)
                    v_sb.append(v_new)

                # Evaluate 3D surface points + small noise
                sb_pts_3d = np.zeros((n_sb_pts, 3), dtype=float)
                for step_i in range(n_sb_pts):
                    surf = ellipsoid_point(u_sb[step_i], v_sb[step_i], a, b, c)
                    noise = self.rng.normal(0.0, 0.05, size=3)
                    sb_pts_3d[step_i] = surf + noise

                side_branches.append({
                    "parent": parent_name,
                    "attachment_index": idx_attach,
                    "points": sb_pts_3d,
                })

        return {
            "ellipsoid_params": {"a_mm": a, "b_mm": b, "c_mm": c},
            "landmarks": landmarks,
            "pca_coefficients_b": b_coeffs.tolist(),
            "shape_vector_126d": syn_126d.tolist(),
            "vessels_3d": vessel_3d,
            "side_branches": side_branches,
        }

    def generate_valid_tree(self, max_attempts: int = 100) -> tuple[dict[str, Any], int, list[str]]:
        """Steps 11 & 12 — Generate candidate tree and validate against Level 6 bounds; retry if invalid up to max_attempts."""
        attempts = 0
        all_rejection_reasons = []

        while attempts < max_attempts:
            attempts += 1
            tree_data = self.generate_single_tree()

            # Step 11: Validate against population thresholds (including 1.0 mm self-intersection)
            is_valid, errors = validate_synthetic_tree(
                vessels=tree_data["vessels_3d"],
                side_branches=tree_data["side_branches"],
                thresholds=self.validation_thresholds,
                min_dist_threshold_mm=1.0,
            )

            if is_valid:
                tree_data["attempts_required"] = attempts
                tree_data["validation"] = {"passed": True, "errors": []}
                return tree_data, attempts, []

            all_rejection_reasons.extend(errors)

        raise RuntimeError(f"Failed to generate valid synthetic tree after {max_attempts} attempts. Reasons: {all_rejection_reasons[:5]}")
