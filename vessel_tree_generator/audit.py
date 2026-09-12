"""Independent quantitative audit for exported 4D LCA research cases.

The audit intentionally separates software correctness, anatomical plausibility,
and clinical validation.  Passing this module never means that a synthetic case
is clinically validated or suitable for patient care.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BRANCHES = ("LMCA", "LAD", "LCX")
COLORS = {"LMCA": "#252525", "LAD": "#d93025", "LCX": "#1a73e8"}

PRIMARY_REFERENCES = [
    {
        "topic": "human LCA length and diameter morphometry",
        "citation": "Manpoong et al., Detailed Morphometric Analysis on Left Coronary Artery in the Population of North-East India, 2023",
        "url": "https://pubmed.ncbi.nlm.nih.gov/37829965/",
        "observations": {
            "sample_size": 100,
            "LMCA_length_mm": "9.13 +/- 3.23",
            "LMCA_diameter_mm": "4.38 +/- 0.58",
            "LAD_length_mm": "109.46 +/- 14.49",
            "LCX_length_mm": "66.27 +/- 11.56",
        },
    },
    {
        "topic": "left-main bifurcation anatomy",
        "citation": "Lujinovic et al., Morphometric analysis of clinically significant parameters of the main trunk of the left coronary artery, 2023",
        "url": "https://pubmed.ncbi.nlm.nih.gov/36944018/",
        "observations": {
            "sample_size": 29,
            "LMCA_length_mm": "median 9.0; range 6.0-13.5",
            "LAD_LCX_angle_deg": "median 89.0; interquartile range 74.5-93.0",
            "LMCA_LAD_angle_deg": "30.83 +/- 9.23",
        },
    },
    {
        "topic": "3D coronary bifurcation angles",
        "citation": "Pflederer et al., Measurement of coronary artery bifurcation angles by multidetector computed tomography, 2006",
        "url": "https://pubmed.ncbi.nlm.nih.gov/17035869/",
        "observations": {"sample_size": 100, "LAD_LCX_angle_deg": "80 +/- 27"},
    },
    {
        "topic": "normal coronary diameters",
        "citation": "Paul et al., A milestone in prediction of the coronary artery dimensions, 2019",
        "url": "https://pubmed.ncbi.nlm.nih.gov/31779861/",
        "observations": {
            "sample_size": 925,
            "LMCA_diameter_mm": "4.18 +/- 0.65",
            "LAD_diameter_mm": "3.22 +/- 0.63",
            "LCX_diameter_mm": "3.07 +/- 0.65",
        },
    },
    {
        "topic": "coronary displacement over the cardiac cycle",
        "citation": "Tan et al., Estimation of the displacement of cardiac substructures and the motion of the coronary arteries using ECG gating, 2013",
        "url": "https://pubmed.ncbi.nlm.nih.gov/24098082/",
        "observations": {
            "coronary_AP_displacement_mm": "2.8-5.9 average",
            "coronary_LR_displacement_mm": "3.5-6.6 average",
            "coronary_SI_displacement_mm": "3.8-5.3 average",
            "summary": "most average 3D displacements were 3-8 mm",
        },
    },
    {
        "topic": "normal and plaque-segment cyclic lumen-area change",
        "citation": "Ge et al., Intravascular ultrasound imaging of angiographically normal coronary arteries, 1994",
        "url": "https://pubmed.ncbi.nlm.nih.gov/8043342/",
        "observations": {
            "normal_area_change": "LMCA 10.2%, proximal LAD 8.3%, mid LAD 9.8%",
            "plaque_segment_area_change": "5.8%",
            "timing_note": "maximum lumen area occurred in early diastole in this cohort",
        },
    },
    {
        "topic": "coronary distensibility and plaque",
        "citation": "Nakatani et al., Assessment of coronary artery distensibility by intravascular ultrasound, 1995",
        "url": "https://pubmed.ncbi.nlm.nih.gov/7796499/",
        "observations": {
            "mean_systolic_lumen_area_mm2": "12.6 +/- 5.0",
            "mean_diastolic_lumen_area_mm2": "11.6 +/- 4.6",
            "interpretation": "atherosclerotic change was associated with impaired distensibility",
        },
    },
    {
        "topic": "reduced diameter pulsatility in CAD",
        "citation": "Numao et al., Pulsatile diameter change of coronary artery lumen estimated by IVUS, 1997",
        "url": "https://pubmed.ncbi.nlm.nih.gov/9253689/",
        "observations": {
            "control_mean_diameter_change_mm": "0.13 +/- 0.12",
            "control_end_diastolic_diameter_mm": "4.52 +/- 0.51",
            "CAD_mean_diameter_change_mm": "0.05 +/- 0.18",
            "CAD_end_diastolic_diameter_mm": "4.53 +/- 0.69",
        },
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _arc_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _normalized_arc(points: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(seg)))
    return cumulative / cumulative[-1] if cumulative[-1] > 0 else cumulative


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1.0e-12:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / denom, -1.0, 1.0))))


def _robust_tangent(points: np.ndarray, *, at_end: bool, fraction: float = 0.10) -> np.ndarray:
    step = max(2, min(len(points) - 1, int(round(fraction * (len(points) - 1)))))
    return points[-1] - points[-1 - step] if at_end else points[step] - points[0]


def _plane_metrics(points: np.ndarray) -> dict[str, Any]:
    centered = points - np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    residual = np.abs(centered @ normal)
    return {
        "normal": normal.tolist(),
        "rms_residual_mm": float(np.sqrt(np.mean(residual**2))),
        "maximum_residual_mm": float(np.max(residual)),
    }


def _segment_distance(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> float:
    """Minimum distance between two closed 3D line segments."""
    u, v, w = p1 - p0, q1 - q0, p0 - q0
    a, b, c = float(u @ u), float(u @ v), float(v @ v)
    d, e = float(u @ w), float(v @ w)
    denom = a * c - b * b
    if a <= 1.0e-15 and c <= 1.0e-15:
        return float(np.linalg.norm(p0 - q0))
    if a <= 1.0e-15:
        s, t = 0.0, np.clip(e / c, 0.0, 1.0)
    elif c <= 1.0e-15:
        s, t = np.clip(-d / a, 0.0, 1.0), 0.0
    else:
        s = np.clip((b * e - c * d) / denom, 0.0, 1.0) if abs(denom) > 1.0e-15 else 0.0
        t = np.clip((b * s + e) / c, 0.0, 1.0)
        s = np.clip((b * t - d) / a, 0.0, 1.0)
    return float(np.linalg.norm(w + s * u - t * v))


def _minimum_unintended_segment_distance(branches: dict[str, np.ndarray]) -> dict[str, Any]:
    best = {"distance_mm": float("inf"), "segments": None}
    for branch, values in branches.items():
        xyz = values[:, :3]
        for i in range(len(xyz) - 1):
            for j in range(i + 3, len(xyz) - 1):
                distance = _segment_distance(xyz[i], xyz[i + 1], xyz[j], xyz[j + 1])
                if distance < best["distance_mm"]:
                    best = {"distance_mm": distance, "segments": [branch, i, branch, j]}
    pairs = (("LMCA", "LAD"), ("LMCA", "LCX"), ("LAD", "LCX"))
    for left, right in pairs:
        a, b = branches[left][:, :3], branches[right][:, :3]
        for i in range(len(a) - 1):
            for j in range(len(b) - 1):
                connected = left == "LMCA" and i >= len(a) - 3 and j <= 1
                daughter_origin = left == "LAD" and right == "LCX" and i <= 1 and j <= 1
                if connected or daughter_origin:
                    continue
                distance = _segment_distance(a[i], a[i + 1], b[j], b[j + 1])
                if distance < best["distance_mm"]:
                    best = {"distance_mm": distance, "segments": [left, i, right, j]}
    return best


def _branch_geometry(values: np.ndarray) -> dict[str, Any]:
    xyz = values[:, :3]
    segment = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    length = float(segment.sum())
    chord = float(np.linalg.norm(xyz[-1] - xyz[0]))
    vectors = np.diff(xyz, axis=0)
    norms = np.linalg.norm(vectors, axis=1)
    unit = vectors / np.maximum(norms[:, None], 1.0e-12)
    turns = np.degrees(np.arccos(np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1.0, 1.0)))
    plane = _plane_metrics(xyz)
    return {
        "point_count": len(values),
        "length_mm": length,
        "chord_mm": chord,
        "distance_metric_arc_over_chord": length / chord if chord > 0 else None,
        "total_turning_angle_deg": float(np.sum(turns)),
        "maximum_local_turn_deg": float(np.max(turns)) if len(turns) else 0.0,
        "proximal_diameter_mm": float(2.0 * values[0, 3]),
        "midpoint_diameter_mm": float(2.0 * values[len(values) // 2, 3]),
        "distal_diameter_mm": float(2.0 * values[-1, 3]),
        "inferior_displacement_mm": float(xyz[0, 2] - xyz[-1, 2]),
        "terminal_xy_displacement_mm": float(np.linalg.norm(xyz[-1, :2] - xyz[0, :2])),
        "plane": plane,
    }


def _disease_metrics(
    healthy: np.ndarray, static: np.ndarray, reduction: np.ndarray
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for index, name in enumerate(BRANCHES):
        s = _normalized_arc(healthy[index, :, :3])
        profile = reduction[index]
        mask = profile > 1.0e-6
        components = 0
        if np.any(mask):
            components = int(mask[0]) + int(np.sum((~mask[:-1]) & mask[1:]))
        maximum = float(np.max(profile))
        results[name] = {
            "maximum_radius_and_diameter_reduction_fraction": maximum,
            "maximum_equivalent_circular_area_reduction_fraction": 1.0 - (1.0 - maximum) ** 2,
            "minimum_lumen_diameter_mm": float(2.0 * np.min(static[index, :, 3])),
            "affected_export_point_count": int(np.sum(mask)),
            "affected_component_count": components,
            "affected_normalized_arc_range": [float(np.min(s[mask])), float(np.max(s[mask]))] if np.any(mask) else None,
            "centerline_change_caused_by_disease_mm": float(np.max(np.linalg.norm(
                static[index, :, :3] - healthy[index, :, :3], axis=1
            ))),
        }
    return results


def _motion_and_pulsatility(
    cine: np.ndarray, static: np.ndarray, reduction: np.ndarray, phases: np.ndarray
) -> tuple[dict[str, Any], dict[str, Any]]:
    displacement = np.linalg.norm(cine[..., :3] - static[None, ..., :3], axis=-1)
    radius_change = cine[..., 3] / static[None, ..., 3] - 1.0
    peak_index = int(np.argmax(np.max(np.abs(radius_change), axis=(1, 2))))
    motion: dict[str, Any] = {
        "frame_count": int(len(cine)),
        "peak_sample_index": peak_index,
        "peak_sample_phase": float(phases[peak_index]),
        "maximum_global_displacement_mm": float(np.max(displacement)),
        "mean_global_displacement_at_peak_mm": float(np.mean(displacement[peak_index])),
        "cycle_closing_error_mm": float(np.max(np.linalg.norm(cine[-1, ..., :3] - cine[0, ..., :3], axis=-1))),
        "branches": {},
    }
    pulsatility: dict[str, Any] = {
        "peak_sample_index": peak_index,
        "peak_sample_phase": float(phases[peak_index]),
        "global_maximum_absolute_radius_change_fraction": float(np.max(np.abs(radius_change))),
        "global_mean_absolute_radius_change_at_peak_fraction": float(np.mean(np.abs(radius_change[peak_index]))),
        "peak_change_sign": "expansion" if float(np.mean(radius_change[peak_index])) >= 0 else "contraction",
        "branches": {},
    }
    for index, name in enumerate(BRANCHES):
        branch_disp = displacement[:, index]
        motion["branches"][name] = {
            "maximum_displacement_mm": float(np.max(branch_disp)),
            "mean_displacement_at_peak_mm": float(np.mean(branch_disp[peak_index])),
            "terminal_displacement_at_peak_mm": float(branch_disp[peak_index, -1]),
        }
        peak_profile = np.abs(radius_change[peak_index, index])
        lesion = reduction[index] > 1.0e-6
        max_lesion_index = int(np.argmax(reduction[index]))
        nonlesion = ~lesion
        healthy_level = float(np.max(peak_profile[nonlesion])) if np.any(nonlesion) else None
        lesion_level = float(peak_profile[max_lesion_index]) if np.any(lesion) else None
        pulsatility["branches"][name] = {
            "maximum_absolute_change_fraction": float(np.max(peak_profile)),
            "mean_absolute_change_fraction": float(np.mean(peak_profile)),
            "maximum_nonlesion_change_fraction": healthy_level,
            "change_at_maximum_lesion_fraction": lesion_level,
            "lesion_to_nonlesion_amplitude_ratio": (
                lesion_level / healthy_level if lesion_level is not None and healthy_level else None
            ),
        }
    return motion, pulsatility


def _make_quantitative_plot(
    output: Path,
    case_id: str,
    cine: np.ndarray,
    static: np.ndarray,
    reduction: np.ndarray,
    phases: np.ndarray,
    audit: dict[str, Any],
) -> None:
    displacement = np.linalg.norm(cine[..., :3] - static[None, ..., :3], axis=-1)
    radius_change = 100.0 * (cine[..., 3] / static[None, ..., 3] - 1.0)
    peak = audit["pulsatility"]["peak_sample_index"]
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    for index, name in enumerate(BRANCHES):
        axes[0, 0].plot(phases, np.max(displacement[:, index], axis=1), "o-", color=COLORS[name], label=name)
        axes[0, 1].plot(phases, np.max(radius_change[:, index], axis=1), "o-", color=COLORS[name], label=name)
        arc = _normalized_arc(static[index, :, :3])
        axes[1, 0].plot(arc, radius_change[peak, index], color=COLORS[name], lw=2.2, label=name)
        axes[1, 1].plot(arc, 100.0 * reduction[index], color=COLORS[name], lw=2.2, label=name)
    axes[0, 0].set(title="Global 4D motion by branch", xlabel="Normalized cardiac phase", ylabel="Maximum displacement (mm)")
    axes[0, 1].set(title="Global radius pulsatility by branch", xlabel="Normalized cardiac phase", ylabel="Maximum radius change (%)")
    axes[1, 0].set(title=f"Local pulsatility profile at sampled peak (phase {phases[peak]:.3f})", xlabel="Normalized branch arc length", ylabel="Local radius change (%)")
    axes[1, 1].set(title="Disease profile (diameter/radius reduction)", xlabel="Normalized branch arc length", ylabel="Reduction (%)")
    axes[1, 0].set_ylim(0.0, max(3.2, 1.1 * float(np.max(np.abs(radius_change[peak])))))
    axes[1, 1].set_ylim(0.0, max(5.0, 110.0 * float(np.max(reduction))))
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(f"{case_id} — quantitative motion, pulsatility and lesion audit", fontsize=16)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def audit_exported_case(case_dir: str | Path, *, write_outputs: bool = True) -> dict[str, Any]:
    """Audit one exported case and optionally write JSON plus a diagnostic PNG."""
    case_dir = Path(case_dir).resolve()
    metadata = _load_json(case_dir / "metadata.json")
    manifest = _load_json(case_dir / "manifest.json")
    static = np.load(case_dir / "geometry_static.npy", allow_pickle=False)
    healthy = np.load(case_dir / "geometry_healthy_reference.npy", allow_pickle=False)
    cine = np.load(case_dir / "geometry_cine.npy", allow_pickle=False)
    reduction = np.load(case_dir / "disease_reduction.npy", allow_pickle=False)
    phases = np.asarray(metadata["phase_values"], dtype=float)
    branches = {name: static[index] for index, name in enumerate(BRANCHES)}

    geometry = {name: _branch_geometry(branches[name]) for name in BRANCHES}
    geometry["angles_deg"] = {
        "LMCA_to_LAD": _angle_deg(
            _robust_tangent(branches["LMCA"][:, :3], at_end=True),
            _robust_tangent(branches["LAD"][:, :3], at_end=False),
        ),
        "LMCA_to_LCX": _angle_deg(
            _robust_tangent(branches["LMCA"][:, :3], at_end=True),
            _robust_tangent(branches["LCX"][:, :3], at_end=False),
        ),
        "LAD_to_LCX": _angle_deg(
            _robust_tangent(branches["LAD"][:, :3], at_end=False),
            _robust_tangent(branches["LCX"][:, :3], at_end=False),
        ),
    }
    lad_normal = np.asarray(geometry["LAD"]["plane"]["normal"])
    lcx_normal = np.asarray(geometry["LCX"]["plane"]["normal"])
    plane_angle = _angle_deg(lad_normal, lcx_normal)
    geometry["LAD_LCX_plane_acute_angle_deg"] = min(plane_angle, 180.0 - plane_angle)
    geometry["minimum_unintended_segment_distance"] = _minimum_unintended_segment_distance(branches)

    role_source = metadata["provenance"]["static_validation"]["metrics"].get("anatomical_direction", {})
    roles = {
        "LAD_more_inferior_than_LCX": geometry["LAD"]["inferior_displacement_mm"] > geometry["LCX"]["inferior_displacement_mm"],
        "LAD_is_longer_than_LCX": geometry["LAD"]["length_mm"] > geometry["LCX"]["length_mm"],
        "LAD_initial_course_is_inferior": float(role_source.get("LAD_initial_inferior_direction_fraction", 0.0)) >= 0.5,
        "LCX_initial_course_is_horizontal": float(role_source.get("LCX_initial_horizontal_direction_fraction", 0.0)) >= 0.5,
        "LCX_more_circumferential_by_terminal_fraction": float(role_source.get("LCX_terminal_lateral_fraction", 0.0)) > float(role_source.get("LAD_terminal_lateral_fraction", 0.0)),
        "source_assignment_gate_accepted": bool(metadata["provenance"]["static_validation"].get("accepted", False)),
    }

    disease = _disease_metrics(healthy, static, reduction)
    motion, pulsatility = _motion_and_pulsatility(cine, static, reduction, phases)
    topology_error = max(
        float(np.max(np.linalg.norm(cine[:, 0, -1, :3] - cine[:, 1, 0, :3], axis=1))),
        float(np.max(np.linalg.norm(cine[:, 0, -1, :3] - cine[:, 2, 0, :3], axis=1))),
    )

    literature_checks = {
        "LMCA_length_in_6_to_13_5_mm_cadaver_range": 6.0 <= geometry["LMCA"]["length_mm"] <= 13.5,
        "LAD_LCX_angle_in_MDCT_mean_plus_minus_2SD": 26.0 <= geometry["angles_deg"]["LAD_to_LCX"] <= 134.0,
        "proximal_LMCA_diameter_in_Paul_mean_plus_minus_2SD": 2.88 <= geometry["LMCA"]["proximal_diameter_mm"] <= 5.48,
        "proximal_LAD_diameter_in_Paul_mean_plus_minus_2SD": 1.96 <= geometry["LAD"]["proximal_diameter_mm"] <= 4.48,
        "proximal_LCX_diameter_in_Paul_mean_plus_minus_2SD": 1.77 <= geometry["LCX"]["proximal_diameter_mm"] <= 4.37,
        "maximum_motion_within_3_to_8_mm_average_context": motion["maximum_global_displacement_mm"] <= 10.0,
        "healthy_radius_pulsatility_consistent_with_rough_3_percent_diameter_change": pulsatility["global_maximum_absolute_radius_change_fraction"] <= 0.05,
    }
    warnings: list[str] = []
    if not all(roles.values()):
        warnings.append("one or more major-vessel anatomical role checks failed")
    failed_literature = [key for key, passed in literature_checks.items() if not passed]
    if failed_literature:
        warnings.append("descriptive external-reference checks outside envelope: " + ", ".join(failed_literature))
    if motion["maximum_global_displacement_mm"] > 8.0:
        warnings.append("maximum point displacement exceeds the 3-8 mm average-motion context; maxima and averages are not equivalent")
    if geometry["minimum_unintended_segment_distance"]["distance_mm"] < 0.25:
        warnings.append("an unintended segment pair is closer than 0.25 mm and needs visual review")

    engineering_pass = bool(
        manifest.get("status") == "PASS"
        and metadata["validation"].get("is_valid")
        and topology_error <= 1.0e-9
        and motion["cycle_closing_error_mm"] <= 1.0e-9
    )
    anatomical_pass = bool(all(roles.values()) and metadata["provenance"]["static_validation"].get("accepted"))
    overall_status = (
        "FAIL"
        if not (engineering_pass and anatomical_pass)
        else "PASS_WITH_CAVEATS"
        if warnings
        else "PASS"
    )
    audit: dict[str, Any] = {
        "schema_version": "1.0.0",
        "case_id": metadata["case_id"],
        "model_scope": metadata["model_scope"],
        "overall_status": overall_status,
        "engineering_validation_pass": engineering_pass,
        "major_vessel_LCA_anatomical_plausibility_pass": anatomical_pass,
        "clinical_validation": False,
        "anatomical_completeness": {
            "complete_coronary_tree": False,
            "included": list(BRANCHES),
            "not_included": ["RCA", "diagonal branches", "obtuse marginal branches", "septal branches", "PDA/PL branches"],
            "statement": "This is a major-vessel LCA scaffold, not a complete human coronary tree.",
        },
        "warnings": warnings,
        "geometry": geometry,
        "anatomical_roles": {"checks": roles, "source_metrics": role_source},
        "disease": disease,
        "motion": motion,
        "pulsatility": pulsatility,
        "topology": {
            "maximum_bifurcation_error_all_phases_mm": topology_error,
            "exact_shared_bifurcation": topology_error <= 1.0e-9,
        },
        "external_reference_context": {
            "checks": literature_checks,
            "interpretation": "Reference ranges are descriptive cross-study sanity checks, not clinical acceptance limits.",
            "primary_references": PRIMARY_REFERENCES,
        },
        "interpretation_limits": [
            "The statistical anatomy is population-derived only for LMCA/LAD/LCX.",
            "The radii, disease profiles, motion amplitudes and compliance factor are mechanistic research settings, not patient-estimated parameters.",
            "The cyclic radius model is scalar and cannot represent anisodiametric wall motion.",
            "No flow, pressure, myocardium, vessel wall, side branches or patient outcome is simulated.",
        ],
    }
    if write_outputs:
        (case_dir / "quantitative_validation.json").write_text(
            json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        visual_dir = case_dir / "visualizations"
        visual_dir.mkdir(exist_ok=True)
        _make_quantitative_plot(
            visual_dir / "quantitative_motion_pulsatility.png",
            metadata["case_id"], cine, static, reduction, phases, audit,
        )
    return audit


def audit_demo_release(root: str | Path, *, write_outputs: bool = True) -> dict[str, Any]:
    root = Path(root).resolve()
    cases = {}
    for name in ("healthy", "focal_lad", "diffuse_lcx", "tandem_lad"):
        cases[name] = audit_exported_case(root / name, write_outputs=write_outputs)
    case_statuses = [item["overall_status"] for item in cases.values()]
    release_status = (
        "FAIL"
        if not all(status.startswith("PASS") for status in case_statuses)
        else "PASS_WITH_CAVEATS"
        if any(status == "PASS_WITH_CAVEATS" for status in case_statuses)
        else "PASS"
    )
    summary = {
        "status": release_status,
        "case_count": len(cases),
        "shared_major_vessel_anatomy": True,
        "cases": {
            name: {
                "case_id": item["case_id"],
                "status": item["overall_status"],
                "engineering_validation_pass": item["engineering_validation_pass"],
                "major_vessel_LCA_anatomical_plausibility_pass": item["major_vessel_LCA_anatomical_plausibility_pass"],
                "warnings": item["warnings"],
            }
            for name, item in cases.items()
        },
    }
    if write_outputs:
        (root / "QUANTITATIVE_AUDIT_SUMMARY.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return summary
