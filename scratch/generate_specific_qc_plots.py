"""Generate QC plots for 1 successful (1.label), 1 RCA-unresolved (82.label), and 1 rejected case (103.label)."""

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pca_ssm_vessel_tree_generator"))

from pca_ssm_vessel_tree_generator.extraction.nifti_loader import load_centerline_archive
from pca_ssm_vessel_tree_generator.extraction.skeleton_graph import build_graph_from_summary
from pca_ssm_vessel_tree_generator.extraction.batch1_qc import save_batch1_qc_plot
from run_batch1_extraction import process_patient_archive

output_dir = PROJECT_ROOT / "outputs" / "batch1_extracted"
centerlines_dir = PROJECT_ROOT / "lca_ssm" / "centerlines"

# 1. Process 1.label (Successful case)
process_patient_archive(
    case_dir=centerlines_dir / "1.label",
    patient_id="1.label",
    output_dir=output_dir,
    generate_qc_plots=True
)

# 2. Process 82.label (RCA-unresolved case)
process_patient_archive(
    case_dir=centerlines_dir / "82.label",
    patient_id="82.label",
    output_dir=output_dir,
    generate_qc_plots=True
)

# 3. Process 103.label (Rejected case)
case_dir = centerlines_dir / "103.label"
archive = load_centerline_archive(case_dir)
G, node_coords, endpoint_nodes, endpoint_coords, junction_nodes = build_graph_from_summary(
    archive["summary_df"], archive["branches_dict"]
)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

for u, v, d in G.edges(data=True):
    c_u = G.nodes[u]["coord"]
    c_v = G.nodes[v]["coord"]
    ax.plot([c_u[0], c_v[0]], [c_u[1], c_v[1]], [c_u[2], c_v[2]], color="#888888", alpha=0.6, linewidth=1.2)

if len(endpoint_coords) > 0:
    ax.scatter(endpoint_coords[:, 0], endpoint_coords[:, 1], endpoint_coords[:, 2], color="#D62828", s=30, label="Degree-1 Endpoints")
j_coords = node_coords[junction_nodes]
if len(j_coords) > 0:
    ax.scatter(j_coords[:, 0], j_coords[:, 1], j_coords[:, 2], color="#2CA02C", s=40, label="Degree-3+ Junctions")

ax.set_title("Patient 103.label — REJECTED (No Valid Downstream Bifurcation Junction)", fontsize=11, fontweight="bold")
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
ax.legend(loc="upper right")
fig.tight_layout()

rej_path = output_dir / "103.label" / "qc_visualization.png"
rej_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(rej_path, dpi=180)
plt.close(fig)

print("Generated specific QC plots for 1.label, 82.label, and 103.label")
