#!/usr/bin/env python
# coding: utf-8

import nibabel as nib
import numpy as np
import napari
from skimage.morphology import skeletonize
from skan import Skeleton, summarize
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Configurable Path
PATIENT = "117.label"
path = BASE_DIR / "nii files" / f"{PATIENT}.nii.gz"

print(f"Processing patient {PATIENT} from {path}...")
img = nib.load(path)
mask = img.get_fdata() > 0
spacing = img.header.get_zooms()[:3]

# 1. Distance Transform
print("Computing distance transform...")
dist_map = distance_transform_edt(mask, sampling=spacing)

# 2. Skeletonize
print("Computing skeleton...")
skel = skeletonize(mask, method="lee")
sk = Skeleton(skel.astype(np.uint8), spacing=spacing)
df = summarize(sk, separator='_')

print("Columns in skan dataframe:", df.columns.tolist())

# 3. Extract Endpoints and Junctions
degrees = sk.degrees
node_coords = sk.coordinates * spacing

junction_indices = np.where(degrees >= 3)[0]
junction_coords = node_coords[junction_indices]

endpoint_indices = np.where(degrees == 1)[0]
endpoint_coords = node_coords[endpoint_indices]

# 4. Compute Radius at Endpoints
endpoint_voxels = np.round(sk.coordinates[endpoint_indices]).astype(int)
endpoint_voxels[:, 0] = np.clip(endpoint_voxels[:, 0], 0, mask.shape[0]-1)
endpoint_voxels[:, 1] = np.clip(endpoint_voxels[:, 1], 0, mask.shape[1]-1)
endpoint_voxels[:, 2] = np.clip(endpoint_voxels[:, 2], 0, mask.shape[2]-1)

endpoint_radii = dist_map[endpoint_voxels[:, 0], endpoint_voxels[:, 1], endpoint_voxels[:, 2]]

# 5. Candidate Root = Largest Radius
root_idx = np.argmax(endpoint_radii)
candidate_root_coord = endpoint_coords[root_idx].reshape(1, 3)
termination_coords = np.delete(endpoint_coords, root_idx, axis=0)

print(f"Found {len(junction_coords)} junctions.")
print(f"Found {len(endpoint_coords)} endpoints. Candidate root radius: {endpoint_radii[root_idx]:.2f}")

# 6. Save Landmarks
landmarks = {
    "patient": PATIENT,
    "spacing": spacing,
    "candidate_root_coord": candidate_root_coord,
    "candidate_root_node": int(endpoint_indices[root_idx]),
    "endpoint_coords": endpoint_coords,
    "endpoint_nodes": endpoint_indices,
    "junction_coords": junction_coords,
    "junction_nodes": junction_indices,
    "endpoint_radii": endpoint_radii,
    "termination_coords": termination_coords
}
np.save(BASE_DIR / "landmarks.npy", landmarks, allow_pickle=True)
print("Saved landmarks.npy")

df.to_csv(BASE_DIR / "summary.csv", index=False)
print("Saved summary.csv")

all_paths = [sk.path_coordinates(i) * spacing for i in range(sk.n_paths)]
all_paths_dict = {f"path_{i}": path for i, path in enumerate(all_paths)}
np.savez(BASE_DIR / "branches.npz", **all_paths_dict)
print("Saved branches.npz")

# 7. Matplotlib Visualization (Optional, keeping from original)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
for i in range(sk.n_paths):
    coords = sk.path_coordinates(i) * spacing
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color='red', linewidth=1)
# Add landmarks to matplotlib
ax.scatter(junction_coords[:, 0], junction_coords[:, 1], junction_coords[:, 2], color='yellow', s=20, label='Bifurcations')
ax.scatter(termination_coords[:, 0], termination_coords[:, 1], termination_coords[:, 2], color='blue', s=20, label='Terminations')
ax.scatter(candidate_root_coord[:, 0], candidate_root_coord[:, 1], candidate_root_coord[:, 2], color='green', s=50, label='Candidate Root')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.legend()
plt.show()

# 8. Napari Visualization
paths_table = df.copy()
paths_table['path_id'] = np.arange(sk.n_paths)
paths_table['branch_type'] = paths_table['branch_type'].astype(str)

viewer = napari.Viewer(ndisplay=3)
viewer.add_shapes(
    all_paths,
    shape_type='path',
    properties=paths_table,
    edge_width=0.5,
    edge_color='branch_distance',
    edge_colormap='viridis',
    name='Centerlines'
)
viewer.add_points(
    junction_coords,
    size=1.0,
    face_color='yellow',
    name='Bifurcations'
)
viewer.add_points(
    termination_coords,
    size=1.0,
    face_color='blue',
    name='Terminations'
)
viewer.add_points(
    candidate_root_coord,
    size=2.0,
    face_color='green',
    name='Candidate Root'
)
viewer.show()
napari.run()
