# Synthetic Coronary Tree Generator — Technical Design Document

## Overview

Generate anatomically plausible 3D coronary artery trees (LCA + RCA) with cardiac phase motion, constrained by an ellipsoidal heart surface learned from ~190 patient CT label datasets.

```
NIfTI labels (190 patients)
  → extract centerlines + landmarks
  → fit two planes + two ellipses per patient
  → normalize all patients into one cardiac frame
  → fit one ellipsoid (heart surface model)
  → learn population statistics (scaffold, landmarks, deviations, tortuosity)
  → generate synthetic trees on the ellipsoid surface
  → add cardiac motion by deforming the ellipsoid
  → validate and reject implausible samples
```

---

## Part 1: Per-Patient Data Extraction

### 1.1 Input

- NIfTI (`.nii` / `.nii.gz`) binary label masks of coronary arteries per patient.
- Labels contain both LCA and RCA trees.
- Each label is a 3D binary segmentation in scanner coordinates.

### 1.2 Centerline Extraction

**Goal:** Convert each binary label into an ordered set of 3D centerline points in physical mm coordinates.

**Steps:**

1. Load NIfTI with `nibabel` — extract the affine matrix.
2. Binarize the label (if not already binary).
3. Keep the largest connected component (remove noise).
4. Skeletonize using `skimage.morphology.skeletonize_3d`.
5. Convert skeleton voxels to a graph using `skan` or `networkx`:
   - Node = skeleton voxel (in physical mm via affine).
   - Edge = neighboring skeleton voxels.
   - Edge weight = Euclidean distance in mm.
6. Collapse chains of degree-2 nodes into graph edges (branch segments).
7. Identify:
   - **Endpoints:** degree 1.
   - **Junctions:** degree >= 3.

**Output per patient:**

```
skeleton_graph: networkx.Graph
  - nodes: 3D points in mm
  - edges: vessel segments with arc lengths
  - endpoints: list of 3D points
  - junctions: list of 3D points
```

### 1.3 Landmark Identification

**Goal:** Find the anatomically meaningful points on each patient's coronary tree.

#### Ostia (2 points)

The two ostia are the two endpoints closest to each other near the geometric center of the full tree. All other endpoints are distal branch tips, far apart. The ostia are the only endpoints that sit close together near the aortic root.

```
ostia = two endpoints with minimum mutual Euclidean distance
        AND both near the centroid of all endpoints
```

Label them:
- **LCA ostium:** the ostium whose tree contains the LMCA bifurcation.
- **RCA ostium:** the other ostium.

If ambiguous (rare), use manual verification.

#### LMCA Bifurcation (1 point)

1. From the LCA ostium, trace the shortest path through the skeleton graph.
2. Find the **first meaningful junction** with degree >= 3 downstream from the LCA ostium.
3. Ignore tiny spurs (branches shorter than a threshold, e.g., 3 mm).
4. That junction is the LMCA bifurcation.

#### LAD and LCX Branches

From the bifurcation, there are two daughter paths:

- **LAD:** the daughter that descends away from the AV groove plane (toward the apex).
- **LCX:** the daughter that stays near the AV groove plane (wraps laterally).

Classification rule (after fitting the coronary plane — see Part 2):

```
daughter_A_out_of_plane = |dot(daughter_A_direction, coronary_normal)|
daughter_B_out_of_plane = |dot(daughter_B_direction, coronary_normal)|

if A_out_of_plane > B_out_of_plane:
    LAD = daughter_A, LCX = daughter_B
else:
    LAD = daughter_B, LCX = daughter_A
```

#### Branch Endpoints

- **LAD endpoint:** the terminal endpoint of the LAD path (near apex).
- **LCX endpoint:** the terminal endpoint of the LCX path (distal circumflex).
- **RCA endpoint:** the terminal endpoint of the longest path from the RCA ostium.

#### RCA Main Branch

Trace the longest path from the RCA ostium through the skeleton graph. This is the RCA main branch.

**Output per patient:**

```
landmarks = {
    "lca_ostium": np.ndarray,      # (3,) mm
    "rca_ostium": np.ndarray,      # (3,) mm
    "bifurcation": np.ndarray,     # (3,) mm
    "lad_endpoint": np.ndarray,    # (3,) mm
    "lcx_endpoint": np.ndarray,    # (3,) mm
    "rca_endpoint": np.ndarray,    # (3,) mm
}
centerlines = {
    "RCA":  np.ndarray,   # (N_rca, 3) ordered from ostium to distal
    "LMCA": np.ndarray,   # (N_lmca, 3) ordered from ostium to bifurcation
    "LAD":  np.ndarray,   # (N_lad, 3) ordered from bifurcation to endpoint
    "LCX":  np.ndarray,   # (N_lcx, 3) ordered from bifurcation to endpoint
}
```

---

## Part 2: Per-Patient Plane Fitting and Cardiac Frame

### 2.1 Fit the Coronary Plane (Plane 1 — AV Groove)

The coronary plane is the AV groove plane, containing the RCA and LCX ring.

**Data:** Combine RCA and LCX centerline points.

**Method:** Least-squares plane fit via SVD.

```python
ring_points = np.vstack([rca_centerline, lcx_centerline])  # Nx3
centroid_cor = ring_points.mean(axis=0)
centered = ring_points - centroid_cor
_, _, Vt = np.linalg.svd(centered, full_matrices=False)
coronary_normal = Vt[2]  # direction of least variance = plane normal
```

**Result:** `coronary_centroid` and `coronary_normal`.

### 2.2 Fit the Interventricular Plane (Plane 2 — Median Plane)

The IV plane contains the LAD descent and is approximately perpendicular to the coronary plane.

**Data:** LAD centerline points + coronary normal.

**Method:**

```python
lad_vec = lad_centerline[-1] - lad_centerline[0]
lad_vec /= np.linalg.norm(lad_vec)

# IV plane normal = perpendicular to both coronary_normal and LAD direction
iv_normal = np.cross(coronary_normal, lad_vec)
iv_normal /= np.linalg.norm(iv_normal)
```

**Result:** `iv_normal`.

### 2.3 Build the Per-Patient Cardiac Frame

The two planes define a right-handed coordinate frame:

```python
# Axis 3 (z): coronary plane normal = base-to-apex direction
axis_z = coronary_normal

# Orient: LAD descends toward apex (negative z)
lad_direction = lad_centerline[-1] - lad_centerline[0]
if np.dot(lad_direction, axis_z) > 0:
    axis_z = -axis_z

# Axis 1 (x): IV plane normal = left-to-right direction
axis_x = iv_normal

# Axis 2 (y): anterior-posterior direction
axis_y = np.cross(axis_z, axis_x)
axis_y /= np.linalg.norm(axis_y)

# Re-derive axis_x to ensure orthogonality
axis_x = np.cross(axis_y, axis_z)
axis_x /= np.linalg.norm(axis_x)

# Origin: coronary plane centroid
origin = centroid_cor

# Rotation matrix: cardiac_frame → scanner_frame
R = np.stack([axis_x, axis_y, axis_z], axis=0)  # 3x3
```

**Per-patient cardiac frame:**

| Axis | Direction | Anatomical meaning |
|------|-----------|-------------------|
| x | IV plane normal | Left ↔ Right |
| y | Cross product | Anterior ↔ Posterior |
| z | Coronary plane normal | Base → Apex (negative = toward apex) |
| origin | Ring centroid | Cardiac center |

### 2.4 Transform All Data Into Cardiac Frame

```python
def to_cardiac_frame(points, origin, R):
    return (points - origin) @ R.T

# Transform all centerlines and landmarks
for name in centerlines:
    centerlines[name] = to_cardiac_frame(centerlines[name], origin, R)
for name in landmarks:
    landmarks[name] = to_cardiac_frame(landmarks[name][np.newaxis], origin, R)[0]
```

After this transform, all patients share the same coordinate convention:
- Ostia have `z > 0` (above the coronary plane, near base).
- LAD has `z` decreasing (descends toward apex).
- LCX and RCA have `z ≈ 0` (stay in the coronary plane).

---

## Part 3: From Per-Patient Frames to One Global Cardiac Frame

### 3.1 The Problem

Each patient has their own cardiac frame `(origin_i, R_i)`. To build a population statistical model, all patients must be in **one shared frame**.

The per-patient frames are already anatomically aligned (ostium at similar position, LMCA along similar direction, LAD descending in similar direction). But there is residual variation in:
- Origin position (ring centroid varies).
- Rotation around the z-axis (coronary plane rotation).
- Scale (heart size differs).

### 3.2 Step 1: Translate All Patients to Shared Origin

Set the global origin to the population mean of all per-patient origins:

```python
global_origin = np.mean([p["origin"] for p in patients], axis=0)
```

For each patient, re-express all points relative to the global origin:

```python
# No rotation change needed — just shift origin
offset = p["origin"] - global_origin
# All points in scanner frame: points_scanner = points_cardiac @ R.T + origin
# To rebase: subtract offset from origin, keep R the same
```

Since we work in cardiac frame coordinates (not scanner coordinates), the origin is already at `(0,0,0)` for each patient. The per-patient frames are already origin-aligned by construction.

### 3.3 Step 2: Resolve Residual Z-Axis Rotation

After per-patient frame alignment, the z-axis (base→apex) is consistent. But there is still a free rotation around z. Fix this using the ostia:

```python
# In each patient's cardiac frame, project LCA ostium onto the coronary plane (z=0)
lca_ost_xy = landmarks["lca_ostium"][:2]  # (x, y) components
lca_ost_angle = np.arctan2(lca_ost_xy[1], lca_ost_xy[0])

# Rotate around z so that LCA ostium sits at a canonical angle (e.g., 0 = +x direction)
delta_theta = -lca_ost_angle  # rotation needed to align LCA ostium to +x

# Apply rotation around z-axis
Rz = np.array([
    [np.cos(delta_theta), -np.sin(delta_theta), 0],
    [np.sin(delta_theta),  np.cos(delta_theta), 0],
    [0, 0, 1]
])

# Rotate all points
for name in centerlines:
    centerlines[name] = centerlines[name] @ Rz.T
for name in landmarks:
    landmarks[name] = Rz @ landmarks[name]
```

Now all patients share:
- Same origin: `(0, 0, 0)`.
- Same z-axis: base → apex.
- Same x-axis: LCA ostium direction projected onto coronary plane.
- Same y-axis: anterior-posterior (derived).

This is the **global cardiac frame**.

### 3.4 Step 3: Decide on Scale Normalization

Two options:

**Option A — Preserve scale (recommended):**
Keep physical mm coordinates. Heart size variation is real anatomy and should be modeled. The ellipsoid semi-axes `(a, b, c)` will capture size variation.

**Option B — Normalize scale:**
Divide all points by a per-patient scale factor (e.g., coronary ellipse semi-major axis). Model shape and size separately. Useful if you want to decouple shape from size.

**Recommendation:** Option A for the first version. Size is part of the scaffold PCA.

### 3.5 Verification

After global alignment, plot all 190 patients' centerlines overlaid in the global cardiac frame. You should see:
- All ostia clustered in the `z > 0` region, near the `+x` direction (LCA) and `-x` direction (RCA).
- All LAD paths descending from `z ≈ 0` toward `z < 0` (apex).
- All LCX paths wrapping in the `z ≈ 0` plane.
- All RCA paths wrapping in the `z ≈ 0` plane, opposite side from LCX.

If any patient looks mirrored or rotated incorrectly, flag for manual QC.

---

## Part 4: Ellipsoid Surface Model

### 4.1 Fit Two Ellipses Per Patient (in Global Cardiac Frame)

#### Coronary Ellipse (in the z = 0 plane)

```python
ring_points = np.vstack([rca_centerline, lcx_centerline])
ring_2d = ring_points[:, :2]  # (x, y) since z ≈ 0

from skimage.measure import EllipseModel
ellipse_cor = EllipseModel()
ellipse_cor.estimate(ring_2d)
# params: (xc, yc, a, b, theta)
```

#### Interventricular Ellipse (in the x = 0 plane)

```python
lad_points_2d = lad_centerline[:, 1:3]  # (y, z) since x ≈ 0

ellipse_iv = EllipseModel()
ellipse_iv.estimate(lad_points_2d)
# params: (yc, zc, b_iv, c, theta_iv)
```

### 4.2 Form the Ellipsoid

The two ellipses define a triaxial ellipsoid:

```
x²/a² + y²/b² + z²/c² = 1
```

- `a` = coronary ellipse semi-axis 1 (x direction)
- `b` = coronary ellipse semi-axis 2 (y direction)
- `c` = IV ellipse semi-axis 2 (z direction, base-to-apex)

In the global cardiac frame, the ellipsoid is centered at the origin with axes aligned to `(x, y, z)`.

### 4.3 Parametric Ellipsoid Surface

```python
def ellipsoid_point(u, v, a, b, c):
    """
    u: azimuthal angle [0, 2π) — around the coronary plane
    v: polar angle [0, π] — from base (v=0) to apex (v=π)
    Returns: (x, y, z) on ellipsoid surface in cardiac frame
    """
    x = a * np.sin(v) * np.cos(u)
    y = b * np.sin(v) * np.sin(u)
    z = c * np.cos(v)
    return np.array([x, y, z])

def ellipsoid_normal(u, v, a, b, c):
    """Outward unit normal at (u, v)."""
    nx = np.sin(v) * np.cos(u) / a
    ny = np.sin(v) * np.sin(u) / b
    nz = np.cos(v) / c
    n = np.array([nx, ny, nz])
    return n / np.linalg.norm(n)

def ellipsoid_tangent_u(u, v, a, b, c):
    """Tangent along u direction (circumferential)."""
    tx = -a * np.sin(v) * np.sin(u)
    ty =  b * np.sin(v) * np.cos(u)
    tz =  0.0
    t = np.array([tx, ty, tz])
    return t / np.linalg.norm(t)

def ellipsoid_tangent_v(u, v, a, b, c):
    """Tangent along v direction (base-to-apex)."""
    tx = a * np.cos(v) * np.cos(u)
    ty = b * np.cos(v) * np.sin(u)
    tz = -c * np.sin(v)
    t = np.array([tx, ty, tz])
    return t / np.linalg.norm(t)
```

### 4.4 Project Patient Data Onto the Ellipsoid

For each patient, project all centerline points and landmarks onto their fitted ellipsoid:

```python
def project_to_ellipsoid(point, a, b, c):
    """
    Find nearest point on ellipsoid surface.
    Returns: (u, v, offset)
    - u, v: surface parameters
    - offset: signed distance along surface normal (mm)
    """
    x, y, z = point
    v = np.arccos(np.clip(z / c, -1, 1))
    u = np.arctan2(y / b, x / a)

    surf = ellipsoid_point(u, v, a, b, c)
    normal = ellipsoid_normal(u, v, a, b, c)
    offset = np.dot(point - surf, normal)

    return u, v, offset
```

For each vessel point, store:
- `(u, v)`: surface projection (where on the ellipsoid it sits).
- `offset`: signed distance along normal (how far off the surface it is).

For each landmark, store the same `(u, v, offset)`.

### 4.5 Expected Offsets

| Landmark | Expected offset | Reason |
|----------|----------------|--------|
| LCA ostium | +10 to +20 mm | Above AV groove, near aortic root |
| RCA ostium | +10 to +20 mm | Same |
| Bifurcation | 0 to +3 mm | Near coronary plane |
| LAD endpoint | 0 to +2 mm | Near apex |
| LCX endpoint | 0 to +2 mm | On AV groove |
| RCA endpoint | 0 to +2 mm | On AV groove |

### 4.6 Resample Vessel Paths to Fixed Point Counts

Arc-length resample each vessel in `(u, v)` space to fixed control point counts:

```python
VESSEL_POINT_COUNTS = {
    "RCA":  15,
    "LMCA": 5,
    "LAD":  12,
    "LCX":  10,
}
```

For each patient, each vessel becomes:
- `u_path`: array of `N` u-values.
- `v_path`: array of `N` v-values.
- `offsets`: array of `N` signed distances.
- `deviations`: array of `N × 3` (in-plane + out-of-plane residual after subtracting offset along normal).

---

## Part 5: Population Statistics

### 5.1 Level 1 — Scaffold Statistics (Ellipsoid Parameters)

Per patient, store:

```python
scaffold = np.array([a, b, c])  # (3,)
```

With only 3 parameters and 190 patients, simple mean/std is sufficient:

```python
scaffold_mean = np.mean(all_scaffolds, axis=0)  # (3,)
scaffold_std = np.std(all_scaffolds, axis=0)    # (3,)
```

Optionally add the ellipse rotation angles if they vary meaningfully.

### 5.2 Level 2 — Landmark Statistics

Per patient, store each landmark as `(u, v, offset)`:

```python
landmark_vector = np.array([
    lca_ostium_u, lca_ostium_v, lca_ostium_offset,
    rca_ostium_u, rca_ostium_v, rca_ostium_offset,
    bifurcation_u, bifurcation_v, bifurcation_offset,
    lad_end_u, lad_end_v, lad_end_offset,
    lcx_end_u, lcx_end_v, lcx_end_offset,
    rca_end_u, rca_end_v, rca_end_offset,
])  # (18,)
```

Statistics: mean + std, or PCA if correlations matter:

```python
landmark_mean = np.mean(all_landmarks, axis=0)  # (18,)
landmark_std = np.std(all_landmarks, axis=0)    # (18,)

# Or PCA:
landmark_centered = all_landmarks - landmark_mean
U, S, Vt = np.linalg.svd(landmark_centered, full_matrices=False)
# Retain components explaining ~95% variance
```

### 5.3 Level 3 — Vessel Deviation PCA (Main Shape Model)

This is the core statistical model.

#### 5.3.1 Compute Deviations

For each patient, for each vessel, for each resampled point:

```python
# Surface point at (u, v)
surf = ellipsoid_point(u, v, a, b, c)

# Normal and tangents at (u, v)
normal = ellipsoid_normal(u, v, a, b, c)
tang_u = ellipsoid_tangent_u(u, v, a, b, c)
tang_v = ellipsoid_tangent_v(u, v, a, b, c)

# Actual centerline point (in cardiac frame)
actual = centerline_point

# Decompose the residual into 3 components:
residual = actual - surf
dev_x = np.dot(residual, tang_u)    # in-plane, circumferential
dev_y = np.dot(residual, tang_v)    # in-plane, base-to-apex
dev_z = np.dot(residual, normal)    # out-of-plane (signed!)

deviation_point = np.array([dev_x, dev_y, dev_z])
```

#### 5.3.2 Concatenate All Vessels Into One Shape Vector

```python
shape_vector = np.concatenate([
    rca_deviations.flatten(),    # 15 * 3 = 45
    lmca_deviations.flatten(),   # 5 * 3  = 15
    lad_deviations.flatten(),    # 12 * 3 = 36
    lcx_deviations.flatten(),    # 10 * 3 = 30
])
# Total: 126 dimensions per patient
# Population: (190, 126)
```

#### 5.3.3 Fit Joint PCA

```python
all_shape_vectors = np.array([p["shape_vector"] for p in patients])  # (190, 126)
dev_mean = all_shape_vectors.mean(axis=0)
dev_centered = all_shape_vectors - dev_mean

U, S, Vt = np.linalg.svd(dev_centered, full_matrices=False)

variance_explained = np.cumsum(S**2) / np.sum(S**2)
k = np.searchsorted(variance_explained, 0.95) + 1
dev_pcs = Vt[:k]           # (k, 126)
dev_eigenvalues = S[:k]**2 / (len(all_shape_vectors) - 1)  # (k,)
```

This PCA captures:
- **Intra-vessel correlations:** adjacent points move together.
- **Inter-vessel correlations:** LAD and LCX deviations are coupled.
- **In-plane / out-of-plane correlations:** vessels that wrap wider also bulge differently.

### 5.4 Level 4 — Tortuosity and Obliquity Statistics

#### 5.4.1 Measure Tortuosity Per Vessel

For each patient and vessel, measure how much the `(u, v)` path deviates from a straight line in parameter space:

```python
def measure_tortuosity(u_path, v_path):
    t = np.linspace(0, 1, len(u_path))
    u_linear = u_path[0] + (u_path[-1] - u_path[0]) * t
    v_linear = v_path[0] + (v_path[-1] - v_path[0]) * t
    u_dev = np.std(u_path - u_linear)
    v_dev = np.std(v_path - v_linear)
    return np.sqrt(u_dev**2 + v_dev**2)

# Per vessel across 190 patients:
tort_stats = {}
for vessel in ["RCA", "LMCA", "LAD", "LCX"]:
    torts = [measure_tortuosity(p[f"{vessel}_u"], p[f"{vessel}_v"]) for p in patients]
    tort_stats[vessel] = {"mean": np.mean(torts), "std": np.std(torts)}
```

#### 5.4.2 Measure Obliquity Per Vessel

Obliquity = total drift in `u` from start to end (how far off the ideal groove the vessel runs):

```python
def measure_obliquity(u_path):
    return u_path[-1] - u_path[0]

obliquity_stats = {}
for vessel in ["LAD", "LCX", "RCA"]:
    obliqs = [measure_obliquity(p[f"{vessel}_u"]) for p in patients]
    obliquity_stats[vessel] = {"mean": np.mean(obliqs), "std": np.std(obliqs)}
```

### 5.5 Level 5 — Side Branch Statistics

From the skeleton graphs, extract per patient:

| Statistic | How to measure | Distribution model |
|-----------|---------------|-------------------|
| Number of side branches per parent vessel | Count junctions on each main path | Poisson or negative binomial |
| Attachment parametric position `t` along parent | Normalize junction position to `[0,1]` | Empirical distribution or beta |
| Side branch length | Arc length of branch path | Gamma or log-normal |
| Side branch direction relative to parent | Angle between parent tangent and branch tangent at attachment | Von Mises (angular) |

### 5.6 Level 6 — Validation Thresholds

From the 190-patient population, compute acceptance bounds for generated trees:

```python
validation_thresholds = {
    "branch_lengths_mm": {
        vessel: (np.percentile(lengths, 2.5), np.percentile(lengths, 97.5))
        for vessel, lengths in population_branch_lengths.items()
    },
    "bifurcation_angle_deg": (
        np.percentile(angles, 2.5),
        np.percentile(angles, 97.5)
    ),
    "max_out_of_plane_mm": {
        vessel: (np.percentile(devs, 2.5), np.percentile(devs, 97.5))
        for vessel, devs in population_max_deviations.items()
    },
    "tortuosity_range": {
        vessel: (np.percentile(torts, 2.5), np.percentile(torts, 97.5))
        for vessel, torts in population_tortuosities.items()
    },
}
```

---

## Part 6: Synthetic Tree Generation

### 6.1 Generation Pipeline Overview

```
1. Sample ellipsoid parameters (a, b, c)
2. Sample landmark (u, v, offset) positions
3. Generate vessel paths as B-splines in (u, v) on the ellipsoid surface
4. Add tortuosity (low-frequency perturbation of control points in (u, v))
5. Add obliquity (linear drift in u from ideal groove)
6. Evaluate B-splines → surface points
7. Add deviation noise from joint PCA
8. Apply ostial offsets (decaying from ostium to surface)
9. Snap bifurcation (LAD[0] = LCX[0] = LMCA[-1])
10. Generate side branches as random surface walks
11. Validate against population thresholds
12. Reject and resample if invalid
```

### 6.2 Step 1 — Sample Ellipsoid

```python
a = np.clip(rng.normal(scaffold_mean[0], scaffold_std[0]),
            scaffold_mean[0] - 2*scaffold_std[0],
            scaffold_mean[0] + 2*scaffold_std[0])
b = np.clip(rng.normal(scaffold_mean[1], scaffold_std[1]),
            scaffold_mean[1] - 2*scaffold_std[1],
            scaffold_mean[1] + 2*scaffold_std[1])
c = np.clip(rng.normal(scaffold_mean[2], scaffold_std[2]),
            scaffold_mean[2] - 2*scaffold_std[2],
            scaffold_mean[2] + 2*scaffold_std[2])
```

### 6.3 Step 2 — Sample Landmarks

```python
landmarks = {}
for name in ["lca_ostium", "rca_ostium", "bifurcation",
             "lad_end", "lcx_end", "rca_end"]:
    stats = landmark_stats[name]
    u = np.clip(rng.normal(stats["u_mean"], stats["u_std"]), 0, 2*np.pi)
    v = np.clip(rng.normal(stats["v_mean"], stats["v_std"]), 0.01, np.pi - 0.01)
    offset = rng.normal(stats["offset_mean"], stats["offset_std"])
    landmarks[name] = {"u": u, "v": v, "offset": offset}
```

### 6.4 Step 3 — Generate Vessel Paths in (u, v)

#### B-Spline in Parameter Space

```python
from geomdl import BSpline, utilities

def surface_spline(u_ctrl, v_ctrl, num_eval):
    """Fit B-splines to u(t) and v(t) separately."""
    cu = BSpline.Curve()
    cu.degree = min(3, len(u_ctrl) - 1)
    cu.ctrlpts = [[c] for c in u_ctrl]
    cu.knotvector = utilities.generate_knot_vector(cu.degree, len(u_ctrl))
    cu.sample_size = num_eval
    u_eval = np.array([p[0] for p in cu.evalpts])

    cv = BSpline.Curve()
    cv.degree = min(3, len(v_ctrl) - 1)
    cv.ctrlpts = [[c] for c in v_ctrl]
    cv.knotvector = utilities.generate_knot_vector(cv.degree, len(v_ctrl))
    cv.sample_size = num_eval
    v_eval = np.array([p[0] for p in cv.evalpts])

    return u_eval, v_eval
```

#### Add Tortuosity

```python
def add_tortuosity(u_ctrl, v_ctrl, strength, rng):
    """Perturb control points with low-frequency sinusoidal modes."""
    n = len(u_ctrl)
    t = np.linspace(0, 1, n)
    envelope = (1 - t) * t * 4  # zero at endpoints, peaks in middle

    u_perturb = np.zeros(n)
    v_perturb = np.zeros(n)
    for k in [1, 2, 3]:
        amp_u = strength * rng.uniform(0.3, 1.0)
        amp_v = strength * rng.uniform(0.3, 1.0)
        phase_u = rng.uniform(0, 2*np.pi)
        phase_v = rng.uniform(0, 2*np.pi)
        u_perturb += amp_u * np.sin(k * np.pi * t + phase_u) * envelope
        v_perturb += amp_v * np.sin(k * np.pi * t + phase_v) * envelope

    return u_ctrl + u_perturb, v_ctrl + v_perturb
```

#### Add Obliquity

```python
def add_obliquity(u_ctrl, obliquity_angle):
    """Drift u linearly from start to end."""
    t = np.linspace(0, 1, len(u_ctrl))
    return u_ctrl + obliquity_angle * t
```

#### Full Vessel Generation

```python
def generate_vessel(vessel_name, start_uv, end_uv, num_ctrl_pts,
                    tort_stats, obliquity_stats, rng):
    # Linear interpolation in (u, v)
    t = np.linspace(0, 1, num_ctrl_pts)
    u_ctrl = start_uv[0] + (end_uv[0] - start_uv[0]) * t
    v_ctrl = start_uv[1] + (end_uv[1] - start_uv[1]) * t

    # Add obliquity
    obliquity = rng.normal(obliquity_stats[vessel_name]["mean"],
                           obliquity_stats[vessel_name]["std"])
    obliquity = np.clip(obliquity, -2*obliquity_stats[vessel_name]["std"],
                        2*obliquity_stats[vessel_name]["std"])
    u_ctrl = add_obliquity(u_ctrl, obliquity)

    # Add tortuosity
    strength = rng.normal(tort_stats[vessel_name]["mean"],
                          tort_stats[vessel_name]["std"])
    strength = np.clip(strength, 0, tort_stats[vessel_name]["mean"] + 2*tort_stats[vessel_name]["std"])
    u_ctrl, v_ctrl = add_tortuosity(u_ctrl, v_ctrl, strength, rng)

    # Fix endpoints
    u_ctrl[0], v_ctrl[0] = start_uv
    u_ctrl[-1], v_ctrl[-1] = end_uv

    # B-spline in (u, v)
    u_path, v_path = surface_spline(u_ctrl, v_ctrl, num_eval=200)

    return u_path, v_path
```

### 6.5 Step 4 — Evaluate on Surface + Add Deviations

```python
def reconstruct_3d(u_path, v_path, deviations, a, b, c):
    """
    Evaluate points on the ellipsoid and add 3D deviations.
    deviations: (N, 3) in [tangent_u, tangent_v, normal] components.
    """
    points = np.zeros((len(u_path), 3))
    for i in range(len(u_path)):
        surf = ellipsoid_point(u_path[i], v_path[i], a, b, c)
        tang_u = ellipsoid_tangent_u(u_path[i], v_path[i], a, b, c)
        tang_v = ellipsoid_tangent_v(u_path[i], v_path[i], a, b, c)
        normal = ellipsoid_normal(u_path[i], v_path[i], a, b, c)
        points[i] = (surf
                     + deviations[i, 0] * tang_u
                     + deviations[i, 1] * tang_v
                     + deviations[i, 2] * normal)
    return points
```

### 6.6 Step 5 — Handle Ostial Segments (Off-Surface)

LMCA and RCA proximal segments start at the ostium (off-surface) and land on the surface. Use a decaying offset:

```python
def generate_ostial_segment(ostium_uv_offset, surface_target_uv,
                             num_pts, a, b, c):
    """
    Segment from ostium (off-surface) to surface target (on-surface).
    Offset decays linearly from ostium offset to 0.
    """
    u_ost, v_ost, offset_ost = ostium_uv_offset
    u_tgt, v_tgt = surface_target_uv

    u_path, v_path = surface_spline(
        [u_ost, u_tgt], [v_ost, v_tgt], num_pts
    )

    t = np.linspace(0, 1, num_pts)
    offsets = offset_ost * (1 - t)  # linear decay

    points = np.zeros((num_pts, 3))
    for i in range(num_pts):
        surf = ellipsoid_point(u_path[i], v_path[i], a, b, c)
        normal = ellipsoid_normal(u_path[i], v_path[i], a, b, c)
        points[i] = surf + offsets[i] * normal

    return points
```

### 6.7 Step 6 — Snap Bifurcation

```python
bif_pt = vessels["LMCA"][-1]
vessels["LAD"][0] = bif_pt
vessels["LCX"][0] = bif_pt
```

### 6.8 Step 7 — Generate Side Branches

```python
def generate_side_branches(parent_vessels, parent_uv_paths, ellipsoid_params,
                           side_branch_stats, rng):
    """
    For each parent vessel, sample side branches as random surface walks.
    """
    a, b, c = ellipsoid_params
    all_branches = []

    for parent_name in ["LAD", "LCX", "RCA"]:
        parent_u = parent_uv_paths[parent_name]
        parent_v = parent_uv_paths[parent_name]
        n_parent = len(parent_u)

        # Sample number of side branches
        mean_count = side_branch_stats[parent_name]["mean_count"]
        num_branches = rng.poisson(mean_count)

        for j in range(num_branches):
            # Sample attachment position (avoid ostium and distal end)
            t_attach = rng.uniform(0.1, 0.85)
            idx = int(t_attach * (n_parent - 1))

            u0 = parent_u[idx]
            v0 = parent_v[idx]

            # Sample branch length
            length = rng.gamma(*side_branch_stats[parent_name]["length_params"])

            # Sample direction: random angle from parent tangent
            angle = rng.uniform(30, 90) * np.pi / 180
            direction = rng.choice([-1, 1])  # left or right

            # Walk on surface
            num_pts = max(10, int(length / 2.0))  # ~2mm spacing
            u_branch = [u0]
            v_branch = [v0]

            du = direction * np.sin(angle) * 0.05
            dv = np.cos(angle) * 0.05

            curvature = rng.uniform(-0.3, 0.3)
            for i in range(1, num_pts):
                turn = curvature * i / num_pts
                du_i = du * np.cos(turn) - dv * np.sin(turn)
                dv_i = du * np.sin(turn) + dv * np.cos(turn)
                u_new = u_branch[-1] + du_i
                v_new = np.clip(v_branch[-1] + dv_i, 0.01, np.pi - 0.01)
                u_branch.append(u_new)
                v_branch.append(v_new)

            # Evaluate on surface + small deviation
            u_arr = np.array(u_branch)
            v_arr = np.array(v_branch)
            surf_pts = np.array([ellipsoid_point(u, v, a, b, c)
                                 for u, v in zip(u_arr, v_arr)])

            # Small random deviation (side branches have less data)
            dev = rng.normal(0, 0.002, surf_pts.shape)
            branch_pts = surf_pts + dev

            all_branches.append({
                "parent": parent_name,
                "attachment_index": idx,
                "points": branch_pts,
            })

    return all_branches
```

### 6.9 Step 8 — Validation

```python
def validate_tree(vessels, side_branches, thresholds):
    errors = []

    # Check branch lengths
    for name, (lo, hi) in thresholds["branch_lengths_mm"].items():
        length = path_length(vessels[name])
        if not (lo <= length <= hi):
            errors.append(f"{name} length {length:.1f} outside [{lo:.1f}, {hi:.1f}]")

    # Check bifurcation angle
    angle = bifurcation_angle(vessels["LAD"], vessels["LCX"])
    lo, hi = thresholds["bifurcation_angle_deg"]
    if not (lo <= angle <= hi):
        errors.append(f"Bifurcation angle {angle:.1f} outside [{lo:.1f}, {hi:.1f}]")

    # Check no self-intersection
    if has_self_intersection(vessels):
        errors.append("Self-intersection detected")

    # Check max out-of-plane deviation
    for name, (lo, hi) in thresholds["max_out_of_plane_mm"].items():
        max_dev = np.max(np.abs(out_of_plane(vessels[name])))
        if max_dev > hi:
            errors.append(f"{name} max deviation {max_dev:.2f} exceeds {hi:.2f}")

    return len(errors) == 0, errors
```

---

## Part 7: Cardiac Motion

### 7.1 Motion Model

Deform the ellipsoid as a function of cardiac phase `φ ∈ [0, 1)`:

- `φ = 0`: end-diastole (relaxed, max volume).
- `φ ≈ 0.35`: end-systole (max contraction).
- `φ = 1`: back to end-diastole.

Three motion components:

| Component | Typical amplitude | Description |
|-----------|------------------|-------------|
| Radial contraction | 12–18% at peak systole | Semi-axes `a`, `b` shrink |
| Longitudinal shortening | 8–12% at peak systole | Semi-axis `c` shrinks |
| Torsion | 5–15° at peak systole | Base rotates more than apex |

### 7.2 Contraction Curve

```python
def contraction_curve(phase):
    """
    Smooth periodic contraction function.
    0 at end-diastole, 1 at peak systole (~35% of cycle), 0 at end-diastole.
    """
    peak_phase = 0.35
    if phase < peak_phase:
        return np.sin(np.pi * phase / (2 * peak_phase))
    else:
        return np.cos(np.pi * (phase - peak_phase) / (2 * (1 - peak_phase)))
```

### 7.3 Deform the Ellipsoid

```python
def deform_ellipsoid(a, b, c, u, v, phase,
                     radial_amplitude=0.15,
                     longitudinal_amplitude=0.10,
                     torsion_amplitude_deg=10):
    """
    Given reference ellipsoid params and a point (u, v) at end-diastole,
    return deformed ellipsoid params and shifted (u, v) at cardiac phase.
    """
    s = contraction_curve(phase)

    # Radial contraction
    radial_factor = 1 - radial_amplitude * s
    a_def = a * radial_factor
    b_def = b * radial_factor

    # Longitudinal shortening
    long_factor = 1 - longitudinal_amplitude * s
    c_def = c * long_factor

    # Torsion: base (v near 0) rotates more than apex (v near π)
    torsion_rad = torsion_amplitude_deg * np.pi / 180 * s
    u_def = u + torsion_rad * (1 - v / np.pi)

    return a_def, b_def, c_def, u_def, v
```

### 7.4 Apply Motion to All Vessel Points

```python
def apply_cardiac_motion(vessels_uv, ellipsoid_params, num_phases=10):
    """
    vessels_uv: dict of {name: (u_array, v_array)} at reference (end-diastole)
    Returns: list of num_phases frames, each a dict of {name: (N, 3) points}
    """
    a, b, c = ellipsoid_params
    motion_frames = []

    for phase in np.linspace(0, 1, num_phases, endpoint=False):
        frame = {}
        for name, (u_arr, v_arr) in vessels_uv.items():
            # Deform ellipsoid at this phase
            a_d, b_d, c_d, u_d, v_d = deform_ellipsoid(
                a, b, c, u_arr, v_arr, phase
            )

            # Reconstruct 3D points on deformed ellipsoid
            points = np.array([
                ellipsoid_point(u_d[i], v_d[i], a_d, b_d, c_d)
                for i in range(len(u_d))
            ])

            # Add the same deviations (they move with the surface)
            # Optionally: deviations could be scaled slightly with contraction
            if name in deviations:
                points += deviations[name]

            frame[name] = points

        motion_frames.append(frame)

    return motion_frames
```

### 7.5 What Happens to Each Component During Motion

| Component | How it moves |
|-----------|-------------|
| Ellipsoid surface | Contracts radially + shortens longitudinally + twists |
| Major vessel centerlines | Follow the ellipsoid automatically (same `(u, v)`, deformed surface) |
| Deviations (off-surface noise) | Stay fixed relative to the surface point they were added to |
| Side branches | Follow the same ellipsoid deformation |
| Ostial offsets | Shrink slightly with contraction (aortic root moves less than epicardium) |
| Bifurcation | Stays connected (both LAD and LCX start from the same deformed LMCA endpoint) |
| Radii | Optionally add pulsatile variation (small, ~2-5% diameter change) |

### 7.6 Output Format for Time-Resolved Data

```python
output = {
    "num_phases": 10,
    "phase_values": [0.0, 0.1, 0.2, ..., 0.9],
    "reference_phase": 0,  # end-diastole
    "ellipsoid_params": {"a": a, "b": b, "c": c},
    "motion_params": {
        "radial_amplitude": 0.15,
        "longitudinal_amplitude": 0.10,
        "torsion_amplitude_deg": 10,
    },
    "frames": [
        {
            "phase": 0.0,
            "vessels": {"RCA": ..., "LMCA": ..., "LAD": ..., "LCX": ...},
            "side_branches": [...],
        },
        ...
    ],
}
```

---

## Part 8: Module Structure

### 8.1 Proposed File Layout

```
coronary_tree_generator/
├── extraction/
│   ├── __init__.py
│   ├── nifti_loader.py          # Load NIfTI, extract affine, binarize
│   ├── skeleton_graph.py        # Skeletonize, build graph, find endpoints/junctions
│   ├── landmark_detector.py     # Find ostia, bifurcation, branch endpoints
│   ├── branch_splitter.py       # Split skeleton into RCA, LMCA, LAD, LCX
│   └── resampler.py             # Arc-length resample to fixed point counts
│
├── alignment/
│   ├── __init__.py
│   ├── plane_fitting.py         # SVD-based plane fit, coronary + IV planes
│   ├── cardiac_frame.py         # Build per-patient frame, global frame alignment
│   ├── ellipse_fitting.py       # Fit 2D ellipses in coronary and IV planes
│   └── ellipsoid.py             # Form ellipsoid, surface evaluation, projection
│
├── statistics/
│   ├── __init__.py
│   ├── scaffold_stats.py        # Ellipsoid param statistics (mean/std or PCA)
│   ├── landmark_stats.py        # Landmark (u,v,offset) statistics
│   ├── deviation_pca.py         # Joint PCA on all vessel deviations
│   ├── tortuosity_stats.py      # Tortuosity and obliquity measurement + stats
│   ├── side_branch_stats.py     # Side branch count, position, length, direction
│   └── validation_thresholds.py # Population-based acceptance bounds
│
├── generation/
│   ├── __init__.py
│   ├── surface_spline.py        # B-spline in (u,v) parameter space
│   ├── vessel_generator.py      # Generate major vessel paths with tortuosity + obliquity
│   ├── deviation_sampler.py     # Sample deviations from joint PCA
│   ├── ostial_segment.py        # Generate off-surface ostial segments
│   ├── side_branch_generator.py # Random surface walks for side branches
│   ├── tree_assembler.py        # Assemble full tree, snap bifurcation
│   └── validator.py             # Validate generated tree against thresholds
│
├── motion/
│   ├── __init__.py
│   ├── contraction_model.py     # Phase-dependent contraction curve
│   ├── ellipsoid_deformer.py    # Deform ellipsoid (radial, longitudinal, torsion)
│   └── motion_applier.py        # Apply motion to all vessel points
│
├── output/
│   ├── __init__.py
│   ├── save_geometry.py         # Save centerlines as npy (MxNx4 for static, TxMxNx4 for cine)
│   ├── save_metadata.py         # Save generation parameters, validation report
│   └── save_visualization.py    # 3D plots, radius profiles, motion animations
│
├── pipeline.py                  # End-to-end: extraction → stats → generation → motion → output
└── config.py                    # Vessel point counts, motion params, file paths
```

### 8.2 Module Dependencies

```
extraction/
  depends on: nibabel, skimage, skan/networkx, scipy

alignment/
  depends on: numpy, skimage (EllipseModel), extraction outputs

statistics/
  depends on: numpy, sklearn (optional for PCA), alignment outputs

generation/
  depends on: geomdl (BSpline), numpy, statistics outputs

motion/
  depends on: numpy only

output/
  depends on: numpy, matplotlib (visualization), vtk (optional mesh)

pipeline.py
  depends on: all modules
```

### 8.3 External Python Packages Required

```
nibabel          # NIfTI I/O
SimpleITK        # Image processing, morphology (optional alternative to skimage)
scikit-image     # skeletonize_3d, EllipseModel
skan             # Skeleton → graph (optional, can use networkx directly)
networkx         # Graph operations
geomdl           # B-spline curves (NURBS-python)
numpy            # Core numerics
scipy            # Interpolation, signal processing
matplotlib       # Visualization
scikit-learn     # Optional: PCA, LinearRegression (can use numpy SVD instead)
```

### 8.4 Data Files Produced by the Statistics Phase

```
stats_output/
├── scaffold_stats.json          # Ellipsoid param mean/std
├── landmark_stats.json          # Landmark (u,v,offset) mean/std per landmark
├── deviation_pca.npz            # PCA components, eigenvalues, mean
├── tortuosity_stats.json        # Tortuosity/obliquity mean/std per vessel
├── side_branch_stats.json       # Side branch distributions
├── validation_thresholds.json   # Population-based acceptance bounds
└── patient_data_aligned.npz     # All 190 patients' aligned data (for QC/debugging)
```

### 8.5 Data Files Produced by the Generation Phase

```
generated_trees/
├── tree_0000/
│   ├── geometry_static.npy      # MxNx4 (branches × points × xyzr)
│   ├── geometry_cine.npy        # TxMxNx4 (phases × branches × points × xyzr)
│   ├── metadata.json            # Generation parameters, validation report
│   ├── ellipsoid_params.json    # a, b, c, motion params
│   └── visualization.png        # 3D plot
├── tree_0001/
│   └── ...
```

---

## Part 9: End-to-End Pipeline Execution

### 9.1 Phase 1 — Data Extraction (run once per dataset)

```bash
python pipeline.py extract \
    --input-dir /path/to/nii_labels \
    --output-dir /path/to/extracted_data \
    --num-workers 4
```

Produces: per-patient centerlines + landmarks in scanner coordinates.

### 9.2 Phase 2 — Alignment and Statistics (run once)

```bash
python pipeline.py compute-stats \
    --input-dir /path/to/extracted_data \
    --output-dir /path/to/stats_output \
    --num-ctrl-pts-rca 15 \
    --num-ctrl-pts-lmca 5 \
    --num-ctrl-pts-lad 12 \
    --num-ctrl-pts-lcx 10
```

Produces: scaffold stats, landmark stats, deviation PCA, tortuosity stats, validation thresholds.

### 9.3 Phase 3 — Tree Generation (run as needed)

```bash
python pipeline.py generate \
    --stats-dir /path/to/stats_output \
    --output-dir /path/to/generated_trees \
    --num-trees 1000 \
    --num-phases 10 \
    --seed 42 \
    --motion \
    --validate
```

Produces: synthetic coronary trees with optional cardiac motion.

### 9.4 Quality Control (run after each phase)

```bash
python pipeline.py qc \
    --input-dir /path/to/extracted_data \
    --output-dir /path/to/qc_report
```

Produces: overlay plots of all aligned patients, outlier detection, manual review flags.

---

## Part 10: Key Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Surface model | Triaxial ellipsoid (3 params) | Simplest surface that captures heart shape; deviation PCA handles the rest |
| Alignment | Ring plane + ostia + LAD direction | Derives cardiac frame from coronary labels alone; no LV/aorta segmentation needed |
| Major vessel generation | B-spline in (u,v) parameter space | Curve stays on surface by construction; no projection needed |
| Shape variation | Joint PCA on all vessel deviations | Captures inter-vessel and intra-vessel correlations |
| Tortuosity | Sinusoidal perturbation of (u,v) control points | Low-frequency, smooth, endpoint-preserving |
| Obliquity | Linear drift in u from ideal groove | Captures vessels running slightly off the groove |
| Ostial segments | Decaying offset from surface | Handles the one region where vessels leave the epicardium |
| Side branches | Random surface walks from parent attachment | Simple, surface-constrained, anatomically plausible |
| Cardiac motion | Deform ellipsoid → vessels follow automatically | No per-vessel motion model needed; torsion + contraction + shortening |
| Validation | Population-based rejection thresholds | Reject implausible samples using 2.5–97.5 percentile bounds from 190 patients |
| Scale | Preserve physical mm | Heart size is real anatomy; modeled by scaffold statistics |

---

## Part 11: Future Extensions

| Extension | How to add |
|-----------|-----------|
| More realistic surface | Add 2-3 harmonic deformation modes to the ellipsoid (e.g., LV bulge, RV flattening) |
| Pulsatile radius variation | Add phase-dependent radius scaling (~2-5% diameter change) |
| Disease/stenosis | Reuse existing `disease_model.py` on generated centerlines |
| Patient-specific motion | Replace synthetic contraction with 4D CT-derived displacement fields |
| Mesh generation | Reuse existing `tube_surface.py` + `tight_mesh.py` on generated centerlines |
| Full coronary tree | Add PDA, PLB, diagonals, obtuse marginals as additional surface branches |
| ShapeWorks comparison | Export generated surfaces and compare SSM modes with ShapeWorks correspondence |
