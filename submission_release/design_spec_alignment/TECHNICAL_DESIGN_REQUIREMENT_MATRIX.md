# Technical Design Requirement Matrix

| Requirement | Document section | Expected method | Current implementation | Status | Action |
|---|---|---|---|---|---|
| Centerline extraction | Part 1.2 | NIfTI affine, skeleton graph, ordered physical centerlines | Extraction modules and immutable raw-case archives | EXACTLY_IMPLEMENTED | None |
| Landmark identification | Part 1.3 | Ostia, LMCA bifurcation, LAD/LCX/RCA terminals | LCA landmarks plus multi-signal RAS daughter assignment; unreliable RCA excluded | EQUIVALENT_IMPLEMENTATION | Retain auditable assignment confidence |
| Coronary plane | Part 2.1 | SVD AV-groove plane from RCA+LCX | Saved SVD plane uses inferred RCA candidate plus LCX; candidate is not annotated RCA truth | DELIBERATE_DATASET_DEVIATION | Keep limitation explicit |
| IV plane | Part 2.2 | Cross-product plane from coronary normal and LAD direction | Measured LAD centroid-SVD plane; median design-normal difference 24.97 deg | EQUIVALENT_IMPLEMENTATION | Preserve measured reference and report comparison |
| Cardiac frame | Parts 2.3-2.4 | Right-handed frame; LAD descends in negative z | Rigid orthonormal frame with determinant, length and LAD orientation checks | EXACTLY_IMPLEMENTED | None |
| Global frame | Part 3 | Canonical z rotation from LCA ostium | Per-case origin removal and LCA-ostium +X canonicalization | EXACTLY_IMPLEMENTED | None |
| Coronary ellipse | Part 4.1 | Ring ellipse supplies a,b | Saved two-plane/two-ellipse measurements supply crown a,b | EXACTLY_IMPLEMENTED | Do not treat as LCX path mold |
| IV ellipse | Part 4.1 | LAD ellipse supplies c | Saved LAD ellipse axis aligned to cardiac z supplies c; robust proxy flags pathological partial arcs | EQUIVALENT_IMPLEMENTATION | Retain quality gates |
| Ellipsoid | Part 4.2 | a,b,c triaxial support scaffold | Axis-aligned per-case scaffold derived from the two ellipse measurements | EXACTLY_IMPLEMENTED | None |
| Surface coordinates | Part 4.3-4.4 | u,v,offset and local tangent/normal residual | All source and generated LCA points have reconstructable surface-relative records | EQUIVALENT_IMPLEMENTATION | Angular/radial mapping is exactly reconstructable, not Euclidean nearest projection |
| Fixed-point resampling | Part 4.6 | LMCA 5, LAD 12, LCX 10 | Arc-length fixed representation with exactly 27 LCA points | EXACTLY_IMPLEMENTED | None |
| Scaffold statistics | Part 5.1 | Population distribution of a,b,c | Joint empirical bootstrap over 52 eligible triples | EQUIVALENT_IMPLEMENTATION | Prefer joint empirical sampling over independent Gaussian axes |
| Landmark statistics | Part 5.2 | Landmarks in u,v,offset | LCA ostium, bifurcation and daughter terminals use joint empirical case bootstrap | EXACTLY_IMPLEMENTED | RCA landmarks excluded |
| Deviation PCA | Part 5.3 | Joint tangent-u/tangent-v/normal PCA | 81-D LCA-only joint PCA: 27 x 3, 13 modes retain 95.55% | EXACTLY_IMPLEMENTED | None |
| Tortuosity | Parts 5.4, 6.4 | Population-derived smooth low-frequency surface variation | Natural empirical path/PCA tortuosity retained; extra random perturbation disabled to prevent double counting | EQUIVALENT_IMPLEMENTATION | Document deliberate no-double-counting policy |
| Obliquity | Parts 5.4, 6.4 | u drift learned from population | Measured and gated as unwrapped u drift; inherited jointly from matched trajectory/PCA | EQUIVALENT_IMPLEMENTATION | None |
| B-spline in u,v | Part 6.4 | Major-vessel B-spline evaluated in surface parameter space | Production uses explicit cubic B-spline in matched cardiac XYZ, then exact surface re-parameterization; literal u/v candidate evaluated on 52 cases and 26/52 LAD controls touch a u-singular pole | PARTIALLY_IMPLEMENTED | Candidate is quantified in uv_spline_equivalence_audit; retain protected stable path until a pole-safe chart is validated |
| Reconstruction | Parts 6.5, 7 | Ellipsoid surface plus local deviation | Full local basis reconstruction with numerical round trip | EQUIVALENT_IMPLEMENTATION | Keep stronger linear-basis solve |
| Ostial offsets | Part 6.6 | Decaying off-surface proximal segment | Empirical LMCA u,v,offset course is retained rather than imposing a generic linear decay | EQUIVALENT_IMPLEMENTATION | None |
| Bifurcation snapping | Part 6.7 | One exact LMCA/LAD/LCX junction | Shared point enforced to machine precision | EXACTLY_IMPLEMENTED | None |
| Validation | Parts 5.6, 6.9 | Population-derived rejection thresholds | Observed real bounds, role gates, topology, course, collision and descriptive central intervals | EQUIVALENT_IMPLEMENTATION | Add surface-role evidence to audit layer |
| Cardiac motion | Part 7 | Radial contraction, shortening, torsion on ellipsoid | Surface-associated points deform coherently; 9 independent phases plus closure frame | EQUIVALENT_IMPLEMENTATION | Preserve corrected closure convention |
| Output representation | Parts 7.6, 8 | Static/cine arrays, metadata and visualization | NPY/NPZ, JSON, VTP/VTM/PVD, PNG/GIF/HTML and reports | EXACTLY_IMPLEMENTED | None |
| RCA population model | Parts 1-8 | Joint LCA+RCA model | RCA candidates are not trusted annotated branch identity and are excluded from the validated generator | DELIBERATE_DATASET_DEVIATION | Do not fabricate RCA |
| Side branches | Parts 5.5, 6.8 | Population-derived diagonals/septals/OM branches | Not part of validated core release because branch identities were not sufficiently validated | NOT_IMPLEMENTED | Keep outside core release |
