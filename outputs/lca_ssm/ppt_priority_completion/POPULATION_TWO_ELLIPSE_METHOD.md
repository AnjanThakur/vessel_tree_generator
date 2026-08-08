# Population two-plane / two-ellipse method

## What is measured

For each of the 200 discovered NIfTI labels, the pipeline looks for the previously extracted, unchanged physical RAS centerlines. 191 cases have usable LMCA/LAD/LCX centerlines. Failures remain visible in `population_case_status.csv`.

1. **SVD plane:** subtract the source-point centroid and use the final right-singular vector as the best-fit plane normal. This produces a centroid, a normal, a deterministic RAS-tied in-plane basis, and point-to-plane residuals. It does not produce an ellipse.
2. **Actual ellipse:** project the unchanged source coordinates into that plane only for measurement, then fit a genuine 2-D ellipse using all support points. The fit produces center, semi-major radius `a`, semi-minor radius `b`, and axial tilt. The source vessel coordinates are never replaced.
3. **Two anatomical roles:** the crown/AV-groove plane uses LCX plus an inferred disconnected RCA candidate where available; that RCA is explicitly not annotated ground truth. The LAD/interventricular plane uses the LAD descent.
4. **Angles:** ellipse theta=0 is the positive major-axis direction whose component along deterministic plane basis `u` is nonnegative. Theta increases toward the positive minor axis in the right-handed `(u,v,normal)` frame. Landmark theta is 360-degree circular. Ellipse tilt is axial with a 180-degree period and is stored in `[-90,90)`.
5. **Angular extent:** each ordered source branch is mapped to nearest ellipse theta, unwrapped along source order, and terminal theta minus initial theta is the signed branch angular extent.
6. **Per-point deviation:** each source point stores normalized branch position `s`, corresponding ellipse reference XYZ, signed in-plane radial residual, signed plane-normal residual, and total Euclidean residual.

## Population model

191 cases have two numerically valid SVD planes and two actual ellipse fits and enter the measured population summaries. Linear variables report N, mean, sample standard deviation, median, IQR, range, P5 and P95. Landmark angles use directional circular statistics; ellipse tilts use axial circular statistics. Pearson correlations are supplied only as a diagnostic for selected continuous linear quantities.

## Source integrity

The measurement pipeline copies arrays in memory, never writes to the source archives, and verifies zero coordinate and segment-length change. SHA-256 hashes are stored for each source array and archive.
