# Coronary4D 10-15 Minute Presentation Script

## 0:00-1:00 - Problem and final scope

**CLICK:** Project Overview.

**SAY:**

“Our project is Coronary4D, a population-derived, disease-aware 4D generator
for the major vessels of the left coronary artery. The validated topology is
LMCA branching into LAD and LCX. The system generates time-corresponded X, Y,
Z and radius values, controlled disease, cardiac motion, pulsatility, and VTK
files. This is a validated engineering prototype; we do not claim clinical
validation.”

**POINT TO:** Engineering PASS and Clinical Validation Not Claimed cards.

**SAY:**

“The original charter mentioned RCA. The available RCA components were not
trusted annotated ground truth, so we excluded RCA rather than make an
unsupported population claim.”

**NEXT:** Dataset & Cohort.

## 1:00-2:15 - Real-data funnel

**POINT TO:** 200 -> 191 -> 181 -> 65 -> 52 funnel.

**SAY:**

“We started with 200 NIfTI coronary label volumes. 191 produced the original
two-plane, two-ellipse LCA records. LAD and LCX identity was confidently
resolved in 181. Sixty-five passed the core source and scaffold anatomy gate.
After ellipsoid, integrity, and fixed-representation checks, 52 cases formed
the PCA cohort.”

**SAY:**

“We did not weaken gates to increase sample size. The synthetic trees are
derivatives of these 52 anatomies; they do not increase the independent patient
count.”

**NEXT:** Anatomical Model.

## 2:15-3:20 - Two planes and two ellipses

**POINT TO:** Two-plane/two-ellipse figure and complete-model visual.

**SAY:**

“SVD answers: what is the best 3D plane for the measured points? Ellipse fitting
then answers: what are the semi-axes, tilt, angular extent, and residuals inside
that plane? We model a crown or coronary plane and an LAD or interventricular
plane.”

**SAY:**

“The real artery is not the ellipse. The ellipse is reference geometry. Across
77,337 pointwise residual records, the maximum source coordinate change and
segment-length change are both zero millimetres.”

**NEXT:** Statistical Shape Model.

## 3:20-4:25 - PCA statistical anatomy

**POINT TO:** 5 + 12 + 10 = 27 and 27 x 3 = 81.

**SAY:**

“Every eligible LCA tree is represented by five LMCA, twelve LAD, and ten LCX
corresponding points. Each point has X, Y and Z, so every patient becomes an
81-value shape vector. PCA learns coordinated anatomical changes rather than
moving points independently.”

**POINT TO:** Cumulative variance chart.

**SAY:**

“The final matrix is 52 by 81. Thirteen modes retain 95.55 percent of variation
in the eligible cohort. This is cohort-specific statistical coverage, not a
claim about all patients.”

**NEXT:** Synthetic Generation.

## 4:25-5:45 - Synthetic generation and B-splines

**POINT TO:** Generation pipeline and 52-tree montage.

**SAY:**

“A generated case selects a matched eligible statistical anatomy, adds a
conservative joint PCA innovation, reconstructs LMCA, LAD and LCX samples, and
uses a genuine composite cubic B-spline. Branch terminals and the shared
bifurcation are exact: LMCA end equals both daughter starts.”

**SAY:**

“The B-spline reproduces the previously accepted stable path to about 2.84
times ten to the minus fourteen millimetres. Unconstrained cubic alternatives
were rejected when they introduced hooks or invalid turns.”

**CLICK:** Generate Demo Case, using the default safe settings.

**WAIT FOR:** PASS result and preview.

**SAY:**

“The new case is written only into the presentation runtime folder. The frozen
statistics are read, never edited.”

**NEXT:** Disease Model.

## 5:45-7:00 - Disease

**POINT TO:** Healthy, focal LAD, diffuse LCX, and tandem LAD comparison.

**SAY:**

“These four cases share identical anatomy. Healthy has no lesion. Focal is a
localized smooth narrowing. Diffuse has a longer narrowed region and smooth
transitions. Tandem contains separated focal lesions.”

**POINT TO:** Zero XYZ difference card.

**SAY:**

“Disease changes radius only. The maximum healthy-versus-disease XYZ difference
is zero millimetres, so geometry and disease severity remain independently
controllable.”

**NEXT:** 4D Cardiac Motion.

## 7:00-8:25 - Motion

**WAIT FOR ANIMATION:** Observe one closed cardiac cycle.

**SAY:**

“Motion combines radial contraction, longitudinal shortening, and base-to-apex
torsion on an anatomical support surface. We do not randomly move each vessel
point. Corresponding points and branch junctions are preserved across phases.”

**CLICK:** Play on the phase scrubber, then pause and move the slider.

**SAY:**

“The maximum demonstrated displacement is 7.8104 millimetres. Ten frames are
stored: nine independent positions plus a repeated closure frame at phase one.”

**NEXT:** Pulsatility & Compliance.

## 8:25-9:25 - Pulsatility and lesion compliance

**POINT TO:** Radius-change curves.

**SAY:**

“Healthy global radius change is 2.9916 percent. A lesion is modeled as less
compliant: focal LAD changes by 1.0483 percent, diffuse LCX by 1.0471 percent,
and tandem LAD by 1.1706 percent. These are about 35 to 39 percent of the
healthy response.”

**POINT TO:** Parametric Compliance Model - Not FSI.

**SAY:**

“This is a scalar parametric compliance model, not full biomechanics or
fluid-structure interaction.”

**NEXT:** Output Format, then VTK.

## 9:25-10:35 - Output and ParaView

**POINT TO:** Tensor layout.

**SAY:**

“The main array has shape 10 by 3 by 50 by 4: phase, branch, point, and
coordinate. Coordinates are X, Y, Z, and radius in millimetres. This is why we
write Tree of t equals x of t, y of t, z of t, and r of t.”

**NEXT:** VTK / ParaView.

**SAY:**

“The single ParaView entry is vtk/cine.pvd. It references one VTM multiblock
tree per phase, containing LMCA, LAD and LCX VTP blocks. We can color by radius
or disease reduction and play the time series. Independent VTK readback passed.”

## 10:35-12:00 - Quantitative validation

**NEXT:** Quantitative Validation.

**POINT TO:** Large validation cards and distribution overlays.

**SAY:**

“All 52 requested population trees were accepted after 64 candidate attempts.
All 52 eligible baselines are represented. Fifty out of fifty real-versus-
generated comparisons passed with zero warnings. The checks include lengths,
angles, tortuosity, landmarks, scaffold parameters, and PCA scores.”

**SAY:**

“Matching the eligible-cohort distributions is engineering evidence. It is not
clinical validation.”

## 12:00-13:10 - Novelty and holdout

**NEXT:** Novelty & Holdout.

**SAY:**

“No canonical output is an exact training duplicate and none lies below the
0.1-millimetre near-duplicate threshold. We quantify nearest-training distance,
baseline RMS displacement, and PCA-space displacement.”

**POINT TO:** Holdout result.

**SAY:**

“The source-grouped five-fold holdout accepts 41 of 52 first draws, or 78.85
percent, with zero source and baseline leakage. This is anatomical generation
acceptance, not classification accuracy and not external validation.”

## 13:10-14:10 - Verification and limitations

**NEXT:** Reproducibility / Tests.

**CLICK:** Run Release Verification.

**SAY:**

“The fast verifier checks manifests, frozen statistics hashes, case checksums,
phase-zero identity, topology, VTK entry points, and internal metric consistency.
The regression record contains 90 passes and zero failures.”

**NEXT:** Limitations.

**SAY:**

“The final release is major-vessel LCA only. It has no population-derived RCA,
side branches, external cohort, hemodynamics, myocardium simulation, or clinical
validation. PCA is linear and radii, taper, disease, motion, pulsatility, and
compliance are parametric.”

## 14:10-15:00 - Close

**NEXT:** Live Demo.

**SAY:**

“Coronary4D therefore provides a reproducible chain from protected real LCA
measurements to statistical anatomy, smooth synthetic generation, controlled
disease, closed-cycle 4D motion, XYZ-radius output, ParaView export, and
quantitative engineering validation—while keeping its clinical and anatomical
scope explicit.”

**STOP. Invite questions.**
