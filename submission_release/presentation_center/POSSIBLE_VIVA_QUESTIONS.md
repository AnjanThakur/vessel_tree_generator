# Coronary4D Possible Viva Questions

## Data and anatomy

1. **Why did only 52 of 200 labels enter PCA?** 200 were discovered, 191 yielded
   the original LCA measurement records, 181 had confident daughter identities,
   65 passed core anatomy, and 52 passed the remaining ellipsoid, integrity, and
   fixed-representation gates. Gates were not weakened to inflate N.
2. **Does generating 52 trees mean there are 104 patients?** No. Synthetic trees
   are derivatives of 52 eligible source anatomies and do not increase independent N.
3. **Why is LAD/LCX assignment important?** LAD should be the dominant descending
   branch and LCX the lateral crown-like branch; reversing them corrupts branch-specific statistics.
4. **Was branch identity selected from radius alone?** No. Assignment used combined
   anatomy including displacement, direction, length, descent, and crown behavior.
5. **Were source centerlines changed?** No. Maximum source coordinate and segment-
   length changes are both 0.0 mm.
6. **What is SVD here?** Singular value decomposition identifies the best-fitting
   plane and its orthogonal directions for 3D points.
7. **What is the difference between SVD and ellipse fitting?** SVD finds a plane;
   ellipse fitting estimates semi-axes, tilt, angular course, and residuals inside it.
8. **Is the real artery assumed to be an ellipse?** No. The ellipse is reference
   geometry; measured points retain residual offsets from it.
9. **Why use two planes?** LAD follows an interventricular descending course while
   LMCA/LCX follow a more crown-like course; one plane would collapse this relationship.
10. **What are landmark theta and angular extent?** Theta locates a landmark around
    the fitted ellipse; angular extent measures the branch's covered ellipse arc.
11. **Why use an ellipsoid support surface?** It provides a common cardiac-relative
    frame for surface coordinates and coherent deformation.
12. **What are u, v, and offset?** They are angular/radial surface-relative
    coordinates plus signed local offset, allowing exact reconstruction in the common frame.
13. **Is that an exact nearest-point projection?** No. It is an angular/radial
    ellipsoid parameterization; the code and audit name the distinction explicitly.
14. **How accurate is surface reconstruction?** 156,690 audited records have a
    maximum reconstruction error of about 2.17e-12 mm.

## Statistical shape model and generation

15. **Why 27 points?** Fixed correspondence uses 5 LMCA, 12 LAD, and 10 LCX
    anatomical samples, retaining branch identity and manageable dimensionality.
16. **Why 81 dimensions?** Twenty-seven points times three XYZ coordinates equals 81.
17. **Why PCA?** PCA captures coordinated linear anatomical variation in a compact,
    reproducible model and supports controlled sampling.
18. **Why 13 modes?** Thirteen modes are the smallest retained set reaching the
    configured variance target for the eligible cohort.
19. **Why retain approximately 95% variance?** It balances population variation
    against high-order noise and instability; actual retained variance is 95.5469%.
20. **Is PCA clinically representative?** No. It represents only the 52 eligible
    anatomies and is not an external or population-prevalence estimate.
21. **Why not use every point in PCA?** Fixed anatomical correspondence reduces
    noise, unequal sampling, and dimensionality while preserving named branch structure.
22. **What is PCA innovation scale 0.04?** It is the conservative final amplitude
    for joint PCA deviation around matched empirical anatomy, calibrated against acceptance and distributions.
23. **Are generated trees simply copied baselines?** No. Canonical outputs have
    nonzero baseline displacement, zero exact duplicates, and zero near duplicates under 0.1 mm.
24. **How many canonical trees were generated?** 52 accepted trees in 64 candidate attempts.
25. **Why exactly 52 canonical outputs?** The release represents each eligible
    baseline once, enabling paired coverage rather than oversampling a subset.
26. **Why a B-spline?** It gives smooth continuous centerlines with explicit basis
    evaluation while preserving exact endpoints and junctions.
27. **Is the final path a real B-spline or a renamed interpolator?** It is a genuine
    composite cubic B-spline constructed from exact cubic Bezier-equivalent controls and knots.
28. **Why not use one global cubic spline?** Trials introduced hooks and invalid
    local turns; the composite shape-preserving construction satisfied anatomy gates.
29. **How is the bifurcation preserved?** LMCA terminal, LAD start, and LCX start
    are explicitly identical before validation and checked at every motion phase.
30. **What does 2.84e-14 mm mean?** It is floating-point-scale numerical difference
    between the real B-spline evaluation and the previously accepted stable path.

## Disease, motion, and output

31. **What is focal stenosis?** A localized smooth fractional radius reduction.
32. **What is diffuse stenosis?** A longer narrowing with smooth entry, plateau,
    and exit behavior.
33. **What is tandem disease?** Multiple separated focal lesions on a branch.
34. **Does disease bend the artery?** No. Disease changes radius only; healthy and
    diseased XYZ coordinates remain identical before motion.
35. **Is severity radius reduction or area reduction?** The input severity is
    fractional radius reduction. Equivalent area reduction is derived, not treated as identical.
36. **How is cardiac motion generated?** Ellipsoid-relative radial contraction,
    longitudinal shortening, and base-to-apex torsion deform corresponding points coherently.
37. **Why is maximum displacement about 7.8 mm?** It is the maximum observed under
    the explicit demonstration parameters, not a universal clinical limit.
38. **What is pulsatility?** Cyclic lumen radius variation over cardiac phase.
39. **What is lesion compliance?** A scalar reduction in local radius response at
    stenotic points relative to the healthy cyclic amplitude.
40. **Is compliance full biomechanics?** No. It is parametric and is not FSI,
    wall mechanics, pressure coupling, or anisotropic deformation.
41. **Why store 10 frames if only nine are independent?** The tenth repeats the
    endpoint at phase one so exported animation closes exactly.
42. **How is temporal mismatch prevented?** The same normalized static-reference
    point correspondence and branch order are reused at every phase.
43. **What exactly is Tree(t)?** A time-indexed coronary representation containing
    x(t), y(t), z(t), and radius r(t) for each branch point.
44. **What is the main tensor shape?** The release demo uses `[10, 3, 50, 4]`:
    phase, branch, point, and XYZ-radius coordinate.
45. **Why include a static array?** It is the phase-zero fixed-size reference and
    provides a simple baseline for consumers that do not require the cine tensor.

## Validation, VTK, and limitations

46. **What does 50/50 population validation prove?** Audited real-versus-generated
    comparisons for key eligible-cohort distributions pass. It does not prove clinical equivalence.
47. **What does 78.85% mean?** It is first-draw anatomical acceptance in an internal
    source-grouped holdout, not classification accuracy.
48. **How was leakage controlled?** Held-out sources and their baselines were blocked
    from training and baseline access; source and baseline leakage counts are zero.
49. **Is the holdout external validation?** No. It tests internal representation
    stability and leakage control within the available dataset.
50. **What is VTK?** A scientific visualization data format family. This release
    uses PVD time series, VTM multiblock trees, and VTP branch polydata.
51. **Why ParaView?** It provides interactive time playback, block inspection,
    tube rendering, and coloring by radius or disease arrays.
52. **How do you know VTK exports are valid?** The exporter reads back written
    files, and the final VTK audit is PASS for all demo cases.
53. **Why is RCA excluded?** Available RCA candidates were not trusted annotated
    ground truth. Excluding them prevents an unsupported population-derived claim.
54. **What side branches are missing?** Diagonal, septal, obtuse marginal, and
    other smaller branches are outside the final topology.
55. **What are the main PCA limitations?** Linear variation, small eligible N,
    selection bias, and limited ability to represent nonlinear or multimodal anatomy.
56. **What are the main physiology limitations?** Parametric radius, taper,
    stenosis, motion, pulsatility, and compliance; no hemodynamics or wall mechanics.
57. **Is the system clinically validated?** No. It is a research/engineering
    prototype with quantitative internal engineering validation.
58. **What should be improved next?** Add expert-reviewed RCA/side branches,
    expand source cohorts, obtain external validation, learn physiology from metadata,
    investigate nonlinear models, and separately validate wall/hemodynamic modules.
59. **How is reproducibility checked?** Fixed seed/configuration, stored metadata,
    SHA-256 manifests, repeatability evidence, tests, and read-only verification.
60. **What is the strongest honest project claim?** A reproducible, population-
    derived major-vessel LCA engineering generator with controlled disease, 4D
    motion, XYZ-radius output, VTK export, and transparent internal validation.
