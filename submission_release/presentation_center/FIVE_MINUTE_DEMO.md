# Coronary4D Five-Minute Demo

## Minute 1 - Problem and pipeline

**CLICK:** Project Overview.

**SAY:** “Coronary4D generates a major-vessel left coronary tree: LMCA into LAD
and LCX. It learns eligible population shape, adds controlled disease, cardiac
motion and cyclic radius, then exports time-corresponded XYZ-radius arrays and
VTK. Engineering validation passes; clinical validation is not claimed.”

**POINT TO:** Complete pipeline and PASS / Not Clinically Validated cards.

## Minute 2 - Statistical anatomy

**CLICK:** Dataset & Cohort, then Statistical Shape Model.

**SAY:** “The verified funnel is 200 source labels, 191 extracted LCA records,
181 resolved LAD/LCX assignments, 65 core anatomy passes, and 52 PCA-eligible
cases. Five LMCA, twelve LAD and ten LCX points create 27 corresponding points,
or 81 XYZ dimensions. Thirteen PCA modes retain 95.55 percent of eligible-cohort
variation.”

## Minute 3 - Generation and disease

**CLICK:** Synthetic Generation, then Disease Model.

**SAY:** “A matched anatomy plus conservative PCA innovation is converted into
genuine composite cubic B-spline branches with exact bifurcation. The four demo
cases share identical XYZ geometry. Focal, diffuse and tandem disease change
radius only; maximum healthy-versus-disease XYZ difference is zero.”

## Minute 4 - Motion and pulsatility

**CLICK:** 4D Cardiac Motion and press Play. Then Pulsatility & Compliance.

**SAY:** “Radial contraction, longitudinal shortening, and torsion generate a
closed cardiac cycle. Ten stored frames mean nine independent positions plus
closure. Maximum displacement is 7.8104 mm. Healthy radius change is 2.9916
percent; lesions respond at about 35 to 39 percent of healthy. This is
parametric compliance, not FSI.”

## Minute 5 - Validation and limitations

**CLICK:** Quantitative Validation, then Limitations.

**SAY:** “Fifty-two of 52 trees were accepted after 64 attempts. Fifty of 50
population comparisons passed, with zero warnings. Novelty found zero exact or
near duplicates. Internal holdout accepted 41 of 52 with zero leakage. Ninety
tests passed. The release remains LCA-only, has 52 effective source anatomies,
uses linear PCA and parametric physiology, and is not clinically validated.”

**END:** “The full evidence and a read-only verifier are available inside this
presentation center.”
