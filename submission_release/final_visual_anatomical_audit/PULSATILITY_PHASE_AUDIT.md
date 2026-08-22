# Pulsatility phase audit

Classification: **PHYSIOLOGICAL_MODEL_ISSUE — FIXED** (literature is mixed; motion and pulse phase are now explicit and independent).

1. **Phase 0** is the end-diastolic reference geometry.
2. **Phase 0.35** is peak modeled mechanical systolic contraction.
3. **Phase 1** repeats the phase-0 end-diastolic state and closes the cycle exactly.
4. Radial heart contraction is maximum at phase 0.35.
5. Coronary radius is maximum at the independent pulse peak, phase 0.60 (early diastole).
6. The radius model assumes modest early-diastolic lumen expansion with lesion-dependent attenuation; it does not reuse the mechanical contraction peak.
7. This implements the charter's request for a phase-offset/approximately phase-inverted diameter response.
8. Coronary IVUS evidence is mixed: PMID 7611122 reported about 2.1% diameter and 8.1% area expansion in mid/late systole, whereas PMID 8043342 reported maximum lumen area in early diastole and approximately 8–10% cyclic area change.
9. The final modeling assumption is explicit: **a conservative parametric early-diastolic expansion response based on PMID 8043342**, while documenting the contrary systolic-expansion observation in PMID 7611122. This is not patient-specific coronary mechanics.

Lesion compliance remains a **parametric scalar compliance approximation**, not patient-specific mechanical compliance.
