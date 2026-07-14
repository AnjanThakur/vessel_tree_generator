# Anatomical LCA Pipeline Summary

## Counts

- Total anatomical trees discovered: 5
- Total anatomical trees processed: 3
- Valid centerline outputs: 3
- Valid radius outputs: 3
- Valid normal tight mesh outputs: 3
- Disease cases generated: 9
- Valid diseased cases: 9
- Valid diseased tight mesh cases: 9

## Average Anatomical Scores

- Average LAD downward score: 0.9789804197376051
- Average LCX lateral score: 1.0
- Average anatomical score: 1.0

## Failed Cases

- patient_0002: centerline_validation - ['LAD control path length 23.352 mm is below 30.000 mm']
- patient_0003: centerline_validation - ['LMCA control path length 3.641 mm is below 5.000 mm']

## Limitations

- This is MVP anatomical realism, not clinical reconstruction.
- Vessels are not fitted to a real heart surface.
- Patient-specific apex, base, and groove landmarks are not available.
- The corrected geometry is rule-based and designed to remain compatible with the current radius, disease, tube, and mesh pipeline.
