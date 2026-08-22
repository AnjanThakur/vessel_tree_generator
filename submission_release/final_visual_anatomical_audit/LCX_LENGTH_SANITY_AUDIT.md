# LCX Length Sanity Audit

**Status:** PASS_WITH_EXPLANATION

## Conclusion

Broader project range; endpoint and segmentation-extent definitions differ. The ten longest eligible source/training LCX paths and ten longest generated LCX paths retain the full selected segmented daughter path to its graph endpoint. Saved multi-signal assignments are consistent, all reviewed generated cases pass the production anatomy/topology gate, and no extraction/path or branch-swap error was found. The long tail is dataset-derived and must not be clamped to literature with a different terminal definition.

## Cause classification

- Endpoint-definition difference: supported.
- Segmentation extent: supported; the full daughter path is retained to its graph endpoint.
- Dataset-derived long tail: supported.
- Extraction/path error: not found.
- Branch-assignment error: not found.

The literature values are not used as clinical limits and no LCX length was clamped.

## Ten longest eligible source/training LCX paths

| Case | Raw arc (mm) | Fixed arc (mm) | Fixed chord (mm) | Assignment | Gate |
|---|---:|---:|---:|---|---|
| 143.label | 154.893 | 127.620 | 82.624 | verified multi-signal | eligible PASS |
| 118.label | 122.588 | 123.831 | 81.452 | verified multi-signal | eligible PASS |
| 76.label | 145.009 | 112.377 | 85.901 | verified multi-signal | eligible PASS |
| 156.label | 105.572 | 109.139 | 71.956 | verified multi-signal | eligible PASS |
| 195.label | 80.959 | 107.594 | 71.044 | verified multi-signal | eligible PASS |
| 94.label | 106.484 | 107.645 | 82.409 | verified multi-signal | eligible PASS |
| 91.label | 128.502 | 106.435 | 71.726 | verified multi-signal | eligible PASS |
| 153.label | 123.378 | 104.796 | 71.251 | verified multi-signal | eligible PASS |
| 37.label | 117.253 | 100.764 | 73.733 | verified multi-signal | eligible PASS |
| 61.label | 118.408 | 101.427 | 69.851 | verified multi-signal | eligible PASS |

## Ten longest generated LCX paths

| Tree | Source | Arc (mm) | Chord (mm) | Assignment | Production gate |
|---|---|---:|---:|---|---|
| tree_0036 | 143.label | 129.387 | 82.624 | verified multi-signal | PASS |
| tree_0027 | 118.label | 125.124 | 81.452 | verified multi-signal | PASS |
| tree_0019 | 76.label | 114.089 | 85.901 | verified multi-signal | PASS |
| tree_0041 | 156.label | 110.096 | 71.956 | verified multi-signal | PASS |
| tree_0023 | 94.label | 108.250 | 82.409 | verified multi-signal | PASS |
| tree_0051 | 195.label | 108.029 | 71.044 | verified multi-signal | PASS |
| tree_0022 | 91.label | 107.346 | 71.726 | verified multi-signal | PASS |
| tree_0039 | 153.label | 106.032 | 71.251 | verified multi-signal | PASS |
| tree_0016 | 61.label | 102.091 | 69.851 | verified multi-signal | PASS |
| tree_0010 | 37.label | 102.024 | 73.733 | verified multi-signal | PASS |

## Endpoint and graph-path definition

Dijkstra shortest-path tree from the radius-selected root; select an accepted major-daughter junction, then follow the chosen LCX daughter to its farthest descendant graph endpoint.

The complete selected segmented daughter path is retained to its graph endpoint. This can include more distal LCX geometry than a literature measurement with a different terminal definition.
