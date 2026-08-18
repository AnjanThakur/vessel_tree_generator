# Person 2 generator

This package owns the synthetic-generation side of the project. It consumes the Person 1 `surface_relative` API and statistical artifacts without modifying them.

Modules:

- `parameter_sampler.py`: joint empirical ellipsoid sampling with an explicit provisional fallback.
- `landmark_sampler.py`: circular `u` and bounded `v` landmark sampling.
- `surface_path_generator.py`: shape-preserving PCHIP paths from exact cardiac controls, projected through Person 1's ellipsoid functions; controlled/manual inputs retain ordinary B-splines.
- `trajectory_sampler.py`: exact case-matched cardiac controls and their local surface deviations.
- `deviation_sampler.py`: one joint PCA innovation draw around the matched baseline; no pointwise white noise.
- `tree_assembler.py`: LMCA/LAD/LCX assembly and exact bifurcation snapping.
- `validator.py`: exact topology, LAD/LCX anatomy, physical self/inter-branch clearance, progression/loop checks, and Person 1 population limits.
- `vtk_export.py`: modular VTP files and one ParaView VTM hierarchy.

Run the controlled Week-1 architecture demo from the repository root:

```powershell
.\.venv\Scripts\python.exe pca_ssm_vessel_tree_generator\run_person2_week1_demo.py
```

The default inputs are deliberately marked as controlled and not population-derived. Real-statistics mode refuses to start unless it receives ellipsoid rows/statistics, landmark statistics, fixed empirical trajectories, PCA arrays, and validation thresholds together.

After completing the Person 1 pipeline, generate one population-derived realization with:

```powershell
.\.venv\Scripts\python.exe pca_ssm_vessel_tree_generator\run_person2_week1_demo.py `
  --stats-dir outputs\lca_ssm\person1_week1\generator_statistics `
  --output-dir outputs\lca_ssm\person2_real_statistics_demo\tree_0001 `
  --clean --max-attempts 100
```

Real mode jointly matches the ellipsoid, landmarks, and exact fixed cardiac controls from one eligible source case. It applies only low-scale, centered PCA innovation to that shape-preserving baseline, preserves exact LMCA/LAD/LCX topology, and rejects candidates that violate population thresholds, self-clearance, or the shared LAD/LCX anatomy gate. Production mode refuses packages containing unresolved daughter assignments. The output VTM can be opened directly in ParaView.

## Complete population cohort

Generate the final 25-tree primary LCA cohort:

```powershell
.\.venv\Scripts\python.exe pca_ssm_vessel_tree_generator\run_person2_population_cohort.py `
  --stats-dir outputs\lca_ssm\person1_week1\generator_statistics `
  --output-dir outputs\lca_ssm\person2_population_cohort `
  --count 25 --pca-scale 0.08 --clean --max-attempts 1000
```

Compare it with the resolved, anatomy-gated real scaffold references:

```powershell
.\.venv\Scripts\python.exe pca_ssm_vessel_tree_generator\validate_person2_population_cohort.py `
  --person1-dir outputs\lca_ssm\person1_week1 `
  --cohort-dir outputs\lca_ssm\person2_population_cohort --clean
```

`synthetic_cohort.vtm` opens all trees as named ParaView blocks. Every tree directory includes XYZ arrays, surface coordinates, parameters, rejection history, quantitative validation, a fixed cardiac X-Z front view, a fixed three-view anatomy preview, VTP branches, landmarks, ellipsoid, and a tree-level VTM. The source schedule is balanced across every eligible baseline.

## Acceptance policy

- Authoritative branch resolution is a training-data prerequisite; unresolved LAD/LCX assignments never enter statistics or PCA.
- Topology, LAD-dominant inferior direction, LCX crown behaviour, LMCA-to-daughter continuity, collisions, physical self-clearance, and real-derived loop/progression limits are hard failures.
- Observed real min/max values are hard population limits.
- P2.5–P97.5 intervals are warnings rather than simultaneous hard gates, avoiding family-wise rejection of valid real tail anatomy.
- Empirical baselines interpolate exact matched cardiac controls with PCHIP before projection to surface coordinates, avoiding azimuth interpolation loops at ellipsoid poles.
- A low-scale coordinated PCA innovation is applied around the exact matched baseline. Independent point noise is never added.

The cohort report keeps descriptive warnings visible. In particular, smooth scaffolds are expected to have lower local turning and tortuosity than noisy full-resolution centerlines; these differences are reported rather than used to falsify individual validity. The final primary cohort intentionally excludes RCA: the available RCA paths are inferred disconnected candidates, so optional RCA support is retained for research/audit use but is not presented as resolved anatomy.
