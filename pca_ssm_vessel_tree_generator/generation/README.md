# LCA statistical generation

This package consumes the frozen LCA population artifacts and generates joined
LMCA/LAD/LCX anatomical scaffolds. It does not require or invent an RCA branch.

## Model contract

The primary local-deviation vector is 81-dimensional:

| Branch | Fixed points | Local values |
|---|---:|---:|
| LMCA | 5 | 15 |
| LAD | 12 | 36 |
| LCX | 10 | 30 |
| Total | 27 | 81 |

The frozen package under
`outputs/lca_ssm/lca_population_model/generator_statistics/` contains the
assignment gate, eligible ellipsoid and landmark statistics, fixed
surface-relative controls, PCA arrays, validation thresholds, and a SHA-256
manifest. Unresolved daughter assignments are never used for fitting or
sampling.

## Generation design

- `parameter_sampler.py`: jointly samples an empirical eligible ellipsoid row.
- `trajectory_sampler.py`: loads case-matched cardiac controls and local surface
  deviations.
- `deviation_sampler.py`: draws one coordinated PCA innovation; it adds no
  independent point noise.
- `surface_path_generator.py`: converts normalized-arc empirical samples into
  shape-preserving cubic Hermite/Bezier controls, assembles an explicit clamped
  SciPy `BSpline`, and densely evaluates it. Endpoints remain exact; the stable
  B-spline form avoids the hooks observed with an unconstrained global cubic.
- `tree_assembler.py`: assembles LMCA/LAD/LCX and snaps the shared bifurcation
  exactly.
- `validator.py`: performs topology, anatomy, continuity, progression, and
  physical-clearance checks.
- `vtk_export.py`: writes branch VTP files and a ParaView VTM hierarchy.

The empirical baseline, ellipsoid parameters, and PCA baseline come from the
same eligible source case. A low-scale PCA innovation supplies coordinated
variation around that baseline.

This is a bootstrap-based statistical generator. It is generative because the
joint PCA innovation makes every canonical tree non-identical to its training
baseline, but derivatives are not new independent patients. The final novelty
and holdout evidence quantifies this distinction.

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe pipeline.py generate --count 52 --pca-scale 0.04 --clean
```

The canonical defaults deliberately cover every one of the 52 eligible
empirical baselines once. The 0.04 innovation scale was selected by a controlled
population rerun: it removed the LMCA local-turn mismatch while preserving the
observed LAD/LCX length and tortuosity distributions.

The canonical output is `outputs/lca_ssm/lca_population_cohort/`. Every accepted
tree contains XYZ branch arrays, parameters and seed provenance, sampling
attempts, quantitative validation, fixed front/multiview previews, and VTK/VTM
geometry.

## Hard acceptance checks

- exact `LMCA[-1] == LAD[0] == LCX[0]` topology;
- LMCA is shorter than both major daughter branches;
- LAD is the dominant inferior/apex-directed branch;
- LCX remains less inferior and more lateral/crown-like than LAD;
- LCX cannot behave as the main apex-descending branch;
- 3D LMCA-to-LAD and LMCA-to-LCX continuity;
- nonlocal self-clearance, inter-branch clearance, and progression checks;
- observed hard population bounds.

Central 95% population intervals produce visible warnings instead of rejecting
all valid tail anatomy. Full-resolution centerline error is descriptive because
the output is a smooth scaffold rather than a reproduction of every noisy source
point.

Optional cardiac motion and radius tapers are prototype design defaults and are
labelled as such in exported metadata. The results are engineering prototypes,
not clinically validated geometries.
