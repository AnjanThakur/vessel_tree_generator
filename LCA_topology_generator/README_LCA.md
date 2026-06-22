# LCA Topology-First Control Point Generator

The LCA generator uses a topology-first workflow because the left coronary
artery starts as LMCA and then divides into LAD and LCX, with RI and other
side branches as future optional extensions. This differs from the RCA-style
workflow where a main vessel can often be generated first and side branches
attached later.

The current active topology is:

- LMCA root trunk
- LAD child branch connected at the LMCA bifurcation
- LCX child branch connected at the same LMCA bifurcation

Control points define the geometry, but the topology defines the parent-child
relationships and connection points.

## Generated Files

Running the generator writes outputs to `LCA_branch_control_points/generated/`
by default:

- `LMCA_ctrl_points.npy`
- `LAD_ctrl_points.npy`
- `LCX_ctrl_points.npy`
- `topology.json`
- `parameter_log.json`
- `validation_report.json`
- `LMCA_mean.npy` and `LMCA_std.npy`
- `LAD_mean.npy` and `LAD_std.npy`
- `LCX_mean.npy` and `LCX_std.npy`
- `population_statistics.json`
- `angle_distribution.json`
- `angle_distribution_plot.png`
- `side_branch_parametric_positions.json`
- `preview_plots/lca_template_0.png`

Each control-point matrix has shape `(K, 20, 3)`, where `K` is the number of
validated templates, `20` is the number of control points per branch, and `3`
stores x/y/z coordinates in millimeters.

## Mean And Std Matrices

The mean and standard deviation matrices are computed per branch across the
population axis:

```python
mean = np.mean(matrix, axis=0)
std = np.std(matrix, axis=0)
```

Each mean/std file has shape `(20, 3)`. These files summarize the population
of generated control-point templates; they are not final vessel centerlines.

## Angle Distribution

`angle_distribution.json` reports the generated LAD-LCX angle near the
bifurcation. The current tangent definition is:

- LAD tangent: `LAD_cp[3] - LAD_cp[0]`
- LCX tangent: `LCX_cp[3] - LCX_cp[0]`

The report includes target angles from `parameter_log.json` when available,
actual generated angles, actual mean/std/min/max, the angle tolerance, and
validation pass/fail counts.

## Side Branch Parametric Positions

`side_branch_parametric_positions.json` stores future optional side branch
attachment locations using normalized parent-vessel position `s`.

Examples:

- D branches attach to LAD over configured `s_range` intervals.
- OM branches attach to LCX over configured `s_range` intervals.
- RI is marked as a future optional trifurcation branch from the LMCA
  bifurcation.
- Left PLV and Left PDA are stored as future dominance-based LCX extensions.

These entries are configuration placeholders only; side branches are not
generated yet.

## Run

From the repository root:

```bash
python -m LCA_topology_generator.generate_lca --k 50 --candidates 1000 --seed 42
```

The defaults are also configurable:

```bash
python -m LCA_topology_generator.generate_lca --k 8 --candidates 100 --seed 42
```
