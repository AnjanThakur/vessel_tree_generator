# Historical archive boundary

The final-deliverable branch intentionally excludes historical bulk outputs,
duplicate evidence packs and non-production RCA/prototype utilities. Nothing
was discarded: the complete pre-cleanup state is preserved on
`latestt_branchh` at commit `7b21ce4`.

## Historical outputs

- `outputs/lca_ssm/person1_week1/`
- `outputs/lca_ssm/person2_population_cohort/`
- `outputs/lca_ssm/ppt_priority_completion/`
- `outputs/lca_ssm/stage1_final_anatomical_model/`
- `outputs/lca_ssm/stage1_parametric_heart_surface/`
- `outputs/lca_ssm/stage1_heart_scaffold_vtk/`
- smoke/tortuosity trial outputs

## Superseded release copies

- `submission_release/final_visual_anatomical_audit/`
- `submission_release/mentor_visualization_pack/`
- `submission_release/original_ppt_evidence/`
- standalone 5 MB Plotly HTML copies inside individual demo cases

Their final conclusions remain represented by the compact frozen statistics,
final audit, final validation, presentation center, project report and the
verified all-52 presentation package.

`submission_release/design_spec_alignment/` remains in the deliverable because
its CSV/JSON evidence is consumed by the nine independent design-alignment
contract tests.

## Excluded code

- The disconnected RCA prototype is not part of the validated LCA model.
- Required radius, disease and tube-mesh functions were consolidated into
  `vessel_tree_generator` before the legacy LCA shell was archived.
- One-off audit/report/presentation builders whose large source evidence was
  archived were moved to `_local_archive/tools/legacy/`. The supported tools
  directory retains release tests, per-case report generation and all-52
  presentation generation.

To inspect any historical path without changing the final branch:

```powershell
git show latestt_branchh:<path>
```

To restore the complete historical working tree, switch back to
`latestt_branchh`.
