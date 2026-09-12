# Per-tree quantitative reports

- `healthy_validation_report.docx`
- `focal_lad_validation_report.docx`
- `diffuse_lcx_validation_report.docx`
- `tandem_lad_validation_report.docx`

Every report is generated from the corresponding exported arrays and
`quantitative_validation.json`. It includes anatomy, branch tortuosity, disease
severity, equivalent circular area reduction, 4D displacement, global and
lesion-local pulsatility, validation evidence, limitations, and primary-source
literature context.

Regenerate the reports from the repository root with:

```powershell
.\.venv\Scripts\python.exe tools\build_tree_reports.py
```

These are research validation reports, not clinical reports.
