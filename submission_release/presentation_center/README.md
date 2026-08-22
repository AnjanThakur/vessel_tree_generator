# Coronary4D Presentation & Verification Center

## Start

PowerShell:

```powershell
.\START_PRESENTATION.ps1
```

From the repository root, the equivalent command is:

```powershell
.\.venv\Scripts\python.exe -m vessel_tree_generator present
```

The application opens at `http://127.0.0.1:8765`. It uses Python's local HTTP
server rather than Streamlit because Streamlit is not installed in the project
environment. No new runtime dependency is required.

The 15-section presentation explains the protected source data, cohort funnel,
two-plane/two-ellipse anatomical scaffold, 27-point statistical model,
synthetic B-spline generation, radius-only disease, 4D motion, pulsatility,
XYZ-radius tensor, VTK/ParaView workflow, quantitative validation, novelty,
holdout, engineering QA, and limitations. All numbers are loaded from the final
release evidence; no dashboard metric is an independently maintained copy.

Fast and full verification are read-only with respect to frozen evidence. Live
generation writes only to `runtime_demo/`, which may be safely recreated. The
validated statistical scope is major-vessel LCA (`LMCA -> LAD + LCX`). RCA and
side branches are not part of the population-derived release. The project is a
research/engineering prototype and is not clinically validated.

If the application cannot be started, open `offline/index.html`. Presenter
scripts, the viva guide, cheat sheet, ParaView instructions, and the independent
verification guide are stored beside this README.
