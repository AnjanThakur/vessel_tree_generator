# Visual validation pack

- Open `validation_dashboard.png` for the complete acceptance overview.
- Open `cardiac_cycle.gif` to inspect the closed 4D motion cycle.
- Open `interactive_tree.html` in a browser to rotate, zoom, hover, play, pause, and select phase.
- Open `tortuosity_analysis.png` for branch curvature and arc/chord metrics.
- Read `visualization_metrics.json` for the exact plotted numbers.

To change disease, generate a new case with `coronary4d generate` (or `python -m vessel_tree_generator generate`) and rerun the `visualize` command on that case. The viewer never modifies geometry or validation data.
