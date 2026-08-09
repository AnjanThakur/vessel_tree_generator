# 91.label — PPT two-plane + two-ellipse visualization

Open `06_combined/ppt_two_plane_two_ellipse_model.vtm` in ParaView.

The SVD plane determines the anatomical plane orientation. The finite plane polygons are display patches only; they are not fitted ellipses. The ellipse is independently fitted inside that plane. The source artery is measured but never replaced. The fitted ellipse is a statistical reference. The residual is the source artery's deviation from that reference.

The RCA component is an inferred candidate, not annotated ground truth. This case contains that inferred candidate.

All coordinates are the saved NIfTI RAS coordinates in millimetres. The source `.vtp` files contain exactly one polyline with the original point order and point count. Residual connectors are a display subsample (every 10th saved correspondence plus the terminal row); their scalar arrays retain the saved residual values.

Selection note: Explicitly required representative case 91.label.
