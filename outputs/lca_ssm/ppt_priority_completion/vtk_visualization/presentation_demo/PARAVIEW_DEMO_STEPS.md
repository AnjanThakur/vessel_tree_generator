# ParaView demo steps

1. Open `91_label_ppt_two_plane_two_ellipse_model.vtm` in ParaView and click **Apply**.
2. In the Pipeline Browser, expand the multiblock dataset and use **Block Colors Distinct Values** if desired.
3. Show the three source branches first; use line width 4–6 and keep the original RAS axes visible.
4. Add `CORONARY_SVD_PLANE` and `LAD_SVD_PLANE` with opacity around 0.15–0.25.
5. Add the two fitted ellipses with line width 5–7.
6. Show `LANDMARKS` using Point Gaussian or Glyph representation.
7. Finally show the residual blocks, color by `euclidean_residual_mm`, and use a thin tube only for presentation visibility.

Narration: The SVD plane determines the anatomical plane orientation. The ellipse is independently fitted inside that plane. The source artery is measured but never replaced. The fitted ellipse is a statistical reference. The residual is the source artery's deviation from that reference. The RCA component is an inferred candidate, not annotated ground truth.
