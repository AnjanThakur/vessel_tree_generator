# Generation Architecture Comparison

| Design step | Current method | Equivalent? | Impact |
|---|---|---|---|
| Sample scaffold | Joint empirical case-matched a,b,c bootstrap | Yes | Preserves cross-axis covariance and avoids pathological raw partial-arc tails |
| Sample surface landmarks | Joint empirical case-matched u,v,offset landmarks | Yes | LCA subset only |
| Build surface trajectory | Matched real fixed surface trajectory and exact cardiac controls | Yes | Empirical bootstrap replaces an independently sampled generic line |
| Add obliquity | Inherited from matched u trajectory and coordinated PCA | Equivalent | Avoids independent double counting |
| Add tortuosity | Inherited from empirical path and PCA; unlearned perturbation disabled | Equivalent | Avoids independent random XYZ noise |
| B-spline in u,v | Shape-preserving cubic B-spline in cardiac XYZ then surface re-parameterization | Partial | Literal candidate evaluated: 26/52 LAD baselines touch a pole; P95 max path difference 5.374 mm |
| Evaluate ellipsoid + deviation | Exact u,v,offset/local-basis reconstruction and joint 81-D PCA innovation | Yes | Full local-basis solve is numerically stronger than independent tangent dot products |
| Enforce bifurcation | Exact shared LMCA[-1]=LAD[0]=LCX[0] | Yes | Machine precision |
| Validate | Population/course/topology/collision gates | Yes | Surface-role audit is added as transparent evidence, not a new hard-coded textbook mold |

The final production path is therefore an **ellipsoid surface-relative statistical generator**, not an ellipse-arc generator. The B-spline chart is the one documented partial alignment.
