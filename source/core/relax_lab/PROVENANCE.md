# Embedded Relax Lab 0.2.0

Origin: the numerical solver of the author's Relax Lab 0.2.0.
Copyright (C) 2026 FreedomWalk. License: GPL-3.0-or-later; the full text is in LICENSE in this directory.

What is used: the Python/NumPy solver only, that is the RobustSolver corner-Jacobian objective with its
quad / triangle targets and PatchSolver support (solver.py, robust.py, targets.py). PolyTailor calls it through
its core/smooth.py adapter and supplies its own brush masks, pin protection, source projection, mirror clipping
and Move / Slide / Lock boundary handling (default Move). The standalone tool's registration, UI, spatial-index
brush, dense preview and scene / reference settings are not included.

In this copy RobustSolver reuses initial face calculations and caches immutable active-corner index / frame
arrays; the optimization equations, step loop and acceptance checks are unchanged.

## Method

The solver measures all four corner Jacobians of a quad and all three of a triangle. Quad targets keep the
average lengths of each opposite edge pair while removing skew; triangles use equilateral targets. The objective
uses a regularized signed determinant:

```
chi(D, epsilon) = (D + sqrt(D*D + epsilon*epsilon)) / 2
E(J) = (||J||_F^2 + 0.05 * (D*D + 1)) / chi(D, epsilon)
```

A small L-BFGS history accelerates relaxation. Proposals are bounded by local edge scale and brush influence,
and backtracking requires energy improvement with positive corner determinants along the accepted step. This is
local fixed-topology optimization, not a global untangling guarantee. Vertices outside the brush
patch stay fixed; mesh-border vertices follow PolyTailor's Move / Slide / Lock boundary setting.

## Research provenance

Original Python/NumPy implementation by the author. Theory informing the design:

- Garanzha et al., *Foldover-free maps in 50 lines of code* (2021):
  https://arxiv.org/abs/2102.03069
- Knupp, *Algebraic Mesh Quality Metrics* (2001):
  https://epubs.siam.org/doi/abs/10.1137/S1064827500371499
- Bouaziz et al., *Shape-Up* (2012):
  https://infoscience.epfl.ch/record/183433/files/shapeup.pdf

The regularized-determinant objective is adapted to a local brush with retained aspect targets and explicit
movement safeguards; it is not a reproduction of any complete paper solver.

Copyright (C) 2026 FreedomWalk. GPL-3.0-or-later; see LICENSE.
