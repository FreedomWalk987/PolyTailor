# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""fallback5 + dopt FAST polish (guards=False), with a time budget.
fill(world, budget_s=None) -> dict(positions, faces). Raises FillRefused when no valid mesh is found or the budget
runs out. The dopt polish is skipped when less than ~15% of the budget remains, and is kept only if the strict
validation still passes and min SJ does not drop. NumPy and Python only."""
import time

import numpy as np

from . import fallback5 as fb
from . import dopt



def fill(world, budget_s=None, cancel=None):
    """cancel: the run's token (set / is_set): the fill refuses / the polish stops at its next check."""
    t0 = time.perf_counter()
    W = np.asarray(world, float)
    P, c, ex, ey = fb.frame(W)
    X, F, info = fb.fill2d(P, budget_s=budget_s, cancel=cancel)
    n = len(P)
    left = None if budget_s is None else budget_s - (time.perf_counter() - t0)
    if (left is None or left > 0.15 * budget_s) and (cancel is None or not cancel.is_set()):
        try:
            q0 = fb.mesh_min(X, F)
            Y = np.asarray(dopt.place(np.asarray(X, float), F, n, preset='FAST', guards=False,
                                      deadline=None if budget_s is None else t0 + budget_s,
                                      cancel=None if cancel is None else cancel.is_set), float)[:, :2]
            Y[:n] = P
            if fb.validate(Y, F, n, [tuple(p) for p in P]) is None and fb.mesh_min(Y, F) >= q0:
                X = Y
        except Exception:
            pass
    pos = c[None, :] + X[:, :1] * ex[None, :] + X[:, 1:2] * ey[None, :]
    pos[:n] = W
    return dict(positions=pos.tolist(), faces=[list(map(int, f)) for f in F])
