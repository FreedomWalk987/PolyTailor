# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Strip-smoothed quad targets for RobustSolver (Relax setting "Even Out Spacing").

RobustSolver's quad target is a rectangle with the quad's own two widths (averages of its
opposite edge pairs) at stroke start, so irregular spacing is kept like intended detail.
Here each width is averaged, in log scale, with the same width of the quads before and
after it in its strip (the quads across the two edges that width is measured along), a
few passes, with an edge-aware weight. Widths across strips are never mixed: holding
loops (narrow strips) and density gradation stay; per-quad noise goes. The idea follows
reference-Jacobian rezoning (Knupp, Margolin & Shashkov 2002: targets from smoothed
Jacobians of the current mesh).
"""
import numpy as np

PASSES = 8
SIGMA = .35          # log-width difference at which a neighbour's weight falls to 1/e


def smooth_strip_targets(solver, passes=PASSES, sigma=SIGMA):
    ids, owners, pts = solver.ids, solver.corner_owners, solver.points
    quads, i = [], 0
    while i < len(ids):
        n = len(set(owners[i].tolist()))
        if n == 4:
            quads.append((i, [int(ids[i + k, 0]) for k in range(4)]))
        i += n
    if not quads:
        return
    edge_of = {}
    for qi, (_row, q) in enumerate(quads):
        for k in range(4):
            edge_of.setdefault(frozenset((q[k], q[(k + 1) % 4])), []).append((qi, k))
    L = np.array([[np.linalg.norm(pts[q[(k + 1) % 4]] - pts[q[k]]) for k in range(4)] for _r, q in quads])
    w = np.log(np.maximum(np.stack(((L[:, 0] + L[:, 2]) / 2, (L[:, 1] + L[:, 3]) / 2), axis=1), 1e-12))
    # width p of quad qi is measured along edges p and p+2; the strip continues across them
    nb_q, nb_p, own_q, own_p = [], [], [], []
    for qi, (_row, q) in enumerate(quads):
        for k in range(4):
            for qj, kj in edge_of[frozenset((q[k], q[(k + 1) % 4]))]:
                if qj != qi:
                    own_p.append(k % 2)
                    nb_p.append(kj % 2)
                    own_q.append(qi); nb_q.append(qj)
    own_q, own_p, nb_q, nb_p = map(np.array, (own_q, own_p, nb_q, nb_p))
    for _ in range(passes):
        if not len(own_q):
            break
        diff = w[nb_q, nb_p] - w[own_q, own_p]
        a = np.exp(-(diff / sigma) ** 2) if sigma else np.ones_like(diff)
        tot, ws = w.copy(), np.ones_like(w)
        np.add.at(tot, (own_q, own_p), a * w[nb_q, nb_p])
        np.add.at(ws, (own_q, own_p), a)
        w = tot / ws
    uv = np.exp(w)
    inverse = solver.inverse.copy()
    for qi, (row, _q) in enumerate(quads):
        for k in range(4):
            inverse[row + k] = np.diag((1 / uv[qi, k % 2], 1 / uv[qi, 1 - k % 2]))
    solver.inverse = np.ascontiguousarray(inverse)
    solver._metric_cache = None
