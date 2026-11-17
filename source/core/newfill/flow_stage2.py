# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""The flow stage's smoothing (NumPy), one colour class at a time, rim fixed; the result does not depend on
the deadline unless the deadline binds.

How the inner loop stays cheap without changing a floating-point operation on the positions:
  * neighbour sums: each colour class has its own (vertex, neighbour) list in a fixed accumulation order, summed
    with np.bincount (no np.add.at over all edges per class);
  * the per-vertex "did one of my faces fall below the floor" test is a boolean scatter, not a Python set loop.
smooth(mesh, sweeps, lam, deadline_s, halvings) -> mesh (rim fixed); no face may drop below the start's min SJ."""
import time
import numpy as np


def _plane(p, n):
    rim = p[:n]; c = rim.mean(0)
    nrm = np.sum([np.cross(rim[i] - c, rim[(i + 1) % n] - c) for i in range(n)], 0)
    return nrm / max(np.linalg.norm(nrm), 1e-30)


def smooth(mesh, sweeps=50, lam=0.5, deadline_s=None, halvings=4, cancel=None):
    t0 = time.perf_counter()
    P = np.asarray(mesh['positions'], float); n = int(mesh['boundary']); cells = [list(f) for f in mesh['faces']]
    if len(P) <= n or not cells:
        return dict(mesh)
    nrm = _plane(P, n)
    groups = {}
    for fi, f in enumerate(cells):
        groups.setdefault(len(f), []).append(fi)
    G_ = [(np.array(ix), np.array([cells[i] for i in ix])) for ix in groups.values()]

    def csj(x):
        out = np.empty(len(cells))
        for ix, F in G_:
            X = x[F]; a = np.roll(X, -1, 1) - X; b = np.roll(X, 1, 1) - X
            out[ix] = ((np.cross(a, b) @ nrm) / np.maximum(np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2), 1e-30)).min(1)
        return out
    E = set()
    for f in cells:
        for i in range(len(f)):
            E.add((min(f[i], f[i - 1]), max(f[i], f[i - 1])))
    E = np.array(sorted(E)); deg = np.bincount(E.ravel(), minlength=len(P)).astype(float)
    vfaces = [[] for _ in range(len(P))]
    for fi, f in enumerate(cells):
        for v in f:
            vfaces[v].append(fi)
    inner = [v for v in range(n, len(P)) if deg[v] > 0]
    colour = {}
    for v in inner:
        used = {colour[u] for fi in vfaces[v] for u in cells[fi] if u in colour}
        c = 0
        while c in used:
            c += 1
        colour[v] = c
    classes = {}
    for v, c in colour.items():
        classes.setdefault(c, []).append(v)
    classes = [np.array(vs) for _, vs in sorted(classes.items())]
    # the original's accumulation order per vertex: all edges with the vertex first (E order), then all with it second
    tgt = np.concatenate([E[:, 0], E[:, 1]]); src = np.concatenate([E[:, 1], E[:, 0]])
    cls_of = np.full(len(P), -1); loc = np.zeros(len(P), int)
    for ci, vs in enumerate(classes):
        cls_of[vs] = ci; loc[vs] = np.arange(len(vs))
    cls_faces, cls_nb = [], []
    for ci, vs in enumerate(classes):
        fl = []; owner = []
        for v in vs:
            for fi in vfaces[v]:
                fl.append(fi); owner.append(v)
        fl = np.array(fl); owner = np.array(owner); sub = {}
        for j, fi in enumerate(fl):
            sub.setdefault(len(cells[fi]), []).append(j)
        grp = [(np.array(js), np.array([cells[fl[j]] for j in js])) for js in sub.values()]
        cls_faces.append((fl, loc[owner], grp))
        m = cls_of[tgt] == ci
        cls_nb.append((loc[tgt[m]], src[m]))

    def csj_sub(x, grp, m):
        out = np.empty(m)
        for js, F in grp:
            X = x[F]; a = np.roll(X, -1, 1) - X; b = np.roll(X, 1, 1) - X
            out[js] = ((np.cross(a, b) @ nrm) / np.maximum(np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2), 1e-30)).min(1)
        return out
    s0 = csj(P)
    if float(np.median(s0)) < 0:   # fix: a clockwise mesh (the S lane's) flips the SJ sign -> no floor at all
        nrm = -nrm
        s0 = csj(P)
    floor = float(s0.min())
    x = P.copy()
    for _ in range(sweeps):
        if (deadline_s is not None and time.perf_counter() - t0 > deadline_s) or (cancel is not None and cancel.is_set()):
            break
        moved_any = False
        for vs, (fl, own_loc, grp), (ti, si) in zip(classes, cls_faces, cls_nb):
            m = len(vs)
            xs = x[si]
            s = np.stack([np.bincount(ti, xs[:, 0], m), np.bincount(ti, xs[:, 1], m), np.bincount(ti, xs[:, 2], m)], 1)
            step = lam * (s / deg[vs, None] - x[vs])
            pending = np.ones(m, bool); base = x[vs].copy()
            for h in range(halvings):
                if not pending.any():
                    break
                x[vs[pending]] = base[pending] + step[pending] / (2 ** h)
                sj = csj_sub(x, grp, len(fl))
                bad = np.zeros(m, bool); bad[own_loc[sj < floor]] = True
                newly = pending & ~bad
                if newly.any():
                    moved_any = True
                pending &= bad
            x[vs[pending]] = base[pending]
        if not moved_any:
            break
    x[:n] = P[:n]
    if float(csj(x).min()) < float(s0.min()) - 1e-9:   # hard post-check: never lower the worst corner
        return dict(mesh)
    return dict(mesh, positions=x.tolist())
