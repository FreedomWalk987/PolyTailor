# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Quality checks: validate() and the quick()/good() definitions, verbatim in
behaviour (vectorised per face size). validate(mesh, world, check=None) -> (ok, reason, normalised mesh);
quick(mesh, world, check=None) -> dict | None; good(q). check() runs between chunks and raises to abort."""
import numpy as np


def _plane(p, n):
    rim = p[:n]
    c = rim.mean(0)
    nrm = np.sum([np.cross(rim[i] - c, rim[(i + 1) % n] - c) for i in range(n)], 0)
    nrm /= max(np.linalg.norm(nrm), 1e-30)
    return nrm


class Stopped(Exception):
    """validate / quick stopped by their check (the run was cancelled or passed its time cap)."""


def _groups(F):
    """{face size: (face indices, (m, k) int array)} in face order within each size."""
    out = {}
    for k in sorted({len(f) for f in F}):
        idx = np.array([i for i, f in enumerate(F) if len(f) == k], int)
        out[k] = (idx, np.array([F[i] for i in idx], int).reshape(len(idx), k))
    return out


def _corner_cross(P, Fk, nrm):
    """(m, k) corner cross products along the normal, and the (m, k, 3) edge vectors a (next) and b (previous)."""
    x = P[Fk]
    a = np.roll(x, -1, 1) - x
    b = np.roll(x, 1, 1) - x
    m, k = Fk.shape
    cr = (np.cross(a, b).reshape(m * k, 3) @ nrm).reshape(m, k)
    return cr, a, b


def validate(mesh, world, check=None):
    """check: optional callable, called between chunks of work; it raises (e.g. Stopped) to abort the check."""
    tick = check if check is not None else (lambda: None)
    try:
        P = np.asarray(mesh['positions'], float)
        F = [list(map(int, f)) for f in mesh['faces']]
        n = int(mesh['boundary'])
    except Exception as e:  # malformed
        return False, f'malformed: {e!r}'[:120], None
    tick()
    if n != len(world):
        return False, f'boundary {n} != rim {len(world)}', None
    if P.ndim != 2 or P.shape[1] != 3 or len(P) < n:
        return False, 'positions shape', None
    if not np.all(np.isfinite(P)):
        return False, 'non-finite position', None
    if np.max(np.abs(P[:n] - world)) > 1e-9:
        return False, 'rim moved/reordered', None
    if not F:
        return False, 'no faces', None
    nv = len(P)
    sizes = [len(f) for f in F]
    tri = sum(s == 3 for s in sizes)
    if any(s not in (3, 4) for s in sizes):
        return False, 'polygon other than quad/triangle', None
    if n % 2 == 0 and tri:
        return False, 'triangle on an even rim', None
    if n % 2 == 1 and tri != 1:
        return False, 'odd rim needs exactly 1 triangle', None
    G = _groups(F)
    for k, (idx, Fk) in G.items():
        if Fk.min() < 0 or Fk.max() >= nv:
            return False, 'bad face indices', None
        for i in range(k):
            for j in range(i + 1, k):
                if np.any(Fk[:, i] == Fk[:, j]):
                    return False, 'bad face indices', None
    used = np.unique(np.concatenate([Fk.ravel() for _, Fk in G.values()]))
    if len(used) != nv:
        return False, 'unused vertex', None
    tick()
    nrm = _plane(P, n)
    sa = np.empty(len(F))
    for k, (idx, Fk) in G.items():
        x = P[Fk]
        sa[idx] = .5 * (np.cross(x, np.roll(x, -1, 1)).sum(1) @ nrm)
    if np.all(sa < 0):
        F = [f[::-1] for f in F]
        G = {k: (idx, Fk[:, ::-1].copy()) for k, (idx, Fk) in G.items()}
        sa = -sa
    if not np.all(sa > 0):
        return False, 'mixed winding or zero-area face', None
    tick()
    ef = {}
    for fi, f in enumerate(F):
        if not fi & 4095:
            tick()
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            ef.setdefault((min(a, b), max(a, b)), []).append((fi, a, b))
    rim_edges = {(min(i, (i + 1) % n), max(i, (i + 1) % n)) for i in range(n)}
    bnd = set()
    for e, lst in ef.items():
        if len(lst) > 2:
            return False, 'non-manifold edge', None
        if len(lst) == 2 and lst[0][1] == lst[1][1]:
            return False, 'inconsistent winding across an edge', None
        if len(lst) == 1:
            bnd.add(e)
    if bnd != rim_edges:
        return False, 'boundary edges != rim edges', None
    if nv - len(ef) + len(F) != 1:
        return False, 'Euler characteristic != 1', None
    tick()
    size = float(np.max(np.ptp(world, axis=0)))
    c = world.mean(0)
    off = np.abs((P[n:] - c) @ nrm) if nv > n else np.zeros(0)
    if off.size and off.max() > 1e-6 * size:
        return False, 'interior vertex off the rim plane', None
    if nv > n:
        u = np.cross(nrm, [1., 0, 0]) if abs(nrm[0]) < .9 else np.cross(nrm, [0, 1., 0])
        u /= np.linalg.norm(u); v = np.cross(nrm, u)
        R = np.stack([(world - c) @ u, (world - c) @ v], 1); Q = np.stack([(P[n:] - c) @ u, (P[n:] - c) @ v], 1)
        inside = np.zeros(len(Q), bool)
        for i in range(n):
            if not i & 63:
                tick()
            a, b = R[i], R[(i + 1) % n]
            cond = (a[1] > Q[:, 1]) != (b[1] > Q[:, 1])
            xint = a[0] + (Q[:, 1] - a[1]) * (b[0] - a[0]) / np.where(b[1] - a[1] == 0, 1e-300, b[1] - a[1])
            inside ^= cond & (Q[:, 0] < xint)
        if not inside.all():
            return False, 'interior vertex outside the rim', None
    tick()
    for k, (idx, Fk) in G.items():
        cr, _, _ = _corner_cross(P, Fk, nrm)
        if np.any(cr <= 0):
            return False, 'inverted/degenerate corner (SJ <= 0)', None
    return True, '', dict(positions=P, faces=F, boundary=n)


def quick(mesh, world, check=None):
    """check: as in validate (raises to abort)."""
    tick = check if check is not None else (lambda: None)
    ok, why, m = validate(mesh, world, check)
    if not ok:
        return None
    Pp = np.asarray(m['positions']); F = m['faces']; n = m['boundary']; nrm = _plane(Pp, n); sj = 9.; nb = {}
    er = []
    for k, (idx, Fk) in _groups(F).items():
        cr, a, b = _corner_cross(Pp, Fk, nrm)
        s = cr / np.maximum(np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2), 1e-30)
        sj = min(sj, float(s.min()))
        L = np.linalg.norm(a, axis=2)
        er.extend((L.max(1) / np.maximum(L.min(1), 1e-30)).tolist())
    tick()
    for fi, f in enumerate(F):
        if not fi & 4095:
            tick()
        for i in range(len(f)):
            nb.setdefault(f[i], set()).update((f[i - 1], f[(i + 1) % len(f)]))
    strip = len(Pp) == n and len(F) >= 3
    rim = Pp[:n]; h = float(np.mean(np.linalg.norm(np.roll(rim, -1, 0) - rim, axis=1)))
    Ar = abs(float(np.cross(rim, np.roll(rim, -1, 0)).sum(0) @ nrm) * .5)
    coarse = strip and (len(F) / max(Ar / h ** 2, 1e-30) < .5 or float(np.percentile(er, 90)) > 4)
    val6 = any(len(nb[v]) >= 6 for v in range(n, len(Pp)) if v in nb)
    mm = dict(positions=Pp.tolist(), faces=[list(map(int, f)) for f in F], boundary=int(n))
    return dict(mesh=mm, minSJ=sj, strip=strip, coarse=coarse, val6=val6, irr=sum(1 for v in range(n, len(Pp)) if v in nb and len(nb[v]) != 4))


def good(q):
    return q is not None and not q['coarse'] and not q['val6']


def as_mesh(r, n):
    return dict(positions=np.asarray(r['positions'], float).tolist(), faces=[list(map(int, f)) for f in r['faces']], boundary=n)
