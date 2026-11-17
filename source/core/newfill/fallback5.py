# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Guaranteed-valid fallback filler (NumPy and Python only).

fill(world (n,3)) -> dict(positions, faces)   (raises FillRefused only when every attempt fails validation)

Pipeline per attempt (density scale s from a schedule):
  1. constrained Delaunay of the rim (ear clipping + Lawson), Rebay frontal interior points at size s*h(x)
  2. Blossom-Quad: exact min-cost perfect matching of the triangles. A triangle pair is NOT offered when the
     merged quad would have a corner >= 179.5 deg at a fixed rim vertex (no optimiser can repair that).
     Odd rim: one dummy node -> exactly one triangle at the best place. Leftover pairs: rim-safe walk + merge.
  3. topology hygiene: doublet collapse (a valence-2 interior vertex can never have two convex corners)
  4. untangle + improve: derivative-free max-min corner SJ per interior vertex (pattern search), then
     guarded Laplacian/angle smoothing that never lowers a vertex's local min
  5. strict validation (the frozen validation rules); on failure the next density scale is tried.
Kept from the literature because it is simply right here: Blossom-Quad's exact matching (it always yields a
valid quad topology when one exists on the triangulation) and Poincare-Hopf parity (odd rim = one triangle).
"""
import math
import time
from collections import defaultdict

import numpy as np

from . import emesh5 as emesh
from .matching import max_weight_matching

SCALES = (1.0746, 0.85, 1.35, 0.65, 0.5, 0.38)
RIM_MAX_DEG = 179.5
LARGE_T = 800          # above this many triangles use greedy + walk-and-merge instead of the exact matcher
SJ_EPS = 1e-6          # strict validity margin (the scorer tests SJ > 0 in 3D floating point)


class FillRefused(Exception):
    pass


class _Ctx:
    """the E run of this process (one job at a time per worker process): deadline (t_end) and cancel token."""


_CTX = _Ctx()


def _check_budget():
    c = getattr(_CTX, 'cancel', None)
    if c is not None and c.is_set():
        raise FillRefused('fallback: cancelled')
    t = getattr(_CTX, 't_end', None)
    if t is not None and time.perf_counter() > t:
        _CTX.budget_hit = True   # this E run's result (or refusal) was cut by its budget
        raise FillRefused('fallback: budget exceeded')


emesh.set_check(_check_budget)


# ------------------------------------------------------------------ frame
def frame(world):
    W = np.asarray(world, float)
    c = W.mean(0)
    nrm = np.zeros(3)
    for i in range(len(W)):
        nrm += np.cross(W[i] - c, W[(i + 1) % len(W)] - c)
    nrm /= np.linalg.norm(nrm)
    ex = W[1] - W[0]
    ex -= nrm * (ex @ nrm)
    ex /= np.linalg.norm(ex)
    ey = np.cross(nrm, ex)
    return np.stack([(W - c) @ ex, (W - c) @ ey], 1), c, ex, ey


# ------------------------------------------------------------------ corner quality (plain loops, fast for small rims)
def corner(a, b, c):
    """SJ at b of the CCW polygon ... a, b, c ..."""
    ux, uy = c[0] - b[0], c[1] - b[1]
    vx, vy = a[0] - b[0], a[1] - b[1]
    d = math.hypot(ux, uy) * math.hypot(vx, vy)
    return (ux * vy - uy * vx) / d if d > 0 else -1.


def mesh_min(X, F):
    """Vectorised min corner SJ over all faces (same values as face_min, numpy)."""
    return float(emesh.FaceSet(F).sj(np.asarray(X, float)).min()) if F else 1.


def face_min(X, f):
    k = len(f)
    return min(corner(X[f[i - 1]], X[f[i]], X[f[(i + 1) % k]]) for i in range(k))


# ------------------------------------------------------------------ recombination with the rim-corner rule
def recombine_safe(tr, fixed_rule=True):
    X = tr.X
    n = tr.n
    tris = tr.alive()
    T = len(tris)
    tid = {}
    for i, t in enumerate(tris):
        for e in emesh.edges_of(t):
            tid[e] = i
    lim = math.sin(math.radians(180 - RIM_MAX_DEG))
    edges = []
    pair_quad = {}
    S = 10000
    for (u, v), i in tid.items():
        j = tid.get((v, u))
        if j is None or j <= i:
            continue
        a = [w for w in tris[i] if w not in (u, v)][0]
        b = [w for w in tris[j] if w not in (u, v)][0]
        Q = [a, u, b, v]
        bad = False
        for k in range(4):
            if fixed_rule and Q[k] < n and corner(X[Q[k - 1]], X[Q[k]], X[Q[(k + 1) % 4]]) <= lim:
                bad = True
                break
        if bad:
            continue
        eta = emesh.quad_eta([X[w] for w in Q])
        c = 1. - eta if eta >= 0 else 2.0
        edges.append((i, j, int(round((4. - c) * S))))
        pair_quad[(i, j)] = Q
    if T > LARGE_T:
        # large meshes: the exact O(V^3) blossom costs too much time. Greedy by cost (Remacle 2012 sec. 3,
        # the classic indirect method), leftovers are then paired by the rim-safe walk-and-merge.
        edges.sort(key=lambda e: -e[2])
        mate = {}
        adj = defaultdict(list)
        for i, j, w in edges:
            adj[i].append(j)
            adj[j].append(i)
            if i not in mate and j not in mate:
                mate[i] = j
                mate[j] = i
        augment_short(T, adj, mate, depth=12)
        F, left = [], 0
        for i in range(T):
            m = mate.get(i)
            if m is None:
                F.append(list(tris[i]))
                left += 1
            elif i < m:
                F.append(pair_quad[(i, m)])
        return F, left
    nv = T
    if T % 2:
        D = nv
        nv += 1
        for i, t in enumerate(tris):
            q = emesh.tri_quality(*(X[w] for w in t))
            edges.append((i, D, int(round((4. - 1. - (1. - q)) * S))))
    else:
        D1, D2 = nv, nv + 1
        nv += 2
        edges.append((D1, D2, 4 * S))
        for i in range(T):
            edges.append((i, D1, -46 * S))
            edges.append((i, D2, -46 * S))
    mate = max_weight_matching(nv, edges)
    F, used, left = [], set(), 0
    for i in range(T):
        if i in used:
            continue
        m = mate[i]
        if m == -1 or m >= T:
            F.append(list(tris[i]))
            used.add(i)
            left += 1
            continue
        F.append(pair_quad[(min(i, m), max(i, m))])
        used.update((i, m))
    return F, left


def augment_short(T, adj, mate, depth=12):
    """Grow a matching by short alternating BFS paths from each unmatched node (no blossom shrinking; on the
    near-planar triangle dual graph short paths pair up almost all leftovers)."""
    for u in range(T):
        if u in mate:
            continue
        prev = {u: None}
        frontier = [u]
        found = None
        for _ in range(depth):
            nxt = []
            for a in frontier:
                for x in adj[a]:
                    if x in prev or x == u:
                        continue
                    if x not in mate:
                        prev[x] = a
                        found = x
                        break
                    y = mate[x]
                    if y in prev:
                        continue
                    prev[x] = a
                    prev[y] = x
                    nxt.append(y)
                if found is not None:
                    break
            if found is not None or not nxt:
                break
            frontier = nxt
        if found is None:
            continue
        # flip along the path found <- ... <- u : pairs (found, prev[found]), (prev[prev], prev[prev[prev]]) ...
        x = found
        while x is not None:
            a = prev[x]
            b = prev[a] if a is not None else None
            mate[x] = a
            mate[a] = x
            x = b


# ------------------------------------------------------------------ hygiene
def remove_doublets(X, F, n):
    changed = True
    while changed:
        _check_budget()
        changed = False
        vf = defaultdict(list)
        for k, f in enumerate(F):
            for v in f:
                vf[v].append(k)
        for v, ks in vf.items():
            if v < n or len(ks) != 2:
                continue
            f1, f2 = F[ks[0]], F[ks[1]]
            if len(f1) + len(f2) not in (7, 8):
                continue
            # union boundary without v: walk f1 from v's successor to v's predecessor, then f2 likewise
            i = f1.index(v)
            p1 = [f1[(i + k) % len(f1)] for k in range(1, len(f1))]       # v+1 ... v-1 in f1
            j = f2.index(v)
            p2 = [f2[(j + k) % len(f2)] for k in range(1, len(f2))]
            if p1[-1] != p2[0] or p2[-1] != p1[0]:
                continue
            g = p1 + p2[1:-1]
            if len(set(g)) != len(g) or len(g) not in (3, 4):
                continue
            F = [f for k, f in enumerate(F) if k not in ks] + [g]
            changed = True
            break
    return F


# ------------------------------------------------------------------ untangler / optimiser
def optimise(X, F, n, sweeps=40, target=0.35):
    X = [tuple(p) for p in X]
    nv = len(X)
    vf = defaultdict(list)
    nb = defaultdict(set)
    for k, f in enumerate(F):
        for i, v in enumerate(f):
            vf[v].append(k)
            nb[v].add(f[(i + 1) % len(f)])
            nb[v].add(f[i - 1])
    inner = [v for v in range(n, nv) if vf[v]]

    def local(v):
        return min(face_min(X, F[k]) for k in vf[v])

    def elen(v):
        p = X[v]
        return sum(math.hypot(X[w][0] - p[0], X[w][1] - p[1]) for w in nb[v]) / max(len(nb[v]), 1)

    dirs = [(math.cos(t), math.sin(t)) for t in [i * math.pi / 4 for i in range(8)]]
    fs = emesh.FaceSet(F)
    inner_arr = np.array(inner, int)
    inc_v, inc_f = [], []
    for v in inner:
        for k in vf[v]:
            inc_v.append(v)
            inc_f.append(k)
    inc_v = np.array(inc_v, int)
    inc_f = np.array(inc_f, int)
    for sw in range(sweeps):
        q = fs.sj(np.array(X))
        vmin = np.full(nv, np.inf)
        np.minimum.at(vmin, inc_v, q[inc_f])
        _check_budget()
        tangled = q.min() <= SJ_EPS
        if tangled and sw < 6:
            # v5: untangling may need good vertices to move, but only near the folds: 3 rings around bad faces
            badf = np.nonzero(q <= SJ_EPS)[0]
            reg = {v for k in badf for v in F[k]}
            for _ in range(3):
                reg |= {w for v in list(reg) for w in nb[v]}
            cand = np.array(sorted(v for v in reg if v >= n and vf[v]), int)
        else:
            cand = inner_arr[vmin[inner_arr] < target]
        if not len(cand):
            break
        order = [int(v) for v in cand[np.argsort(vmin[cand])]]
        moved = False
        for v in order:
            q0 = local(v)
            if q0 >= target and not (tangled and sw < 6 and len(cand) < len(inner_arr) + 1):
                break                                  # order is ascending: the rest are fine
            p = X[v]
            # candidates: Laplacian position first
            cx = sum(X[w][0] for w in nb[v]) / len(nb[v])
            cy = sum(X[w][1] for w in nb[v]) / len(nb[v])
            best, bq = p, q0
            X[v] = (cx, cy)
            ql = local(v)
            if ql > bq + 1e-12:
                best, bq = (cx, cy), ql
            step = 0.25 * elen(v)
            cur = best
            for _ in range(5):
                improved = False
                for dx, dy in dirs:
                    c = (cur[0] + dx * step, cur[1] + dy * step)
                    X[v] = c
                    qc = local(v)
                    if qc > bq + 1e-12:
                        best, bq, improved = c, qc, True
                if improved:
                    cur = best
                else:
                    step *= 0.5
            X[v] = best
            if bq > q0 + 1e-9:
                moved = True
        if not moved:
            break
    return np.array(X)


# ------------------------------------------------------------------ validation (frozen rules, 2D)
def validate(X, F, n, P):
    tri = sum(len(f) == 3 for f in F)
    if any(len(f) not in (3, 4) for f in F):
        return 'polygon'
    if (n % 2 == 0 and tri) or (n % 2 == 1 and tri != 1):
        return 'triangle count %d' % tri
    ec = defaultdict(int)
    for f in F:
        if len(set(f)) != len(f):
            return 'degenerate face'
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            ec[(min(a, b), max(a, b))] += 1
    if any(c > 2 for c in ec.values()):
        return 'non-manifold'
    bnd = {e for e, c in ec.items() if c == 1}
    rim = {(min(i, (i + 1) % n), max(i, (i + 1) % n)) for i in range(n)}
    if bnd != rim:
        return 'boundary != rim'
    used = {v for f in F for v in f}
    if used != set(range(len(X))):
        return 'unused vertex'
    if len(X) - len(ec) + len(F) != 1:
        return 'euler'
    if mesh_min(X, F) <= SJ_EPS:
        return 'corner SJ <= %g' % SJ_EPS
    Pa = np.asarray(P, float)
    for v in range(n, len(X)):
        if not emesh.point_in_polygon(tuple(X[v]), Pa):
            return 'vertex outside'
    return None


def fixed_corner_bad(X, F, n):
    """A corner made only of fixed rim vertices can never be repaired by moving interior vertices."""
    for f in F:
        k = len(f)
        for i in range(k):
            a, b, c = f[i - 1], f[i], f[(i + 1) % k]
            if a < n and b < n and c < n and corner(X[a], X[b], X[c]) <= SJ_EPS:
                return True
    return False


def compact(X, F, n):
    used = sorted({v for f in F for v in f} | set(range(n)))
    remap = {v: i for i, v in enumerate(used)}
    return np.asarray(X)[used], [[remap[v] for v in f] for f in F]


def attempt(P, scale, accept=1.35):
    n = len(P)
    _check_budget()
    tr = emesh.cdt_polygon(P)
    size = emesh.SizeField(P, scale=scale)
    emesh.frontal_insert(tr, size, accept=accept)
    F, left = recombine_safe(tr)
    X = np.array(tr.X)
    if left > (n % 2):
        X, F, _ = merge_pairs_safe(X, F, n)
        if sum(len(f) == 3 for f in F) > n % 2:
            from . import topofix
            X, F, _ = topofix.merge_triangle_pairs(X, F, n)
            if sum(len(f) == 3 for f in F) > n % 2:
                raise RuntimeError('fixed rim corner: stuck triangle pairs')   # rim-bound: the ring is the remedy
    F = remove_doublets(X, F, n)
    X, F = compact(X, F, n)
    if fixed_corner_bad(X, F, n):
        raise RuntimeError('fixed rim corner')
    X = optimise(X, F, n)
    X[:n] = P
    q0 = mesh_min(X, F)
    if q0 > 0:
        _check_budget()
        Xs = emesh.smooth(X, F, n, iters=20, mode='laplace' if len(F) > 1500 else 'laplace+angle')
        if mesh_min(Xs, F) >= q0 - 0.02:
            X = Xs
    return X, F


# ------------------------------------------------------------------ escalation: boundary ring + core
def _segs_cross(p1, p2, p3, p4):
    d1 = emesh.orient(p3, p4, p1)
    d2 = emesh.orient(p3, p4, p2)
    d3 = emesh.orient(p1, p2, p3)
    d4 = emesh.orient(p1, p2, p4)
    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)


def simple_polygon(Q):
    n = len(Q)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if _segs_cross(Q[i], Q[(i + 1) % n], Q[j], Q[(j + 1) % n]):
                return False
    return True


def ring_points(P, frac=0.3):
    """Inner partner Q_i of every rim vertex on the interior-angle bisector. Offsets shrink until the inner
    polygon is simple, lies inside the rim and every ring quad (P_i, P_i+1, Q_i+1, Q_i) is strictly convex."""
    n = len(P)
    Pt = [tuple(p) for p in P]
    el = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    d0 = np.empty(n)
    B = np.empty((n, 2))
    for i in range(n):
        a = P[i - 1] - P[i]
        b = P[(i + 1) % n] - P[i]
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        th = math.atan2(b[0] * a[1] - b[1] * a[0], a @ b) % (2 * math.pi)     # interior angle
        ang = math.atan2(b[1], b[0]) + th / 2                                  # rotate b CCW by th/2
        B[i] = (math.cos(ang), math.sin(ang))
        # local thickness: distance to non-adjacent rim segments
        thick = min(emesh.seg_dist(Pt[i], Pt[j], Pt[(j + 1) % n]) for j in range(n) if j != i and (j + 1) % n != i) if n > 3 else el.min()
        d0[i] = min(el[i - 1], el[i], thick) * frac * max(math.sin(min(th, math.pi) / 2), 0.2)
    for it in range(30):
        _check_budget()
        Q = P + B * d0[:, None]
        Qt = [tuple(q) for q in Q]
        ok = all(emesh.point_in_polygon(q, Pt) for q in Qt) and simple_polygon(Qt)
        if ok:
            for i in range(n):
                j = (i + 1) % n
                if face_min({0: Pt[i], 1: Pt[j], 2: Qt[j], 3: Qt[i]}, [0, 1, 2, 3]) <= 1e-6:
                    ok = False
                    break
            if ok:
                # ring must not cross the rim
                for i in range(n):
                    for j in range(n):
                        if j in ((i - 1) % n, i, (i + 1) % n):
                            continue
                        if _segs_cross(Pt[i], Qt[i], Pt[j], Pt[(j + 1) % n]):
                            ok = False
                            break
                    if not ok:
                        break
        if ok:
            return Q
        d0 *= 0.6
    return None


def attempt_ring(P, scale, rings=1, frac=0.3):
    n = len(P)
    layers = [np.asarray(P, float)]
    for _ in range(rings):
        Q = ring_points(layers[-1], frac)
        if Q is None:
            raise RuntimeError('ring')
        layers.append(Q)
    X = [tuple(p) for L in layers for p in L]
    F = []
    for r in range(rings):
        o, oq = r * n, (r + 1) * n
        for i in range(n):
            j = (i + 1) % n
            F.append([o + i, o + j, oq + j, oq + i])
    core = layers[-1]
    _check_budget()
    tr = emesh.cdt_polygon(core)
    size = emesh.SizeField(core, scale=scale)
    emesh.frontal_insert(tr, size, accept=1.35)
    # strict pairing first (convex corners at the inner ring give better quads); relax only if pairs get stuck
    Xc = np.array(tr.X)
    Fc, left = recombine_safe(tr, fixed_rule=True)
    if left > (n % 2):
        Xc2, Fc2, _ = merge_pairs_safe(Xc, Fc, n)
        if sum(len(f) == 3 for f in Fc2) > n % 2:
            Fc, left = recombine_safe(tr, fixed_rule=False)   # the core rim is free (inner ring vertices)
            Xc2, Fc2, _ = merge_pairs_safe(Xc, Fc, n, fixed_n=0)
        Xc, Fc = Xc2, Fc2
    base = rings * n
    off = len(X)
    gid = lambda v: base + v if v < n else off + v - n
    X += [tuple(p) for p in Xc[n:]]
    F += [[gid(v) for v in f] for f in Fc]
    X = np.array(X)
    F = remove_doublets(X, F, n)
    X, F = compact(X, F, n)
    if fixed_corner_bad(X, F, n):
        raise RuntimeError('fixed rim corner')
    X = optimise(X, F, n, sweeps=60)
    X[:n] = P
    q0 = mesh_min(X, F)
    if q0 > SJ_EPS:
        _check_budget()
        Xs = emesh.smooth(X, F, n, iters=20, mode='laplace' if len(F) > 1500 else 'laplace+angle')
        if mesh_min(Xs, F) >= q0 - 0.02:
            X = Xs
    return X, F


SEC_PER_FACE = 1.1e-3      # estimated seconds per face of a plain attempt, used to size the density
LATE_STAR_S = 0.4          # grace for the late star template once the budget is gone with nothing valid
Q_RETRY = 0.25             # a valid result below this min SJ is kept, and the next density is tried while time allows


def est_faces(P, scale):
    """Expected quad count: area x mean(1 / h(x)^2) over a deterministic sample of interior points, with the same
    size field the mesher uses (a mean rim spacing underestimates rims that mix long and very short edges)."""
    P = np.asarray(P, float)
    key = (P.tobytes(), scale)
    if key in _EST:
        return _EST[key]
    x, y = P[:, 0], P[:, 1]
    A = 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))
    sz = emesh.SizeField(P, scale=scale, grid=32)
    lo, hi = P.min(0), P.max(0)
    g = np.linspace(0.025, 0.975, 20)
    Q = np.stack(np.meshgrid(lo[0] + g * (hi[0] - lo[0]), lo[1] + g * (hi[1] - lo[1]), indexing='ij'), -1).reshape(-1, 2)
    inv = [1. / sz(tuple(q)) ** 2 for q in Q if emesh.point_in_polygon(tuple(q), P)]
    if not inv:
        el = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
        r = A / max((float(el.mean()) * scale) ** 2, 1e-30)
    else:
        r = A * float(np.mean(inv))
    if len(_EST) > 64:
        _EST.clear()
    _EST[key] = r
    return r



def clear_caches():
    """Empty the estimate cache (add-on unregister)."""
    _EST.clear()


_EST = {}


def fill2d(P, scales=SCALES, budget_s=None, cancel=None):
    """v5: returns the best valid mesh found within the budget.
    * Tight budget (estimated plain cost > half the budget) and a strictly star-shaped rim: the kernel-star template
      runs FIRST (cheap, provably valid), so a valid answer exists early; the normal plans then try to beat it.
      Coarsening the interior against a dense rim was tried and rejected: it creates rim ears and fails.
    * A valid result below Q_RETRY triggers at most ONE more plan while time allows; the better one is returned.
    * On budget exhaustion the best valid mesh so far is returned (FillRefused only when there is none).
    cancel: the run's token (set / is_set), polled with the deadline by _check_budget."""
    errs = []
    n = len(P)
    t0 = time.perf_counter()
    _CTX.t_end, _CTX.cancel = (None if budget_s is None else t0 + budget_s), cancel
    _CTX.budget_hit = False
    k = 1.0
    star_first = budget_s is not None and est_faces(P, 1.0746) * SEC_PER_FACE > 0.5 * budget_s
    plans = ([('plain', s * k, 0, 0.3) for s in scales[:3]] + [('ring', s * k, 1, 0.3) for s in (1.0746, 1.4, 0.75)]
             + [('ring', s * k, 2, 0.3) for s in (1.0, 0.6)] + [('ring', 1.0 * k, 2, f) for f in (0.1, 0.45)]
             + [('ring', 1.0 * k, 3, f) for f in (0.15, 0.35)] + [('star', 0.0, 0, 0.0)])
    if n == 3:
        return np.asarray(P, float), [[0, 1, 2]], dict(plan='tri')
    best = None
    nvalid = 0
    if star_first:
        try:
            X, F = attempt_star(P)
            if validate(X, F, n, [tuple(p) for p in P]) is None:
                best = (mesh_min(X, F), X, F, dict(plan='star', rings=0, scale=0.0, frac=0.0, attempts=1, coarsen=k, star_first=True))
        except Exception as e:  # not star-shaped: continue with the normal plans
            errs.append('star-first: %s' % type(e).__name__)
        plans = [p for p in plans if p[0] != 'star']
    star_tried = star_first
    for kind, s, r, fr in plans:
        if (budget_s is not None and best is None and not star_tried and kind != 'star'
                and time.perf_counter() - t0 > 0.5 * budget_s):
            star_tried = True       # half the budget gone without a valid mesh: secure one with the star template
            try:
                Xs, Fs = attempt_star(P)
                if validate(Xs, Fs, n, [tuple(p) for p in P]) is None:
                    best = (mesh_min(Xs, Fs), Xs, Fs, dict(plan='star', rings=0, scale=0.0, frac=0.0, attempts=len(errs) + 1, coarsen=k))
            except FillRefused:
                break
            except Exception as e:
                errs.append('star-mid: %s' % type(e).__name__)
        try:
            _check_budget()
            X, F = attempt(P, s) if kind == 'plain' else (attempt_star(P) if kind == 'star' else attempt_ring(P, s, r, fr))
        except FillRefused:
            if best is None and not star_tried:
                star_tried = True   # budget exhausted with nothing valid: the star template is cheap, run it anyway
                t_end, _CTX.t_end = _CTX.t_end, time.perf_counter() + LATE_STAR_S   # a short grace, not unbounded
                try:
                    Xs, Fs = attempt_star(P)
                    if validate(Xs, Fs, n, [tuple(p) for p in P]) is None:
                        best = (mesh_min(Xs, Fs), Xs, Fs, dict(plan='star', rings=0, scale=0.0, frac=0.0, attempts=len(errs) + 1, coarsen=k, late=True))
                except Exception as e:
                    errs.append('star-late: %s' % type(e).__name__)
                finally:
                    _CTX.t_end = t_end
            if best is not None:
                best[3]['budget_hit'] = True
                break               # budget exhausted: return the best valid mesh so far
            raise
        except Exception as e:      # any failure -> next attempt
            errs.append('%s%d %.2f/%.2f: %s' % (kind, r, s, fr, type(e).__name__))
            continue
        bad = validate(X, F, n, [tuple(p) for p in P])
        if bad is None:
            q = mesh_min(X, F)
            if best is None or q > best[0]:
                best = (q, X, F, dict(plan=kind, rings=r, scale=s, frac=fr, attempts=len(errs) + 1, coarsen=k))
            nvalid += 1
            if q >= Q_RETRY or kind == 'star' or nvalid >= 2:
                break
            cost = est_faces(P, 1.0746) * SEC_PER_FACE
            if (budget_s is not None and (time.perf_counter() - t0) + cost > budget_s) or (budget_s is None and cost > 3.0):
                break               # no time for a retry (unbudgeted calls retry only when it is cheap)
            errs.append('%s%d %.2f/%.2f: valid, minSJ %.3f (retry)' % (kind, r, s, fr, q))
            continue
        errs.append('%s%d %.2f/%.2f: %s' % (kind, r, s, fr, bad))
    if best is not None:
        return best[1], best[2], best[3]
    raise FillRefused('fallback: no valid mesh (%s)' % '; '.join(errs))


# ------------------------------------------------------------------ v5: rim-safe pairing of leftover triangles
def _fixed_bad_face(X, f, n):
    k = len(f)
    for i in range(k):
        a, b, c = f[i - 1], f[i], f[(i + 1) % k]
        if a < n and b < n and c < n and corner(X[a], X[b], X[c]) <= SJ_EPS:
            return True
    return False


def merge_pairs_safe(X, F, n, max_tries=6, fixed_n=None):
    """Pair leftover triangles by walking one to the other (pentagon re-splits) and merging them, like
    topofix.merge_triangle_pairs, but never creating a corner made only of fixed rim vertices that is flat or
    inverted (no optimiser can repair it). A bad final merge is retried on a snapshot: walking from the other
    end, then with the next-nearest partner."""
    from collections import deque
    from . import topofix
    from .cleanup import Mesh, is_edge
    m = Mesh(X, F, n)
    merged = 0
    fn_ = n if fixed_n is None else fixed_n      # vertices < fn_ are fixed (the real rim); core rims move

    def bfs(src, avoid):
        prev = {src: None}
        dq = deque([src])
        hits = []
        while dq and len(hits) < 3:
            k = dq.popleft()
            f = m.F[k]
            for i in range(len(f)):
                for kk in m.edge_faces(f[i], f[(i + 1) % len(f)]):
                    if kk in prev:
                        continue
                    prev[kk] = k
                    if len(m.F[kk]) == 3:
                        if kk not in avoid:
                            hits.append(kk)
                    else:
                        dq.append(kk)
        return prev, hits

    def walk(src, hit, prev, log):
        path = [hit]
        while path[-1] != src:
            path.append(prev[path[-1]])
        path.reverse()
        cur = src
        for i in range(1, len(path) - 1):
            q, nxt = path[i], path[i + 1]
            best = None
            for op in topofix.candidate_ops(m, {cur}):
                if op[0] != 'tri' or op[1] != (cur, q):
                    continue
                tri, quad = op[2]
                fn = m.F[nxt]
                if not any(tri[j] in fn and tri[(j + 1) % 3] in fn and is_edge(fn, tri[j], tri[(j + 1) % 3]) for j in range(3)):
                    continue
                if _fixed_bad_face(m.V, quad, fn_) or _fixed_bad_face(m.V, tri, fn_):
                    continue
                s = min(topofix.sj(m.V, tri), topofix.sj(m.V, quad))
                if best is None or s > best[0]:
                    best = (s, op)
            if best is None:
                return False
            removed, added, Vsave, _ = topofix.apply(m, best[1])
            log.append((removed, added, Vsave))
            cur = [k for k in added if len(m.F[k]) == 3][0]
        f1, f2 = m.F[cur], m.F[hit]
        sh = [(f1[j], f1[(j + 1) % 3]) for j in range(3) if f1[j] in f2 and f1[(j + 1) % 3] in f2 and is_edge(f2, f1[j], f1[(j + 1) % 3])]
        if not sh:
            return False
        u, v = sh[0]
        w1 = [x for x in f1 if x not in (u, v)][0]
        w2 = [x for x in f2 if x not in (u, v)][0]
        quad = [u, w2, v, w1]
        if _fixed_bad_face(m.V, quad, fn_):
            return False
        m.remove(cur)
        m.remove(hit)
        m.add(quad)
        return True

    stuck = set()
    for _ in range(len(F)):
        _check_budget()
        tris_all = [k for k, f in m.F.items() if len(f) == 3]
        if len(tris_all) <= n % 2:
            break
        tris = [k for k in tris_all if k not in stuck]
        if not tris:
            break
        src = tris[0]
        done = False
        tries = 0
        prev, hits = bfs(src, set())
        for hit in hits:
            for a, b in ((src, hit), (hit, src)):
                if tries >= max_tries:
                    break
                tries += 1
                log = []                                   # undo log instead of a snapshot
                pv = prev if a == src else bfs(a, set())[0]
                if b in pv and walk(a, b, pv, log):
                    done = True
                    break
                for removed, added, Vsave in reversed(log):
                    topofix.revert(m, removed, added, Vsave)
            if done:
                break
        if done:
            merged += 1
        else:
            stuck.add(src)
    Fo = [m.F[k] for k in sorted(m.F)]
    return m.V, Fo, merged


# ------------------------------------------------------------------ v5: kernel-star template (last resort, provably valid)
def kernel_point(P):
    """Centroid of the polygon's kernel (intersection of the left half-planes of all edges), or None when the
    rim is not strictly star-shaped. Sutherland-Hodgman clipping of the bbox, O(n^2)."""
    P = np.asarray(P, float)
    n = len(P)
    lo, hi = P.min(0), P.max(0)
    K = [np.array(v, float) for v in ((lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1]))]
    scale = float(np.max(hi - lo))
    for i in range(n):
        if not i & 31:
            _check_budget()
        a, b = P[i], P[(i + 1) % n]
        d = b - a
        side = lambda q: d[0] * (q[1] - a[1]) - d[1] * (q[0] - a[0])
        out = []
        for j in range(len(K)):
            p, q = K[j], K[(j + 1) % len(K)]
            sp, sq = side(p), side(q)
            if sp >= 0:
                out.append(p)
            if (sp >= 0) != (sq >= 0):
                t = sp / (sp - sq)
                out.append(p + t * (q - p))
        K = out
        if len(K) < 3:
            return None
    A = np.array(K)
    x, y = A[:, 0], A[:, 1]
    cr = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = 0.5 * cr.sum()
    if area <= 1e-12 * scale * scale:
        return None
    c = np.array([((x + np.roll(x, -1)) * cr).sum(), ((y + np.roll(y, -1)) * cr).sum()]) / (6 * area)
    return c


def attempt_star(P):
    """Ring quads (P_i, P_i+1, Q_i+1, Q_i) with Q_i on a tiny circle around a kernel point K in the direction of P_i,
    plus a quadrangulation of the co-circular inner polygon. For a strictly star-shaped rim every corner is
    strictly convex for a small enough radius (corners at P are those of triangle P_i P_i+1 K; corners at Q are
    90 deg + half the angular gap < 180 deg). The radius is halved until the mesh validates, then optimised."""
    P = np.asarray(P, float)
    n = len(P)
    K = kernel_point(P)
    if K is None:
        raise RuntimeError('star: rim not star-shaped')
    U = P - K
    U /= np.linalg.norm(U, axis=1)[:, None]
    dmin = min(float(np.min(np.linalg.norm(P - K, axis=1))),
               min(emesh.seg_dist(tuple(K), tuple(P[i]), tuple(P[(i + 1) % n])) for i in range(n)))
    F = [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    inner = list(range(n, 2 * n))
    k = 1
    if n % 2:
        F.append([inner[0], inner[1], inner[2]])
        k = 2
    # quads 0,k,k+1,k+2 / 0,k+2,k+3,k+4 ... on co-circular points (all convex)
    while k + 2 < n:
        F.append([inner[0], inner[k], inner[k + 1], inner[k + 2]])
        k += 2
    r = 0.25 * dmin
    for _ in range(40):
        _check_budget()
        X = np.vstack([P, K + r * U])
        if validate(X, F, n, [tuple(p) for p in P]) is None:
            X = optimise(X, F, n, sweeps=60)
            X[:n] = P
            return X, F
        r *= 0.5
    raise RuntimeError('star: no valid radius')
