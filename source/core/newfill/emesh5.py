# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""mesher core, pure Python + numpy (runs in Blender's bundled Python).

Pipeline pieces (reimplemented from the published methods):
  cdt_polygon       ear clipping + Lawson flips -> constrained Delaunay triangulation of the rim
  frontal_insert    Rebay (1993) frontal-Delaunay point insertion, the method behind Gmsh algorithm 6;
                    Bowyer-Watson cavities that never cross the rim (rim edges are constraints)
  recombine         Blossom-Quad (Remacle et al. 2012): min-cost perfect matching on the triangle
                    dual graph, cost 1 - eta(quad); odd rims via one dummy node (exact best triangle)
  smooth            fixed-rim smart Laplacian + angle-based (Zhou & Shimada 2000) with a validity guard
The rim (first n vertices, CCW) is never moved or split.
"""
import heapq
import math
from collections import defaultdict

import numpy as np



# ------------------------------------------------------------------ predicates
def orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def incircle(a, b, c, d):
    adx, ady = a[0] - d[0], a[1] - d[1]
    bdx, bdy = b[0] - d[0], b[1] - d[1]
    cdx, cdy = c[0] - d[0], c[1] - d[1]
    return ((adx * adx + ady * ady) * (bdx * cdy - cdx * bdy) - (bdx * bdx + bdy * bdy) * (adx * cdy - cdx * ady)
            + (cdx * cdx + cdy * cdy) * (adx * bdy - bdx * ady))


def circumcenter(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-300:
        return None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    return ((a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d, (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d)


def point_in_polygon(p, P):
    A = P if isinstance(P, np.ndarray) else np.asarray(P, float)
    x, y = p
    xi, yi = A[:, 0], A[:, 1]
    xj, yj = np.roll(xi, 1), np.roll(yi, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        cross = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
    return bool(np.count_nonzero(cross) % 2)


def seg_dist(p, a, b):
    ab = (b[0] - a[0], b[1] - a[1])
    L = ab[0] ** 2 + ab[1] ** 2
    t = 0. if L == 0 else max(0., min(1., ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / L))
    return math.hypot(p[0] - a[0] - t * ab[0], p[1] - a[1] - t * ab[1])


# ------------------------------------------------------------------ triangulation container
class Tri:
    def __init__(self, pts, n):
        self.X = [tuple(map(float, p)) for p in pts]
        self.n = n                                   # rim vertices 0..n-1, CCW, constrained edges (i,i+1)
        self.T = []                                  # [a,b,c] CCW or None (dead)
        self.E = {}                                  # directed edge (a,b) -> triangle index

    def constrained(self, a, b):
        n = self.n
        return a < n and b < n and ((a + 1) % n == b or (b + 1) % n == a)

    def add(self, a, b, c):
        t = len(self.T)
        self.T.append([a, b, c])
        self.E[(a, b)] = t
        self.E[(b, c)] = t
        self.E[(c, a)] = t
        return t

    def kill(self, t):
        a, b, c = self.T[t]
        for e in ((a, b), (b, c), (c, a)):
            if self.E.get(e) == t:
                del self.E[e]
        self.T[t] = None

    def alive(self):
        return [t for t in self.T if t is not None]

    def flip_ok(self, a, b):
        """Edge a->b shared by t1=(a,b,c) and t2=(b,a,d): legal-izing flip test."""
        if self.constrained(a, b):
            return False
        t1, t2 = self.E.get((a, b)), self.E.get((b, a))
        if t1 is None or t2 is None:
            return False
        c = [v for v in self.T[t1] if v != a and v != b][0]
        d = [v for v in self.T[t2] if v != a and v != b][0]
        X = self.X
        if incircle(X[a], X[b], X[c], X[d]) <= 1e-30 * 0:
            return False
        # convexity of the quad a d b c
        if orient(X[c], X[a], X[d]) <= 0 or orient(X[d], X[b], X[c]) <= 0:
            return False
        return True

    def flip(self, a, b):
        t1, t2 = self.E[(a, b)], self.E[(b, a)]
        c = [v for v in self.T[t1] if v != a and v != b][0]
        d = [v for v in self.T[t2] if v != a and v != b][0]
        self.kill(t1)
        self.kill(t2)
        self.add(c, a, d)
        self.add(d, b, c)
        return c, d

    def lawson(self, stack=None):
        if stack is None:
            stack = [e for e in list(self.E) if e[0] < e[1]]
        guard = 0
        while stack and guard < 200000:
            guard += 1
            a, b = stack.pop()
            if (a, b) not in self.E or (b, a) not in self.E:
                continue
            if self.flip_ok(a, b):
                c, d = self.flip(a, b)
                stack += [(a, d), (d, b), (b, c), (c, a)]


def cdt_polygon(P):
    """Ear clipping of the CCW polygon P, then Lawson flips (constrained Delaunay).
    v5: ear validity and quality are cached per vertex; after clipping an ear only its two neighbours are
    re-evaluated, so the whole clipping costs O(n^2) instead of O(n^3)."""
    n = len(P)
    tr = Tri(P, n)
    X = tr.X
    prev = [(i - 1) % n for i in range(n)]
    nxt = [(i + 1) % n for i in range(n)]
    alive = [True] * n
    reflex = set()

    def is_reflex(i):
        return orient(X[prev[i]], X[i], X[nxt[i]]) <= 0

    for i in range(n):
        if is_reflex(i):
            reflex.add(i)

    def ear(i):
        i0, i2 = prev[i], nxt[i]
        if i in reflex:
            return None
        for j in reflex:                          # only reflex vertices can lie inside a convex ear
            if j in (i0, i, i2):
                continue
            if orient(X[i0], X[i], X[j]) >= 0 and orient(X[i], X[i2], X[j]) >= 0 and orient(X[i2], X[i0], X[j]) >= 0:
                return None
        return tri_quality(X[i0], X[i], X[i2])

    q = {i: ear(i) for i in range(n)}
    m = n
    guard = 0
    while m > 3 and guard < 4 * n:
        guard += 1
        best = None
        for i, v in q.items():
            if v is not None and (best is None or v > q[best]):
                best = i
        if best is None:
            raise RuntimeError('ear clipping failed')
        i0, i2 = prev[best], nxt[best]
        tr.add(i0, best, i2)
        alive[best] = False
        del q[best]
        nxt[i0], prev[i2] = i2, i0
        reflex.discard(best)
        m -= 1
        for k in (i0, i2):
            if is_reflex(k):
                reflex.add(k)
            else:
                reflex.discard(k)
        for k in (i0, i2):
            q[k] = ear(k)
    rest = [i for i in range(n) if alive[i]]
    a = rest[0]
    tr.add(a, nxt[a], nxt[nxt[a]])
    tr.lawson()
    return tr


def tri_quality(a, b, c):
    """Normalised radius ratio-like quality: 4*sqrt(3)*area / sum(l^2), 1 = equilateral."""
    A = orient(a, b, c) / 2
    s = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (b[0] - c[0]) ** 2 + (b[1] - c[1]) ** 2 + (c[0] - a[0]) ** 2 + (c[1] - a[1]) ** 2
    return 4 * math.sqrt(3) * A / s if s > 0 else 0.


# ------------------------------------------------------------------ size field
class SizeField:
    """Shepard interpolation of rim spacing. v5: values come from a grid over the rim bbox (cell <= h_min / 2)
    (bilinear), built once with numpy, so a query is O(1) instead of O(n)."""
    def __init__(self, P, scale=1.0, grid=None):
        P = np.asarray(P, float)
        el = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
        self.P = P
        self.h = 0.5 * (el + np.roll(el, 1)) * scale
        self.hmean = float(el.mean()) * scale
        lo, hi = P.min(0), P.max(0)
        pad = 1e-9 + 0.01 * (hi - lo)
        self.lo, self.hi = lo - pad, hi + pad
        if grid is None:                       # cell <= half the smallest rim spacing, 32..256 nodes per side
            grid = int(min(256, max(32, np.ceil(float(np.max(self.hi - self.lo)) / (0.5 * max(float(el.min()), 1e-12))) + 1)))
        self.g = grid
        gx = np.linspace(self.lo[0], self.hi[0], grid)
        gy = np.linspace(self.lo[1], self.hi[1], grid)
        Q = np.stack(np.meshgrid(gx, gy, indexing='ij'), -1).reshape(-1, 2)
        H = np.empty(len(Q))
        for k in range(0, len(Q), 512):
            if CHECK is not None:
                CHECK()
            d2 = ((Q[k:k + 512, None, :] - P[None, :, :]) ** 2).sum(-1)
            w = 1. / np.maximum(d2, 1e-30) ** 1.5
            H[k:k + 512] = (w @ self.h) / w.sum(1)
        self.H = H.reshape(grid, grid)
        self.dx = (self.hi - self.lo) / (grid - 1)

    def __call__(self, x):
        u = (x[0] - self.lo[0]) / self.dx[0]
        v = (x[1] - self.lo[1]) / self.dx[1]
        i = min(max(int(u), 0), self.g - 2)
        j = min(max(int(v), 0), self.g - 2)
        fu = min(max(u - i, 0.), 1.)
        fv = min(max(v - j, 0.), 1.)
        H = self.H
        return float((1 - fu) * ((1 - fv) * H[i, j] + fv * H[i, j + 1]) + fu * ((1 - fv) * H[i + 1, j] + fv * H[i + 1, j + 1]))


CHECK = None   # v5: optional cooperative budget callback, called every 256 heap pops (raises to abort)


def set_check(fn):
    """Set the cooperative budget callback (None: none)."""
    global CHECK
    CHECK = fn


def frontal_insert(tr, size, accept=1.35, max_points=20000, min_sep=0.55):
    X = tr.X
    P = np.asarray([X[i] for i in range(tr.n)], float)
    K = 1. / math.sqrt(3.)                           # circumradius of the equilateral triangle of side h, / h

    rho_cache = {}

    def rho(t):
        r = rho_cache.get(t)
        if r is not None:
            return r
        r = rho_raw(t)
        rho_cache[t] = r
        return r

    def rho_raw(t):
        a, b, c = (X[v] for v in tr.T[t])
        cc = circumcenter(a, b, c)
        if cc is None:
            return 1e9
        R = math.hypot(a[0] - cc[0], a[1] - cc[1])
        g = ((a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3)
        return R / (K * size(g))

    acc = {}
    heap = []

    def is_front_edge(t, a, b):
        if tr.constrained(a, b):
            return True
        o = tr.E.get((b, a))
        return o is not None and acc.get(o, False)

    def push(t):
        if tr.T[t] is None or acc.get(t):
            return
        r = rho(t)
        if r < accept:
            acc[t] = True
            for v0, v1 in edges_of(tr.T[t]):
                o = tr.E.get((v1, v0))
                if o is not None and not acc.get(o):
                    heapq.heappush(heap, (-rho(o), o))
            return
        a, b, c = tr.T[t]
        if any(is_front_edge(t, u, v) for u, v in ((a, b), (b, c), (c, a))):
            heapq.heappush(heap, (-r, t))

    for t in range(len(tr.T)):
        if tr.T[t] is not None:
            push(t)
    inserted = 0
    RA = np.asarray(P, float)
    RB = np.roll(RA, -1, 0)
    RD = RB - RA
    RL = np.maximum((RD ** 2).sum(1), 1e-300)

    def rim_clear(x):
        q = np.asarray(x)
        t = np.clip(((q - RA) * RD).sum(1) / RL, 0., 1.)
        return float(np.sqrt((((RA + t[:, None] * RD) - q) ** 2).sum(1)).min())
    grid = _Grid(X, max(float(np.median(size.h)) * 0.5, 1e-12))
    for i in range(len(X)):
        grid.add(i)
    pops = 0
    while heap and inserted < max_points:
        negr, t = heapq.heappop(heap)
        pops += 1
        if CHECK is not None and not pops & 255:
            CHECK()
        if tr.T[t] is None or acc.get(t):
            continue
        a, b, c = tr.T[t]
        fronts = [(u, v) for u, v in ((a, b), (b, c), (c, a)) if is_front_edge(t, u, v)]
        if not fronts:
            continue
        u, v = min(fronts, key=lambda e: (X[e[0]][0] - X[e[1]][0]) ** 2 + (X[e[0]][1] - X[e[1]][1]) ** 2)
        A, B = X[u], X[v]
        M = ((A[0] + B[0]) / 2, (A[1] + B[1]) / 2)
        l = math.hypot(B[0] - A[0], B[1] - A[1])
        nx, ny = -(B[1] - A[1]) / l, (B[0] - A[0]) / l     # left normal = into triangle (CCW)
        p = l / 2
        cc = circumcenter(*(X[w] for w in tr.T[t]))
        q = (cc[0] - M[0]) * nx + (cc[1] - M[1]) * ny if cc else p
        hm = size(M)
        rm = max(K * hm, p)
        if q > 1e-12 * l:
            rm = min(rm, (p * p + q * q) / (2 * q))
        dist = rm + math.sqrt(max(rm * rm - p * p, 0.))
        x = (M[0] + nx * dist, M[1] + ny * dist)
        ok = point_in_polygon(x, P)    # needed: in non-convex rims the locate walk can stop at the rim for inside points
        if ok:
            hx = size(x)
            # separation from rim (vectorised point-segment distance) and from points (spatial hash)
            ok = rim_clear(x) >= min_sep * hx
        if ok:
            ok = grid.nearest_ok(x, min_sep * hx)
        newt = insert_point(tr, x, t) if ok else None
        if newt is not None:
            grid.add(len(X) - 1)
        if newt is None:
            acc[t] = True                             # give up on this triangle: accept it as is
            for v0, v1 in edges_of(tr.T[t]):
                o = tr.E.get((v1, v0))
                if o is not None and not acc.get(o):
                    heapq.heappush(heap, (-rho(o), o))
            continue
        inserted += 1
        for nt in newt:
            push(nt)
            for v0, v1 in edges_of(tr.T[nt]):
                o = tr.E.get((v1, v0))
                if o is not None and not acc.get(o) and tr.T[o] is not None:
                    heapq.heappush(heap, (-rho(o), o))
    return inserted


class _Grid:
    def __init__(self, X, cell):
        self.X = X
        self.c = cell
        self.g = defaultdict(list)

    def add(self, i):
        p = self.X[i]
        self.g[(int(math.floor(p[0] / self.c)), int(math.floor(p[1] / self.c)))].append(i)

    def nearest_ok(self, x, r):
        cx, cy = int(math.floor(x[0] / self.c)), int(math.floor(x[1] / self.c))
        k = int(math.ceil(r / self.c))
        r2 = r * r
        for gx in range(cx - k, cx + k + 1):
            for gy in range(cy - k, cy + k + 1):
                for i in self.g.get((gx, gy), ()):
                    p = self.X[i]
                    if (p[0] - x[0]) ** 2 + (p[1] - x[1]) ** 2 < r2:
                        return False
        return True


def edges_of(t):
    a, b, c = t
    return ((a, b), (b, c), (c, a))


def locate(tr, x, t0):
    X = tr.X
    t = t0
    for _ in range(10000):
        a, b, c = tr.T[t]
        moved = False
        for u, v in ((a, b), (b, c), (c, a)):
            if orient(X[u], X[v], x) < 0:
                o = tr.E.get((v, u))
                if o is None:
                    return None
                t = o
                moved = True
                break
        if not moved:
            return t
    return None


def insert_point(tr, x, t0):
    t = locate(tr, x, t0)
    if t is None:
        return None
    X = tr.X
    cav = {t}
    stack = [t]
    while stack:
        s = stack.pop()
        for u, v in edges_of(tr.T[s]):
            if tr.constrained(u, v):
                continue
            o = tr.E.get((v, u))
            if o is None or o in cav:
                continue
            a, b, c = (X[w] for w in tr.T[o])
            if incircle(a, b, c, x) > 0:
                cav.add(o)
                stack.append(o)
    bnd = []
    for s in cav:
        for u, v in edges_of(tr.T[s]):
            o = tr.E.get((v, u))
            if o is None or o not in cav:
                bnd.append((u, v))
    for u, v in bnd:
        if orient(X[u], X[v], x) <= 1e-14 * ((X[u][0] - X[v][0]) ** 2 + (X[u][1] - X[v][1]) ** 2):
            return None
    k = len(X)
    X.append(tuple(x))
    for s in cav:
        tr.kill(s)
    return [tr.add(u, v, k) for u, v in bnd]


# ------------------------------------------------------------------ Blossom-Quad recombination
def quad_eta(Q):
    """Remacle et al. 2012 quad quality eta = max(0, 1 - 2/pi * max_k |pi/2 - alpha_k|), with sign check."""
    worst = 0.
    for k in range(4):
        a, b, c = Q[k - 1], Q[k], Q[(k + 1) % 4]
        o = orient(a, b, c)
        if o <= 0:
            return -1.
        v1 = (a[0] - b[0], a[1] - b[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        ang = math.atan2(abs(v1[0] * v2[1] - v1[1] * v2[0]), v1[0] * v2[0] + v1[1] * v2[1])
        worst = max(worst, abs(math.pi / 2 - ang))
    return max(0., 1. - 2. / math.pi * worst)


def build_adj(nv, F):
    vf = [[] for _ in range(nv)]
    nbr = [set() for _ in range(nv)]
    for fi, f in enumerate(F):
        k = len(f)
        for i, v in enumerate(f):
            vf[v].append(fi)
            nbr[v].add(f[(i + 1) % k])
            nbr[v].add(f[i - 1])
    return vf, nbr


class FaceSet:
    """Vectorised per-face corner scaled Jacobian for mixed quads/triangles (triangle SJ normalised)."""
    def __init__(self, F):
        self.F = F
        self.q = [i for i, f in enumerate(F) if len(f) == 4]
        self.t = [i for i, f in enumerate(F) if len(f) == 3]
        self.Q = np.array([F[i] for i in self.q], int).reshape(-1, 4)
        self.Tt = np.array([F[i] for i in self.t], int).reshape(-1, 3)

    def sj(self, V):
        out = np.empty(len(self.F))
        for idx, A, norm in ((self.q, self.Q, 1.), (self.t, self.Tt, math.sin(math.pi / 3))):
            if not len(idx):
                continue
            x = V[A]
            a = np.roll(x, -1, 1) - x
            b = np.roll(x, 1, 1) - x
            cr = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
            la = np.linalg.norm(a, axis=2)
            lb = np.linalg.norm(b, axis=2)
            out[idx] = (cr / np.maximum(la * lb, 1e-300)).min(1) / norm
        return out


def smooth(V, F, n, iters=30, mode='laplace+angle'):
    """Fixed-rim smoothing. Jacobi sweeps; a vertex move is kept only if the min SJ of its faces does not drop
    below min(old, 0.98*old+...) (smart Laplacian, Freitag 1997 style), so the mesh never tangles."""
    V = np.array(V, float)
    nv = len(V)
    vf, nbr = build_adj(nv, F)
    inner = np.array([v for v in range(n, nv) if vf[v]], int)
    if not len(inner):
        return V
    fs = FaceSet(F)
    # neighbour lists (edge neighbours)
    nl = [sorted(nbr[v]) for v in range(nv)]
    maxd = max(len(nl[v]) for v in inner)
    NB = np.full((len(inner), maxd), -1, int)
    for r, v in enumerate(inner):
        NB[r, :len(nl[v])] = nl[v]
    mask = NB >= 0
    cnt = mask.sum(1)
    vf_inner = [vf[v] for v in inner]
    for it in range(iters):
        if CHECK is not None:
            CHECK()
        Xn = V[np.where(mask, NB, 0)]
        lap = (Xn * mask[..., None]).sum(1) / cnt[:, None]
        new = lap
        if 'angle' in mode and it >= iters // 3:
            # angle-based (Zhou-Shimada): rotate each neighbour edge to bisect the adjacent angles
            new = 0.5 * lap + 0.5 * angle_target(V, inner, nl, vf, F)
        old_sj = fs.sj(V)
        trial = V.copy()
        trial[inner] = new
        new_sj = fs.sj(trial)
        accept = np.ones(len(inner), bool)
        for r, fl in enumerate(vf_inner):
            o = min(old_sj[f] for f in fl)
            nn = min(new_sj[f] for f in fl)
            if nn < min(o, 0.0) + 1e-9 or nn < o - 0.02:
                accept[r] = False
        V2 = V.copy()
        V2[inner[accept]] = new[accept]
        # Jacobi conflicts: verify, revert offending vertices
        s2 = fs.sj(V2)
        bad = s2 < np.minimum(old_sj, 0) - 1e-12
        if bad.any():
            badv = {v for fi in np.nonzero(bad)[0] for v in F[fi]}
            for v in badv:
                if v >= n:
                    V2[v] = V[v]
        V = V2
    return V


def angle_target(V, inner, nl, vf, F):
    out = np.empty((len(inner), 2))
    for r, v in enumerate(inner):
        x = V[v]
        acc = np.zeros(2)
        k = 0
        # ordered ring from faces: for each neighbour j, bisect the angle between its two adjacent ring edges
        for j in nl[v]:
            # find the two faces containing edge (v,j); take the far neighbours of v in those faces
            others = []
            for fi in vf[v]:
                f = F[fi]
                if j in f:
                    i = f.index(v)
                    a, b = f[(i + 1) % len(f)], f[i - 1]
                    others.append(b if a == j else a)
            if len(others) != 2:
                continue
            pj = V[j]
            d1 = V[others[0]] - pj
            d2 = V[others[1]] - pj
            d1 /= max(np.linalg.norm(d1), 1e-300)
            d2 /= max(np.linalg.norm(d2), 1e-300)
            bis = d1 + d2
            nb = np.linalg.norm(bis)
            L = np.linalg.norm(x - pj)
            if nb < 1e-9:
                continue
            bis /= nb
            if bis @ (x - pj) < 0:
                bis = -bis
            acc += pj + bis * L
            k += 1
        out[r] = acc / k if k else x
    return out
