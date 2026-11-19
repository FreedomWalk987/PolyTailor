# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""rectilinear-labelling pattern fill (pure Python + numpy).

Idea:
  * Every rim vertex gets a label: H (convex corner, 1 face), F (flat, 2 faces),
    R (reflex corner, 3 faces) or S(d) (soft convex corner: 2 faces on the rim,
    one valence-3 pole d rows inside, i.e. an O-grid corner).
  * The labels turn the rim into a closed rectilinear (orthogonal) polygon in an
    integer parameter plane.  Closure (sum of turns = 360 deg, x and y return to 0)
    is the exact integer compatibility condition; it is solved by a cyclic DP over
    states (quarter-turns, x, y) with per-vertex costs from the rim angles.
  * The mesh is the unit-cell tiling of that polygon; each soft corner excises a
    d x d block at the param corner and glues its two inner edges (cone 270 deg).
  * Odd rims: one triangle (ear at a sharp corner, or a rim edge plus one interior
    vertex), the rest is the even problem.
  * Geometry: uniform-Laplacian (Tutte) map with the rim fixed, then a
    parallelogram/Laplacian smoother with an inversion guard.

Entry point: fill(world) -> dict(positions, faces).
"""
import math
import numpy as np
from . import dp_fast

DIRS = np.array([(1, 0), (0, 1), (-1, 0), (0, -1)], int)


class FillRefused(Exception):
    pass


class BudgetExceeded(FillRefused):
    pass


import time as _time


class _Ctx:
    """the A run of this process (one job at a time per worker process): its deadline and cancel token."""


_CTX = _Ctx()


def set_deadline(t, cancel=None):
    """This A run: absolute perf_counter deadline (None: none) and cancel token (set / is_set, or None)."""
    _CTX.deadline, _CTX.cancel = t, cancel


def deadline():
    return getattr(_CTX, 'deadline', None)


def cancelled():
    c = getattr(_CTX, 'cancel', None)
    return c is not None and c.is_set()


def time_left():
    if cancelled():
        return -1.0   # a cancelled run is out of time
    d = getattr(_CTX, 'deadline', None)
    return None if d is None else d - _time.perf_counter()


def check_budget():
    if cancelled():
        raise BudgetExceeded('cancelled')
    d = getattr(_CTX, 'deadline', None)
    if d is not None and _time.perf_counter() > d:
        raise BudgetExceeded('time budget exceeded')

def _drive(gen):
    """run a generator to its end; its value."""
    try:
        while True:
            next(gen)
    except StopIteration as done:
        return done.value


def _call_steps(fn, *args, **kwargs):
    """call fn; when it returns a generator (a *_steps function), step it here."""
    import inspect
    out = fn(*args, **kwargs)
    if inspect.isgenerator(out):
        out = yield from out
    return out


def _unique_pairs(pairs, nv):
    """np.unique(pairs, axis=0) for non-negative int pairs below nv: the same rows, order and dtype, in O(k log k)
    on one int64 key (axis=0 sorts a structured view, ~20x slower)."""
    pairs = np.asarray(pairs)
    if len(pairs) == 0:
        return np.unique(pairs, axis=0)
    base = max(int(nv), int(pairs.max()) + 1)
    key = np.unique(pairs[:, 0].astype(np.int64) * base + pairs[:, 1].astype(np.int64))
    return np.stack([key // base, key % base], 1).astype(pairs.dtype, copy=False)


def tick():
    """a budget check and a yield point: a slice point (a no-op for _drive)."""
    check_budget()
    yield



# ----------------------------------------------------------------- geometry utils
def plane_frame(world):
    P = np.asarray(world, float)
    c = P.mean(0)
    nrm = np.zeros(3)
    for i in range(len(P)):
        a, b = P[i] - c, P[(i + 1) % len(P)] - c
        nrm += np.cross(a, b)
    ln = np.linalg.norm(nrm)
    if ln < 1e-30:
        raise FillRefused('degenerate rim')
    nrm /= ln
    u = P[1] - P[0]
    u -= nrm * (u @ nrm)
    u /= np.linalg.norm(u)
    v = np.cross(nrm, u)
    xy = np.stack([(P - c) @ u, (P - c) @ v], 1)
    return xy, (c, u, v, nrm)   # CCW by construction (normal from Newell sum)


def interior_angles(xy):
    a = np.roll(xy, 1, 0) - xy
    b = np.roll(xy, -1, 0) - xy
    cr = b[:, 0] * a[:, 1] - b[:, 1] * a[:, 0]
    th = np.degrees(np.arctan2(cr, (a * b).sum(1))) % 360.0
    return th


# ----------------------------------------------------------------- labelling DP
def vertex_costs(th, prm):
    """Return dict label -> cost array (inf = forbidden) for H, F, R."""
    th = np.asarray(th, float)
    def sj(f):
        a = th / f
        return np.where(a < 180.0, np.sin(np.radians(np.minimum(a, 180.0))), 0.0)
    out = {}
    for lab, f in (('H', 1), ('F', 2), ('R', 3), ('Q', 4)):
        s = sj(f)
        mism = np.abs(f - th / 90.0) > 0.65
        c = prm['wq'] * (1.0 - s) + prm['wlow'] * np.maximum(0.0, prm['sj_lo'] - s) ** 2 * 100.0 + prm['wm'] * mism
        if lab == 'H':
            c = np.where(th < prm['h_max'], c, np.inf)
        if lab == 'R':
            c = np.where(th > prm['r_min'], c, np.inf)
        if lab == 'Q':
            c = np.where(th > prm.get('q_min', 285.0), c, np.inf)
        tau = prm.get('tau')
        if tau is not None:
            c = np.where(s >= tau, c, np.inf)
        out[lab] = c
    return out


def medial_depth(xy):
    """Per rim vertex: radius of the largest inscribed disk touching it (in units of the local edge length)."""
    n = len(xy)
    e_prev = xy - np.roll(xy, 1, 0); e_next = np.roll(xy, -1, 0) - xy
    def nrm(e):
        L = np.linalg.norm(e, axis=1, keepdims=True)
        return np.stack([-e[:, 1], e[:, 0]], 1) / np.maximum(L, 1e-300)
    nv = nrm(e_prev) + nrm(e_next)
    nv /= np.maximum(np.linalg.norm(nv, axis=1, keepdims=True), 1e-300)
    D = xy[None, :, :] - xy[:, None, :]            # p_j - p_i
    num = (D ** 2).sum(-1)
    den = 2.0 * (D * nv[:, None, :]).sum(-1)
    idx = np.arange(n)
    gap = np.abs(((idx[None, :] - idx[:, None]) + n // 2) % n - n // 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where((den > 1e-30) & (gap > 1), num / den, np.inf)
    rho = r.min(1)
    h = 0.5 * (np.linalg.norm(e_prev, axis=1) + np.linalg.norm(e_next, axis=1))
    rho = np.where(np.isfinite(rho), rho, h)
    return rho / np.maximum(h, 1e-300)


# ----------------------------------------------------------------- mesh from labels
class _UF:
    def __init__(self):
        self.p = {}
    def f(self, a):
        p = self.p
        p.setdefault(a, a)
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a
    def u(self, a, b):
        a, b = self.f(a), self.f(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def build_mesh_steps(labels, n):
    """Walk the labelled rim in the param plane; return (faces over keys, rim keys, #soft) or raise.
    Labels: H (+1 turn), F, R (-1), Q (-2, f=4 slit tip), S(d) (soft convex corner, d x d block excised and glued).
    Vertex identity: cells around a grid point are joined across non-rim edges, so slits keep two sides."""
    _y = 0   # yield points (a slice point)
    k = 0
    pos = np.array([0, 0])
    rim_pts = []
    poly = []
    softs = []
    soft_rim = []
    for i in range(n):
        lab, d = labels[i]
        if lab == 'S':
            a = DIRS[k % 4]; b = DIRS[(k + 1) % 4]
            rA = pos.copy()
            c = pos + d * a
            poly.append(tuple(c))
            softs.append((c.copy(), a.copy(), b.copy(), d))
            soft_rim.append((i, d))
            pos = c + d * b
            k += 1
            rim_pts.append((tuple(int(v) for v in rA), tuple(int(v) for v in pos)))
        else:
            rim_pts.append((tuple(int(v) for v in pos), tuple(int(v) for v in pos)))
            if lab == 'H':
                poly.append(tuple(pos)); k += 1
            elif lab == 'R':
                poly.append(tuple(pos)); k -= 1
            elif lab == 'Q':
                poly.append(tuple(pos)); k -= 2
        pos = pos + DIRS[k % 4]
    if k != 4 or pos[0] != 0 or pos[1] != 0:
        raise FillRefused('labelling does not close')
    if len(poly) < 4:
        raise FillRefused('too few param corners')
    P = np.array(poly)
    xmin, ymin = P.min(0); xmax, ymax = P.max(0)
    W, Hh = int(xmax - xmin), int(ymax - ymin)
    if W <= 0 or Hh <= 0:
        raise FillRefused('flat param polygon')
    diff = np.zeros((W + 1, Hh), int)
    m = len(P)
    for i in range(m):
        x0, y0 = P[i]; x1, y1 = P[(i + 1) % m]
        if x0 == x1 and y0 != y1:
            s = 1 if y1 > y0 else -1
            ya, yb = sorted((y0, y1))
            diff[0, ya - ymin:yb - ymin] += s
            diff[x0 - xmin, ya - ymin:yb - ymin] -= s
    wind = np.cumsum(diff, 0)[:W]
    if wind.min() < 0 or wind.max() > 1:
        raise FillRefused('param polygon overlaps itself')
    inside = wind == 1
    for si, (c, a, b, d) in enumerate(softs):
        for s_ in range(d):
            for t in range(d):
                p0 = c - (s_ + 0.5) * a + (t + 0.5) * b
                cx, cy = int(math.floor(p0[0])) - xmin, int(math.floor(p0[1])) - ymin
                if not (0 <= cx < W and 0 <= cy < Hh) or not inside[cx, cy]:
                    ex = FillRefused('soft block outside polygon')
                    ex.soft_vertex = soft_rim[si]
                    raise ex
                inside[cx, cy] = False
    rim_edges = set()
    for i in range(n):
        p0 = rim_pts[i][1]; p1 = rim_pts[(i + 1) % n][0]
        rim_edges.add((min(p0, p1), max(p0, p1)))

    def ins(x, y):
        cx, cy = x - xmin, y - ymin
        return 0 <= cx < W and 0 <= cy < Hh and bool(inside[cx, cy])
    uf = _UF()
    cells = [(int(cx + xmin), int(cy + ymin)) for cx, cy in np.argwhere(inside)]
    pts = set()
    for (x, y) in cells:
        _y += 1
        if not _y & 63:
            yield
        for p in ((x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)):
            pts.add(p)
            uf.f((p, (x, y)))
    for (px, py) in pts:
        _y += 1
        if not _y & 63:
            yield
        around = {'ll': (px - 1, py - 1), 'lr': (px, py - 1), 'ul': (px - 1, py), 'ur': (px, py)}
        links = (('ul', 'ur', (px, py + 1)), ('lr', 'ur', (px + 1, py)), ('ll', 'lr', (px, py - 1)), ('ll', 'ul', (px - 1, py)))
        for c1, c2, q in links:
            a1, a2 = around[c1], around[c2]
            if ins(*a1) and ins(*a2):
                e = (min((px, py), q), max((px, py), q))
                if e not in rim_edges:
                    uf.u(((px, py), a1), ((px, py), a2))

    def key_at(p, cell):
        return uf.f((tuple(int(v) for v in p), tuple(cell)))

    def any_key(p):
        p = tuple(int(v) for v in p)
        for cell in ((p[0] - 1, p[1] - 1), (p[0], p[1] - 1), (p[0] - 1, p[1]), (p[0], p[1])):
            if ins(*cell):
                return key_at(p, cell)
        raise FillRefused('seam point without cell')
    for c, a, b, d in softs:
        q = c - d * a + d * b
        for t in range(1, d + 1):
            uf.u(any_key(q - t * b), any_key(q + t * a))
    faces = []
    for (x, y) in cells:
        _y += 1
        if not _y & 63:
            yield
        f = [uf.f(key_at(p, (x, y))) for p in ((x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1))]
        if len(set(f)) != 4:
            raise FillRefused('degenerate glued cell')
        faces.append(f)

    def left_cell(p0, p1):
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        mx, my = 0.5 * (p0[0] + p1[0]) - 0.5 * dy, 0.5 * (p0[1] + p1[1]) + 0.5 * dx
        return (int(math.floor(mx)), int(math.floor(my)))
    rim = []
    for i in range(n):
        ra, rb = rim_pts[i]
        cin = left_cell(rim_pts[i - 1][1], ra)
        cout = left_cell(rb, rim_pts[(i + 1) % n][0])
        if not ins(*cin) or not ins(*cout):
            raise FillRefused('rim edge without interior cell')
        r1 = uf.f(key_at(ra, cin)); r2 = uf.f(key_at(rb, cout))
        if r1 != r2:
            raise FillRefused('rim vertex split')
        rim.append(r1)
    if len(set(rim)) != n:
        raise FillRefused('rim vertices merged')
    return faces, rim, len(softs)


def index_mesh_steps(faces_k, rim_k):
    ids = {k: i for i, k in enumerate(rim_k)}
    nxt = len(ids)
    faces = []
    for j, f in enumerate(faces_k):
        if j and not j & 63:
            yield
        row = []
        for k in f:
            if k not in ids:
                ids[k] = nxt; nxt += 1
            row.append(ids[k])
        faces.append(row)
    return np.array(faces, int), nxt


def check_topology_steps(faces, n, nv):
    """Disk check: boundary edges == rim cycle, Euler = 1, edge use <= 2."""
    from collections import Counter
    E = Counter()
    for j, f in enumerate(faces):
        if j and not j & 63:
            yield
        L = len(f)
        for i in range(L):
            a, b = f[i], f[(i + 1) % L]
            E[(a, b)] += 1
    und = Counter()
    for j, ((a, b), c) in enumerate(E.items()):
        if j and not j & 1023:
            yield
        if c > 1:
            return False
        und[(min(a, b), max(a, b))] += 1
    if max(und.values()) > 2:
        return False
    bnd = {e for e, c in und.items() if c == 1}
    rim = {(min(i, (i + 1) % n), max(i, (i + 1) % n)) for i in range(n)}
    if bnd != rim:
        return False
    yield
    used = set(v for f in faces for v in f)
    if len(used) != nv:
        return False
    return nv - len(und) + len(faces) == 1


# ----------------------------------------------------------------- smoothing
def tutte_steps(xy_rim, faces_list, n, nv, iters=None):
    """Uniform Laplacian with fixed rim, conjugate gradient."""
    edges = set()
    for j, f in enumerate(faces_list):
        if j and not j & 63:
            yield   # a slice point
        L = len(f)
        for i in range(L):
            a, b = f[i], f[(i + 1) % L]
            edges.add((min(a, b), max(a, b)))
    e = np.array(sorted(edges), int)
    yield
    deg = np.bincount(e.ravel(), minlength=nv).astype(float)
    m = nv - n
    pos = np.zeros((nv, 2))
    pos[:n] = xy_rim
    if m == 0:
        return pos
    ii = e[:, 0]; jj = e[:, 1]
    b = np.zeros((nv, 2))
    mask = (ii >= n) & (jj < n)
    np.add.at(b, ii[mask], pos[jj[mask]])
    mask2 = (jj >= n) & (ii < n)
    np.add.at(b, jj[mask2], pos[ii[mask2]])
    b = b[n:]
    both = (ii >= n) & (jj >= n)
    I2, J2 = ii[both] - n, jj[both] - n
    d = deg[n:]
    yield
    def A(x):
        y = d[:, None] * x
        y -= np.stack([np.bincount(I2, x[J2, k], m) + np.bincount(J2, x[I2, k], m) for k in (0, 1)], 1)
        return y
    x = np.tile(xy_rim.mean(0), (m, 1))
    r = b - A(x); p = r.copy(); rs = (r * r).sum(0)
    bb = max(1e-300, (b * b).sum())
    for it in range(iters or 2 * m + 50):
        if it % 32 == 0:
            yield from tick()
        elif it % 8 == 0:
            yield
        Ap = A(p)
        al = rs / np.maximum((p * Ap).sum(0), 1e-300)
        x += al * p
        r -= al * Ap
        rn = (r * r).sum(0)
        if rn.max() < 1e-22 * bb:
            break
        p = r + (rn / np.maximum(rs, 1e-300)) * p
        rs = rn
    pos[n:] = x
    return pos


def corner_sj(pos, F):
    """Scaled Jacobian per face corner (F: (m,L) int array); returns (m,L)."""
    X = pos[F]
    a = np.roll(X, -1, 1) - X
    b = np.roll(X, 1, 1) - X
    cr = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    la = np.sqrt((a * a).sum(-1)); lb = np.sqrt((b * b).sum(-1))
    return cr / np.maximum(la * lb, 1e-300)


class Smoother:
    """Winslow (inverse-Laplace) Jacobi iteration on regular vertices, iso-parametric blend elsewhere."""

    def __init__(self, faces_q, n, nv, tri=None):
        _drive(self._setup_steps(faces_q, n, nv, tri))

    @classmethod
    def create_steps(cls, faces_q, n, nv, tri=None):
        """a Smoother built in steps (the sliced run)."""
        sm = cls.__new__(cls)
        yield from sm._setup_steps(faces_q, n, nv, tri)
        return sm

    def _setup_steps(self, faces_q, n, nv, tri=None):
        F = np.asarray(faces_q, int).reshape(-1, 4)
        self.F, self.n, self.nv = F, n, nv
        self.tri = np.asarray([tri], int) if tri is not None else None
        v = F.reshape(-1); nx = np.roll(F, -1, 1).reshape(-1); op = np.roll(F, -2, 1).reshape(-1); pv = np.roll(F, 1, 1).reshape(-1)
        self.cv, self.cnx, self.cop, self.cpv = v, nx, op, pv
        self.cnt = np.bincount(v, minlength=nv).astype(float)
        e = [np.stack([v, nx], 1), np.stack([nx, v], 1)]
        if tri is not None:
            t = np.asarray(tri)
            te = np.array([(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])
            e += [te, te[:, ::-1]]
        e = _unique_pairs(np.concatenate(e), nv)
        yield
        self.e = e
        self.deg = np.bincount(e[:, 0], minlength=nv).astype(float)
        tri_v = set(tri) if tri is not None else set()
        corners = {}
        for i in range(len(v)):
            if i and not i & 63:
                yield
            corners.setdefault(int(v[i]), []).append((int(nx[i]), int(op[i]), int(pv[i])))
        reg, st = [], []
        for vv in range(n, nv):
            if not (vv - n + 1) & 63:
                yield
            cs = corners.get(vv, [])
            if len(cs) != 4 or vv in tri_v or self.deg[vv] != 4:
                continue
            by_nx = {c[0]: c for c in cs}
            c0 = cs[0]; seq = [c0]
            ok = True
            for _ in range(3):
                nxt = by_nx.get(seq[-1][2])
                if nxt is None:
                    ok = False; break
                seq.append(nxt)
            if not ok or by_nx.get(seq[-1][2]) is not c0:
                continue
            # E NE N NW W SW S SE
            st.append([seq[0][0], seq[0][1], seq[1][0], seq[1][1], seq[2][0], seq[2][1], seq[3][0], seq[3][1]])
            reg.append(vv)
        self.reg = np.array(reg, int)
        self.st = np.array(st, int).reshape(-1, 8)
        isreg = np.zeros(nv, bool); isreg[self.reg] = True
        self.irr = np.array([vv for vv in range(n, nv) if not isreg[vv]], int)
        yield

    def minsj(self, pos):
        m = corner_sj(pos, self.F).min() if len(self.F) else 1.0
        if self.tri is not None:
            m = min(m, corner_sj(pos, self.tri).min())
        return float(m)

    def step(self, pos, w_par=0.5):
        new = pos.copy()
        if len(self.reg):
            S = pos[self.st]                       # (m,8,2)
            E, NE, N, NW, W, SW, S_, SE = [S[:, k] for k in range(8)]
            xi = 0.5 * (E - W); eta = 0.5 * (N - S_)
            al = (eta * eta).sum(1); ga = (xi * xi).sum(1); be = (xi * eta).sum(1)
            cross = 0.25 * (NE - NW - SE + SW)
            den = 2.0 * (al + ga)
            new[self.reg] = (al[:, None] * (E + W) + ga[:, None] * (N + S_) - 2.0 * be[:, None] * cross) / np.maximum(den, 1e-300)[:, None]
        if len(self.irr):
            lap = np.zeros_like(pos)
            lap[:, 0] = np.bincount(self.e[:, 0], pos[self.e[:, 1], 0], self.nv)
            lap[:, 1] = np.bincount(self.e[:, 0], pos[self.e[:, 1], 1], self.nv)
            lap /= np.maximum(self.deg, 1)[:, None]
            par = np.zeros_like(pos)
            q = pos[self.cnx] + pos[self.cpv] - pos[self.cop]
            par[:, 0] = np.bincount(self.cv, q[:, 0], self.nv); par[:, 1] = np.bincount(self.cv, q[:, 1], self.nv)
            par /= np.maximum(self.cnt, 1)[:, None]
            tgt = np.where(self.cnt[:, None] > 0, (1 - w_par) * lap + w_par * par, lap)
            new[self.irr] = tgt[self.irr]
        return new

    def run_steps(self, pos, iters=200, relax=1.0, check=20):
        """the Python smoothing loop."""
        return (yield from self._py_run_steps(pos, iters, relax, check))

    def run(self, *args, **kwargs):
        """run_steps run to its end (the synchronous form)."""
        return _drive(self.run_steps(*args, **kwargs))

    def _py_run_steps(self, pos, iters=200, relax=1.0, check=20):
        best, bq = pos, self.minsj(pos)
        cur = pos
        for it in range(iters):
            if it % 10 == 0:
                yield from tick()
            else:
                yield   # one smoothing iteration per step
            nxt = self.step(cur)
            cur = cur + relax * (nxt - cur)
            if (it + 1) % check == 0 or it == iters - 1:
                q = self.minsj(cur)
                if q >= bq - 1e-9 or (bq <= 0 and q > bq):
                    best, bq = cur, q
                elif q < 0 < bq:
                    cur = best
                    relax *= 0.5
        return best, bq

    def _py_run(self, *args, **kwargs):
        """_py_run_steps run to its end (the synchronous form)."""
        return _drive(self._py_run_steps(*args, **kwargs))


def polish_steps(pos, F_all, n, rounds=4, thr=0.5):
    """Greedy per-vertex search maximising the local minimum scaled Jacobian (bad region only)."""
    nv = len(pos)
    faces_of = [[] for _ in range(nv)]
    for fi, f in enumerate(F_all):
        for v in f:
            faces_of[v].append(fi)
    Fl = [list(f) for f in F_all]

    def face_min(P, f):
        L = len(f); m = 9.0
        for i in range(L):
            x0, y0 = P[f[i]]; x1, y1 = P[f[(i + 1) % L]]; x2, y2 = P[f[i - 1]]
            ax, ay, bx, by = x1 - x0, y1 - y0, x2 - x0, y2 - y0
            d = math.sqrt((ax * ax + ay * ay) * (bx * bx + by * by))
            s = (ax * by - ay * bx) / d if d > 0 else -1.0
            if s < m:
                m = s
        return m
    P = [tuple(map(float, p)) for p in pos]
    fq = [face_min(P, f) for f in Fl]
    yield
    dirs = [(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
    for _ in range(rounds):
        yield from tick()
        bad = sorted({v for fi, q in enumerate(fq) if q < thr for v in Fl[fi] if v >= n})
        if not bad:
            break
        moved = False
        for bi, v in enumerate(bad):
            if bi and not bi & 15:
                yield   # a slice point
            fs = faces_of[v]
            base = min(fq[fi] for fi in fs)
            x, y = P[v]
            ln = min(math.dist(P[v], P[u]) for fi in fs for u in Fl[fi] if u != v)
            bestq, bestp = base, None
            for scale in (0.25, 0.1, 0.04):
                for dx, dy in dirs:
                    P[v] = (x + scale * ln * dx, y + scale * ln * dy)
                    q = min(face_min(P, Fl[fi]) for fi in fs)
                    if q > bestq + 1e-6:
                        bestq, bestp = q, P[v]
                if bestp is not None:
                    break
            if bestp is not None:
                P[v] = bestp
                for fi in fs:
                    fq[fi] = face_min(P, Fl[fi])
                moved = True
            else:
                P[v] = (x, y)
        if not moved:
            break
    return np.array(P)


def optimize_sj_steps(pos, F_all, n, hinge, iters=60, lam=0.02, mu=0.0, r0=0.26):
    """Raise the corner scaled Jacobians sj_c towards `hinge` (rim fixed) by backtracking gradient steps on
    E = 100 * sum_c w_c * max(0, hinge - sj_c)^2      (w_c = 0.3 at rim corners, 1 inside)
        + mu * sum_c max(0, |0.5 * ln(la_c^2 / lb_c^2)| - r0)^2      (the two corner edges' length ratio)
        + lam * Laplacian edge energy, scaled by the mean hinge term.
    A step is kept only if E drops and the minimum sj stays above max(0, the starting minimum); the step of each
    vertex is scaled by its local edge length^2. realize passes hinge = prm['hinge'] (0.72), raised by the bound."""
    nv = len(pos)
    if nv == n:
        return pos
    v0, v1, v2 = [], [], []
    for f in F_all:
        L = len(f)
        for i in range(L):
            v0.append(f[i]); v1.append(f[(i + 1) % L]); v2.append(f[i - 1])
    v0 = np.array(v0); v1 = np.array(v1); v2 = np.array(v2)
    tgt = np.full(len(v0), hinge, float)
    rimw = np.where(v0 < n, 0.3, 1.0)
    e = _unique_pairs(np.sort(np.stack([v0, v1], 1), 1), nv)
    free = np.arange(nv) >= n

    def energy_grad(P, need_grad=True):
        a = P[v1] - P[v0]; b = P[v2] - P[v0]
        la2 = (a * a).sum(1); lb2 = (b * b).sum(1)
        la = np.sqrt(la2); lb = np.sqrt(lb2)
        c = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        s = c / np.maximum(la * lb, 1e-300)
        hv = np.maximum(0.0, tgt - s) * np.sqrt(rimw)
        w = hv * hv * 100.0
        E = w.sum()
        if mu:
            r = 0.5 * np.log(la2 / np.maximum(lb2, 1e-300))
            ex = np.sign(r) * np.maximum(0.0, np.abs(r) - r0)
            E = E + mu * (ex * ex).sum()
        el = P[e[:, 0]] - P[e[:, 1]]
        scale = np.mean(la2)
        E_l = lam * (el * el).sum() / scale * w.sum() / len(w)
        if not need_grad:
            return E + E_l, s
        dEds = -200.0 * hv * np.sqrt(rimw)
        ab = la * lb
        dsa = np.stack([b[:, 1], -b[:, 0]], 1) / ab[:, None] - (s / la2)[:, None] * a
        dsb = np.stack([-a[:, 1], a[:, 0]], 1) / ab[:, None] - (s / lb2)[:, None] * b
        ga = dEds[:, None] * dsa; gb = dEds[:, None] * dsb
        if mu:
            ga = ga + (mu * 2 * ex / la2)[:, None] * a
            gb = gb - (mu * 2 * ex / lb2)[:, None] * b
        G = np.zeros_like(P)
        for k in (0, 1):
            G[:, k] = np.bincount(v1, ga[:, k], nv) + np.bincount(v2, gb[:, k], nv) - np.bincount(v0, ga[:, k] + gb[:, k], nv)
            ge = 2 * lam * el[:, k] / scale * w.sum() / len(w)
            G[:, k] += np.bincount(e[:, 0], ge, nv) - np.bincount(e[:, 1], ge, nv)
        return E + E_l, s, G
    # local length^2 per vertex
    el = pos[e[:, 0]] - pos[e[:, 1]]; l2 = (el * el).sum(1)
    L2 = (np.bincount(e[:, 0], l2, nv) + np.bincount(e[:, 1], l2, nv)) / np.maximum(np.bincount(e.ravel(), minlength=nv), 1)
    P = pos.copy()
    _, s0 = energy_grad(P, False)
    ref = s0.min()
    E, s, G = energy_grad(P, True)
    floor = ref - 1e-9
    eta = 0.05
    for it in range(iters):
        yield from tick()
        D = -G * L2[:, None]
        D[~free] = 0.0
        gn = np.sqrt((D * D).sum(1)).max()
        if gn <= 0:
            break
        step = eta / (gn / np.sqrt(L2[free].mean()))     # max move = eta * typical edge length
        ok = False
        for _ in range(6):
            Q = P + step * D
            E2, s2 = energy_grad(Q, False)
            if s2.min() > max(0.0, floor) and E2 < E:
                ok = True
                break
            step *= 0.5
        if not ok:
            break
        P = Q
        E, s, G = energy_grad(P, True)
        eta = min(eta * 1.3, 0.3)
    return P


def quick_metrics(pos, F_all, n):
    """minSJ (raw, triangles included as in the rubric), interior irregulars, tolerant rim mismatches, strip flag."""
    nb = [set() for _ in range(len(pos))]
    fc = np.zeros(len(pos), int)
    for f in F_all:
        L = len(f)
        for i in range(L):
            nb[f[i]].add(f[(i + 1) % L]); nb[f[(i + 1) % L]].add(f[i]); fc[f[i]] += 1
    sj = 9.0
    quads = np.array([f for f in F_all if len(f) == 4], int).reshape(-1, 4)
    if len(quads):
        sj = min(sj, corner_sj(pos, quads).min())
    tris = np.array([f for f in F_all if len(f) == 3], int).reshape(-1, 3)
    if len(tris):
        sj = min(sj, corner_sj(pos, tris).min())
    irr = sum(1 for v in range(n, len(pos)) if len(nb[v]) != 4)
    th = interior_angles(pos[:n])
    btol = int(np.sum(np.abs(fc[:n] - th / 90.0) > 0.65))
    strip = (len(pos) == n and len(F_all) >= 3)
    return dict(minSJ=float(sj), irr=irr, btol=btol, strip=strip)


def sj_eff(sj, prm):
    cap = prm['sj_cap']
    return min(sj, cap) + prm['sj_slope'] * max(0.0, sj - cap)


def select_key(m, prm):
    s = sj_eff(m['minSJ'], prm) - prm['k_irr'] * m['irr'] - prm['k_btol'] * m['btol']
    if m['minSJ'] < 0.3:
        s -= prm['k_terr']
    if m['strip']:
        s -= prm['k_strip']
    return s


# ----------------------------------------------------------------- driver
DEFAULT = dict(wq=1.0, wlow=1.0, sj_lo=0.55, wm=1.5, h_max=172.0, r_min=190.0, wT=0.6,
               soft=True, s_max=186.0, s_min=100.0, wpole=1.2, wd=0.3, dmax=40, dfrac=0.55,
               k_irr=0.33, k_btol=0.19, k_terr=1.0, k_strip=0.3, sj_cap=0.6, sj_slope=0.4, iters=200, polish=True, opt_iters=60, lam=0.0, hinge=0.72, hinge_gap=0.08, hinge_max=0.9, hinge_hi=0.85, hinge_gap_hi=0.04, mu=0.2)
VARIANTS = [dict(), dict(soft=False), dict(wpole=0.5), dict(wpole=3.0), dict(wpole=3.0, wm=0.5, sj_lo=0.35),
            dict(wpole=1.2, dfrac=0.35), dict(wpole=1.2, dfrac=0.8), dict(wT=1.5), dict(wT=0.25, wpole=2.0),
            dict(dfrac=1.0, dwin=99, wd=0.02, wpole=1.5), dict(r_min=168.0, wm=0.6, wpole=2.0),
            dict(r_min=168.0, wm=0.6, wpole=1.0, wT=0.2),
            dict(r_min=168.0, wm=1.0, wpole=0.8, wT=0.2, s_min=120.0, sj_lo=0.8, wlow=3.0)]


def labellings_steps(xy, prm, variants):
    """Closing labellings of the rim, per variant: attempt 0 of every variant is solved batched / bound-pruned
    (dp_fast.solve_steps, exact), the nogood retries by the windowed single DP (dp_fast.label_dp_steps, pruned by the best
    labelling found so far). Results equal the plain cyclic labelling DP's unless the fill's deadline expires inside
    the DP."""
    from . import apattern as A   # this module (dp_fast reads its costs and budget check)
    th = interior_angles(xy)
    rho = medial_depth(xy)
    start = int(np.argmin(np.abs(th - 180.0)))
    seen, out = set(), []
    n = len(xy)
    ps = []
    for var in variants:
        p = dict(prm, **var); p['_rho'] = rho; p['_xy'] = xy; p['_forbid'] = set(); p['_forbid_lab'] = set()
        ps.append(p)
    first = (yield from dp_fast.solve_steps(th, ps, start, A))
    pool = [r[0] for r in first if r is not None and r[0] is not None]
    prune = n >= dp_fast.PRUNE_MIN_N
    for vi, var in enumerate(variants):
        p = ps[vi]
        forbid = set()
        flab = set()
        for attempt in range(prm.get('nogood', 4) if prm.get('_relaxed') is None else 10):
            p['_forbid'] = forbid
            p['_forbid_lab'] = flab
            try:
                yield from tick()   # the build / index / topology work below never runs past the deadline or a cancel
                if attempt == 0:
                    if first[vi] is None:
                        raise BudgetExceeded('time budget exceeded')
                    labels, cost = first[vi]
                else:
                    P = dp_fast._prep(th, p, start, A)
                    ub = min([dp_fast.path_cost(l, P) for l in pool] + [np.inf]) if (prune and P['W'] is not None) else None
                    labels, cost = (yield from dp_fast.label_dp_steps(th, p, start, A, ub=ub, P=P))
                    if labels is not None:
                        pool.append(labels)
            except BudgetExceeded:
                if out:
                    return out, th
                raise
            if labels is None:
                break
            try:
                fk_, rk_, _ = yield from build_mesh_cached_steps(labels, n)
                yield from tick()
                fq_, nv_ = yield from index_mesh_steps(fk_, rk_)
                yield from tick()
                if not (yield from check_topology_steps(fq_.tolist(), n, nv_)):
                    raise FillRefused('topology')
            except BudgetExceeded:
                if out:
                    return out, th
                raise
            except FillRefused as ex:
                sv = getattr(ex, 'soft_vertex', None)
                if sv is None:
                    # overlap / seam failure: retry with a soft corner (or corner label) of this labelling forbidden
                    corner = [(i, l[0]) for i, l in enumerate(labels) if l[0] == 'R' and (i, 'R') not in flab] + [(i, l[0]) for i, l in enumerate(labels) if l[0] == 'H' and (i, 'H') not in flab]
                    if not any(l[0] == 'S' for l in labels):    # no soft corner to blame -> forbid one corner label
                        if not corner:
                            break
                        flab = flab | {corner[attempt % len(corner)]}
                        continue
                    if prm.get('_relaxed') is not None:      # forbid one (vertex, d) at a time, cycling
                        sl = [(i, l[1]) for i, l in enumerate(labels) if l[0] == 'S' and (i, l[1]) not in forbid]
                        if not sl:
                            break
                        forbid = forbid | {sl[attempt % len(sl)]}
                        continue
                    sl = [(i, 0) for i, l in enumerate(labels) if l[0] == 'S' and (i, 0) not in forbid]
                    if not sl:
                        break
                    forbid = forbid | ({sl[attempt % len(sl)]} if attempt < 2 else set(sl))
                    continue
                forbid = forbid | {sv}
                continue
            key = tuple(labels)
            if key not in seen:
                seen.add(key)
                out.append((labels, vi))
            break
    if not out and prm.get('_relaxed') is None:
        rp = dict(prm, h_max=179.0, r_min=150.0, kband=2.6, wm=0.3, _relaxed=True)
        return (yield from labellings_steps(xy, rp, [dict(), dict(soft=False), dict(dmax=2), dict(s_min=30.0, s_max=200.0, dwin=3, wpole=0.8), dict(s_min=30.0, s_max=200.0, dwin=3, wpole=0.8, kband=3.5)]))
    return out, th


def label_bound(labels, th):
    """Upper bound of the candidate's min SJ from the rim angles, interior irregulars and tolerant mismatches."""
    f = np.array([{'H': 1, 'F': 2, 'R': 3, 'S': 2, 'Q': 4}[l[0]] for l in labels], float)
    a = th / f
    sj = np.where(a < 180.0, np.sin(np.radians(np.minimum(a, 179.999))), 0.0)
    irr = sum(1 for l in labels if l[0] == 'S')
    btol = int(np.sum(np.abs(f - th / 90.0) > 0.65))
    return float(sj.min()), irr, btol



def clear_caches():
    """Empty the build cache (add-on unregister)."""
    _BUILD_CACHE.clear()


_BUILD_CACHE = {}


def build_mesh_cached_steps(labels, n):
    key = (tuple(labels), n)
    r = _BUILD_CACHE.get(key)
    if r is None:
        if len(_BUILD_CACHE) > 256:
            _BUILD_CACHE.clear()
        try:
            r = yield from build_mesh_steps(labels, n)
        except FillRefused as ex:
            r = ex
        _BUILD_CACHE[key] = r
    if isinstance(r, FillRefused):
        raise r.with_traceback(None)   # a fresh traceback per hit (no frames piling up on the cached object)
    faces, rim, nsoft = r
    return [list(f) for f in faces], list(rim), nsoft   # copies: the cache stays safe whatever a caller does


def mesh_from_labels_steps(labels, n):
    fk, rk, nsoft = yield from build_mesh_cached_steps(labels, n)
    faces, nv = yield from index_mesh_steps(fk, rk)
    return faces, nv


def realize_steps(xy, faces, nv, n, prm, tri=None, bound=None):
    F_all = [list(map(int, f)) for f in faces] + ([list(tri)] if tri is not None else [])
    pos = (yield from tutte_steps(xy, F_all, n, nv))
    sm = yield from Smoother.create_steps(faces, n, nv, tri)
    pos, q = (yield from sm.run_steps(pos, iters=prm['iters']))
    if prm.get('opt_iters', 0) and q > 0:
        hg = prm.get('hinge')
        if hg is not None and bound is not None and prm.get('hinge_gap') is not None:
            gap = prm['hinge_gap'] if bound < prm.get('hinge_hi', 0.85) else prm.get('hinge_gap_hi', prm['hinge_gap'])
            hg = max(hg, min(bound - gap, prm.get('hinge_max', 0.9)))
        pos = (yield from optimize_sj_steps(pos, F_all, n, iters=prm['opt_iters'], lam=prm.get('lam', 0.02), mu=prm.get('mu', 0.0), hinge=hg))
    q = corner_sj(pos, np.asarray(faces, int)).min() if len(faces) else 1.0
    if 0 < q < 0.45:
        pos = (yield from polish_steps(pos, F_all, n))
    return pos, F_all


def plans_even_steps(xy, prm, variants, tri_map=None, extra_bound=1.0, tag=None):
    n = len(xy)
    out = []
    labs, th = (yield from labellings_steps(xy, prm, variants))
    for labels, vi in labs:
        b, irr, btol = label_bound(labels, th)
        b = min(b, extra_bound)
        ub = sj_eff(b, prm) - prm['k_irr'] * irr - prm['k_btol'] * btol - (prm['k_terr'] if b < 0.3 else 0.0)
        out.append(dict(ub=ub, b=b, labels=labels, vi=vi, xy=xy, n=n, tri_map=tri_map, tag=tag))
    return out


def run_plan_steps(pl, prm):
    xy, n = pl['xy'], pl['n']
    faces, nv = yield from mesh_from_labels_steps(pl['labels'], n)
    yield from tick()
    if pl['tri_map'] is None:
        Fq, nvf, tri, xyf, nf = faces, nv, None, xy, n
    else:
        Fq, nvf, tri, xyf, nf = pl['tri_map'](faces, nv)
    F_all = [list(map(int, f)) for f in Fq] + ([list(tri)] if tri is not None else [])
    if not (yield from check_topology_steps(F_all, nf, nvf)):
        raise FillRefused('topology')
    yield from tick()
    pos, F_all = (yield from realize_steps(xyf, Fq, nvf, nf, prm, tri, pl.get('b')))
    m = quick_metrics(pos, F_all, nf)
    if m['minSJ'] <= 0 and prm.get('untangle') is not None:
        pos = (yield from _call_steps(prm['untangle'], pos, F_all, nf))
        m = quick_metrics(pos, F_all, nf)
    if m['minSJ'] <= 0:
        raise FillRefused('inverted after smoothing')
    return select_key(m, prm), pos, F_all, dict(labels=pl['labels'], m=m, vi=pl['vi'], tri=pl['tag'])


def fill_xy_steps(xy, prm=None):
    prm = dict(DEFAULT, **(prm or {}))
    n = len(xy)
    plans = []
    if n % 2 == 0:
        plans = (yield from plans_even_steps(xy, prm, prm.get('variants') or VARIANTS))
    else:
        th = interior_angles(xy)
        sharp = [int(v) for v in np.argsort(th)[:2]]
        tries = [('ear', v) for v in sharp if th[v] < 125.0]
        flat = np.abs(th - 180.0) + np.abs(np.roll(th, -1) - 180.0)
        tries += [('edge', int(i)) for i in np.argsort(flat)[:2]]
        for v in sharp[:2]:
            for e in ((v - 1) % n, v):
                if ('edge', e) not in tries:
                    tries.append(('edge', e))
        for kind, v in tries:
            plans += (yield from _odd_plans_steps(xy, prm, kind, v))
    plans.sort(key=lambda p: -p['ub'])
    cands, err, done = [], None, 0
    best = -np.inf
    for pl in plans:
        if pl['ub'] <= best + 1e-9 or done >= prm.get('max_real', 6):
            break
        try:
            c = (yield from run_plan_steps(pl, prm))
        except BudgetExceeded:
            if cands:
                break   # deadline or cancel: the best candidate so far
            raise
        except FillRefused as ex:
            err = err or ex
            continue
        done += 1
        cands.append(c)
        best = max(best, c[0])
    if not cands:
        raise err or FillRefused('no candidate')
    bestc = max(cands, key=lambda c: c[0])
    info = dict(bestc[3]); info['ncand'] = len(cands); info['nplans'] = len(plans)
    return bestc[1], bestc[2], info


def _odd_plans_steps(xy, prm, kind, v):
    n = len(xy)
    if kind == 'ear':
        keep = [i for i in range(n) if i != v]
        sub = xy[keep]
        tri = [(v - 1) % n, v, (v + 1) % n]
        T = xy[tri]
        tsj = corner_sj(T, np.array([[0, 1, 2]])).min()

        def tmap(faces, nv):
            m = n - 1
            mp = np.array(keep + list(range(n, n + nv - m)))
            return mp[faces], nv + 1, tri, xy, n
        return (yield from plans_even_steps(sub, prm, prm.get('odd_variants') or ODD_VARIANTS, tmap, extra_bound=tsj, tag=(kind, v)))
    a, b = xy[v], xy[(v + 1) % n]
    t = b - a
    L = np.linalg.norm(t)
    nrm = np.array([-t[1], t[0]]) / L
    x = 0.5 * (a + b) + nrm * L * 0.75
    sub = np.concatenate([xy[:v + 1], x[None], xy[v + 1:]])
    tri = [v, (v + 1) % n, n]

    def tmap(faces, nv):
        m = n + 1
        mp = np.array(list(range(v + 1)) + [n] + list(range(v + 1, n)) + list(range(n + 1, n + 1 + nv - m)))
        return mp[faces], nv, tri, xy, n
    return (yield from plans_even_steps(sub, prm, prm.get('odd_variants') or ODD_VARIANTS, tmap, extra_bound=0.866, tag=(kind, v)))


ODD_VARIANTS = [dict(), dict(soft=False), dict(r_min=168.0, wm=0.6, wpole=2.0)]


# ---------------------------------------------------------------- round 2: tau-ladder frontier
# For each rim-SJ floor tau the DP returns the cheapest closing labelling whose every rim vertex has
# sin(theta/f) >= tau; costs are poles first, tolerant mismatches second, shape-follow third.
TAU_BASE = dict(wq=0.2, wlow=0.0, wm=0.6, wpole=1.0, r_min=190.0, wT=0.6, s_min=60.0)
