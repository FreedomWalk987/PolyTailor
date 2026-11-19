# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""D-opt: vertex-placement optimizer for a fixed planar quad(-dominant) topology.

Pure numpy (Blender's bundled Python, no
scipy).  Vertices 0..n_fixed-1 (the drawn rim) never move; topology never changes.

Per face corner c (edges e1 -> next, e2 -> prev, D = e1 x e2 > 0, ccw faces):
  shape   kappa_c = |A W^-1|_F^2 / (2 det(A W^-1))       Knupp condition number = TMOP
                                                        shape metric mu_2; W = unit square
                                                        (quads) or equilateral (triangle)
  size    s_c     = (tau + 1/tau)/2 - 1, tau = D / T_c   TMOP size term; T_c from a harmonic
                                                        size field off the rim spacing
  min SJ  g_c     = 1/SJ_c = |e1||e2|/D                  P_p = (mean g^p)^(1/p) -> max g,
                                                        i.e. maximise min SJ (smooth max-min)
  floor   SJ_c > t                                      log-type barrier (stage C)
  guards  |ln A_f/A_g| < cap_e,  |ln l_i/l_j| < cap_f   "do no harm" barriers: adjacent-area
                                                        ratio and face edge ratio may not get
                                                        worse than the input's p95 / p90 level
  flow    cos^2(inner edge, rim chord) at 2-face rim vertices (first-row orthogonality)
All terms except flow are barriers (-> inf at their limit); the Armijo line search rejects
non-finite energies, so every accepted iterate is fold-free and inside the guards.

Stages (L-BFGS each):
  0  untangle (Garanzha-style regularised barrier) if input folds
  A  TMOP:   mean kappa + w_size mean s + w_orth flow                       (smooth, fair)
  B  max-min SJ: P_p(g clipped at the immovable-corner cap) + w_reg * A, p = 8,16,32
     (skipped when an immovable rim corner already sets min SJ)
  C  lexicographic: A's objective under SJ_c > (B's min SJ - slack), started from A if
     A is feasible (else from B)
  pick: among results within pick_tol of the best min SJ (never below the input's), the
     latest stage (C > B > A > input).

API:  optimize(positions, faces, n_fixed, **opts) -> (positions_out, info)
positions (V,2) or (V,3); 3D is projected to the rim's best-fit plane and lifted back with
harmonic heights.  faces: index lists (quads; triangles allowed).
"""
import math
import time

import numpy as np


def _drive(gen):
    """run a generator to its end; its value."""
    try:
        while True:
            next(gen)
    except StopIteration as done:
        return done.value


__version__ = "5.1"

SIN60 = math.sin(math.pi / 3)
_W_TRI_INV = np.linalg.inv(np.array([[1.0, 0.5], [0.0, SIN60]]))

DEFAULTS = dict(
    w_size=1.0,          # TMOP size weight (A, C)
    w_orth=2.0,          # first-row orthogonality weight (A, C)
    w_aspect=10.0,       # in-face edge-length ratio weight (A, C)
    iters_a=60,          # L-BFGS iteration caps per stage
    p_schedule=(8, 32),
    iters_b=40,          # per p value
    iters_c=60,
    ftol=1e-6,           # stop a stage when the relative energy drop stays below this
    w_reg=0.05,          # stage B regulariser weight (shape+size)
    untangle=True,
    stage_a=True,
    stage_b=True,
    stage_c=True,
    floor_slack=0.002,   # stage C floor = B's min SJ - slack
    floor_band=0.1,
    w_floor=1.0,
    pick_tol=0.01,
    cap_tol=1e-3,
    guards=True,         # do-no-harm barriers on adjacent-area ratio and face edge ratio
    adj_min_cap=1.0,     # global caps are at least these ratios (1.0: the input's p95 / p90)
    edge_min_cap=1.0,
    guard_slack=0.02,    # log units
    guard_band=0.1,
    w_guard=1.0,
    size_target="harmonic",
    time_budget=None,    # seconds from the call; None = iteration caps only
    deadline=None,       # absolute time.perf_counter() value (shared budget of a whole fill); min with time_budget
    cancel=None,         # callable() -> True to stop (UI cancel / caller's step budget); polled every iteration
    on_budget="best",    # stop and return the best valid stage so far
    raw_adj_cap=2.0,     # absolute guard caps used when the input had to be untangled
    raw_edge_cap=3.0,
)

# Named option sets. FULL = DEFAULTS (the scored config). FAST: the latency preset.
PRESETS = dict(
    FULL={},
    FAST=dict(iters_a=40, p_schedule=(16,), iters_b=40, iters_c=40),
)


# ----------------------------------------------------------------------------- topology tables
class Corners:
    """Vectorised corner / pair tables for a fixed topology (faces ccw)."""

    def __init__(self, faces, nv, n_fixed):
        _drive(self._setup_steps(faces, nv, n_fixed))

    @classmethod
    def create_steps(cls, faces, nv, n_fixed):
        """a Corners built in steps (the sliced A)."""
        c = cls.__new__(cls)
        yield from c._setup_steps(faces, nv, n_fixed)
        return c

    def _setup_steps(self, faces, nv, n_fixed):
        o, a, b, tri, fid = [], [], [], [], []
        for fi, f in enumerate(faces):
            if fi and not fi & 63:
                yield   # a slice point
            m = len(f)
            for k in range(m):
                o.append(f[k]); a.append(f[(k + 1) % m]); b.append(f[(k - 1) % m])
                tri.append(m == 3); fid.append(fi)
        self.o = np.array(o, np.int64); self.a = np.array(a, np.int64); self.b = np.array(b, np.int64)
        self.tri = np.array(tri, bool); self.fid = np.array(fid, np.int64)
        self.nv = nv; self.n_fixed = n_fixed; self.nf = len(faces)
        # corners whose three vertices are all fixed are constant
        self.live = ~((self.o < n_fixed) & (self.a < n_fixed) & (self.b < n_fixed))
        self.any_tri = bool(self.tri.any())
        w = np.broadcast_to(np.eye(2), (len(o), 2, 2)).copy()
        w[self.tri] = _W_TRI_INV
        self.w00, self.w01, self.w10, self.w11 = w[:, 0, 0], w[:, 0, 1], w[:, 1, 0], w[:, 1, 1]
        self.wdet = self.w00 * self.w11 - self.w01 * self.w10
        self.scatter = np.concatenate((self.a, self.b, self.o))
        self.free = np.arange(n_fixed, nv)
        self.kc = np.where(self.tri, 1.0 / 6, 0.25)          # face area = sum kc * D
        first, pairs = {}, []
        yield
        for fi, f in enumerate(faces):
            if fi and not fi & 63:
                yield
            for k in range(len(f)):
                e = (min(f[k], f[(k + 1) % len(f)]), max(f[k], f[(k + 1) % len(f)]))
                if e in first:
                    pairs.append((first[e], fi))
                else:
                    first[e] = fi
        self.adj = np.array(pairs, np.int64).reshape(-1, 2)
        self.edge_faces = first
        ep, c0 = [], 0
        yield
        for f in faces:
            m = len(f)
            ep += [(c0 + i, c0 + j) for i in range(m) for j in range(i + 1, m)]
            c0 += m
        yield
        self.epairs = np.array(ep, np.int64).reshape(-1, 2)
        self.epair_face = self.fid[self.epairs[:, 0]] if len(ep) else np.zeros(0, np.int64)
        self.orth_v = self.orth_w = self.orth_t = None
        self._scatter2()

    def _scatter2(self):
        self.scatter2 = np.concatenate((self.scatter, self.scatter + self.nv))
        if self.orth_w is not None:
            self.scatter2o = np.concatenate((self.scatter, self.orth_w, self.scatter + self.nv, self.orth_w + self.nv))

    def flip(self):
        """Reverse every face's orientation (swap next/prev at each corner)."""
        self.a, self.b = self.b, self.a
        self.scatter = np.concatenate((self.a, self.b, self.o))
        self._scatter2()

    def set_orth(self, x, faces):
        """Rim vertices with exactly 2 faces and one interior edge: (rim v, inner w, unit chord t)."""
        n = self.n_fixed
        fc = np.zeros(n, int)
        nb = {}
        for f in faces:
            m = len(f)
            for k in range(m):
                if f[k] < n:
                    fc[f[k]] += 1
                for u_, v_ in ((f[k], f[(k + 1) % m]), (f[(k + 1) % m], f[k])):
                    nb.setdefault(u_, set()).add(v_)
        vs, ws, ts = [], [], []
        for i in range(n):
            if fc[i] != 2:
                continue
            inner = [w for w in nb.get(i, ()) if not ((w == (i + 1) % n) or (w == (i - 1) % n))]
            if len(inner) != 1 or inner[0] < n:
                continue
            t = x[(i + 1) % n] - x[i - 1]
            t = t / max(np.linalg.norm(t), 1e-300)
            vs.append(i); ws.append(inner[0]); ts.append(t)
        if vs:
            self.orth_v = np.array(vs, np.int64); self.orth_w = np.array(ws, np.int64); self.orth_t = np.array(ts)
            self._scatter2()

    def sj(self, x):
        e1 = x[self.a] - x[self.o]; e2 = x[self.b] - x[self.o]
        d = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        return d / np.maximum(np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1), 1e-300)

    def face_areas(self, D):
        return np.bincount(self.fid, self.kc * D, self.nf)

    def adj_log(self, x):
        e1 = x[self.a] - x[self.o]; e2 = x[self.b] - x[self.o]
        A = self.face_areas(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
        return np.log(np.maximum(A[self.adj[:, 0]], 1e-300) / np.maximum(A[self.adj[:, 1]], 1e-300))

    def edge_log(self, x):
        ll = np.log(np.maximum(np.linalg.norm(x[self.a] - x[self.o], axis=1), 1e-300))
        return ll[self.epairs[:, 0]] - ll[self.epairs[:, 1]]


def _bar(z):
    """Barrier b(z) = (1-z)^2/z for 0 < z < 1 (0 above): value sum and derivative."""
    return (1 - z) ** 2 / z, -2 * (1 - z) / z - (1 - z) ** 2 / (z * z)


# ----------------------------------------------------------------------------- objective
class Objective:
    def __init__(self, C, T, w_shape=1.0, w_size=0.0, w_sj=0.0, p=16, floor=None, w_floor=0.0,
                 band=0.1, cap=None, guards=None, w_guard=1.0, guard_band=0.1, w_orth=0.0, w_aspect=0.0):
        self.C, self.T = C, T
        self.w_shape, self.w_size, self.w_sj, self.p = w_shape, w_size, w_sj, p
        self.floor = None if floor is None else np.asarray(floor, float)
        self.w_floor, self.band = w_floor, band
        self.gcap = None if cap is None else 1.0 / max(cap, 1e-6)
        self.m = C.live
        self.count = max(int(self.m.sum()), 1)
        self.wl = self.m / self.count
        self.guards, self.w_guard, self.gband = guards, w_guard, guard_band
        self.w_orth = w_orth if C.orth_v is not None else 0.0
        self.w_aspect = w_aspect if len(C.epairs) else 0.0

    def __call__(self, x, grad=True):
        C, m = self.C, self.m
        X = x[:, 0]; Y = x[:, 1]
        xo = X[C.o]; yo = Y[C.o]
        x1 = X[C.a] - xo; y1 = Y[C.a] - yo; x2 = X[C.b] - xo; y2 = Y[C.b] - yo
        D = x1 * y2 - y1 * x2
        if not (D > 0).all():
            return math.inf, None
        nc = len(D)
        E = 0.0
        buf = np.zeros((5, nc))
        cD, c1x, c1y, c2x, c2y = buf
        s11 = x1 * x1 + y1 * y1; s22 = x2 * x2 + y2 * y2
        wl = self.wl
        if self.w_shape:
            if C.any_tri:
                w00, w01, w10, w11 = C.w00, C.w01, C.w10, C.w11
                m1x = x1 * w00 + x2 * w10; m1y = y1 * w00 + y2 * w10
                m2x = x1 * w01 + x2 * w11; m2y = y1 * w01 + y2 * w11
                J = D * C.wdet
                kap = (m1x * m1x + m1y * m1y + m2x * m2x + m2y * m2y) / (2 * J)
                if grad:
                    k = self.w_shape * wl / J
                    c1x += k * (m1x * w00 + m2x * w01); c1y += k * (m1y * w00 + m2y * w01)
                    c2x += k * (m1x * w10 + m2x * w11); c2y += k * (m1y * w10 + m2y * w11)
            else:
                kap = (s11 + s22) / (2 * D)
                if grad:
                    k = self.w_shape * wl / D
                    c1x += k * x1; c1y += k * y1; c2x += k * x2; c2y += k * y2
            if grad:
                cD -= self.w_shape * wl * kap / D
            E += self.w_shape * float(np.dot(kap, wl))
        if self.w_size:
            tau = D / self.T
            E += self.w_size * float(np.dot(0.5 * (tau + 1 / tau) - 1, wl))
            if grad:
                cD += self.w_size * wl * 0.5 * (1 - 1 / (tau * tau)) / self.T
        if self.w_sj or self.floor is not None:
            l12 = np.sqrt(s11 * s22)
        if self.w_sj:
            g = l12 / D
            gm_all = g[m]
            if self.gcap is not None:
                gm_all = np.maximum(gm_all, self.gcap)
                act = m & (g > self.gcap)
            else:
                act = m
            p = self.p
            gm = float(gm_all.max())
            P = gm * float(np.mean((gm_all / gm) ** p)) ** (1.0 / p)
            E += self.w_sj * P
            if grad and act.any():
                idx = np.nonzero(act)[0]
                ga = g[idx]
                w = (P / gm) ** (1 - p) * (ga / gm) ** (p - 1) / len(gm_all) * self.w_sj * ga
                t1 = w / s11[idx]; t2 = w / s22[idx]
                c1x[idx] += t1 * x1[idx]; c1y[idx] += t1 * y1[idx]
                c2x[idx] += t2 * x2[idx]; c2y[idx] += t2 * y2[idx]
                cD[idx] -= w / D[idx]
        if self.floor is not None and self.w_floor:
            sj = D / l12
            z = (sj - self.floor) / self.band
            if not (z[m] > 0).all():
                return math.inf, None
            idx = np.nonzero(m & (z < 1))[0]
            if len(idx):
                bv, bd = _bar(z[idx])
                E += self.w_floor * float(bv.sum()) / self.count
                if grad:
                    db = bd / self.band * (self.w_floor / self.count) * sj[idx]
                    cD[idx] += db / D[idx]
                    t1 = db / s11[idx]; t2 = db / s22[idx]
                    c1x[idx] -= t1 * x1[idx]; c1y[idx] -= t1 * y1[idx]
                    c2x[idx] -= t2 * x2[idx]; c2y[idx] -= t2 * y2[idx]
        rr = None; dpair = None
        if self.guards:
            gb = self.gband
            cap = self.guards.get("adj")
            if cap is not None:
                A = C.face_areas(D)
                fa, fb = C.adj[:, 0], C.adj[:, 1]
                r = np.log(A[fa] / A[fb])
                z = (cap - np.abs(r)) / gb
                if not (z > 0).all():
                    return math.inf, None
                idx = np.nonzero(z < 1)[0]
                if len(idx):
                    bv, bd = _bar(z[idx]); n_ = len(z)
                    E += self.w_guard * float(bv.sum()) / n_
                    if grad:
                        dr = bd * (-np.sign(r[idx])) / gb * (self.w_guard / n_)
                        fa_, fb_ = fa[idx], fb[idx]
                        dA = np.bincount(fa_, dr / A[fa_], C.nf) - np.bincount(fb_, dr / A[fb_], C.nf)
                        cD += dA[C.fid] * C.kc
            cap = self.guards.get("edge")
            if cap is not None:
                ll = 0.5 * np.log(s11)
                ci, cj = C.epairs[:, 0], C.epairs[:, 1]
                r = ll[ci] - ll[cj]
                rr = r
                z = (cap - np.abs(r)) / gb
                if not (z > 0).all():
                    return math.inf, None
                idx = np.nonzero(z < 1)[0]
                if len(idx):
                    bv, bd = _bar(z[idx]); n_ = len(z)
                    E += self.w_guard * float(bv.sum()) / n_
                    if grad:
                        dpair = np.zeros(len(z))
                        dpair[idx] = bd * (-np.sign(r[idx])) / gb * (self.w_guard / n_)
        if self.w_aspect:
            # aspect: mean over in-face edge pairs of ln(l_i/l_j)^2 (all pairs: E7 = max/min edge)
            ci, cj = C.epairs[:, 0], C.epairs[:, 1]
            if rr is None:
                ll = 0.5 * np.log(s11)
                rr = ll[ci] - ll[cj]
            n_a = len(rr)
            E += self.w_aspect * float(np.dot(rr, rr)) / n_a
            if grad:
                da = 2 * rr * (self.w_aspect / n_a)
                dpair = da if dpair is None else dpair + da
        if grad and dpair is not None:
            ci, cj = C.epairs[:, 0], C.epairs[:, 1]
            t = (np.bincount(ci, dpair, nc) - np.bincount(cj, dpair, nc)) / s11
            c1x += t * x1; c1y += t * y1
        if self.w_orth:
            d = x[C.orth_w] - x[C.orth_v]
            dd = d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]
            dt = d[:, 0] * C.orth_t[:, 0] + d[:, 1] * C.orth_t[:, 1]
            n_o = len(dd)
            E += self.w_orth * float(np.sum(dt * dt / dd)) / n_o
        if not grad:
            return E, None
        g1x = c1x + cD * y2; g1y = c1y - cD * x2
        g2x = c2x - cD * y1; g2y = c2y + cD * x1
        nv = C.nv
        if self.w_orth:
            k = self.w_orth / n_o
            a_ = 2 * k * dt / dd; b_ = 2 * k * dt * dt / (dd * dd)
            w = np.concatenate((g1x, g2x, -(g1x + g2x), a_ * C.orth_t[:, 0] - b_ * d[:, 0],
                                g1y, g2y, -(g1y + g2y), a_ * C.orth_t[:, 1] - b_ * d[:, 1]))
            g = np.bincount(C.scatter2o, w, 2 * nv)
        else:
            w = np.concatenate((g1x, g2x, -(g1x + g2x), g1y, g2y, -(g1y + g2y)))
            g = np.bincount(C.scatter2, w, 2 * nv)
        return E, g.reshape(2, nv).T


# ----------------------------------------------------------------------------- solver
def lbfgs_steps(fun, x0, free, iterations, memory=8, deadline=None, ftol=0.0, scale=1.0, stop=None):
    """Minimise fun over x[free] (x: (V,2)); fun(x) -> (E, G). Armijo backtracking; a
    non-finite energy (barrier wall) is a rejected step. Returns x, E, iterations."""
    x = x0.copy()
    E, G = fun(x)
    if not math.isfinite(E):
        return x, E, 0
    g = G[free].ravel()
    S, Y = [], []
    stall = 0
    it = -1
    for it in range(iterations):
        yield   # a slice point
        if (deadline is not None and time.perf_counter() > deadline) or (stop is not None and stop()):
            break
        q = g.copy(); al = []
        for s_, y_ in reversed(list(zip(S, Y))):
            rho = 1.0 / np.dot(y_, s_); a_ = rho * np.dot(s_, q); q -= a_ * y_; al.append((rho, a_))
        if S:
            q *= np.dot(S[-1], Y[-1]) / np.dot(Y[-1], Y[-1])
        for (s_, y_), (rho, a_) in zip(zip(S, Y), reversed(al)):
            q += s_ * (a_ - rho * np.dot(y_, q))
        d = -q
        slope = float(np.dot(g, d))
        if slope >= 0:
            d = -g; slope = -float(np.dot(g, g)); S.clear(); Y.clear()
        if slope == 0:
            break
        step = 1.0
        if not S:   # first / restarted step: move at most 1% of the rim spacing
            step = min(1.0, 1e-2 * scale / max(float(np.abs(d).max()), 1e-300))
        while True:
            xn = x.copy(); xn[free] += (step * d).reshape(-1, 2)
            En, Gn = fun(xn)
            if math.isfinite(En) and En <= E + 1e-4 * step * slope:
                break
            step *= 0.5
            if step < 1e-14:
                return x, E, it
        gn = Gn[free].ravel()
        s_ = (xn[free] - x[free]).ravel(); y_ = gn - g
        if np.dot(s_, y_) > 1e-300:
            S.append(s_); Y.append(y_)
            if len(S) > memory:
                S.pop(0); Y.pop(0)
        rel = (E - En) / max(abs(E), 1e-300)
        x, E, g = xn, En, gn
        stall = stall + 1 if rel < ftol else 0
        if rel <= 1e-12 or stall >= 3:
            break
    return x, E, it + 1


# ----------------------------------------------------------------------------- helpers
def laplace_steps(pos, n_fixed, faces, iterations=400):
    """Uniform graph-Laplacian (Tutte) placement of free vertices (CG)."""
    out = np.array(pos, float).copy()
    nv = len(out); m = nv - n_fixed
    if m <= 0:
        return out
    ed = set()
    for fi, f in enumerate(faces):
        if fi and not fi & 63:
            yield   # a slice point
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            ed.add((min(a, b), max(a, b)))
    e = np.array(sorted(ed), np.int64)
    deg = np.bincount(e.ravel(), minlength=nv).astype(float)[n_fixed:]
    fr = e >= n_fixed
    both = fr[:, 0] & fr[:, 1]
    ea, eb = e[both, 0] - n_fixed, e[both, 1] - n_fixed
    one = fr[:, 0] ^ fr[:, 1]
    fi = np.where(fr[one, 0], e[one, 0], e[one, 1]) - n_fixed
    fx = np.where(fr[one, 0], e[one, 1], e[one, 0])
    dim = out.shape[1]
    rhs = np.column_stack([np.bincount(fi, out[fx, k], m) for k in range(dim)])
    ia = np.concatenate((ea, eb)); ib = np.concatenate((eb, ea))

    def apply(z):
        return deg[:, None] * z - np.column_stack([np.bincount(ia, z[ib, k], m) for k in range(dim)])
    z = np.repeat(out[:n_fixed].mean(0)[None], m, 0)
    r = rhs - apply(z); pp = r.copy(); rs = float(np.sum(r * r)); sc = max(float(np.sum(rhs * rhs)), 1e-300)
    for it in range(iterations):
        if not it & 7:
            yield
        if rs <= 1e-18 * sc:
            break
        ap = apply(pp); alpha = rs / float(np.sum(pp * ap)); z += alpha * pp; r -= alpha * ap
        rn = float(np.sum(r * r)); pp = r + rn / rs * pp; rs = rn
    out[n_fixed:] = z
    return out


def laplace(*args, **kwargs):
    """laplace_steps run to its end."""
    return _drive(laplace_steps(*args, **kwargs))


def size_targets_steps(x, C, faces, n_fixed):
    """Per-corner target D from a harmonic size field: h at a rim vertex = mean adjacent rim
    edge, harmonic inward; face target area = mean(h)^2 (triangle: equilateral area),
    normalised so the targets sum to the polygon area."""
    rim = x[:n_fixed]
    el = np.linalg.norm(np.roll(rim, -1, 0) - rim, axis=1)
    h_rim = 0.5 * (el + np.roll(el, 1))
    hp = np.zeros((len(x), 1)); hp[:n_fixed, 0] = h_rim
    h = (yield from laplace_steps(hp, n_fixed, faces))[:, 0]
    cnt = np.bincount(C.fid, minlength=C.nf)
    hm = np.bincount(C.fid, h[C.o], C.nf) / cnt
    tf = hm * hm * np.where(cnt == 3, 0.5 * SIN60, 1.0)
    area = 0.5 * float(np.sum(rim[:, 0] * np.roll(rim[:, 1], -1) - np.roll(rim[:, 0], -1) * rim[:, 1]))
    tf *= abs(area) / max(float(tf.sum()), 1e-300)
    return np.where(C.tri, 2 * tf[C.fid], tf[C.fid])   # quad corner D ~ area, triangle D = 2 area


def untangle_steps(x, C, iterations=400, theta=1.0 / 128, stop_valid=True, stop=None):
    """Regularised-barrier untangling (Garanzha et al. 2021)."""
    e1 = x[C.a] - x[C.o]
    s = float(np.mean(np.linalg.norm(e1, axis=1))) or 1.0

    def make(eps):
        def fun(z):
            e1 = (z[C.a] - z[C.o]) / s; e2 = (z[C.b] - z[C.o]) / s
            x1, y1, x2, y2 = e1[:, 0], e1[:, 1], e2[:, 0], e2[:, 1]
            m1x = x1 * C.w00 + x2 * C.w10; m1y = y1 * C.w00 + y2 * C.w10
            m2x = x1 * C.w01 + x2 * C.w11; m2y = y1 * C.w01 + y2 * C.w11
            det = m1x * m2y - m1y * m2x
            root = np.sqrt(eps * eps + det * det); chi = 0.5 * (det + root); dchi = 0.5 * (1 + det / root)
            fro = m1x * m1x + m1y * m1y + m2x * m2x + m2y * m2y
            f = ((1 - theta) * fro + theta * (det * det + 1)) / chi
            dfd = theta * 2 * det / chi - f / chi * dchi
            k = (1 - theta) * 2 / chi
            # dE/dm = k m + dfd * cof(m);  cof wrt (m1, m2): d det/dm1 = (m2y, -m2x), d det/dm2 = (-m1y, m1x)
            d1x = k * m1x + dfd * m2y; d1y = k * m1y - dfd * m2x
            d2x = k * m2x - dfd * m1y; d2y = k * m2y + dfd * m1x
            g1x = (d1x * C.w00 + d2x * C.w01) / s; g1y = (d1y * C.w00 + d2y * C.w01) / s
            g2x = (d1x * C.w10 + d2x * C.w11) / s; g2y = (d1y * C.w10 + d2y * C.w11) / s
            gx = np.bincount(C.scatter, np.concatenate((g1x, g2x, -(g1x + g2x))), C.nv)
            gy = np.bincount(C.scatter, np.concatenate((g1y, g2y, -(g1y + g2y))), C.nv)
            return float(f.sum()), np.column_stack((gx, gy)), det
        return fun
    eps = 1.0
    for _ in range(30):
        yield   # a slice point
        if stop is not None and stop():
            break
        fun = make(eps)
        E0 = fun(x)[0]
        x, E1, _ = (yield from lbfgs_steps(lambda z: fun(z)[:2], x, C.free, max(iterations // 4, 1), scale=s, stop=stop))
        mind = float(fun(x)[2].min())
        if mind > 0 and stop_valid:
            break
        sigma = max(1 - E1 / E0, 0.1) if E0 > 0 else 0.1
        mu = (1 - sigma) * 0.5 * (mind + math.sqrt(eps * eps + mind * mind))
        eps = 2 * math.sqrt(max(mu * (mu - mind), 0.0)) if mind < mu else 1e-8
        if mind > 0 and E0 > 0 and (E0 - E1) / E0 < 1e-4:
            break
    return x


def untangle(*args, **kwargs):
    """untangle_steps run to its end (the synchronous form)."""
    return _drive(untangle_steps(*args, **kwargs))


def orient_faces_steps(faces, n_fixed):
    """Consistently orient a face list (BFS over shared edges) so the rim edges run
    0->1->...->n-1->0 or its reverse coherently. Returns (faces, changed)."""
    owner = {}
    for fi, f in enumerate(faces):
        if fi and not fi & 63:
            yield   # a slice point
        for k in range(len(f)):
            e = (min(f[k], f[(k + 1) % len(f)]), max(f[k], f[(k + 1) % len(f)]))
            owner.setdefault(e, []).append(fi)
    out = [list(f) for f in faces]
    seen = [False] * len(out)
    changed = False

    def dirs(f):
        return {(f[k], f[(k + 1) % len(f)]) for k in range(len(f))}
    for root in range(len(out)):
        if seen[root]:
            continue
        seen[root] = True
        stack = [root]
        while stack:
            fi = stack.pop()
            if fi and not fi & 63:
                yield
            d = dirs(out[fi])
            for (u, v) in d:
                for gj in owner[(min(u, v), max(u, v))]:
                    if gj == fi or seen[gj]:
                        continue
                    if (u, v) in dirs(out[gj]):        # same direction: neighbour is reversed
                        out[gj] = out[gj][::-1]; changed = True
                    seen[gj] = True
                    stack.append(gj)
    return out, changed


def _plane_frame(P, n_fixed):
    rim = P[:n_fixed]
    c = rim.mean(0)
    nrm = np.sum(np.cross(rim - c, np.roll(rim, -1, 0) - c), 0)
    nrm /= max(np.linalg.norm(nrm), 1e-300)
    ref = np.array([1.0, 0, 0]) if abs(nrm[0]) < .9 else np.array([0, 1.0, 0])
    u = np.cross(nrm, ref); u /= np.linalg.norm(u); v = np.cross(nrm, u)
    return c, u, v, nrm


# ----------------------------------------------------------------------------- driver
def optimize_steps(positions, faces, n_fixed, **opts):
    """Optimize free vertex positions; the rim (first n_fixed) is fixed. See module doc."""
    o = dict(DEFAULTS)
    o.update(PRESETS[opts.pop("preset", "FULL")])
    o.update(opts)
    t0 = time.perf_counter()
    deadline = None if o["time_budget"] is None else t0 + float(o["time_budget"])
    live = o["deadline"] if callable(o["deadline"]) else None   # a callable deadline is read at every check
    if o["deadline"] is not None and live is None:
        deadline = float(o["deadline"]) if deadline is None else min(deadline, float(o["deadline"]))
    cancel = o["cancel"]
    hit = [False]

    def stop():
        if hit[0]:
            return True
        late = live() if live is not None else None
        late = deadline if late is None else (late if deadline is None else min(deadline, late))
        if (late is not None and time.perf_counter() > late) or (cancel is not None and cancel()):
            hit[0] = True
        return hit[0]

    def check(stage):
        """Between stages: honour the budget (skip the remaining stages)."""
        if stop():
            info["budget_hit"] = stage
            return True
        return False
    P = np.asarray(positions, float)
    faces = [list(map(int, f)) for f in faces]
    lift = None
    if P.shape[1] == 3:
        c, u, v, nrm = _plane_frame(P, n_fixed)
        x = np.column_stack(((P - c) @ u, (P - c) @ v))
        lift = (c, u, v, nrm, (P - c) @ nrm)
    else:
        x = P.copy()
    info = dict(stages=[], version=__version__)
    faces, reor = yield from orient_faces_steps(faces, n_fixed)
    yield   # a slice point
    if reor:
        info["reoriented"] = True
    C = yield from Corners.create_steps(faces, len(x), n_fixed)
    yield
    # orientation vs the plane: total signed area = sum over corners of x_o x x_a / 2
    xo, xa = x[C.o], x[C.a]
    ar = 0.5 * float(np.sum(xo[:, 0] * xa[:, 1] - xo[:, 1] * xa[:, 0]))
    if ar < 0:
        C.flip()
    work = [f[::-1] for f in faces] if ar < 0 else faces
    rim = x[:n_fixed]
    scale = float(np.mean(np.linalg.norm(np.roll(rim, -1, 0) - rim, axis=1)))
    info["flipped_faces"] = ar < 0
    sj0 = C.sj(x)
    valid_in = bool((sj0 > 0).all())
    info["in"] = dict(valid=valid_in, minSJ=float(sj0.min()))
    cands = []
    if valid_in:
        cands.append(("input", x.copy()))
    elif o["untangle"]:
        # raw / folded input: its ratios are not a standard to keep; use absolute caps
        o["adj_min_cap"] = max(o["adj_min_cap"], o["raw_adj_cap"])
        o["edge_min_cap"] = max(o["edge_min_cap"], o["raw_edge_cap"])
        x = (yield from untangle_steps(x, C, stop=stop))
        info["stages"].append(("untangle",))
        if not (C.sj(x) > 0).all():
            check("untangle")
        if (C.sj(x) > 0).all():
            cands.append(("untangle", x.copy()))
    if not cands:
        info.update(ok=False, seconds=time.perf_counter() - t0)
        return np.asarray(positions, float).copy(), info
    if not C.live.any() or len(C.free) == 0:
        o = dict(o, stage_a=False, stage_b=False, stage_c=False)
    if o["w_orth"]:
        C.set_orth(x, work)
    T = yield from size_targets_steps(x, C, work, n_fixed)
    yield
    guards = None
    if o["guards"]:
        guards = {}
        ra = np.abs(C.adj_log(x)); re_ = np.abs(C.edge_log(x))
        if len(ra):
            ga = max(float(np.percentile(ra, 95)), math.log(o["adj_min_cap"]))
            guards["adj"] = np.maximum(ga, ra) + o["guard_slack"]
        if len(re_):
            fmax = np.zeros(C.nf); np.maximum.at(fmax, C.epair_face, re_)
            ge = max(float(np.percentile(fmax, 90)), math.log(o["edge_min_cap"]))
            guards["edge"] = np.maximum(ge, re_) + o["guard_slack"]
    yield
    gk = dict(guards=guards or None, w_guard=o["w_guard"], guard_band=o["guard_band"])
    lk = dict(stop=stop, ftol=o["ftol"], scale=scale)
    if check("setup"):
        o = dict(o, stage_a=False, stage_b=False, stage_c=False)
    if o["stage_a"]:
        fa = Objective(C, T, w_shape=1.0, w_size=o["w_size"], w_orth=o["w_orth"], w_aspect=o["w_aspect"], **gk)
        x, _E, its = (yield from lbfgs_steps(fa, x, C.free, o["iters_a"], **lk))
        info["stages"].append(("A", its))
        cands.append(("A", x.copy()))
        if check("A"):
            o = dict(o, stage_b=False, stage_c=False)
    sj_now = C.sj(x)
    cap = min(float(sj_now[~C.live].min()) if (~C.live).any() else 1.0, 1.0)
    info["cap"] = cap
    if o["stage_b"] and float(sj_now[C.live].min()) >= cap - o["cap_tol"]:
        info["skip_b"] = "at cap"            # an immovable rim corner already sets min SJ
        o = dict(o, stage_b=False, stage_c=False)
    if o["stage_b"]:
        for p in (o["p_schedule"] if hasattr(o["p_schedule"], "__iter__") else (o["p_schedule"],)):
            fb = Objective(C, T, w_shape=o["w_reg"], w_size=o["w_reg"] * o["w_size"], w_sj=1.0, p=p, cap=cap, **gk)
            x, _E, its = (yield from lbfgs_steps(fb, x, C.free, o["iters_b"], **lk))
            info["stages"].append(("B", p, its))
            if stop():
                break
        cands.append(("B", x.copy()))
        if check("B"):
            o = dict(o, stage_c=False)
    if o["stage_c"] and o["stage_b"]:
        sjb = C.sj(x)
        t = min(float(sjb[C.live].min()), cap) - o["floor_slack"]
        floor = np.minimum(np.full(len(sjb), t), sjb - 1e-9)
        fc = Objective(C, T, w_shape=1.0, w_size=o["w_size"], w_orth=o["w_orth"], w_aspect=o["w_aspect"], floor=floor,
                       w_floor=o["w_floor"], band=o["floor_band"], **gk)
        xa = dict(cands).get("A")
        if xa is not None:
            ea = fc(xa, grad=False)[0]
            if math.isfinite(ea) and ea < fc(x, grad=False)[0]:
                x = xa.copy(); info["c_start"] = "A"
        x, _E, its = (yield from lbfgs_steps(fc, x, C.free, o["iters_c"], **lk))
        info["stages"].append(("C", its))
        cands.append(("C", x.copy()))
        check("C")
    rank = dict(input=0, untangle=0, A=1, B=2, C=3)
    st = [(nm, y, C.sj(y)) for nm, y in cands]
    top = max(float(s_.min()) for _n, _y, s_ in st)
    floor_in = float(sj0.min()) if valid_in else -math.inf
    ok = [(rank[nm], float(s_.mean()), nm, y) for nm, y, s_ in st
          if float(s_.min()) >= top - o["pick_tol"] and float(s_.min()) >= floor_in - 1e-12]
    _r, _m, name, x = max(ok, key=lambda t_: (t_[0], t_[1]))
    info.update(picked=name, out=dict(minSJ=float(C.sj(x).min())), ok=True)
    if lift is not None and float(np.abs(lift[4][:n_fixed]).max()) <= 1e-12 * max(scale, 1e-30):
        c, u, v, nrm, _hgt = lift          # flat rim: no height solve
        out = c + x[:, :1] * u + x[:, 1:2] * v
    elif lift is not None:
        c, u, v, nrm, hgt = lift
        hp = np.zeros((len(x), 1)); hp[:n_fixed, 0] = hgt[:n_fixed]
        h = (yield from laplace_steps(hp, n_fixed, faces))[:, 0]
        out = c + x[:, :1] * u + x[:, 1:2] * v + h[:, None] * nrm
    else:
        out = x.copy()
    out[:n_fixed] = P[:n_fixed]
    if reor:
        info["faces"] = faces              # callers must use these (consistent winding)
    info["seconds"] = time.perf_counter() - t0
    return out, info


def optimize(*args, **kwargs):
    """optimize_steps run to its end (the synchronous form)."""
    return _drive(optimize_steps(*args, **kwargs))


def place(positions, faces, n_rim, preset="FAST", **opts):
    """Stable entry point: new (V,3)/(V,2) positions; the input unchanged if D-opt fails.
    Same options as optimize(); preset="FAST" (default, ~50 ms) | "FULL".
    Budget: deadline=<abs perf_counter> / time_budget=<s> / cancel=<callable>; the best valid stage reached
    is returned.
    If the input's faces had inconsistent winding, use optimize() to get the repaired faces."""
    out, info = optimize(positions, faces, n_rim, preset=preset, **opts)
    return out if info.get("ok") else np.asarray(positions, float).copy()


