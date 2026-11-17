# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Geometry helpers (numpy only, no bpy).

* turning angles for an ordered planar boundary
* boundary-locked Laplace placement (matrix-free conjugate gradients)
* flip detection and foldover-free untangling after Garanzha et al. 2021,
  "Foldover-free maps in 50 lines of code" (own implementation, L-BFGS in numpy)
"""

import math
import time
from contextlib import contextmanager

import numpy as np

from .. import memo_jobs

_DEADLINE = [[None]]   # [the deadline cell in use]; the cell holds the absolute deadline on clock() (None: none)
_CLOCK = [time]        # [the clock the fill's budget code reads]: time, or a work-unit clock while a step runs


def clock():
    """The clock the fill's budget code reads (.monotonic() / .perf_counter())."""
    return _CLOCK[0]


def push_clock(c):
    """Make `c` the clock until pop_clock(token); returns the token."""
    old = _CLOCK[0]
    _CLOCK[0] = c
    return old


def pop_clock(token):
    """Put back the clock that push_clock returned."""
    _CLOCK[0] = token


def push_deadline(cell):
    """Make `cell` (a one-item list) the deadline cell until pop_deadline(token); returns the token."""
    old = _DEADLINE[0]
    _DEADLINE[0] = cell
    return old


def pop_deadline(token):
    """Put back the deadline cell that push_deadline returned."""
    _DEADLINE[0] = token


def deadline_at():
    """The current deadline (absolute, on clock()); None: no deadline."""
    return _DEADLINE[0][0]


def set_deadline_at(value):
    """Set the current deadline cell's value."""
    _DEADLINE[0][0] = value


class TimeUp(Exception):
    """The fill ran out of its time budget."""


@memo_jobs.recorded_budget
@contextmanager
def time_budget(seconds):
    """Every optimisation inside stops when the budget is used up (no freezes)."""
    old = _DEADLINE[0][0]
    _DEADLINE[0][0] = _CLOCK[0].monotonic() + seconds
    try:
        yield
    finally:
        _DEADLINE[0][0] = old


@memo_jobs.recorded_read
def time_left():
    return float("inf") if _DEADLINE[0][0] is None else _DEADLINE[0][0] - _CLOCK[0].monotonic()


def check_time():
    if time_left() <= 0:
        raise TimeUp("time budget used up")

def turning(p2):
    """Exterior (left-turn) angle at each vertex in radians, ccw loop."""
    a = p2 - np.roll(p2, 1, axis=0)
    b = np.roll(p2, -1, axis=0) - p2
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    dot = np.sum(a * b, axis=1)
    return np.arctan2(cross, dot)

# ------------------------------------------------------------------ placement
def laplace(fixed_pos, count, quads, iterations=400, tol=1e-9):
    """Positions of vertices n..count-1 with vertices 0..n-1 fixed (uniform weights).

    fixed_pos: (n, dim). Returns (count, dim). Matrix-free CG on the graph Laplacian.
    """
    fixed_pos = np.asarray(fixed_pos, float)
    n, dim = fixed_pos.shape
    edges = set()
    for q in quads:
        for a, b in zip(q, q[1:] + q[:1]):
            edges.add((min(a, b), max(a, b)))
    e = np.array(sorted(edges), dtype=np.int64).reshape(-1, 2)
    pos = np.zeros((count, dim))
    pos[:n] = fixed_pos
    m = count - n
    if m == 0:
        return pos
    pos[n:] = fixed_pos.mean(axis=0)
    deg = np.bincount(e.ravel(), minlength=count).astype(float)[n:]
    free = (e >= n)
    both = free[:, 0] & free[:, 1]
    ea, eb = e[both, 0] - n, e[both, 1] - n
    # right-hand side: sum of fixed neighbours
    rhs = np.zeros((m, dim))
    one = free[:, 0] ^ free[:, 1]
    fi = np.where(free[one, 0], e[one, 0], e[one, 1]) - n
    fx = np.where(free[one, 0], e[one, 1], e[one, 0])
    np.add.at(rhs, fi, fixed_pos[fx])

    def apply(x):
        y = deg[:, None] * x
        np.add.at(y, ea, -x[eb])
        np.add.at(y, eb, -x[ea])
        return y

    x = pos[n:]
    r = rhs - apply(x)
    p = r.copy()
    rs = np.sum(r * r)
    scale = max(np.sum(rhs * rhs), 1e-30)
    for _ in range(iterations):
        if rs <= tol * tol * scale:
            break
        ap = apply(p)
        alpha = rs / np.sum(p * ap)
        x += alpha * p
        r -= alpha * ap
        rs_new = np.sum(r * r)
        p = r + (rs_new / rs) * p
        rs = rs_new
    pos[n:] = x
    return pos


def _corner_triangles(quads):
    q = np.asarray(quads, dtype=np.int64)
    tris = []
    for k in range(4):
        tris.append(np.stack([q[:, k], q[:, (k + 1) % 4], q[:, (k + 3) % 4]], axis=1))
    return np.concatenate(tris)


def min_corner_area(p2, quads):
    """Smallest signed corner-triangle area (<= 0 means a flipped or degenerate quad)."""
    t = _corner_triangles(quads)
    a = p2[t[:, 1]] - p2[t[:, 0]]
    b = p2[t[:, 2]] - p2[t[:, 0]]
    return float(np.min(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))


_PERP_V = np.array([1.0, -1.0])     # (v1, -v0) from v reversed
_PERP_U = np.array([-1.0, 1.0])     # (-u1, u0) from u reversed


class WorkUnitClock:
    """Base of the work-unit clocks (fill_progressive.WUClock). While one runs the current step (geom.clock()), the
    untangle charges it (_charge) and its clock reads made inside an untangle solve are free (_UNTANGLE)."""

    def charge(self, units):
        """Add work units."""
        self.ticks += units

    def set_state(self, ticks, quantum):
        """Set the tick count and the quantum (memo_jobs replays a recorded step under its recorded quantum)."""
        self.ticks = ticks
        self.quantum = quantum


_UNTANGLE = [0]   # untangle solves running now: a work-unit clock does not count the reads made inside them


def _charge(units):
    """Add work units to the work-unit clock running the current step, if any."""
    c = _CLOCK[0]
    if isinstance(c, WorkUnitClock):
        c.charge(units)


def untangle_steps(p2, n_fixed, quads, outer=30, inner=150, theta=1.0 / 128, step_limit=96):
    """Foldover-free placement (_untangle_python_steps). On a work-unit clock the solve costs 1 unit per call plus
    1 unit per energy+gradient evaluation (_lbfgs_steps); the clock reads made inside it are free."""
    _charge(1)
    job = _quiet_steps(_untangle_python_steps(p2, n_fixed, quads, outer, inner, theta, step_limit))
    try:
        _UNTANGLE[0] += 1
        try:
            val = next(job)
        finally:
            _UNTANGLE[0] -= 1
        while True:
            sent = yield val
            _UNTANGLE[0] += 1
            try:
                val = job.send(sent)
            finally:
                _UNTANGLE[0] -= 1
    except StopIteration as done:
        return done.value
    finally:
        job.close()


def _quiet_steps(gen):
    """drive a solve generator with np.errstate set around each resumed segment only (never across a yield)."""
    try:
        while True:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                try:
                    u = next(gen)
                except StopIteration as done:
                    return done.value
            yield u
    finally:
        gen.close()


def _untangle_python_steps(p2, n_fixed, quads, outer=30, inner=150, theta=1.0 / 128, step_limit=96):
    """Foldover-free placement in 2D with vertices < n_fixed locked.

    Energy per quad-corner triangle (reference: unit right angle scaled to the mean
    edge length): (1-theta) * |J|^2 / chi(D) + theta * (D^2 + 1) / chi(D),
    chi(D, eps) = (D + sqrt(eps^2 + D^2)) / 2, with eps decreased as in the paper.
    Returns (positions, ok).
    """
    p2 = np.array(p2, float)
    # A wall-clock cutoff makes mirrored patches stop at different layouts under
    # UI load. Cooperative iteration limits give each patch the same work budget.
    budget=[step_limit]
    fallback=p2.copy() if min_corner_area(p2,quads)>0 else None
    t = _corner_triangles(quads)
    free = np.arange(n_fixed, len(p2))
    if not len(free):
        return p2, min_corner_area(p2, quads) > 0
    e = p2[t[:, 1]] - p2[t[:, 0]]
    s = float(np.mean(np.linalg.norm(e, axis=1))) or 1.0
    scatter = np.concatenate((t[:, 1], t[:, 2], t[:, 0]))

    buf = p2.copy()   # one position buffer per solve (rim rows never change)
    n2 = 2 * len(p2)
    scatter2 = np.concatenate((2 * scatter, 2 * scatter + 1))

    def parts(x, eps):
        p = buf
        p[free] = x.reshape(-1, 2)
        T = p[t]   # one gather (m, 3, 2); the same values as p[t0], p[t1], p[t2]
        u = (T[:, 1] - T[:, 0]) / s
        v = (T[:, 2] - T[:, 0]) / s
        det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        return p, u, v, det

    memo = [None]   # the last evaluation (x object, eps) -> intermediates, reused by its gradient call

    def energy(x, eps, grad=True):   # runs under the solve's per-segment np.errstate (_quiet_steps)
        m = memo[0]
        if m is not None and m[0] is x and m[1] == eps:
            u, v, det, chi, dchi, f, E = m[2]
        else:
            _p, u, v, det = parts(x, eps)
            root = np.sqrt(eps * eps + det * det)
            chi = 0.5 * (det + root)
            dchi = 0.5 * (1 + det / root)
            fro = np.sum(u * u, 1) + np.sum(v * v, 1)
            f = ((1 - theta) * fro + theta * (det * det + 1)) / chi
            E = float(np.sum(f))
            memo[0] = (x, eps, (u, v, det, chi, dchi, f, E))
        if not grad:
            return E, det
        dfd = (theta * 2 * det) / chi - f / chi * dchi           # d f / d det
        gu = (1 - theta) * 2 * u / chi[:, None] + dfd[:, None] * (v[:, ::-1] * _PERP_V)
        gv = (1 - theta) * 2 * v / chi[:, None] + dfd[:, None] * (u[:, ::-1] * _PERP_U)
        # one bincount per axis sums in the same order as add.at over t1, t2, t0
        w = np.concatenate((gu / s, gv / s, -(gu + gv) / s))
        g = np.bincount(scatter2, np.concatenate((w[:, 0], w[:, 1])), n2).reshape(-1, 2)   # one bincount
        return E, det, g[free].ravel()

    x = p2[free].ravel()
    eps = 1.0
    _E, det = energy(x, eps, grad=False)
    for _ in range(outer):
        if time_left() <= 0 or budget[0]<=0:
            break
        E0 = energy(x, eps, grad=False)[0]
        x = yield from _lbfgs_steps(lambda z: energy(z, eps), x, inner,budget=budget,
                                    fun_e=lambda z: energy(z, eps, grad=False)[0])
        E1, det = energy(x, eps, grad=False)
        mindet = float(np.min(det))
        if mindet>0:
            candidate=p2.copy()
            candidate[free]=x.reshape(-1,2)
            if fallback is None or quad_quality(candidate,quads).mean()>quad_quality(fallback,quads).mean():
                fallback=candidate
        sigma = max(1.0 - E1 / E0, 0.1) if E0 > 0 else 0.1
        mu = (1 - sigma) * 0.5 * (mindet + math.sqrt(eps * eps + mindet * mindet))
        if mindet < mu:
            eps = 2 * math.sqrt(max(mu * (mu - mindet), 0.0))
        else:
            eps = 1e-8
        if mindet > 0 and E0 > 0 and (E0 - E1) / E0 < 1e-4:
            break
        yield None
    p = p2.copy()
    p[free] = x.reshape(-1, 2)
    if fallback is not None and (min_corner_area(p,quads)<=0 or quad_quality(fallback,quads).mean()>quad_quality(p,quads).mean()):
        p=fallback
    return p, min_corner_area(p, quads) > 0


def _lbfgs_steps(fun, x, iterations, memory=8,budget=None,fun_e=None):
    """Minimal L-BFGS with Armijo backtracking. fun(x) -> (E, aux, grad). Each fun call (energy + gradient) charges
    1 work unit (_charge); fun_e calls (energy only) charge nothing."""
    _charge(1)
    E, _aux, g = fun(x)
    S, Y = [], []
    for iteration in range(iterations):
        if time_left() <= 0 or (budget is not None and budget[0]<=0):
            return x
        if budget is not None:
            budget[0]-=1
        q = g.copy()
        alphas = []
        for s_, y_ in reversed(list(zip(S, Y))):
            rho = 1.0 / np.dot(y_, s_)
            a = rho * np.dot(s_, q)
            q -= a * y_
            alphas.append((rho, a))
        if S:
            q *= np.dot(S[-1], Y[-1]) / np.dot(Y[-1], Y[-1])
        for (s_, y_), (rho, a) in zip(zip(S, Y), reversed(alphas)):
            b = rho * np.dot(y_, q)
            q += s_ * (a - b)
        d = -q
        slope = np.dot(g, d)
        if slope >= 0:
            d, slope = -g, -np.dot(g, g)
            S.clear(); Y.clear()
        step = 1.0
        trials=0
        while True:
            xn = x + step * d
            if fun_e is None:
                _charge(1)
                En, _a, gn = fun(xn)
            else:   # energy only on a trial point; the gradient for the accepted one (same arithmetic)
                En = fun_e(xn)
            if np.isfinite(En) and En <= E + 1e-4 * step * slope:
                if fun_e is not None:
                    _charge(1)
                    En, _a, gn = fun(xn)
                break
            step *= 0.5
            if step < 1e-12:
                return x
            trials+=1
            if trials%8==0:
                yield None
        s_, y_ = xn - x, gn - g
        if np.dot(s_, y_) > 1e-12:
            S.append(s_); Y.append(y_)
            if len(S) > memory:
                S.pop(0); Y.pop(0)
        if abs(E - En) <= 1e-12 * max(1.0, abs(E)):
            return xn
        x, E, g = xn, En, gn
        if iteration % 4 == 3:
            yield None
    return x


def _winslow_stencil(count, n_fixed, quads):
    """Ordered neighbourhoods of the free vertices.

    Regular vertices (4 quads around) get the structured 3x3 stencil: edge
    neighbours e0..e3 counter-clockwise and the diagonal between e_i and e_i+1.
    Every other free vertex is listed with its edge neighbours for averaging."""
    around = {}
    for q in quads:
        for k in range(4):
            v = q[k]
            if v >= n_fixed:
                # counter-clockwise around v: q[k+1] -> q[k+2] (diagonal) -> q[k-1]
                around.setdefault(v, {})[q[(k + 1) % 4]] = (q[(k + 2) % 4], q[(k + 3) % 4])
    regular, other = [], []
    for v, links in around.items():
        ring = None
        if len(links) == 4:
            start = next(iter(links))
            ring, diag, current = [start], [], start
            for _ in range(4):
                if current not in links:
                    ring = None
                    break
                d, nxt = links[current]
                diag.append(d)
                current = nxt
                if current == start:
                    break
                ring.append(current)
            if ring is not None and (len(ring) != 4 or current != start):
                ring = None
        if ring is not None:
            regular.append([v] + ring + diag)
        else:
            neighbours = set(links)
            for d, prev in links.values():
                neighbours.add(prev)
            other.append((v, sorted(neighbours)))
    return np.asarray(regular, dtype=np.int64).reshape(-1, 9), other


def winslow_steps(p2, n_fixed, quads, iterations=400, relax=0.6):
    """Elliptic (Winslow) smoothing of a fold-free quad placement with vertices below
    n_fixed locked: grid lines straighten and spacing grades smoothly from the
    boundary. Poles average their neighbours. Returns the smoothed positions, or the
    input when smoothing would fold a quad or lower the worst quad quality."""
    p2 = np.array(p2, float)
    if len(p2) <= n_fixed:
        return p2
    reg, other = _winslow_stencil(len(p2), n_fixed, quads)
    if other:
        ov = np.array([v for v, _ in other], dtype=np.int64)
        flat = np.array([w for _, ws in other for w in ws], dtype=np.int64)
        owner = np.repeat(np.arange(len(other)), [len(ws) for _, ws in other])
        degree = np.bincount(owner, minlength=len(other)).astype(float)
    start_quality = quad_quality(p2, quads)
    best = p2.copy()
    x = p2.copy()
    scale = max(float(np.ptp(p2[:n_fixed], axis=0).max()), 1e-30)
    for it in range(iterations):
        new = x.copy()
        if len(reg):
            xv, x0, x1, x2, x3, d01, d12, d23, d30 = x[reg].transpose(1, 0, 2)
            xi = (x0 - x2) * .5
            eta = (x1 - x3) * .5
            alpha = np.sum(eta * eta, axis=1)
            gamma = np.sum(xi * xi, axis=1)
            beta = np.sum(xi * eta, axis=1)
            cross = (d01 - d12 + d23 - d30) * .25
            denom = np.maximum(2 * (alpha + gamma), 1e-30)
            target = (alpha[:, None] * (x0 + x2) + gamma[:, None] * (x1 + x3)
                      - 2 * beta[:, None] * cross) / denom[:, None]
            new[reg[:, 0]] = xv + relax * (target - xv)
        if other:
            near = x[flat]
            mean = np.column_stack((np.bincount(owner, near[:, 0], len(other)),
                                    np.bincount(owner, near[:, 1], len(other))))
            mean /= degree[:, None]
            new[ov] = x[ov] + relax * (mean - x[ov])
        move = float(np.abs(new - x).max())
        x = new
        if it % 20 == 19 or move < 1e-5 * scale:
            if min_corner_area(x, quads) <= 0:
                break
            best = x.copy()
            if move < 1e-5 * scale:
                break
            yield None
    if min_corner_area(best, quads) <= 0:
        return p2
    if quad_quality(best, quads).min() < min(start_quality.min(), .9) - 1e-9:
        return p2
    return best


def quad_quality(p2, quads):
    """Per quad: the smallest corner sine (1 = right angles, <= 0 = folded).
    All four corners in one pass: per corner the same element-wise arithmetic as _quad_quality_corners (gather,
    subtract, 2-element norm, cross, divide) and the minimum over the same five values (1 and the four corner sines);
    np.linalg.norm over an axis of length 2 is the same sqrt(x0*x0 + x1*x1) whatever the array rank."""
    q = np.asarray(quads, dtype=np.int64)
    if q.ndim != 2 or q.shape[1] != 4 or not len(q):
        return _quad_quality_corners(p2, quads)
    P = p2[q]                                   # (m, 4, 2): o for k = 0..3
    a = P[:, [1, 2, 3, 0]] - P                  # p2[q[:, (k+1) % 4]] - o
    b = P[:, [3, 0, 1, 2]] - P                  # p2[q[:, (k+3) % 4]] - o
    la = np.linalg.norm(a, axis=2)
    lb = np.linalg.norm(b, axis=2)
    s = (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]) / np.maximum(la * lb, 1e-30)
    return np.minimum(np.ones(len(q)), s.min(axis=1))


def _quad_quality_corners(p2, quads):
    """quad_quality corner by corner (for quad arrays the one-pass form does not take)."""
    q = np.asarray(quads, dtype=np.int64)
    worst = np.ones(len(q))
    for k in range(4):
        o = p2[q[:, k]]
        a = p2[q[:, (k + 1) % 4]] - o
        b = p2[q[:, (k + 3) % 4]] - o
        la = np.linalg.norm(a, axis=1)
        lb = np.linalg.norm(b, axis=1)
        s = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / np.maximum(la * lb, 1e-30)
        worst = np.minimum(worst, s)
    return worst
