# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Topo Fill laid out on the source surface.

The source triangles under an even outline are flattened (a free-boundary conformal map, LSCM), the outline goes to
the usual planner at its flat positions, and the planned interior is mapped back onto those triangles, so it lies on
the source. The plain path (plan on one plane, then move each point to the nearest source point) is kept whenever
this cannot be done cleanly: no single mesh source under the outline, a flat patch, a patch over TRI_CAP triangles,
or triangles still flipped inside the outline after the trim pass. A candidate that folds on the source is refused on
its own, as the plain path refuses folding candidates.

prepare_steps()  the flattened patch of an outline (a Surface), or None for the plain path; cooperative.
planned_on()     the Surface a plan was planned on (its rim is that Surface's flat outline), else None.
part_steps()     one part of a crease split: the plane part or the part laid out on the source, by minQ."""

import math
import time

import numpy as np

from . import fill

ENABLED = True   # tests only (no setting): False keeps the plain path everywhere

TRI_CAP = 5000            # patch triangles at most (more: the plain path)
SOURCE_TRIS_MAX = 500000    # larger sources are not read (the plain path)
SLICE = .03               # seconds of work per step while a source mesh is read
NGON_CORNERS = 128        # a concave polygon with more corners keeps its fan (one ear clip stays within a step)
FACING = -.2              # patch triangles face the outline's side: normal . outline normal above this
SAMPLES = 4               # closest-point samples per outline edge (the triangles under the rim)
OFF_SOURCE = .25          # an outline vertex farther than this x h from the source is not on it
FLAT = 1e-6               # a patch within this x its size (or x its distance from the origin) of one plane is flat
MISS = .05                # a planned point this far (x h) outside the flat patch: the layout does not fit it
CG_ITERS, CG_TOL, CG_STEP = 2000, 1e-10, 20   # conjugate gradient: iterations, tolerance, iterations per step
_FOLDS = "This layout folds; try another choice (Ctrl+Wheel)"

_MESHES = {}   # source mesh key -> _Mesh (local space; a moved object keeps its entry); one source at a time
_MESHES_MAX = 1
_MAIN = {}     # (outline, sources) -> Surface or None, for whole outlines (planned_on reads these)
_PARTS = {}    # the same for the parts of a crease split
_KEEP = 8


def _remember(table, key, value, limit):
    table.pop(key, None)
    table[key] = value
    while len(table) > limit:
        table.pop(next(iter(table)))


# ---------------------------------------------------------------------------------------------- source mesh

class _Mesh:
    """one source mesh in local space: vertex coordinates, triangles (polygon fans), the first triangle of each
    polygon and the triangles around each vertex. A polygon of more than 4 corners is checked when the patch walk
    first reaches it (fix_steps): a fan with a triangle turned over or without area is replaced by an ear clip in
    the same rows, so first and the polygon of each triangle stay as read; vt_tris keeps the fans."""

    def __init__(self, co, tri, first, vt_start, vt_tris, ngons=False):
        self.co, self.tri, self.first, self.vt_start, self.vt_tris = co, tri, first, vt_start, vt_tris
        self.ngons = ngons   # the source has polygons of more than 4 corners
        self.checked = {}    # polygon -> 'fan' (its fan is valid), 'ear' (ear clipped) or 'kept' (fan kept: no ear clip)
        self.clipped = np.zeros(0, dtype=np.int64)   # the ear-clipped polygons, sorted

    def around(self, verts):
        """the triangles touching any of verts (with repeats)."""
        a, b = self.vt_start[verts], self.vt_start[verts + 1]
        count = b - a
        if not count.sum():
            return np.zeros(0, dtype=np.int32)
        offset = np.repeat(a - np.concatenate(([0], np.cumsum(count)[:-1])), count)
        rows = self.vt_tris[np.arange(int(count.sum())) + offset]
        if len(self.clipped):
            # each corner of a polygon is in one of its fan triangles, so vt_tris finds an ear-clipped polygon by
            # any of its corners; its own triangles are then tested against verts
            poly = np.searchsorted(self.first, rows, side='right') - 1
            mine = np.isin(poly, self.clipped)
            if mine.any():
                own = np.concatenate([np.arange(self.first[p], self.first[p + 1])
                                      for p in np.unique(poly[mine]).tolist()])
                rows = np.concatenate((rows[~mine], own[np.isin(self.tri[own], verts).any(1)]))
        return rows

    def around_steps(self, verts):
        """around(), with the polygons of more than 4 corners among those triangles checked first; cooperative."""
        rows = self.around(verts)
        if self.ngons and len(rows):
            poly = np.searchsorted(self.first, rows, side='right') - 1
            big = np.unique(poly[self.first[poly + 1] - self.first[poly] > 2]).tolist()
            new = [p for p in big if p not in self.checked]
            if new:
                yield from self.fix_steps(new)
                rows = self.around(verts)
        return rows

    def fix_steps(self, polys):
        """check each polygon of more than 4 corners in polys once: a valid fan stays, an invalid one is ear
        clipped (_ngon). Cooperative: a step every SLICE seconds, at most one polygon of NGON_CORNERS per step."""
        if not self.ngons:
            return
        clock = time.perf_counter()
        for p in polys:
            p = int(p)
            r0, r1 = int(self.first[p]), int(self.first[p + 1])
            if r1 - r0 < 3 or p in self.checked:
                continue   # a triangle or a quad keeps its fan
            fan = self.tri[r0:r1]
            corners = np.concatenate((fan[0, :2], fan[:, 2]))
            how, T = _ngon(self.co[corners].astype(np.float64)) if len(corners) <= NGON_CORNERS else ('kept', None)
            if how == 'ear':
                self.tri[r0:r1] = corners[T]
                self.clipped = np.union1d(self.clipped, [p])
            self.checked[p] = how
            if time.perf_counter() - clock >= SLICE:
                yield None
                clock = time.perf_counter()


def _mesh_key(obj):
    me = obj.data
    return (obj.name, me.as_pointer(), len(me.vertices), len(me.polygons), len(me.loops))


def _mesh_steps(obj):
    """the cached _Mesh of a source object, read once in steps of about SLICE seconds (None when it is too large or
    unreadable). Triangles: each polygon as a fan from its first corner (the whole-source reads are single calls, so
    the polygon starts and corner vertices are read rather than the slower loop triangles); a concave polygon of
    more than 4 corners is ear clipped later, only where the patch walk reaches it (_Mesh.fix_steps)."""
    key = _mesh_key(obj)
    hit = _MESHES.get(key)
    if hit is not None:
        return hit
    me = obj.data
    nv, np_, nl = len(me.vertices), len(me.polygons), len(me.loops)
    nt = nl - 2 * np_
    if not np_ or nt <= 0 or nt > SOURCE_TRIS_MAX or nv > SOURCE_TRIS_MAX:
        return None
    _MESHES.clear()   # one source at a time (never two in memory)
    co = np.empty(nv * 3, dtype=np.float32)
    me.vertices.foreach_get('co', co)
    yield None
    start = np.empty(np_, dtype=np.int32)
    me.polygons.foreach_get('loop_start', start)
    yield None
    corner = np.empty(nl, dtype=np.int32)
    attr = me.attributes.get('.corner_vert')
    if attr is not None and attr.domain == 'CORNER' and attr.data_type == 'INT' and len(attr.data) == nl:
        attr.data.foreach_get('value', corner)
    else:
        me.loops.foreach_get('vertex_index', corner)
    yield None
    total = np.diff(np.append(start, nl))
    if (total < 3).any():
        return None   # polygon corners not in order
    first = np.zeros(np_ + 1, dtype=np.int64)
    np.cumsum(total - 2, out=first[1:])
    chunk, clock = 1 << 16, time.perf_counter()

    def due():
        nonlocal clock
        if time.perf_counter() - clock < SLICE:
            return False
        clock = time.perf_counter()
        return True
    tri = np.empty((nt, 3), dtype=np.int32)
    for a in range(0, np_, chunk):
        b = min(a + chunk, np_)
        k = total[a:b] - 2
        base = np.repeat(start[a:b], k)
        j = np.arange(int(k.sum())) - np.repeat(first[a:b] - first[a], k)
        rows = slice(int(first[a]), int(first[b]))
        tri[rows, 0] = corner[base]
        tri[rows, 1] = corner[base + j + 1]
        tri[rows, 2] = corner[base + j + 2]
        if due():
            yield None
    flat = tri.ravel()
    counts = np.zeros(nv, dtype=np.int64)
    for s in range(0, 3 * nt, 2 * chunk):
        counts += np.bincount(flat[s:s + 2 * chunk], minlength=nv)
        if due():
            yield None
    vt_start = np.zeros(nv + 1, dtype=np.int64)
    np.cumsum(counts, out=vt_start[1:])
    cursor = vt_start[:-1].copy()
    vt_tris = np.empty(3 * nt, dtype=np.int32)
    for s in range(0, 3 * nt, 2 * chunk):   # counting sort of the corners by vertex, chunk by chunk (stable)
        v = flat[s:s + 2 * chunk]
        order = np.argsort(v, kind='stable')
        vs = v[order]
        head = np.flatnonzero(np.concatenate(([True], vs[1:] != vs[:-1])))
        runs = np.diff(np.concatenate((head, [len(vs)])))
        rank = np.arange(len(vs)) - np.repeat(head, runs)
        vt_tris[cursor[vs] + rank] = (order + s) // 3
        cursor[vs[head]] += runs
        if due():
            yield None
    mesh = _Mesh(co.reshape(-1, 3), tri, first, vt_start, vt_tris, ngons=bool((total > 4).any()))
    _remember(_MESHES, key, mesh, _MESHES_MAX)
    return mesh


def _sources(srcs):
    """(object, matrix, inverse, normal matrix) of each mesh source."""
    out = []
    for obj in srcs:
        try:
            if obj.type != 'MESH' or obj.data is None:
                continue
            mw = obj.matrix_world.copy()
            out.append((obj, mw, mw.inverted(), np.array(mw, dtype=np.float64)))
        except (ValueError, ReferenceError, AttributeError):
            continue
    return out


def _closest_steps(objs, P):
    """per point: (object index, world point, polygon index, distance); index -1 when no source has a surface.
    Cooperative: a step every 32 points."""
    from mathutils import Vector
    out = []
    for i, p in enumerate(P):
        if i and i % 32 == 0:
            yield None
        world = Vector((float(p[0]), float(p[1]), float(p[2])))
        best = (-1, None, -1, math.inf)
        for k, (obj, mw, mi, _m) in enumerate(objs):
            try:
                ok, co, _n, face = obj.closest_point_on_mesh(mi @ world)
            except RuntimeError:
                continue
            if not ok:
                continue
            cw = mw @ co
            d = (cw - world).length
            if d < best[3]:
                best = (k, np.array(cw.to_tuple()), int(face), d)
        out.append(best)
    return out


# ---------------------------------------------------------------------------------------------- geometry

def _ngon(P):
    """how a polygon P (m, 3) of more than 4 corners is triangulated, in its best-fit plane: ('fan', None) when the
    fan from its first corner is valid (every fan triangle turns the polygon's way, none without area); ('ear', T)
    with T (m - 2, 3) rows into P from ear clipping, in the polygon's winding; ('kept', None) when it cannot be ear
    clipped (no area, folded over in its plane, no ear left): the fan stays."""
    m = len(P)
    Q = P - P.mean(0)
    size = float(np.abs(Q).max())
    newell = np.cross(Q, np.roll(Q, -1, 0)).sum(0)   # twice the area vector: its direction is the winding
    eps = 1e-9 * size * size
    if not size > 0 or not float(np.linalg.norm(newell)) > eps:
        return 'kept', None
    e = np.linalg.svd(Q)[2]   # best-fit plane: e[0], e[1] span it, e[2] is its normal
    n = e[2] if float(e[2] @ newell) > 0 else -e[2]
    x, y = Q @ e[0], Q @ np.cross(n, e[0])
    area2 = float(x @ np.roll(y, -1) - np.roll(x, -1) @ y)
    if not area2 > eps:
        return 'kept', None   # folded over in its plane
    fx, fy = x[1:] - x[0], y[1:] - y[0]
    if (fx[:-1] * fy[1:] - fy[:-1] * fx[1:] > eps).all():
        return 'fan', None

    def turn(a, b, c):
        return (x[b] - x[a]) * (y[c] - y[a]) - (y[b] - y[a]) * (x[c] - x[a])
    nxt, prv = list(range(1, m)) + [0], [m - 1] + list(range(m - 1))
    alive = np.ones(m, dtype=bool)
    # corners that are not strictly convex (reflex, straight or repeated): the only ones that can lie in an ear
    reflex = np.array([turn(prv[i], i, nxt[i]) <= eps for i in range(m)])

    def ear(b):
        a, c = prv[b], nxt[b]
        if turn(a, b, c) <= eps:
            return False
        test = alive & reflex
        test[[a, b, c]] = False
        if not test.any():
            return True
        qx, qy = x[test], y[test]
        ax, ay, bx, by, cx, cy = x[a], y[a], x[b], y[b], x[c], y[c]
        inside = (((bx - ax) * (qy - ay) - (by - ay) * (qx - ax) >= -eps) &
                  ((cx - bx) * (qy - by) - (cy - by) * (qx - bx) >= -eps) &
                  ((ax - cx) * (qy - cy) - (ay - cy) * (qx - cx) >= -eps))
        # a corner at the very position of a, b or c (a repeated point) does not block the ear
        inside &= ~(((qx == ax) & (qy == ay)) | ((qx == bx) & (qy == by)) | ((qx == cx) & (qy == cy)))
        return not inside.any()
    out, state = [], [None] * m
    k, i, miss, tests, fresh = m, 0, 0, 0, True
    while k > 3:
        if state[i] is None:
            tests += 1
            if tests > 8 * m:
                return 'kept', None   # bounded work
            state[i] = ear(i)
        if state[i]:
            a, c = prv[i], nxt[i]
            out.append((a, i, c))
            alive[i] = False
            nxt[a], prv[c] = c, a
            for j in (a, c):
                state[j] = None
                reflex[j] = turn(prv[j], j, nxt[j]) <= eps
            k, miss, fresh, i = k - 1, 0, False, c
            continue
        miss += 1
        if miss > k:
            if fresh:
                return 'kept', None   # no ear left: degenerate
            state, miss, fresh = [None] * m, 0, True   # a clipped corner may have been the only one blocking an ear
        i = nxt[i]
    out.append((prv[i], i, nxt[i]))
    T = np.array(out, dtype=np.int64)
    a2 = (x[T[:, 1]] - x[T[:, 0]]) * (y[T[:, 2]] - y[T[:, 0]]) - (y[T[:, 1]] - y[T[:, 0]]) * (x[T[:, 2]] - x[T[:, 0]])
    if not a2.min() > eps or abs(float(a2.sum()) - area2) > 1e-6 * area2:
        return 'kept', None
    return 'ear', T


def _inside(Q, R):
    """even-odd point in polygon: Q (m, 2) in the polygon R (n, 2)."""
    x, y = Q[:, :1], Q[:, 1:2]
    x1, y1 = R[None, :, 0], R[None, :, 1]
    x2, y2 = np.roll(R[:, 0], -1)[None], np.roll(R[:, 1], -1)[None]
    with np.errstate(divide='ignore', invalid='ignore'):
        cross = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
    return cross.sum(1) % 2 == 1


def _bary(p, a, b, c):
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11, d20, d21 = v0 @ v0, v0 @ v1, v1 @ v1, v2 @ v0, v2 @ v1
    den = d00 * d11 - d01 * d01
    if abs(den) < 1e-300:
        return np.array([1., 0., 0.])
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return np.array([1 - v - w, v, w])


def _lscm_steps(V, T, pins, pin_uv, x0):
    """free-boundary conformal map of the triangles T over V, two pinned vertices; Jacobi-preconditioned conjugate
    gradient on the normal equations, warm-started at x0. Returns uv (n, 2)."""
    n = len(V)
    p0, p1, p2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    e1 = p1 - p0
    l1 = np.maximum(np.linalg.norm(e1, axis=1), 1e-300)
    ex = e1 / l1[:, None]
    nn = np.cross(e1, p2 - p0)
    a2 = np.maximum(np.linalg.norm(nn, axis=1), 1e-300)
    ey = np.cross(nn / a2[:, None], ex)
    X = np.zeros((len(T), 3, 2))
    X[:, 1, 0] = l1
    X[:, 2, 0] = ((p2 - p0) * ex).sum(1)
    X[:, 2, 1] = ((p2 - p0) * ey).sum(1)
    g = np.empty((len(T), 3, 2))
    for i in range(3):
        d = X[:, (i + 2) % 3] - X[:, (i + 1) % 3]
        g[:, i, 0], g[:, i, 1] = -d[:, 1] / a2, d[:, 0] / a2
    sa = np.sqrt(a2 / 2)
    t = np.arange(len(T))
    rows, cols, vals = [], [], []
    for i in range(3):
        for part, (ju, gv) in enumerate(((-g[:, i, 1], g[:, i, 0]), (g[:, i, 0], g[:, i, 1]))):
            r = 2 * t + part
            rows += [r, r]
            cols += [T[:, i], n + T[:, i]]
            vals += [sa * ju, -sa * gv]
    rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    m = 2 * len(T)
    free = np.ones(2 * n, bool)
    xp = np.zeros(2 * n)
    for pi, uv in zip(pins, pin_uv):
        free[pi] = free[n + pi] = False
        xp[pi], xp[n + pi] = uv

    def apply(x):
        return np.bincount(cols, vals * np.bincount(rows, vals * (x * free)[cols], minlength=m)[rows],
                           minlength=2 * n) * free
    b = -np.bincount(cols, vals * np.bincount(rows, vals * xp[cols], minlength=m)[rows], minlength=2 * n) * free
    D = np.bincount(cols, vals * vals, minlength=2 * n)
    D = np.where(free & (D > 0), D, 1.0)
    x = x0 * free
    r = b - apply(x)
    z = r / D
    p = z.copy()
    rz = r @ z
    b2 = max(float(b @ b), 1e-300)
    for it in range(CG_ITERS):
        if r @ r < CG_TOL * CG_TOL * b2:
            break
        Ap = apply(p)
        den = p @ Ap
        if not den > 0:
            break
        al = rz / den
        x += al * p
        r -= al * Ap
        z = r / D
        rz2 = r @ z
        p = z + (rz2 / rz) * p
        rz = rz2
        if it % CG_STEP == CG_STEP - 1:
            yield None
    x = x * free + xp
    return np.column_stack((x[:n], x[n:]))


def _signed(uv, T):
    d1, d2 = uv[T[:, 1]] - uv[T[:, 0]], uv[T[:, 2]] - uv[T[:, 0]]
    return d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]


# ---------------------------------------------------------------------------------------------- the surface

class Surface:
    """the flattened source patch under one outline. flat: the outline at its flat positions (world space, in the
    plane of the outline's frame), handed to the planner; map_steps() takes a plan planned on it back to the
    source."""

    def __init__(self, world, flat, frame, uv, T, V, normals, ok, h, key, vids):
        self.world, self.flat = world, flat
        self.c, self.e1, self.e2, self.N = frame
        self.uv, self.T, self.V, self.normals, self.ok = uv, T, V, normals, ok
        self.h, self.key, self.vids = h, key, vids
        self.plain = False   # the outline's main search went back to the plain path (its flat plans still map)
        self.seconds = 0.0   # seconds, mapped, folds: counters for tests only
        self.mapped = self.folds = 0

    def _fail(self):
        self.folds += 1
        raise fill.FillError(_FOLDS)

    def matches(self, plan):
        """the plan was planned on this flat outline."""
        try:
            P = np.asarray(plan['positions'], float)
            nb, order = plan['boundary'], plan['order']
            if nb != len(self.flat) or plan.get('virtual') is not None or P.ndim != 2 or len(P) < nb:
                return False
            ids = np.asarray(order[:nb], dtype=np.int64)
            return bool(np.allclose(P[:nb], self.flat[ids], rtol=0.0, atol=self.h * 1e-6))
        except (KeyError, TypeError, ValueError, IndexError):
            return False

    def _locate(self, Q):
        """per flat point (m, 2): (patch triangle, barycentric weights, distance outside the patch)."""
        uv, T = self.uv, self.T[self.ok]
        ids = np.flatnonzero(self.ok)
        a, b, c = uv[T[:, 0]], uv[T[:, 1]], uv[T[:, 2]]
        v0, v1 = b - a, c - a
        den = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
        k = np.empty(len(Q), dtype=np.int64)
        w = np.empty((len(Q), 3))
        miss = np.zeros(len(Q))
        for s in range(0, len(Q), 64):
            q = Q[s:s + 64]
            d = q[:, None, :] - a[None]
            vv = (d[..., 0] * v1[None, :, 1] - d[..., 1] * v1[None, :, 0]) / den
            ww = (v0[None, :, 0] * d[..., 1] - v0[None, :, 1] * d[..., 0]) / den
            uu = 1 - vv - ww
            worst = np.minimum(np.minimum(uu, vv), ww)
            best = np.argmax(worst, axis=1)
            r = np.arange(len(q))
            bw = np.clip(np.column_stack((uu[r, best], vv[r, best], ww[r, best])), 0, None)
            bw /= np.maximum(bw.sum(1), 1e-300)[:, None]
            k[s:s + 64], w[s:s + 64] = ids[best], bw
            out = worst[r, best] < -1e-9
            if out.any():
                tri = T[best[out]]
                near = (bw[out, :, None] * uv[tri]).sum(1)
                miss[s:s + 64][out] = np.linalg.norm(near - q[out], axis=1)
        return k, w, miss

    def map_steps(self, plan):
        """the plan (planned on self.flat) on the source: the rim at its own vertices, each interior point by its
        flat position on the patch. Raises FillError when this candidate folds on the source."""
        P = np.asarray(plan['positions'], float)
        nb = plan['boundary']
        out = P.copy()
        out[:nb] = self.world[np.asarray(plan['order'][:nb], dtype=np.int64)]
        Q = np.column_stack(((P[nb:] - self.c) @ self.e1, (P[nb:] - self.c) @ self.e2))
        if len(Q):
            k, w, miss = self._locate(Q)
            yield None
            if float(miss.max()) > MISS * self.h:
                self._fail()
            out[nb:] = np.einsum('ij,ijk->ik', w, self.V[self.T[k]])
        orient = 1.0 if float(np.dot(plan['normal_ccw'], self.N)) >= 0 else -1.0
        faces = fill.faces_of(plan)
        cen = np.array([P[list(f)].mean(0) for f in faces])
        tri, _w, _m = self._locate(np.column_stack(((cen - self.c) @ self.e1, (cen - self.c) @ self.e2)))
        if min_q(out, faces, self.normals[tri], orient) <= 0:
            self._fail()
        self.mapped += 1
        plan = dict(plan, positions=out, normal_ccw=orient * self.N)
        fill._accept_plan(plan)
        return plan

    def still_valid(self, srcs):
        """the source vertices of the patch are where they were read."""
        obj = next((o for o, _mw, _mi, _m in _sources(srcs) if _mesh_key(o) == self.key[0]), None)
        if obj is None:
            return False
        try:
            mw = np.array(obj.matrix_world, dtype=np.float64)
            verts = obj.data.vertices
            co = np.array([tuple(verts[int(i)].co) for i in self.vids], dtype=np.float64)
        except (ReferenceError, AttributeError, IndexError):
            return False
        W = co @ mw[:3, :3].T + mw[:3, 3]
        return W.shape == self.V.shape and bool(np.allclose(W, self.V, rtol=0.0, atol=self.h * 1e-6))


def min_q(positions, faces, normals, orient):
    """the smallest corner sine of the faces measured against the source normal at each face (one per face);
    a face turned over against it counts negative."""
    P = np.asarray(positions, float)
    worst = 1.0
    for f, nrm in zip(faces, normals):
        Q = P[list(f)]
        d1 = np.roll(Q, -1, 0) - Q
        d0 = np.roll(d1, 1, 0)
        s = orient * (np.cross(d0, d1) @ nrm) / np.maximum(
            np.linalg.norm(d0, axis=1) * np.linalg.norm(d1, axis=1), 1e-300)
        worst = min(worst, float(s.min()))
    return worst


def _src_ids(srcs):
    out = []
    for obj in srcs:
        try:
            out.append((obj.name, obj.data.as_pointer() if obj.data is not None else 0,
                        tuple(map(tuple, obj.matrix_world))))
        except (ReferenceError, AttributeError):
            out.append(None)
    return tuple(out)


def prepare_steps(world, srcs, main=True):
    """the Surface of an even outline on the projection sources, or None for the plain path. main: a whole outline
    (planned_on finds its plans); False: a part of a crease split."""
    if not ENABLED or not srcs:
        return None
    W = np.asarray(world, dtype=np.float64)
    n = len(W)
    if n < 4 or n % 2 or W.shape != (n, 3) or not np.isfinite(W).all():
        return None
    table = _MAIN if main else _PARTS
    key = (W.tobytes(), _src_ids(srcs))
    if key in table:
        hit = table[key]
        if hit is None:
            return None
        if hit.still_valid(srcs):
            return hit
    started = time.perf_counter()
    try:
        surface = yield from _build_steps(W, srcs)
        if surface is not None and not surface.still_valid(srcs):   # the cached mesh is older than the source
            _MESHES.pop(surface.key[0], None)
            surface = yield from _build_steps(W, srcs)
            if surface is not None and not surface.still_valid(srcs):
                surface = None
    except (ValueError, IndexError, ReferenceError, AttributeError, RuntimeError, MemoryError):
        surface = None   # a source that cannot be read this way: the plain path
    if surface is not None:
        surface.seconds = time.perf_counter() - started
    _remember(table, key, surface, _KEEP)
    return surface


def known(world, srcs):
    """(True, Surface or None) when prepare_steps has an answer for this whole outline without any work, else
    (False, None)."""
    if not ENABLED or not srcs:
        return True, None
    W = np.asarray(world, dtype=np.float64)
    key = (W.tobytes(), _src_ids(srcs))
    if key not in _MAIN:
        return False, None
    hit = _MAIN[key]
    if hit is None:
        return True, None
    return (True, hit) if hit.still_valid(srcs) else (False, None)


def planned_on(plan):
    """the whole-outline Surface whose flat outline this plan was planned on, else None."""
    for surface in reversed(list(_MAIN.values())):
        if surface is not None and surface.matches(plan):
            return surface
    return None


def _build_steps(W, srcs):
    n = len(W)
    h = float(np.median(np.linalg.norm(np.roll(W, -1, 0) - W, axis=1)))
    if not h > 0:
        return None
    objs = _sources(srcs)
    if not objs:
        return None
    S = np.array([W[i] + (W[(i + 1) % n] - W[i]) * (s / SAMPLES) for i in range(n) for s in range(SAMPLES)])
    hits = yield from _closest_steps(objs, S)
    yield None
    k0 = hits[0][0]
    if k0 < 0 or any(hk != k0 for hk, _p, _f, _d in hits):
        return None   # off the sources, or across two objects
    if max(hits[i * SAMPLES][3] for i in range(n)) > OFF_SOURCE * h:
        return None
    obj, mw_, _mi, mw = objs[k0]
    mesh = yield from _mesh_steps(obj)
    if mesh is None:
        return None
    M3, tr = mw[:3, :3], mw[:3, 3]

    def world(verts):
        return mesh.co[verts].astype(np.float64) @ M3.T + tr
    flip = 1.0 if np.linalg.det(M3) >= 0 else -1.0   # a mirroring matrix turns the triangle winding over

    def normals(ts):
        P = world(mesh.tri[ts].ravel()).reshape(-1, 3, 3)
        nn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]) * flip
        return P.mean(1), nn / np.maximum(np.linalg.norm(nn, axis=1), 1e-300)[:, None]
    rimq = np.array([hits[i * SAMPLES][1] for i in range(n)])
    rimf = np.array([hits[i * SAMPLES][2] for i in range(n)], dtype=np.int64)
    polys = np.unique([f for _k, _p, f, _d in hits])
    if polys.max() + 1 >= len(mesh.first):
        return None
    yield from mesh.fix_steps(polys.tolist())
    seeds = np.unique(np.concatenate([np.arange(mesh.first[p], mesh.first[p + 1]) for p in polys]))
    if not len(seeds):
        return None
    c = rimq.mean(0)
    _s, _v, vt = np.linalg.svd(rimq - c)
    e1, e2 = vt[0], vt[1]
    N = np.cross(e1, e2)   # a right-handed frame: triangles facing N run counter-clockwise in (e1, e2)
    _cs, fs = normals(seeds)
    if float(fs.mean(0) @ N) < 0:
        N, e2 = -N, -e2
    R2 = np.column_stack(((rimq - c) @ e1, (rimq - c) @ e2))
    size = float(np.ptp(R2, axis=0).max())
    if not size > 0:
        return None

    def facing(ts):
        cen, fn = normals(ts)
        return (fn @ N > FACING) & (np.abs((cen - c) @ N) < size), cen

    # grow from the triangles under the rim through triangles inside the outline (plus the rim triangles' ring)
    seen = set(seeds.tolist())
    keep = set(seeds.tolist())
    inner = set()   # the walked triangles inside the outline (not the rim triangles or their ring)
    ring = np.unique((yield from mesh.around_steps(np.unique(mesh.tri[seeds]))))
    ring = ring[~np.isin(ring, seeds)]
    seen.update(ring.tolist())
    fine, _cen = facing(ring) if len(ring) else (np.zeros(0, bool), None)
    front = np.concatenate((seeds, ring[fine]))
    keep.update(ring[fine].tolist())
    while len(front):
        nb = np.unique((yield from mesh.around_steps(np.unique(mesh.tri[front]))))
        nb = nb[[t not in seen for t in nb.tolist()]] if len(nb) else nb
        if not len(nb):
            break
        seen.update(nb.tolist())
        fine, cen = facing(nb)
        inside = _inside(np.column_stack(((cen - c) @ e1, (cen - c) @ e2)), R2)
        front = nb[fine & inside]
        keep.update(front.tolist())
        inner.update(front.tolist())
        if len(keep) > TRI_CAP:
            return None
        yield None
    # the part joined to the rim triangles across edges
    K = np.array(sorted(keep), dtype=np.int64)
    E = np.sort(np.stack((mesh.tri[K][:, [0, 1]], mesh.tri[K][:, [1, 2]], mesh.tri[K][:, [2, 0]]), 1), axis=2)
    E = E.astype(np.int64)
    code = E[..., 0] * (len(mesh.co) + 1) + E[..., 1]
    owner = np.repeat(np.arange(len(K)), 3)
    order = np.argsort(code.ravel(), kind='stable')
    sc, so = code.ravel()[order], owner[order]
    same = np.flatnonzero(sc[1:] == sc[:-1])
    adj = {}
    for a, b in zip(so[same].tolist(), so[same + 1].tolist()):
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    local = {t: i for i, t in enumerate(K.tolist())}
    comp = {local[t] for t in seeds.tolist()}
    stack = list(comp)
    while stack:
        for y in adj.get(stack.pop(), ()):
            if y not in comp:
                comp.add(y)
                stack.append(y)
    pf = K[sorted(comp)]
    yield None
    vids, T = np.unique(mesh.tri[pf], return_inverse=True)
    T = T.reshape(-1, 3)
    if flip < 0:
        T = T[:, ::-1].copy()   # the winding as seen in world space
    V = world(vids)
    # a flat source under the outline (within float32 noise, far from the world or the mesh origin too): the plain
    # path, exactly as without this module. The rim triangles and those inside the outline count, not the rim's ring
    # (it may reach over a crease).
    under = np.isin(pf, np.concatenate((seeds, np.fromiter(inner, dtype=np.int64, count=len(inner)))))
    vu = np.unique(mesh.tri[pf[under]]) if under.any() else np.zeros(0, dtype=np.int64)
    if len(vu) < 4:
        return None   # three points or fewer under the outline lie in one plane: flat
    Vu = world(vu)
    reach = max(size, float(np.abs(Vu).max()), float(np.abs(mesh.co[vu]).max()) * float(np.abs(M3).max()))
    pc = Vu.mean(0)
    pn = np.linalg.svd(Vu - pc)[2][2]
    if float(np.abs((Vu - pc) @ pn).max()) <= FLAT * reach:
        return None
    rim_tris = np.isin(pf, seeds)
    uv, ruv, T, V, pf, vids = yield from _flatten_steps(V, T, pf, vids, rimq, rimf, mesh, (c, e1, e2, N), rim_tris,
                                                        world, flip)
    if uv is None:
        return None
    # align the flat patch with the outline's plane (no mirroring), so the flat outline lies where the outline is
    A, B = ruv - ruv.mean(0), R2 - R2.mean(0)
    U, _s, Vt = np.linalg.svd(A.T @ B)
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] = -U[:, -1]
    rot = U @ Vt
    uv = (uv - ruv.mean(0)) @ rot + R2.mean(0)
    ruv = A @ rot + R2.mean(0)
    flat = c + ruv[:, :1] * e1 + ruv[:, 1:2] * e2
    sa = _signed(uv, T)
    _cen, fn = normals(pf)
    return Surface(W, flat, (c, e1, e2, N), uv, T, V, fn, sa > 0, h, (_mesh_key(obj),), vids)


def _rim_uv(uv, T, V, pf, rimq, rimf, mesh):
    """flat positions of the rim's closest points (barycentric on their own triangle)."""
    out = np.empty((len(rimq), 2))
    where = {}
    for i, t in enumerate(pf.tolist()):
        where[t] = i
    for i, (q, f) in enumerate(zip(rimq, rimf)):
        cands = [where[t] for t in range(mesh.first[f], mesh.first[f + 1]) if t in where]
        if not cands:
            return None
        best, bw = None, None
        for j in cands:
            w = _bary(q, *V[T[j]])
            if best is None or w.min() > bw.min():
                best, bw = j, w
        if bw.min() < -.05:
            return None   # the closest point is not on its own triangle: the polygon is not the patch's
        bw = np.clip(bw, 0, None)
        out[i] = (bw / max(bw.sum(), 1e-300)) @ uv[T[best]]
    return out


def _solve_steps(V, T, plan2):
    i0 = int(np.argmax(((V - V.mean(0)) ** 2).sum(1)))
    i1 = int(np.argmax(((V - V[i0]) ** 2).sum(1)))
    uv = yield from _lscm_steps(V, T, [i0, i1], [plan2[i0], plan2[i1]], np.concatenate((plan2[:, 0], plan2[:, 1])))
    a3 = float(np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1).sum())
    sa = _signed(uv, T)
    if sa.sum() < 0:
        uv[:, 1] = -uv[:, 1]
        sa = -sa
    return (uv - uv.mean(0)) * math.sqrt(a3 / max(abs(float(sa.sum())), 1e-300))


def _flatten_steps(V, T, pf, vids, rimq, rimf, mesh, frame, rim_tris, world, flip):
    """(uv, rim uv, T, V, patch triangles, vertex ids); uv None when triangles stay flipped inside the outline. One
    trim pass: the patch cut to the outline (plus the rim triangles and their ring) is solved again and the one with
    fewer flips inside the outline is kept."""
    c, e1, e2, _N = frame

    def solve(V_, T_, pf_):
        plan2 = np.column_stack(((V_ - c) @ e1, (V_ - c) @ e2))
        uv_ = yield from _solve_steps(V_, T_, plan2)
        ruv_ = _rim_uv(uv_, T_, V_, pf_, rimq, rimf, mesh)
        if ruv_ is None:
            return uv_, None, 1 << 30, None
        cen = (uv_[T_[:, 0]] + uv_[T_[:, 1]] + uv_[T_[:, 2]]) / 3
        inside = _inside(cen, ruv_)
        return uv_, ruv_, int(((_signed(uv_, T_) <= 0) & inside).sum()), inside
    uv, ruv, flips, inside = yield from solve(V, T, pf)
    if ruv is None:
        return None, None, T, V, pf, vids
    if flips:
        ring = np.isin(T, np.unique(T[rim_tris])).any(1)
        keep = inside | ring
        if not keep.all():
            pf2 = pf[keep]
            vids2, T2 = np.unique(mesh.tri[pf2], return_inverse=True)
            T2 = T2.reshape(-1, 3)
            if flip < 0:
                T2 = T2[:, ::-1].copy()
            V2 = world(vids2)
            uv2, ruv2, flips2, _in2 = yield from solve(V2, T2, pf2)
            if ruv2 is not None and flips2 < flips:
                uv, ruv, flips, T, V, pf, vids = uv2, ruv2, flips2, T2, V2, pf2, vids2
    if flips:
        return None, None, T, V, pf, vids
    return uv, ruv, T, V, pf, vids


# ---------------------------------------------------------------------------------------------- crease split parts

_WORK = [0.0]   # seconds of work in part_steps so far (the crease search does not count it against its caps)
PART_SECONDS = 3.0   # work the part layouts of one crease search may take; after it, parts stay on the plane


def work_seconds():
    return _WORK[0]


def part_steps(pts, srcs, solve, plane, normals_at, budget):
    """_part_steps with its work time counted in _WORK and taken from budget['left'] (seconds, one per crease search);
    a spent budget: the plane part at once."""
    if budget['left'] <= 0:
        return plane, None
    job = _part_steps(pts, srcs, solve, plane, normals_at)
    try:
        while True:
            t = time.perf_counter()
            try:
                next(job)
            except StopIteration as done:
                spent = time.perf_counter() - t
                _WORK[0] += spent
                budget['left'] -= spent
                return done.value
            spent = time.perf_counter() - t
            _WORK[0] += spent
            budget['left'] -= spent
            if budget['left'] <= 0:
                return plane, None   # out of time: the plane part
            yield None
    finally:
        job.close()


def _part_steps(pts, srcs, solve, plane, normals_at):
    """one part of a crease split: (plan, 'plane' | 'surface' | None). plane: the part planned on its plane (the plain
    path; None when it did not fill); solve(points, project) plans a part; normals_at(points) -> source normals (None
    when a point has no source). A flat source under the part: the plane part at once (prepare_steps). Otherwise the
    part laid out on the source is taken when it fills and its minQ on the source is higher."""
    surface = yield from prepare_steps(pts, srcs, main=False)
    if surface is None:
        return plane, None
    try:
        alt = yield from solve(surface.flat, surface.map_steps)
    except fill.FillError:
        alt = None
    if alt is None:
        return plane, 'plane'
    if plane is None:
        return alt, 'surface'
    q_alt, q_plane = _plan_q(alt, normals_at, surface.N), _plan_q(plane, normals_at, surface.N)
    if q_alt is None or q_plane is None:
        return plane, 'plane'
    return (alt, 'surface') if q_alt > q_plane else (plane, 'plane')


def _plan_q(plan, normals_at, N):
    faces = fill.faces_of(plan)
    P = np.asarray(plan['positions'], float)
    cen = np.array([P[list(f)].mean(0) for f in faces])
    normals = normals_at(cen)
    if normals is None:
        return None
    orient = 1.0 if float(np.dot(plan['normal_ccw'], N)) >= 0 else -1.0
    return min_q(P, faces, normals, orient)
