# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Topo Fill along a source crease.

When two outline vertices sit on a clear fold of the projection source (about 30 degrees or more, sharp or a small
bevel), the outline is split along the fold: new vertices are traced on the fold between the two, and each part is
filled on its own and projected onto its own side. The fill then has an edge row on the fold and no face bridges it.

tail_steps()   all of it, run after the main search ends (it never slows or changes the main search):
detect_steps() outline vertex pairs on a crease (source face normals near the outline, plus a path of sharp source
               edges for low-poly sources); None when there is none, and then nothing else here runs.
fast_steps()   one quick split fill (each part: the direct grid, else the main solve), offered as a candidate at once.
full_steps()   the wider search: 2-3 seam vertex counts, each part filled with the full candidate set, or all parts as
               one panel network (each panel placed on its own plane).
Only even outlines in Topo mode (with projection sources) are considered."""

import heapq
import itertools
import math
import time

import numpy as np

from . import fill, fill_surface

ENABLED = True   # tests only (no setting): False skips the detection, so Fill runs exactly as without it

# detection on source face normals (angles in degrees, distances in outline edge lengths h)
THETA = 20.0       # a crease turns the normal by more than this
DIR_TOL = 35.0     # the crease direction at both vertices is parallel to their chord within this
NEAR, FAR = .5, .9   # normal probes this far along the outline edges (and across the seam)
FLAT = .3          # each side is flat: the normal changes from NEAR to FAR by less than this x the turn
AT = .3            # the turn is at the vertex: raw normals this far along the two edges differ by THETA / 2
SMOOTH_R = .25     # probe normals are averaged over 5 samples within this radius
MAX_TURN = 150.0   # a turn above this is faces with opposite winding (normals not recalculated), not a crease
VERTEX_ON = .25    # each vertex of a pair is at most this far from the crease (measured on the source)
# detection on sharp source edges (low-poly sources)
EDGE_THETA = 30.0    # a sharp edge: dihedral angle at least this
EDGE_ON = .15        # an outline vertex this close to a sharp edge is on it
EDGE_LEN = 1.3       # the sharp-edge path between two vertices is at most this x their chord
EDGE_CLEAR = .25     # and keeps this far from the rest of the outline
EDGE_FACES_MAX = 50000   # larger sources use the normal probes only (their mesh is not read)
# the split
SEAM_ALONG = .35     # seam tracing: the side normals are read this far along the outline edges at both ends
SEAM_ITERS = 12      # bisection steps per seam vertex
FULL_SECONDS = 8.0   # the wider search's comparison stops after this much own work (its re-run: PART_SECONDS more)
DETECT_SECONDS = 1.0   # detection gives up (no crease) after this much of its own work time
FAST_SECONDS = 2.0     # the quick split fill gives up (no quick fill) after this much of its own work time
GAP_SKIP = .1          # a part whose plane fill lies this close (x h) to the source everywhere is not laid out


class _Miss(Exception):
    """a probe found no source surface"""


class Probe:
    """Closest points and face normals on the projection sources, in world space. The normal is the hit face's own
    normal (closest_point_on_mesh ignores smooth shading), turned by each object's normal matrix."""

    def __init__(self, srcs):
        from mathutils import Vector
        self._vector = Vector
        self.data = []
        self.queries = 0
        for obj in srcs:
            try:
                mw = obj.matrix_world.copy()
                mi = mw.inverted()
            except (ValueError, ReferenceError, AttributeError):
                continue
            self.data.append((obj, mw, mi, mi.to_3x3().transposed()))

    def point(self, p):
        """(point, unit normal) as arrays; raises _Miss when no source has a surface."""
        world = self._vector((float(p[0]), float(p[1]), float(p[2])))
        best, best_d = None, math.inf
        self.queries += 1
        for obj, mw, mi, nm in self.data:
            try:
                ok, co, nrm, _face = obj.closest_point_on_mesh(mi @ world)
            except RuntimeError:
                continue
            if not ok:
                continue
            cw = mw @ co
            d = (cw - world).length
            if d < best_d:
                best_d, best = d, (cw, nm @ nrm)
        if best is None or best[1].length == 0.0:
            raise _Miss
        return np.array(best[0].to_tuple()), np.array(best[1].normalized().to_tuple())

    def normals(self, P):
        return np.array([self.point(p)[1] for p in np.atleast_2d(P)])

    def smooth(self, P, r):
        """normals averaged over the hit and 4 samples at radius r in its tangent plane."""
        out = []
        for p in np.atleast_2d(P):
            q, nk = self.point(p)
            e1 = _unit(_cross(nk, (1., 0., 0.) if abs(nk[0]) < .9 else (0., 1., 0.)))
            e2 = _cross(nk, e1)
            m = nk + sum(self.point(q + r * d)[1] for d in (e1, -e1, e2, -e2))
            out.append(_unit(m))
        return np.array(out)


def _ang(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b))))))


def _unit(v):
    return v / max(math.sqrt(float(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])), 1e-30)


def _cross(a, b):
    """the cross product of two 3-vectors (np.cross is slow on single vectors)"""
    return np.array((a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]))


class Crease:
    """the detected pairs of one outline, with the queries the split reuses."""

    def __init__(self, world, h, probe, srcs, pairs, paths=None):
        self.world, self.h, self.probe, self.srcs, self.pairs = world, h, probe, srcs, pairs
        self.paths = paths or {}   # (i, j) -> the sharp-edge path between them (pairs found on source edges)
        self.detect_seconds = 0.0
        self._seams, self._sides = {}, {}
        self.log = []   # one row per split tried: engine, seam counts, result
        self.layout = dict(left=fill_surface.PART_SECONDS)   # work left for part layouts on the source
        self.plane_only = False   # full_steps compares the splits: parts that fill on the plane stay there
        self.trial = None   # the current comparison try's parts (set by _split_steps for the 'parts' engine)

    def seam(self, i, j, k):
        """k - 1 points on the crease from vertex i to vertex j: along the pair's sharp-edge path at equal lengths, or
        equal steps along the chord, each moved sideways on the source to where the normal turns half way (bisection);
        raises _Miss when a point finds no turn."""
        key = (i, j, k)
        if key not in self._seams:
            path = self.paths.get((i, j))
            if path is not None:
                self._seams[key] = np.array([self.probe.point(p)[0] for p in _resample(path, k)[1:-1]],
                                            float).reshape(-1, 3)
            else:
                self._seams[key] = _trace(self.world, self.h, self.probe, i, j, k)
        return self._seams[key]

    def sides(self, i, j):
        """the source normals on the two sides of the pair: the side of the outline arc from i to j first."""
        if (i, j) not in self._sides:
            W, n, pr = self.world, len(self.world), self.probe
            a = pr.point(W[i] + NEAR * (W[(i + 1) % n] - W[i]))[1] + pr.point(W[j] + NEAR * (W[(j - 1) % n] - W[j]))[1]
            b = pr.point(W[i] + NEAR * (W[(i - 1) % n] - W[i]))[1] + pr.point(W[j] + NEAR * (W[(j + 1) % n] - W[j]))[1]
            self._sides[(i, j)] = (_unit(a), _unit(b))
        return self._sides[(i, j)]


def _resample(poly, k):
    """k + 1 points at equal lengths along a polyline (its ends included)."""
    poly = np.asarray(poly, float)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    at = np.concatenate(([0.0], np.cumsum(seg)))
    t = np.linspace(0.0, at[-1], k + 1)
    return np.column_stack([np.interp(t, at, poly[:, c]) for c in range(3)])


def _trace(W, h, probe, i, j, k):
    a, b = W[i], W[j]
    n = len(W)
    ch = _unit(b - a)
    ends = [a + SEAM_ALONG * (W[(i - 1) % n] - a), a + SEAM_ALONG * (W[(i + 1) % n] - a),
            b + SEAM_ALONG * (W[(j - 1) % n] - b), b + SEAM_ALONG * (W[(j + 1) % n] - b)]
    off = _unit(_cross(probe.normals(ends).sum(0), ch))
    out = []
    for p in a + np.outer(np.arange(1, k) / k, b - a):
        q = probe.point(p)[0]
        lo, hi = -h, h
        nlo, nhi = probe.point(q + lo * off)[1], probe.point(q + hi * off)[1]
        if _ang(nlo, nhi) < 5:
            raise _Miss   # no turn within h of the chord: the crease bends away from it
        for _ in range(SEAM_ITERS):
            mid = (lo + hi) / 2
            nm = probe.point(q + mid * off)[1]
            if _ang(nm, nlo) < _ang(nm, nhi):
                lo = mid
            else:
                hi = mid
        out.append(probe.point(q + (lo + hi) / 2 * off)[0])
    return np.array(out, float).reshape(-1, 3)


# ---------------------------------------------------------------------------------------------- detection

def _median_edge(W):
    return float(np.median(np.linalg.norm(np.roll(W, -1, 0) - W, axis=1)))


def detect(world, srcs):
    """detect_steps run to its end (no time limit)."""
    job = detect_steps(world, srcs)
    try:
        while True:
            next(job)
    except StopIteration as done:
        return done.value
    finally:
        job.close()


def detect_steps(world, srcs):
    """The crease pairs of an even outline on the projection sources, cooperatively: returns a Crease, or None
    (nothing found, or the outline is odd / too short). Pairs are (i, j) outline indices, strongest first, no vertex
    used twice."""
    if not ENABLED or not srcs:
        return None
    started = time.perf_counter()
    W = np.asarray(world, float)
    n = len(W)
    if n < 6 or n % 2 or W.shape != (n, 3) or not np.isfinite(W).all():
        return None
    h = _median_edge(W)
    if not h > 0:
        return None
    probe = Probe(srcs)
    if not probe.data:
        return None
    found = yield from _normal_pairs(W, h, probe)
    used = {v for i, j, _s in found for v in (i, j)}
    paths = {}
    for i, j, s, path in (yield from _edge_pairs(W, h, srcs)):
        if i not in used and j not in used:
            found.append((i, j, s))
            paths[(i, j)] = path
            used |= {i, j}
    if not found:
        return None
    crease = Crease(W, h, probe, srcs, [(i, j) for i, j, _s in found], paths)
    crease.detect_seconds = time.perf_counter() - started
    return crease


def _clear_of_rim(W, h, poly, i, j):
    """the polyline from vertex i to vertex j keeps EDGE_CLEAR x h from the outline (sampled every h / 4): the first
    and last 0.5 h are not tested, and the two outline edges at i (at j) only from 0.75 h away from i (j) on."""
    n = len(W)
    poly = np.asarray(poly, float)
    pts = []
    for a, b in zip(poly[:-1], poly[1:]):
        m = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / (h / 4))))
        pts.append(a + np.outer(np.arange(m) / m, b - a))
    pts.append(poly[-1:])
    P = np.vstack(pts)
    di, dj = np.linalg.norm(P - W[i], axis=1), np.linalg.norm(P - W[j], axis=1)
    keep = (di > .5 * h) & (dj > .5 * h)
    P, di, dj = P[keep], di[keep], dj[keep]
    if not len(P):
        return True
    A, B = W, np.roll(W, -1, 0)
    e = B - A
    t = np.clip(np.einsum('pqc,qc->pq', P[:, None, :] - A[None], e) / np.maximum(np.einsum('qc,qc->q', e, e), 1e-30),
                0, 1)
    d = np.linalg.norm(A[None] + t[:, :, None] * e[None] - P[:, None, :], axis=2)
    for q in ((i - 1) % n, i):
        d[di <= .75 * h, q] = np.inf
    for q in ((j - 1) % n, j):
        d[dj <= .75 * h, q] = np.inf
    return bool(d.min() >= EDGE_CLEAR * h)


def _normal_pairs(W, h, probe):
    """pairs from source face normals: at both vertices the normal turns sharply across the vertex with flat sides,
    the crease direction follows the chord, the turn continues at every point of the traced seam, and the seam keeps
    clear of the rest of the outline."""
    n = len(W)
    prev, nxt = np.roll(W, 1, 0) - W, np.roll(W, -1, 0) - W
    r = SMOOTH_R * h

    def sharp(na, nb, ma, mb):
        t = _ang(na, ma)
        return t if THETA < t < MAX_TURN and max(_ang(na, nb), _ang(ma, mb)) < FLAT * t else 0.0
    side = {}
    for i in range(n):
        yield None
        try:
            if not THETA / 2 < _ang(probe.point(W[i] + AT * prev[i])[1], probe.point(W[i] + AT * nxt[i])[1]) < MAX_TURN:
                continue
            la, lb, ra, rb = probe.smooth([W[i] + NEAR * prev[i], W[i] + FAR * prev[i],
                                           W[i] + NEAR * nxt[i], W[i] + FAR * nxt[i]], r)
        except _Miss:
            continue
        if sharp(la, lb, ra, rb) > 0:
            side[i] = (la, ra)
    cand = sorted(side)
    cdir = {i: _unit(_cross(*side[i])) for i in cand}
    pairs = []
    for ii, i in enumerate(cand):
        for j in cand[ii + 1:]:
            if min((i - j) % n, (j - i) % n) < 2:
                continue
            ch = W[j] - W[i]
            length = float(np.linalg.norm(ch))
            if length < 1e-12:
                continue
            ch = ch / length
            if (min(_ang(cdir[i], ch), _ang(cdir[i], -ch)) > DIR_TOL
                    or min(_ang(cdir[j], ch), _ang(cdir[j], -ch)) > DIR_TOL):
                continue
            yield None
            try:
                strength, poly = _seam_turns(W, h, probe, i, j, max(2, int(round(length / h))), side, sharp)
                if strength is not None and (max(_off_crease(W, h, probe, i, j), _off_crease(W, h, probe, j, i))
                                             > VERTEX_ON * h or not _clear_of_rim(W, h, poly, i, j)):
                    strength = None
            except _Miss:
                strength = None
            if strength is not None:
                pairs.append((i, j, strength))
    return _strongest(pairs)


def _seam_turns(W, h, probe, i, j, k, side, sharp):
    """(the mean turn along the traced seam, the seam with its ends), or (None, None) when the crease does not
    continue sharply at every point."""
    poly = np.vstack(([W[i]], _trace(W, h, probe, i, j, k), [W[j]]))
    avg = side[i][0] + side[i][1] + side[j][0] + side[j][1]
    turns = []
    for q in range(1, len(poly) - 1):
        off = _unit(_cross(avg, _unit(poly[q + 1] - poly[q - 1])))
        m = poly[q]
        x = probe.smooth([m + NEAR * h * off, m + FAR * h * off, m - NEAR * h * off, m - FAR * h * off], SMOOTH_R * h)
        t = _ang(x[0], x[2])
        if t <= THETA or not sharp(x[0], x[1], x[2], x[3]) > 0:
            return None, None
        turns.append(t)
    return (float(np.mean(turns)), poly) if turns else (None, None)


def _off_crease(W, h, probe, i, j):
    """how far vertex i is from the crease on the source: from its surface point, the point across the chord to j
    where the normal turns half way (bisection within h each way); inf when no turn is found there."""
    n = len(W)
    ends = [W[i] + SEAM_ALONG * (W[(i - 1) % n] - W[i]), W[i] + SEAM_ALONG * (W[(i + 1) % n] - W[i])]
    off = _unit(_cross(probe.normals(ends).sum(0), _unit(W[j] - W[i])))
    q = probe.point(W[i])[0]
    lo, hi = -h, h
    nlo, nhi = probe.point(q + lo * off)[1], probe.point(q + hi * off)[1]
    if _ang(nlo, nhi) < 5:
        return math.inf
    for _ in range(SEAM_ITERS):
        mid = (lo + hi) / 2
        nm = probe.point(q + mid * off)[1]
        if _ang(nm, nlo) < _ang(nm, nhi):
            lo = mid
        else:
            hi = mid
    return float(np.linalg.norm(probe.point(q + (lo + hi) / 2 * off)[0] - q))


def _strongest(pairs):
    pairs = sorted(pairs, key=lambda x: -x[2])
    used, out = set(), []
    for i, j, s in pairs:
        if i in used or j in used:
            continue
        used |= {i, j}
        out.append((i, j, s))
    return out


def _sharp_edges(srcs, lo, hi):
    """(points (m, 3), edges (e, 2) into points, dihedral degrees) of the sharp source edges that pass through the
    box lo..hi, world space; mesh sources up to EDGE_FACES_MAX faces. The dihedral follows each face pair's winding:
    two faces that run their shared edge the same way (normals not recalculated) are compared with one normal turned."""
    pts, edges, angles = [], [], []
    base = 0
    for obj in srcs:
        try:
            if obj.type != 'MESH':
                continue
            me = obj.data
            nf = len(me.polygons)
            if not nf or nf > EDGE_FACES_MAX:
                continue
            nv, ne, nl = len(me.vertices), len(me.edges), len(me.loops)
            co = np.empty(nv * 3)
            me.vertices.foreach_get('co', co)
            ev = np.empty(ne * 2, dtype=np.int64)
            me.edges.foreach_get('vertices', ev)
            mw = np.array(obj.matrix_world, dtype=np.float64)
        except (ReferenceError, AttributeError, RuntimeError):
            continue
        V = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
        E = ev.reshape(-1, 2)
        P0, D = V[E[:, 0]], V[E[:, 1]] - V[E[:, 0]]
        tmin, tmax = np.zeros(len(E)), np.ones(len(E))
        with np.errstate(divide='ignore', invalid='ignore'):   # slab test: the edge segment meets the box
            for c in range(3):
                flat = np.abs(D[:, c]) < 1e-30
                t1, t2 = (lo[c] - P0[:, c]) / D[:, c], (hi[c] - P0[:, c]) / D[:, c]
                tmin = np.where(flat, np.where((P0[:, c] >= lo[c]) & (P0[:, c] <= hi[c]), tmin, 2.0),
                                np.maximum(tmin, np.minimum(t1, t2)))
                tmax = np.where(flat, tmax, np.minimum(tmax, np.maximum(t1, t2)))
        near = tmin <= tmax
        if not near.any():
            continue
        try:
            fn = np.empty(nf * 3)
            me.polygons.foreach_get('normal', fn)
            tot = np.empty(nf, dtype=np.int64)
            me.polygons.foreach_get('loop_total', tot)
            le = np.empty(nl, dtype=np.int64)
            me.loops.foreach_get('edge_index', le)
            lv = np.empty(nl, dtype=np.int64)
            me.loops.foreach_get('vertex_index', lv)
            nmat = np.linalg.inv(mw[:3, :3]).T
        except (ReferenceError, AttributeError, RuntimeError, np.linalg.LinAlgError):
            continue
        count = np.bincount(le, minlength=ne)
        loops = np.flatnonzero(near[le] & (count[le] == 2))
        loops = loops[np.argsort(le[loops], kind='stable')]   # the two loops of each such edge, side by side
        first, second = loops[0::2], loops[1::2]
        if not len(first):
            continue
        face = np.repeat(np.arange(nf), tot)
        N = fn.reshape(-1, 3) @ nmat.T
        N /= np.maximum(np.linalg.norm(N, axis=1), 1e-30)[:, None]
        e = le[first]
        forward = lv == E[le, 0]
        sign = np.where(forward[first] != forward[second], 1.0, -1.0)
        cos = np.einsum('ij,ij->i', N[face[first]], N[face[second]]) * sign
        dihedral = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        sharp = dihedral >= EDGE_THETA
        if not sharp.any():
            continue
        pts.append(V)
        edges.append(E[e[sharp]] + base)
        angles.append(dihedral[sharp])
        base += len(V)
    if not edges:
        return None
    return np.vstack(pts), np.vstack(edges), np.concatenate(angles)


def _edge_pairs(W, h, srcs):
    """pairs joined by a path of sharp source edges, with that path: both vertices within EDGE_ON x h of a sharp
    edge, the shortest path along sharp edges at most EDGE_LEN x the chord and clear of the rest of the outline."""
    n = len(W)
    lo, hi = W.min(0), W.max(0)
    pad = .45 * float(np.linalg.norm(hi - lo)) + h
    table = _sharp_edges(srcs, lo - pad, hi + pad)
    yield None
    if table is None:
        return []
    V, S, angles = table
    A, B = V[S[:, 0]], V[S[:, 1]]
    adj = {}
    for (a, b), g in zip(S.tolist(), angles.tolist()):
        length = float(np.linalg.norm(V[a] - V[b]))
        adj.setdefault(a, []).append((b, length, g))
        adj.setdefault(b, []).append((a, length, g))
    ab = B - A
    ab2 = np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-30)
    seeds, on_edge = {}, {}
    for v in range(n):
        t = np.clip(np.einsum('ij,ij->i', W[v] - A, ab) / ab2, 0, 1)
        d = np.linalg.norm(A + ab * t[:, None] - W[v], axis=1)
        k = int(np.argmin(d))
        if d[k] <= EDGE_ON * h:
            a, b = int(S[k, 0]), int(S[k, 1])
            length = float(np.sqrt(ab2[k]))
            seeds[v] = {a: float(t[k]) * length, b: (1 - float(t[k])) * length}
            on_edge[v] = (k, float(t[k]) * length, float(d[k]))
    found = []
    cand = sorted(seeds)
    for ii, i in enumerate(cand):
        yield None
        dist, prev, heap = {}, {}, []
        for s0, c0 in seeds[i].items():
            if c0 < dist.get(s0, math.inf):
                dist[s0] = c0
                heapq.heappush(heap, (c0, s0))
        while heap:
            d0, u = heapq.heappop(heap)
            if d0 > dist.get(u, math.inf):
                continue
            for w, length, _g in adj.get(u, ()):
                nd = d0 + length
                if nd < dist.get(w, math.inf):
                    dist[w] = nd
                    prev[w] = u
                    heapq.heappush(heap, (nd, w))
        for j in cand[ii + 1:]:
            if min((i - j) % n, (j - i) % n) < 2:
                continue
            best = min(((dist.get(s1, math.inf) + c1, s1) for s1, c1 in seeds[j].items()), default=(math.inf, None))
            chord = float(np.linalg.norm(W[j] - W[i]))
            (ki, ti, di), (kj, tj, dj) = on_edge[i], on_edge[j]
            if ki == kj and abs(ti - tj) <= best[0]:   # both on one long sharp edge: the stretch between them
                best = (abs(ti - tj), None)
                path = [int(S[ki, 0]), int(S[ki, 1])]
                poly = np.array([W[i], W[j]])
            if best[0] > EDGE_LEN * chord or chord < 2 * h:
                continue
            if best[1] is not None:
                path = [best[1]]
                while path[-1] in prev:
                    path.append(prev[path[-1]])
                poly = np.vstack(([W[i]], V[path[::-1]], [W[j]]))   # from vertex i along the sharp edges to j
            if not _clear_of_rim(W, h, poly, i, j):
                continue
            turns = [g for a, b in zip(path, path[1:]) for w, _l, g in adj[a] if w == b]
            turns += [float(angles[ki]), float(angles[kj])]
            # the vertices nearest the sharp edges and the straightest path first
            found.append((i, j, float(np.mean(turns)), (best[0] + di + dj) / chord, set(path), poly))
    found.sort(key=lambda x: (x[3], -x[2]))
    used, taken, out = set(), set(), []
    for i, j, g, _r, verts, poly in found:
        if i in used or j in used or verts & taken:   # one pair per stretch of crease
            continue
        used |= {i, j}
        taken |= verts
        out.append((i, j, g, poly))
    return out


# ---------------------------------------------------------------------------------------------- the split

class _SideProject:
    """Projection of a part's interior onto the source, kept on the part's own side of its seams: a hit within h of a
    seam whose normal is nearer the other side's normal is taken again from the point moved onto the own side's
    tangent plane, and put on the seam when it is still across."""

    def __init__(self, probe, seams, h):
        self.probe, self.seams, self.h = probe, seams, h
        self.fixes = 0

    @staticmethod
    def _on_seam(poly, p):
        best, bd = None, math.inf
        for a, b in zip(poly[:-1], poly[1:]):
            e = b - a
            t = min(1.0, max(0.0, float((p - a) @ e) / max(float(e @ e), 1e-30)))
            q = a + t * e
            d = float(np.linalg.norm(p - q))
            if d < bd:
                best, bd = q, d
        return best, bd

    def _across(self, q, nq, poly, own, other):
        return self._on_seam(poly, q)[1] < self.h and _ang(nq, own) > _ang(nq, other)

    def steps(self, plan):
        plan = dict(plan, positions=np.array(plan['positions'], float))
        pos = plan['positions']
        for idx in range(plan['boundary'], len(pos)):
            p = pos[idx]
            try:
                q, nq = self.probe.point(p)
            except _Miss:
                continue
            for poly, own, other in self.seams:
                if not self._across(q, nq, poly, own, other):
                    continue
                self.fixes += 1
                s = self._on_seam(poly, p)[0]
                try:
                    q2, n2 = self.probe.point(p - float((p - s) @ own) * own)
                    q = s if self._across(q2, n2, poly, own, other) else q2
                except _Miss:
                    q = s
                break
            pos[idx] = q
            if idx % 16 == 0:
                yield None
        fill._accept_plan(plan)
        return plan


class _Vert:
    """a stand-in for a mesh vertex (index, co) for the panel network."""
    __slots__ = ('index', 'co')

    def __init__(self, index, co):
        self.index, self.co = index, tuple(float(c) for c in co)


def _normals_at(probe):
    """source normals at points (None when a point finds no source)."""
    def at(P):
        try:
            return probe.normals(P)
        except _Miss:
            return None
    return at


def _drive(job):
    """run a planner generator cooperatively; its updates are not passed on."""
    try:
        while True:
            try:
                next(job)
            except StopIteration as done:
                return done.value
            yield None
    finally:
        job.close()


def _part_settings(settings):
    return {k: v for k, v in settings.items() if k != '_face_turns'}


def _part_fast(pts, settings, project):
    """one part: the direct grid, else the main solve; projected."""
    from . import fill_progressive
    plan = None
    try:
        plan = yield from _drive(fill_progressive.deterministic_steps(fill.direct_steps(pts, settings), .00005))
        if plan is not None:
            return (yield from project(plan))
    except fill.FillError:
        pass
    try:
        plan = yield from _drive(fill_progressive.initial_steps(pts, settings))
        return (yield from project(plan))
    except fill.FillError:
        return None


def _part_full(pts, settings, project):
    """one part: the main search's candidate set (direct grid, main solve, network seed, refinement, polished
    solve), projected; the main search's default among them."""
    from . import fill_progressive, fill_layouts
    pf = fill_layouts.Portfolio(ranking='FLOW', quality_floor=settings.get('_quality_floor', .20))
    pf.rank_r1 = True
    frames = []

    def add(plan):
        if plan is None:
            return
        try:
            plan = yield from project(plan)
        except fill.FillError:
            return
        if plan.get('_legacy_layout'):
            pf.add_legacy(plan)
        else:
            pf.add(plan)
    jobs = (lambda: fill_progressive.deterministic_steps(fill.direct_steps(pts, settings), .00005),
            lambda: fill_progressive.initial_steps(pts, dict(settings, _legacy_frames=frames), {}))
    first = None
    for make in jobs:
        try:
            plan = yield from _drive(make())
        except fill.FillError:
            plan = None
        if plan is not None and not plan.get('_direct'):
            first = plan
        yield from add(plan)
    seed = None
    try:
        seed = yield from _drive(fill_progressive.seed_steps(pts, settings))
    except (fill.FillError, ValueError):
        seed = None
    yield from add(seed)
    incumbent = seed if seed is not None else first
    try:
        refined = yield from _drive(fill_progressive.refine_steps(pts, dict(settings, _network_seeded=True), incumbent))
        if refined is not None and refined is not incumbent:
            yield from add(refined)
    except (fill.FillError, ValueError):
        pass
    try:
        yield from add((yield from _drive(fill_progressive.legacy_steps(pts, settings, frames))))
    except fill.FillError:
        pass
    return pf.plan


def _finish(positions, faces, n, seams, groups):
    """the merged fill: faces run the outline in its order; the plan normal is the mean of the parts' own normals
    (`groups`: the face count of each part, in order), so a steep fold leans no part's faces past it; validated."""
    P = np.asarray(positions, float)
    faces = [tuple(int(v) for v in f) for f in faces]
    if any((f[t], f[(t + 1) % len(f)]) == (1, 0) for f in faces for t in range(len(f))):
        faces = [f[::-1] for f in faces]
    normal, start = np.zeros(3), 0
    for count in groups:
        part = np.zeros(3)
        for f in faces[start:start + count]:
            Q = P[list(f)]
            part += np.cross(Q, np.roll(Q, -1, 0)).sum(0)
        normal += _unit(part)
        start += count
    plan = dict(positions=P, quads=faces, boundary=n, order=list(range(n)), virtual=None, collapse=None,
                normal_ccw=_unit(normal), pattern='crease', shift=0, rotation=0, corners=(), anchors=[], layout=None,
                _crease_candidate=True, crease_seams=seams)
    fill.validate_plan(plan)
    return plan


def _merge(A, n, solved, seams):
    """independent part fills over the outline + seam points A: one plan; each part's faces run its own outline."""
    positions, faces, groups = list(A), [], []
    for ids, plan in solved:
        m = plan['boundary']
        remap = {k: ids[o] for k, o in enumerate(plan['order'][:m]) if o is not None}
        base = len(positions)
        for k in range(m, len(plan['positions'])):
            remap[k] = base + k - m
        positions.extend(np.asarray(plan['positions'], float)[m:])
        own = [tuple(remap[x] for x in f) for f in fill.faces_of(plan)]
        a, b = ids[0], ids[1]
        if any((f[t], f[(t + 1) % len(f)]) == (b, a) for f in own for t in range(len(f))):
            own = [f[::-1] for f in own]
        faces.extend(own)
        groups.append(len(own))
    return _finish(positions, faces, n, seams, groups)


def _layout(crease, pairs, ranks):
    """the parts of a split: (points, part loops as indices into points, seams per part, seam counts), or None when
    a pair does not lie on one part or has no seam count of that rank. The seam count k of a pair keeps every part
    even; the counts nearest chord / h come first."""
    W, h = crease.world, crease.h
    n = len(W)
    parts, part_seams, points, ks = [list(range(n))], {0: []}, list(W), []
    for (i, j), rank in zip(pairs, ranks):
        host = next((q for q, loop in enumerate(parts) if i in loop and j in loop), None)
        if host is None:
            return None
        loop = parts[host]
        size = len(loop)
        a, b = loop.index(i), loop.index(j)
        arc1 = [loop[(a + s) % size] for s in range((b - a) % size + 1)]
        arc2 = [loop[(b + s) % size] for s in range((a - b) % size + 1)]
        k0 = float(np.linalg.norm(W[j] - W[i])) / h
        valid = sorted((k for k in range(1, int(k0) + 5) if (k + len(arc1) - 1) % 2 == 0), key=lambda k: (abs(k - k0), k))
        if rank >= len(valid):
            return None
        k = valid[rank]
        seam = crease.seam(i, j, k)
        sid = list(range(len(points), len(points) + k - 1))
        points.extend(seam)
        n_a, n_b = crease.sides(i, j)
        poly = np.vstack(([W[i]], seam, [W[j]]))
        old = part_seams.pop(host)
        parts[host] = arc1 + sid[::-1]
        parts.append(arc2 + sid)
        part_seams[host] = old + [(poly, n_a, n_b)]
        part_seams[len(parts) - 1] = old + [(poly, n_b, n_a)]
        ks.append(k)
    if any(len(loop) < 4 or len(loop) % 2 for loop in parts):
        return None
    return np.asarray(points, float), parts, part_seams, ks


def _split_steps(crease, pairs, ranks, engine, settings):
    """one split fill: engine 'fast' / 'parts' (each part filled on its own) or 'panels' (one panel network, each
    panel on its own plane). None when it does not fill."""
    started = time.perf_counter()
    row = dict(engine=engine, pairs=[list(p) for p in pairs], ranks=list(ranks), ok=False)
    crease.log.append(row)
    try:
        layout = _layout(crease, pairs, ranks)
    except _Miss:
        layout = None
    if layout is None:
        row['why'] = 'layout'
        return None
    A, parts, part_seams, ks = layout
    row['k'] = ks
    n = len(crease.world)
    settings = _part_settings(settings)
    projectors = [_SideProject(crease.probe, part_seams[q], crease.h) for q in range(len(parts))]
    try:
        if engine == 'panels':
            plan = yield from _panels(A, n, parts, projectors, settings, ks)
        else:
            solve = _part_fast if engine == 'fast' else _part_full
            solved, plain, needs = [], [], []
            for q, loop in enumerate(parts):
                part = yield from solve(A[loop], settings, projectors[q].steps)
                plain.append((loop, part))
                if part is not None and crease.plane_only:
                    how = None
                    needs.append(not _near_source(part, crease))
                    row.setdefault('near', []).append(not needs[-1])
                elif part is not None and _near_source(part, crease):
                    how = 'plane'
                else:
                    needs.append(False)   # (during the comparison) laid out here already
                    part, how = yield from fill_surface.part_steps(   # or the part laid out on the source
                        A[loop], crease.srcs, lambda pts, project: solve(pts, settings, project), part,
                        _normals_at(crease.probe), crease.layout)
                row.setdefault('layout', []).append(how)
                if part is None:
                    row['why'] = 'part %d' % q
                    return None
                solved.append((loop, part))
            if crease.plane_only and engine == 'parts':   # what a re-run of this split would reuse
                crease.trial = dict(pairs=pairs, ranks=ranks, A=A, n=n, ks=ks, solved=solved, plain=plain,
                                    needs=needs, hows=list(row['layout']))
            try:
                plan = _merge(A, n, solved, ks)
            except fill.FillError:   # parts laid out on the source that do not join: the plane parts
                if all(p is s for (_l, p), (_m, s) in zip(plain, solved)) or any(p is None for _l, p in plain):
                    raise
                plan = _merge(A, n, plain, ks)
                row['layout'], row['retry'] = ['plane'] * len(parts), 'plane parts'
    except (fill.FillError, ValueError, _Miss) as exc:
        row['why'] = str(exc)[:60]
        return None
    finally:
        row['seconds'] = round(time.perf_counter() - started, 3)
        row['fixes'] = sum(p.fixes for p in projectors)
    from . import fill_flow
    m = fill_flow.metrics(plan)
    row.update(ok=True, poles=int(m['poles']), minQ=round(float(m['minQ']), 3))
    return plan


def _near_source(plan, crease):
    """the plane fill of a part lies within GAP_SKIP x crease.h of the source at every face centre; False when a
    centre has no source."""
    h = crease.h
    P = np.asarray(plan['positions'], float)
    probe = Probe(crease.srcs)
    try:
        for f in fill.faces_of(plan):
            c = P[list(f)].mean(0)
            if not float(np.linalg.norm(probe.point(c)[0] - c)) < GAP_SKIP * h:
                return False
    except (_Miss, KeyError, TypeError, ValueError, IndexError):
        return False
    return True


def _panels(A, n, parts, projectors, settings, ks):
    from . import fill_multi
    verts = [_Vert(i, p) for i, p in enumerate(A)]
    loops = [[verts[q] for q in loop] for loop in parts]
    order = iter(projectors)

    def project(plan):   # the panel network projects its panels in the order of `loops`
        return (yield from next(order).steps(plan))
    plan = yield from fill_multi.plan_steps(np.eye(4), loops, settings, project, own_planes=True)
    groups = [len(fill.faces_of(p)) for p in plan['panel_plans']]   # the network lists its faces panel by panel
    return _finish(plan['positions'], fill.faces_of(plan), n, ks, groups)


def _rank(plan):
    from . import fill_flow
    m = fill_flow.metrics(plan)
    return (int(m['poles']), -round(float(m['minQ']), 3))


def _pair_sets(crease):
    """all pairs, then the strongest alone (when the pairs together do not split)."""
    sets = [list(crease.pairs)]
    if len(crease.pairs) > 1:
        sets.append(list(crease.pairs[:1]))
    return sets


def fast_steps(crease, settings):
    """the quick split fill: the nearest seam count (else the next), each part the direct grid or the main solve."""
    for pairs in _pair_sets(crease):
        for rank in (0, 1):
            plan = yield from _split_steps(crease, pairs, [rank] * len(pairs), 'fast', settings)
            if plan is not None:
                return plan
    return None


def _timed(job, budget):
    """run a generator cooperatively while its own work time stays within budget['left'] seconds (checked at every
    step); (finished, value): False when the budget ran out first (the generator is closed)."""
    try:
        while True:
            if budget['left'] <= 0:
                return False, None
            t, w = time.perf_counter(), fill_surface.work_seconds()   # part layouts on the source are not counted
            try:
                next(job)
            except StopIteration as done:
                budget['left'] -= time.perf_counter() - t - (fill_surface.work_seconds() - w)
                return True, done.value
            budget['left'] -= time.perf_counter() - t - (fill_surface.work_seconds() - w)
            yield None
    finally:
        job.close()


def full_steps(crease, settings, rank=None):
    """the wider search: 3 seam counts (2 per pair with more than one pair), nearest chord / h first, x both engines;
    the first fill by `rank` (a key function; default: the fewest interior poles, then the highest minQ), eligible
    fills only. Stops early once a seam count gives a fill that ranks first with no interior pole, and with its best
    so far once FULL_SECONDS of its own work are used (checked at every step). Every try has its parts on the plane
    (a part that does not fill on the plane is laid out on the source as before); the split of the best fill is
    then tried once more with its parts laid out on the source where they win (fill_surface.part_steps, the
    crease search's PART_SECONDS), and that fill is taken when it ranks first or equal. That second run is made
    only when a part of the split would be laid out (its plane fill not near the source), with its own budget of
    fill_surface.PART_SECONDS: the comparison uses at most FULL_SECONDS of its own work and the second run at most
    PART_SECONDS (each plus one step), 8 + 3 s; part layouts on the source are bounded by the crease search's
    PART_SECONDS. The second run reuses the comparison's parts of that split and lays out only the parts that
    are not near the source (_relayout_steps); no plane part is solved again."""
    from . import fill_flow
    floor = settings.get('_quality_floor', .20)
    rank = rank or _rank
    crease.plane_only = True   # the splits are compared with plane parts
    try:
        best, trial = yield from _plane_search(crease, settings, rank, floor)
    finally:
        crease.plane_only, crease.trial = False, None
    if trial is not None and any(trial['needs']) and crease.layout['left'] > 0:
        finished, plan = yield from _timed(_relayout_steps(crease, trial, settings),
                                           dict(left=fill_surface.PART_SECONDS))
        if finished and plan is not None and fill_flow.eligible(plan, floor) and rank(plan) <= rank(best):
            best = plan
    return best


def _plane_search(crease, settings, rank, floor):
    """full_steps' loop over the splits: (best fill, record) or (None, None). The record is that of the best's
    split's 'parts' try when that try solved every part (also when its join failed, and also when the best came
    from the split's 'panels' try); None otherwise."""
    from . import fill_flow
    best, kept, budget = None, None, dict(left=FULL_SECONDS)
    for pairs in _pair_sets(crease):
        counts = 3 if len(pairs) == 1 else 2
        for ranks in itertools.product(range(counts), repeat=len(pairs)):
            record = None
            for engine in ('parts', 'panels'):
                crease.trial = None
                finished, plan = yield from _timed(_split_steps(crease, pairs, ranks, engine, settings), budget)
                if engine == 'parts':
                    record = crease.trial   # this split's parts (also for its panels try, which comes next)
                if not finished:
                    return best, kept
                if plan is not None and fill_flow.eligible(plan, floor) and (best is None or rank(plan) < rank(best)):
                    best, kept = plan, record
            key = rank(best) if best is not None else None
            if key is not None and key[-2] == 0 and not any(key[:-2]):   # first on every count before minQ, no pole
                return best, kept
        if best is not None:
            return best, kept
    return best, kept


def _relayout_steps(crease, trial, settings):
    """the picked split once more, from the comparison's record: its parts are reused, fill_surface.part_steps runs
    only for the parts recorded as not near the source, then the parts are joined. None when it does not fill."""
    started = time.perf_counter()
    row = dict(engine='parts', pairs=[list(p) for p in trial['pairs']], ranks=list(trial['ranks']), ok=False,
               k=trial['ks'], reused=True)
    crease.log.append(row)
    A, n, ks = trial['A'], trial['n'], trial['ks']
    settings = _part_settings(settings)
    solved, plain = [], []
    try:
        for q, ((loop, part), (_l, flat), need, how) in enumerate(zip(trial['solved'], trial['plain'],
                                                                     trial['needs'], trial['hows'])):
            if need:
                part, how = yield from fill_surface.part_steps(   # or the part laid out on the source
                    A[loop], crease.srcs, lambda pts, project: _part_full(pts, settings, project), part,
                    _normals_at(crease.probe), crease.layout)
            elif flat is not None:
                how = 'plane'   # near the source: stays on the plane
            row.setdefault('layout', []).append(how)
            if part is None:
                row['why'] = 'part %d' % q
                return None
            solved.append((loop, part))
            plain.append((loop, flat))
        try:
            plan = _merge(A, n, solved, ks)
        except fill.FillError:   # parts laid out on the source that do not join: the plane parts
            if all(p is s for (_l, p), (_m, s) in zip(plain, solved)) or any(p is None for _l, p in plain):
                raise
            plan = _merge(A, n, plain, ks)
            row['layout'], row['retry'] = ['plane'] * len(solved), 'plane parts'
    except (fill.FillError, ValueError, _Miss) as exc:
        row['why'] = str(exc)[:60]
        return None
    finally:
        row['seconds'] = round(time.perf_counter() - started, 3)
    from . import fill_flow
    m = fill_flow.metrics(plan)
    row.update(ok=True, poles=int(m['poles']), minQ=round(float(m['minQ']), 3))
    return plan


def tail_steps(world, srcs, settings, rank=None, base=True):
    """everything for one outline after the main search: detection (at most DETECT_SECONDS of its own work; none
    found: returns None at once), then the quick split fill, yielded as {'candidate': plan}, then the wider search,
    whose fill is returned. Yields {'crease': Crease} once the pairs are known."""
    finished, crease = yield from _timed(detect_steps(world, srcs), dict(left=DETECT_SECONDS))
    if not finished or crease is None:
        return None
    yield {'crease': crease}
    finished, plan = yield from _timed(fast_steps(crease, settings), dict(left=FAST_SECONDS))
    if plan is not None:
        yield {'candidate': plan}
    elif not base:   # no fill at all yet and no quick crease fill: the fallback search should not wait
        return None
    return (yield from full_steps(crease, settings, rank))
