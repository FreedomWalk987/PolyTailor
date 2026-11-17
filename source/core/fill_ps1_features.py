# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""PS-1 feature extraction (numpy only; no scipy).

TOPOLOGICAL definitions that also work for non-planar (3D) vertices. On planar outlines the
values equal those of the planar (2D) detectors.

Inputs everywhere:  positions (n,2|3) array-like, faces = list of vertex-index sequences
(quads; triangles tolerated), boundary = derived from edge incidence (edges used by 1 face).

Public:
    mesh_features(positions, faces) -> dict   (all PS-1 terms; see FEATURE_KEYS)
    wpct(values, weights, q)                 weighted "lower" percentile used by wp90 terms
"""
from collections import defaultdict
import math
import numpy as np



# ----------------------------------------------------------------------------- small helpers
def _p3(positions):
    p = np.asarray(positions, float)
    if p.ndim != 2 or p.shape[1] not in (2, 3):
        raise ValueError("positions must be (n,2) or (n,3)")
    if p.shape[1] == 2:
        p = np.c_[p, np.zeros(len(p))]
    return p


def wpct(values, weights, q):
    """Length-weighted LOWER percentile (no interpolation): the smallest value whose
    cumulative weight fraction is >= q/100.  Identical to detectors.wpct."""
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    if len(v) == 0:
        return float("nan")
    o = np.argsort(v, kind="mergesort")
    v, w = v[o], w[o]
    c = np.cumsum(w) / w.sum()
    return float(v[min(int(np.searchsorted(c, q / 100.0)), len(v) - 1)])


def _angle_deg(a, b):
    """Unsigned angle in degrees between two vectors (0..180), atan2 form (no acos round-off
    near 0/180 degrees).  Returns 0 if either vector is exactly zero."""
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(a, b))), float(a @ b)))


# ----------------------------------------------------------------------------- topology
class Topo:
    """Edge/vertex incidence of a polygon mesh given as vertex-index faces."""

    def __init__(self, n_vertices, faces):
        self.faces = [tuple(int(v) for v in f) for f in faces]
        self.n = int(n_vertices)
        self.edge_faces = defaultdict(list)          # sorted (a,b) -> [face ids]
        self.neighbors = defaultdict(set)
        self.corner_pairs = defaultdict(set)         # v -> {sorted(prev,next)} per incident face corner
        self.triangle_verts = set()
        for fi, f in enumerate(self.faces):
            m = len(f)
            if m != 4:
                self.triangle_verts.update(f)
            for i, v in enumerate(f):
                w = f[(i + 1) % m]
                e = (v, w) if v < w else (w, v)
                self.edge_faces[e].append(fi)
                self.neighbors[v].add(w)
                self.neighbors[w].add(v)
                self.corner_pairs[v].add(tuple(sorted((f[i - 1], w))))
        self.boundary_edges = sorted(e for e, fs in self.edge_faces.items() if len(fs) == 1)
        self.interior_edges = sorted(e for e, fs in self.edge_faces.items() if len(fs) == 2)
        self.boundary_verts = {v for e in self.boundary_edges for v in e}
        # poles: interior vertices whose unique-neighbour (edge) valence != 4
        self.poles = sorted(v for v, ns in self.neighbors.items()
                            if v not in self.boundary_verts and len(ns) != 4)
        # opposite[(v,a)] = the neighbour b of v that is NOT co-facial with a (straight-through
        # continuation) -- only at interior, valence-4, all-quad vertices
        self.opposite = {}
        for v, ns in self.neighbors.items():
            if v in self.boundary_verts or v in self.triangle_verts or len(ns) != 4:
                continue
            for a in sorted(ns):
                other = [b for b in ns if b != a and tuple(sorted((a, b))) not in self.corner_pairs[v]]
                if len(other) == 1:
                    self.opposite[(v, a)] = other[0]


# ----------------------------------------------------------------------------- element quality
def corner_quality(p, faces):
    """(minQ, signed_minQ): unsigned / face-normal-projected corner sine, as fill_flow.metrics."""
    minimum, signed = 1.0, 1.0
    for count in sorted({len(f) for f in faces}):
        rings = np.asarray([f for f in faces if len(f) == count], dtype=int)
        x = p[rings]
        a = np.roll(x, -1, axis=1) - x
        b = np.roll(x, 1, axis=1) - x
        cross = np.cross(a, b)
        normals = cross.sum(axis=1)
        normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-30)
        denom = np.maximum(np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2), 1e-30)
        minimum = min(minimum, float((np.linalg.norm(cross, axis=2) / denom).min()))
        projected = np.sum(cross * normals[:, None, :], axis=2)
        signed = min(signed, float((projected / denom).min()))
    return minimum, signed


def face_areas(p, faces):
    """Vector-area magnitude 0.5*||sum x_i x x_{i+1}|| per face (planar = shoelace |area|)."""
    out = np.empty(len(faces))
    for i, f in enumerate(faces):
        x = p[list(f)]
        out[i] = 0.5 * float(np.linalg.norm(np.cross(x, np.roll(x, -1, axis=0)).sum(axis=0)))
    return out


# ----------------------------------------------------------------------------- density terms
def density_terms(p, faces):
    """D = CV(area) (population std / mean)."""
    a = face_areas(p, faces)
    if len(a) < 2 or a.mean() <= 0:
        return {"D": 0.0}
    return {"D": float(a.std() / a.mean())}


# ----------------------------------------------------------------------------- flow terms
def turn_p95(p, topo):
    """T: unweighted np.percentile(95) of the turning angle (deg, 0..180) through each pair of
    opposite neighbours at every interior, valence-4, all-quad vertex (two samples per vertex).
    None when there is no sample (matches fill_flow.metrics 'turn_p95')."""
    turns = []
    for (v, a), b in topo.opposite.items():
        if a < b:
            x, y = p[v] - p[a], p[b] - p[v]
            turns.append(_angle_deg(x, y))
    return float(np.percentile(turns, 95)) if turns else None


def primal_chains(topo):
    """Maximal straight-through edge chains over INTERIOR edges.  Continue only through
    interior valence-4 all-quad vertices (poles, boundary vertices and triangle vertices
    terminate).  Returns [(vertex_path, closed)].  Deterministic: edges visited in sorted
    order, first extended forward from b, then backward from a."""
    used = set()
    chains = []
    for e in topo.interior_edges:
        if e in used:
            continue
        a, b = e
        used.add(e)
        path = [a, b]
        closed = False
        while True:                                   # forward from b
            nxt = topo.opposite.get((path[-1], path[-2]))
            if nxt is None:
                break
            ne = (path[-1], nxt) if path[-1] < nxt else (nxt, path[-1])
            if ne in used:
                closed = True
                break
            used.add(ne)
            path.append(nxt)
        if not closed:
            path.reverse()
            while True:                               # backward = forward from a
                nxt = topo.opposite.get((path[-1], path[-2]))
                if nxt is None:
                    break
                ne = (path[-1], nxt) if path[-1] < nxt else (nxt, path[-1])
                if ne in used:
                    break
                used.add(ne)
                path.append(nxt)
        chains.append((path, closed))
    return chains


def chain_dev_wp90(p, topo):
    """cdev: for every OPEN primal chain with >= 3 vertices and chord > 1e-12:
         dev = max_k dist(x_k, line(x_0, x_end)) / |x_end - x_0|     (lateral drift / chord)
       weight = polyline length; value = weighted lower P90 over chains.  0.0 if none."""
    vals, wts = [], []
    for path, closed in primal_chains(topo):
        if closed or len(path) < 3:
            continue
        pts = p[path]
        chord = pts[-1] - pts[0]
        cl = float(np.linalg.norm(chord))
        if cl < 1e-12:
            continue
        u = chord / cl
        rel = pts - pts[0]
        perp = rel - (rel @ u)[:, None] * u[None, :]
        dev = float(np.linalg.norm(perp, axis=1).max() / cl)
        L = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        vals.append(dev)
        wts.append(L)
    return (wpct(vals, wts, 90), len(vals)) if vals else (0.0, 0)


def dual_strips(topo):
    """Dual strips of the quad faces: sequences of 'rung' edges, each rung crossed to the
    opposite edge of the next quad.  Returns [(rung_edges, closed)].  Partition of ALL edges
    (boundary rungs included) for an all-quad mesh; walking stops at a non-quad face."""
    faces, ef = topo.faces, topo.edge_faces

    def opp(fi, e):
        f = faces[fi]
        if len(f) != 4:
            return None
        for i in range(4):
            a, b = f[i], f[(i + 1) % 4]
            if ((a, b) if a < b else (b, a)) == e:
                c, d = f[(i + 2) % 4], f[(i + 3) % 4]
                return (c, d) if c < d else (d, c)
        return None

    seen = set()
    strips = []
    for e0 in sorted(ef):
        if e0 in seen:
            continue

        def walk(e, fi):
            seq = []
            while True:
                oe = opp(fi, e)
                if oe is None:
                    break
                seq.append(oe)
                seen.add(oe)
                fs = ef[oe]
                if len(fs) != 2:
                    break
                fi2 = fs[0] if fs[1] == fi else fs[1]
                if oe == e0:
                    break
                e, fi = oe, fi2
                if len(seq) > 4 * len(faces) + 4:
                    break
            return seq

        seen.add(e0)
        fs0 = ef[e0]
        a = walk(e0, fs0[0])
        b = walk(e0, fs0[1]) if len(fs0) == 2 else []
        if a and a[-1] == e0:
            strips.append(([e0] + a[:-1], True))
        else:
            strips.append((b[::-1] + [e0] + a, False))
    return strips


def strip_tot_wp90(p, topo):
    """stot: for every OPEN dual strip with >= 3 rungs: centre-line through rung midpoints; value =
       sum of unsigned turning angles (deg) between consecutive centre-line segments; weight =
       centre-line length; weighted lower P90.  0.0 if no strip.
       CLOSED strips (dual loops) are skipped: they encircle poles, i.e. structure already charged by P,
       and their value is otherwise start-rung dependent.
    """
    vals, wts = [], []
    for rungs, closed in dual_strips(topo):
        if len(rungs) < 3 or closed:
            continue
        mid = np.array([0.5 * (p[a] + p[b]) for a, b in rungs])
        seg = np.diff(mid, axis=0)
        tot = sum(_angle_deg(seg[i], seg[i + 1]) for i in range(len(seg) - 1))
        vals.append(tot)
        wts.append(float(np.linalg.norm(seg, axis=1).sum()))
    return (wpct(vals, wts, 90), len(vals)) if vals else (0.0, 0)


# ----------------------------------------------------------------------------- entry point
def mesh_features(positions, faces):
    p = _p3(positions)
    faces = [tuple(int(v) for v in f) for f in faces]
    topo = Topo(len(p), faces)
    minq, sminq = corner_quality(p, faces)
    out = {"faces": len(faces), "poles": len(topo.poles), "minQ": minq, "signed_minQ": sminq}
    out.update(density_terms(p, faces))
    out["T"] = turn_p95(p, topo)
    out["cdev"] = chain_dev_wp90(p, topo)[0]
    out["stot"] = strip_tot_wp90(p, topo)[0]
    return out
