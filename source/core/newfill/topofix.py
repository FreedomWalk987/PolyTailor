# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Fixed-rim triangle moves for the odd-rim fill (merge_triangle_pairs here, fallback5.merge_pairs_safe): the
triangle move re-splits a triangle and an adjacent quad (their pentagon, 5 ways), which moves the triangle by one
face; two triangles walked next to each other merge into a quad.
"""
import math

import numpy as np

from .cleanup import Mesh, is_edge


def sj(V, f):
    x = V[f]
    a = np.roll(x, -1, 0) - x
    b = np.roll(x, 1, 0) - x
    cr = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    s = cr / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-300)
    if len(f) == 3:
        s = s / math.sin(math.pi / 3)
    return float(s.min())


def candidate_ops(m, R):
    """the triangle moves of the triangles in R (its callers pass one triangle)."""
    ops = []
    for k in R:
        f = m.F.get(k)
        if f is None:
            continue
        if len(f) == 3:
            # triangle move: pentagon with each adjacent quad, re-split
            for i in range(3):
                u, v = f[i], f[(i + 1) % 3]
                ef = [kk for kk in m.edge_faces(u, v) if kk != k]
                if len(ef) != 1 or len(m.F[ef[0]]) != 4:
                    continue
                g = m.F[ef[0]]
                # pentagon CCW: f = u v w ; g contains v u in order v,u,x,y
                w = f[(i + 2) % 3]
                j = g.index(v)
                x, y = g[(j + 2) % 4], g[(j + 3) % 4]
                pent = [u, x, y, v, w] if g[(j + 1) % 4] == u else None
                if pent is None:
                    continue
                for s in range(5):
                    tri = [pent[s], pent[(s + 1) % 5], pent[(s + 2) % 5]]
                    quad = [pent[(s + 2) % 5], pent[(s + 3) % 5], pent[(s + 4) % 5], pent[s]]
                    if set(tri) == set(f):
                        continue
                    ops.append(('tri', (k, ef[0]), (tri, quad)))
    return ops


def apply(m, op):
    kind, ks, data = op   # a triangle move: replace the triangle and the quad by the re-split pair
    removed = []
    added = []
    Vsave = m.V.copy()
    for k in ks:
        removed.append((k, m.F[k]))
        m.remove(k)
    for f in data:
        added.append(m.add(list(f)))
    touched = [k for k in added]
    return removed, added, Vsave, touched


def revert(m, removed, added, Vsave):
    for k in added:
        m.remove(k)
    for k, f in removed:
        m.F[k] = f
        for v in f:
            m.vf[v].add(k)
    m.V = Vsave


def merge_triangle_pairs(V, F, n):
    """Fixed-rim version of Blossom-Quad's extra-edge repair (Remacle 2012 sec. 5.3): while two triangles
    remain, walk one to the other along a shortest dual path by pentagon re-splits, then merge the adjacent pair
    into a quad. Never touches the rim. Returns (V, F, merged_pairs)."""
    m = Mesh(V, F, n)
    merged = 0
    from collections import deque
    from . import emesh5
    for _ in range(len(F)):
        if emesh5.CHECK is not None:
            emesh5.CHECK()
        tris = [k for k, f in m.F.items() if len(f) == 3]
        if len(tris) < 2:
            break
        # BFS from tris[0] to the nearest other triangle
        src = tris[0]
        prev = {src: None}
        dq = deque([src])
        hit = None
        while dq and hit is None:
            k = dq.popleft()
            f = m.F[k]
            for i in range(len(f)):
                for kk in m.edge_faces(f[i], f[(i + 1) % len(f)]):
                    if kk not in prev:
                        prev[kk] = k
                        if len(m.F[kk]) == 3:
                            hit = kk
                            break
                        dq.append(kk)
                if hit is not None:
                    break
        if hit is None:
            break
        path = [hit]
        while path[-1] != src:
            path.append(prev[path[-1]])
        path.reverse()                     # src ... hit
        cur = src
        ok = True
        for i in range(1, len(path) - 1):
            q = path[i]
            nxt = path[i + 1]
            ops = [op for op in candidate_ops(m, {cur}) if op[0] == 'tri' and op[1] == (cur, q)]
            best = None
            for op in ops:
                tri, quad = op[2]
                # the new triangle must share an edge with nxt
                fn = m.F[nxt]
                share = any(is_edge(fn, tri[j], tri[(j + 1) % 3]) for j in range(3) if tri[j] in fn and tri[(j + 1) % 3] in fn)
                if not share:
                    continue
                s = min(sj(m.V, tri), sj(m.V, quad))
                if best is None or s > best[0]:
                    best = (s, op)
            if best is None:
                ok = False
                break
            removed, added, Vsave, touched = apply(m, best[1])
            cur = [k for k in added if len(m.F[k]) == 3][0]
        if not ok:
            break
        # merge cur and hit: they share an edge (u, v)
        f1, f2 = m.F[cur], m.F[hit]
        sh = [(f1[j], f1[(j + 1) % 3]) for j in range(3) if f1[j] in f2 and f1[(j + 1) % 3] in f2 and is_edge(f2, f1[j], f1[(j + 1) % 3])]
        if not sh:
            break
        u, v = sh[0]
        w1 = [x for x in f1 if x not in (u, v)][0]
        w2 = [x for x in f2 if x not in (u, v)][0]
        # f1 has u->v; quad CCW = u, w2, v, w1
        quad = [u, w2, v, w1]
        m.remove(cur)
        m.remove(hit)
        m.add(quad)
        merged += 1
    Fo = [m.F[k] for k in sorted(m.F)]
    return m.V, Fo, merged
