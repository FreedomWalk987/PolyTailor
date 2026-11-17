# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Grid Relax (Alt in the Relax tool): a force-based relax brush.

Every pass sums a few geometric pushes per vertex, weights them by the brush
falloff, caps the move and lets the caller project the vertex back onto the surface:

- edges toward the patch's average edge length,
- each vertex toward the centre of its neighbours (border vertices: their border
  neighbours; corners stay), harder the further it is off,
- face corners toward an even distance from the face centre,
- face corners toward equal interior angles, by turning the two neighbouring
  vertices around the corner in the face plane.

Small capped steps (a few percent of the local edge length per pass) make the
brush converge smoothly instead of jumping. Everything is computed with array
operations on topology cached per patch. No Blender imports.
"""
import numpy as np

SPEED = 10.0        # overall pace: per second, relative to the brush radius
EDGE = 2.0          # edge-length push
CENTRE = 5.0        # push toward the neighbours' centre
RADIUS = 5.0        # face corners toward an even distance from the face centre
ANGLE = 0.2         # face corners toward equal interior angles
MAX_EDGE = 0.05     # largest move per pass, in mean incident edge lengths
MAX_RADIUS = 0.1    # largest move per pass, in brush radii


def _sum(rows, values, count):
    return np.stack([np.bincount(rows, values[:, k], count) for k in range(3)], axis=1)


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 1e-18)


class Topology:
    """Index arrays of one patch, reused while the brush stays over the same faces.

    border: rows on a real open border (their centre uses border neighbours only)."""

    def __init__(self, cells, count, border=None):
        self.count = count
        border = np.zeros(count, dtype=bool) if border is None else np.asarray(border, dtype=bool)
        edges, uses = {}, {}
        fan, fan_owner = [], []
        groups = {}
        for i, cell in enumerate(cells):
            k = len(cell)
            for a, b in zip(cell, cell[1:] + cell[:1]):
                key = (a, b) if a < b else (b, a)
                uses[key] = uses.get(key, 0) + 1
                edges.setdefault(key, len(edges))
            for j in range(1, k - 1):
                fan.append((cell[0], cell[j], cell[j + 1]))
                fan_owner.append(i)
            groups.setdefault(k, []).append((i, cell))
        self.cells = len(cells)
        self.edges = np.array(list(edges), dtype=np.int64).reshape(-1, 2)
        self.fan = np.array(fan, dtype=np.int64).reshape(-1, 3)
        self.fan_owner = np.array(fan_owner, dtype=np.int64)
        # faces grouped by corner count: (corner count, face ids, (m, k) vertex rows)
        self.faces = [(k, np.array([i for i, _ in g]), np.array([c for _, c in g], dtype=np.int64))
                      for k, g in groups.items()]
        # neighbour pairs for the centre push
        a, b = self.edges[:, 0], self.edges[:, 1]
        outer = np.array([uses[tuple(e)] == 1 for e in self.edges.tolist()], dtype=bool)
        both = np.concatenate([np.column_stack([a, b]), np.column_stack([b, a])])
        along = np.concatenate([outer, outer])
        owner_border = border[both[:, 0]]
        keep = ~owner_border | (owner_border & along & border[both[:, 1]])
        self.pairs = both[keep]
        degree = np.bincount(np.concatenate([a, b]), minlength=count)
        self.corner = border & (degree == 2)      # border corners keep their place

    def face_normals(self, x):
        p = x[self.fan]
        return _sum(self.fan_owner, np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), self.cells)

    def link_length(self, x):
        e = self.edges
        length = np.linalg.norm(x[e[:, 0]] - x[e[:, 1]], axis=1)
        total = np.bincount(e[:, 0], length, self.count) + np.bincount(e[:, 1], length, self.count)
        count = np.bincount(e.ravel(), minlength=self.count)
        return total / np.maximum(count, 1)


def pushes(x, topo, scale):
    """Summed pushes per vertex for one pass; scale carries strength / radius / time."""
    n = topo.count
    total = np.zeros((n, 3))
    e = topo.edges
    if EDGE and len(e):
        vec = x[e[:, 1]] - x[e[:, 0]]
        length = np.linalg.norm(vec, axis=1)
        push = vec * (EDGE * (length.mean() - length) * scale)[:, None]
        total += _sum(e[:, 1], push, n) - _sum(e[:, 0], push, n)
    if CENTRE and len(topo.pairs):
        own, other = topo.pairs[:, 0], topo.pairs[:, 1]
        count = np.bincount(own, minlength=n)
        centre = _sum(own, x[other], n) / np.maximum(count, 1)[:, None]
        off = np.where(count[:, None] > 0, centre - x, 0.0)
        off[topo.corner] = 0.0
        total += off * (np.linalg.norm(off, axis=1) * CENTRE * scale)[:, None]
    normals = _unit(topo.face_normals(x))
    for k, ids, rows in topo.faces:
        p = x[rows]                                         # (m, k, 3)
        if RADIUS:
            rel = p - p.mean(axis=1, keepdims=True)
            r = np.linalg.norm(rel, axis=2)
            push = rel * ((r.mean(axis=1, keepdims=True) - r) * RADIUS * scale)[:, :, None]
            total += _sum(rows.ravel(), push.reshape(-1, 3), n)
        if ANGLE:
            z = normals[ids]
            target = np.pi * (k - 2) / k
            prev, nxt = np.roll(p, 1, axis=1), np.roll(p, -1, axis=1)
            to_prev, to_next = prev - p, nxt - p                 # from each corner
            d_prev, d_next = _unit(to_prev), _unit(to_next)
            # interior angle, signed in the face plane
            sin = np.einsum("mkc,mc->mk", np.cross(d_next, d_prev), z)
            cos = np.einsum("mkc,mkc->mk", d_next, d_prev)
            error = target - np.arctan2(sin, cos)
            reach = (np.linalg.norm(to_prev, axis=2) + np.linalg.norm(to_next, axis=2)) ** 2
            mag = error * ANGLE * scale * reach
            # open the corner: previous vertex turns one way, next vertex the other
            turn_prev = _unit(np.cross(d_prev, z[:, None, :])) * -mag[:, :, None]
            turn_next = _unit(np.cross(d_next, z[:, None, :])) * mag[:, :, None]
            total += _sum(np.roll(rows, 1, axis=1).ravel(), turn_prev.reshape(-1, 3), n)
            total += _sum(np.roll(rows, -1, axis=1).ravel(), turn_next.reshape(-1, 3), n)
    return total


def relax(x, topo, weight, scale, radius, project, passes=2):
    """Move rows of x (world, (n, 3)) with weight > 0 by `passes` capped steps.

    weight: per-row brush falloff (0 = fixed); scale: strength x pressure x time /
    radius for the whole tick; project(points, active) puts active rows back on the
    surface and applies Slide / Mirror / Glue."""
    x = np.asarray(x, dtype=float)
    weight = np.clip(np.asarray(weight, dtype=float), 0.0, 1.0)
    active = weight > 0
    if not active.any() or scale <= 0:
        return x
    reference = topo.face_normals(x)
    for _ in range(passes):
        move = pushes(x, topo, SPEED * scale / passes) * weight[:, None]
        move[~active] = 0.0
        size = np.linalg.norm(move, axis=1)
        biggest = size.max()
        if biggest <= 1e-18:
            break
        move *= min(1.0, MAX_RADIUS * radius / biggest)
        size = np.linalg.norm(move, axis=1)
        limit = MAX_EDGE * topo.link_length(x)
        move *= np.minimum(1.0, np.divide(limit, size, out=np.ones_like(size), where=size > 0))[:, None]
        proposal = project(x + move, active)
        x = _unflip(proposal, x, topo, reference)
    return x


def _unflip(points, fallback, topo, reference):
    """Undo the move of every vertex of a face that turned over."""
    for _ in range(3):
        flipped = (topo.face_normals(points) * reference).sum(axis=1) <= 0
        if not flipped.any():
            return points
        rows = np.unique(topo.fan[np.isin(topo.fan_owner, np.flatnonzero(flipped))])
        points = points.copy()
        points[rows] = fallback[rows]
    return points
