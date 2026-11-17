# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Pole-free lattice fill for rectilinear outlines (Fill AUTO only).

Every boundary vertex turns by roughly a multiple of 90 degrees: +1 (convex
corner), 0 (straight) or -1 (reflex corner). Walking the loop one lattice step per
edge in the current direction gives every boundary vertex integer coordinates.
When the walk closes on itself (opposite side counts balance) and stays simple,
the hole is exactly a set of lattice cells: filling those cells is a quad grid with
no interior poles at all. Rectangles, L, U, T and stepped outlines with matching
counts get the grid a modeller would draw; anything else returns None and the
template / partition planner runs as before.

Only array planning happens here: outer vertices never move, and the result passes
the same whole-disk validator and quality gates as a template fill.
"""
import math

import numpy as np

from . import boundary_preview
from .patterns import geom

GEOMETRIC_MIN_QUALITY = .3   # worst corner sine of a grid read from outline angles
TOLERANCE = math.radians(30)     # how far a turn may be from its multiple of 90 degrees
STEPS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def eligible(settings):
    """without manual corners."""
    return not (settings.get('anchors') or settings.get('_corners') is not None)


def quarter_turns(p):
    """Per vertex -1 / 0 / +1 quarter turns, or None when a turn is ambiguous."""
    turns = geom.turning(p)
    q = np.rint(turns / (math.pi / 2)).astype(int)
    if np.any(np.abs(turns - q * math.pi / 2) > TOLERANCE) or np.any(np.abs(q) > 1):
        return None
    if int(q.sum()) != 4:
        return None
    return q


def lattice_walk(q):
    """Integer coordinates of the boundary vertices, or None when the walk does not
    close or touches itself."""
    n = len(q)
    direction = 0
    point = (0, 0)
    coords = [point]
    for i in range(n):
        if i:
            direction = (direction + q[i]) % 4
        dx, dy = STEPS[direction]
        point = (point[0] + dx, point[1] + dy)
        coords.append(point)
    if coords[-1] != coords[0]:
        return None
    coords = coords[:-1]
    if len(set(coords)) != n:
        return None
    return coords


def _inside(poly, x, y):
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def lattice_quads(coords):
    """Quads of the lattice cells inside the walk (counter-clockwise), and the total
    vertex count; boundary vertices keep their loop indices."""
    area = sum(coords[i - 1][0] * coords[i][1] - coords[i][0] * coords[i - 1][1]
               for i in range(len(coords))) / 2
    if area <= 0:
        return None
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    index = {c: i for i, c in enumerate(coords)}
    quads = []
    for y in range(min(ys), max(ys)):
        for x in range(min(xs), max(xs)):
            if not _inside(coords, x + .5, y + .5):
                continue
            ring = []
            for corner in ((x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)):
                if corner not in index:
                    index[corner] = len(index)
                ring.append(index[corner])
            quads.append(tuple(ring))
    if len(quads) != int(round(area)) or not quads or len(quads) > boundary_preview.MAX_QUADS:
        return None
    return quads, len(index)


def closes(points):
    """Whether this outline's lattice walk closes (it can be filled without poles)."""
    q = quarter_turns(np.asarray(points, float))
    return q is not None and lattice_walk(q) is not None


def odd_split(points, lengths, turns=None):
    """For an odd loop: the edge whose temporary midpoint makes the lattice walk close
    (longest such edge), or None. The midpoint is a straight (0) turn, so it only
    adds one step to the side it lands on."""
    for q in (mesh_turns(turns, len(points)), quarter_turns(np.asarray(points, float))):
        if q is None:
            continue
        for k in np.argsort(-np.asarray(lengths), kind="stable"):
            if lattice_walk(np.insert(q, k + 1, 0)) is not None:
                return int(k)
    return None


def mesh_turns(turns, n):
    """Quarter turns read from the mesh around a hole (faces at each border vertex,
    minus 2), when they are usable: -1 / 0 / +1 summing to one full turn."""
    if turns is None or len(turns) != n:
        return None
    q = np.asarray(turns, dtype=int)
    if np.any(np.abs(q) > 1) or int(q.sum()) != 4:
        return None
    return q


def plan_steps(points, turns=None):
    """Lattice fill result (boundary_preview result format) or None.

    turns: optional quarter turns from the surrounding quads. When their walk closes,
    the fill continues the surrounding grid exactly (every border vertex ends up
    with four edges); otherwise the outline's own angles are used."""
    boundary = np.asarray(points, float)
    n = len(boundary)
    if n < 4 or n % 2:
        return None
    coords = None
    from_mesh = True
    q = mesh_turns(turns, n)
    if q is not None:
        coords = lattice_walk(q)
    if coords is None:
        from_mesh = False
        q = quarter_turns(boundary)
        if q is None:
            return None
        coords = lattice_walk(q)
    if coords is None:
        return None
    built = lattice_quads(coords)
    if built is None:
        return None
    quads, total = built
    yield None
    origin = boundary.mean(axis=0)
    scale = float(np.ptp(boundary, axis=0).max())
    unit = (boundary - origin) / scale
    placed = yield from boundary_preview.place_steps(unit, total, quads)
    if placed is None:
        return None
    positions, metrics, score = placed
    # Angles alone can read a rounded outline as rectilinear; then the grid comes
    # out as slivers. Keep it only when its quads are reasonably square.
    if not from_mesh and geom.quad_quality(positions, quads).min() < GEOMETRIC_MIN_QUALITY:
        return None
    positions = positions * scale + origin
    positions[:n] = boundary
    corners = tuple(int(i) for i in np.flatnonzero(q))
    return dict(positions=positions, quads=tuple(quads), pattern='grid', rotation=0,
                corners=corners, anchors=[], metrics=metrics, score=score)
