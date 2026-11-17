# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Lattice walk and cell quads for rectilinear outlines (used by the coupled medial lattice, network_medial).

Every boundary vertex turns by roughly a multiple of 90 degrees: +1 (convex corner), 0 (straight) or -1 (reflex
corner). Walking the loop one lattice step per edge in the current direction gives every boundary vertex integer
coordinates; when the walk closes on itself and stays simple, the hole is exactly a set of lattice cells.
"""
from . import boundary_preview

STEPS = ((1, 0), (0, 1), (-1, 0), (0, -1))


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


def lattice_quad_steps(coords):
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
            if (x-min(xs)) % 16 == 0:
                yield None
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


