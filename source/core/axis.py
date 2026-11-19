# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""World-axis mouse distances and homogeneous clipping for viewport guides."""
import math

from mathutils import Vector
from mathutils.geometry import intersect_line_plane

from . import raycast


def direction(name):
    return Vector((name == "X", name == "Y", name == "Z"))


def distance(context, start, xy, pivot, name):
    """Same fixed axis-containing plane and end-on fallback used by the curve stroke."""
    axis = direction(name)
    _, view = raycast.ray(context, start)
    normal = view-axis*view.dot(axis)
    if normal.length_squared > 1e-3:
        normal.normalize()
        points = []
        for coord in (start, xy):
            origin, ray = raycast.ray(context, coord)
            if abs(ray.dot(normal)) < 1e-6:
                break
            point = intersect_line_plane(origin, origin+ray, pivot, normal)
            if point is None:
                break
            points.append(point)
        if len(points) == 2:
            return (points[1]-points[0]).dot(axis)
    first = raycast.on_view_plane(context, start, pivot)
    pixel = raycast.on_view_plane(context, start+Vector((0, 1)), pivot)
    return 0.0 if first is None or pixel is None else (xy.y-start.y)*(pixel-first).length


def projected_line(matrix, origin, name, width, height):
    """Clip an infinite world line to the visible frustum, before dividing by w.

    This avoids selecting an arbitrary world length or projecting behind-camera
    endpoints. A line facing the viewer can project to a single marker position.
    """
    if name not in {"X", "Y", "Z"} or width <= 0 or height <= 0:
        return None
    p = matrix @ Vector((*origin, 1.0))
    d = matrix @ Vector((*direction(name), 0.0))
    if not all(math.isfinite(v) for v in (*p, *d)):
        return None
    lower, upper = -math.inf, math.inf
    planes = [(p[3]-1e-8, d[3])]
    planes.extend((p[3]+sign*p[i], d[3]+sign*d[i]) for i in range(3) for sign in (-1, 1))
    for a, b in planes:
        if abs(b) < 1e-30:
            if a < 0:
                return None
        elif b > 0:
            lower = max(lower, -a/b)
        else:
            upper = min(upper, -a/b)
    if lower > upper or not math.isfinite(lower) or not math.isfinite(upper):
        return None
    ends = []
    for t in (lower, upper):
        q = p+d*t
        if q[3] <= 0:
            return None
        ends.append((max(0., min(width, width*.5*(1+q[0]/q[3]))),
                     max(0., min(height, height*.5*(1+q[1]/q[3])))))
    return tuple(ends)
