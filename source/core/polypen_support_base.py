# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Build preview drawing: dots, dashed lines and fills through core.draw."""
from contextlib import contextmanager
import bpy
from mathutils import Vector
from .. import prefs, palette
from . import draw as batch_draw

CC_2D_POINTS = "POINTS"
CC_2D_LINES = "LINES"
CC_2D_TRIANGLES = "TRIS"


def _clip_to_viewport(a, b, pad=64.0):
    """Clip a 2D segment to the viewport (Liang-Barsky); None when fully outside.

    Points behind the view are projected far off-screen; without clipping a dashed
    preview line toward one was drawn as ~100,000 dash batches and froze Blender.
    """
    import gpu
    try:
        _x, _y, w, h = gpu.state.viewport_get()
    except Exception:  # noqa: BLE001 - no GPU context: keep the segment as is
        return a, b
    x0, y0, x1, y1 = -pad, -pad, w + pad, h + pad
    dx, dy = b.x - a.x, b.y - a.y
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a.x - x0), (dx, x1 - a.x), (-dy, a.y - y0), (dy, y1 - a.y)):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return None
    return a + (b - a) * t0, a + (b - a) * t1


class Drawing:
    @staticmethod
    def scale(value):
        return value * prefs.ui_scale(bpy.context)

    @staticmethod
    @contextmanager
    def draw(context, primitive):
        batch = DrawBatch(context, primitive)
        yield batch


class DrawBatch:
    def __init__(self, context, primitive):
        self.context, self.primitive = context, primitive
        self.rgba = (1, 1, 1, 0.85)
        self.size, self.width = 8, 2
        self.border_width, self.border_color = 0, (1, 1, 1, 1)
        self.vertices = []
        self.dashes = False

    def point_size(self, size):
        self.size = Drawing.scale(size)
        return self

    def line_width(self, width):
        self.width = Drawing.scale(width)
        return self

    def border(self, *, width, color):
        self.border_width, self.border_color = Drawing.scale(width), tuple(color)
        return self

    def color(self, color):
        values = tuple(color)
        self.rgba = values if len(values) == 4 else (*values, 0.12)
        if self.primitive == CC_2D_TRIANGLES:
            self.rgba = palette.preview_fill(self.context)
        return self

    def stipple(self, **kwargs):
        self.dashes = True
        return self

    def vertex(self, point):
        if point is None:
            self.vertices.clear()
            return self
        point = Vector(point)
        if self.primitive == CC_2D_POINTS:
            if self.rgba[3] > 0:
                batch_draw.disc(point, self.size / 2, self.rgba, segments=16)
            if self.border_width:
                batch_draw.ring(point, self.size / 2, self.border_color, self.border_width, segments=16)
            return self
        self.vertices.append(point)
        count = 2 if self.primitive == CC_2D_LINES else 3
        if len(self.vertices) == count:
            if count == 2:
                a, b = self.vertices
                clipped = _clip_to_viewport(a, b)
                if clipped is None:
                    self.vertices.clear()
                    return self
                a, b = clipped
                if self.dashes and (b-a).length > 0:
                    direction = (b-a).normalized()
                    length = (b-a).length
                    step = Drawing.scale(10)
                    distance = 0.0
                    while distance < length:
                        batch_draw.polyline([a+direction*distance,
                            a+direction*min(length, distance+step/2)], self.rgba, self.width)
                        distance += step
                else:
                    batch_draw.polyline(self.vertices, self.rgba, self.width)
            else:
                batch_draw._draw("TRIS", [tuple(p) for p in self.vertices], self.rgba)
            self.vertices.clear()
        return self
