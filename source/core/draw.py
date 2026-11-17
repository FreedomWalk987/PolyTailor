# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Small 2D drawing helpers built on gpu and blf."""

import math

import blf
import gpu
from gpu_extras.batch import batch_for_shader

def _shader():
    return gpu.shader.from_builtin("UNIFORM_COLOR")


def _line_shader():
    return gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")


def _draw(prim, coords, color, indices=None):
    shader = _shader()
    batch = batch_for_shader(shader, prim, {"pos": coords}, indices=indices)
    shader.uniform_float("color", color)
    batch.draw(shader)


def circle_points(center, radius, segments=48, start=0.0, end=math.tau):
    cx, cy = center
    n = max(2, segments)
    return [
        (cx + math.cos(start + (end - start) * i / n) * radius,
         cy + math.sin(start + (end - start) * i / n) * radius)
        for i in range(n + 1)
    ]


def polyline(coords, color, width=1.0, closed=False):
    if len(coords) < 2:
        return
    coords = list(coords)
    if closed:
        coords.append(coords[0])
    shader = _line_shader()
    batch = batch_for_shader(shader, "LINE_STRIP", {"pos": [(x, y, 0.0) for x, y in coords]})
    shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
    shader.uniform_float("lineWidth", width)
    shader.uniform_float("color", color)
    batch.draw(shader)


def ring(center, radius, color, width=1.0, segments=48):
    polyline(circle_points(center, radius, segments), color, width)


def segments(coords,color,width=1.0):
    """Batched independent screen-space line segments with explicit pixel width."""
    if not coords:
        return
    shader=_line_shader()
    batch=batch_for_shader(shader,"LINES",{"pos":[(x,y,0.0) for x,y in coords]})
    shader.uniform_float("viewportSize",gpu.state.viewport_get()[2:])
    shader.uniform_float("lineWidth",width)
    shader.uniform_float("color",color)
    batch.draw(shader)


def _fan(center, rim, color):
    coords = [tuple(center)] + list(rim)
    indices = [(0, i, i + 1) for i in range(1, len(rim))]
    _draw("TRIS", coords, color, indices)


def disc(center, radius, color, segments=32):
    _fan(center, circle_points(center, radius, segments), color)


def rounded_rect(x0, y0, x1, y1, radius, color, segments=4):
    """a filled rectangle with rounded corners (one triangle fan), e.g. the HUD backdrop."""
    r = max(0., min(radius, (x1-x0)/2, (y1-y0)/2))
    rim = []
    for cx, cy, start in ((x1-r, y0+r, -math.pi/2), (x1-r, y1-r, 0.), (x0+r, y1-r, math.pi/2), (x0+r, y0+r, math.pi)):
        rim.extend(circle_points((cx, cy), r, segments, start, start+math.pi/2))
    rim.append(rim[0])
    _fan(((x0+x1)/2, (y0+y1)/2), rim, color)


def dashed(coords, color, width=1.0, dash=6.0, gap=4.0):
    """Polyline drawn as dashes (screen pixels)."""
    pts = [tuple(c) for c in coords]
    on, left, seg = True, dash, [pts[0]] if pts else []
    for a, b in zip(pts, pts[1:]):
        ax, ay = a
        dx, dy = b[0] - ax, b[1] - ay
        length = math.hypot(dx, dy)
        pos = 0.0
        while length - pos > left:
            pos += left
            p = (ax + dx * pos / length, ay + dy * pos / length)
            if on:
                seg.append(p)
                polyline(seg, color, width)
            seg = [p]
            on = not on
            left = dash if on else gap
        left -= length - pos
        if on:
            seg.append(b)
    if on and len(seg) > 1:
        polyline(seg, color, width)


def text(xy, string, color, size=11, shadow=False, shadow_alpha=.8):
    blf.size(0, size)
    blf.color(0, *color)
    blf.position(0, xy[0], xy[1], 0)
    if shadow:   # blurred shadow like Blender's overlay text; thin CJK strokes stay legible on light meshes
        blf.enable(0, blf.SHADOW)
        blf.shadow(0, 5, 0., 0., 0., shadow_alpha)   # fades with a toast
        blf.shadow_offset(0, 1, -1)
    try:
        blf.draw(0, string)
    finally:
        if shadow:
            blf.disable(0, blf.SHADOW)


_plain_text = text
labels = {}   # kind -> (text, xy, monotonic time) of the last cursor label drawn (toast() places its text there)


def cursor_label(context, xy, text, kind='info', alpha=1.0, size=None):
    """the one at-the-cursor label (Fill choice, Build segments, warnings, tool toast).
    kind 'info': main text colour; 'warning': palette.WARNING. Blurred shadow; size as the Fill label."""
    import time
    from .. import palette, prefs
    if size is None:
        size = max(11, round(13*max(.5, min(3., prefs.ui_scale(context)))))
    color = palette.WARNING if kind == 'warning' else palette.main(context)
    alpha = max(0., min(1., alpha))
    _plain_text(xy, text, (*color[:3], color[3]*alpha), size, shadow=True, shadow_alpha=.8*alpha)
    labels[kind] = (text, tuple(xy), time.monotonic())
    return size


# one fading toast at a time: [text, start, seconds, window xy or None]
_toast = []
_toast_hooks = []   # (show, ended) of the HUD, whose draw handler draws the toast (ui/hud.py)


def toast_owner(show=None, ended=None):
    """the HUD: show() when a toast starts (its draw handler draws it), ended() when the toast has faded."""
    _toast_hooks[:] = (show, ended) if show is not None else ()


def toast_showing():
    import time
    return bool(_toast) and time.monotonic()-_toast[1] < _toast[2]


def _toast_tick():
    """Redraw the 3D views while the toast fades, and once more to erase it."""
    import bpy
    import time
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
    except (AttributeError, ReferenceError):
        pass
    if not _toast or time.monotonic()-_toast[1] >= _toast[2]:
        _toast.clear()
        if _toast_hooks:
            try:
                _toast_hooks[1]()   # the HUD handler goes again if PolyTailor is not in use
            except Exception:
                pass
        return None
    return 1/30


def toast(context, text, seconds=.8, xy=None):
    """Show text at the cursor, fading out over seconds; drawn by the HUD draw handler.
    xy: window position (event.mouse_x / mouse_y); default: the last cursor label's place."""
    import bpy
    import time
    if xy is None and context is not None and context.region is not None and labels:
        last = max(labels.values(), key=lambda item: item[2])
        xy = (last[1][0]+context.region.x, last[1][1]+context.region.y)
    _toast[:] = [text, time.monotonic(), max(.05, float(seconds)), xy]
    if _toast_hooks:
        _toast_hooks[0]()   # the HUD draw handler draws it (also when PolyTailor is not in use)
    if not bpy.app.timers.is_registered(_toast_tick):
        bpy.app.timers.register(_toast_tick, first_interval=0.)


def unregister():
    """the toast's redraw timer does not outlive the add-on."""
    import bpy
    if bpy.app.timers.is_registered(_toast_tick):
        bpy.app.timers.unregister(_toast_tick)
    _toast.clear()


def draw_toast(context):
    """Called by the HUD draw handler in every 3D view WINDOW region: draws the toast in the region under it."""
    import time
    if not _toast or context.region is None:
        return
    string, start, seconds, xy = _toast
    age = time.monotonic()-start
    if age >= seconds:
        return
    region = context.region
    if xy is None:
        local = (region.width/2, region.height/2)
    else:
        local = (xy[0]-region.x, xy[1]-region.y)
        if not (0 <= local[0] < region.width and 0 <= local[1] < region.height):
            return
    from .. import prefs
    scale = max(.5, min(3., prefs.ui_scale(context)))
    with Blend():
        cursor_label(context, (local[0]+18*scale, local[1]+14*scale), string, alpha=1.-age/seconds)


class Blend:
    """Context manager enabling alpha blending for 2D overlays."""

    def __enter__(self):
        self.previous = gpu.state.blend_get()
        gpu.state.blend_set("ALPHA")
        return self

    def __exit__(self, *exc):
        gpu.state.blend_set(self.previous)
        return False
