# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Persistent GPU wheel: settings, sliders and nested menus share one modal owner.

Drawing follows the Blender theme; releasing the key that opened the wheel picks the card under the
cursor, as in Blender's own pie menus.
"""
import math
import time
import traceback
from array import array
from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace

import blf
import bpy
import gpu
from gpu_extras.batch import batch_for_shader

from .. import prefs, palette
from ..core import draw
from ..ops.common import active_tool_id
from . import pie
from .wheel_geometry import arrange
from .custom_layout import (Layout, contains, hit, selected, set_choice, enum_selection_text,
                            icon_name, format_label, value_label, numeric_text, parse_numeric)
from ..i18n import tr as iface_, tr_rpt   # add-on context first

_active = None
_registered = False
_icons = {}
_icon_shader = None
_icon_batch = None
_warmup_handle = None
_warmup_jobs = deque()
_warmup_pending = set()
_opening_draw = False
_opening_alpha = 1.0
_TOOL_ICONS = ("polypen", "fill", "tweak", "relax",
               "pin", "unpin", "glue", "unglue")


def icon_px(scale):
    """Tool icons are rasterized from vector geometry at the exact pixel size they are
    drawn, so the wheel never scales a bitmap."""
    return max(8, round(16 * scale))


def tool_texture(tool, size=16):
    """Full-color family04 artwork at exact pixels, cached separately per theme.
    Rasterizing and uploads stay in the warmup callback."""
    if not tool.startswith("ppo.") or tool[4:] not in _TOOL_ICONS:
        return None
    light = pie.light_theme()
    key = f"{tool}@{size}@{int(light)}"
    if key not in _icons:
        _schedule_asset("tool", (tool[4:], size, light))
    return _icons.get(key)


def _live_area(session):
    """(window, area) of a wheel session, found again among the open windows by pointer; (None, None) when either
    is gone (a stored Window / Area reference is not cleared when Blender frees it)."""
    keys = getattr(session, "area_keys", None)
    wm = getattr(bpy.context, "window_manager", None)
    for window in (wm.windows if wm is not None and keys else ()):
        if window.as_pointer() == keys[0]:
            screen = window.screen
            area = next((a for a in (screen.areas if screen else ()) if a.as_pointer() == keys[1]), None)
            return (window, area) if area is not None else (None, None)
    return None, None


def watch_context():
    """Blender can stop dispatching a modal after a mode/workspace change."""
    active = _active
    if active is None:
        return None
    try:
        window, area = _live_area(active)
        invalid = (not _registered or area is None or active.obj.mode != "EDIT" or active.obj.data != active.mesh or
                   window.view_layer.objects.active != active.obj)
    except (ReferenceError, AttributeError, RuntimeError):   # a removed object
        invalid = True
    if invalid:
        active.restore_drag()
        active.close()
        return None
    return .1


def _read_vco(path):
    """Triangle corners (n, 3, 2) in 0..1, y up, and one RGBA per triangle (n, 4)."""
    import numpy as np
    data = path.read_bytes()
    if data[:4] != b"VCO\0":
        raise ValueError(f"not VCO geometry: {path.name}")
    count = (len(data) - 8) // 18
    xy = np.frombuffer(data, np.uint8, count * 6, 8).reshape(count, 3, 2) / 255.0
    rgba = np.frombuffer(data, np.uint8, count * 12, 8 + count * 6).reshape(count, 3, 4)[:, 0] / 255.0
    return xy, rgba


def _raster_gpu(xy, rgba, n):
    """Premultiplied n x n coverage image (row 0 at the bottom), drawn on the GPU into an
    offscreen buffer: one draw call, about a millisecond for any icon."""
    import numpy as np
    pos = np.zeros((len(xy) * 3, 3), np.float32)
    pos[:, :2] = xy.reshape(-1, 2) * 2 - 1           # 0..1 -> clip space
    colour = np.repeat(np.concatenate((rgba[:, :3] * rgba[:, 3:], rgba[:, 3:]), 1), 3, 0).astype(np.float32)
    offscreen = gpu.types.GPUOffScreen(n, n)
    try:
        with offscreen.bind():
            framebuffer = gpu.state.active_framebuffer_get()
            framebuffer.clear(color=(0.0, 0.0, 0.0, 0.0))
            with gpu.matrix.push_pop(), gpu.matrix.push_pop_projection():
                gpu.matrix.load_identity()
                gpu.matrix.load_projection_matrix(gpu.matrix.get_projection_matrix().Identity(4))
                shader = gpu.shader.from_builtin("SMOOTH_COLOR")
                batch = batch_for_shader(shader, "TRIS", {"pos": pos.tolist(), "color": colour.tolist()})
                previous = gpu.state.blend_get()
                gpu.state.blend_set("ALPHA_PREMULT")   # triangles composited in order
                try:
                    batch.draw(shader)
                finally:
                    gpu.state.blend_set(previous)
            pixels = framebuffer.read_color(0, 0, n, n, 4, 0, "FLOAT")
    finally:
        offscreen.free()
    pixels.dimensions = n * n * 4
    return np.array(pixels.to_list(), np.float64).reshape(n, n, 4)


def _raster_cpu(xy, rgba, n):
    """Same image as _raster_gpu (sample centres, triangles of either winding, in order),
    for a GPU backend without offscreen buffers."""
    import numpy as np
    image = np.zeros((n, n, 4))
    tris = xy * n
    low = np.maximum(np.floor(tris.min(1)), 0).astype(int).tolist()
    high = np.minimum(np.ceil(tris.max(1)), n).astype(int).tolist()
    centres = np.arange(n) + .5
    colours = np.concatenate((rgba[:, :3] * rgba[:, 3:], rgba[:, 3:]), 1)
    for tri, (x0, y0), (x1, y1), colour, a in zip(tris.tolist(), low, high, colours, rgba[:, 3].tolist()):
        if x0 >= x1 or y0 >= y1:
            continue
        px, py = centres[x0:x1], centres[y0:y1, None]
        (ax, ay), (bx, by), (cx, cy) = tri
        if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) < 0:      # either winding, like the GPU
            (bx, by), (cx, cy) = (cx, cy), (bx, by)
        inside = (((bx - ax) * (py - ay) - (by - ay) * (px - ax) >= 0)
                  & ((cx - bx) * (py - by) - (cy - by) * (px - bx) >= 0)
                  & ((ax - cx) * (py - cy) - (ay - cy) * (px - cx) >= 0))
        block = image[y0:y1, x0:x1]
        block[inside] = colour if a >= 1 else colour + block[inside] * (1 - a)
    return image


def _vector_texture(path, size, samples=4, light=False):
    """Rasterize Blender VCO triangle geometry (the toolbar icon format, y up) to a
    size x size texture: `samples`^2 coverage samples per pixel, straight alpha for the
    icon shader. The GPU draws the samples (no frame stall); the CPU path is a fallback."""
    import numpy as np
    xy, rgba = _read_vco(path)
    rgba = palette.icon_colors(rgba, light)
    n = size * samples
    try:
        image = _raster_gpu(xy, rgba, n)
    except Exception:
        image = _raster_cpu(xy, rgba, n)
    image = image.reshape(size, samples, size, samples, 4).mean((1, 3))
    alpha = image[..., 3:]
    image[..., :3] = np.divide(image[..., :3], alpha, out=np.zeros_like(image[..., :3]), where=alpha > 0)
    pixels = image.astype(np.float32).ravel()
    buffer = gpu.types.Buffer("FLOAT", len(pixels), pixels.tolist())
    return gpu.types.GPUTexture((size, size), format="RGBA16F", data=buffer)


def _schedule_asset(kind, key=None):
    job = (kind, key)
    if job not in _warmup_pending:
        _warmup_pending.add(job)
        _warmup_jobs.append(job)
    _schedule_warmup()


def _schedule_warmup():
    global _warmup_handle
    if not _registered or bpy.app.background:
        return
    if _warmup_handle is None:
        _warmup_handle = bpy.types.SpaceView3D.draw_handler_add(_warmup_draw, (), "WINDOW", "POST_PIXEL")
    if not bpy.app.timers.is_registered(_warmup_tick):
        _warmup_idle[:] = [0, 0]
        bpy.app.timers.register(_warmup_tick, first_interval=.02, persistent=True)


_warmed = [False]   # the warm-up runs once, after PolyTailor is first used (not at add-on load)


def warm_up_once():
    if _registered and not _warmed[0]:
        _warmed[0] = True
        request_warmup()


def request_warmup():
    if not _registered or bpy.app.background:
        return
    if _icon_shader is None:
        _schedule_asset("shader")
    # Custom layout fitting may reduce scale on shorter viewports; other sizes
    # are rasterized on first use.
    scale = max(.65, min(1.5, prefs.ui_scale()))
    for name in _TOOL_ICONS:
        light = pie.light_theme()
        if f"ppo.{name}@{icon_px(scale)}@{int(light)}" not in _icons:
            _schedule_asset("tool", (name, icon_px(scale), light))


_warmup_idle = [0, 0]   # jobs left at the last tick, ticks without progress


def _warmup_tick():
    if not _registered or not _warmup_jobs:
        _stop_warmup()
        return None
    # Timers only request redraws. GPU calls run in an actual viewport context.
    _warmup_ready[0] = True
    found = False
    for window in bpy.context.window_manager.windows:
        if window.screen is None:
            continue
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
                found = True
    left = len(_warmup_jobs)
    _warmup_idle[:] = [left, _warmup_idle[1] + 1 if left == _warmup_idle[0] else 0]
    if not found or _warmup_idle[1] >= 25:
        # no 3D View, or the views do not draw: stop; the next entry into PolyTailor (or the wheel's draw) resumes
        _stop_warmup()
        _warmed[0] = False
        _warmup_idle[:] = [0, 0]
        return None
    return .02


def _stop_warmup():
    global _warmup_handle
    if _warmup_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_warmup_handle, "WINDOW")
        _warmup_handle = None


def _warmup_draw():
    # One asset per frame, even with several 3D views drawing in the same cycle.
    # The timer rearms the next step; never run every waiting step on first W.
    if not _registered or not _warmup_jobs or not _warmup_ready[0]:
        return
    _warmup_ready[0] = False
    kind, key = _warmup_jobs.popleft()
    try:
        if kind == "shader":
            _ensure_icon_shader()
            with draw.Blend():
                draw._draw("TRIS", [(-10, -10), (-9, -10), (-9, -9)], (0, 0, 0, 0))
                draw.polyline([(-10, -10), (-9, -9)], (0, 0, 0, 0))
                labels = " ".join(format_label(label) for label in (
                    "Build", "Tweak", "Relax", "Relax / Tweak",
                    "Select Mode", "Snapping", "Constraint Tools", "Stop at Edge", "Snap Targets", "3D", "Topo"))
                draw.text((-10000, -10000), labels, (0, 0, 0, 0), 11)
        elif kind == "tool":
            name, size, light = key
            path = pie.vector_dat(name)
            _icons[f"ppo.{name}@{size}@{int(light)}"] = _vector_texture(path, size, light=light) if path.exists() else None
            if _icons[f"ppo.{name}@{size}@{int(light)}"] is not None:
                paint_icon(_icons[f"ppo.{name}@{size}@{int(light)}"], (-10, -10, -9, -9))
    except Exception:
        # A missing GPU feature / broken optional image must not disable the menu.
        if kind == "tool":
            _icons[f"ppo.{key[0]}@{key[1]}@{int(key[2])}"] = None
        traceback.print_exc()
    finally:
        _warmup_pending.discard((kind, key))


_warmup_ready = [True]


def opening_fraction(started, duration, now):
    if duration <= 0:
        return 1.0
    t = max(0.0, min(1.0, (now - started) / duration))
    return 1.0 - (1.0 - t) ** 3


def opening_rect(rect, origin, amount):
    """Translate a fixed-size rectangle out of the invocation point."""
    cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
    dx, dy = (origin[0] - cx) * (1 - amount), (origin[1] - cy) * (1 - amount)
    return rect[0] + dx, rect[1] + dy, rect[2] + dx, rect[3] + dy


def opening_color(color):
    return (*color[:3], color[3] * _opening_alpha)


@contextmanager
def opening_panel(session, root, amount):
    """Fade paired brush settings from transparent, independently of the cards."""
    global _opening_alpha
    previous = _opening_alpha
    try:
        if root is session.south and session.tool in {"ppo.relax", "ppo.tweak"} and amount < 1:
            _opening_alpha = max(0.0, amount)
        yield
    finally:
        _opening_alpha = previous


@contextmanager
def opening_geometry(session, amount):
    """Translate whole panels/cards for drawing; retain final sizes and hit bounds."""
    global _opening_draw, _opening_alpha
    if amount >= 1:
        yield
        return
    amount = max(0.0, amount)
    roots = list(session.roots)
    if session.popup:
        roots.append(session.popup)
    saved = []
    cards = session.cards
    previous, previous_alpha = _opening_draw, _opening_alpha
    try:
        _opening_draw = True
        _opening_alpha = .25 + .75 * amount
        for root in roots:
            rect = root.rect
            translated = opening_rect(rect, session.center, amount)
            dx, dy = translated[0] - rect[0], translated[1] - rect[1]
            # One translation per panel preserves every child/control's layout.
            for node in {id(n): n for n in [root, *root.walk()]}.values():
                saved.append((node, node.rect))
                x0, y0, x1, y1 = node.rect
                node.rect = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
        session.cards = [(label, tool, opening_rect(rect, session.center, amount))
                         for label, tool, rect in cards]
        yield
    finally:
        for node, rect in saved:
            node.rect = rect
        session.cards = cards
        _opening_draw, _opening_alpha = previous, previous_alpha


def font_size(scale):
    return max(9, round(11 * scale))


def annulus_vertices(center, inner, outer, start=0.0, sweep=math.tau, segments=48):
    x, y = center
    vertices = []
    for i in range(segments):
        a, b = start + sweep * i / segments, start + sweep * (i + 1) / segments
        ia = (x + inner * math.cos(a), y + inner * math.sin(a))
        ib = (x + inner * math.cos(b), y + inner * math.sin(b))
        oa = (x + outer * math.cos(a), y + outer * math.sin(a))
        ob = (x + outer * math.cos(b), y + outer * math.sin(b))
        vertices.extend((ia, oa, ob, ia, ob, ib))
    return vertices


def direction_arc(center, mouse, scale):
    dx, dy = mouse[0] - center[0], mouse[1] - center[1]
    if math.hypot(dx, dy) <= 20 * scale:
        return []
    return annulus_vertices(center, 12 * scale, 20 * scale,
                            math.atan2(dy, dx) - .34, .68, 12)


def paint_center(origin, mouse, style, scale):
    """A stationary ring with a pointer-facing segment, independent of opening."""
    with draw.Blend():
        draw._draw("TRIS", annulus_vertices(origin, 12 * scale, 20 * scale), tuple(style.inner))
        arc = direction_arc(origin, mouse, scale)
        if arc:
            draw._draw("TRIS", arc, palette.accent())
        for radius in (12, 20):
            draw.ring(origin, radius * scale, (*style.outline[:3], 1), 1)
        x, y = origin
        text((x - 65 * scale, y + 23 * scale, x + 65 * scale, y + 43 * scale),
             "PolyTailor", (*style.text[:3], .65), scale, True)


def _ensure_icon_shader():
    global _icon_shader, _icon_batch
    if _icon_shader is None:
        interface = gpu.types.GPUStageInterfaceInfo("pen_case_icon_uv")
        interface.smooth("VEC2", "uv")
        info = gpu.types.GPUShaderCreateInfo()
        info.push_constant("MAT4", "ModelViewProjectionMatrix")
        info.push_constant("VEC4", "rect")
        info.push_constant("VEC4", "tint")
        info.sampler(0, "FLOAT_2D", "image")
        info.vertex_in(0, "VEC2", "pos")
        info.vertex_out(interface)
        info.fragment_out(0, "VEC4", "fragColor")
        info.vertex_source("void main() { uv=pos; vec2 p=mix(rect.xy,rect.zw,pos); gl_Position=ModelViewProjectionMatrix*vec4(p,0.0,1.0); }")
        info.fragment_source("void main() { fragColor=texture(image,uv)*tint; }")
        shader = gpu.shader.create_from_info(info)
        batch = batch_for_shader(shader, "TRI_FAN", {"pos": ((0, 0), (1, 0), (1, 1), (0, 1))})
        _icon_shader, _icon_batch = shader, batch
    return _icon_shader


def paint_icon(texture, rect, tint=(1, 1, 1, 1)):
    # If W is pressed before idle warmup, keep labels and fallback glyphs usable.
    if texture is None or _icon_shader is None:
        return
    shader = _icon_shader
    shader.bind()
    shader.uniform_float("ModelViewProjectionMatrix", gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix())
    shader.uniform_float("rect", rect)
    shader.uniform_sampler("image", texture)
    shader.uniform_float("tint", opening_color(tint))
    previous = gpu.state.blend_get()
    try:
        gpu.state.blend_set("ALPHA")
        _icon_batch.draw(shader)
    finally:
        gpu.state.blend_set(previous)


# ---------------------------------------------------------------- vector tool icons
# The wheel draws tool icons straight from their vector primitives (icons/vector/
# <name>.prim): capsules, rings and convex quads, each pixel covered by its exact
# distance to the shape. Any size and colour, no rasterizing, caching or warmup.
# The VCO texture path above stays as the fallback when the shader is unavailable.
_vector_shader = None
_vector_batch = None
_vector_data = {}
_vector_failed = [False]

_VECTOR_FRAGMENT = """
float sdf(int i, vec2 p)
{
  vec4 h = texelFetch(prims, ivec2(0, i), 0);
  vec4 q = texelFetch(prims, ivec2(1, i), 0);
  vec4 r = texelFetch(prims, ivec2(2, i), 0);
  if (h.x < 0.5) {                       /* capsule: a, b, radius */
    vec2 a = h.zw, ba = q.xy - h.zw, pa = p - a;
    float t = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * t) - q.z;
  }
  if (h.x < 1.5) {                       /* ring: centre, radius, half width */
    return abs(length(p - h.zw) - q.x) - q.y;
  }
  vec2 v[4] = vec2[4](h.zw, q.xy, q.zw, r.xy);   /* convex quad */
  float area = 0.0;
  for (int k = 0; k < 4; k++) {
    vec2 a = v[k], b = v[(k + 1) % 4];
    area += a.x * b.y - a.y * b.x;
  }
  float s = area < 0.0 ? -1.0 : 1.0, d = -1e9;
  for (int k = 0; k < 4; k++) {
    vec2 a = v[k], e = v[(k + 1) % 4] - v[k];
    vec2 n = normalize(vec2(e.y, -e.x)) * s;
    d = max(d, dot(p - a, n));
  }
  return d;
}

vec4 role_colour(int role)
{
  return role == 0 ? params.main_colour : (role == 1 ? params.dim_colour : params.accent_colour);
}

void main()
{
  vec2 p = uv * 256.0;
  float unit = params.misc.x, border = params.misc.y;
  int count = int(params.misc.z + 0.5);
  vec4 colour = vec4(0.0);
  float dmin = 1e9, outline = 1e9;
  int role = -1;
  for (int i = 0; i <= count; i++) {
    int next = -1;
    float d = 1e9;
    if (i < count) {
      next = int(texelFetch(prims, ivec2(0, i), 0).y + 0.5);
      d = sdf(i, p);
    }
    if (next != role && role >= 0) {     /* one run of a colour: union, then paint */
      vec4 c = role_colour(role);
      float a = c.a * clamp(0.5 - dmin / unit, 0.0, 1.0);
      colour = vec4(c.rgb * a, a) + colour * (1.0 - a);
      dmin = 1e9;
    }
    role = next;
    dmin = min(dmin, d);
    outline = min(outline, d);
  }
  if (border > 0.0) {                    /* theme icon border, under the icon */
    float a = border * clamp(1.5 - outline / unit, 0.0, 1.0) * (1.0 - colour.a);
    colour += vec4(0.0, 0.0, 0.0, a);
  }
  fragColor = colour;
}
"""


def _ensure_vector_shader():
    global _vector_shader, _vector_batch
    if _vector_shader is None and not _vector_failed[0]:
        try:
            interface = gpu.types.GPUStageInterfaceInfo("ppo_vector_icon_uv")
            interface.smooth("VEC2", "uv")
            info = gpu.types.GPUShaderCreateInfo()
            # colours and sizes in a uniform buffer: push constants stay within 128 bytes
            info.typedef_source("struct IconParams { vec4 main_colour; vec4 dim_colour; vec4 accent_colour; vec4 misc; };")
            info.uniform_buf(0, "IconParams", "params")
            info.push_constant("MAT4", "ModelViewProjectionMatrix")
            info.push_constant("VEC4", "rect")
            info.sampler(0, "FLOAT_2D", "prims")
            info.vertex_in(0, "VEC2", "pos")
            info.vertex_out(interface)
            info.fragment_out(0, "VEC4", "fragColor")
            info.vertex_source("void main() { uv=pos; vec2 p=mix(rect.xy,rect.zw,pos); gl_Position=ModelViewProjectionMatrix*vec4(p,0.0,1.0); }")
            info.fragment_source(_VECTOR_FRAGMENT)
            shader = gpu.shader.create_from_info(info)
            _vector_batch = batch_for_shader(shader, "TRI_FAN", {"pos": ((0, 0), (1, 0), (1, 1), (0, 1))})
            _vector_shader = shader
        except Exception:
            _vector_failed[0] = True
            traceback.print_exc()
    return _vector_shader


def _vector_texture_data(name):
    """The icon's primitives as a float texture (3 texels per primitive), once."""
    if name not in _vector_data:
        import struct
        try:
            data = (pie.ICONS / "vector" / f"{name}.prim").read_bytes()
            if data[:4] != b"PRIM":
                raise ValueError(f"not icon primitives: {name}")
            count = struct.unpack("<I", data[4:8])[0]
            values = array("f")
            values.frombytes(data[8:8 + count * 48])
            buffer = gpu.types.Buffer("FLOAT", count * 12, values.tolist())
            _vector_data[name] = (gpu.types.GPUTexture((3, count), format="RGBA32F", data=buffer), count)
        except Exception:
            _vector_data[name] = None
            traceback.print_exc()
    return _vector_data[name]


def clear_vector_icons():
    global _vector_shader, _vector_batch
    _vector_data.clear()
    _vector_shader = _vector_batch = None
    _vector_failed[0] = False


def _overlap_insets(area, region):
    """(left, bottom, right, top) pixels of the WINDOW region covered by overlapping header / tool header /
    toolbar / sidebar regions (Region Overlap draws the viewport under them)."""
    l = b = r = t = 0
    try:
        x0, y0, x1, y1 = region.x, region.y, region.x + region.width, region.y + region.height
        for other in area.regions:
            if other == region or other.width <= 1 or other.height <= 1:
                continue
            ox0, oy0, ox1, oy1 = other.x, other.y, other.x + other.width, other.y + other.height
            if ox1 <= x0 or ox0 >= x1 or oy1 <= y0 or oy0 >= y1:
                continue   # does not overlap the viewport
            if other.type in {"HEADER", "TOOL_HEADER"}:
                if oy0 >= (y0 + y1) / 2:
                    t = max(t, y1 - oy0)
                else:
                    b = max(b, oy1 - y0)
            elif other.type in {"TOOLS", "UI"}:
                if ox0 <= (x0 + x1) / 2:
                    l = max(l, ox1 - x0)
                else:
                    r = max(r, x1 - ox0)
    except (AttributeError, ReferenceError):
        pass
    return l, b, r, t


def _desaturate(rgba, amount):
    grey = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
    return tuple(grey + (c - grey) * amount for c in rgba[:3]) + (rgba[3],)


def icon_colours(style, selected=False):
    """Main / dim / accent colours of a tool icon. Custom (default) uses the three
    colours set in the preferences, as they are. Native follows Blender: the widget's
    text colour (selected text when chosen), the inactive alpha for secondary parts,
    the Accent Color with the theme's icon saturation and alpha."""
    settings = prefs.get()
    if getattr(settings, "icon_colours", "CUSTOM") == "CUSTOM":
        acc = tuple(settings.icon_accent)
        if palette.close(acc, palette.DARK):   # default follows 主題色
            acc = (*palette.accent()[:3], acc[3])
        return tuple(settings.icon_main), tuple(settings.icon_dim), acc, 0.0
    ui = bpy.context.preferences.themes[0].user_interface
    text = tuple(style.text_sel if selected else style.text)[:3]
    alpha = float(ui.icon_alpha)
    main = (*text, alpha)
    dim = (*text, .45 * alpha)
    accent = _desaturate(palette.accent(), float(ui.icon_saturation))
    return main, dim, (*accent[:3], alpha), float(ui.icon_border_intensity)


def paint_tool_icon(name, rect, colours, opacity=1.0):
    """Draw a tool icon from its vector primitives; False when that is unavailable
    (the caller falls back to the VCO texture)."""
    shader = _ensure_vector_shader()
    data = _vector_texture_data(name) if shader is not None else None
    if data is None:
        return False
    texture, count = data
    main, dim, accent, border = colours
    fade = opening_color((1, 1, 1, opacity))[3]
    size = max(1.0, rect[2] - rect[0])
    shader.bind()
    shader.uniform_float("ModelViewProjectionMatrix", gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix())
    shader.uniform_float("rect", rect)
    values = [*main[:3], main[3] * fade, *dim[:3], dim[3] * fade, *accent[:3], accent[3] * fade,
              256.0 / size, border * fade, float(count), 0.0]
    block = gpu.types.GPUUniformBuf(gpu.types.Buffer("FLOAT", 16, values))   # alive until drawn
    shader.uniform_block("params", block)
    shader.uniform_sampler("prims", texture)
    previous = gpu.state.blend_get()
    try:
        gpu.state.blend_set("ALPHA_PREMULT")
        _vector_batch.draw(shader)
    finally:
        gpu.state.blend_set(previous)
    del block
    return True


def rounded(rect, color, outline=None, scale=1, corners=(True,True,True,True)):
    x0,y0,x1,y1 = rect
    radius = min(3*scale, (x1-x0)/2, (y1-y0)/2)
    points = []
    for (x,y,start), curved in zip(((x1,y0,-math.pi/2),(x1,y1,0),
                                   (x0,y1,math.pi/2),(x0,y0,math.pi)),corners):
        r = radius if curved else 0
        cx = x + (-r if x==x1 else r)
        cy = y + (r if y==y0 else -r)
        points.extend((cx+r*math.cos(start+i*math.pi/12),
                       cy+r*math.sin(start+i*math.pi/12)) for i in range(7 if curved else 1))
    center = ((x0+x1)/2, (y0+y1)/2)
    coords = [p for i,a in enumerate(points) for p in (center,a,points[(i+1)%len(points)])]
    draw._draw("TRIS", coords, opening_color(color))
    if outline:
        draw.polyline(points, opening_color(outline), scale, closed=True)


def text(rect, value, color, scale, centered=False, right=False):
    size = font_size(scale)
    blf.size(0, size)
    width = rect[2]-rect[0]-(2 if centered else 10)*scale
    label = str(value)
    if blf.dimensions(0,label)[0] > max(0,width):
        original = label
        low,high = 0,len(original)
        while low<high:
            middle = (low+high+1)//2
            if blf.dimensions(0,original[:middle]+"…")[0]<=width:
                low=middle
            else:
                high=middle-1
        label = original[:low]+"…" if blf.dimensions(0,"…")[0]<=width else ""
    length = blf.dimensions(0, label)[0]
    x = rect[2]-length-5*scale if right else (rect[0]+rect[2]-length)/2 if centered else rect[0]+5*scale
    previous = gpu.state.blend_get()
    draw.text((x,(rect[1]+rect[3])/2-4*scale), label, opening_color(color), size)
    gpu.state.blend_set(previous)
    return label


def symbol(name,rect,color,scale):
    """Native-style monochrome marks for controls without extension PNG icons."""
    try:   # the GPU-drawn marks of ui/glyphs.py first; the marks below are the fallback
        from . import glyphs
        if glyphs.draw(name, rect, opening_color(color), scale):
            return
    except Exception:   # noqa: BLE001  (a mark that fails to draw falls back to the marks below)
        pass
    x,y = (rect[0]+rect[2])/2,(rect[1]+rect[3])/2
    point = lambda a,b: (x+a*scale,y+b*scale)
    if name == "CHECKMARK":
        draw.polyline((point(-5,0),point(-1,-4),point(6,5)),opening_color(color),1.5*scale)
    elif name in {"CHECKBOX","CHECKBOX_HLT"}:
        draw.polyline([point(a,b) for a,b in ((-5,-5),(5,-5),(5,5),(-5,5))],opening_color(color),scale,closed=True)
        if name.endswith("HLT"):
            symbol("CHECKMARK",rect,color,scale*.7)
    elif name == "TRASH":
        draw.polyline([point(a,b) for a,b in ((-4,3),(-3,-5),(3,-5),(4,3))],opening_color(color),scale)
        draw.segments((point(-6,5),point(6,5),point(-2,7),point(2,7)),opening_color(color),scale)
    elif name.startswith("SNAP"):
        draw.polyline([point(a,b) for a,b in ((-5,5),(-5,-2),(-3,-5),(3,-5),(5,-2),(5,5))],opening_color(color),1.8*scale)
    elif name == "LOOP_BACK":
        draw.polyline(draw.circle_points((x,y),5*scale,segments=14,start=-2,end=2.6),opening_color(color),scale)
        draw.polyline((point(-6,2),point(-5,6),point(-1,5)),opening_color(color),scale)
    elif name == "AUTOMERGE":
        for a,b in ((-5,-4),(5,-4),(0,5)):
            draw.ring(point(a,b),1.5*scale,opening_color(color),scale,segments=8)
        draw.segments((point(-3,-4),point(3,-4),point(-4,-2),point(-1,3),point(4,-2),point(1,3)),opening_color(color),scale)
    elif name == "GREASEPENCIL":
        draw.polyline([point(a,b) for a,b in ((-5,-5),(-3,0),(4,7),(7,4),(0,-3),(-5,-5))],opening_color(color),scale)
    elif name in {"CURSOR","ONLY","EXCLUDE"}:
        draw.polyline([point(a,b) for a,b in ((-4,6),(-3,-5),(0,-1),(5,-3),(-4,6))],opening_color(color),scale)
        if name=="ONLY":
            symbol("CHECKMARK",(x+2*scale,y-6*scale,x+11*scale,y+3*scale),color,scale*.7)
        elif name=="EXCLUDE":
            draw.segments((point(1,5),point(6,5)),opening_color(color),scale)
    elif name == "MOD_MIRROR":
        draw.segments((point(0,-6),point(0,6)),opening_color(color),scale)
        for side in (-1,1):
            draw.polyline([point(a*side,b) for a,b in ((2,-4),(6,-4),(2,4))],opening_color(color),scale,closed=True)
    elif name in {"RADIOBUT_ON","RADIOBUT_OFF"}:
        draw.ring((x,y),5*scale,opening_color(color),scale,segments=16)
        if name.endswith("ON"):
            draw.disc((x,y),2.5*scale,opening_color(color),segments=12)
    elif name == "MOD_LATTICE":
        for offset in (-5,0,5):
            draw.segments((point(offset,-5),point(offset,5),point(-5,offset),point(5,offset)),opening_color(color),scale)
    else:
        draw.polyline([point(a,b) for a,b in ((-5,-5),(5,-5),(5,5),(-5,5))],opening_color(color),scale,closed=True)
        draw.segments((point(-5,-5),point(5,5)),opening_color(color),scale)


_NOT_KEYS = frozenset({"NONE", "PEN", "ERASER", "TEXTINPUT", "WINDOW_DEACTIVATE"})
_NOT_KEY_PREFIXES = ("TRACKPAD", "NDOF_", "TIMER", "ACTIONZONE_", "XR_", "EVT_")


def key_trigger(event):
    """The keyboard key of the event that opened the wheel (its release picks the card under the cursor), or None
    for anything else: mouse, tablet and NDOF buttons, timers, a menu click."""
    kind = event.type
    if kind in _NOT_KEYS or "MOUSE" in kind or kind.startswith(_NOT_KEY_PREFIXES):
        return None
    return kind


def _widget_key(node):
    """what identifies a wheel widget across a refresh (the tree is rebuilt, the values stay)."""
    return (node.rect, node.kind, getattr(node, "owner", None), getattr(node, "name", None),
            getattr(node, "idname", None), getattr(node, "value", None))


class WheelSession:
    """Plain Python lifetime survives Blender disposing a modal operator's RNA."""

    def invoke(self, context, event, operator):
        global _active
        if event.is_repeat:
            return {"CANCELLED"}
        if _active is not None:
            _active.close()
        _active = self
        self.operator = operator
        self.wm = context.window_manager
        self.window, self.area, self.region = context.window, context.area, context.region
        self.area_keys = (self.window.as_pointer(), self.area.as_pointer())   # for watch_context / close
        self.obj = context.edit_object
        self.mesh = self.obj.data
        self.size = (self.region.width, self.region.height)
        self.mouse = (event.mouse_region_x, event.mouse_region_y)
        self.origin = self.mouse
        self.tool = active_tool_id(context)
        self.popup = self.popup_source = None
        self.popup_scroll = 0
        self.popup_max_scroll = 0
        self.drag = self.edit = None
        self.press = None
        self.last_click = None   # (widget key, popup source) of the last click that acted on a widget
        self.numeric_click = None
        self.status = ""
        self.closed = False
        self.trigger = key_trigger(event)   # the (rebindable) key that opened the wheel; None for a click
        self.held = event.type == self.trigger
        self.started = time.monotonic()
        self.hover_started = self.started
        self.serial = time.monotonic_ns()
        self.refresh(context, initial=True)
        # Keep the center ring consistent with the fitted panel scale.
        self.center_scale = self.scale
        self.selection_threshold = max(34, getattr(context.preferences.view, "pie_menu_threshold", 12)) * self.center_scale
        self.opening_duration = .12
        self.opening_started = None
        self.opening_finished = self.opening_duration == 0
        self.next_refresh = time.monotonic() + .12
        self.handle = bpy.types.SpaceView3D.draw_handler_add(self.paint, (), "WINDOW", "POST_PIXEL")
        self.animation_timer = not self.opening_finished
        self.timer = context.window_manager.event_timer_add(1 / 60 if self.animation_timer else .12, window=self.window)
        if not bpy.app.timers.is_registered(watch_context):
            bpy.app.timers.register(watch_context,first_interval=.1)
        context.window_manager.modal_handler_add(operator)
        self.area.tag_redraw()
        return {"RUNNING_MODAL"}

    def refresh(self, context, initial=False):
        # Never replace a tree while an interaction owns one of its RNA nodes.
        if not initial and (self.drag or self.edit or self.press):
            return
        self.fixed, self.snapping, self.selection, self.south = (Layout() for _ in range(4))
        self.roots = (self.fixed, self.snapping, self.selection, self.south)
        pie.draw_constraints(context, self.fixed)
        pie.draw_snapping(context, self.snapping)
        pie.draw_selection(context, self.selection)
        fill_slot = pie._settings_section(context, self.south, self.tool)
        if fill_slot is not None:
            fill_slot()
        def headings(root, tool):
            if tool in {"ppo.tweak", "ppo.relax"}:
                for node in root.walk():
                    if getattr(node, "heading", False):
                        node.height_units = 22
        headings(self.south, self.tool)
        if initial:
            # Measure every reachable settings variant without mutating scene RNA.
            measured = []
            for tool in ("ppo.polypen", "ppo.fill", "ppo.relax", "ppo.tweak"):
                root = Layout()
                fill_slot = pie._settings_section(context, root, tool, maximum=True)
                if fill_slot is not None:
                    fill_slot()
                headings(root, tool)
                measured.append(root.height(1))
            self.settings_height = max(measured)
            self.preferred_scale = max(.65, min(2., prefs.ui_scale(context)))
        self.south_width_units = 484 if self.tool in {"ppo.tweak", "ppo.relax"} else 264
        self.scale, self.center, placements, cards = arrange(
            [root.height(1) for root in self.roots], self.size, self.origin,
            self.preferred_scale, 484, self.settings_height, _overlap_insets(self.area, self.region))
        # Reserve both brush columns for every tool so switching tools stays anchored.
        cx, cy = self.center
        placements[-1] = (cx-self.south_width_units*self.scale/2,
                          placements[-1][1], self.south_width_units*self.scale)
        self.center_scale = self.scale
        self.width = 248*self.scale
        self.south_width = self.south_width_units*self.scale
        for root, (x, top, width) in zip(self.roots, placements):
            root.place(x, top, width, self.scale)
        self.cards = [card for card in cards if card[1] in pie.AVAILABLE_TOOLS]
        if self.popup_source:
            self.build_popup(context, *self.popup_source)

    def build_popup(self, context, source, anchor):
        if self.popup_source is None or self.popup_source[0] is not source:
            self.popup_scroll = 0
        root = Layout()
        if source.kind == "prop":
            root.label(text=source.text or source.rna.name, text_ctxt="Poly Toolset")
            for item in source.rna.enum_items:
                if item.identifier:
                    root.add("choice",owner=source.owner,name=source.name,value=item.identifier,
                             text=iface_(item.name or item.identifier,
                                         getattr(source.rna, "translation_context", "") or "*"),
                             description=iface_(item.description),
                             flag=source.rna.is_enum_flag,icon=icon_name(item.icon))
        elif source.idname == "VIEW3D_PT_snapping":
            # Reuse the installed Blender version's layout and conditional fields.
            # Only the rendering and lifetime belong to the persistent wheel.
            root.label(text="Snapping", text_ctxt="Poly Toolset")
            cls = bpy.types.VIEW3D_PT_snapping
            cls.draw(SimpleNamespace(layout=root),context)
            native = root.children.pop()
            for child in native.children:
                child.parent = root
                root.children.append(child)
            extra = root.box().column(align=True)
            extra.label(text="PolyTailor", text_ctxt="Poly Toolset")
            pie.draw_snap_extras(context, extra)   # the same rows as the sidebar's Snapping
        else:
            cls = getattr(bpy.types, source.idname)
            root.label(text=cls.bl_label, text_ctxt="Poly Toolset")
            cls.draw(SimpleNamespace(layout=root),context)
        available = self.size[1]-50
        snapping = getattr(source,"idname",None)=="VIEW3D_PT_snapping"
        if snapping and root.height(self.scale)-root.children[0].height(self.scale)-2*self.scale<=available:
            root.children.pop(0)
        self.popup_max_scroll = 0
        if root.height(self.scale)>available:
            title, items = root.children[0],root.children[1:]
            budget = available-title.height(self.scale)-30*self.scale
            # Lists and rows have different heights. Fit whole controls rather
            # than estimating a fixed number of rows and clipping long groups.
            start, used = len(items), 0
            while start>0 and used+items[start-1].height(self.scale)+2*self.scale<=budget:
                start -= 1
                used += items[start].height(self.scale)+2*self.scale
            self.popup_max_scroll = max(0,start)
            self.popup_scroll = min(self.popup_max_scroll,self.popup_scroll)
            visible, used = [], 0
            for item in items[self.popup_scroll:]:
                height = item.height(self.scale)+2*self.scale
                if visible and used+height>budget:
                    break
                visible.append(item)
                used += height
            root.children = [title]+visible
            root.label(text=iface_("Wheel: scroll options"), text_ctxt="Poly Toolset")
        width = (220 if snapping else 295)*self.scale
        height = root.height(self.scale)
        x = anchor[2]+8*self.scale
        if x+width>self.size[0]-12:
            x = anchor[0]-width-8*self.scale
        x = min(self.size[0]-width-12,max(12,x))
        top = max(height+12,min(self.size[1]-12,anchor[3]))
        root.place(x,top,width,self.scale)
        self.popup, self.popup_source = root, (source,anchor)

    def close(self):
        global _active
        if self.closed:
            return
        self.closed = True
        if getattr(self,"handle",None) is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self.handle,"WINDOW")
            self.handle = None
        if getattr(self,"timer",None) is not None:
            try:
                self.wm.event_timer_remove(self.timer)
            except (ReferenceError,RuntimeError):
                pass
            self.timer = None
        if _active is self:
            _active = None
        _window, area = _live_area(self)
        if area is not None:   # the area is still open
            area.tag_redraw()

    def cancel(self, context):
        self.restore_drag()
        self.close()

    def restore_drag(self):
        if self.drag:
            node, original, _, _ = self.drag
            self.drag = None
            try:
                setattr(node.owner,node.name,original)
            except ReferenceError:
                pass

    def _popup_token(self):
        """the open popup's source node (kept across a refresh of the same popup), None without a popup."""
        return self.popup_source[0] if self.popup is not None and self.popup_source else None

    def widget_at(self, xy):
        if self.popup and contains(self.popup.rect,xy):
            return hit(self.popup,xy)
        return next((node for root in self.roots if (node := hit(root,xy)) is not None), None)

    def card_at(self, xy):
        if self.popup and contains(self.popup.rect,xy):
            return None
        if any(contains(root.rect,xy) for root in self.roots):
            return None
        direct = next((c for c in self.cards if contains(c[2],xy)), None)
        if direct is not None:
            return direct
        if math.dist(xy, self.origin) < self.selection_threshold:
            return None
        if math.dist(xy, self.center) < 50*self.scale:
            return None
        dx, dy = xy[0] - self.center[0], xy[1] - self.center[1]
        sector = math.floor(math.atan2(dy, dx) / (math.pi / 4) + .5) % 8
        target = {3:"ppo.polypen", 4:"ppo.fill", 5:"ppo.relax"}.get(sector)
        return next((c for c in self.cards if c[1] == target), None)


    def activate(self, context, node):
        if not node.is_enabled:
            return
        self.status = ""
        if node.kind in {"menu","popover"} or node.kind == "prop" and node.rna.type == "ENUM":
            self.build_popup(context,node,node.rect)
            return
        if node.kind == "choice":
            set_choice(node)
            snapping = self.popup_source and getattr(self.popup_source[0],"idname",None)=="VIEW3D_PT_snapping"
            if not node.flag and not snapping:
                self.popup = self.popup_source = None
        elif node.kind == "prop":
            if node.rna.type == "BOOLEAN":
                setattr(node.owner,node.name,not getattr(node.owner,node.name))
        elif node.kind == "operator":
            namespace,name = node.idname.split(".")
            operation = getattr(getattr(bpy.ops,namespace),name)
            if operation.poll():
                options = dict(vars(node.options))
                if node.idname == "mesh.select_mode":
                    options["use_extend"] = bool(getattr(self, "shift", False))   # Shift+click adds
                result = operation("EXEC_DEFAULT",**options)
                if "CANCELLED" in result:
                    self.status = iface_("Action unavailable for the current selection")
                if "FINISHED" in result:
                    # switch_tool, or Topo chosen in a Blender tool, may change the tool
                    self.tool = active_tool_id(context)
            else:
                self.status = iface_("Action unavailable for the current selection")
            # Nested menus return to the still-open parent wheel.
            self.popup = self.popup_source = None
        if not self.closed:
            self.refresh(context)

    def choose_tool(self, context, card, toast=False):
        self.close()
        bpy.ops.ppo.switch_tool(tool=card[1])   # no hidden Auto Merge action
        show = getattr(draw, "toast", None) if toast else None   # the tool-switch toast
        if show is not None:
            from ..i18n import tr
            name = {"ppo.polypen": "Build", "ppo.fill": "Fill", "ppo.tweak": "Tweak",
                    "ppo.relax": "Relax"}.get(active_tool_id(context))
            if name:
                show(context, tr(name))

    def numeric(self, node, value):
        rna = node.rna
        if not math.isfinite(value):
            raise ValueError("Value must be finite")
        value = max(rna.hard_min,min(rna.hard_max,value))
        setattr(node.owner,node.name,round(value) if rna.type == "INT" else value)

    def modal(self, context, event):
        try:
            return self.handle_event(context,event)
        except Exception:
            traceback.print_exc()
            self.restore_drag()
            self.close()
            self.operator.report({"ERROR"},tr_rpt("PolyTailor Wheel error; see console"))
            return {"CANCELLED"}

    def handle_event(self, context, event):
        if (self.closed or not _registered or context.mode != "EDIT_MESH" or
                context.edit_object != self.obj or context.area != self.area or
                (self.region.width,self.region.height) != self.size):
            self.restore_drag()
            self.close()
            return {"CANCELLED"}
        if event.type == "WINDOW_DEACTIVATE":
            self.restore_drag()
            self.close()
            return {"CANCELLED"}
        if event.type in {"ESC","RIGHTMOUSE"} and event.value == "PRESS":
            if self.edit:
                self.edit = None
            elif self.drag:
                self.restore_drag()
            elif self.popup:
                self.popup = self.popup_source = None
            else:
                self.close()
                return {"CANCELLED"}
            self.area.tag_redraw()
            return {"RUNNING_MODAL"}
        if event.type == "TIMER":
            if self.opening_finished and self.animation_timer:
                self.wm.event_timer_remove(self.timer)
                self.timer = self.wm.event_timer_add(.12, window=self.window)
                self.animation_timer = False
            now = time.monotonic()
            if now >= self.next_refresh:
                if not self.drag and not self.edit and not self.press:
                    self.refresh(context)
                self.next_refresh = now + .12
                self.area.tag_redraw()
            if not self.opening_finished:
                self.area.tag_redraw()
            return {"RUNNING_MODAL"}
        if event.type in {"LEFTMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
            # Real widget interactions finish the transition; unrelated events
            # (modifier keys, timer reports, redraw notifications) must not skip it.
            self.opening_finished = True
        if self.edit:
            node, value, replace = self.edit
            if event.value == "PRESS":
                chars = {"ZERO":"0","ONE":"1","TWO":"2","THREE":"3","FOUR":"4","FIVE":"5",
                         "SIX":"6","SEVEN":"7","EIGHT":"8","NINE":"9","PERIOD":".","MINUS":"-",
                         **{f"NUMPAD_{i}":str(i) for i in range(10)},"NUMPAD_PERIOD":".","NUMPAD_MINUS":"-"}
                if event.type in {"RET","NUMPAD_ENTER"}:
                    try:
                        self.numeric(node,parse_numeric(node,value))
                        self.edit = None
                        self.refresh(context)
                    except ValueError:
                        self.status = iface_("Enter a valid number")
                elif event.type == "A" and event.ctrl:
                    self.edit = (node,value,True)
                elif event.type == "BACK_SPACE":
                    self.edit = (node,"" if replace else value[:-1],False)
                elif event.type in chars and len(value)<24:
                    self.edit = (node,chars[event.type] if replace else value+chars[event.type],False)
            self.area.tag_redraw()
            return {"RUNNING_MODAL"}
        if event.type in {"MOUSEMOVE","INBETWEEN_MOUSEMOVE"}:
            self.mouse = (event.mouse_region_x,event.mouse_region_y)
            self.hover_started = time.monotonic()
            if self.drag:
                node,original,current,previous = self.drag
                step = 1 if node.rna.type == "INT" else .005*max(1,abs(original))
                current += (self.mouse[0]-previous)*step*(.1 if event.shift else 1)
                current = max(node.rna.hard_min,min(node.rna.hard_max,current))
                self.numeric(node,current)
                self.drag = (node,original,current,self.mouse[0])
            self.area.tag_redraw()
            return {"RUNNING_MODAL"}
        if event.type == self.trigger:
            if event.value == "RELEASE":
                card = self.card_at(self.mouse)
                self.held = False
                if card and math.dist(self.mouse,self.origin)>=self.selection_threshold:
                    self.opening_finished = True
                    self.choose_tool(context,card)
                    return {"FINISHED"}
                # a long press without movement always leaves the wheel open
            return {"RUNNING_MODAL"}
        if event.value == "PRESS" and not self.popup:
            # number keys as directions of the cards: 7 NW Build, 4 W Fill, 1 SW brush card.
            # Other digits do nothing (consumed below; the wheel stays open).
            target = {"SEVEN":"ppo.polypen","FOUR":"ppo.fill","ONE":"ppo.relax",
                      "NUMPAD_7":"ppo.polypen","NUMPAD_4":"ppo.fill","NUMPAD_1":"ppo.relax"}.get(event.type)
            if target:
                card = next((c for c in self.cards if c[1]==target),None)
                if card:
                    self.choose_tool(context,card,toast=True)
                    return {"FINISHED"}
        double = event.type == "LEFTMOUSE" and event.value == "DOUBLE_CLICK"
        if double:
            node = self.widget_at(self.mouse)
            if node and node.kind == "prop" and node.rna.type in {"INT","FLOAT"}:
                if node.is_enabled:
                    self.drag = None
                    self.edit = (node,numeric_text(node,getattr(node.owner,node.name)),True)
                    self.area.tag_redraw()
                return {"RUNNING_MODAL"}
            # the widget just clicked, clicked again quickly (a toggle), with the same popup state: the second
            # click is a press like the first. Anything else (another widget, a popup closed or opened in between)
            # is swallowed as a plain double click.
            last, self.last_click = getattr(self, "last_click", None), None
            if (node is None or last is None or last[0] != _widget_key(node)
                    or last[1] is not self._popup_token()):
                return {"RUNNING_MODAL"}
        if event.type == "LEFTMOUSE" and (event.value == "PRESS" or double):
            if self.popup and not contains(self.popup.rect,self.mouse):
                self.popup = self.popup_source = None
            node = self.widget_at(self.mouse)
            self.press = node or self.card_at(self.mouse) or (
                "POPUP" if self.popup and contains(self.popup.rect,self.mouse) else "OUTSIDE")
            if node and node.kind == "prop" and node.rna.type in {"INT","FLOAT"} and node.is_enabled:
                value = getattr(node.owner,node.name)
                self.drag = (node,value,float(value),self.mouse[0])
            return {"RUNNING_MODAL"}
        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            clicked = None   # recorded before the click acts (it may open or close a popup)
            if self.drag:
                node,original,current,_ = self.drag
                previous = self.numeric_click
                if abs(current-original)<1e-8:
                    now = time.monotonic()
                    if previous and previous[0] == node.owner and previous[1] == node.name and now-previous[2]<.5:
                        self.edit = (node,numeric_text(node,getattr(node.owner,node.name)),True)
                        self.numeric_click = None
                    else:
                        self.numeric_click = (node.owner,node.name,now)
                else:
                    self.numeric_click = None
                self.drag = None
                self.refresh(context)
            elif isinstance(self.press,Layout):
                node = self.widget_at(self.mouse)
                if node and node.rect == self.press.rect:
                    clicked = (_widget_key(node), self._popup_token())
                    self.shift = event.shift
                    self.activate(context,node)
            elif isinstance(self.press,tuple):
                card = self.card_at(self.mouse)
                if card == self.press:
                    self.choose_tool(context,card)
            elif self.press == "OUTSIDE" and not any(contains(root.rect,self.mouse) for root in self.roots):
                self.close()
            self.press = None
            self.last_click = clicked
            self.area.tag_redraw()
            return {"FINISHED"} if self.closed else {"RUNNING_MODAL"}
        if event.type in {"WHEELUPMOUSE","WHEELDOWNMOUSE"} and event.value == "PRESS":
            if self.popup and contains(self.popup.rect,self.mouse) and self.popup_max_scroll:
                self.popup_scroll = max(0,min(self.popup_max_scroll,self.popup_scroll+(1 if event.type=="WHEELDOWNMOUSE" else -1)))
                self.build_popup(context,*self.popup_source)
                self.area.tag_redraw()
                return {"RUNNING_MODAL"}
            node = self.widget_at(self.mouse)
            if node and node.kind == "prop" and node.rna.type in {"INT","FLOAT"} and node.is_enabled:
                amount = 1 if event.type == "WHEELUPMOUSE" else -1
                step = 1 if node.rna.type=="INT" else math.radians(1) if node.rna.subtype=="ANGLE" else .01
                self.numeric(node,getattr(node.owner,node.name)+amount*step)
                self.refresh(context)
                self.area.tag_redraw()
            return {"RUNNING_MODAL"}
        return {"RUNNING_MODAL"}

    def paint(self):
        context = bpy.context
        if self.closed or context.area != self.area or context.region != self.region:
            return
        try:
            ui = context.preferences.themes[0].user_interface
            regular = ui.wcol_regular
            foreground = (*regular.text[:3],1)
            outline = (*regular.outline[:3],1)
            accent = palette.accent(context)
            hover = self.widget_at(self.mouse)
            hovered_card = self.card_at(self.mouse)
            hovered_tool = hovered_card[1] if hovered_card else None
            now = time.monotonic()
            if self.opening_started is None:
                self.opening_started = now
            amount = 1.0 if self.opening_finished else opening_fraction(self.opening_started, self.opening_duration, now)
            self.opening_finished = amount >= 1.0
            paint_center(self.center, self.mouse, ui.wcol_pie_menu, getattr(self, "center_scale", self.scale))
            with opening_geometry(self, amount), draw.Blend():
                for root in self.roots:
                    with opening_panel(self, root, amount):
                        self.paint_layout(root,regular,foreground,outline,accent,hover)
                for label,tool,rect in self.cards:
                    label = format_label(label)
                    active = tool == self.tool or tool == "ppo.relax" and self.tool == "ppo.tweak"
                    over = tool == hovered_tool
                    style = ui.wcol_pie_menu
                    color = palette.selection_fill(style.inner,context) if active else palette.selection_fill(style.inner,context,.12) if over else tuple(style.inner)
                    rounded(rect,color,accent if active else tuple(style.outline),self.scale)
                    fg = (*style.text[:3],1)
                    icon_tool = "ppo.tweak" if tool == "ppo.relax" and self.tool == "ppo.tweak" else tool
                    px = icon_px(self.scale)
                    shortcut = {"ppo.polypen":"7","ppo.fill":"4","ppo.relax":"1"}[tool]
                    # whole pixels: the icon's pixel grid meets the screen's
                    x,y = math.floor(rect[0]+8*self.scale+.5),math.floor((rect[1]+rect[3]-px)/2+.5)
                    drawn = icon_tool.startswith("ppo.") and paint_tool_icon(
                        icon_tool[4:],(x,y,x+px,y+px),icon_colours(style,active),fg[3])
                    texture = None if drawn else tool_texture(icon_tool, px)
                    if drawn or texture is not None:
                        if texture is not None:
                            paint_icon(texture,(x,y,x+px,y+px),(1,1,1,fg[3]))
                        text((x+18*self.scale,rect[1],rect[2]-18*self.scale,rect[3]),label,fg,self.scale)
                    else:
                        text((rect[0]+6*self.scale,rect[1],rect[2]-18*self.scale,rect[3]),label,fg,self.scale)
                    text((rect[2]-18*self.scale,rect[1],rect[2]-2*self.scale,rect[3]),shortcut,(*fg[:3],.65),self.scale,True)
                footer = (self.south.rect[0],max(6,self.south.rect[1]-27*self.scale),self.south.rect[2],self.south.rect[1]-6*self.scale)
                if self.status or self.edit:
                    with opening_panel(self, self.south, amount):
                        text(footer,self.status or iface_("Enter: confirm | Esc: cancel"),foreground,self.scale)
                if self.popup:
                    root = self.popup
                    background = tuple(ui.wcol_menu_back.inner)
                    rounded((root.rect[0]-6*self.scale,root.rect[1]-6*self.scale,root.rect[2]+6*self.scale,root.rect[3]+6*self.scale),background,outline,self.scale)
                    self.paint_layout(root,regular,foreground,outline,accent,hover)
                if hover and hover.description and not self.drag and not self.edit and time.monotonic()-self.hover_started>.7:
                    description = hover.description
                    if hover.kind == "operator":
                        namespace,name = hover.idname.split(".")
                        cls = getattr(bpy.types,namespace.upper()+"_OT_"+name,None)
                        callback = getattr(cls,"description",None)
                        if callback:
                            try:
                                description = iface_(callback(context,hover.options))
                            except AttributeError:
                                pass
                    self.paint_tooltip(description,foreground)
        except Exception:
            traceback.print_exc()
            self.restore_drag()
            self.close()

    def paint_layout(self,root,regular,foreground,outline,accent,hover):
        for node in root.walk():
            if node.kind == "box":
                box = bpy.context.preferences.themes[0].user_interface.wcol_box
                rounded(node.rect,tuple(box.inner),tuple(box.outline),self.scale)
                continue
            if node.kind in {"row","column","separator"}:
                continue
            color = foreground if node.is_enabled else (*foreground[:3],.35)
            if node.kind == "label":
                if node.heading:
                    style = bpy.context.preferences.themes[0].user_interface.wcol_pie_menu
                    rounded(node.rect,tuple(style.inner),scale=self.scale)
                    text(node.rect,node.text,color,self.scale)
                else:
                    muted = tuple(.8*c+.2*b for c,b in zip(color[:3],regular.inner[:3]))+(color[3],)
                    rect = node.rect
                    mark = getattr(node,"icon","")
                    if mark:
                        x = rect[0]+3*self.scale
                        symbol(mark,(x,rect[1],x+16*self.scale,rect[3]),muted,self.scale)
                        rect = (rect[0]+20*self.scale,rect[1],rect[2],rect[3])
                    text(rect,node.text,muted,self.scale)
                continue
            chosen = selected(node)
            checkbox = node.kind=="prop" and node.rna.type=="BOOLEAN" and not node.toggle and not node.icon_only
            snapping_control = (root is self.popup and self.popup_source
                                and getattr(self.popup_source[0],"idname",None)=="VIEW3D_PT_snapping")
            native_checkbox = checkbox and snapping_control
            style = bpy.context.preferences.themes[0].user_interface.wcol_radio if node.kind=="choice" else regular
            fill = palette.selection_fill(style.inner) if chosen else tuple(style.inner)
            if not node.is_enabled:
                fill = tuple(.4*c+.6*b for c,b in zip(fill[:3],regular.inner[:3]))+(1,)
            if node is hover and node.is_enabled and not chosen:
                fill = tuple(min(1,c+.08) for c in fill[:3])+(1,)
            brush_checkbox = checkbox and node.name=="include_occluded" and self.tool in {"ppo.tweak","ppo.relax"}
            if not checkbox or brush_checkbox:
                corners = (True,True,True,True)
                parent = node.parent
                if parent and parent.kind=="row" and parent.align and len(parent.children)>1:
                    first,last = node is parent.children[0],node is parent.children[-1]
                    corners = (last,last,first,first)
                elif parent and parent.kind=="column" and parent.align and len(parent.children)>1:
                    first,last = node is parent.children[0],node is parent.children[-1]
                    corners = (last,first,first,last)
                rounded(node.rect,fill,tuple(style.outline),self.scale,corners)
            label = node.text
            right_value = None
            mark = getattr(node,"icon","")
            tool_icon = None
            if node.kind=="operator" and node.icon_value:
                tool_icon = next((name for name in _TOOL_ICONS if pie.icon(name)==node.icon_value),None)
            if node.kind == "prop":
                value = getattr(node.owner,node.name)
                if node.rna.type in {"INT","FLOAT"}:
                    formatted = numeric_text(node,value)
                    if self.edit and self.edit[0].owner == node.owner and self.edit[0].name == node.name:
                        formatted = self.edit[1]+"|"
                    right_value = formatted
                    label = value_label(label)
                    span = node.rna.soft_max-node.rna.soft_min
                    fraction = max(0,min(1,(value-node.rna.soft_min)/span)) if span>0 else 0
                    if fraction>0:
                        x0,y0,x1,y1 = node.rect
                        if node.slider:
                            rounded((x0,y0,x0+(x1-x0)*fraction,y1),palette.selection_fill(style.inner,amount=.35),scale=self.scale)
                elif node.rna.type == "ENUM":
                    value_text = enum_selection_text(node)
                    if node.rna.is_enum_flag and len(value)>2:
                        value_text = value_text.split(", ")[0]+f" (+{len(value)-1})"
                    label = value_label(label)
                    separator = "" if not label or label.endswith("：") else " "
                    label += separator+value_text+" ▾"
                elif node.rna.type == "BOOLEAN":
                    if node.name=="use_snap":
                        mark = "SNAP_ON" if value else "SNAP_OFF"
                    elif node.name=="use_mesh_automerge":
                        mark = "AUTOMERGE_ON" if value else "AUTOMERGE_OFF"
                    elif node.name=="show_viewport":
                        mark = "RESTRICT_VIEW_OFF" if value else "RESTRICT_VIEW_ON"
                    elif (not node.toggle or mark) and not native_checkbox:
                        mark = mark or ("CHECKBOX_HLT" if value else "CHECKBOX_DEHLT")
            elif node.kind in {"menu","popover"}:
                label += " ▾"
            color = (*style.text[:3],1) if chosen and node.is_enabled and not checkbox else color
            rect = node.rect
            if native_checkbox:
                option = bpy.context.preferences.themes[0].user_interface.wcol_option
                y = (rect[1]+rect[3])/2
                bounds = (rect[0]+5*self.scale,y-6*self.scale,rect[0]+17*self.scale,y+6*self.scale)
                fill = palette.accent() if value else tuple(option.inner)
                rounded(bounds,fill,tuple(option.outline),self.scale*.65)
                if value:
                    symbol("CHECKMARK",bounds,(.06,.06,.06,1),self.scale*.7)
                rect = (rect[0]+18*self.scale,rect[1],rect[2],rect[3])
            icon_only = getattr(node,"icon_only",False)
            icon_rect = ((rect[0]+rect[2])/2-8*self.scale,rect[1],(rect[0]+rect[2])/2+8*self.scale,rect[3]) if icon_only else (rect[0]+4*self.scale,rect[1],rect[0]+18*self.scale,rect[3])
            if node.kind=="prop" and node.name=="include_occluded":
                mark = "CHECKBOX_HLT" if value else "CHECKBOX_DEHLT"
            if tool_icon is not None:
                y = (rect[1]+rect[3])/2
                if not icon_only:
                    icon_rect = (rect[0]+4*self.scale,rect[1],rect[0]+20*self.scale,rect[3])
                px = icon_px(self.scale)
                x,bottom = math.floor(icon_rect[0]+.5),math.floor(y-px/2+.5)
                if not paint_tool_icon(tool_icon,(x,bottom,x+px,bottom+px),
                                       icon_colours(style,chosen and node.is_enabled),color[3]):
                    texture = tool_texture("ppo."+tool_icon,px)
                    if texture is not None:
                        paint_icon(texture,(x,bottom,x+px,bottom+px),(1,1,1,color[3]))
            elif mark:
                symbol(mark,icon_rect,color,self.scale)
            if not icon_only:
                if mark or tool_icon is not None:
                    rect = (rect[0]+(20 if tool_icon is not None else 18)*self.scale,rect[1],rect[2],rect[3])
                if right_value is not None:
                    if not label:
                        text(rect,right_value,color,self.scale,centered=True)
                        continue
                    blf.size(0,font_size(self.scale))
                    value_width = min((rect[2]-rect[0])*.60,max(50*self.scale,blf.dimensions(0,right_value)[0]+10*self.scale))
                    text((rect[2]-value_width-5*self.scale,rect[1],rect[2]-5*self.scale,rect[3]),right_value,color,self.scale,right=True)
                    rect = (rect[0]+5*self.scale,rect[1],rect[2]-value_width-5*self.scale,rect[3])
                snapping_toggle = node.kind=="prop" and node.toggle and snapping_control
                text(rect,label,color,self.scale,(node.kind=="choice" and not mark) or snapping_toggle)

    def paint_tooltip(self,value,foreground):
        blf.size(0,font_size(self.scale))
        width = min(350*self.scale,self.size[0]-30)
        lines, line = [], ""
        for char in value:
            if blf.dimensions(0,line+char)[0]>width-16*self.scale:
                lines.append(line)
                line = ""
            line += char
        if line:
            lines.append(line)
        lines = lines[:5]
        height = (len(lines)*17+12)*self.scale
        x = max(8,min(self.size[0]-width-8,self.mouse[0]+15*self.scale))
        y = max(height+8,self.mouse[1]-15*self.scale)
        rounded((x,y-height,x+width,y),(.04,.04,.04,.98),(.25,.25,.25,1),self.scale)
        for index,line in enumerate(lines):
            text((x+4*self.scale,y-(index+1)*17*self.scale-4*self.scale,x+width-4*self.scale,y-index*17*self.scale-4*self.scale),line,(.9,.9,.9,1),self.scale)


class PPO_OT_persistent_pie(bpy.types.Operator):
    bl_idname = "ppo.persistent_pie"
    bl_label = "PolyTailor"
    bl_translation_context = "Poly Toolset"
    bl_description = "Change settings in the W wheel; Esc or a click outside closes it"
    bl_options = {"INTERNAL"}
    ppo_mesh_read_only = True

    @classmethod
    def poll(cls, context):
        return pie.PPO_OT_call_pie.poll(context)

    def invoke(self, context, event):
        self.session = WheelSession()
        result = self.session.invoke(context,event,self)
        p = prefs.get(context)
        if "RUNNING_MODAL" in result and getattr(p, "wheel_open_count", 3) < 3:
            p.wheel_open_count += 1   # the sidebar's W hint hides after the third open
            try:
                context.preferences.is_dirty = True   # kept with the preferences when they are saved
            except (AttributeError, TypeError):
                pass
        return result

    def modal(self, context, event):
        return self.session.modal(context,event)

    def cancel(self, context):
        self.session.cancel(context)


def register():
    global _registered
    bpy.utils.register_class(PPO_OT_persistent_pie)
    _registered = True


def unregister():
    global _registered, _icon_shader, _icon_batch
    _registered = False
    _warmed[0] = False
    if bpy.app.timers.is_registered(_warmup_tick):
        bpy.app.timers.unregister(_warmup_tick)
    _stop_warmup()
    _warmup_jobs.clear()
    _warmup_pending.clear()
    _warmup_ready[0] = True
    if bpy.app.timers.is_registered(watch_context):
        bpy.app.timers.unregister(watch_context)
    if _active is not None:
        _active.restore_drag()
        _active.close()
    _icons.clear()
    _icon_shader = None
    _icon_batch = None
    clear_vector_icons()
    bpy.utils.unregister_class(PPO_OT_persistent_pie)
