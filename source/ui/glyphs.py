# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""In-house monochrome UI marks for the GPU-drawn W wheel.

GPU draw handlers cannot draw Blender's built-in icons, so the wheel draws these simple line marks in code. They keep the meaning and
rough shape of Blender's icons (cube / magnet / eye / checkbox ...) at 16 px.

    draw(name, rect, color, scale) -> bool    # True when drawn; False -> the caller's own fallback (custom_pie.symbol)

`color` is the final RGBA (the caller applies its opening fade). Coordinates below are in a 16-unit box centred on
the rect, y up.
"""
import math

from ..core import draw as gd
import gpu
from gpu_extras.batch import batch_for_shader

_DIM = .55     # "off" states: same mark, lower alpha

# ---- joined strokes (strokes must meet cleanly)
# Blender's POLYLINE shader draws every segment as its own quad: sharp corners (the checkmark, arrows, cubes) show
# notches and hairline gaps. Glyph strokes are therefore built here as ONE triangle mesh per stroke: mitred joins
# (bevel past a 4x miter limit), square-extended ends where strokes must meet, and a 1 px alpha feather on both sides
# for anti-aliasing. One draw call per stroke, no self-overlap, so translucent ("off") marks have no dark joints.

def _offset(pts, d, closed):
    n = len(pts)
    out = []
    for i in range(n):
        if not closed and i in (0, n - 1):
            a, b = (pts[0], pts[1]) if i == 0 else (pts[-2], pts[-1])
            dx, dy = b[0] - a[0], b[1] - a[1]
            ln = math.hypot(dx, dy) or 1.
            out.append((pts[i][0] - dy / ln * d, pts[i][1] + dx / ln * d))
            continue
        p0, p1, p2 = pts[i - 1], pts[i], pts[(i + 1) % n]
        d0x, d0y = p1[0] - p0[0], p1[1] - p0[1]
        d1x, d1y = p2[0] - p1[0], p2[1] - p1[1]
        l0, l1 = math.hypot(d0x, d0y) or 1., math.hypot(d1x, d1y) or 1.
        n0 = (-d0y / l0, d0x / l0)
        n1 = (-d1y / l1, d1x / l1)
        mx, my = n0[0] + n1[0], n0[1] + n1[1]
        ml = math.hypot(mx, my)
        if ml < 1e-6:
            out.append((p1[0] + n1[0] * d, p1[1] + n1[1] * d))
            continue
        mx, my = mx / ml, my / ml
        k = d / max(mx * n1[0] + my * n1[1], .25)    # miter length, limited to 4x
        out.append((p1[0] + mx * k, p1[1] + my * k))
    return out


def _stroke(pts, color, width, closed=False, cap=0.):
    """A joined, anti-aliased stroke through `pts` (pixel coords). cap: extend open ends by this many px."""
    pts = [tuple(q) for q in pts]
    pts = [q for i, q in enumerate(pts) if i == 0 or math.hypot(q[0] - pts[i - 1][0], q[1] - pts[i - 1][1]) > 1e-4]
    if closed and len(pts) > 2 and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 1e-4:
        pts = pts[:-1]
    if len(pts) < 2:
        return
    if cap and not closed:
        for idx, other in ((0, 1), (-1, -2)):
            dx, dy = pts[idx][0] - pts[other][0], pts[idx][1] - pts[other][1]
            ln = math.hypot(dx, dy) or 1.
            pts[idx] = (pts[idx][0] + dx / ln * cap, pts[idx][1] + dy / ln * cap)
    core = max(width / 2. - .5, 0.)
    rows = [(-core - 1., 0.), (-core, 1.), (core, 1.), (core + 1., 0.)]
    lines = [(_offset(pts, d, closed), a) for d, a in rows]
    seq = list(range(len(pts))) + ([0] if closed else [])
    pos, col = [], []
    r, g, b, a0 = color
    for (la, aa), (lb, ab) in zip(lines, lines[1:]):
        if aa == 0. and ab == 0.:
            continue
        for i, j in zip(seq, seq[1:]):
            quad = ((la[i], aa), (la[j], aa), (lb[j], ab), (lb[i], ab))
            for k in (0, 1, 2, 0, 2, 3):
                pos.append(quad[k][0]); col.append((r, g, b, a0 * quad[k][1]))
    sh = gpu.shader.from_builtin('SMOOTH_COLOR')
    batch_for_shader(sh, 'TRIS', {'pos': pos, 'color': col}).draw(sh)


def _pl(coords, color, width=1., closed=False):
    _stroke(coords, color, width, closed, cap=width / 2.)


def _seg(coords, color, width=1.):
    for i in range(0, len(coords) - 1, 2):
        _stroke([coords[i], coords[i + 1]], color, width, cap=width / 2.)


def _ring(center, radius, color, width=1., segments=48):
    _stroke(gd.circle_points(center, radius, segments)[:-1], color, width, closed=True)


def _ctx(rect, color, scale):
    """Points on the pixel grid (centres at .5): 1 px strokes stay crisp at every UI scale."""
    x, y = math.floor((rect[0] + rect[2]) / 2) + .5, math.floor((rect[1] + rect[3]) / 2) + .5
    return (lambda a, b: (round(x + a * scale - .5) + .5, round(y + b * scale - .5) + .5)), (x, y)


def _arrowhead(tip, direction, size, color):
    """A filled triangle arrowhead at `tip`, pointing along `direction` (unit vector)."""
    dx, dy = direction
    bx, by = tip[0] - dx * size, tip[1] - dy * size
    nx, ny = -dy * size * .6, dx * size * .6
    gd._draw("TRIS", [tip, (bx + nx, by + ny), (bx - nx, by - ny)], color)


def _square(p, color, w, a=5):
    _pl([p(-a, -a), p(a, -a), p(a, a), p(-a, a)], color, w, closed=True)


def _fill(p, coords, color):
    pts = [p(a, b) for a, b in coords]
    tris = [c for i in range(1, len(pts) - 1) for c in (pts[0], pts[i], pts[i + 1])]
    gd._draw("TRIS", tris, color)


def _cube(p, color, w):
    """Front square plus a back square offset up-right: Blender's mesh select-mode cube, flattened."""
    _pl([p(-6, -6), p(2, -6), p(2, 2), p(-6, 2)], color, w, closed=True)
    _pl([p(-2, 6), p(6, 6), p(6, -2)], color, w)
    _seg((p(-6, 2), p(-2, 6), p(2, 2), p(6, 6), p(2, -6), p(6, -2)), color, w)


def _magnet(p, color, w):
    _pl([p(-5, 5), p(-5, -1), p(-3, -4), p(3, -4), p(5, -1), p(5, 5)], color, w)
    _seg((p(-6.5, 5), p(-3.5, 5), p(3.5, 5), p(6.5, 5)), color, w)


def _alpha(color, f):
    return (*color[:3], color[3] * f)


def draw(name, rect, color, scale):
    p, (cx, cy) = _ctx(rect, color, scale)
    w = max(1., scale)
    n = (name or "").upper()
    if n in {"VERTEXSEL", "EDGESEL", "FACESEL"}:
        _cube(p, _alpha(color, .6), w)
        if n == "VERTEXSEL":
            gd.disc(p(2, 2), 2.2 * scale, color, segments=12)
        elif n == "EDGESEL":
            _seg((p(-6, 2), p(2, 2)), color, 2.2 * scale)
        else:
            _fill(p, ((-6, -6), (2, -6), (2, 2), (-6, 2)), _alpha(color, .7))
    elif n in {"SNAP_ON", "SNAP_OFF"}:
        _magnet(p, color if n == "SNAP_ON" else _alpha(color, _DIM), (1.8 if n == "SNAP_ON" else 1.2) * scale)
    elif n == "SNAP_VERTEX":
        _square(p, _alpha(color, .6), w)
        gd.disc(p(0, 0), 2.2 * scale, color, segments=12)
    elif n == "SNAP_EDGE":
        _seg((p(-6, -5), p(6, 5)), color, 2 * scale)
    elif n == "SNAP_MIDPOINT":
        _seg((p(-6, -5), p(6, 5)), _alpha(color, .6), w)
        gd.disc(p(0, 0), 2.2 * scale, color, segments=12)
    elif n == "SNAP_PERPENDICULAR":
        _pl([p(-5, 6), p(-5, -5), p(6, -5)], color, 1.6 * scale)
        _pl([p(-5, -1), p(-1, -1), p(-1, -5)], _alpha(color, .7), w)
    elif n == "SNAP_FACE":
        _fill(p, ((-5, -5), (5, -5), (5, 5), (-5, 5)), _alpha(color, .45))
        _square(p, color, w)
    elif n == "SNAP_FACE_NEAREST":
        _fill(p, ((-6, -6), (6, -6), (6, -2), (-6, -2)), _alpha(color, .45))
        _seg((p(0, 6), p(0, -1)), color, 1.4 * scale)
        _pl([p(-2.5, 1.5), p(0, -1), p(2.5, 1.5)], color, 1.4 * scale)
    elif n == "SNAP_VOLUME":
        _cube(p, color, w)
        gd.disc(p(-2, -2), 1.6 * scale, color, segments=10)
    elif n == "SNAP_INCREMENT":
        _seg((p(-6, -3), p(6, -3)), color, 1.4 * scale)
        _seg(tuple(q for i, h in ((-6, 6), (-3, 3), (0, 5), (3, 3), (6, 6)) for q in (p(i, -3), p(i, -3 + h))), color, w)
    elif n == "SNAP_GRID":
        for o in (-5, 0, 5):
            _seg((p(o, -6), p(o, 6), p(-6, o), p(6, o)), _alpha(color, .8), w)
    elif n in {"AUTOMERGE_ON", "AUTOMERGE_OFF"}:
        c = color if n.endswith("ON") else _alpha(color, _DIM)
        for a, b in ((-5, -4), (5, -4), (0, 5)):
            _ring(p(a, b), 1.6 * scale, c, w, segments=8)
        _seg((p(-3, -4), p(3, -4), p(-4, -2), p(-1, 3), p(4, -2), p(1, 3)), c, w)
    elif n in {"RESTRICT_VIEW_OFF", "RESTRICT_VIEW_ON", "HIDE_OFF", "HIDE_ON"}:
        visible = n.endswith("OFF")
        upper = [p(6.5 * math.cos(t), 4 * math.sin(t)) for t in [i * math.pi / 10 for i in range(11)]]
        lower = [p(6.5 * math.cos(t), -4 * math.sin(t)) for t in [i * math.pi / 10 for i in range(11)]]
        _pl(lower, color, w)
        if visible:
            _pl(upper, color, w)
            gd.disc(p(0, 0), 2 * scale, color, segments=12)
        else:   # closed eye: the lid line plus lashes
            _seg((p(-4, -3), p(-5, -5.5), p(0, -4), p(0, -6.5), p(4, -3), p(5, -5.5)), color, w)
    elif n in {"CHECKBOX_DEHLT", "CHECKBOX_HLT"}:
        _square(p, color, w)
        if n == "CHECKBOX_HLT":
            _pl((p(-3.5, 0), p(-1, -2.8), p(4, 3)), color, 1.6 * scale)
    elif n in {"LOCKED", "UNLOCKED"}:
        _pl([p(-4, -6), p(4, -6), p(4, 0), p(-4, 0)], color, w, closed=True)
        arc = gd.circle_points(p(0, 2.5) if n == "LOCKED" else p(2, 2.5), 2.8 * scale, segments=10, start=0, end=math.pi)
        _pl(arc, color, w)
        if n == "LOCKED":
            _seg((p(-2.8, 0), p(-2.8, 2.5)), color, w)
    elif n == "ERROR":
        _pl([p(0, 6), p(6.5, -5), p(-6.5, -5)], color, w, closed=True)
        _seg((p(0, 2.5), p(0, -1.5)), color, 1.4 * scale)
        gd.disc(p(0, -3.4), .9 * scale, color, segments=8)
    elif n in {"X", "CANCEL", "PANEL_CLOSE"}:
        _seg((p(-4.5, -4.5), p(4.5, 4.5), p(-4.5, 4.5), p(4.5, -4.5)), color, 1.4 * scale)
    elif n in {"CLIPUV_HLT", "CLIPUV_DEHLT"}:
        gd.dashed([p(-6, -6), p(6, -6), p(6, 6), p(-6, 6), p(-6, -6)], _alpha(color, .7), w, 2 * scale, 2 * scale)
        _square(p, color if n.endswith("HLT") else _alpha(color, _DIM), w, 3)
    elif n == "EDITMODE_HLT":
        _square(p, _alpha(color, .7), w)
        for a, b in ((-5, -5), (5, -5), (5, 5), (-5, 5)):
            gd.disc(p(a, b), 1.7 * scale, color, segments=8)
    elif n in {"MESH_DATA", "OUTLINER_DATA_MESH"}:
        _pl([p(0, 6), p(6, -5), p(-6, -5)], _alpha(color, .7), w, closed=True)
        for a, b in ((0, 6), (6, -5), (-6, -5)):
            gd.disc(p(a, b), 1.6 * scale, color, segments=8)
    elif n == "OUTLINER_OB_MESH":
        _pl([p(0, 6), p(6, -5), p(-6, -5)], color, 1.4 * scale, closed=True)
    elif n == "MESH_PLANE":
        _pl([p(-6, -4), p(3, -4), p(6, 4), p(-3, 4)], color, w, closed=True)
    elif n == "MOD_MESHDEFORM":
        _square(p, _alpha(color, .7), w, 6)
        _pl([p(0, 4), p(4, 0), p(0, -4), p(-4, 0)], color, w, closed=True)
    elif n == "MOD_SIMPLIFY":
        _square(p, color, w)                                   # a quad reduced to two triangles
        _seg((p(-5, -5), p(5, 5)), _alpha(color, .7), w)
    elif n in {"RESTRICT_SELECT_OFF", "RESTRICT_SELECT_ON"}:
        c = color if n.endswith("OFF") else _alpha(color, _DIM)
        _pl([p(-3, 6), p(-3, -4), p(-.5, -1.5), p(1.5, -6), p(3, -5.3), p(1, -1), p(4, -1)], c, w, closed=True)
        if n.endswith("ON"):   # excluded: a slash, so it never reads as a dimmed "all"
            _seg((p(-6, -6), p(6, 6)), color, w)
    elif n == "LOOP_BACK":   # reset: one counter-clockwise turn with a filled head (Blender's LOOP_BACK shape)
        r = 5.
        a0, a1 = math.radians(-40), math.radians(195)   # counter-clockwise from lower right, over the top, to the left
        arc = [p(r * math.cos(a0 + (a1 - a0) * i / 16), r * math.sin(a0 + (a1 - a0) * i / 16)) for i in range(17)]
        _pl(arc, color, 1.4 * scale)
        t = (-math.sin(a1), math.cos(a1))                 # counter-clockwise tangent at the arc end
        ex, ey = r * math.cos(a1), r * math.sin(a1)
        _arrowhead(p(ex + t[0] * 2.8, ey + t[1] * 2.8), t, 4.0 * scale, color)
    elif n == "CHECKMARK":
        _pl((p(-5, 0), p(-1.5, -3.5), p(5, 4)), color, 1.8 * scale)
    elif n == "TRASH":
        _pl([p(-4, 3), p(-3.2, -6), p(3.2, -6), p(4, 3)], color, w)
        _seg((p(-6, 4), p(6, 4), p(-2, 6), p(2, 6), p(-1.2, 1.5), p(-1.2, -4), p(1.2, 1.5), p(1.2, -4)), color, w)
    elif n == "MOD_MIRROR":
        gd.dashed([p(0, -7), p(0, 7)], _alpha(color, .7), w, 2 * scale, 1.5 * scale)
        _pl([p(-2, -5), p(-6, 0), p(-2, 5)], color, w, closed=True)
        _fill(p, ((2, -5), (6, 0), (2, 5)), _alpha(color, .8))
    elif n == "MOD_LATTICE":
        _square(p, color, w, 6)
        _seg((p(-2, -6), p(-2, 6), p(2, -6), p(2, 6), p(-6, -2), p(6, -2), p(-6, 2), p(6, 2)), _alpha(color, .7), w)
    elif n in {"RADIOBUT_ON", "RADIOBUT_OFF"}:
        _ring((cx, cy), 5 * scale, color, w, segments=20)
        if n.endswith("ON"):
            gd.disc((cx, cy), 2.5 * scale, color, segments=14)
    elif n == "GREASEPENCIL":
        _pl([p(-6, -6), p(-5, -2), p(3, 6), p(6, 3), p(-2, -5)], color, w, closed=True)
        _seg((p(-5, -2), p(-2, -5), p(1.5, 4.5), p(4.5, 1.5)), _alpha(color, .7), w)
    elif n == "TIME":
        _ring(p(0, 0), 6 * scale, color, w, segments=20)
        _seg((p(0, 0), p(0, 4), p(0, 0), p(3, -1.5)), color, w)
    elif n == "FILE_REFRESH":
        _pl(gd.circle_points(p(0, 0), 5.5 * scale, segments=16, start=.6, end=5.6), color, 1.3 * scale)
        _pl([p(2.2, 6.5), p(4.6, 3.3), p(1, 2.4)], color, 1.3 * scale)
    else:
        return False
    return True
