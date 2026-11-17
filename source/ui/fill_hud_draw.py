# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Fill HUD drawing.

Pure drawing: no operator access. ui/fill_hud.py builds the input dict:

    mouse (x, y) · scale · size · bounds (left, right, bottom) · region
    fill = {n (int | None while searching), k (-1: none current), busy, word, edges_text, note, first,
            phase ('fallback' while the wider search runs), fallback (bool: result came from it),
            thumbs: [{segs (E,2,2), rim (idx | None), tris (T,3,2), faces, loops ([(P,2)] | None), poles ((Q,2))}]}

Layout:
  - one box at the cursor (label offset, flip left / clamp at the region edges);
  - header "k / n · word" + the edge count; "k / …" + spinner while searching; thumbnails grow as candidates arrive;
  - a silent fallback result is simply n = 1 with one thumbnail; nothing found -> no box (the caller draws nothing).

GPU note: blf.draw() resets the blend mode to NONE, so every fill / line below sets ALPHA blending first.
"""
import math
import time

import gpu

from ..core import draw as gd   # aliased: this module's own draw() would shadow it
from .. import palette

GAP = 8            # px between the bar row and the strip
PAD = 6
BAR_W, BAR_H = 176, 24
CELL_W, CELL_H, CELL_GAP = 58, 48, 6   # 58x48: dense candidates still show their rim shape
BACKDROP = (0., 0., 0., .6)        # .6: text stays >= 4.5:1 over light-grey meshes
CELL_BG = (.11, .11, .11, .88)
LINE = (.86, .86, .86)          # thumbnail strokes (neutral: the accent is reserved for "current")
_side = {}                      # region key -> [left, above]: flip hysteresis


def _alpha():
    gpu.state.blend_set("ALPHA")


def _rect(x0, y0, x1, y1, r, color):
    _alpha()
    gd.rounded_rect(x0, y0, x1, y1, r, color)


def _frame(x0, y0, x1, y1, r, color, width):
    """Rounded outline (matches the rounded backdrop)."""
    pts = []
    for cx, cy, start in ((x1 - r, y0 + r, -math.pi / 2), (x1 - r, y1 - r, 0.), (x0 + r, y1 - r, math.pi / 2),
                          (x0 + r, y0 + r, math.pi)):
        pts.extend(gd.circle_points((cx, cy), r, 4, start, start + math.pi / 2))
    _alpha()
    gd.polyline(pts, color, width, closed=True)


def _segs(coords, color, width):
    if coords:
        _alpha()
        gd.segments(coords, color, width)


def _tris(coords, color):
    if coords:
        _alpha()
        gd._draw("TRIS", coords, color)


def _text(xy, s, color, size):
    gd.text(xy, s, color, size, shadow=True)


def _text_w(s, size):
    import blf
    blf.size(0, size)
    return blf.dimensions(0, s)[0]


def _spinner(ctx, c, r, s, color=None):
    """color None: the accent (active side). The dimmed side passes a light neutral grey, one step brighter than the
    dimmed box, so "still searching" stays legible without orange."""
    phase = (time.perf_counter() * 1.25 % 1.) * math.tau
    _alpha()
    gd.ring(c, r, palette.secondary(ctx, .45), 1.2 * s)
    gd.polyline(gd.circle_points(c, r, 24, phase, phase + math.tau * .72), color or palette.accent(ctx), 2. * s)


def _place(hud, w, row_h, strip_h, ctx, key):
    """(x, bar_top, above) of the group. anchored at the cursor with the
    old Fill cursor label's offset - left edge 18 px right of the cursor, the bar's text on the label's baseline
    (6 px below the cursor) - on every redraw. Flips left of the cursor at the right edge and puts the strip above the
    bar at the bottom edge (40 px hysteresis against flicker), then clamps the whole group into the region."""
    s, size = hud['scale'], hud['size']
    mx, my = hud['mouse']
    left, right, bottom = hud['bounds']
    margin = 12 * s
    try:
        ceiling = ctx.region.height - margin
    except AttributeError:
        ceiling = my + 10000
    side = _side.setdefault(key, [False, False])
    # horizontal
    x = mx + 18 * s
    over = x + w - (right - margin)
    if not side[0] and over > 0:
        side[0] = True
    elif side[0] and over < -40 * s:
        side[0] = False
    if side[0]:
        x = mx - 18 * s - w
    x = max(left + margin, min(x, right - margin - w))
    # vertical: the bar text sits on the old label baseline; the strip hangs below, or above near the bottom edge
    top = my - 6 * s + (BAR_H * s) / 2 + .36 * size + PAD * s
    gap = (GAP * s + strip_h) if strip_h else 0.
    room = top - row_h - gap - (bottom + margin)
    if not side[1] and room < 0:
        side[1] = True
    elif side[1] and room > 40 * s:
        side[1] = False
    above = side[1] and strip_h > 0
    hi = top + (gap if above else 0.)              # group top
    lo = top - row_h - (0. if above else gap)       # group bottom
    if hi > ceiling:
        top -= hi - ceiling
        lo -= hi - ceiling
    if lo < bottom + margin:
        top += (bottom + margin) - lo
    return x, top, above


def _tr(msg):
    """The add-on's own translation (i18n CONTEXT); plain English when unavailable (mock / tests)."""
    try:
        from ..i18n import tr
        return tr(msg)
    except Exception:   # noqa: BLE001
        return msg


def draw(ctx, hud):
    """One box at the cursor. Header "k / n · word" + edge count; "k / …" + spinner while searching
    (n still open); the confirmed candidates' thumbnails grow as they arrive; the current one is accented."""
    s, size = hud['scale'], hud['size']
    acc = palette.accent(ctx)
    main = palette.main(ctx)
    muted = palette.muted(ctx)
    f = hud['fill']
    pad = PAD * s
    thumbs = [t for t in (f.get('thumbs') or []) if t is not None]
    n, k, busy = f.get('n'), int(f.get('k', -1) if f.get('k') is not None else -1), bool(f.get('busy'))
    if busy:
        head = (f"{k + 1 if k >= 0 else len(thumbs)} / …") if thumbs else "… / …"
    else:
        head = (f"{k + 1} / {n}" if k >= 0 else f"– / {n}") if n else "… / …"
    if not busy and n and f.get('word'):
        head += f" · {f['word']}"
    fallback_tag = ''
    if f.get('phase') == 'fallback' and busy:      # the wider search is announced
        head = _tr("Searching wider...")
    elif f.get('fallback') and not busy and n:      # its result keeps a quiet marker in the muted tail colour
        fallback_tag = " · " + _tr("Wider search")
    tail = f.get('edges_text', '')
    if f.get('note'):
        tail = f"{tail} · {f['note']}" if tail else f['note']
    cw, ch, cg = CELL_W * s, CELL_H * s, CELL_GAP * s
    head_h = size + 10 * s
    spin_w = 22 * s if busy else 0.
    head_w = pad + 6 * s + _text_w(head + fallback_tag, size) + 16 * s + _text_w(tail, size) + spin_w + pad
    slots = len(thumbs)
    cells_w = (pad + 12 * s + slots * cw + (slots - 1) * cg + 12 * s + pad) if slots else 0.
    w = max(head_w, cells_w, (BAR_W * .6) * s)
    h = pad + head_h + (ch + pad if slots else 0.)
    key = (hud.get('region') or 0)
    x, top, _above = _place(hud, w, h, 0., ctx, key)
    sy1, sy0 = top, top - h
    _rect(x, sy0, x + w, sy1, 6 * s, BACKDROP)
    _frame(x + .5, sy0 + .5, x + w - .5, sy1 - .5, 6 * s, (*acc[:3], .9), 1.6 * s)
    ty = sy1 - pad - size * .85
    _text((x + pad + 6 * s, ty), head, main if (thumbs or n) else muted, size)
    if fallback_tag:
        _text((x + pad + 6 * s + _text_w(head, size), ty), fallback_tag, (*muted[:3], .9), size)
    tail_x = x + w - pad - spin_w - _text_w(tail, size) - 4 * s
    edges = f.get('edges_text', '') or ''
    if edges and f.get('edges_odd') and tail.startswith(edges):   # odd rim: the edge count in the warning colour
        _text((tail_x, ty), edges, palette.warning(1.0), size)   # the same amber as the thumbnails' odd triangles
        if tail[len(edges):]:
            _text((tail_x + _text_w(edges, size), ty), tail[len(edges):], (*muted[:3], .9), size)
    else:
        _text((tail_x, ty), tail, (*muted[:3], .9), size)
    if busy:
        _spinner(ctx, (x + w - pad - 9 * s, sy1 - pad - head_h / 2 + 2 * s), 6 * s, s)
    if not thumbs:
        return
    s_acc = acc[:3]

    def fa(c):   # an RGBA colour
        return (*c[:3], c[3] if len(c) > 3 else 1.)
    first = int(f.get('first', 0))
    cx0 = x + pad + 12 * s
    cy0 = sy0 + pad
    if n and first > 0:
        _text((x + pad, cy0 + ch / 2 - size * .35), "‹", fa(muted), size)
    if n and first + len(thumbs) < n:
        _text((x + w - pad - 8 * s, cy0 + ch / 2 - size * .35), "›", fa(muted), size)
    inset = 6 * s
    k_fit = min(cw, ch) - 2 * inset
    interior, interior_dense, rims, cur_rim, odd, flows, poles, cur_poles = [], [], [], [], [], [], [], []
    for j, th in enumerate(thumbs):
        cx = cx0 + j * (cw + cg)
        i = first + j
        current = i == k
        _rect(cx, cy0, cx + cw, cy0 + ch, 4 * s, fa(CELL_BG))
        if current:
            _rect(cx, cy0, cx + cw, cy0 + ch, 4 * s, (*acc[:3], .22))
        mx_, my_ = cx + cw / 2, cy0 + ch / 2
        segs = th.get('segs')
        rim_idx = th.get('rim')
        rim_set = set(rim_idx) if rim_idx is not None else set()
        loops = th.get('loops')   # "main flow lines only" when the data side provides loops
        # many interior lines in a 58 px cell (e.g. a striped candidate, ~20+ long parallel edges) hatch into a blob:
        # draw them faint so the rim still carries the shape
        crowded = segs is not None and loops is None and len(segs) - len(rim_set) > 14
        bucket = interior_dense if crowded else interior
        if segs is not None:
            for e, (a, b) in enumerate(segs):
                if e not in rim_set and (loops is not None or th.get('faces', 0) > 60):
                    continue   # flow lines replace the interior; dense with no loops: rim only, never a hatched blob
                p = ((mx_ + a[0] * k_fit, my_ + a[1] * k_fit), (mx_ + b[0] * k_fit, my_ + b[1] * k_fit))
                (cur_rim if current else rims).extend(p) if e in rim_set else bucket.extend(p)
        if loops is not None:
            for poly in loops:
                pts = [(mx_ + v[0] * k_fit, my_ + v[1] * k_fit) for v in poly]
                for a, b in zip(pts, pts[1:]):
                    flows += [a, b]
            pl = th.get('poles')
            if pl is not None:
                (cur_poles if current else poles).extend((mx_ + v[0] * k_fit, my_ + v[1] * k_fit) for v in pl)
        tris = th.get('tris')
        if tris is not None:
            for t in tris:
                odd.extend((mx_ + v[0] * k_fit, my_ + v[1] * k_fit) for v in t)
        if current:
            _frame(cx - 1, cy0 - 1, cx + cw + 1, cy0 + ch + 1, 4 * s, acc, 2.2 * s)
        else:
            _frame(cx, cy0, cx + cw, cy0 + ch, 4 * s, fa((1., 1., 1., .12)), 1.)
    dense = any(th.get('faces', 0) > 60 and th.get('loops') is None for th in thumbs)   # dense: interior recedes, the rim carries the shape
    _tris(odd, fa((*palette.WARNING[:3], .35)))
    _segs(interior, fa((*LINE, .24 if dense else .55)), 1.)
    _segs(interior_dense, fa((*LINE, .2)), 1.)
    _segs(rims, fa((*LINE, 1.)), 1.6 * s)
    _segs(cur_rim, (*acc[:3], 1.), 1.6 * s)
    _segs(flows, fa((*LINE, .75)), 1. * s)
    _alpha()
    for p in poles:       # poles: orange dots with a thin dark ring
        gd.disc(p, 1.4 * s + 1.6, fa((0., 0., 0., .9)))
        gd.disc(p, 1.4 * s, fa((*s_acc, 1.)))   # radius 1.4 px x UI scale
    for p in cur_poles:   # on the orange current cell an orange dot vanishes: white dot, dark ring
        gd.disc(p, 1.4 * s + 1.6, (0., 0., 0., .9))
        gd.disc(p, 1.4 * s, (1., 1., 1., 1.))
