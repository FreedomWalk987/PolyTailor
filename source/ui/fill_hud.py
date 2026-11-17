# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Data side of the Fill HUD.

build_fill(op, ctx, s, size) -> hud['fill'], one plain dict per frame; draw_hud wraps it into the input of
fill_hud_draw.draw(ctx, hud), which does all the drawing. Thumbnail source geometry = each candidate's faces projected on the rim's best-fit
plane (one orientation per outline), normalised to [-0.5, 0.5] and cached per plan per outline (reset on an outline
change)."""

import numpy as np

VISIBLE = 5
_warned = [False]


def applicable(op):
    """the one engine's box: a candidate shown or the search running for an outline (Legacy, or New as the
    silent fallback); with nothing found the caller's text label explains why."""
    if op.loop is None or getattr(op, 'error', ''):
        return False
    return (op.plan is not None or op.job is not None or getattr(op, 'background', None) is not None
            or getattr(op, 'triangle_job', None) is not None)


def _cache(op):
    ids = tuple(v.index for v in op.loop) if op.loop is not None else None
    c = op.__dict__.get('_hud_thumbs')
    if c is None or c.get('ids') != ids:   # a new outline: no stale thumbnails
        c = op.__dict__['_hud_thumbs'] = dict(ids=ids, basis=None, plans={})
    return c


def _basis(op, c):
    if c['basis'] is None:
        from ..core import fill
        W = np.asarray(fill.world_points(op.state.mw, op.loop), float)
        ctr = W.mean(0); _, _, vt = np.linalg.svd(W - ctr, full_matrices=False)
        c['basis'] = (ctr, vt[:2])
    return c['basis']


def _thumb(op, c, plan):
    """Cached thumbnail dict of a plan: segs (E,2,2) in [-0.5, 0.5], rim (indices into segs of the boundary edges),
    tris (K,3,2) odd triangles, faces (count)."""
    hit = c['plans'].get(id(plan))
    if hit is not None and hit[0] is plan:
        return hit[1]
    from ..core import fill
    out = dict(segs=np.zeros((0, 2, 2)), rim=None, tris=np.zeros((0, 3, 2)), faces=0)
    try:
        P = np.asarray(plan['positions'], float)
        if P.ndim == 2 and P.shape[1] >= 2:
            if P.shape[1] >= 3:
                ctr, B = _basis(op, c)
                Q = (P[:, :3] - ctr) @ B.T
            else:
                Q = P[:, :2]
            lo, hi = Q.min(0), Q.max(0); span = max(float((hi - lo).max()), 1e-12)
            Q = (Q - (lo + hi) / 2) / span
            faces = [list(map(int, f)) for f in fill.faces_of(plan)]
            # weld coincident vertices (a plan with per-face vertices would make every edge a "rim" edge and
            # leave no loop to walk: a hatched thumbnail)
            K = np.round((P[:, :3] if P.shape[1] >= 3 else P[:, :2]) / (span * 1e-5)).astype(np.int64)
            _u, first, inv = np.unique(K, axis=0, return_index=True, return_inverse=True)
            inv = np.asarray(inv).reshape(-1)
            if len(faces) > FLOW_MAX_FACES:   # very large plans, vectorised; only the rim and poles are drawn
                out = _thumb_large(faces, inv, Q[first], (len(P), len(first)))
                c['plans'][id(plan)] = (plan, out)
                return out
            if len(first) < len(P):
                Q = Q[first]
                welded = []
                for f in faces:
                    g = [int(inv[v]) for v in f]
                    g = [v for i, v in enumerate(g) if v != g[i - 1]]   # drop collapsed (repeated) corners
                    if len(g) >= 3:
                        welded.append(g)
                faces = welded
            out_welded = (len(P), len(first))
            uses = {}; tris = []
            for f in faces:
                for i in range(len(f)):
                    a, b = f[i], f[(i + 1) % len(f)]
                    e = (min(a, b), max(a, b)); uses[e] = uses.get(e, 0) + 1
                if len(f) == 3:
                    tris.append(f)
            E = sorted(uses)
            out = dict(segs=Q[np.array(E, int)] if E else out['segs'],
                       rim=np.array([i for i, e in enumerate(E) if uses[e] == 1], int),   # an edge of one face = the rim
                       tris=Q[np.array(tris, int)] if tris else out['tris'], faces=len(faces), welded=out_welded)
            try:
                loops, poles = _flow(faces, uses, Q, c.get('scale') or 1.)
                out['loops'] = None if loops is None else [Q[np.array(L, int)] for L in loops]
                out['poles'] = Q[np.array(poles, int)] if poles else np.zeros((0, 2))
            except Exception:   # noqa: BLE001  (the drawing falls back to segs)
                out['loops'], out['poles'] = None, np.zeros((0, 2))
    except Exception:   # noqa: BLE001  (a plan without usable geometry)
        pass
    c['plans'][id(plan)] = (plan, out)
    return out


def _thumb_large(faces, inv, Q, welded):
    """the thumbnail of a plan above FLOW_MAX_FACES faces in numpy only: segs = the rim edges
    (the drawing shows rim + poles, no interior), tris, poles; bounded by array work, not per-edge Python."""
    n = np.fromiter((len(f) for f in faces), int, len(faces))
    A = inv[np.fromiter((v for f in faces for v in f), np.int64, int(n.sum()))]
    start = np.repeat(np.cumsum(n) - n, n); pos = np.arange(len(A)) - start
    B = A[start + (pos + 1) % np.repeat(n, n)]
    keep = A != B
    lo, hi = np.minimum(A, B)[keep], np.maximum(A, B)[keep]
    m = int(max(A.max(initial=0), 0)) + 1
    key, cnt = np.unique(lo * m + hi, return_counts=True)
    ea, eb = key // m, key % m
    deg = np.bincount(np.concatenate([ea, eb]), minlength=m)
    rk = cnt == 1
    rimv = np.zeros(m, bool); rimv[ea[rk]] = True; rimv[eb[rk]] = True
    poles = np.nonzero((deg > 0) & (deg != 4) & ~rimv)[0]
    tri = n == 3
    T = A.reshape(-1)[np.repeat(tri, n)].reshape(-1, 3) if tri.any() else None
    segs = np.stack([Q[ea[rk]], Q[eb[rk]]], 1) if rk.any() else np.zeros((0, 2, 2))
    return dict(segs=segs, rim=np.arange(len(segs)), tris=Q[T] if T is not None else np.zeros((0, 3, 2)),
                faces=len(faces), welded=welded, loops=[], poles=Q[poles] if len(poles) else np.zeros((0, 2)))


FLOW_MAX_FACES = 1500   # above this: no flow lines (rim + poles only) to bound the first frame
FIT_PX = 36             # the thumbnail fit box at UI scale 1 (fill_hud_draw: min(58, 48) - 2 x 6 px inset)
GRID_PX = 4.5           # drawn loops of one family stay at least about this far apart on screen


def _flow(faces, uses, Q=None, scale=1.0):
    """Grid lines of a fill: (loops, poles) as vertex-index lists.
    Complete edge loops of both directions, walked through interior valence-4 vertices surrounded by quads (straight
    across: the edge sharing no face with the incoming one), stopping at the rim or at a pole. Loops are layered by
    quad rows from the rim (the loop one quad row in = depth 1, the next = 2, ...) and every k-th layer is kept, k
    chosen so the kept lines are about GRID_PX apart at the UI scale `scale` (from the mean interior edge length in
    thumbnail units, Q); loops of a family with no rim contact are kept, and a family with no kept row keeps its middle
    row (both directions stay visible). Poles = interior vertices with valence != 4. Above
    FLOW_MAX_FACES: no loops (rim + poles only)."""
    import math
    vfaces, deg = {}, {}
    for fi, f in enumerate(faces):
        for v in f:
            vfaces.setdefault(v, []).append(fi)
    for (a, b) in uses:
        deg[a] = deg.get(a, 0) + 1; deg[b] = deg.get(b, 0) + 1
    rimv = {v for e, n in uses.items() if n == 1 for v in e}
    poles = [v for v, d in deg.items() if v not in rimv and d != 4]
    if len(faces) > FLOW_MAX_FACES:   # very large plans: rim + poles only, no extraction
        return [], poles
    ef_cache = {}

    def efaces(e):   # lazy: only along walked loops
        hit = ef_cache.get(e)
        if hit is None:
            hit = ef_cache[e] = set(vfaces.get(e[0], ())) & set(vfaces.get(e[1], ()))
        return hit

    def straight(e, v):   # the edge leaving v straight across from e (None: stop here)
        fs = vfaces.get(v, ())
        if v in rimv or deg.get(v) != 4 or len(fs) != 4 or any(len(faces[fi]) != 4 for fi in fs):
            return None
        es = set()
        for fi in fs:
            f = faces[fi]; i = f.index(v)
            for w in (f[i - 1], f[(i + 1) % 4]):
                es.add((min(v, w), max(v, w)))
        ef = efaces(e)
        nxt = [g for g in es if g != e and not (efaces(g) & ef)]
        return nxt[0] if len(nxt) == 1 else None

    interior = [e for e, n in uses.items() if n != 1]
    loop_of, loops = {}, []
    for e0 in interior:
        if e0 in loop_of:
            continue
        lid = len(loops); chain = [e0[0], e0[1]]; loop_of[e0] = lid
        for back in (False, True):   # grow from e0's second end, then from its first end
            e, v = e0, (e0[0] if back else e0[1])
            while True:
                g = straight(e, v)
                if g is None or g in loop_of or uses.get(g) == 1:
                    break
                loop_of[g] = lid
                v = g[0] if g[1] == v else g[1]
                chain.insert(0, v) if back else chain.append(v)
                e = g
        loops.append(chain)
    # layers: quad rows from the rim (multi-source BFS over "opposite sides of one quad")
    adj = [set() for _ in loops]; rim_nb = set()
    for f in faces:
        if len(f) != 4:
            continue
        es = [(min(f[i], f[(i + 1) % 4]), max(f[i], f[(i + 1) % 4])) for i in range(4)]
        for p, q in ((es[0], es[2]), (es[1], es[3])):
            lp, lq = loop_of.get(p), loop_of.get(q)
            if lp is not None and lq is not None and lp != lq:
                adj[lp].add(lq); adj[lq].add(lp)
            elif lp is None and lq is not None:
                rim_nb.add(lq)
            elif lq is None and lp is not None:
                rim_nb.add(lp)
    depth = [None] * len(loops); frontier = sorted(rim_nb)
    for l in frontier:
        depth[l] = 1
    while frontier:
        nxt = []
        for l in frontier:
            for m in adj[l]:
                if depth[m] is None:
                    depth[m] = depth[l] + 1; nxt.append(m)
        frontier = nxt
    k = 1
    if Q is not None and interior:   # keep every k-th row: drawn lines about GRID_PX apart at this UI scale
        E = np.asarray(interior, int)
        mean = float(np.hypot(*(Q[E[:, 0], :2] - Q[E[:, 1], :2]).T).mean())
        k = max(1, math.ceil(GRID_PX / max(mean * FIT_PX * scale, 1e-6)))
    keep = {l for l, d in enumerate(depth) if d is None or d % k == 0}
    seen = [False] * len(loops)
    for s0 in range(len(loops)):   # a family (loops linked across quads) with no kept row keeps its middle row
        if seen[s0]:
            continue
        comp, stack = [], [s0]; seen[s0] = True
        while stack:
            l = stack.pop(); comp.append(l)
            for m in adj[l]:
                if not seen[m]:
                    seen[m] = True; stack.append(m)
        if not any(l in keep for l in comp):
            ds = sorted(depth[l] for l in comp)
            mid = ds[len(ds) // 2]
            keep.update(l for l in comp if depth[l] == mid)
    return [loops[l] for l in sorted(keep) if len(loops[l]) >= 2], poles


def _more_list(op):
    """(plans, current index, still computing) of the Legacy list as the wheel walks it. The current
    cell is the shown plan by identity (None when the shown plan is not in the list: nothing is highlighted)."""
    from ..core import fill_layouts
    odd, pf, shown, busy = getattr(op, 'odd_wheel', None), getattr(op, 'portfolio', None), op.plan, op.job is not None
    if odd:
        plans = list(odd)
    else:
        plans = [pf.legacy if k == fill_layouts.LEGACY_KEY else pf.entries[k][1] for k in _ranked(op, pf)]
    cur = next((i for i, p in enumerate(plans) if p is shown), None)
    return plans, cur, busy


def _ranked(op, pf):
    """pf.ranked_keys() (the guarded flow score shortlist: the default first, then the R1 band, then S) cached per
    frame state: recomputed only when the candidates, the selection, the lock or the ranking inputs change (the HUD
    draws every frame)."""
    try:
        sig = (id(pf), tuple((k, v[0], id(v[1])) for k, v in pf.entries.items()), id(pf.legacy), pf.selected, pf.locked,
               getattr(pf, 'default', None), pf.ranking,
               getattr(pf, 'quality_floor', None), getattr(pf, 'legacy_slot', None), getattr(pf, 'legacy_match', None), getattr(pf, 'limit', None))
        hash(sig)
    except Exception:   # noqa: BLE001  (an unhashable field: no cache)
        return pf.ranked_keys()
    hit = op.__dict__.get('_hud_ranked')
    if hit is not None and hit[0] == sig:
        return hit[1]
    keys = pf.ranked_keys()
    op.__dict__['_hud_ranked'] = (sig, keys)
    return keys


def _bounds(ctx, s):
    try:
        from .hud import hud_bounds
        left, right, bottom, _size, _step = hud_bounds(ctx)
    except Exception:   # noqa: BLE001
        left, right, bottom = 12 * s, ctx.region.width - 12 * s, 26 * s
    return (left, right, bottom)


def _preview(op):
    """(bbox, start) of the drawn preview in region pixels, from the operator's cached paint geometry."""
    g = op.__dict__.get('_paint_geometry')
    if not g:
        return None, None
    hit = op.__dict__.get('_hud_preview')
    if hit is not None and hit[0] is g:   # the paint geometry is rebuilt only with the plan or the view
        return hit[1]
    op.__dict__['_hud_preview'] = (g, _preview_calc(g))
    return op.__dict__['_hud_preview'][1]


def _preview_calc(g):
    try:
        tris, _yellow, lines, start = g
        pts = [p for p in (lines or [])] + [p for p in (tris or [])]
        if not pts:
            return None, (tuple(start) if start is not None else None)
        A = np.asarray(pts, float)[:, :2]
        return (float(A[:, 0].min()), float(A[:, 1].min()), float(A[:, 0].max()), float(A[:, 1].max())), \
            (tuple(map(float, start[:2])) if start is not None else None)
    except Exception:   # noqa: BLE001
        return None, None


def displayed_plans(op):
    """the Legacy list exactly as the box shows it (and as the wheel steps it): ranked order, or after T
    (progressive) the ARRIVAL order, never re-sorted while shown. Returns (plans, index of the shown plan or None)."""
    plans, cur, _b = _more_list(op)
    if getattr(op, '_went_progressive', False):
        order = op.__dict__.setdefault('_arrival', [])
        live = {id(p) for p in plans}
        order[:] = [p for p in order if id(p) in live]
        known = {id(p) for p in order}
        order.extend(p for p in plans if id(p) not in known)
        plans = list(order)
        cur = next((i for i, p in enumerate(plans) if p is op.plan), None)
    else:
        op.__dict__['_arrival'] = []
    return plans, cur


def _rim_odd(op):
    """the rim whose count the header shows (the plan's, else the hovered loop's) has an odd number of edges."""
    try:
        if op.plan is not None:
            n = sum(1 for o in op.plan['order'] if o is not None)
        elif op.loop is not None:
            n = len(op.loop)
        else:
            return False
    except (KeyError, TypeError):
        return False
    return n % 2 == 1


def build_fill(op, ctx, s, size):
    """The drawing input: hud['fill'] = {n (None while searching), k (-1: none current), busy, word,
    edges_text, note, first, thumbs, phase ('legacy' / 'fallback')}."""
    from ..ops.fill import _rim_count
    from ..i18n import tr
    busy = (op.job is not None or getattr(op, 'background', None) is not None
            or getattr(op, 'triangle_job', None) is not None)
    fallback = bool(getattr(op, '_fallback_here', lambda: False)())
    edges = _rim_count(op.plan) if op.plan is not None else (f"{len(op.loop)} " + tr("edges") if op.loop is not None else '')
    edges_odd = _rim_odd(op)
    notes = []   # step 4: the New run still works after Legacy is done; then a new #1 is announced
    if getattr(op, '_step4_tail', lambda: False)():
        notes.append(tr('Improving fill preview...'))
    if op._better_note():
        notes.append(tr('Better result found'))
    note = ' · '.join(notes)
    stage = op._display_stage() if hasattr(op, '_display_stage') else 'done'
    if fallback:
        entries = list(getattr(op, 'newfill_entries', None) or ([op.plan] if op.plan is not None else []))
        plans, cur = entries, (0 if entries else None)
        busy = busy or not entries   # busy held across the hand-over until the fallback result arrives
    elif stage == 'waiting':
        plans, cur = [], None   # "… / …" and the spinner; no cells, no preview yet
    else:
        plans, cur = displayed_plans(op)
    n = len(plans)
    c = _cache(op)
    if c.get('scale') != s:   # the GRID row spacing is in screen pixels: a UI scale change rebuilds the thumbnails
        c['scale'], c['plans'] = s, {}
    if len(c['plans']) > 2 * n + VISIBLE:
        live = {id(p) for p in plans}
        c['plans'] = {k: v for k, v in c['plans'].items() if k in live}
    if not n:
        return dict(n=None, k=-1, busy=busy, word='', edges_text=edges, edges_odd=edges_odd, note=note, first=0, thumbs=[],
                    phase='fallback' if fallback else 'legacy', fallback=False)
    slots = min(n, VISIBLE)
    anchor = cur if cur is not None else 0
    first = 0 if n <= slots else min(max(0, anchor - slots // 2), n - slots)
    thumbs = [_thumb(op, c, plans[i]) for i in range(first, first + slots)]
    word = ''
    if cur is not None and not fallback:
        try:
            word = op._label_of(plans[cur], plans) or ''
        except Exception:   # noqa: BLE001
            pass
    return dict(n=None if (busy and not fallback) else n, k=cur if cur is not None else -1, busy=busy, word=word,
                edges_text=edges, edges_odd=edges_odd, note=note, first=first, thumbs=thumbs,
                phase='fallback' if fallback else 'legacy',
                fallback=bool(fallback and not busy))   # the fallback's result keeps a quiet marker


def draw_hud(op, ctx, xy, s, size):
    """The Fill HUD box (fill_hud_draw.draw(ctx, hud) with hud['fill']). Returns the input dict,
    or False on any error: the caller then falls back to the text label."""
    try:
        bbox, start = _preview(op)
        hud = dict(region=getattr(op, 'region', None) or (ctx.region.as_pointer() if ctx.region else 0),
                   mouse=(op.mouse.x, op.mouse.y), scale=s, size=size, bounds=_bounds(ctx, s),
                   preview_bbox=bbox, start_xy=start, fill=build_fill(op, ctx, s, size))
        from . import fill_hud_draw
        fill_hud_draw.draw(ctx, hud)
        return hud
    except ReferenceError:
        raise
    except Exception:   # noqa: BLE001  (the HUD must never break the Fill modal)
        if not _warned[0]:
            _warned[0] = True
            import traceback
            traceback.print_exc()
        return False

