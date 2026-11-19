# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""P1 cut candidates.

Scene gate: the cheap concavity gate of fill_cuts (a reflex turn below NECK_TURN);
convex and shallow outlines return at once. On a hit:
  * "no cut": the whole outline as one Takayama patch (P1 corner choice);
  * S  : one straight cut per reflex cluster (QuadWild extension, far end = rim vertex);
  * SS : two non-crossing straight cuts.
Each cut gets one shared integer segment count (both sides use the same samples);
every piece is filled with the add-on's Takayama patterns. The rim is never moved.
Ranking is untouched: candidates join the normal pool. numpy only.
"""
import itertools
import math
import numpy as np
from .patterns import geom

THR = 40.0            # deg: corner / reflex threshold
SNAP = 2              # far endpoint may move +-SNAP rim vertices from the ray hit
KMAX = 8              # max segments on one cut
CLUSTER_GAP = 2
MAX_FILL = 24         # cut layouts filled per call (plus the no-cut patch)
MAX_COMBOS = 80       # cut combinations examined per call
MAX_SINGLES = 24      # shortest single cuts kept
WALL = 2.0            # s (work clock): bound per Fill call
K_LAYOUTS = 4         # stop after this many valid pre-ranked cut layouts
EMIT = 50.0           # deg: a piece node turning below -EMIT keeps a reflex corner
CAP = .25             # s (work clock): one piece's Takayama search
STEP_SETS = 64           # best_piece_steps yields before every this many corner sets (no clock read: tick-neutral)
MAX_CORNER_SETS = 20000  # above this many corner sets for one piece ...
KEEP_WEAK = 12           # ... only the strongest weak turns stay corner candidates (<= 1507 sets)


def reflex_clusters(t, thr=THR):
    m = len(t)
    idx = [i for i in range(m) if t[i] < -thr]
    if not idx:
        return []
    groups = [[idx[0]]]
    for i in idx[1:]:
        if i-groups[-1][-1] <= CLUSTER_GAP:
            groups[-1].append(i)
        else:
            groups.append([i])
    if len(groups) > 1 and (groups[0][0]+m-groups[-1][-1]) <= CLUSTER_GAP:
        groups[0] = groups[-1]+groups[0]
        groups.pop()
    return groups


def _seg_hit(o, d, a, b):
    e = b-a
    den = d[0]*e[1]-d[1]*e[0]
    if abs(den) < 1e-14:
        return None
    w = a-o
    s = (w[0]*e[1]-w[1]*e[0])/den
    u = (w[0]*d[1]-w[1]*d[0])/den
    return (s, u) if s > 1e-9 and -1e-9 <= u <= 1+1e-9 else None


def _ray_rim(P, i, d):
    m = len(P)
    best = None
    for a in range(m):
        b = (a+1) % m
        if a == i or b == i:
            continue
        h = _seg_hit(P[i], d, P[a], P[b])
        if h and (best is None or h[0] < best[0]):
            best = (h[0], a, h[1])
    return best


def _orient(p, q, r):
    return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])


def _seg_inside(P, i, j):
    m = len(P)
    A, B = P[i], P[j]
    for a in range(m):
        b = (a+1) % m
        if len({a, b, i, j}) < 4:
            continue
        if _orient(A, B, P[a])*_orient(A, B, P[b]) < 0 and _orient(P[a], P[b], A)*_orient(P[a], P[b], B) < 0:
            return False
    mid = (A+B)/2
    inside = False
    for a in range(m):
        x1, y1 = P[a]
        x2, y2 = P[(a+1) % m]
        if (y1 > mid[1]) != (y2 > mid[1]) and mid[0] < (x2-x1)*(mid[1]-y1)/(y2-y1)+x1:
            inside = not inside
    return inside


def cut_candidates_steps(P, cluster):
    """Cooperative: yields None before every ray (no clock read); returns the sorted cut candidates."""
    m = len(P)
    r0, r1 = cluster[0], cluster[-1]
    out = set()
    for r, d in ((r0, P[r0]-P[r0-1]), (r1, P[r1]-P[(r1+1) % m]), (r0, P[r1]-P[(r1+1) % m]), (r1, P[r0]-P[r0-1])):
        for rr in sorted(set(cluster) | {r}):
            yield None   # F2b-1: a long reflex cluster (zig-zag rim) casts hundreds of O(n) rays
            h = _ray_rim(P, rr, d/(np.linalg.norm(d)+1e-300))
            if not h:
                continue
            _, a, u = h
            near = a if u < .5 else (a+1) % m
            for s in range(-SNAP, SNAP+1):
                j = (near+s) % m
                if min((j-rr) % m, (rr-j) % m) < 2:
                    continue
                if _seg_inside(P, rr, j):
                    out.add(tuple(sorted((int(rr), int(j)))))
    return sorted(out)


def _crosses(P, c1, c2):
    if len(set(c1) | set(c2)) < 4:
        return False
    a, b, c, d = P[c1[0]], P[c1[1]], P[c2[0]], P[c2[1]]
    return _orient(a, b, c)*_orient(a, b, d) < 0 and _orient(c, d, a)*_orient(c, d, b) < 0


def split_pieces(m, cuts):
    pieces = [list(range(m))]
    for i, j in cuts:
        for k, pc in enumerate(pieces):
            if i in pc and j in pc:
                a, b = sorted((pc.index(i), pc.index(j)))
                if b-a < 2 or (len(pc)-b+a) < 2:
                    return None
                pieces[k:k+1] = [pc[a:b+1], pc[b:]+pc[:a+1]]
                break
        else:
            return None
    return pieces


def _piece_geometry(P, ids, ks):
    pts, rim_of = [], []
    n, m = len(P), len(ids)
    for q in range(m):
        a, b = ids[q], ids[(q+1) % m]
        rim_of.append(len(pts))
        pts.append(P[a])
        if (a+1) % n != b:
            k = ks[tuple(sorted((a, b)))]
            for s in range(1, k):
                pts.append(P[a]*(1-s/k)+P[b]*s/k)
    return np.array(pts), rim_of


def _turning_deg(P):
    a = P-np.roll(P, 1, axis=0)
    b = np.roll(P, -1, axis=0)-P
    return np.degrees(np.arctan2(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0], (a*b).sum(1)))


def _side_counts(npts, corners):
    c = sorted(corners)
    return [(c[(i+1) % len(c)]-c[i]) % npts for i in range(len(c))]


def _cc_able(s):
    n, T = len(s), sum(s)
    if T % 2:
        return False
    if n == 3:
        return all(2*x < T for x in s)
    if n == 4:
        return s[0] == s[2] and s[1] == s[3]
    if n == 5:
        return all(s[i]+s[(i+1) % 5]+s[(i+2) % 5] > s[(i+3) % 5]+s[(i+4) % 5] for i in range(5))
    return False


def _size_dev(pts, corners, s):
    e = np.linalg.norm(np.roll(pts, -1, 0)-pts, axis=1)
    h = np.median(e)
    c = sorted(corners)
    dev = 0.
    for i, x in enumerate(s):
        L = sum(e[(c[i]+q) % len(pts)] for q in range(x))
        dev += (x/max(L/h, 1e-9)-1)**2
    return dev



def clear_caches():
    """Empty the solvability memo (add-on unregister)."""
    _SOLVABLE.clear()


_SOLVABLE = {}


def solvable(s):
    """count-level check against the add-on's 15 Takayama patterns (any rotation).
    Every rotation of s is tried, so the answer is rotation-invariant: it is kept per lexicographically smallest
    rotation. Its clock reads (geom.check_time) happen only under an active deadline."""
    s = list(s)
    key = min(tuple(s[r:] + s[:r]) for r in range(len(s)))
    if key not in _SOLVABLE:
        s = list(key)
        from .patterns import chords
        ok = False
        for pat in chords.library(True):
            if getattr(pat, 'n', None) != len(s):
                continue
            for r in range(len(s)):
                g = pat.solve_steps(tuple(s[r:]+s[:r]))
                try:
                    while True:
                        next(g)
                except StopIteration as done:
                    if done.value is not None:
                        ok = True
                if ok:
                    break
            if ok:
                break
        _SOLVABLE[key] = ok
    return _SOLVABLE[key]


def best_piece_steps(P, ids, ks, thr=THR):
    """P1 corner choice for one piece: corners are rim vertices only. Cooperative: yields None before every
    STEP_SETS corner sets (no clock read, so the work-unit clock is unchanged); returns the choice or None.
    The key is (not cc, abs(len(c)-4), d) and `best` is replaced only on a strict `<`. The prefix (not cc,
    abs(len(c)-4)) needs only the side counts: if it is strictly greater than the best prefix, the full key cannot be
    smaller, so solvable() and _size_dev() are skipped. The argmin and its tie order do not depend on the skip."""
    pts, rim_of = _piece_geometry(P, ids, ks)
    t = _turning_deg(pts)
    rim_set = set(rim_of)
    strong = [i for i in rim_of if t[i] > thr]
    weak = [i for i in rim_of if thr * .5 < t[i] <= thr]
    sets = sum(math.comb(len(weak), nc - len(strong)) for nc in (3, 4, 5) if 0 <= nc - len(strong) <= len(weak))
    if sets > MAX_CORNER_SETS:   # a zig-zag / fluted rim: millions of sets, seconds to a minute of work
        keep = set(sorted(weak, key=lambda i: (-t[i], i))[:KEEP_WEAK])
        weak = [i for i in weak if i in keep]   # rim order kept
    best = None
    npts = len(pts)
    tried = 0
    for nc in (3, 4, 5):
        extra = nc - len(strong)
        if extra < 0 or extra > len(weak):
            continue
        pre_len = abs(nc - 4)
        for combo in itertools.combinations(weak, extra):
            tried += 1
            if tried % STEP_SETS == 0:
                yield None   # F2b-1: a cancel point; Blender redraws between steps
            c = sorted(strong + list(combo))
            s = _side_counts(npts, c)
            if not (2 <= len(s) <= 6 and sum(s) % 2 == 0 and min(s) >= 1):
                continue
            cc = _cc_able(s)
            if best is not None and (not cc, pre_len) > best[0][:2]:
                continue                      # cannot beat the incumbent on the key prefix
            if not solvable(s):
                continue
            d = _size_dev(pts, c, s)
            key = (not cc, abs(len(c) - 4), d)
            if best is None or key < best[0]:
                inv = {k: ids[q] for q, k in enumerate(rim_of)}
                best = (key, dict(corners=[inv[i] for i in c if i in rim_set], cc=cc, dev=d, s=list(s), cpos=c, pts=pts))
    return None if best is None else best[1]


def _has_reflex(P, ids):
    """A3 emitter check: the piece keeps a reflex logical corner (turn < -EMIT)."""
    return bool(np.min(_turning_deg(np.asarray([P[i] for i in ids]))) < -EMIT)


def _closure(L):
    """Continuous CC-ability defect on side lengths (0 = closable)."""
    n, T = len(L), float(sum(L))
    if T <= 0:
        return 1.
    if n == 4:
        return (abs(L[0]-L[2])+abs(L[1]-L[3]))/T
    if n == 3:
        return sum(max(0., 2*x-T) for x in L)/T
    if n == 5:
        return sum(max(0., L[(i+3) % 5]+L[(i+4) % 5]-L[i]-L[(i+1) % 5]-L[(i+2) % 5]) for i in range(5))/T
    return .5


def _proxy(plan_pieces):
    """A3 pre-score of a layout from the continuous side lengths of its pieces."""
    sc = 0.
    for b in plan_pieces:
        pts, c = b['pts'], b['cpos']
        e = np.linalg.norm(np.roll(pts, -1, 0)-pts, axis=1)
        L = [float(sum(e[(c[i]+q) % len(pts)] for q in range(b['s'][i]))) for i in range(len(c))]
        sc += _closure(L)+.25*abs(len(c)-4)
    return sc


def plan_layouts(P, audit, started):
    """Cut layouts: [(seams, counts, pieces, corners)], no-cut first."""
    n = len(P)
    out = []
    whole = yield from best_piece_steps(P, list(range(n)), {})
    if whole is not None:
        out.append(((), (), [list(range(n))], [whole['corners']], 'none', (not whole['cc'], whole['dev'])))
    t = _turning_deg(P)
    clusters = reflex_clusters(t)
    audit['clusters'] = len(clusters)
    cand = []
    for cl in clusters:
        cand.append((yield from cut_candidates_steps(P, cl)))
    singles = sorted({c for cs in cand for c in cs}, key=lambda c: (float(np.linalg.norm(P[c[0]]-P[c[1]])), c))[:MAX_SINGLES]
    h = max(float(np.median(np.linalg.norm(np.roll(P, -1, axis=0)-P, axis=1))), 1e-12)
    combos = [(c,) for c in singles]
    for a, b in itertools.combinations(singles, 2):
        if len(set(a) | set(b)) == 4 and not _crosses(P, a, b):
            combos.append((a, b))
    combos = combos[:MAX_COMBOS]
    audit['combos'] = len(combos)
    cache = {}
    layouts = []
    for combo in combos:
        if geom.clock().perf_counter()-started > WALL*.5:
            audit['layout_budget'] = True
            break
        pieces = split_pieces(n, combo)
        if pieces is None:
            continue
        best = None
        ranges = []
        for a, b in combo:
            target = max(1, min(KMAX, round(float(np.linalg.norm(P[a]-P[b]))/h)))
            ranges.append(range(max(1, target-2), min(KMAX, target+2)+1))
        for kv in itertools.product(*ranges):
            ks = dict(zip(combo, kv))
            plan, ok, cc_all, dev, info = [], True, True, 0., []
            for ids in pieces:
                key = (tuple(ids), tuple(sorted((e, k) for e, k in ks.items() if set(e) <= set(ids))))
                if key not in cache:
                    cache[key] = yield from best_piece_steps(P, ids, ks)
                b = cache[key]
                if b is None:
                    ok = False
                    break
                plan.append(b['corners'])
                info.append(b)
                cc_all &= b['cc']
                dev += b['dev']
            if ok:
                rank = (not cc_all, dev)
                if best is None or rank < best[0]:
                    best = (rank, kv, plan, info)
        if best is not None:
            reflex = sum(_has_reflex(P, ids) for ids in pieces)
            layouts.append((combo, best[1], pieces, best[2], 'S' if len(combo) == 1 else 'SS', best[0],
                            (bool(reflex), _proxy(best[3])+reflex)))
        yield None
    layouts.sort(key=lambda L: (L[5], len(L[0]), L[0]))
    # the base's top layout is
    # filled first (quality guard); the rest go best-first by the continuous
    # pre-score, emitter-complete layouts before the rest, and filling stops
    # after K_LAYOUTS valid layouts (candidate_steps).
    chosen = layouts[:1]
    clean = [L for L in layouts if not L[6][0]]
    order = sorted(layouts, key=lambda L: (L[6][0] and len(clean) >= K_LAYOUTS, L[6][1], L[5], len(L[0]), L[0]))
    for L in order:
        if len(chosen) >= MAX_FILL:
            break
        if L not in chosen and not (L[6][0] and len(clean) >= K_LAYOUTS):
            chosen.append(L)
    audit['layouts'] = len(layouts)
    return out+[L[:6] for L in chosen]


def gate(p):
    """True when P1 should run (a reflex turn, neck, tail or fan root exists)."""
    from . import fill_cuts
    return not fill_cuts.gate(p)


def candidate_steps(points, settings=None, audit=None):
    """Cooperative generator; yields None or {'candidate': {...}} like fill_cuts."""
    from . import network_fill, boundary_preview
    from .patterns.validate import validate_steps, InvalidFill
    from .fill_partition import _capped_steps, _capped_site, PartitionError
    settings = settings or {}
    audit = {} if audit is None else audit
    floor = float(settings.get('_quality_floor', .25))
    started = geom.clock().perf_counter()
    points = np.asarray(points, float)
    n = len(points)
    if n < 8 or n > 128 or n % 2:
        audit['status'] = 'ineligible'
        return
    origin = points.mean(0)
    scale = float(np.ptp(points, axis=0).max())
    p = (points-origin)/scale
    if not gate(p):
        audit['gate'] = 'miss'
        return
    audit['gate'] = 'hit'
    layouts = yield from plan_layouts(p, audit, started)
    audit.update(emitted=0, failed=0, invalid=0)
    bad = set()                   # pieces (ids, cut counts, corners) that already failed are never searched twice
    valid = 0
    for seams, counts, pieces, corners, kind, _ in layouts:
        if seams and valid >= K_LAYOUTS+1:   # base top layout + K pre-ranked
            audit['k_stop'] = True
            break
        sig = [(tuple(pc), tuple((s, c) for s, c in zip(seams, counts) if set(s) <= set(pc)), tuple(cs))
               for pc, cs in zip(pieces, corners)]
        if any(x in bad for x in sig):
            audit['skipped'] = audit.get('skipped', 0)+1
            continue
        if geom.clock().perf_counter()-started > WALL:
            audit['status'] = 'budget'
            return
        parent, rings = network_fill._expand(p, [list(x) for x in pieces], list(seams), list(counts))
        positions = list(parent)
        quads, failed = [], False
        for ring, cs, key in zip(rings, corners, sig):
            local = parent[ring]
            planned = [('p', ring.index(c)) for c in cs]
            try:
                result = yield from _capped_steps(boundary_preview.job(
                    local, [('p', j) for j in range(len(local))],
                    dict(pattern='AUTO', _policy='legacy', _best_of=1, _takayama=True, _corners=planned),
                    site=_capped_site(CAP)), CAP)
            except (PartitionError, boundary_preview.PreviewError, geom.TimeUp):
                result = None
            if result is None:
                bad.add(key)
                failed = True
                break
            remap = ring+list(range(len(positions), len(positions)+len(result['positions'])-len(ring)))
            positions.extend(result['positions'][len(ring):])
            quads.extend(tuple(remap[v] for v in q) for q in result['quads'])
            yield None
        if failed:
            audit['failed'] += 1
            continue
        positions = np.asarray(positions)
        try:
            yield from validate_steps(p, positions, quads)
        except InvalidFill:
            audit['invalid'] += 1
            continue
        mm = network_fill.metrics(positions, quads)
        if mm['qmin'] < floor:
            audit['invalid'] += 1
            continue
        paths, k = [], n
        for (a, b), count in zip(seams, counts):
            paths.append((a,)+tuple(range(k, k+count-1))+(b,))
            k += count-1
        world = positions*scale+origin
        world[:n] = points
        audit['emitted'] += 1
        valid += bool(seams)
        audit.setdefault('kinds', []).append(kind)
        yield dict(candidate=dict(
            positions=world, quads=tuple(quads), status='filled', method='p1_cuts',
            internal_seams=tuple(seams), internal_counts=tuple(counts), internal_paths=paths,
            p1_kind=kind, **mm))
    audit['seconds'] = round(geom.clock().perf_counter()-started, 3)
