# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Fill: quad patches for open loops of an existing mesh (holes, wire outlines).

The loop is laid flat on its best-fit plane. AUTO first continues the surrounding
grid (core/fill_lattice), then tries a plain grid on balanced corners, then the
shared pattern engine (core/boundary_preview: Gmsh-style patterns, integer
chords, foldover-free placement), and can decompose deep concavities into
shared-edge quad patches (core/fill_partition). The patch is lifted back to 3D by harmonic heights,
or onto the retopology sources in Topo mode. An odd loop gets one temporary
midpoint on its longest edge; collapsing it afterwards leaves one triangle.
"""
import math
from collections import Counter
from itertools import combinations

import numpy as np

from . import boundary_preview, fill_lattice, fill_partition
from .patterns import geom

PATTERNS = boundary_preview.PATTERNS
SOLVE_SECONDS = 1.0
BEST_OF = 4          # AUTO compares up to this many valid template layouts
BALANCED_CANDIDATES = 10       # sharpest vertices tried as the corners of a plain grid
BALANCED_TRIES = 4             # best-ranked balanced corner sets placed
BALANCED_MIN_TURN = math.radians(10)   # a vertex turning less is no corner of a plain grid
BALANCED_MIN_QUALITY = .25     # a plain grid must keep its worst corner sine above this
BALANCED_MEAN_QUALITY = .75    # and its mean above this, else the templates decide
SPARE_MARGIN = .15             # templates with poles must beat a weaker plain grid by this
SPARE_MIN_QUALITY = .2         # a weaker plain grid is still kept as the alternative above this
SPARE_MEAN_QUALITY = .7
MAX_POINTS = boundary_preview.MAX_BOUNDARY


class FillError(ValueError):
    """str(exc) is the technical detail (logs, Show Debug Info); user_text(str(exc)) is the text for the user."""


# guard cases the user cannot act on read as one plain message unless Show Debug Info
PLAIN_FAILURE = "This outline cannot be filled; try another selection"
_TECHNICAL = ("Invalid ", "Loop too short", "Panel network: ")


# the shown wording of internal refusal messages (the raised texts stay unchanged)
_SHOWN = {
    'Fill corners no longer available': 'The outline changed; preview again',
    'Odd count': 'Odd number of edges: quads need an even count',
    'Preview boundary limit (256 points)': 'Outline too long to preview (max 256 edges)',
    'No valid pattern for this point plan': 'No pattern fits these points',
    'No valid pattern above policy quality floor': 'No pattern fits these points',
    'Loop too long to fill': 'Outline too long to fill (max 256 edges)',
    'This loop is already filled': 'This outline is already filled',
    'Select one hole, or panels that share edges': 'Select one outline, or panels that share edges',
    'Panel network would create a non-manifold seam': 'These panels would give a seam edge more than two faces',
}


def user_text(message, debug=False):
    """The Fill error text to show: the plain message for a technical guard case, unless debug.
    only a text with its own translation entry (or a _SHOWN wording) is shown; any other text (an
    internal exception's) reads as PLAIN_FAILURE."""
    if debug or not message:
        return message
    if str(message) in _SHOWN:
        return _SHOWN[str(message)]
    if str(message).startswith(_TECHNICAL):
        return PLAIN_FAILURE
    from .. import i18n
    return message if str(message) in i18n.ZH_HANT or str(message) in i18n.TERMS else PLAIN_FAILURE


# ------------------------------------------------------------------ loops

def _open(edge):
    return not edge.hide and not any(v.hide for v in edge.verts) and len(edge.link_faces) <= 1


def _cycle_blocks(edges):
    """Iterative biconnected blocks: dangling wires cannot hide a closed outline."""
    links = {}
    for i, edge in enumerate(edges):
        a, b = edge.verts
        links.setdefault(a, []).append((b, i))
        links.setdefault(b, []).append((a, i))
    discovery, low, parent, pending = {}, {}, {}, []
    for root in links:
        if root in discovery:
            continue
        discovery[root] = low[root] = len(discovery)
        stack = [(root, iter(links[root]))]
        while stack:
            vertex, neighbors = stack[-1]
            try:
                other, edge_id = next(neighbors)
            except StopIteration:
                stack.pop()
                if vertex in parent:
                    previous, parent_edge = parent[vertex]
                    low[previous] = min(low[previous], low[vertex])
                    if low[vertex] >= discovery[previous]:
                        block = []
                        while pending:
                            item = pending.pop()
                            block.append(edges[item])
                            if item == parent_edge:
                                break
                        if len(block) >= 3:
                            yield block
                continue
            if vertex in parent and edge_id == parent[vertex][1]:
                continue
            if other not in discovery:
                parent[other] = (vertex, edge_id)
                discovery[other] = low[other] = len(discovery)
                pending.append(edge_id)
                stack.append((other, iter(links[other])))
            elif discovery[other] < discovery[vertex]:
                low[vertex] = min(low[vertex], discovery[other])
                pending.append(edge_id)


def _cycles(edges):
    """Closed simple vertex cycles of an edge set (components with a branch are skipped)."""
    links = {}
    for e in edges:
        a, b = e.verts
        links.setdefault(a, []).append(b)
        links.setdefault(b, []).append(a)
    seen, loops = set(), []
    for start in links:
        if start in seen:
            continue
        component, stack = [], [start]
        seen.add(start)
        while stack:
            v = stack.pop()
            component.append(v)
            for w in links[v]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        if len(component) < 3 or any(len(links[v]) != 2 for v in component):
            continue
        loop, prev = [component[0]], None
        while True:
            a, b = links[loop[-1]]
            nxt = b if a is prev else a
            if nxt is loop[0]:
                break
            prev = loop[-1]
            loop.append(nxt)
        if len(loop) == len(component):
            loops.append(loop)
    return loops


def _planar_faces(edges):
    """Regions enclosed by a branched block of open edges, e.g. a wire outline split
    by a strip of faces: walk the block's planar embedding (best-fit plane) and
    return every bounded face as a simple vertex cycle."""
    links = {}
    for e in edges:
        a, b = e.verts
        links.setdefault(a, []).append(b)
        links.setdefault(b, []).append(a)
    verts = list(links)
    co = np.array([tuple(v.co) for v in verts])
    centre, u, v, _n = _frame(co)
    xy = {vert: (float((c - centre) @ u), float((c - centre) @ v)) for vert, c in zip(verts, co)}
    around = {}
    for a, ns in links.items():
        ax, ay = xy[a]
        around[a] = sorted(ns, key=lambda b: math.atan2(xy[b][1] - ay, xy[b][0] - ax))
    used, faces = set(), []
    for a in verts:
        for b in links[a]:
            if (a, b) in used:
                continue
            cycle, x, y = [], a, b
            while (x, y) not in used and len(cycle) <= len(edges):
                used.add((x, y))
                cycle.append(x)
                ring = around[y]
                x, y = y, ring[ring.index(x) - 1]      # next edge clockwise from the arrival
            if (x, y) != (a, b) or len(cycle) < 3 or len(set(cycle)) != len(cycle):
                continue
            pts = [xy[c] for c in cycle]
            area = sum(pts[i - 1][0] * pts[i][1] - pts[i][0] * pts[i - 1][1] for i in range(len(pts)))
            if area > 1e-12:                          # bounded faces wind one way, the outside the other
                faces.append(cycle)
    return faces


class ClosedLoop(list):
    """2.6.2: a loop closed through an existing edge (2.6.3: or a straight run of them; see closing_chain);
    a cache only."""


def _path(edges):
    """The vertices of a simple open path (2 ends, at least 2 edges) in order, else None."""
    links = {}
    for e in edges:
        a, b = e.verts
        links.setdefault(a, []).append(b)
        links.setdefault(b, []).append(a)
    if len(edges) < 2 or any(len(ns) > 2 for ns in links.values()):
        return None
    ends = [v for v, ns in links.items() if len(ns) == 1]
    if len(ends) != 2:
        return None
    path, prev = [ends[0]], None
    while len(path) <= len(links):
        nxt = [w for w in links[path[-1]] if w is not prev]
        if not nxt:
            break
        prev = path[-1]
        path.append(nxt[0])
    return path if len(path) == len(links) and path[-1] is ends[1] else None


def _rim_edges(loop):
    out = []
    for a, b in zip(loop, loop[1:] + loop[:1]):
        out.append(next((e for e in a.link_edges if e.other_vert(a) is b), None))
    return out


def _covers(loop, chain):
    """target guard: a face on the rim (open rim edges and the closing chain's edges, 2.6.3) that is
    nearly coplanar with the loop and lies inside it means the loop would cover existing geometry."""
    from mathutils import geometry
    co = np.array([tuple(v.co) for v in loop], dtype=np.float64)
    centre, u, v, n = _frame(co)
    xy = [(float((p - centre) @ u), float((p - centre) @ v)) for p in co]
    margin = 1e-4 * float(np.linalg.norm(co.max(axis=0) - co.min(axis=0)))
    rim = set(loop)
    chain = set(chain)
    # the loop's orientation decides each chain edge's interior side
    turning = sum(xy[i - 1][0] * xy[i][1] - xy[i][0] * xy[i - 1][1] for i in range(len(xy)))
    sides = {}   # chain edge -> (index of its first vertex in loop order)
    faces = set()
    for i, edge in enumerate(_rim_edges(loop)):
        if edge is None:
            continue
        if edge in chain:
            sides[edge] = i
        faces.update(edge.link_faces)
    for face in faces:   # hidden faces count too
        fn = np.array(tuple(geometry.normal([w.co for w in face.verts])), dtype=np.float64)
        if abs(float(fn @ n)) < .5:
            continue
        to2 = lambda q: (float((np.array(tuple(q), dtype=np.float64) - centre) @ u),
                         float((np.array(tuple(q), dtype=np.float64) - centre) @ v))
        fxy = [to2(w.co) for w in face.verts]
        probes = ([to2(face.calc_center_median())] + [to2(w.co) for w in face.verts if w not in rim]
                  + [((fxy[j - 1][0] + fxy[j][0]) * .5, (fxy[j - 1][1] + fxy[j][1]) * .5) for j in range(len(fxy))])
        if any(_inside(pt, xy) and _edge_distance(pt, xy) > margin for pt in probes):
            return True   # the face (partly) lies inside the loop
        own = set(face.verts)
        mids = [((xy[i - 1][0] + xy[i][0]) * .5, (xy[i - 1][1] + xy[i][1]) * .5) for i in range(len(xy))]
        if any(_inside(pt, fxy) and _edge_distance(pt, fxy) > margin
               for pt in [pt for w, pt in zip(loop, xy) if w not in own] + mids):
            return True   # the loop (partly) lies inside the face
        if any(_crosses(xy[i - 1], xy[i], fxy[j - 1], fxy[j], margin)
               for i in range(len(xy)) for j in range(len(fxy))):
            return True   # their outlines cross
        ring = list(face.verts)
        for edge, i in sides.items():
            if face not in edge.link_faces:
                continue
            # side test: across each chain edge the loop interior must lie opposite the face. the
            # interior side comes from the loop's orientation, the face's side from its own
            # winding in the same frame (exact for n-gons, reflex corners and noisy
            # collinear vertices): a face lies left of the edge it runs forward along when its signed area is
            # positive, right of it when negative
            va, vb = loop[i], loop[(i + 1) % len(loop)]
            k = ring.index(va)
            forward = ring[(k + 1) % len(ring)] is vb
            area = sum(fxy[j - 1][0] * fxy[j][1] - fxy[j][0] * fxy[j - 1][1] for j in range(len(fxy)))
            if area == 0.:
                continue
            if ((area > 0) == forward) == (turning > 0):
                return True   # the face lies on the loop's side of the chain edge
    return False


def _crosses(a, b, c, d, eps):
    """Proper crossing of segments ab and cd (shared or touching ends do not count)."""
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    scale = eps * max(math.hypot(b[0] - a[0], b[1] - a[1]), math.hypot(d[0] - c[0], d[1] - c[1]), 1e-30)
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return ((o1 > scale and o2 < -scale) or (o1 < -scale and o2 > scale)) and            ((o3 > scale and o4 < -scale) or (o3 < -scale and o4 > scale))


def _edge_distance(pt, poly):
    best = float('inf')
    for i in range(len(poly)):
        ax, ay = poly[i - 1]; bx, by = poly[i]
        dx, dy = bx - ax, by - ay
        t = 0. if dx == dy == 0 else max(0., min(1., ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / (dx * dx + dy * dy)))
        best = min(best, math.hypot(pt[0] - ax - t * dx, pt[1] - ay - t * dy))
    return best


def _usable(edge):
    """an existing edge a loop may close through: exactly 2 faces, it and its vertices visible."""
    return (not _open(edge) and len(edge.link_faces) == 2 and not edge.hide
            and not any(v.hide for v in edge.verts))


def _chain_tol(a, b):
    """straightness tolerance: relative to the chain's length, floored for float32 coordinates."""
    pa, pb = np.array(tuple(a.co), dtype=np.float64), np.array(tuple(b.co), dtype=np.float64)
    return max(1e-4 * float(np.linalg.norm(pb - pa)), 1e-6 * float(max(np.abs(pa).max(), np.abs(pb).max())))


def _chain_param(a, b, w, tol):
    """Parameter t of w's projection on the segment ab, when w lies within tol of that segment; else None."""
    pa, pb, pw = (np.array(tuple(x.co), dtype=np.float64) for x in (a, b, w))
    d = pb - pa
    length2 = float(d @ d)
    if length2 == 0.:
        return None
    t = float((pw - pa) @ d) / length2
    if not 0. <= t <= 1.:
        return None
    return t if float(np.linalg.norm(pa + d * t - pw)) <= tol else None


def _walk_chain(a, b, path):
    """the intermediate vertices of the unique straight run of existing edges from a to b
    (each step to exactly one candidate strictly farther along the segment ab), else None."""
    tol = _chain_tol(a, b)
    current, reached, middle = a, 0., []
    for _ in range(64):
        candidates = []
        for edge in current.link_edges:
            if _open(edge):
                continue
            w = edge.other_vert(current)
            if w is not b and w in path:
                continue
            t = _chain_param(a, b, w, tol)
            if t is not None and t > reached:
                candidates.append((edge, w, t))
        if len(candidates) != 1:
            return None
        edge, w, t = candidates[0]
        if not _usable(edge):
            return None
        if w is b:
            return middle
        if any(_open(x) for x in w.link_edges):
            return None   # a wire or border from a chain vertex
        middle.append(w)
        current, reached = w, t
    return None


def closing_chain(loop):
    """the rim edges of the loop's single contiguous run of non-open edges, when every other rim
    edge is open (at least 2), every chain edge is usable, the run is straight and advances monotonically
    from its first to its last vertex, its middle vertices have no open edges, and the loop does not cover
    existing geometry; else None. A run of one edge is exactly 2.6.2's closing edge. Derived from the
    mesh, so any copy of a loop gives the same answer."""
    n = len(loop)
    if n < 3 or len(set(loop)) != n:
        return None
    rim = _rim_edges(loop)
    if any(edge is None for edge in rim):
        return None
    closed = [not _open(edge) for edge in rim]
    starts = [i for i in range(n) if closed[i] and not closed[i - 1]]
    if len(starts) != 1 or n - sum(closed) < 2:
        return None
    start, count = starts[0], sum(closed)
    chain = tuple(rim[(start + k) % n] for k in range(count))
    if not all(closed[(start + k) % n] for k in range(count)) or not all(_usable(e) for e in chain):
        return None
    a, b = loop[start], loop[(start + count) % n]
    tol, reached = _chain_tol(a, b), 0.
    for k in range(1, count):
        w = loop[(start + k) % n]
        t = _chain_param(a, b, w, tol)
        if t is None or t <= reached or t >= 1. or any(_open(x) for x in w.link_edges):
            return None
        reached = t
    if _covers(loop, chain):
        return None
    return chain


def _chain_middles(loop, chain):
    """{middle chain vertex: (first chain vertex, last chain vertex, t along them)} for UV."""
    if len(chain) < 2:
        return {}
    rim = _rim_edges(loop)
    n = len(loop)
    start = next(i for i in range(n) if rim[i] in chain and rim[i - 1] not in chain)
    a, b = loop[start], loop[(start + len(chain)) % n]
    pa, pb = np.array(tuple(a.co), dtype=np.float64), np.array(tuple(b.co), dtype=np.float64)
    d = pb - pa
    out = {}
    for k in range(1, len(chain)):
        w = loop[(start + k) % n]
        out[w] = (a, b, float((np.array(tuple(w.co), dtype=np.float64) - pa) @ d) / float(d @ d))
    return out


def _closable(path):
    """the loop an open path closes into through existing edges (one edge between its ends, or a
    straight run of them), before the target guard; else None."""
    a, b = path[0], path[-1]
    shared = next((e for e in a.link_edges if e.other_vert(a) is b), None)
    if shared is not None:
        return path if not _open(shared) else None
    middle = _walk_chain(a, b, set(path))
    return None if middle is None else path + middle[::-1]


def open_loops(bm):
    """Every closed loop of open edges (border edges, or wire edges drawn as an outline).
    Where open edges branch (an outline crossed by faces), each enclosed region counts.
    an open path whose ends share an existing 2-face edge is closed through it."""
    edges = [e for e in bm.edges if _open(e)]
    loops = []
    for block in _cycle_blocks(edges):
        degree = Counter(v for e in block for v in e.verts)
        loops.extend(_cycles(block) if all(d == 2 for d in degree.values()) else _planar_faces(block))
    for component in _components(edges):
        path = _path(component)
        loop = _closable(path) if path is not None else None   # or a straight run of edges
        if loop is not None and closing_chain(loop) is not None:
            loops.append(ClosedLoop(loop))
    return loops


def _components(edges):
    links = {}
    for e in edges:
        for v in e.verts:
            links.setdefault(v, []).append(e)
    seen, out = set(), []
    for e in edges:
        if e in seen:
            continue
        stack, comp = [e], []
        seen.add(e)
        while stack:
            x = stack.pop()
            comp.append(x)
            for v in x.verts:
                for y in links[v]:
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
        out.append(comp)
    return out


def world_points(mw, verts):
    """World positions in double precision: mathutils products are single
    precision, which crushes an outline's shape on objects far from the origin."""
    m = np.array(mw, dtype=np.float64)
    co = np.array([tuple(v.co) for v in verts], dtype=np.float64)
    return co @ m[:3, :3].T + m[:3, 3]


def _filled_loop(loop):
    common = set(loop[0].link_faces)
    for vertex in loop[1:]:
        common.intersection_update(vertex.link_faces)
    return any(set(face.verts) == set(loop) for face in common)


def selected_outline(bm):
    """Selection diagnostic; never guess a closing edge or bridge two open chains."""
    edges = [e for e in bm.edges if e.select and not e.hide]
    if not edges:
        return None, ""
    # an open path, alone or with the existing edge between its ends, closes through that edge
    full = [e for e in edges if not _open(e)]
    path = _path([e for e in edges if _open(e)])
    loop = _closable(path) if path is not None else None   # or a straight run of edges
    if loop is not None and (not full or set(full) == set(_rim_edges(loop)[len(path) - 1:])):
        if closing_chain(loop) is not None:
            return ClosedLoop(loop), ""
        return None, "Select one closed loop without branches"   # closable shape refused by the guard
    if full:
        return None, "Select only open edges (at most one face each)"
    loops = _cycles(edges)
    if len(loops) == 1 and len(loops[0]) == len(edges):
        if _filled_loop(loops[0]):
            return None, "This loop is already filled"
        return loops[0], ""
    degree = Counter(v for edge in edges for v in edge.verts)
    if any(d == 1 for d in degree.values()):
        return None, "Close the selected outline before filling"
    return None, "Select one closed loop without branches"


def _inside(xy, poly):
    x, y = xy
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-30) + xi:
            inside = not inside
        j = i
    return inside


def _area(poly):
    return abs(sum(poly[i][0] * poly[i - 1][1] - poly[i - 1][0] * poly[i][1] for i in range(len(poly)))) * .5


def screen_loop(context, mw, loop):
    from . import raycast
    pts = [raycast.to_screen(context, mw @ v.co) for v in loop]
    return None if any(p is None for p in pts) else [(p.x, p.y) for p in pts]


def is_hole(context, mw, loop, poly):
    """A loop whose neighbouring faces lie inside it is an outer border, not a hole.
    a loop closed through an existing edge passed the in-plane target guard instead."""
    from . import raycast
    if closing_chain(loop) is not None:
        return True
    for v in loop:
        for f in v.link_faces:
            c = raycast.to_screen(context, mw @ f.calc_center_median())
            if c is not None and _inside((c.x, c.y), poly):
                return False
    return True


def pick(context, mw, loops, chosen, xy):
    """The loop to fill under the cursor: the selected loop first, then the smallest hole."""
    if chosen is not None:
        return chosen
    best = None
    for loop in loops:
        poly = screen_loop(context, mw, loop)
        if poly is None or not _inside((xy.x, xy.y), poly):
            continue
        if not is_hole(context, mw, loop, poly):
            continue
        area = _area(poly)
        if best is None or area < best[0]:
            best = (area, loop)
    return best[1] if best else None


# ------------------------------------------------------------------ solve

def _frame(points):
    centre = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centre)
    u, v, n = vt[0], vt[1], vt[2]
    return centre, u, v, n


def face_turns(loop):
    """Quarter turns of a hole read from the quads around it: a border vertex with 2
    faces goes straight, 3 is a convex corner of the hole (+1), 1 a reflex one (-1).
    None for wire outlines or when any neighbouring face is not a quad (and for a loop
    closed through an existing edge, whose faces across that edge are not around the hole)."""
    if closing_chain(loop) is not None:
        return None
    turns = []
    for vertex in loop:
        faces = [f for f in vertex.link_faces if not f.hide]
        if not faces or any(len(f.verts) != 4 for f in faces):
            return None
        turns.append(len(faces) - 2)
    return tuple(turns)


def _turns_in(order, turns):
    """Face turns in solved order (a temporary midpoint goes straight)."""
    if turns is None:
        return None
    return [0 if source is None else turns[source] for source in order]


def _balanced_steps(xy, refs, template):
    """A plain grid on four corners whose opposite sides have equal point counts.

    Strips, bent tubes and near-rectangles have such a choice among their sharpest
    vertices; it gives a pole-free grid that stays the same while the outline is
    nudged. Corner sets are ranked by how close their turns are to 90 degrees (a
    continuous score, no thresholds), and a grid is kept only when its quads are
    reasonably square; otherwise the templates decide."""
    n = len(xy)
    if n < 6:
        return None, None
    turn = geom.turning(np.asarray(xy, dtype=np.float64))
    candidates = sorted(int(i) for i in np.argsort(-turn, kind="stable")[:BALANCED_CANDIDATES])
    ranked = []
    for cs in combinations(candidates, 4):
        s = [(cs[(k + 1) % 4] - cs[k]) % n for k in range(4)]
        if s[0] == s[2] and s[1] == s[3] and min(float(turn[c]) for c in cs) >= BALANCED_MIN_TURN:
            ranked.append((round(sum(abs(float(turn[c]) - math.pi / 2) for c in cs), 9), cs))
    yield None
    from .patterns.validate import validate_steps, InvalidFill
    boundary = np.asarray(xy, dtype=np.float64)
    origin = boundary.mean(axis=0)
    scale = max(float(np.ptp(boundary, axis=0).max()), 1e-30)
    unit = (boundary - origin) / scale
    spare = None
    for _score, cs in sorted(ranked)[:BALANCED_TRIES]:
        positions, quads = _ladder_grid(unit, cs)
        if geom.min_corner_area(positions, quads) <= 0:
            continue                    # these corners do not suit the outline
        positions = yield from geom.winslow_steps(positions, n, quads)
        quality = geom.quad_quality(positions, quads)
        if quality.min() < SPARE_MIN_QUALITY or quality.mean() < SPARE_MEAN_QUALITY:
            continue
        try:
            metrics = yield from validate_steps(unit, positions, quads)
        except InvalidFill:
            continue
        positions = positions * scale + origin
        positions[:n] = boundary
        result = dict(positions=positions, quads=tuple(quads), pattern="quad", rotation=0,
                      corners=tuple(cs), anchors=[refs[i] for i in cs], metrics=metrics,
                      score=float(quality.mean()))
        boundary_preview.fill_selection.checkpoint(template, result)
        if quality.min() >= BALANCED_MIN_QUALITY and quality.mean() >= BALANCED_MEAN_QUALITY:
            return result, None
        if spare is None or quality.min() > spare[0]:
            spare = (float(quality.min()), result)
    return None, (spare[1] if spare else None)


def _ladder_grid(unit, cs):
    """Structured grid between four corners with opposite sides of equal counts,
    placed by transfinite interpolation with arc-length parameters: rows follow the
    two long sides like the rungs of a ladder, which keeps bent strips unfolded
    (a uniform Laplace start folds on the inner side of a tight bend)."""
    n = len(unit)
    a, b = (cs[1] - cs[0]) % n, (cs[2] - cs[1]) % n
    bottom = unit[[(cs[0] + i) % n for i in range(a + 1)]]
    right = unit[[(cs[1] + j) % n for j in range(b + 1)]]
    top = unit[[(cs[2] + a - i) % n for i in range(a + 1)]]
    left = unit[[(cs[3] + b - j) % n for j in range(b + 1)]]

    def params(side):
        d = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(side, axis=0), axis=1))))
        return d / max(float(d[-1]), 1e-30)
    ub, ut, vl, vr = params(bottom), params(top), params(left), params(right)
    index = {}
    for i in range(a + 1):
        index[i, 0] = (cs[0] + i) % n
        index[i, b] = (cs[2] + a - i) % n
    for j in range(b + 1):
        index[0, j] = (cs[3] + b - j) % n
        index[a, j] = (cs[1] + j) % n
    positions = list(unit)
    for i in range(1, a):
        for j in range(1, b):
            v = (vl[j] + vr[j]) / 2
            u = (1 - v) * ub[i] + v * ut[i]
            v = (1 - u) * vl[j] + u * vr[j]
            p = ((1 - v) * bottom[i] + v * top[i] + (1 - u) * left[j] + u * right[j]
                 - ((1 - u) * (1 - v) * bottom[0] + u * (1 - v) * bottom[a] + (1 - u) * v * top[0] + u * v * top[a]))
            index[i, j] = len(positions)
            positions.append(p)
    quads = [(index[i, j], index[i + 1, j], index[i + 1, j + 1], index[i, j + 1])
             for i in range(a) for j in range(b)]
    return np.asarray(positions, dtype=np.float64), quads


def _planar_steps(xy, refs, settings, turns=None):
    """AUTO order: a pole-free lattice grid when the surrounding quads or the outline's
    corners balance; then a plain grid on balanced corners; then the templates, with
    time reserved for decomposition on deep concavities."""
    if boundary_preview.fill_selection.enabled(settings):
        return (yield from boundary_preview.fill_selection.planar_steps(xy, refs, settings, turns))
    template = {k: v for k, v in settings.items() if k != "_face_turns"}
    spare = None
    if fill_lattice.eligible(settings):
        grid = yield from fill_lattice.plan_steps(xy, turns)
        if grid is not None:
            boundary_preview.fill_selection.checkpoint(settings, grid)
            return grid
        grid, spare = yield from _balanced_steps(xy, refs, template)
        if grid is not None:
            boundary_preview.fill_selection.checkpoint(settings, grid)
            return grid
        template["_best_of"] = BEST_OF     # AUTO: compare a few valid layouts
    if spare is None:
        return (yield from _template_steps(xy, refs, settings, template))
    # A pole-free grid that missed the quality bar is kept as the alternative: it is
    # used when the templates fail, or when they only manage with poles and are not
    # clearly better shaped.
    try:
        result = yield from _template_steps(xy, refs, settings, template)
    except (geom.TimeUp, boundary_preview.PreviewError):
        return spare
    if _interior_poles(result) and _worst(result) < _worst(spare) + SPARE_MARGIN:
        return spare
    return result


def _worst(result):
    return float(geom.quad_quality(np.asarray(result["positions"], dtype=np.float64)[:, :2], result["quads"]).min())


def _interior_poles(result):
    quads = [tuple(q) for q in result["quads"]]
    edges = Counter(tuple(sorted(e)) for q in quads for e in zip(q, q[1:] + q[:1]))
    boundary = {v for e, count in edges.items() if count == 1 for v in e}
    valence = Counter(v for e in edges for v in e)
    return sum(1 for v, d in valence.items() if v not in boundary and d != 4)


def _template_steps(xy, refs, settings, template):
    """Existing templates first, with time reserved for decomposition on deep
    concavities."""
    if not fill_partition.eligible(xy, settings):
        return (yield from boundary_preview.job([tuple(p) for p in xy], refs, template,
                                                site=('fill._template_steps', 'direct')))
    # Count CPU execution only, not time between timer events. The nested deadline
    # is always removed before yielding back to Blender.
    quick = boundary_preview.job([tuple(p) for p in xy], refs, dict(settings),
                                 site=('fill._template_steps', 'quick', repr(0.0)))   # spent = 0.0 at its first step
    spent = 0.0
    try:
        while spent < .08:
            started = geom.clock().monotonic()
            try:
                with geom.time_budget(min(.08-spent, geom.time_left())):
                    next(quick)
            except StopIteration as done:
                return done.value
            except (geom.TimeUp, boundary_preview.PreviewError):
                break
            spent += geom.clock().monotonic()-started
            yield None
    finally:
        quick.close()
    geom.check_time()
    try:
        if boundary_preview.fill_selection.enabled(settings):
            result = yield from fill_partition.solve_steps(
                xy, takayama=boundary_preview.chords.takayama_enabled(settings),
                policy_settings=boundary_preview.fill_selection.child_settings(settings))
            if boundary_preview.fill_selection.choose([result], settings) is None:
                raise fill_partition.PartitionError("Partition below policy quality floor")
            return result
        if boundary_preview.chords.takayama_enabled(settings):
            return (yield from fill_partition.solve_steps(xy, takayama=True))
        return (yield from fill_partition.solve_steps(xy))
    except fill_partition.PartitionError:
        # A failed partition does not bypass existing geometric checks.
        geom.check_time()
        return (yield from boundary_preview.job([tuple(p) for p in xy], refs, template,
                                                site=('fill._template_steps', 'fallback', repr(spent))))


def solve_steps(world, settings):
    """Fill plan for an ordered loop of world points.

    settings: pattern / rotation / anchors (as for boundary_preview.job).
    Returns dict(positions=(total, 3) world, quads, boundary=n of real loop points,
    virtual=index of the temporary midpoint or None, collapse=its real neighbour,
    order=loop indices in solved order, pattern, rotation, corners, anchors)."""
    pts = np.asarray(world, dtype=np.float64)
    n = len(pts)
    if n < 3:
        raise FillError("Loop too short")
    if n + n % 2 > MAX_POINTS:
        raise FillError("Loop too long to fill")
    if pts.shape != (n, 3) or not np.isfinite(pts).all():
        raise FillError("Invalid boundary")
    centre, u, v, normal, xy, pts, order = _planar(pts)
    if n == 3:
        plan = dict(positions=pts.copy(), quads=[(0, 1, 2)], boundary=3,
                    virtual=None, collapse=None, order=order, normal_ccw=np.cross(u, v),
                    pattern="triangle", rotation=0, corners=(0, 1, 2),
                    anchors=[("p", i) for i in range(3)], shift=0, layout=None)
        validate_plan(plan)
        return plan
    return (yield from _solve_rest(pts, xy, centre, u, v, normal, order, settings))


def _planar(pts):
    """The solver's plane: (centre, u, v, normal, xy, pts, order); the loop counter-clockwise in (u, v).
    Raises FillError for an edge-on / degenerate loop."""
    n = len(pts)
    centre, u, v, normal = _frame(pts)
    # The same outline must plan the same way however the object is moved or turned:
    # fix the sign of each in-plane axis by the skew of the points along it, and drop
    # coordinate noise far below the outline's size (it reorders near-equal choices).
    for axis in (u, v):
        if float(np.sum(((pts - centre) @ axis) ** 3)) < 0:
            axis *= -1
    normal = np.cross(u, v)
    xy = np.column_stack(((pts - centre) @ u, (pts - centre) @ v))
    size = float(f"{max(float(np.ptp(xy, axis=0).max()), 1e-30):.6g}")     # the same for every placement
    xy = np.round(xy / size, 9) * size
    area = float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1))) * .5
    order = list(range(n))
    if area < 0:                       # the engine wants a counter-clockwise loop
        order.reverse()
    xy, pts = xy[order], pts[order]
    scale = max(float(np.ptp(xy, axis=0).max()), 1e-30)
    if abs(area) <= scale * scale * 1e-12:
        raise FillError("This outline encloses no area; select another one")
    return centre, u, v, normal, xy, pts, order


def _solve_rest(pts, xy, centre, u, v, normal, order, settings):
    """solve_steps after the planar frame (n >= 4)."""
    n = len(pts)
    face_turns = settings.get("_face_turns")
    if face_turns is not None and len(face_turns) != n:
        face_turns = None
    virtual = collapse = None
    if n % 2:                          # one temporary midpoint on the longest edge
        lengths = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        k = int(np.argmax(lengths))
        if fill_lattice.eligible(settings):
            # on a rectilinear outline, the edge whose midpoint balances the sides
            balanced = fill_lattice.odd_split(xy, lengths, _turns_in(order, face_turns))
            k = k if balanced is None else balanced
        if settings.get("_triangle_edge") is not None:
            edge = int(settings["_triangle_edge"])
            if not 0 <= edge < n:
                raise FillError("Invalid triangle site")
            k = next(i for i in range(n)
                     if {order[i], order[(i + 1) % n]} == {edge, (edge + 1) % n})
        xy = np.insert(xy, k + 1, (xy[k] + xy[(k + 1) % n]) / 2, axis=0)
        pts = np.insert(pts, k + 1, (pts[k] + pts[(k + 1) % n]) / 2, axis=0)
        order = order[:k + 1] + [None] + order[k + 1:]
        virtual, collapse = k + 1, k
    shift = 0   # the rim keeps its own start
    refs = [("p", i) for i in range(len(xy))]
    if boundary_preview.fill_selection.recovery_enabled(settings):
        settings = dict(settings, _validated_pool=[])
    job = _planar_steps(xy, refs, settings, _turns_in(order, face_turns))
    spent = 0.0
    try:
        while True:
            started = geom.clock().monotonic()
            # Do not leave the module-wide geometry deadline installed across a
            # yield. The budget counts computation, not time waiting for UI input.
            with geom.time_budget(max(0.0, SOLVE_SECONDS - spent)):
                next(job)
            spent += geom.clock().monotonic() - started
            if spent > SOLVE_SECONDS:
                raise geom.TimeUp("time budget used up")
            yield None
    except StopIteration as done:
        result = done.value
    except geom.TimeUp:
        result = boundary_preview.fill_selection.best_checkpoint(settings)
        if result is None:
            raise FillError("No fill found in time")
    except boundary_preview.PreviewError as exc:
        raise FillError(str(exc))
    finally:
        job.close()
        capture = settings.get('_legacy_capture')
        if capture is not None:
            capture.append(dict(centre=centre, u=u, v=v, normal=normal, pts=pts,
                                order=order, virtual=virtual, collapse=collapse, shift=shift,
                                pool=list(settings.get('_validated_pool', []))))
    return (yield from _finish_steps(result, xy, pts, centre, u, v, normal, virtual, collapse, order, shift))


def _finish_steps(result, xy, pts, centre, u, v, normal, virtual, collapse, order, shift):
    """The chosen layout to a Fill plan: final polish, heights, world space, validation."""
    m = len(xy)
    # Final polish of the chosen layout only (Fill): elliptic smoothing
    # straightens grid lines and grades spacing from the border. It returns the
    # input unchanged if it would fold a quad or lower the worst quad quality.
    plane = yield from geom.winslow_steps(result["positions"], m, result["quads"])
    result = dict(result, positions=plane)
    # heights above the plane: harmonic from the loop, then back to world space
    heights = (pts - centre) @ normal
    h = geom.laplace(np.column_stack((heights, np.zeros(m))), len(plane), result["quads"], iterations=240)[:, 0]
    world_pos = centre + plane[:, :1] * u + plane[:, 1:2] * v + h[:, None] * normal
    world_pos[:m] = pts
    plan = dict(positions=world_pos, quads=[tuple(q) for q in result["quads"]], boundary=m,
                virtual=virtual, collapse=collapse, order=order, normal_ccw=np.cross(u, v),
                pattern=result["pattern"], rotation=result["rotation"],
                corners=tuple(result["corners"]), anchors=result["anchors"],
                shift=shift, layout=dict(result, total=len(plane)))
    _accept_plan(plan)
    return plan


DIRECT_SECONDS = .25   # work-clock cap of the instant direct grid (it takes about 1-90 ms of work)


def direct_steps(world, settings):
    """The instant direct grid of an even rim: the lattice (the surrounding quads' turns, then the outline's own
    angles), else four balanced corners. Only a pole-free grid counts. It is finished as solve_steps finishes its
    pick and marked _direct. None when neither gives one. Searches nothing else (the Legacy search runs as before)."""
    pts = np.asarray(world, dtype=np.float64)
    n = len(pts)
    if n < 4 or n % 2 or n > MAX_POINTS or pts.shape != (n, 3) or not np.isfinite(pts).all():
        return None
    if not fill_lattice.eligible(settings):
        return None
    try:
        centre, u, v, normal, xy, pts, order = _planar(pts)
    except FillError:
        return None
    face_turns = settings.get("_face_turns")
    if face_turns is not None and len(face_turns) != n:
        face_turns = None
    refs = [("p", i) for i in range(n)]
    template = {key: value for key, value in settings.items()
                if key != '_face_turns' and key not in ('_validated_pool', '_incumbent') and not key.startswith('_legacy')}

    def balanced():
        grid, _spare = yield from _balanced_steps(xy, refs, template)
        return grid
    result = None
    for make in (lambda: fill_lattice.plan_steps(xy, _turns_in(order, face_turns)), balanced):
        job, spent, got = make(), 0.0, None
        try:
            while True:
                started = geom.clock().monotonic()
                with geom.time_budget(max(0.0, DIRECT_SECONDS - spent)):
                    next(job)
                spent += geom.clock().monotonic() - started
                if spent > DIRECT_SECONDS:
                    raise geom.TimeUp("time budget used up")
                yield None
        except StopIteration as done:
            got = done.value
        except (geom.TimeUp, boundary_preview.PreviewError, fill_partition.PartitionError):
            got = None
        finally:
            job.close()
        if (got is not None and boundary_preview.fill_selection.topology_poles(
                got["quads"], len(got["positions"]), n) == 0):
            result = got
            break
    if result is None:
        return None
    try:
        plan = yield from _finish_steps(result, xy, pts, centre, u, v, normal, None, None, order, 0)
    except FillError:
        return None
    return dict(plan, _direct=True)


MARKS_MAX = 64   # certain_fail(): marks of each kind drawn at most


def _newell(pts):
    """the loop's Newell normal (unit), or None for a loop with no area."""
    c = pts.mean(axis=0)
    nw = np.sum(np.cross(pts - c, np.roll(pts, -1, axis=0) - c), axis=0)
    length = float(np.linalg.norm(nw))
    return nw / length if length > 0 else None


def _self_crosses(pts, normal):
    """the loop projected along `normal` crosses itself (non-neighbour edges, proper crossings)."""
    if normal is None:
        return False
    axis = np.array((1.0, 0.0, 0.0)) if abs(float(normal[0])) < .9 else np.array((0.0, 1.0, 0.0))
    u = np.cross(normal, axis)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    c = pts.mean(axis=0)
    p = np.column_stack(((pts - c) @ u, (pts - c) @ v))
    r = np.roll(p, -1, axis=0) - p
    n = len(p)
    size = max(float(np.ptp(p, axis=0).max()), 1e-30)

    def cross(x, y):
        return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
    qp = p[None, :, :] - p[:, None, :]
    rs = cross(r[:, None, :], r[None, :, :])
    with np.errstate(divide='ignore', invalid='ignore'):
        t = cross(qp, r[None, :, :]) / rs
        s = cross(qp, r[:, None, :]) / rs
    idx = np.arange(n)
    gap = np.abs(idx[:, None] - idx[None, :])
    hit = ((np.abs(rs) > 1e-12 * size * size) & (t > 1e-9) & (t < 1 - 1e-9) & (s > 1e-9) & (s < 1 - 1e-9)
           & (gap > 1) & (gap < n - 1))
    return bool(hit.any())


def certain_fail(world):
    """Provable reasons no fill exists, located on the rim. None, or dict(segs=[(a, b) world point pairs],
    dots=[world points]) for: a rim crossing itself on both its best-fit (SVD) plane and its Newell plane (both edges,
    a dot halfway between the two crossing points), coincident non-neighbour points (a dot on each), a zero-length span
    (a dot) and a back-and-forth spike whose tip is convex on both planes (a dot on the tip).
    At most 2 x MARKS_MAX segments and MARKS_MAX dots in all."""
    pts = np.asarray(world, dtype=np.float64)
    n = len(pts)
    if n < 4 or n > MAX_POINTS or pts.shape != (n, 3) or not np.isfinite(pts).all():
        return None
    size = float(np.ptp(pts, axis=0).max())
    if size <= 0:
        return None
    eps = 1e-6 * size
    segs, dots = [], []
    nxt = np.roll(pts, -1, axis=0)
    lengths = np.linalg.norm(nxt - pts, axis=1)
    for i in np.flatnonzero(lengths < eps):
        dots.append(tuple((pts[i] + nxt[i]) / 2))
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    idx = np.arange(n)
    gap = np.abs(idx[:, None] - idx[None, :])
    near = (d < eps) & (gap > 1) & (gap < n - 1) & (idx[:, None] < idx[None, :])
    for i, j in zip(*np.nonzero(near)):
        dots += [tuple(pts[i]), tuple(pts[j])]
    centre, u, v, normal = _frame(pts)
    xy = np.column_stack(((pts - centre) @ u, (pts - centre) @ v))
    orient = 1.0 if float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1))) >= 0 else -1.0
    a = pts - np.roll(pts, 1, axis=0)
    b = nxt - pts
    la, lb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    ok = (la >= eps) & (lb >= eps)
    cosine = np.einsum('ij,ij->i', a, b) / np.where(ok, la * lb, 1.0)
    a2, b2 = xy - np.roll(xy, 1, axis=0), np.roll(xy, -1, axis=0) - xy
    convex = orient * (a2[:, 0] * b2[:, 1] - a2[:, 1] * b2[:, 0]) >= 0   # a reflex slit tip can still fill
    nw = _newell(pts)
    if nw is not None:   # convex on the Newell plane too (a bent rim can turn a reflex tip over on one plane)
        cn = np.cross(a, b) @ nw
        convex &= cn >= 0
    for i in np.flatnonzero(ok & convex & (cosine < -math.cos(math.radians(1.0)))):
        dots.append(tuple(pts[i]))
    if not _self_crosses(pts, _newell(pts)):   # a crossing on one plane only proves nothing
        dots = dots[:MARKS_MAX]
        return dict(segs=[], dots=dots) if dots else None
    p, r = xy, np.roll(xy, -1, axis=0) - xy

    def cross(x, y):
        return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
    qp = p[None, :, :] - p[:, None, :]
    rs = cross(r[:, None, :], r[None, :, :])
    with np.errstate(divide='ignore', invalid='ignore'):
        t = cross(qp, r[None, :, :]) / rs
        s = cross(qp, r[:, None, :]) / rs
    hit = ((np.abs(rs) > 1e-12 * size * size) & (t > 1e-9) & (t < 1 - 1e-9) & (s > 1e-9) & (s < 1 - 1e-9)
           & (gap > 1) & (gap < n - 1) & (idx[:, None] < idx[None, :]))
    for i, j in zip(*np.nonzero(hit)):
        segs += [(tuple(pts[i]), tuple(nxt[i])), (tuple(pts[j]), tuple(nxt[j]))]
        x1 = pts[i] + t[i, j] * (nxt[i] - pts[i])
        x2 = pts[j] + s[i, j] * (nxt[j] - pts[j])
        dots.append(tuple((x1 + x2) / 2))
        if len(segs) >= 2 * MARKS_MAX:
            break
    segs, dots = segs[:2 * MARKS_MAX], dots[:MARKS_MAX]
    if not segs and not dots:
        return None
    return dict(segs=segs, dots=dots)


def project_steps(context, plan, srcs):
    """Project cooperatively and check the final geometry, not just the flat seed."""
    from mathutils import Vector
    from . import raycast
    pos = plan["positions"]
    for i in range(plan["boundary"], len(pos)):
        hit = raycast.nearest_on_sources(context, Vector(pos[i]), srcs)
        if hit is not None:
            pos[i] = tuple(hit.world)
        yield None
    _accept_plan(plan)
    return plan


def faces_of(plan):
    """Final faces as index tuples into plan positions; the midpoint collapses into its
    neighbour, turning the one face holding both into a triangle."""
    out = []
    vi, ci = plan["virtual"], plan["collapse"]
    for q in plan["quads"]:
        face = [ci if i == vi else i for i in q] if vi is not None else list(q)
        dedup = [i for k, i in enumerate(face) if i != face[k - 1]]
        out.append(tuple(dedup))
    return out


def _accept_plan(plan):
    """An odd loop can collapse its midpoint to either endpoint; use a valid one."""
    try:
        validate_plan(plan)
        return
    except FillError as first:
        if plan["virtual"] is None:
            raise
        old = plan["collapse"]
        left = (plan["virtual"] - 1) % plan["boundary"]
        right = (plan["virtual"] + 1) % plan["boundary"]
        plan["collapse"] = right if old == left else left
        try:
            validate_plan(plan)
        except FillError:
            plan["collapse"] = old
            raise first


def validate_plan(plan):
    """Validate the actual triangle/quad disk after collapse or source projection."""
    positions = np.asarray(plan["positions"], dtype=float)
    faces = faces_of(plan)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all() or not faces:
        raise FillError("Invalid fill geometry")
    normal = np.asarray(plan["normal_ccw"], dtype=float)
    ids = [i for i, source in enumerate(plan["order"]) if source is not None]
    scale = max(float(np.ptp(positions[ids], axis=0).max()), 1e-30)
    area_floor = scale * scale * 1e-12
    edges, used = Counter(), set()
    groups = {3: [], 4: []}
    for face in faces:
        if len(face) not in (3, 4) or len(set(face)) != len(face) or min(face) < 0 or max(face) >= len(positions):
            raise FillError("Invalid fill geometry")
        groups[len(face)].append(face)
        used.update(face)
        edges.update(tuple(sorted((a, b))) for a, b in zip(face, face[1:] + face[:1]))
    for group in groups.values():
        if not group:
            continue
        p = positions[np.asarray(group,dtype=np.int64)]
        directions = np.roll(p,-1,axis=1)-p
        crosses = np.cross(directions,np.roll(directions,-1,axis=1))
        turns = crosses @ normal
        # Keep the previous per-face operation for comparisons close enough to
        # the threshold that a batched dot product's rounding could matter.
        error = 16*np.finfo(float).eps*np.sum(np.abs(crosses*normal),axis=2)
        borderline = np.any(np.abs(turns-area_floor)<=error,axis=1)
        for row in np.flatnonzero(borderline):
            local = p[row]
            delta = np.roll(local,-1,axis=0)-local
            turns[row] = np.cross(delta,np.roll(delta,-1,axis=0)) @ normal
        if np.any(turns <= area_floor):
            raise FillError("This layout folds; try another choice (Ctrl+Wheel)")
    boundary = {tuple(sorted((a, b))) for a, b in zip(ids, ids[1:] + ids[:1])}
    expected_used = set(ids) | set(range(plan["boundary"], len(positions)))
    if (used != expected_used or len(used) - len(edges) + len(faces) != 1 or
            {edge for edge, count in edges.items() if count == 1} != boundary or
            any(count not in (1, 2) for count in edges.values())):
        raise FillError("Invalid fill connectivity")


# ------------------------------------------------------------------ commit

def commit(bm, mw, loop, plan, view_dir, correct_attributes=True):
    """Create the patch in the edit mesh. Returns the new faces. With
    correct_attributes (Blender's Correct Face Attributes), UV and colours are
    continued from the rim; plan['uv_warning'] says when they could not be."""
    import bmesh
    from mathutils import Vector
    validate_plan(plan)
    mi = np.linalg.inv(np.array(mw, dtype=np.float64))     # double, see world_points
    verts = {}
    for idx, src in enumerate(plan["order"]):
        if src is not None:
            verts[idx] = loop[src]
    if len(verts) != len(loop) or any(not v.is_valid for v in loop):
        raise FillError("The outline changed; preview again")
    # the rim must still be where the plan was solved (the multi-panel path checks it too)
    rim = sorted(verts)
    planned = np.asarray(plan["positions"], dtype=np.float64)[rim]
    current = world_points(mw, [verts[i] for i in rim])
    size = max(float(np.ptp(planned, axis=0).max()), 1e-30)
    if current.shape != planned.shape or not np.allclose(current, planned, rtol=0.0, atol=size * 1e-5):
        raise FillError("The outline changed; preview again")
    chain = set(closing_chain(loop) or ())   # re-derived from the mesh at commit
    for a, b in zip(loop, loop[1:] + loop[:1]):
        edge = bm.edges.get((a, b))
        if edge is None or not (_open(edge) or edge in chain):
            raise FillError("The outline changed; preview again")
    if _filled_loop(loop):
        raise FillError("This loop is already filled")
    faces = faces_of(plan)
    # winding: opposite to an existing face across a shared loop edge, else facing the view
    flip = None
    ids = [i for i, source in enumerate(plan["order"]) if source is not None]
    for a, b in zip(ids, ids[1:] + ids[:1]):
        va, vb = verts[a], verts[b]
        edge = bm.edges.get((va, vb))
        if not edge.link_faces or edge in chain:
            continue
        other = edge.link_faces[0]
        ring = list(other.verts)
        k = ring.index(va)
        direction = ring[(k + 1) % len(ring)] is vb
        if flip is not None and direction != flip:
            raise FillError("The faces around the outline have inconsistent normals; recalculate normals first")
        flip = direction
    if flip is None and chain:
        # opposite to the faces across the closing chain that are nearly coplanar with the loop;
        # every such face votes; when they disagree the strictly most coplanar face decides (2.6.2's
        # rule), and on a tie the view decides
        from mathutils import geometry
        n = _frame(np.array([tuple(v.co) for v in loop], dtype=np.float64))[3]
        votes, ranked = set(), []
        for a, b in zip(ids, ids[1:] + ids[:1]):
            va, vb = verts[a], verts[b]
            edge = bm.edges.get((va, vb))
            if edge not in chain:
                continue
            for face in edge.link_faces:
                level = abs(float(np.dot(tuple(geometry.normal([w.co for w in face.verts])), n)))
                if level >= .5:
                    ring = list(face.verts)
                    k = ring.index(va)
                    vote = ring[(k + 1) % len(ring)] is vb
                    votes.add(vote)
                    ranked.append((level, vote))
        if len(votes) == 1:
            flip = votes.pop()
        elif ranked:
            ranked.sort(key=lambda r: -r[0])
            if all(ranked[0][0] - level > 1e-6 for level, vote in ranked if vote != ranked[0][1]):
                flip = ranked[0][1]
    if flip is None:   # the view decides; a mirroring matrix_world (negative determinant) reverses the shown normal
        flip = ((float(np.dot(plan["normal_ccw"], view_dir)) > 0)
                != (float(np.linalg.det(np.array(mw, dtype=np.float64)[:3, :3])) < 0))
    from .face_attributes import FillUV
    rim_faces = ({f for e in _rim_edges(loop) if e is not None and e not in chain for f in e.link_faces}
                 if chain else None)
    uv_source = ((FillUV(bm, loop) if rim_faces is None else
                  FillUV(bm, loop, rim_faces=rim_faces, along=_chain_middles(loop, chain)))
                 if correct_attributes else None)
    plan["uv_warning"] = None
    created, new_vertices, new_edges = [], [], []
    try:
        for i in range(plan["boundary"], len(plan["positions"])):
            vertex = bm.verts.new(Vector(mi[:3, :3] @ np.asarray(plan["positions"][i], dtype=np.float64) + mi[:3, 3]))
            new_vertices.append(vertex)
            verts[i] = vertex
        for face in faces:
            vs = [verts[i] for i in face]
            if flip:
                vs.reverse()
            if bm.faces.get(vs) is not None:
                raise FillError("This loop is already filled")
            for a, b in zip(vs, vs[1:] + vs[:1]):
                edge = bm.edges.get((a, b))
                if edge is None:
                    new_edges.append((a, b))
                elif len(edge.link_faces) >= 2 and edge not in chain:
                    raise FillError("This fill would give an edge more than two faces")
            created.append(bm.faces.new(vs))
        for face in created:
            face.normal_update()
        if uv_source is not None and uv_source.has_rim and (
                len(bm.loops.layers.uv) or len(bm.loops.layers.color) or len(bm.loops.layers.float_color)):
            # Native attribute copying can invalidate Python face wrappers. Rebind
            # by vertex identity even on failure, so rollback still owns every face.
            rings = [tuple(face.verts) for face in created]
            try:
                if not chain:   # the faces across a closing edge (chain) never donate
                    bmesh.ops.face_attribute_fill(bm, faces=created, use_normals=False, use_data=True)
                else:
                    donor = next((f for e in _rim_edges(loop) if e is not None and e not in chain
                                  for f in e.link_faces if f not in created), None)
                    if donor is not None:
                        for face in created:
                            face.material_index, face.smooth = donor.material_index, donor.smooth
            finally:
                created[:] = [face for ring in rings if (face := bm.faces.get(ring)) is not None]
            uv_source.apply(created)
            plan["uv_warning"] = uv_source.issue
    except Exception:
        # Roll back only this transaction, including edges implicitly created by
        # BMFaceSeq.new before an error. Existing geometry/selection is untouched.
        for face in reversed(created):
            if face.is_valid:
                bm.faces.remove(face)
        for a, b in reversed(new_edges):
            edge = bm.edges.get((a, b))
            if edge is not None and edge.is_valid and not edge.link_faces:
                bm.edges.remove(edge)
        for vertex in reversed(new_vertices):
            if vertex.is_valid:
                bm.verts.remove(vertex)
        raise
    finally:
        if uv_source is not None:
            uv_source.close()
    return created

