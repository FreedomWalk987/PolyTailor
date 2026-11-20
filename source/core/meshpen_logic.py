# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Build: what a click inserts, and the edit itself.

For every mouse move `update` looks at what is under the mouse (a point on the surface,
a hovered vertex / edge / face) and the selection, and decides an action; `commit`
carries it out; `draw` shows the preview. The pen operator (ops/polypen.py) uses the
same interface for every mode.
"""

from enum import Enum
from bisect import bisect_right
from heapq import heappop, heappush
from itertools import chain as _chain
from math import acos, hypot

import bmesh
import numpy as np
from bmesh.types import BMEdge, BMFace, BMVert
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import intersect_line_line, intersect_line_line_2d, intersect_line_plane

from .polypen_support import (
    is_bmvert_hidden,
    nearest_point_valid_sources,
    raycast_point_valid_sources,
    raycast_valid_sources,
)
from .polypen_support_base import CC_2D_LINES, CC_2D_POINTS, CC_2D_TRIANGLES, Drawing
from . import raycast
from .. import palette

HOVER_PX = 10        # hover radius of vertices, edges and faces
EDGE_END_SHARE = 0.05  # this share at each end of an edge belongs to its vertices
DASH_INSET = 8       # preview lines stop this far from their end points


class Action(Enum):
    NONE = 0
    VERT = 1
    VERT_EDGE = 2
    EDGE_TRI = 3
    EDGE_QUAD = 4
    TRI_QUAD = 5
    EDGE_BRIDGE = 6
    SPLIT_EDGE = 7
    EDGE_SPLIT_EDGE = 8
    SPLIT_QUAD = 9
    SPLIT_QUAD_CENTER = 10
    SPLIT_QUAD_EDGES = 11
    VERT_SPLIT_EDGE = 12
    WIRE_SPLIT_EDGE_FACE = 13
    WIRE_VERT_SPLIT_FACE = 14


# ---------------------------------------------------------------------------------------
# screen helpers

def to_region(context, world):
    """Region position of a world point; points behind a perspective view are pushed
    far out in a stable direction instead of giving None."""
    region, rv3d = context.region, context.region_data
    h = rv3d.perspective_matrix @ Vector((world[0], world[1], world[2], 1.0))
    if h.w > 0.0:   # the same arithmetic as bpy_extras' location_3d_to_region_2d
        half_w, half_h = region.width / 2.0, region.height / 2.0
        return Vector((half_w + half_w * (h.x / h.w), half_h + half_h * (h.y / h.w)))
    return Vector((region.width / 2 * (1 + h.x / 1e-4), region.height / 2 * (1 + h.y / 1e-4)))


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def segment_distance(p, a, b, lo=EDGE_END_SHARE, hi=1.0 - EDGE_END_SHARE):
    """Distance from p to the part lo..hi of segment a-b."""
    ab = b - a
    n2 = ab.length_squared
    if n2 <= 1e-5:
        return (p - a).length
    return (p - (a + ab * _clamp((p - a).dot(ab) / n2, lo, hi))).length


def segment_closest(p, a, b):
    ab = b - a
    n2 = ab.length_squared
    if n2 < 1e-5:
        return a
    return a + ab * _clamp(ab.dot(p - a) / n2, 0.0, 1.0)


NEAR_EDGES_MIN = 32       # at most this many selected edges keep the plain per-edge loops
_PROJECTION_ERROR = 4e-6  # relative rounding bound of the single-precision projection (about 30x the real one)


class SelectedEdgeScreen:
    """The end points of a fixed list of edges (object space), read into one numpy array once.

    bounds() narrows candidates down only: for a screen point p it gives each edge's screen distance in double
    precision and a margin that bounds the difference to the distance the exact code computes in single precision
    (segment_distance / segment_closest on to_region or the view's to_screen). An edge with an end behind, at or near
    the view plane is flagged unsafe (always kept). The exact tests still run on the kept edges, so picks do not change.
    """

    def __init__(self, edges):
        self.edges = list(edges)   # the iteration order of the (unchanged) selection set
        n = len(self.edges)
        self.co = np.fromiter(_chain.from_iterable(_chain(e.verts[0].co, e.verts[1].co) for e in self.edges),
                              dtype=np.float64, count=6 * n).reshape(n * 2, 3)

    def bounds(self, persmat, mw, width, height, p, lo, hi):
        """(distance, margin, unsafe) per edge: the exact distance lies within distance +- margin unless unsafe."""
        P = np.array(persmat, dtype=np.float64)
        M = np.array(mw, dtype=np.float64)
        PM, A = P @ M, np.abs(P) @ np.abs(M)
        co = self.co
        h = co @ PM[:, :3].T + PM[:, 3]
        err = (np.abs(co) @ A[:, :3].T + A[:, 3]) * _PROJECTION_ERROR
        w, ew = h[:, 3], err[:, 3]
        bad = ~(w > 4.0 * ew) | ~np.isfinite(w)
        w = np.where(bad, 1.0, w)
        den = np.where(bad, 1.0, w - ew)
        hw, hh = width / 2.0, height / 2.0
        qx, qy = h[:, 0] / w, h[:, 1] / w
        sx, sy = hw + hw * qx, hh + hh * qy
        point_err = (hw * (err[:, 0] + np.abs(qx) * ew) / den + hh * (err[:, 1] + np.abs(qy) * ew) / den
                     + 1e-6 * (hw + hh + np.abs(sx) + np.abs(sy)))
        sx, sy, point_err, bad = sx.reshape(-1, 2), sy.reshape(-1, 2), point_err.reshape(-1, 2), bad.reshape(-1, 2)
        px, py = float(p[0]), float(p[1])
        ax, ay, bx, by = sx[:, 0], sy[:, 0], sx[:, 1], sy[:, 1]
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / np.where(l2 > 0.0, l2, 1.0), lo, hi)
        dist = np.where(l2 < 1e-5, np.hypot(px - ax, py - ay), np.hypot(ax + dx * t - px, ay + dy * t - py))
        margin = (point_err.max(axis=1) + 1e-5 * (np.abs(ax) + np.abs(ay) + np.abs(bx) + np.abs(by) + abs(px) + abs(py) + 1.0)
                  + 0.004)   # 0.004 px: the short-edge branch (|ab| < 0.0032 px) of the exact code
        unsafe = bad[:, 0] | bad[:, 1] | ~np.isfinite(dist) | ~np.isfinite(margin)
        return dist, margin, unsafe

    def within(self, persmat, mw, width, height, p, limit, lo=0.0, hi=1.0):
        """The edges (in list order) whose exact distance to p can be at most limit."""
        dist, margin, unsafe = self.bounds(persmat, mw, width, height, p, lo, hi)
        return [self.edges[i] for i in np.flatnonzero(unsafe | (dist - margin <= limit))]

    def nearest_candidates(self, persmat, mw, width, height, p, lo, hi):
        """The edges (in list order) that can have the smallest exact distance to p (ties included)."""
        dist, margin, unsafe = self.bounds(persmat, mw, width, height, p, lo, hi)
        upper = np.where(unsafe, np.inf, dist + margin).min() if len(dist) else np.inf
        if not np.isfinite(upper):
            return list(self.edges)
        return [self.edges[i] for i in np.flatnonzero(unsafe | (dist - margin <= upper))]

    def ordered(self, persmat, mw, width, height, p, lo, hi, key, keep=None, mask=None):
        """The edges in exactly the order of sorted(edges, key=key) (stable on the list order), without those the
        mask (bool per position) or keep(position) rejects, computed lazily: key() runs in lower-bound order, and an
        edge is handed out once its (key, position) is below the lower bound of every edge not handed out yet whose
        key is not computed yet (rejected edges are never handed out, so they do not count)."""
        dist, margin, unsafe = self.bounds(persmat, mw, width, height, p, lo, hi)
        lower = np.where(unsafe, -np.inf, dist - margin)
        todo = np.argsort(lower, kind="stable")
        if mask is not None:
            todo = todo[mask[todo]]
        todo = todo.tolist()
        lower = lower.tolist()
        heap, i, n = [], 0, len(todo)
        while heap or i < n:
            if heap and (i >= n or heap[0][0] < lower[todo[i]]):
                yield self.edges[heappop(heap)[1]]
                continue
            j = todo[i]
            i += 1
            if keep is None or keep(j):
                heappush(heap, (key(self.edges[j]), j))


class QuadRings:
    """Components of the relation "edge -> the opposite edge of a quad it is on" (over all quads, no other
    condition), found from the selected edges on demand and kept per selection snapshot, with each component's quads
    (the quads that have an edge in it) and their centres and squared corner radii (numpy).

    _find_loop_cut's walk from a start edge only moves along that relation, so every quad it can enter has an edge in
    the start's component; it can only end on a quad where inside_face(hit) holds or that has the hovered edge.
    possible() is that necessary condition (inside_face's own first test, the centre distance, is narrowed in numpy
    with a rounding margin, then inside_face itself runs). It holds also for rings that cross themselves."""

    def __init__(self, edges):
        self.edges = edges   # the selected edges (SelectedEdgeScreen order)
        self.position = {e: i for i, e in enumerate(edges)}
        self.selected = [-1] * len(edges)   # position -> component (-1: not found yet)
        self.unknown = len(edges)
        self.of = {}       # edge -> component
        self.faces = []    # component -> its quads
        self.data = []     # component -> (centres, squared radii, coordinate scale) or None (a degenerate quad met)

    def _assign(self, edge, c):
        self.of[edge] = c
        i = self.position.get(edge)
        if i is not None:
            self.selected[i] = c
            self.unknown -= 1

    def at(self, i):
        """The component of the selected edge at position i."""
        c = self.selected[i]
        return c if c >= 0 else self.component(self.edges[i])

    def component(self, start):
        c = self.of.get(start)
        if c is not None:
            return c
        c = len(self.faces)
        self._assign(start, c)
        stack, faces, seen, degenerate = [start], [], set(), False
        while stack:
            edge = stack.pop()
            for loop in edge.link_loops:
                face = loop.face
                if len(face.loops) != 4:
                    continue
                if face not in seen:
                    seen.add(face)
                    faces.append(face)
                # the quad's opposite edge (opposite_edge(face, edge), without its per-edge vertex sets); a face that
                # uses the edge twice gives both, so the component stays a superset of the walk
                far = loop.link_loop_next.link_loop_next.edge
                a, b = edge.verts
                fa, fb = far.verts
                if fa == a or fa == b or fb == a or fb == b:   # not a proper quad: leave it to the original walk
                    degenerate = True
                elif far not in self.of:
                    self._assign(far, c)
                    stack.append(far)
        if degenerate or any(len(f.verts) != 4 for f in faces):
            data = None    # leave this component to the original walk
        else:
            pts = np.fromiter(_chain.from_iterable(v.co for f in faces for v in f.verts), dtype=np.float64,
                              count=12 * len(faces)).reshape(len(faces), 4, 3)
            mid = pts.mean(axis=1)
            data = (mid, ((pts - mid[:, None, :]) ** 2).sum(axis=2).max(axis=1), float(np.abs(pts).max(initial=0.0)))
        self.faces.append(faces)
        self.data.append(data)
        return c

    def possible(self, c, hit, he):
        """Can a walk from a start edge in component c end on a target? (necessary condition)
        - the hovered edge as the far edge: then the walk reached it, so it is in c;
        - inside_face(hit) on a quad the walk can enter: from an edge e in c that does not touch the hovered edge,
          unless the walk refuses that step (the far edge touches the hovered edge and is not it)."""
        data = self.data[c]
        if data is None:
            return True
        if he is not None and self.of.get(he) == c:
            return True
        mid, r2, scale = data
        if not len(mid):
            return False
        q = np.array((hit[0], hit[1], hit[2]), dtype=np.float64)
        d2 = ((mid - q) ** 2).sum(axis=1)
        scale = max(scale, float(np.abs(q).max()))
        slack = 1e-5 * scale * (np.sqrt(d2) + np.sqrt(r2)) + 1e-10 * scale * scale   # single-precision rounding
        faces = self.faces[c]
        return any(self._enterable(faces[i], c, he) and inside_face(hit, faces[i])
                   for i in np.flatnonzero(d2 <= r2 + slack))

    def _enterable(self, face, c, he):
        """The walk can enter this quad from an edge of c. An edge that touches the hovered edge is never walked
        from: as a start it is skipped, as a far edge refused."""
        ends = set(he.verts) if he is not None else None
        for loop in face.loops:
            edge = loop.edge
            if self.of.get(edge) != c or (ends is not None and edge != he and not ends.isdisjoint(edge.verts)):
                continue
            far = loop.link_loop_next.link_loop_next.edge
            if ends is None or far == he or ends.isdisjoint(far.verts):
                return True
        return False


# ---------------------------------------------------------------------------------------
# mesh helpers

def shared(*groups):
    """Elements in every group, built up one group at a time (the set order Build's
    choices in tie cases have always followed)."""
    out = set(groups[0])
    for g in groups[1:]:
        out &= set(g)
    return out


def other_vert(edge, vert):
    a, b = edge.verts
    return b if vert == a else a if vert == b else None


def touching(e0, e1):
    return not set(e0.verts).isdisjoint(e1.verts)


def common_vert(e0, e1):
    return next(iter(set(e0.verts) & set(e1.verts)), None)


def centroid(face):
    return sum((v.co for v in face.verts), Vector()) / len(face.verts)


def inside_face(co, face):
    """co lies within the face (even-odd test in the face plane, so concave faces are judged
    right), and not farther from its centre than its farthest corner."""
    pts = [v.co for v in face.verts]
    mid = sum(pts, Vector()) / len(pts)
    if (co - mid).length_squared > max((p - mid).length_squared for p in pts):
        return False
    n = Vector()
    for a, b in zip(pts, pts[1:] + pts[:1]):   # Newell normal (robust for concave faces)
        n += Vector(((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y)))
    if n.length_squared == 0.0:
        return False
    u = n.orthogonal().normalized()
    w = n.normalized().cross(u)
    q = [((p - mid).dot(u), (p - mid).dot(w)) for p in pts]
    x, y = (co - mid).dot(u), (co - mid).dot(w)
    inside = False
    for (x1, y1), (x2, y2) in zip(q, q[1:] + q[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def wire_from(vert):
    """Vertices of the face-less chain that ends at `vert`, from the vertex with faces it
    starts at (the anchor) to `vert`; None if vert has faces, or the chain forks or
    dead-ends."""
    if not vert.is_wire:
        return None
    chain, used = [vert], set()
    while chain[-1].is_wire:
        left = set(chain[-1].link_edges) - used
        if len(left) != 1:
            return None
        edge = next(iter(left))
        used.add(edge)
        chain.append(other_vert(edge, chain[-1]))
    chain.reverse()
    return chain


def wire_face_to(vert, hovered):
    """(wire, face) when the wire ending at vert can be closed through a face of its
    anchor that `hovered` (vertex or edge) is on."""
    wire = wire_from(vert)
    if not wire:
        return None
    faces = list(set(hovered.link_faces) & set(wire[0].link_faces))
    if len(faces) == 1:
        return wire, faces[0]
    if len(faces) == 2:
        near = wire[1].co
        f0, f1 = faces
        return (wire, f0) if (near - centroid(f0)).length < (near - centroid(f1)).length else (wire, f1)
    return None


def wire_face_around(vert):
    """(wire, face of the anchor that contains vert)."""
    wire = wire_from(vert)
    if not wire:
        return None
    for face in wire[0].link_faces:
        if inside_face(vert.co, face):
            return wire, face
    return None


def split_lists(face, path):
    """The two corner lists that divide face along `path` (vertex list from one corner of
    the face to another, the inner ones not in the face), or None when that cannot make
    two valid faces."""
    ring = [loop.vert for loop in face.loops]
    first, last = path[0], path[-1]
    if first not in ring or last not in ring:
        return None

    def arc(start, stop):   # face corners strictly after start, up to before stop
        i = ring.index(start)
        rot = ring[i:] + ring[:i]
        return rot[1:rot.index(stop)]

    lists = (path + arc(last, first), path[::-1] + arc(first, last))
    if any(len(set(vs)) != len(vs) or len(vs) < 3 for vs in lists):
        return None
    return lists


def point_on_edge(a, b, origin, direction):
    """Point of segment a-b closest to the ray (origin, direction); one space for all."""
    found = intersect_line_line(a, b, origin, origin + direction)
    ab = b - a
    if found is None or not ab.length_squared:
        return (a + b) / 2
    t = (found[0] - a).dot(ab) / ab.length_squared
    return a.lerp(b, min(1.0, max(0.0, t)))


def point_on_plane(co, normal, origin, direction):
    """Where the ray crosses the plane through co with normal (one space), else co."""
    found = intersect_line_plane(origin, origin + direction, co, normal)
    return found if found is not None else co.copy()


def rebuild_split(bm, face, path):
    """Replace face by two faces divided by `path` (see split_lists)."""
    lists = split_lists(face, path)
    if lists is None:
        raise ValueError("This path cannot split the face")
    for verts in lists:
        copy_face_data(bm, bm.faces.new(verts), face)
    bm.faces.remove(face)


_FACE_LAYER_KINDS = ("int", "float", "float_vector", "float_color", "string", "bool")


def copy_face_data(bm, new, old):
    """Carry material, smoothing, face attributes and interpolated loop data (UVs,
    colors) over from the face being replaced; call before `old` is removed.
    BMFace.copy_from is not used: with a different corner count it invalidates `new`.
    Vertex data is left alone (vert=False) so pins and creases are not interpolated."""
    new.material_index = old.material_index
    new.smooth = old.smooth
    for kind in _FACE_LAYER_KINDS:
        for layer in getattr(bm.faces.layers, kind, {}).values():
            new[layer] = old[layer]
    new.copy_from_face_interp(old, False)


def quad_corners_from(face, edge):
    """Face corners starting at the edge's vertex that follows the other one in the
    face's winding (so the edge is corners 0 and 3)."""
    ring = list(face.verts)
    i0, i1 = (ring.index(v) for v in edge.verts)
    start = i0 if (i1 + 1) % 4 == i0 else i1
    return ring[start:] + ring[:start]


def tri_from_edge(face, edge):
    """Triangle corners ordered so that the edge is corners 0 and 1."""
    a, b, c = face.verts
    ends = set(edge.verts)
    if ends == {a, b}:
        return a, b, c
    if ends == {b, c}:
        return b, c, a
    return c, a, b


def opposite_edge(face, edge):
    return next((e for e in face.edges if not touching(edge, e)), None)


def faces_viewer(context, mw, face):
    """Is the face's front side toward the view (perspective and orthographic)?"""
    xy = to_region(context, mw @ face.calc_center_median())
    if xy is None:
        return False
    _origin, direction = raycast.ray(context, xy)
    return direction.dot(mw.inverted_safe().transposed().to_3x3() @ face.normal) < 0


def crossings(context, mw, start, a, b):
    """Edges crossed on screen by the segment a-b, walking face to face from `start`
    (vertex or edge): [(edge, crossing point)]. An edge start is listed first (at a).

    Like Blender's Knife, the walk keeps to faces facing the view when `start` touches
    one: on a closed mesh the screen line also crosses the hidden back faces, and taking
    those made the cut run across the back and through the volume. Where `start` only
    touches back faces (an open sheet seen from behind), every face is walked as before."""
    found = []
    walk = lambda faces: faces
    if isinstance(start, BMVert):
        v, seen = start, set()
        while v.is_wire and v not in seen:   # a wire: walk to where it meets faces
            seen.add(v)
            nxt = next((other_vert(e, v) for e in v.link_edges), None)
            if nxt is None:
                break
            v = nxt
        pending = list(v.link_faces)
        done = set(start.link_edges)
    else:
        pending = list(start.link_faces)
        done = {start}
        found.append((start, a))
    front = [f for f in pending if faces_viewer(context, mw, f)]
    if front:
        pending = front
        walk = lambda faces: [f for f in faces if faces_viewer(context, mw, f)]
    while pending:
        face = pending.pop(0)
        # Where several edges of the face are crossed (the line through a corner), the
        # first one in set order is taken, as Build always did: its choice in these
        # cases follows the same set order, so this keeps it identical.
        for edge in set(face.edges) - done:
            e0, e1 = (to_region(context, mw @ u.co) for u in edge.verts)
            hit = intersect_line_line_2d(a, b, e0, e1)
            if hit:
                done.add(edge)
                found.append((edge, hit))
                beyond = next((f for f in walk(edge.link_faces) if f != face), None)
                if beyond:
                    pending.append(beyond)
                break
    return found


def quad_uv(p, a, b, c, d, steps=100):
    """(u, v) of point p in the screen quad a-b-c-d: p lies on the line from
    e = a + u (b - a) to f = d + u (c - d), at v of the way from e to f. u by bisection."""
    lo, hi = 0.0, 1.0
    u = 0.5
    ab = b - a
    for _ in range(steps):
        e, f = a + (b - a) * u, d + (c - d) * u
        if ab.dot((f - e).normalized()) < ab.dot((p - e).normalized()):
            lo = u
        else:
            hi = u
        u = lo + (hi - lo) * 0.5
    e, f = a + (b - a) * u, d + (c - d) * u
    return u, (p - e).length / (f - e).length


# ---------------------------------------------------------------------------------------
# hover lookups

def _shared_faces_bvh(bm, cache):
    if "faces" not in cache:
        cache["faces"] = BVHTree.FromBMesh(bm)
    return cache["faces"]


class _VertLookup:
    """Hovered vertex: loose vertices plus the corners of the face nearest to P."""

    def __init__(self, bm, cache):
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        self.bm = bm
        self.faces = _shared_faces_bvh(bm, cache)
        self.loose = [v for v in bm.verts if not v.link_faces]
        self.tree = BVHTree.FromPolygons([v.co for v in self.loose],
                                         [[i, i, i] for i in range(len(self.loose))], all_triangles=True)
        self.bmv = None

    @property
    def is_valid(self):
        return self.bm.is_valid and (self.bmv is None or self.bmv.is_valid) and all(v.is_valid for v in self.loose)

    def find(self, context, mw, co, keep=None):
        self.bmv = None
        if not self.is_valid or not co:
            return
        cands = []
        i = self.tree.find_nearest(co, 1.84467e19)[2]
        if i is not None:
            cands.append(self.loose[i])
        j = self.faces.find_nearest(co, 1.84467e19)[2]
        if j is not None:
            cands.extend(self.bm.faces[j].verts)
        cands = [v for v in cands if keep(v)] if keep else [v for v in cands if not v.select]
        if not cands:
            return
        here = to_region(context, mw @ co)
        best = min(cands, key=lambda v: (here - to_region(context, mw @ v.co)).length)
        if (here - to_region(context, mw @ best.co)).length <= Drawing.scale(HOVER_PX):
            self.bmv = best


class _EdgeLookup:
    """Hovered edge: loose edges plus the edges of the face nearest to P."""

    def __init__(self, bm, cache):
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        self.bm = bm
        self.faces = _shared_faces_bvh(bm, cache)
        self.loose = [e for e in bm.edges if not e.link_faces]
        self.tree = BVHTree.FromPolygons([v.co for e in self.loose for v in e.verts],
                                         [[2 * i, 2 * i + 1, 2 * i + 1] for i in range(len(self.loose))],
                                         all_triangles=True)
        self.bme = None
        self.co2d = None

    @property
    def is_valid(self):
        return self.bm.is_valid and (self.bme is None or self.bme.is_valid) and all(e.is_valid for e in self.loose)

    def find(self, context, mw, co, keep=None):
        self.bme = None
        if not self.is_valid:
            return
        cands = []
        i = self.tree.find_nearest(co, 1.84467e19)[2]
        if i is not None:
            cands.append(self.loose[i])
        j = self.faces.find_nearest(co, 1.84467e19)[2]
        if j is not None:
            cands.extend(self.bm.faces[j].edges)
        if keep:
            cands = [e for e in cands if keep(e)]
        if not cands:
            return
        here = to_region(context, mw @ co)
        ends = lambda e: [to_region(context, mw @ v.co) for v in e.verts]
        best = min(cands, key=lambda e: segment_distance(here, *ends(e)))
        if segment_distance(here, *ends(best)) > Drawing.scale(HOVER_PX):
            return
        self.bme = best
        self.co2d = segment_closest(here, *ends(best))


class _FaceLookup:
    """Hovered face: the face nearest to P, if P is on screen close to it."""

    def __init__(self, bm, cache):
        self.bm = bm
        self.faces = _shared_faces_bvh(bm, cache)
        self.bmf = None

    @property
    def is_valid(self):
        return self.bm.is_valid and (self.bmf is None or self.bmf.is_valid)

    def find(self, context, mw, co, keep=None):
        self.bmf = None
        if not self.is_valid or not co:
            return
        near, _n, j, _d = self.faces.find_nearest(co, 1.84467e19)
        if j is not None and (to_region(context, mw @ co) - to_region(context, mw @ near)).length < Drawing.scale(HOVER_PX):
            try:
                self.bmf = self.bm.faces[j]
            except IndexError:
                self.bm.faces.ensure_lookup_table()
                self.bmf = self.bm.faces[j]
        if self.bmf is not None and not (keep(self.bmf) if keep is not None else not self.bmf.select):
            self.bmf = None


# ---------------------------------------------------------------------------------------
# edge-based modes

def line_intersection(p0, p1, q0, q1):
    """Intersection of the infinite 2D lines p0-p1 and q0-q1 (None when parallel)."""
    den = (p0.x - p1.x) * (q0.y - q1.y) - (p0.y - p1.y) * (q0.x - q1.x)
    if den == 0:
        return None
    t = ((p0.x - q0.x) * (q0.y - q1.y) - (p0.y - q0.y) * (q0.x - q1.x)) / den
    return Vector((p0.x + t * (p1.x - p0.x), p0.y + t * (p1.y - p0.y)))


def new_edge_ends(context, p0, p1, mouse, mi, stability, min_ratio=1.1):
    """Quad mode: where the far edge of a new quad goes (local positions of its two ends),
    for the selected edge p0-p1 (screen) and the mouse at the middle of the far edge.

    The far edge is as long as the selected one on screen. Its direction comes from a
    pivot on the selected edge's line: the point where the perpendicular to the
    selected-edge-to-mouse direction, at `stability` of the way toward the mouse, meets
    that line - but never closer to the selected edge's middle than 1.1 edge lengths. At
    stability 1 the far edge is square to the selected-edge-to-mouse direction (parallel to
    the selected edge only when the mouse is straight out from it); lower values fan it out.
    If an end misses every surface, the edge is shortened (halved, up to 32 times)."""
    v01 = p1 - p0
    length = v01.length
    if length < 1e-8:
        return None, None
    along = v01 / length
    mid = p0 + v01 / 2
    toward_mouse = mouse - mid
    if toward_mouse.length < 0.0001:
        return None, None
    probe = mid + toward_mouse * _clamp(stability, 0.01, 0.99)
    normal = Vector((-toward_mouse.y, toward_mouse.x))
    if normal.dot(v01) < 0:
        normal.negate()
    pivot = line_intersection(p0, p1, probe, probe + normal)
    if pivot is None:
        a = raycast_point_valid_sources(context, mouse + v01 / 2)
        b = raycast_point_valid_sources(context, mouse - v01 / 2)
        if a is None or b is None:
            return None, None
        return mi @ a, mi @ b
    offset = along.dot(pivot - mid)
    if abs(offset) < length * min_ratio:
        offset = length * min_ratio * (1 if offset > 0 else -1)
        pivot = mid + along * offset
    direction = (mouse - pivot).normalized()
    if direction.dot(normal) < 0:
        length = -length
    for _ in range(32):
        a = raycast_point_valid_sources(context, mouse + direction * (length / 2))
        b = raycast_point_valid_sources(context, mouse - direction * (length / 2))
        if a and b:
            return mi @ a, mi @ b
        length /= 2
    return None, None


def orient_far_side(p0, p1, p2, p3, a, b):
    """Order the far pair (p2, p3) / (a, b) so that p0 p1 p2 p3 runs around a quad:
    p2 nearer p1's side, and the sides p0-p3 and p1-p2 do not cross."""
    v01 = p1 - p0
    if v01.dot(p2 - p0) - v01.dot(p3 - p0) < 0:
        p2, p3, a, b = p3, p2, b, a
    if intersect_line_line_2d(p0, p3, p1, p2):
        p2, p3, a, b = p3, p2, b, a
    return p2, p3, a, b


def remaining_corner(face, keep_out):
    """The one face corner not in keep_out, or None."""
    rest = set(face.verts) - keep_out
    return next(iter(rest)) if len(rest) == 1 else None


def junction_toward(pc, po, a, b):
    """Screen point for a junction vertex: from corner pc toward the opposite corner po,
    as far as the mean of pc's distances to a and b, times 4/3."""
    return pc + (po - pc).normalized() * (((pc - a).length + (pc - b).length) / 1.5)


def flip_toward_view(context, mi, face):
    face.normal_update()
    if (mi.to_3x3() @ (context.region_data.view_rotation @ Vector((0.0, 0.0, -1.0)))).normalized().dot(face.normal) > 0:
        face.normal_flip()


class BuildStroke:
    """Screen-space stroke with pinned corners and curvature-weighted arc length.

    RDP removes subpixel jitter first. One restrained smoothing pass affects only
    non-corners; endpoints and detected sharp turns retain their exact input point.
    Resampling apportions segments between corner-delimited spans, then inverts
    the cumulative weighted length within each span. No mesh edits occur here.
    """

    MAX_SEGMENTS = 1024
    # dwell jitter: a held-still hand still sends small moves. No new sample
    # within DEAD_ZONE x tolerance (1.25 px x UI scale -> 4 px x UI scale) of the last
    # one, and a run of DWELL_SAMPLES or more samples that stays within DWELL_RADIUS x
    # the dead zone (8 px x UI scale) of its first sample collapses into one sample.
    DEAD_ZONE = 3.2
    DWELL_RADIUS = 2.0
    DWELL_SAMPLES = 4

    def __init__(self, anchor, tolerance=1.25):
        self.raw = [Vector(anchor)]
        self.tolerance = tolerance
        self.dead_zone = self.DEAD_ZONE * tolerance
        self.points = self.raw[:]
        self.corners = []
        self.weights = []
        self.segments = 1
        self.suggested_segments = 1
        self.wheel_offset = 0

    @staticmethod
    def simplify(points, tolerance):
        if len(points) < 3:
            return [p.copy() for p in points]
        keep = {0, len(points) - 1}
        stack = [(0, len(points) - 1)]
        while stack:
            lo, hi = stack.pop()
            a, b = points[lo], points[hi]
            ab = b - a
            length2 = ab.length_squared
            best, index = tolerance * tolerance, None
            for i in range(lo + 1, hi):
                t = _clamp((points[i] - a).dot(ab) / length2, 0, 1) if length2 else 0
                d = (points[i] - (a + ab * t)).length_squared
                if d > best:
                    best, index = d, i
            if index is not None:
                keep.add(index)
                stack.extend(((lo, index), (index, hi)))
        return [points[i].copy() for i in sorted(keep)]

    # cursor backtrack while the stroke is held: moving back along the path cuts the strip back (retract)
    width = 0.0            # the strip's screen width (px); 0: unknown (retract off)
    RETRACT_CAPTURE = 0.95  # x width: the cursor this close to an earlier part of the path, moving against it
    RETRACT_EXCLUDE = 1.0   # x width of arc right behind the tip that never captures (a curve's own end)

    def append(self, xy, final=False):
        xy = Vector(xy)
        if self.width > 0:
            if getattr(self, "_emptied", False) and len(self.raw) == 1 and \
                    (xy - self.raw[0]).length < self.RETRACT_CAPTURE * self.width:
                return False   # retracted to the start: stays empty (release adds nothing)
            if final and self._release_jitter(xy):
                return False   # a tiny pull-back while letting go: the tip stays
            if self._retract(xy, final):
                return True
        return self._append_plain(xy, final)

    def _append_plain(self, xy, final):
        if (xy - self.raw[-1]).length >= (0.01 if final else self.dead_zone):
            self.raw.append(xy)
            self._reach = max(getattr(self, "_reach", 0.0), (xy - self.raw[0]).length)   # farthest from the start
            return True
        return False

    def _retract(self, xy, final):
        """cut the path back to the nearest earlier point the cursor returns to while moving against the path."""
        raw = self.raw
        if len(raw) < 2:
            return False
        width = self.width
        move = xy - raw[-1]
        if move.length < (0.01 if final else self.dead_zone):
            return False
        arc = [0.0] * len(raw)   # arc length from the tip backward, per sample
        for k in range(len(raw) - 2, -1, -1):
            arc[k] = arc[k + 1] + (raw[k + 1] - raw[k]).length
        best = None
        for k in range(len(raw) - 1):
            if arc[k + 1] < self.RETRACT_EXCLUDE * width and k + 1 != len(raw) - 1:
                continue
            a, b = raw[k], raw[k + 1]
            ab = b - a
            if ab.length_squared < 1e-12:
                continue
            t = _clamp((xy - a).dot(ab) / ab.length_squared, 0, 1)
            q = a + ab * t
            if arc[k] - t * ab.length < self.RETRACT_EXCLUDE * width and move.dot(ab) >= 0:
                continue
            dist = (xy - q).length
            if best is None or dist < best[0]:
                best = (dist, k, q, ab)
        if best is None or best[0] > self.RETRACT_CAPTURE * width or move.dot(best[3]) >= 0:
            return False
        _, k, q, _ = best
        del raw[k + 1:]
        far = getattr(self, "_reach", 0.0) > 1.5 * width   # a stroke that once reached far cancels near its start
        near = self.RETRACT_CAPTURE * width if far else max(2 * self.dead_zone, 0.2 * width)
        if (q - raw[0]).length < near:   # cut back to the start: no stroke
            del raw[1:]
            self._emptied = True
        elif (q - raw[-1]).length > 0.01:
            raw.append(q)
        return True

    def _release_jitter(self, xy):
        """a backward move shorter than the dwell dead zone (the hand settling as the button is released)."""
        raw = self.raw
        if len(raw) < 2:
            return False
        move = xy - raw[-1]
        return move.length < self.dead_zone and move.dot(raw[-1] - raw[-2]) < 0

    def collapsed(self):
        """The raw samples with dwell clusters collapsed (see DEAD_ZONE): a cluster
        becomes its mean; one holding the stroke's first or last sample keeps that
        sample exactly (the anchor and the release point never move)."""
        raw, radius = self.raw, self.dead_zone * self.DWELL_RADIUS
        result, i, n = [], 0, len(raw)
        while i < n:
            j = i + 1
            while j < n and (raw[j] - raw[i]).length <= radius:
                j += 1
            if j - i >= self.DWELL_SAMPLES:
                point = raw[0] if i == 0 else raw[-1] if j == n else sum(raw[i:j], Vector((0, 0))) / (j - i)
                result.append(point.copy())
                i = j
            else:
                result.append(raw[i].copy())
                i += 1
        return result

    @staticmethod
    def turns(points):
        angles = [0.0] * len(points)
        for i in range(1, len(points) - 1):
            a, b = points[i] - points[i-1], points[i+1] - points[i]
            if a.length_squared > 1e-12 and b.length_squared > 1e-12:
                angles[i] = acos(_clamp(a.normalized().dot(b.normalized()), -1, 1))
        return angles

    def prepare(self, corner_angle):
        points = self.simplify(self.collapsed(), self.tolerance)
        # The cleaned screen-space path already measures visible curvature.
        # Keep enough initial segments to describe it even when world-space
        # density would reduce a centimetre-scale stroke to a single chord.
        # This is an automatic suggestion, not the wheel's hard lower bound.
        self.suggested_segments = 1
        angles = self.turns(points)
        # Retain the strongest corners if an exceptionally long stroke exceeds
        # the interactive segment limit; deterministic ties follow input order.
        corners = [i for i in range(1, len(points)-1) if angles[i] >= corner_angle]
        self.corners = sorted(sorted(corners, key=lambda i: (-angles[i], i))[:self.MAX_SEGMENTS-1])
        fixed = set(self.corners)
        self.points = [p.copy() if i == 0 or i == len(points)-1 or i in fixed else
                       p * .75 + (points[i-1] + points[i+1]) * .125
                       for i, p in enumerate(points)]
        angles = self.turns(self.points)
        lengths = [(b-a).length for a, b in zip(self.points, self.points[1:])]
        gain = sum(lengths) * .18 / max(corner_angle, .1)
        self.weights = [length + gain * (angles[i] + angles[i+1]) * .5
                        for i, length in enumerate(lengths)]

    def density_segments(self, density):
        """Screen-length density with an angular budget, independent of metres.

        Density 4 gives one segment per 80 logical pixels. Smooth curvature
        adds at most one segment per 30 degrees; pinned corners stay exact.
        """
        from math import ceil, pi
        length = sum((b-a).length for a, b in zip(self.points, self.points[1:]))
        turn = sum(self.turns(self.points))
        # tolerance tracks UI scale, so high DPI keeps the same logical spacing.
        spacing = 320.*max(self.tolerance/1.25, .25)/max(float(density), .01)
        base = max(self.minimum_segments, ceil(length/spacing), ceil(turn/(pi/6)))
        return min(self.MAX_SEGMENTS, base)

    @property
    def minimum_segments(self):
        return len(self.corners) + 1

    def set_segments(self, number):
        self.segments = max(self.minimum_segments, min(self.MAX_SEGMENTS, int(number)))

    def sample(self):
        if len(self.points) < 2:
            return self.points[:]
        bounds = [0] + self.corners + [len(self.points)-1]
        totals = [sum(self.weights[a:b]) for a, b in zip(bounds, bounds[1:])]
        extra = self.segments - len(totals)
        total = sum(totals)
        shares = [extra * w / total if total else 0 for w in totals]
        counts = [1 + int(s) for s in shares]
        for i in sorted(range(len(counts)), key=lambda i: (-(shares[i] % 1), i))[:extra-sum(int(s) for s in shares)]:
            counts[i] += 1
        result = [self.points[0].copy()]
        for a, b, count in zip(bounds, bounds[1:], counts):
            cumulative = [0.0]
            for weight in self.weights[a:b]:
                cumulative.append(cumulative[-1] + weight)
            for j in range(1, count):
                target = cumulative[-1] * j / count
                k = min(len(cumulative)-2, bisect_right(cumulative, target)-1)
                span = cumulative[k+1]-cumulative[k]
                t = (target-cumulative[k]) / span if span > 1e-12 else 0
                result.append(self.points[a+k].lerp(self.points[a+k+1], t))
            result.append(self.points[b].copy())
        return result

    def reference(self):
        """Pivot, reference vector and bounding diagonal for the damped Ctrl+move transform.

        A short or nearly closed stroke has no useful start-to-end vector, so the
        farthest sample from the pivot stands in for it.
        """
        anchor = self.points[0]
        pts = self.sample() or self.points
        diag = hypot(max(p.x for p in pts) - min(p.x for p in pts), max(p.y for p in pts) - min(p.y for p in pts))
        ref = self.points[-1] - anchor
        if ref.length < .25 * diag:
            ref = max((p - anchor for p in pts), key=lambda d: d.length_squared)
        return anchor, ref, diag

    def transformed_by(self, xform):
        """A 2D similarity about the pivot, xform = (real, imag) of a complex factor."""
        anchor = self.points[0]
        real, imag = xform
        return [anchor + Vector((real*d.x-imag*d.y, imag*d.x+real*d.y))
                for d in (p-anchor for p in self.sample())]



def strip_rows(points, ends, normal):
    """Sweep a constant world-width pair along a sampled centerline.

    The first row is the actual selected edge. Subsequent rows use the
    view-plane normal and centered tangents, including a bisector at corners.
    Width is independent of the similarity transform applied to the centerline.
    """
    if len(points) < 2:
        return [tuple(ends)]
    width = (ends[1] - ends[0]).length
    rows = [tuple(p.copy() for p in ends)]
    first = normal.cross(points[1] - points[0])
    sign = -1 if first.dot(ends[1] - ends[0]) < 0 else 1
    previous = (ends[1] - ends[0]).normalized()
    for i in range(1, len(points)):
        incoming = (points[i] - points[i-1]).normalized()
        tangent = incoming
        if i < len(points)-1:
            tangent = incoming + (points[i+1] - points[i]).normalized()
        across = normal.cross(tangent)
        across = across.normalized() * sign if across.length > 1e-8 else previous
        offset = across * (width / 2)
        rows.append((points[i] - offset, points[i] + offset))
        previous = across
    return rows


def strip_cell_normal(near, far, reference, epsilon):
    """Normal of the quad near[0], near[1], far[1], far[0], or None when it is
    collapsed, non-convex, or flipped against the previous cell's normal."""
    (a, b), (d, c) = near, far
    normal = (b-a).cross(c-a) + (c-a).cross(d-a)
    if normal.length < epsilon:
        return None
    cell = (a, b, c, d)
    limit = -normal.length_squared * 1e-8
    if any((cell[(i+1)%4]-cell[i]).cross(cell[(i+2)%4]-cell[(i+1)%4]).dot(normal) < limit for i in range(4)):
        return None
    if reference is not None and normal.dot(reference) <= 0:
        return None
    return normal


STRIP_SHRINK = .7      # inner-side offset factor per relief pass
STRIP_PASSES = 24      # passes per row before a fold counts as impossible


def _strip_inner_side(near, far, reference, epsilon):
    """The rail (0 or 1) with the reflex corners of a cell folded at a bend, or
    None when the cell is collapsed, flipped or twisted (not a bend)."""
    (a, b), (d, c) = near, far
    normal = (b-a).cross(c-a) + (c-a).cross(d-a)
    if normal.length < epsilon or (reference is not None and normal.dot(reference) <= 0):
        return None
    cell = (a, b, c, d)
    turns = [(cell[(i+1)%4]-cell[i]).cross(cell[(i+2)%4]-cell[(i+1)%4]).dot(normal) for i in range(4)]
    limit = -normal.length_squared * 1e-8
    second = turns[0] < limit or turns[1] < limit
    first = turns[2] < limit or turns[3] < limit
    if first == second:
        return None
    return 0 if first else 1


def relieve_strip(rows, centers, project=None):
    """Keep a swept strip valid at tight bends by narrowing the inner side.

    Where a cell would fold, the inner-side offsets of both its rows shrink
    toward the centerline (a local miter limit) until the cell is valid; the
    selected edge (row 0) never moves. Returns (rows, narrowed rows, folded):
    folded is True only when no narrowing gives valid cells. Strokes that do not
    fold are returned unchanged.
    """
    rows = [tuple(row) for row in rows]
    count = len(rows)
    epsilon = max((rows[0][1] - rows[0][0]).length_squared * 1e-10, 1e-20)
    offsets = [[p - centers[i] for p in row] for i, row in enumerate(rows)]
    scale = [[1., 1.] for _ in range(count)]
    original = list(rows)
    normals = [None] * count
    narrowed = set()

    def place(i):
        row = []
        for side in (0, 1):
            if scale[i][side] == 1.:
                row.append(original[i][side])
            else:
                p = centers[i] + offsets[i][side] * scale[i][side]
                row.append(project(p) if project is not None else p)
        rows[i] = tuple(row)

    i, budget = 1, STRIP_PASSES * count
    while i < count:
        normal = strip_cell_normal(rows[i-1], rows[i], normals[i-1], epsilon)
        if normal is not None:
            normals[i] = normal
            i += 1
            continue
        budget -= 1
        changed = False
        if i == 1:
            # The first cell folds only when the stroke does not leave the
            # selected edge (e.g. drawn along it): not a bend, refuse.
            return rows, len(narrowed), True
        side = _strip_inner_side(rows[i-1], rows[i], normals[i-1], epsilon)
        for side in () if side is None else (side,):
            for k in (i-1, i):
                if k == 0 or scale[k][side] == 0.:
                    continue
                shrunk = scale[k][side] * STRIP_SHRINK
                if shrunk < .02:
                    shrunk = 0.
                if shrunk == 0. and scale[k][1-side] == 0.:
                    continue    # both rails on the centerline: no quad left
                scale[k][side] = shrunk
                place(k)
                narrowed.add(k)
                changed = True
        if not changed or budget <= 0:
            return rows, len(narrowed), True
        i = max(1, i-1)     # the previous cell changed too
    return rows, len(narrowed), False


# ---------------------------------------------------------------------------------------
# Row strip from an edge chain. A single edge (K = 2) keeps strip_rows /
# relieve_strip above unchanged; these functions handle K >= 3 chain vertices.
# Every tolerance is relative to the chain's own length or measured in screen pixels,
# so the same stroke behaves the same on a 2 cm and a 20 cm object.

CHAIN_MAX = 256       # chain vertices accepted for a row strip
CELL_BUDGET = 8000    # rows x (K-1) cells in one preview


def chain_from_edges(edges):
    """Order one connected set of edges as an open chain.

    Returns (verts, None), or (None, reason) with reason one of CLOSED, BRANCHED,
    LONG, DEGENERATE. The walk starts at the end vertex with the lower index; the
    strip does not depend on that choice (orientation is taken from the stroke)."""
    adjacency = {}
    for edge in edges:
        a, b = edge.verts
        adjacency.setdefault(a, []).append((b, edge))
        adjacency.setdefault(b, []).append((a, edge))
    if any(len(links) >= 3 for links in adjacency.values()):
        return None, "BRANCHED"
    ends = [v for v, links in adjacency.items() if len(links) == 1]
    if not ends:
        return None, "CLOSED"
    if len(adjacency) > CHAIN_MAX:
        return None, "LONG"
    vert = min(ends, key=lambda v: v.index)
    order, came = [vert], None
    while True:
        step = [(other, edge) for other, edge in adjacency[vert] if edge is not came]
        if not step:
            break
        vert, came = step[0]
        order.append(vert)
    if len(order) != len(adjacency):
        return None, "BRANCHED"
    lengths = [(b.co - a.co).length for a, b in zip(order, order[1:])]
    total = sum(lengths)
    if total <= 0 or min(lengths) <= total * 1e-6:
        return None, "DEGENERATE"
    return order, None


def chain_sides_with_faces(verts):
    """BOTH when the chain already has faces on both sides (an edge with two faces, or
    one-sided edges whose faces lie on opposite sides); else the winding vote:
    > 0 the existing faces run along the chain order, < 0 against it, 0 no faces."""
    vote = 0
    seen = set()
    for a, b in zip(verts, verts[1:]):
        edge = next((e for e in a.link_edges if e.other_vert(a) is b), None)
        if edge is None:
            continue
        loops = list(edge.link_loops)
        if len(loops) >= 2:
            return "BOTH"
        if loops:
            along = loops[0].vert is a
            seen.add(along)
            vote += 1 if along else -1
    if len(seen) == 2:
        return "BOTH"
    return vote


def chain_profile(points, normal):
    """Rigid profile of a world-space chain about its arc-length midpoint.

    Returns dict(pivot, coords, sides, offsets, half, x_axis): coords are (x, y, z) in
    the frame X = chord in the view plane, Y = N x X, Z = N; sides -1 / 0 / +1 tell
    which half of the arc each vertex is on; offsets are the signed arc distances from
    the pivot. None when the chain is seen end-on (no in-plane chord)."""
    lengths = [(b - a).length for a, b in zip(points, points[1:])]
    total = sum(lengths)
    half = total / 2
    cumulative = [0.0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    pivot = points[-1].copy()
    for k, length in enumerate(lengths):
        if cumulative[k + 1] >= half:
            t = (half - cumulative[k]) / length if length > 0 else 0.0
            pivot = points[k].lerp(points[k + 1], t)
            break
    chord = points[-1] - points[0]
    x_axis = chord - normal * chord.dot(normal)
    if x_axis.length <= total * 1e-6:
        # Chord points at the viewer (e.g. a U seen end-on): use the widest span.
        far = max(points, key=lambda p: ((p - points[0]) - normal * (p - points[0]).dot(normal)).length)
        x_axis = (far - points[0]) - normal * (far - points[0]).dot(normal)
        if x_axis.length <= total * 1e-6:
            return None
    x_axis = x_axis.normalized()
    y_axis = normal.cross(x_axis)
    coords = [((p - pivot).dot(x_axis), (p - pivot).dot(y_axis), (p - pivot).dot(normal)) for p in points]
    tolerance = total * 1e-9
    offsets = [c - half for c in cumulative]
    sides = [-1 if o < -tolerance else 1 if o > tolerance else 0 for o in offsets]
    return dict(pivot=pivot, coords=coords, sides=sides, offsets=offsets, half=half, x_axis=x_axis,
                total=total)


def chain_rows(points, chain, profile, normal):
    """Sweep the chain's rigid profile along the sampled centerline `points`.

    Row 0 is the real chain. Row i is the profile rotated about the view axis so that
    its chord is perpendicular to the stroke there (the same frames as strip_rows),
    then moved to the stroke sample. Pairwise distances inside a row stay constant."""
    rows = [tuple(p.copy() for p in chain)]
    if len(points) < 2:
        return rows
    first = normal.cross(points[1] - points[0])
    sign = -1 if first.dot(chain[-1] - chain[0]) < 0 else 1
    previous = profile["x_axis"]
    for i in range(1, len(points)):
        incoming = (points[i] - points[i-1]).normalized()
        tangent = incoming
        if i < len(points)-1:
            tangent = incoming + (points[i+1] - points[i]).normalized()
        across = normal.cross(tangent)
        across = across.normalized() * sign if across.length > 1e-8 else previous
        forward = normal.cross(across)
        rows.append(tuple(points[i] + across * x + forward * y + normal * z for x, y, z in profile["coords"]))
        previous = across
    return rows


def chain_epsilon(row):
    shortest = min((b - a).length for a, b in zip(row, row[1:]))
    return max(shortest * shortest * 1e-10, 1e-30)


def chain_cells_valid(rows, epsilon):
    """Every cell of a K-column strip is convex, non-collapsed and not flipped
    against the cell before it in its own column."""
    normals = [None] * (len(rows[0]) - 1)
    for near, far in zip(rows, rows[1:]):
        for j in range(len(near) - 1):
            normal = strip_cell_normal((near[j], near[j+1]), (far[j], far[j+1]), normals[j], epsilon)
            if normal is None:
                return False
            normals[j] = normal
    return True


def relieve_chain(rows, centers, profile, project=None):
    """relieve_strip for K columns: at a tight bend the folding column and every
    column further out on that side shrink toward the stroke (x0.7 per pass); the
    chain (row 0) never moves. Returns (rows, narrowed rows, folded, along): along is
    True when the first row already folds (a chain edge runs along the stroke)."""
    rows = [tuple(row) for row in rows]
    count, width = len(rows), len(rows[0])
    epsilon = chain_epsilon(rows[0])
    offsets = [[p - centers[i] for p in row] for i, row in enumerate(rows)]
    original = list(rows)
    scale = [[1.] * width for _ in range(count)]
    normals = [[None] * (width - 1) for _ in range(count)]
    sides, distance = profile["sides"], [abs(o) for o in profile["offsets"]]
    narrowed = set()

    def place(i):
        row = []
        for j in range(width):
            if scale[i][j] == 1.:
                row.append(original[i][j])
            else:
                p = centers[i] + offsets[i][j] * scale[i][j]
                row.append(project(p, j) if project is not None else p)
        rows[i] = tuple(row)

    i, budget = 1, STRIP_PASSES * count * (width - 1)
    while i < count:
        bad = None
        for j in range(width - 1):
            reference = normals[i-1][j] if i > 1 else None
            normal = strip_cell_normal((rows[i-1][j], rows[i-1][j+1]), (rows[i][j], rows[i][j+1]),
                                       reference, epsilon)
            if normal is None:
                bad = j
                break
            normals[i][j] = normal
        if bad is None:
            i += 1
            continue
        if i == 1:
            # Unrelieved first cells fail only when a chain edge runs along the stroke.
            return rows, len(narrowed), True, 1 not in narrowed
        budget -= 1
        changed = False
        inner = _strip_inner_side((rows[i-1][bad], rows[i-1][bad+1]), (rows[i][bad], rows[i][bad+1]),
                                  normals[i-1][bad], epsilon)
        # The reflex column; when the cell cannot tell (a curved profile turned at a
        # sharp corner), narrow both of its columns.
        for column in ((bad + inner,) if inner is not None else (bad, bad + 1)):
            side = sides[column]
            outward = [k for k in range(width) if side and sides[k] == side and distance[k] >= distance[column]]
            for k in (i-1, i):
                if k == 0:
                    continue
                touched = False
                for j in outward:
                    if scale[k][j] == 0.:
                        continue
                    shrunk = scale[k][j] * STRIP_SHRINK
                    scale[k][j] = 0. if shrunk < .02 else shrunk
                    touched = True
                if touched:
                    place(k)
                    narrowed.add(k)
                    changed = True
        if not changed or budget <= 0:
            return rows, len(narrowed), True, False
        i = max(1, i-1)
    return rows, len(narrowed), False, False


def _cross2(o, a, b):
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)


def _segments_cross(a, b, c, d):
    """Proper crossing of 2D segments a-b and c-d (touching ends do not count)."""
    scale = max((b - a).length_squared, (d - c).length_squared) * 1e-9
    d1, d2 = _cross2(c, d, a), _cross2(c, d, b)
    d3, d4 = _cross2(a, b, c), _cross2(a, b, d)
    return ((d1 > scale and d2 < -scale) or (d1 < -scale and d2 > scale)) and \
           ((d3 > scale and d4 < -scale) or (d3 < -scale and d4 > scale))


def strip_overlaps(rows_screen):
    """Grid segments of a strip (screen space) that cross a non-neighbouring segment.

    rows_screen: rows of 2D points (None = off screen). Segments are row edges
    (i, j)-(i, j+1) and column edges (i, j)-(i+1, j); pairs whose first ends lie in
    neighbouring cells (|di| <= 1 and |dj| <= 1) are not compared. Uniform grid in
    pixels (cell = median segment length). Returns the crossing segments as
    ((i, j), (i2, j2)) index pairs."""
    segments = []
    count = len(rows_screen)
    for i, row in enumerate(rows_screen):
        for j, p in enumerate(row):
            if p is None:
                continue
            if j + 1 < len(row) and row[j+1] is not None:
                segments.append(((i, j), (i, j+1)))
            if i + 1 < count and rows_screen[i+1][j] is not None:
                segments.append(((i, j), (i+1, j)))
    if len(segments) < 2:
        return []

    def point(key):
        return rows_screen[key[0]][key[1]]
    lengths = sorted((point(b) - point(a)).length for a, b in segments)
    size = max(lengths[len(lengths) // 2], 1.0)
    grid, big = {}, []
    for index, (a, b) in enumerate(segments):
        pa, pb = point(a), point(b)
        x0, x1 = int(min(pa.x, pb.x) // size), int(max(pa.x, pb.x) // size)
        y0, y1 = int(min(pa.y, pb.y) // size), int(max(pa.y, pb.y) // size)
        if (x1 - x0 + 1) * (y1 - y0 + 1) > 256:
            big.append(index)
            continue
        for gx in range(x0, x1 + 1):
            for gy in range(y0, y1 + 1):
                grid.setdefault((gx, gy), []).append(index)
    pairs = set()
    for bucket in grid.values():
        for m in range(len(bucket)):
            for n in range(m + 1, len(bucket)):
                pairs.add((bucket[m], bucket[n]))
    for index in big:
        for other in range(len(segments)):
            if other != index:
                pairs.add((min(index, other), max(index, other)))
    hits = set()
    for m, n in pairs:
        (a, b), (c, d) = segments[m], segments[n]
        if abs(a[0] - c[0]) <= 1 and abs(a[1] - c[1]) <= 1:
            continue
        if _segments_cross(point(a), point(b), point(c), point(d)):
            hits.add(m)
            hits.add(n)
    return [segments[k] for k in sorted(hits)]


class MeshPenLogic:
    """Decides and performs Build insertions (interface used by ops/polypen.py)."""

    def __init__(self, context, event):
        self.mw = context.edit_object.matrix_world
        self.mi = self.mw.inverted()
        self.bm = self.em = None
        self.vert_lookup = self.edge_lookup = self.face_lookup = None
        self.selected = None
        self.bme = None          # an edge remembered between moves (edge-based modes)
        self.mouse = None
        self.hit = None
        self.state = Action.NONE
        # The owner immediately plans at the real cursor. A warm-up update at
        # (0, 0) built the lookups, raycast and selection only to plan again.

    # -- selection snapshot (taken once, dropped by the pen after a commit) --
    def _selection(self):
        bm = self.bm
        return {BMVert: {v for v in bm.verts if v.select and not v.hide},
                BMEdge: {e for e in bm.edges if e.select and not e.hide},
                BMFace: {f for f in bm.faces if f.select and not f.hide}}

    def _screen(self, context, co):
        return to_region(context, self.mw @ co)

    def _edge_screen_distance(self, context, edge, co):
        a, b = (self._screen(context, v.co) for v in edge.verts)
        return segment_distance(self._screen(context, co), a, b)

    def update(self, context, event, insert_mode, parallel_stable, quad_preserve, constrain_edge_vert):
        self.insert_mode = insert_mode
        self.parallel_stable = parallel_stable
        self.quad_preserve = quad_preserve and insert_mode in ("TRI/QUAD", "QUAD-ONLY")
        self.constrain = constrain_edge_vert

        if not self.bm or not self.bm.is_valid:
            self.bm = bmesh.from_edit_mesh(context.edit_object.data)
            self.em = context.edit_object.data
            self.selected = None
            self.vert_lookup = self.edge_lookup = self.face_lookup = None
        cache = {}
        if self.vert_lookup is None or not self.vert_lookup.is_valid:
            self.vert_lookup = _VertLookup(self.bm, cache)
        if self.edge_lookup is None or not self.edge_lookup.is_valid:
            self.edge_lookup = _EdgeLookup(self.bm, cache)
        if self.face_lookup is None or not self.face_lookup.is_valid:
            self.face_lookup = _FaceLookup(self.bm, cache)
        if self.selected is None:
            self.selected = self._selection()
        # Tri/Quad or quads only with one selected vertex: a curve stroke from it
        if insert_mode in ("TRI/QUAD", "QUAD-ONLY") and len(self.selected[BMVert]) == 1:
            self.insert_mode = "EDGE-ONLY"
            self.quad_preserve = False

        self.mouse = Vector((event.mouse_region_x, event.mouse_region_y))
        self.state = Action.NONE
        self.nothing_to_build = False   # set where the click could only fail (see _decide)
        found = raycast_valid_sources(context, self.mouse)
        self.hit = found["co_local"] if found else None
        if self.hit is None:
            return

        shown = lambda v: not is_bmvert_hidden(context, v)
        self.vert_lookup.find(context, self.mw, self.hit, keep=shown)
        if self.vert_lookup.bmv:
            self.hit = Vector(self.vert_lookup.bmv.co)
        self.edge_lookup.find(context, self.mw, self.hit, keep=lambda e: all(map(shown, e.verts)))
        self.face_lookup.find(context, self.mw, self.hit, keep=lambda f: all(map(shown, f.verts)))
        self.state = self._decide(context)

    # -- decision ---------------------------------------------------------------------------
    def _decide(self, context):
        try:
            action = self._choose(context)
        except ZeroDivisionError:   # geometry seen exactly edge-on: no screen direction to build along
            self.nothing_to_build = True
            return Action.NONE
        if action != Action.NONE and self._would_fail(context, action):
            # Build used to fail on this click (an error, the click undone): there is
            # nothing to build, so nothing happens
            self.nothing_to_build = True
            return Action.NONE
        return action

    def _would_fail(self, context, action):
        hv = self.vert_lookup.bmv
        if action == Action.EDGE_QUAD:
            ends = [v for v in self.far_verts if v]
            corners = list(self.bme.verts) + ends
            if len(set(corners)) != len(corners):
                return True          # both new corners on the same vertex
            if len(ends) == 2 and any(set(f.verts) == set(corners) for f in corners[0].link_faces):
                return True          # that quad exists already
        elif action == Action.EDGE_TRI:
            a, b = self.bme.verts      # a triangle on its own edge, where no face is there to split
            return bool(hv) and hv in (a, b) and not set(a.link_faces) & set(b.link_faces)
        elif action == Action.EDGE_SPLIT_EDGE and hv:
            return not self._cut_from_edge(context)       # every cut is on the vertex's own edges
        elif action == Action.WIRE_VERT_SPLIT_FACE:
            wire, face = wire_face_to(self.bmv, hv)
            return split_lists(face, wire + [hv]) is None
        elif action == Action.TRI_QUAD and hv:   # the tip on the triangle's own corner would collapse it
            return self.bmf is not None and hv in self.bmf.verts
        return False

    def _choose(self, context):
        mode = self.insert_mode
        hv, he = self.vert_lookup.bmv, self.edge_lookup.bme
        sel = self.selected
        if mode is None or mode == "VERT-ONLY":
            return Action.VERT
        if not sel[BMVert]:
            return Action.SPLIT_EDGE if he and not hv else Action.VERT

        if he and not he.hide:
            b = self.bme
            if b and b.is_valid and b != he:
                shared = set(b.link_faces) & set(he.link_faces)
                if len(shared) == 1:
                    face = next(iter(shared))
                    if self.quad_preserve and len(face.verts) == 4 and touching(b, he):
                        return Action.SPLIT_QUAD_CENTER
                    return Action.SPLIT_QUAD
            if he.select:
                return Action.SPLIT_EDGE
            if self._find_loop_cut(context):
                return Action.SPLIT_QUAD_EDGES

        if not sel[BMEdge] or mode == "EDGE-ONLY":
            self.bmv = min(sel[BMVert], key=lambda v: (self.hit - v.co).length)
            if hv:
                if hv == self.bmv and (hv.link_faces or not hv.link_edges) and not crossings(
                        context, self.mw, hv, self._screen(context, hv.co), self._screen(context, hv.co)):
                    # the selected vertex itself, where an edge to itself cannot be made
                    # (the end of a loose edge is still picked up, as before)
                    self.nothing_to_build = True
                    return Action.NONE
                found = wire_face_to(self.bmv, hv)
                if found:
                    if set(self.bmv.link_edges) & set(hv.link_edges):
                        self.nothing_to_build = True
                        return Action.NONE     # already joined: nothing to build
                    if inside_face(self.bmv.co, found[1]):
                        return Action.WIRE_VERT_SPLIT_FACE
                    # a wire outside that face (e.g. out from a corner and back onto its neighbour) closes
                    # with a plain edge, as onto any other vertex (Ctrl+Wheel segments keep working)
                return Action.VERT_EDGE
            if he:
                found = wire_face_to(self.bmv, he)
                if found and inside_face(self.bmv.co, found[1]):
                    return Action.WIRE_SPLIT_EDGE_FACE   # a wire outside that face: an ordinary edge target below
                if any(v in he.verts for v in sel[BMVert]):
                    return Action.SPLIT_EDGE
                return Action.VERT_SPLIT_EDGE
            return Action.VERT_EDGE
        return self._decide_edge_modes(context)

    def _near_cache(self):
        """SelectedEdgeScreen of the selected edges, kept as long as this selection snapshot."""
        cached = getattr(self, "_near", None)
        if cached is None or cached[0] is not self.selected:
            screen = SelectedEdgeScreen(self.selected[BMEdge])
            cached = (self.selected, screen, QuadRings(screen.edges))
            self._near = cached
        return cached[1]

    def _rings(self):
        self._near_cache()
        return self._near[2]

    def selected_chain(self, edge):
        """(size, chain_from_edges(the selected edges connected to edge), or None for a single edge), kept per
        selection snapshot for every edge walked (the selection and the mesh do not change while it is kept).
        The walk stops at the first vertex with 3 or more selected edges: chain_from_edges returns (None, "BRANCHED")
        for any component with such a vertex, so a hover near a large selected area does not walk all of it; the
        size is then reported as the edges found (more than 1), which is all the caller reads it for."""
        cached = getattr(self, "_chains", None)
        if cached is None or cached[0] is not self.selected:
            cached = (self.selected, {})
            self._chains = cached
        found = cached[1].get(edge)
        if found is None:
            edges = self.selected[BMEdge]
            component, stack, branched = {edge}, [edge], False
            while stack and not branched:
                current = stack.pop()
                for vert in current.verts:
                    links = [other for other in vert.link_edges if other in edges]
                    if len(links) >= 3:
                        branched = True
                        break
                    for other in links:
                        if other not in component:
                            component.add(other)
                            stack.append(other)
            if branched:
                found = (max(2, len(component)), (None, "BRANCHED"))
            else:   # the whole connected component
                found = (len(component), chain_from_edges(component) if len(component) > 1 else None)
            for other in component:
                cached[1][other] = found
        return found

    def near_selected_edges(self, persmat, mw, width, height, p, limit):
        """The selected edges (selection order) whose segment can be within limit px of p on screen
        (persmat, mw, width, height: the projection the caller's exact check uses)."""
        edges = self.selected[BMEdge]
        if len(edges) <= NEAR_EDGES_MIN:
            return edges
        return self._near_cache().within(persmat, mw, width, height, p, limit)

    def _projection(self, context):
        return context.region_data.perspective_matrix, self.mw, context.region.width, context.region.height

    def _nearest_selected_edge(self, context, edges):
        if edges is self.selected[BMEdge] and len(edges) > NEAR_EDGES_MIN:
            # only edges that can be the nearest (same set order, so ties resolve as before)
            edges = self._near_cache().nearest_candidates(*self._projection(context), self._screen(context, self.hit),
                                                          EDGE_END_SHARE, 1.0 - EDGE_END_SHARE)
        return min(edges, key=lambda e: self._edge_screen_distance(context, e, self.hit))

    def _decide_edge_modes(self, context):
        """Triangle, Tri/Quad and Quad modes with selected edges (spec 3.8)."""
        mode = self.insert_mode
        hv, he, hf = self.vert_lookup.bmv, self.edge_lookup.bme, self.face_lookup.bmf
        edges = self.selected[BMEdge]

        # hovering another edge: bridge to it (or cut between both when faces lie between)
        if mode in ("TRI/QUAD", "QUAD-ONLY") and he and not hv:
            near = self._nearest_selected_edge(context, edges)
            sel_face = next((f for f in near.link_faces if f.select), None)
            if (mode == "QUAD-ONLY" or not sel_face or len(sel_face.verts) != 3) and \
                    not any(he in f.edges for f in near.link_faces):
                self.bme = near
                self.bme_hovered = he
                v0, v1 = near.verts
                v2, v3 = he.verts
                p0, p1, p2, p3 = (self._screen(context, v.co) for v in (v0, v1, v2, v3))
                p2, p3, v2, v3 = orient_far_side(p0, p1, p2, p3, v2, v3)
                if v2 in near.verts or v3 in near.verts:
                    return Action.EDGE_TRI     # the edges touch: as if hovering the far vertex
                at = self._screen(context, self.hit)
                w = (p3 - p2).normalized().dot(at - p2) / (p3 - p2).length
                if len(crossings(context, self.mw, near, p0 + (p1 - p0) * w, at)) > 1:
                    return Action.SPLIT_QUAD
                self.bridge_far = [v2, v3]
                return Action.EDGE_BRIDGE

        if mode == "QUAD-ONLY":
            near = self._nearest_selected_edge(context, edges)
            v0, v1 = near.verts
            p0, p1 = self._screen(context, v0.co), self._screen(context, v1.co)
            a, b = new_edge_ends(context, p0, p1, self.mouse, self.mi, self.parallel_stable)
            if a is None or b is None:
                return Action.NONE
            p2, p3 = self._screen(context, a), self._screen(context, b)
            _p2, _p3, a, b = orient_far_side(p0, p1, p2, p3, a, b)
            self.far_verts = []
            # an end on an existing vertex uses it: unselected and shown, the hover rule (never a hidden one)
            usable = lambda v: not v.select and not is_bmvert_hidden(context, v)
            for co in (a, b):
                self.vert_lookup.find(context, self.mw, co, keep=usable)
                self.far_verts.append(self.vert_lookup.bmv)
            self.vert_lookup.bmv = None
            self.far_cos = [v.co if v else co for v, co in zip(self.far_verts, (a, b))]
            self.bme = near
            return Action.EDGE_QUAD

        faces = self.selected[BMFace]
        if mode == "TRI/QUAD" and len(faces) == 1 and len(next(iter(faces)).edges) == 3:
            self.bmf = next(iter(faces))
            self.bme = self._nearest_selected_edge(context, self.bmf.edges)
            return Action.TRI_QUAD

        if self._find_loop_cut(context):
            return Action.SPLIT_QUAD_EDGES

        if len(edges) == 1 and hv:
            only = next(iter(edges))
            self.bme = only
            if only.is_boundary and self._border_side(only) > 0:
                return Action.EDGE_SPLIT_EDGE
            return Action.EDGE_TRI
        if len(edges) == 1 and hf:
            self.bme = next(iter(edges))
            return Action.EDGE_SPLIT_EDGE
        if len(self.selected[BMVert]) == 2 and len(edges) == 1:
            self.bme = next(iter(edges))
            return Action.EDGE_TRI
        if len(edges) > 1:
            self.bme = self._nearest_selected_edge(context, edges)
            return Action.EDGE_TRI
        return Action.NONE

    def _cut_from_edge(self, context):
        """Edges cut from the middle of the selected edge toward P; ending on a hovered
        vertex, the cuts through that vertex's own edges at the end are left out."""
        a, b = self.bme.verts
        start = self._screen(context, a.co + (b.co - a.co) / 2)
        cuts = crossings(context, self.mw, self.bme, start, self._screen(context, self.hit))
        hv = self.vert_lookup.bmv
        if hv:
            around = set(hv.link_edges)
            while cuts and cuts[-1][0] in around:
                cuts.pop()
        return cuts

    def _border_side(self, edge):
        """Positive when P is on the side of the border edge where its face is (3D)."""
        face = next(iter(edge.link_faces), None)
        v0, v1 = edge.verts
        along = (v1.co - v0.co).normalized()
        to_mid = centroid(face) - v0.co
        return (to_mid - along * along.dot(to_mid)).dot(self.hit - v0.co)

    # -- Junctions: a third vertex so that a cut through a quad corner keeps quads --
    def _junction_vert(self, target):
        """(corner, opposite) when the selected vertex and the hovered vertex `target`
        are two neighbours of one corner of a shared face."""
        src = self.bmv
        corners = {common_vert(e0, e1) for e0 in src.link_edges for e1 in target.link_edges if touching(e0, e1)}
        if len(corners) != 1:
            return None, None
        corner = next(iter(corners))
        face = next(iter(set(src.link_faces) & set(target.link_faces)), None)
        if not face:
            return None, None
        out = ({other_vert(e, target) for e in target.link_edges} | {other_vert(e, src) for e in src.link_edges}
               | {corner, src, target})
        return corner, remaining_corner(face, out)

    def _junction_edge(self, edge):
        """(corner, opposite) when the selected vertex is next to a corner of the hovered
        edge in a shared face."""
        src = self.bmv
        touch = [e for e in src.link_edges if touching(e, edge)]
        if len(touch) != 1:
            return None, None
        corner = common_vert(touch[0], edge)
        face = next(iter(set(src.link_faces) & set(edge.link_faces)), None)
        if not face:
            return None, None
        out = set(edge.verts) | {other_vert(e, src) for e in src.link_edges} | {corner, src}
        return corner, remaining_corner(face, out)

    def _junction_wire(self, edge):
        """For a one-edge wire into a quad that has one extra corner (5 corners): the
        quad corner across from the hovered edge, joined to the wire end after the split."""
        found = wire_face_to(self.bmv, edge)
        if not found:
            return None
        wire, face = found
        if len(wire) != 2 or len(face.verts) != 5:
            return None
        anchor_edges = set(wire[0].link_edges)
        if edge in anchor_edges:
            return None
        far = next((e for e in face.edges if not any(touching(e, a) for a in anchor_edges)), None)
        if not far or far == edge:
            return None
        return other_vert(far, common_vert(edge, far))

    def _loop_cut_starts(self, context, he):
        """The start edges of _find_loop_cut in exactly the old order (sorted by screen distance, stable on the
        set order), without the starts that provably cannot reach a target face (QuadRings.possible); the distance
        of those starts is never computed."""
        edges = self.selected[BMEdge]
        key = lambda e: self._edge_screen_distance(context, e, self.hit)
        if len(edges) <= NEAR_EDGES_MIN:
            yield from sorted(edges, key=key)
            return
        rings, screen = self._rings(), self._near_cache()
        args = (*self._projection(context), self._screen(context, self.hit), EDGE_END_SHARE, 1.0 - EDGE_END_SHARE, key)
        if not rings.unknown:   # every selected edge's component is known: decide them all at once
            comps = np.array(rings.selected, dtype=np.int64)
            verdict = np.zeros(len(rings.faces), dtype=bool)
            for c in np.unique(comps).tolist():
                verdict[c] = rings.possible(c, self.hit, he)
            yield from screen.ordered(*args, mask=verdict[comps])
            return
        verdict = {}

        def usable(i):
            c = rings.at(i)
            found = verdict.get(c)
            if found is None:
                found = verdict[c] = rings.possible(c, self.hit, he)
            return found
        yield from screen.ordered(*args, keep=usable)

    def _find_loop_cut(self, context):
        """Cut across a row of quads from a selected edge toward the mouse (see spec 3.6)."""
        he = self.edge_lookup.bme
        order = self._loop_cut_starts(context, he)   # lazy, the same order as sorting every selected edge
        target = None
        for start in order:
            came = {}   # face -> edge it was entered by, edge -> face it was reached from
            if he and touching(start, he):
                continue
            if start.is_boundary and self._border_side(start) < 0:
                continue          # the mouse is on the open side of a border edge
            frontier = [start]
            while frontier and target is None:
                edge = frontier.pop(0)
                for face in edge.link_faces:
                    if len(face.edges) != 4 or face in came:
                        continue
                    far = opposite_edge(face, edge)
                    if he and he != far and touching(he, far):
                        continue
                    came[face] = edge
                    came[far] = face
                    if he == far or inside_face(self.hit, face):
                        target = face
                        break
                    frontier.append(far)
            if target is not None:
                break
        if target is None:
            return False

        wire = not he or he not in target.edges
        chain, pairs = [], []
        face, visited = target, set()
        while face in came and face not in visited:
            visited.add(face)
            edge = came[face]
            chain.append(edge)
            ring = quad_corners_from(face, edge)
            pairs.append((ring[0], ring[3]))
            if edge not in came:
                break
            face = came[edge]
        chain.reverse()
        pairs.reverse()
        if not wire and chain[-1] != he:
            ring = list(target.verts)
            v0, v1 = he.verts
            if (ring.index(v1) + 1) % 4 == ring.index(v0):
                v0, v1 = v1, v0
            chain.append(he)
            pairs.append((v0, v1))

        corners = [self._screen(context, v.co) for v in quad_corners_from(target, came[target])]
        u, v = quad_uv(self.mouse, *corners)
        self.bme = None
        self.cut = {"start": start, "edges": chain, "pairs": pairs, "uv": (u, v), "wire": wire}
        return True

    # -- commit -----------------------------------------------------------------------------
    def _mirror_side(self, context, co, refs):
        """Move co onto a mirror plane where it crossed to the other side than refs. Each Mirror modifier's plane
        is its mirror object's (else the object's own), with that modifier's merge threshold; co changes in place."""
        obj = context.edit_object
        for m in obj.modifiers:
            if m.type != "MIRROR" or not m.show_viewport:
                continue
            to_plane = m.mirror_object.matrix_world.inverted_safe() @ obj.matrix_world if m.mirror_object else None
            t = m.merge_threshold

            def side(p, i, t=t, to_plane=to_plane):
                x = (to_plane @ p)[i] if to_plane is not None else p[i]
                return 0 if abs(x) <= t else (1 if x > 0 else -1)
            for i in range(3):
                if not m.use_axis[i]:
                    continue
                ref = next((side(v.co, i) for v in refs if side(v.co, i) != 0), 0)
                if ref != 0 and side(co, i) != ref:
                    if to_plane is None:
                        co[i] = 0
                    else:
                        p = to_plane @ co
                        p[i] = 0
                        co[:] = to_plane.inverted_safe() @ p
        return co

    def _source_point(self, context, xy):
        """Local point on a source under xy, or None (no sources, e.g. 3D mode, or a miss)."""
        sources = raycast.sources(context)
        if not sources:
            return None
        hit = raycast.ray_cast_sources(context, *raycast.ray(context, xy), sources)
        return self.mi @ hit.world if hit else None

    def _edge_point(self, context, edge, xy, source=True):
        """Where a vertex that splits `edge` under screen point xy goes: on a source
        under xy (retopology), else on the edge itself. Off the sources the general
        fallback - a view plane at a nearby vertex's depth - put cut vertices off the
        mesh, which twisted the cut faces (always the case in 3D mode: no sources).
        source=False: None where a source is under xy (the caller keeps its own point)."""
        found = self._source_point(context, xy)
        if found is not None:
            return found if source else None
        origin, direction = raycast.ray(context, xy)
        a, b = (self.mw @ v.co for v in edge.verts)
        return self.mi @ point_on_edge(a, b, origin, direction)

    def _face_point(self, context, face, xy):
        """A new vertex inside `face` under xy: on a source, else on the face's plane."""
        found = self._source_point(context, xy)
        if found is not None:
            return found
        origin, direction = raycast.ray(context, xy)
        normal = (self.mw.inverted_safe().transposed().to_3x3() @ face.normal).normalized()
        return self.mi @ point_on_plane(self.mw @ face.calc_center_median(), normal, origin, direction)

    def _split_edge_at_hit(self, context, edge, snap):
        """Split edge in its middle and move the new vertex to the mouse point (or onto
        the edge's line with Constrain); off the sources, onto the edge under the mouse."""
        a, b = edge.verts
        on_edge = self._edge_point(context, edge, self.mouse, source=False)
        _e, vert = bmesh.utils.edge_split(edge, a, 0.5)
        if on_edge is not None:
            vert.co = on_edge
        elif not self.constrain:
            vert.co = self.hit
        else:
            d = (b.co - a.co).normalized()
            vert.co = a.co + d * d.dot(self.hit - a.co)
            snap.append(vert)
        return vert

    def _chain_through(self, context, start, cuts, stop):
        """Split the crossed edges in order from vertex `start`, each new vertex on the
        surface under its crossing point and connected to the previous one. Returns
        (last vertex, first new vertex)."""
        prev, first = start, None
        for edge, xy in cuts:
            if edge == stop:
                break
            co = self._edge_point(context, edge, xy)
            _e, vert = bmesh.utils.edge_split(edge, edge.verts[0], 0.5)
            vert.co = co
            bmesh.ops.connect_verts(self.bm, verts=[prev, vert])
            first = first or vert
            prev = vert
        return prev, first

    def _close_wire(self, first_new):
        """When the selected vertex ends a wire inside its anchor's face, link it to the
        first vertex of the new cut and divide that face along the wire."""
        found = wire_face_around(self.bmv)
        if found and first_new:
            wire, face = found
            self.bm.edges.new((wire[-1], first_new))
            if first_new in face.verts:
                rebuild_split(self.bm, face, wire + [first_new])
            # else the cut left through another face: the wire stays a loose edge joined
            # to the cut (Build used to fail here and undo the whole click)


    def commit(self, context, event):
        if self.state == Action.NONE:
            return
        if not context.tool_settings.mesh_select_mode[0]:
            context.tool_settings.mesh_select_mode[0] = True
        snap, now, later = [], [], []
        free = not self.constrain
        bm = self.bm
        hv, he = self.vert_lookup.bmv, self.edge_lookup.bme
        act = self.state

        if act == Action.VERT:
            now = [hv or bm.verts.new(self.hit)]
            free = True

        elif act == Action.SPLIT_EDGE:
            a, b = he.verts
            on_edge = self._edge_point(context, he, self.mouse, source=False)
            _e, vert = bmesh.utils.edge_split(he, a, 0.5)
            if on_edge is not None:
                vert.co = on_edge
            elif not self.constrain:
                vert.co = self.hit
            else:
                d = (b.co - a.co).normalized()
                vert.co = a.co + d * d.dot(self.hit - a.co)
            snap.append(vert)
            now = [vert]

        elif act == Action.SPLIT_QUAD_EDGES:
            _u, v = self.cut["uv"]
            made = [bmesh.utils.edge_split(e, p[0], v)[1] for e, p in zip(self.cut["edges"], self.cut["pairs"])]
            for a, b in zip(made, made[1:]):
                bmesh.ops.connect_verts(bm, verts=[a, b])
            if self.cut["wire"]:
                tip = bm.verts.new(self.hit)
                bm.edges.new((tip, made[-1]))
                now, free = [tip], True
            else:
                now = [made[-1]]
            snap.extend(made)

        elif act == Action.VERT_EDGE:
            src = self.bmv
            cuts = crossings(context, self.mw, src, self._screen(context, src.co), self._screen(context, self.hit))
            if hv:
                cuts = [(e, xy) for e, xy in cuts if e not in hv.link_edges]
                dst = hv
            else:
                dst = bm.verts.new(self._mirror_side(context, self.hit, [src]))
                snap.append(dst)
            corner = opposite = third = None
            if hv and self.quad_preserve:
                corner, opposite = self._junction_vert(hv)
                if opposite:
                    reach = ((src.co - corner.co).length + (hv.co - corner.co).length) / 2
                    third = corner.co + (opposite.co - corner.co).normalized() * reach
                    third = self.mi @ nearest_point_valid_sources(context, self.mw @ third)
            pick = None
            if not cuts:
                if any(dst in f.verts for f in src.link_faces):
                    made = bmesh.ops.connect_verts(bm, verts=[src, dst])
                    if third:
                        _e, mid = bmesh.utils.edge_split(next(iter(made["edges"]), None), src, 0.5)
                        mid.co = third
                        bmesh.ops.connect_verts(bm, verts=[mid, opposite])
                else:
                    pick = next(iter(shared(src.link_edges, dst.link_edges)), None)
                    if not pick:
                        pick = bm.edges.new((src, dst))
                        free = True
                    wire = wire_from(dst)
                    if wire and any(inside_face(dst.co, f) for f in wire[0].link_faces):
                        pick = None
            else:
                last, first = self._chain_through(context, src, cuts, he)
                self._close_wire(first)
                if hv:
                    bmesh.ops.connect_verts(bm, verts=[last, dst])
                else:
                    bm.edges.new((last, dst))
                    free = True
            now = [dst]
            later = [pick] if pick and self.insert_mode != "EDGE-ONLY" else []

        elif act in (Action.WIRE_VERT_SPLIT_FACE, Action.WIRE_SPLIT_EDGE_FACE):
            if act == Action.WIRE_VERT_SPLIT_FACE:
                dst = hv
                wire, face = wire_face_to(self.bmv, dst)
            else:
                wire, face = wire_face_to(self.bmv, he)
                opposite = self._junction_wire(he) if self.quad_preserve else None
                dst = self._split_edge_at_hit(context, he, snap)
            bm.edges.new((wire[-1], dst))
            rebuild_split(bm, face, wire + [dst])
            if act == Action.WIRE_SPLIT_EDGE_FACE and opposite:
                bmesh.ops.connect_verts(bm, verts=[self.bmv, opposite])
            now = [dst]

        elif act == Action.VERT_SPLIT_EDGE:
            src = self.bmv
            cuts = [(e, xy) for e, xy in crossings(context, self.mw, src, self._screen(context, src.co),
                                                   self._screen(context, self.hit)) if e != he]
            third = opposite = None
            if self.quad_preserve:
                corner, opposite = self._junction_edge(he)
                if opposite:
                    reach = ((src.co - corner.co).length + (self.hit - corner.co).length) / 1.5
                    third = corner.co + (opposite.co - corner.co).normalized() * reach
                    third = self.mi @ nearest_point_valid_sources(context, self.mw @ third)
            dst = self._split_edge_at_hit(context, he, snap)
            if not cuts:
                if any(dst in f.verts for f in src.link_faces):
                    made = bmesh.ops.connect_verts(bm, verts=[src, dst])
                    if third:
                        _e, mid = bmesh.utils.edge_split(made["edges"][0], src, 0.5)
                        mid.co = third
                        bmesh.ops.connect_verts(bm, verts=[mid, opposite])
                else:
                    bm.edges.new((src, dst))
            else:
                last, first = self._chain_through(context, src, cuts, he)
                bmesh.ops.connect_verts(bm, verts=[last, dst])
                self._close_wire(first)
            now = [dst]

        elif act == Action.EDGE_TRI:
            a, b = self.bme.verts
            if hv:
                tip = hv
            else:
                tip = bm.verts.new(self._mirror_side(context, self.hit, self.bme.verts))
                snap.append(tip)
            face = next(iter(shared(a.link_faces, b.link_faces, tip.link_faces)), None)
            now = [tip]
            if face:      # the tip is a corner of a face the edge is on: divide that face
                if not shared(a.link_edges, tip.link_edges):
                    part, _l = bmesh.utils.face_split(face, a, tip)
                    later.append(part)
                    face = next(iter(shared(b.link_faces, tip.link_faces)), None)
                if not shared(b.link_edges, tip.link_edges):
                    part, _l = bmesh.utils.face_split(face, b, tip)
                    later.append(part)
            else:
                face = bm.faces.new((a, b, tip))
                flip_toward_view(context, self.mi, face)
            later.append(face)
            free = True

        elif act == Action.EDGE_BRIDGE:
            a, b = self.bme.verts
            c, d = self.bridge_far
            face = bm.faces.new((a, b, c, d))
            flip_toward_view(context, self.mi, face)
            now, later = [c, d], [face]

        elif act == Action.EDGE_QUAD:
            a, b = self.bme.verts
            far = []
            for vert, co in zip(self.far_verts, self.far_cos):
                if not vert or not vert.is_valid or vert.hide:   # never weld onto a hidden vertex
                    vert = bm.verts.new(self._mirror_side(context, co, self.bme.verts))
                    snap.append(vert)
                far.append(vert)
            face = bm.faces.new((a, b, far[0], far[1]))
            flip_toward_view(context, self.mi, face)
            now, later, free = far, [face], True

        elif act == Action.TRI_QUAD:
            if hv:
                tip = hv
            else:
                tip = bm.verts.new(self._mirror_side(context, self.hit, self.bmf.verts))
                snap.append(tip)
            _e, mid = bmesh.utils.edge_split(self.bme, self.bme.verts[0], 0.5)
            bmesh.ops.weld_verts(bm, targetmap={mid: tip})
            now, later, free = [tip], [self.bmf], True

        elif act == Action.EDGE_SPLIT_EDGE:
            cuts = self._cut_from_edge(context)
            prev = None
            for edge, xy in cuts:
                co = self._edge_point(context, edge, xy)
                _e, vert = bmesh.utils.edge_split(edge, edge.verts[0], 0.5)
                vert.co = co
                if prev:
                    bmesh.ops.connect_verts(bm, verts=[prev, vert])
                prev = vert
            if hv:
                bmesh.ops.connect_verts(bm, verts=[prev, hv])
                tip = hv
            elif he:
                tip = prev     # hovering an edge: the cut ends on it (Build used to fail)
            else:
                tip = bm.verts.new(self.hit)
                bm.edges.new((prev, tip))
            now, free = [tip], True

        elif act == Action.SPLIT_QUAD:
            e0, e1 = self.bme, he
            a0, b0 = e0.verts
            a1, b1 = e1.verts
            s0, t0, s1, t1 = (self._screen(context, v.co) for v in (a0, b0, a1, b1))
            pt = self.edge_lookup.co2d
            d0, d1 = t0 - s0, t1 - s1
            if d0.dot(d1) < 0:
                s0, t0, a0, b0 = t0, s0, b0, a0
            f = d1.dot(pt - s1) / d1.length / d1.length
            cuts = crossings(context, self.mw, self.bme, s0 + (t0 - s0) * f, s1 + (t1 - s1) * f)
            _e, n0 = bmesh.utils.edge_split(e0, a0, f)
            _e, n1 = bmesh.utils.edge_split(e1, a1, f)
            snap.extend([n0, n1])
            if len(cuts) > 1:
                prev = n0
                for edge, xy in cuts:
                    if edge == e0 or edge == e1:
                        continue
                    co = self._edge_point(context, edge, xy)
                    _e, vert = bmesh.utils.edge_split(edge, edge.verts[0], 0.5)
                    vert.co = co
                    bmesh.ops.connect_verts(bm, verts=[prev, vert])
                    prev = vert
                if not set(prev.link_faces).isdisjoint(n1.link_faces):
                    bmesh.ops.connect_verts(bm, verts=[prev, n1])
                else:
                    bm.edges.new((prev, n1))
            else:
                bmesh.ops.connect_verts(bm, verts=[n0, n1])
            now = [n1]

        elif act == Action.SPLIT_QUAD_CENTER:
            e0, e1 = self.bme, he
            c = common_vert(e0, e1)
            v0, v1 = other_vert(e0, c), other_vert(e1, c)
            face = next(iter(set(e0.link_faces) & set(e1.link_faces)))
            o = next(iter(set(face.verts) - {v0, v1, c}))
            po, pc, p0, p1 = (self._screen(context, v.co) for v in (o, c, v0, v1))
            pt = self.edge_lookup.co2d
            r0, r1 = p0 - pc, p1 - pc
            f = r1.dot(pt - pc) / r1.length / r1.length
            q0, q1 = pc + r0 * f, pc + r1 * f
            qn = junction_toward(pc, po, q0, q1)
            co0, co1 = self._edge_point(context, e0, q0), self._edge_point(context, e1, q1)
            con = self._face_point(context, face, qn)
            _e, n0 = bmesh.utils.edge_split(e0, c, 0.5)
            _e, n1 = bmesh.utils.edge_split(e1, c, 0.5)
            n0.co, n1.co = co0, co1
            mid = bm.verts.new(con)
            parts = (bm.faces.new([c, n0, mid, n1]), bm.faces.new([n0, v0, o, mid]),
                     bm.faces.new([n1, mid, o, v1]))
            for part in parts:
                copy_face_data(bm, part, face)
            bm.faces.remove(face)       # the quad becomes three quads around `mid`
            for part in parts:
                flip_toward_view(context, self.mi, part)
            now = [n1]

        else:
            raise NotImplementedError(f"Build action {act.name} is not handled by MeshPenLogic")

        for vert in snap:
            if vert.is_valid:
                vert.co = self.mi @ nearest_point_valid_sources(context, self.mw @ vert.co)
        bm.select_history.clear()
        for vert in bm.verts:
            vert.select_set(False)
        for elem in now:
            if elem:
                bm.select_history.add(elem)
                elem.select_set(True)
        bm.select_flush(True)
        bm.select_flush(False)
        # PolyPenState.commit owns the single normal/tessellation sync after
        # the edit. Flushing here also synchronized the same topology twice.
        self.vert_lookup = None
        self.hit = None
        self.selected = None
        return now, later, free

    # -- preview ----------------------------------------------------------------------------
    def draw(self, context):
        try:
            self._draw(context)
        except ZeroDivisionError:
            pass    # an edge of zero length on screen (vertices on top of each other): no preview

    def _draw(self, context):
        if not self.mouse or not self.hit or not self.bm or not self.bm.is_valid:
            return
        if self.vert_lookup is None or not self.vert_lookup.is_valid:
            return
        if self.edge_lookup is None or not self.edge_lookup.is_valid:
            return
        if self.state in (Action.EDGE_TRI, Action.EDGE_QUAD, Action.TRI_QUAD) and not (self.bme and self.bme.is_valid):
            self.bme = None
            return
        theme = context.preferences.themes[0].view_3d
        hi = palette.preview_border(context)
        solid = (hi[0], hi[1], hi[2], 1.0)
        clear = (hi[0], hi[1], hi[2], 0.0)
        hollow = (theme.face_select[0], theme.face_select[1], theme.face_select[2], 0.0)
        size = theme.vertex_size + 4
        s = lambda co: self._screen(context, co)
        hv, he = self.vert_lookup.bmv, self.edge_lookup.bme
        inset = Drawing.scale(DASH_INSET)

        def dots(filled=(), ringed=()):
            with Drawing.draw(context, CC_2D_POINTS) as b:
                b.point_size(size)
                if filled:
                    b.color(solid)
                    for p in filled:
                        b.vertex(p)
                if ringed:
                    b.border(width=2, color=solid)
                    b.color(hollow)
                    for p in ringed:
                        b.vertex(p)

        def fill(triangles):
            with Drawing.draw(context, CC_2D_TRIANGLES) as b:
                b.color(theme.face_select)
                for a, c, e in triangles:
                    b.vertex(a).vertex(c).vertex(e)

        def dashes(segments):
            with Drawing.draw(context, CC_2D_LINES) as b:
                b.line_width(2)
                b.stipple(pattern=[5, 5], offset=0, color=hollow)
                b.color(solid)
                for a, c in segments:
                    b.vertex(a).vertex(c)

        if hv:
            with Drawing.draw(context, CC_2D_POINTS) as b:
                b.point_size(size)
                b.border(width=2, color=solid)
                b.color(clear)
                b.vertex(s(hv.co))

        act = self.state
        if act == Action.VERT:
            if not hv:
                dots(filled=[s(self.hit)])

        elif act == Action.SPLIT_EDGE:
            if not he:
                return
            p0, p1 = (s(v.co) for v in he.verts)
            pt = self.edge_lookup.co2d
            d = (p1 - p0).normalized() * inset
            dots(filled=[pt], ringed=[p0, p1])
            dashes([(p0 + d, pt - d), (p1 - d, pt + d)])

        elif act == Action.SPLIT_QUAD_EDGES:
            _u, v = self.cut["uv"]
            pts = [s(a.co + (b.co - a.co) * v) for a, b in self.cut["pairs"]]
            if self.cut["wire"]:
                pts.append(self.mouse)
            dots(ringed=pts)
            dashes([(a + (c - a).normalized() * inset, c - (c - a).normalized() * inset)
                    for a, c in zip(pts, pts[1:])])

        elif act == Action.WIRE_VERT_SPLIT_FACE:
            p0, pt = s(self.bmv.co), s(hv.co)
            d = (pt - p0).normalized() * inset
            dots(ringed=[p0, pt])
            dashes([(p0 + d, pt - d)])

        elif act in (Action.VERT_SPLIT_EDGE, Action.WIRE_SPLIT_EDGE_FACE):
            if not he:
                return
            p0, p1 = (s(v.co) for v in he.verts)
            pt = self.edge_lookup.co2d
            pn = s(self.bmv.co)
            po = pj = None          # Junctions: the opposite corner and the third vertex
            if self.quad_preserve:
                if act == Action.WIRE_SPLIT_EDGE_FACE:
                    far = self._junction_wire(he)
                    if far:
                        po, pj = s(far.co), s(self.bmv.co)
                else:
                    corner, far = self._junction_edge(he)
                    if far:
                        po = s(far.co)
                        pj = junction_toward(s(corner.co), po, pt, pn)
            cuts = [xy for e, xy in crossings(context, self.mw, self.bmv, pn, pt) if e != he]
            d = (p1 - p0).normalized() * inset
            dots(filled=[pt], ringed=[p0, p1, pn] + ([po, pj] if po else []) + cuts)
            lines = [(p0 + d, pt - d), (p1 - d, pt + d)]
            lines += [(pj, pt), (pj, pn), (pj, po)] if po else [(pn, pt)]
            dashes(lines)

        elif act in (Action.VERT_EDGE, Action.EDGE_SPLIT_EDGE):
            if act == Action.VERT_EDGE:
                start = self.bmv
                p0 = s(self.bmv.co)
            else:
                start = self.bme
                a, b = self.bme.verts
                p0 = s(a.co + (b.co - a.co) / 2)
            pt = s(hv.co if hv else self.hit)
            diff = pt - p0
            po = pj = None
            if act == Action.VERT_EDGE and self.quad_preserve and hv:
                corner, far = self._junction_vert(hv)
                if far:
                    po = s(far.co)
                    pj = junction_toward(s(corner.co), po, pt, s(hv.co))
            cuts = crossings(context, self.mw, start, p0, pt)
            if hv:
                cuts = [(e, xy) for e, xy in cuts if e not in hv.link_edges]
            dots(filled=[] if hv else [pt], ringed=[p0] + ([po, pj] if po else []) + [xy for _e, xy in cuts])
            if diff.length > inset:
                d = diff.normalized() * inset
                dashes([(p0, pj), (pt, pj), (po, pj)] if po else [(p0 + d, pt - d)])

        elif act == Action.EDGE_TRI:
            p0, p1 = (s(v.co) for v in self.bme.verts)
            pt = s(self.hit)
            to = lambda a, b: (b - a).normalized() * inset
            dots(filled=[] if hv else [pt], ringed=[p0, p1])
            dashes([(p0 + to(p0, pt), pt - to(p0, pt)), (p1 + to(p1, pt), pt - to(p1, pt)),
                    (p0 + to(p0, p1), p1 - to(p0, p1))])
            fill([(pt, p0, p1)])

        elif act == Action.EDGE_BRIDGE:
            p0, p1 = (s(v.co) for v in self.bme.verts)
            p2, p3 = (s(v.co) for v in self.bridge_far)
            to = lambda a, b: (b - a).normalized() * inset
            dots(ringed=[p0, p1, p2, p3])
            dashes([(p0 - to(p3, p0), p3 + to(p3, p0)), (p1 + to(p1, p2), p2 - to(p1, p2)),
                    (p0 + to(p0, p1), p1 - to(p0, p1)), (p2 + to(p2, p3), p3 - to(p2, p3))])
            fill([(p0, p1, p2), (p0, p2, p3)])

        elif act == Action.EDGE_QUAD:
            p0, p1 = (s(v.co) for v in self.bme.verts)
            p2, p3 = (s(co) for co in self.far_cos)
            to = lambda a, b: (b - a).normalized() * inset
            new = [p for p, v in zip((p2, p3), self.far_verts) if not v]
            old = [p for p, v in zip((p2, p3), self.far_verts) if v]
            dots(filled=new, ringed=[p0, p1] + old)
            lines = []
            if (p0 - p3).length > inset:
                lines.append((p0 - to(p3, p0), p3 + to(p3, p0)))
            if (p2 - p1).length > inset:
                lines.append((p1 + to(p1, p2), p2 - to(p1, p2)))
            lines += [(p2 + to(p2, p3), p3 - to(p2, p3)), (p0 + to(p0, p1), p1 - to(p0, p1))]
            dashes(lines)
            fill([(p0, p1, p2), (p0, p2, p3)])

        elif act == Action.TRI_QUAD:
            v0, v1, v2 = tri_from_edge(self.bmf, self.bme)
            p0, p1, p2 = s(v0.co), s(v1.co), s(v2.co)
            pt = s(self.hit)
            to = lambda a, b: (b - a).normalized() * inset
            dots(filled=[] if hv else [pt], ringed=[p0, p1, p2])
            dashes([(p0 + to(p0, pt), pt - to(p0, pt)), (p1 + to(p1, pt), pt - to(p1, pt)),
                    (p0 + to(p0, p2), p2 - to(p0, p2)), (p1 + to(p1, p2), p2 - to(p1, p2))])
            fill([(p0, pt, p1), (p0, p1, p2)])

        elif act in (Action.SPLIT_QUAD, Action.SPLIT_QUAD_CENTER):
            e0, e1 = self.bme, he
            if act == Action.SPLIT_QUAD:
                a0, b0 = e0.verts
                a1, b1 = e1.verts
                if (b0.co - a0.co).dot(b1.co - a1.co) < 0:
                    a1, b1 = b1, a1
            else:
                a0 = a1 = common_vert(e0, e1)
                b0, b1 = other_vert(e0, a0), other_vert(e1, a1)
            s0, t0, s1, t1 = (s(v.co) for v in (a0, b0, a1, b1))
            pt = self.edge_lookup.co2d
            r0, r1 = t0 - s0, t1 - s1
            _u0, u1 = r0 / r0.length, r1 / r1.length
            along = u1.dot(pt - s1)
            q1 = s1 + u1 * along
            q0 = s0 + r0 * (along / r1.length)
            cuts = [xy for _e, xy in crossings(context, self.mw, self.bme, q0, q1)]
            po = pj = None
            if act == Action.SPLIT_QUAD_CENTER:
                c = common_vert(e0, e1)
                if c:
                    face = next(iter(set(e0.link_faces) & set(e1.link_faces)))
                    o = next(iter(set(face.verts) - {other_vert(e0, c), other_vert(e1, c), c}), None)
                    if o:
                        po = s(o.co)
                        pj = junction_toward(s(c.co), po, q0, q1)
            dots(ringed=[q0, q1] + ([po, pj] if po else []) + cuts)
            if po:
                dashes([(q0, pj), (q1, pj), (po, pj)])
            else:
                d = (q1 - q0).normalized() * inset
                dashes([(q0 + d, q1 - d)])
