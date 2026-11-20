# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Snapping to source elements: Vertex, Edge and Edge Center.

PolyTailor places points on the source surface itself (Face Project / Face Nearest, see
core.raycast) and snaps to these three Blender snap targets itself, in Build,
left-drag, G and E. Doing it here rather than in Blender's transform keeps derived
objects out (Blender snaps an object sharing the edit mesh as edit data), keeps Ctrl
held in Build, and works on a plain click.

The target is chosen like Blender's own snapping (transform_snap_object.cc): within
SNAP_DISTANCE pixels (Blender's SNAP_MIN_DISTANCE, 30 px, independent of the UI scale)
the closest vertex, or the closest point on an edge when that is nearer. On an edge,
its parameter under the cursor decides: with Vertex ticked the outer parts of the edge
snap to its end vertex, with Edge Center the middle part to its centre, the rest to
the edge itself (snap_edge_points_impl). Every source near the cursor is searched -
not only the one under it - and elements hidden behind a source are skipped. With "Snap to Itself" the mesh being edited is
searched too, except the vertices being moved.

The other Blender targets (Increment, Grid, Face, Volume, Edge Perpendicular), and
every element target while "Use Blender Snapping" is on, leave left-drag / G / E to
Blender's own move (native_needed).
"""

import time

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree

from . import draw, raycast
from .. import prefs

TARGETS = {"VERTEX", "EDGE", "EDGE_MIDPOINT"}
SNAP_DISTANCE = 30          # pixels, Blender's SNAP_MIN_DISTANCE (not scaled by the UI scale)
SEARCH_SCALE = 1.25         # vertices gathered within the snap distance x this, then filtered
KEEP = 16                   # closest candidates kept per object and kind (the rest cannot win)
FULL_SCAN_LIMIT = 20_000    # off-cursor objects up to this many vertices are fully projected

_cache = {}     # evaluated object name -> {"sig", "kd", "co", "adj"}, see _mesh_data
last = None     # (world point, monotonic time, kind) of the latest snap, for the marker


# --- Blender settings -------------------------------------------------------------

def targets(context):
    """The ticked snap targets PolyTailor handles (empty while snapping is off)."""
    ts = context.scene.tool_settings
    if not ts.use_snap:
        return set()
    return set(getattr(ts, "snap_elements_base", set())) & TARGETS


def native_needed(context):
    """True when a ticked snap target is one PolyTailor does not implement: the move is then
    handed to Blender's own translate (native_translate)."""
    ts = context.scene.tool_settings
    base = set(getattr(ts, "snap_elements_base", set()))
    if not ts.use_snap or not base:
        return False
    return bool(base - TARGETS) or bool(getattr(context.scene.ppo, "use_native_snap", False))


def native_translate(context, **kwargs):
    """bpy.ops.transform.translate with the snap Target Selection passed explicitly.

    Blender remembers translate's last-used use_snap_self / use_snap_edit /
    use_snap_nonedit / use_snap_selectable (they are not SKIP_SAVE), and a remembered
    value overrides the scene's Target Selection: a translate once run with Include
    Active kept snapping onto the retopology mesh itself. Pass the scene values (PolyTailor's
    setup: other objects only) every time.
    """
    ts = context.scene.tool_settings
    return bpy.ops.transform.translate(
        "INVOKE_DEFAULT",
        use_snap_self=ts.use_snap_self or bool(getattr(context.scene.ppo, "snap_self", False)),
        use_snap_edit=ts.use_snap_edit,
        use_snap_nonedit=ts.use_snap_nonedit, use_snap_selectable=ts.use_snap_selectable,
        **kwargs)


# --- per-mesh data ------------------------------------------------------------------

def _signature(mesh):
    n = len(mesh.vertices)
    sample = tuple(tuple(mesh.vertices[i].co) for i in {0, n // 2, n - 1}) if n else ()
    return (mesh.as_pointer(), n, len(mesh.edges), sample)


def _mesh_data(obj, need_edges):
    """KD tree of the evaluated mesh vertices (local space) + vertex -> edge adjacency
    (CSR: neighbours of v are other[offsets[v]:offsets[v + 1]])."""
    mesh = obj.data
    sig = _signature(mesh)
    entry = _cache.get(obj.name)
    if entry is None or entry["sig"] != sig:
        n = len(mesh.vertices)
        co = np.empty(n * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        kd = KDTree(n)
        for i, c in enumerate(co):
            kd.insert(c, i)
        kd.balance()
        entry = {"sig": sig, "kd": kd, "co": co, "adj": None, "edges": None}
        _cache[obj.name] = entry
    if need_edges and entry["adj"] is None:
        e = np.empty(len(mesh.edges) * 2, dtype=np.int64)
        mesh.edges.foreach_get("vertices", e)
        e = e.reshape(-1, 2)
        both = np.concatenate([e, e[:, ::-1]])
        order = np.argsort(both[:, 0], kind="stable")
        offsets = np.searchsorted(both[order, 0], np.arange(len(entry["co"]) + 1))
        entry["adj"] = (offsets, both[order, 1])
        entry["edges"] = e
    return entry


def clear_cache(*_args):
    _cache.clear()


# --- search -------------------------------------------------------------------------

class _Screen:
    """Projection of one object's local points to region pixels."""

    def __init__(self, context, obj):
        region = context.region
        self.hw, self.hh = region.width / 2.0, region.height / 2.0
        self.pm = (np.array(context.region_data.perspective_matrix, dtype=np.float64)
                   @ np.array(obj.matrix_world, dtype=np.float64))

    def __call__(self, local):
        """(pixels n x 2, clip w, in front of the camera) for local points n x 3."""
        h = np.c_[local, np.ones(len(local))] @ self.pm.T
        w = h[:, 3]
        ok = w > 1e-9
        w = np.where(ok, w, 1.0)
        return np.c_[self.hw + self.hw * h[:, 0] / w, self.hh + self.hh * h[:, 1] / w], w, ok


def _world_radius(context, xy, depth_world, pixels):
    """World length of `pixels` on screen at the depth of depth_world, or None."""
    a = raycast.on_view_plane(context, xy, depth_world)
    b = raycast.on_view_plane(context, xy + Vector((pixels, 0.0)), depth_world)
    return None if a is None or b is None else (a - b).length


def _kd_range(data, obj, centre_local, world_length):
    scale = min(abs(v) for v in obj.matrix_world.to_scale()) or 1.0
    near = data["kd"].find_range(centre_local, world_length / scale)
    return np.fromiter((index for _co, index, _d in near), dtype=np.int64)


def _nearby_vertices(context, obj, data, project, xy, hit, ray, radius):
    """Indices of obj's vertices around the cursor to test, and the face under it.

    Three cases: the object under the cursor (KD query around the hit point), small
    objects (every vertex projected), dense objects off the cursor (KD query around
    the surface point closest to the cursor ray)."""
    reach = radius * SEARCH_SCALE
    co = data["co"]
    mw = obj.matrix_world
    if hit is not None and hit.obj is not None and hit.obj.name == obj.name:
        length = _world_radius(context, xy, hit.world, reach)
        if length is None:
            return None, -1
        return _kd_range(data, obj, mw.inverted() @ hit.world, length), hit.face
    if len(co) <= FULL_SCAN_LIMIT:
        scr, _w, ok = project(co.astype(np.float64))
        d2 = ((scr - np.array((xy[0], xy[1]))) ** 2).sum(axis=1)
        return np.nonzero(ok & (d2 <= reach ** 2))[0], -1
    origin, direction = ray
    centre = mw @ (sum((Vector(c) for c in obj.bound_box), Vector()) / 8)
    ray_point = origin + direction * (centre - origin).dot(direction)
    try:
        ok, seed_local, _normal, _face = obj.closest_point_on_mesh(mw.inverted() @ ray_point)
    except RuntimeError:
        return None, -1
    if not ok:
        return None, -1
    seed = mw @ seed_local
    seed_xy = raycast.to_screen(context, seed)
    if seed_xy is None or (seed_xy - xy).length > reach * 4:
        return None, -1
    length = _world_radius(context, xy, seed, reach)
    under = raycast.on_view_plane(context, xy, seed)
    if length is None or under is None:
        return None, -1
    return _kd_range(data, obj, seed_local, length + (seed - under).length), -1


def _edge_set(obj, data, verts, hit_face):
    """Edges (n x 2 vertex indices) to test: all of a small object, otherwise those at
    the gathered vertices plus the edges of the face under the cursor."""
    if len(data["co"]) <= FULL_SCAN_LIMIT:
        return data["edges"]
    offsets, other = data["adj"]
    verts = np.asarray(verts, dtype=np.int64)
    counts = offsets[verts + 1] - offsets[verts]
    first = np.repeat(verts, counts)
    # neighbours of every gathered vertex, gathered without a Python loop
    starts = np.repeat(offsets[verts] - np.cumsum(counts) + counts, counts) + np.arange(counts.sum())
    second = other[starts]
    pairs = [np.stack([np.minimum(first, second), np.maximum(first, second)], axis=1)]
    if 0 <= hit_face < len(obj.data.polygons):
        pairs.append(np.array([sorted(k) for k in obj.data.polygons[hit_face].edge_keys], dtype=np.int64))
    return np.unique(np.concatenate(pairs).reshape(-1, 2), axis=0)


def _gather(mw, co, verts, edges, project, mouse, r2, wanted, found_v, found_e):
    """Append (screen distance squared, world point) for vertices, and
    (screen distance squared, world point, world end 0, world end 1) for the closest
    point of each edge, within the snap distance."""
    if "VERTEX" in wanted and len(verts):
        vco = co[verts].astype(np.float64)
        scr, _w, ok = project(vco)
        d2 = ((scr - mouse) ** 2).sum(axis=1)
        hits = np.nonzero(ok & (d2 <= r2))[0]
        hits = hits[np.argsort(d2[hits])[:KEEP]]
        found_v += [(float(d2[k]), mw @ Vector(vco[k])) for k in hits]
    if not wanted & {"EDGE", "EDGE_MIDPOINT"} or edges is None or not len(edges):
        return
    p0, p1 = co[edges[:, 0]].astype(np.float64), co[edges[:, 1]].astype(np.float64)
    s0, w0, ok0 = project(p0)
    s1, w1, ok1 = project(p1)
    seg = s1 - s0
    l2 = (seg ** 2).sum(axis=1)
    t = np.clip(((mouse - s0) * seg).sum(axis=1) / np.where(l2 > 1e-12, l2, 1.0), 0.0, 1.0)
    d2 = ((s0 + seg * t[:, None] - mouse) ** 2).sum(axis=1)
    hits = np.nonzero(ok0 & ok1 & (l2 > 1e-12) & (d2 <= r2))[0]
    hits = hits[np.argsort(d2[hits])[:KEEP]]
    # screen parameter -> 3D parameter (perspective-correct interpolation)
    t, w0, w1 = t[hits], w0[hits], w1[hits]
    den = t * w0 + (1 - t) * w1
    t3 = t * w0 / np.where(den > 1e-12, den, 1.0)
    a, b = p0[hits], p1[hits]
    pts = a + (b - a) * t3[:, None]
    for k in range(len(hits)):
        found_e.append((float(d2[hits[k]]), mw @ Vector(pts[k]), mw @ Vector(a[k]), mw @ Vector(b[k])))


def _self_data(context, exclude):
    """The edit mesh as (object, local co n x 3, edges n x 2), without hidden elements
    and without the vertices in exclude (the ones being moved)."""
    obj = context.edit_object
    if obj is None or obj.type != "MESH":
        return None
    bm = bmesh.from_edit_mesh(obj.data)
    skip = set(exclude or ())
    keep = [v for v in bm.verts if not v.hide and v not in skip]
    if not keep:
        return None
    index = {v: i for i, v in enumerate(keep)}
    co = np.array([tuple(v.co) for v in keep], dtype=np.float64)
    edges = np.array([(index[e.verts[0]], index[e.verts[1]]) for e in bm.edges
                      if not e.hide and e.verts[0] in index and e.verts[1] in index],
                     dtype=np.int64).reshape(-1, 2)
    return obj, co, edges


def _ray_line_factor(origin, direction, a, b):
    """Parameter on the line a-b of its point closest to the view ray (Blender's
    isect_ray_line_v3), or None when they are parallel."""
    u = b - a
    w = a - origin
    uu, ud, dd = u.dot(u), u.dot(direction), direction.dot(direction)
    den = uu * dd - ud * ud
    if abs(den) < 1e-12:
        return None
    return (ud * w.dot(direction) - dd * w.dot(u)) / den


def _choose(context, xy, ray, found_v, found_e, wanted, r2, visible):
    """Blender's choice between the gathered vertices and edge points: (world, kind)."""
    best_v = next(((d2, world) for d2, world in sorted(found_v, key=lambda c: c[0])
                   if visible(world)), None)
    limit = best_v[0] if best_v else float("inf")
    best_e = next((c for c in sorted(found_e, key=lambda c: c[0])
                   if c[0] < limit - 1e-6 and visible(c[1])), None)
    if best_e is None:
        return (best_v[1], "VERTEX") if best_v else None
    _d2, on_edge, a, b = best_e
    use_v, use_e, use_m = ("VERTEX" in wanted), ("EDGE" in wanted), ("EDGE_MIDPOINT" in wanted)
    result = (on_edge, "EDGE") if use_e else None
    lam = _ray_line_factor(ray[0], ray[1], a, b) if (use_v or use_m) else None
    if lam is None:
        return result
    modes = use_e + use_v + use_m
    span = 1.0 / (2 * modes - 1)

    def near(world):
        pxy = raycast.to_screen(context, world)
        return pxy is not None and (pxy - xy).length_squared <= r2 and visible(world)

    if use_m:
        mid_span = span * (modes - 1)
        if mid_span < lam < 1.0 - mid_span and near((a + b) / 2):
            result = ((a + b) / 2, "EDGE_MIDPOINT")
    if use_v and (lam < span or lam > 1.0 - span):   # last, like Blender: wins over the centre
        end = a if lam < 0.5 else b
        if near(end):
            result = (end, "VERTEX")
    return result


def snap(context, xy, objs=None, exclude=None):
    """World point of the element snap near screen point xy, or None.

    exclude: BMVerts of the edit mesh being moved (only matters with Snap to Itself).
    """
    global last
    last = None
    wanted = targets(context)
    if not wanted:
        return None
    srcs = raycast.sources(context) if objs is None else objs
    snap_self = bool(getattr(context.scene.ppo, "snap_self", False))
    if not srcs and not snap_self:
        return None
    ray = raycast.ray(context, xy)
    hit = raycast.ray_cast_sources(context, *ray, srcs) if srcs else None
    radius = float(getattr(prefs.get(context), "snap_distance", SNAP_DISTANCE))
    r2 = radius * radius
    mouse = np.array((xy[0], xy[1]))
    need_edges = bool(wanted & {"EDGE", "EDGE_MIDPOINT"})
    found_v, found_e = [], []
    vertex_limit = int(getattr(prefs.get(context), "snap_vertex_limit", 300_000))
    for obj in srcs:
        if obj.type != "MESH" or not len(obj.data.vertices):
            continue
        if len(obj.data.vertices) > vertex_limit:
            continue   # e.g. a scan: indexing it would stall; surface snapping still uses it
        project = _Screen(context, obj)
        # cheap rejection: the object's screen bounding box must reach the cursor
        box, _w, ok = project(np.array([tuple(c) for c in obj.bound_box], dtype=np.float64))
        if ok.all():
            lo, hi = box.min(axis=0) - radius, box.max(axis=0) + radius
            if not (lo[0] <= xy[0] <= hi[0] and lo[1] <= xy[1] <= hi[1]):
                continue
        data = _mesh_data(obj, need_edges)
        verts, hit_face = _nearby_vertices(context, obj, data, project, xy, hit, ray, radius)
        if verts is not None:
            edges = _edge_set(obj, data, verts, hit_face) if need_edges else None
            _gather(obj.matrix_world, data["co"], verts, edges, project, mouse, r2, wanted,
                    found_v, found_e)
    if snap_self:
        own = _self_data(context, exclude)
        if own is not None:
            obj, co, edges = own
            _gather(obj.matrix_world, co, np.arange(len(co)), edges, _Screen(context, obj),
                    mouse, r2, wanted, found_v, found_e)
    offset = getattr(getattr(context.space_data, "overlay", None), "retopology_offset", 0.01)

    def visible(world):
        depth = (world - ray[0]).length
        return not raycast.is_occluded(context, world, srcs, bias=offset + 0.01 * depth)

    chosen = _choose(context, xy, ray, found_v, found_e, wanted, r2, visible)
    if chosen is None:
        return None
    world, kind = chosen
    last = (world.copy(), time.monotonic(), kind)
    return world


def for_move(context, mouse_xy, delta_xy, mw, originals, srcs, active=None, *, exclude=None):
    """Snap for a move of several vertices, like Blender's Snap Base.

    The snap target is searched around the mouse cursor (as in Blender). The base point
    follows Tool Settings > Snap Base: Closest (the moved vertex nearest the target),
    Center (bounding box), Median, or Active. Returns (screen delta, {vertex: world},
    world offset): the screen delta that brings the base point onto the target, the
    base vertex placed exactly on it, and the base -> target offset (for axis locks) -
    or None when nothing is in range.

    exclude optionally includes proportionally moved neighbors as well; they must
    not become stationary self-snap targets while the selected snap base moves.
    """
    if not targets(context) or not originals:
        return None
    target = snap(context, mouse_xy, srcs, exclude=originals.keys() if exclude is None else exclude)
    if target is None:
        return None
    target_xy = raycast.to_screen(context, target)
    if target_xy is None:
        return None
    worlds = {v: mw @ co for v, co in originals.items()}
    mode = getattr(context.scene.tool_settings, "snap_target", "CLOSEST")
    base_vert = None
    if mode == "ACTIVE" and active in worlds:
        base_vert = active
    elif mode in {"CENTER", "MEDIAN", "ACTIVE"}:   # Active without an active vertex: median
        pts = list(worlds.values())
        if mode == "CENTER":
            lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
            hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
            base = (lo + hi) / 2
        else:
            base = sum(pts, Vector((0.0, 0.0, 0.0))) / len(pts)
    if mode == "CLOSEST" or (base_vert is None and len(worlds) == 1):
        best = None
        for v, world in worlds.items():
            xy = raycast.to_screen(context, world)
            if xy is None:
                continue
            d = (xy + delta_xy - target_xy).length_squared
            if best is None or d < best[0]:
                best = (d, v)
        base_vert = best[1] if best else None
    if base_vert is None and mode == "CLOSEST":
        return None   # no moved vertex is on screen: no snap base
    if base_vert is not None:
        base = worlds[base_vert]
    base_xy = raycast.to_screen(context, base)
    if base_xy is None:
        return None
    return target_xy - base_xy, ({base_vert: target} if base_vert is not None else {}), target - base


# --- marker -------------------------------------------------------------------------

# Blender's snap symbols (view3d_cursor_snap.cc, cursor_point_draw), unit size
_SHAPES = {
    "VERTEX": [((-1, -1), (-1, 1), (1, 1), (1, -1), (-1, -1))],                 # square
    "EDGE": [((-1, -1), (1, 1), (-1, 1), (1, -1), (-1, -1))],                   # bow tie
    "EDGE_MIDPOINT": [((-1, -1), (0, 0.866), (1, -1), (-1, -1))],               # triangle
}


def _marker_color(context):
    """Blender draws the snap target in the theme's Active Object colour, alpha 192."""
    try:
        theme = context.preferences.themes[0].view_3d
        return (*tuple(theme.object_active)[:3], 192 / 255)
    except (AttributeError, IndexError):
        return (1.0, 0.667, 0.25, 192 / 255)


def _marker_size(context):
    """2.5 x the theme vertex size, like Blender's symbols."""
    try:
        vertex_size = context.preferences.themes[0].view_3d.vertex_size
    except (AttributeError, IndexError):
        vertex_size = 3
    return 2.5 * vertex_size * prefs.ui_scale(context)


def _draw_marker():
    context = bpy.context
    # shown while a move / placement runs, like Blender's (cleared by finish())
    if last is None or context.mode != "EDIT_MESH":
        return
    xy = raycast.to_screen(context, last[0])
    if xy is None:
        return
    size = _marker_size(context)
    color = _marker_color(context)
    with draw.Blend():
        # vertex / edge / edge centre only: surface placement (Face Project / Nearest,
        # the everyday case) shows no marker, to keep the view calm
        for shape in _SHAPES.get(last[2], ()):
            draw.polyline([(xy.x + x * size, xy.y + y * size) for x, y in shape], color, width=1.5)


def finish():
    """A move or placement ended: drop the marker."""
    global last
    last = None


_handle = None


def start():
    """while PolyTailor is in use (session)."""
    global _handle
    if _handle is None:
        _handle = bpy.types.SpaceView3D.draw_handler_add(_draw_marker, (), "WINDOW", "POST_PIXEL")


def stop():
    global _handle
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, "WINDOW")
        _handle = None


def unregister():
    stop()
    clear_cache()
