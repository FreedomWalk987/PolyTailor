# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Surface queries against source objects, with a view-plane fallback.

Sources are the visible mesh-like objects in Object Mode, except the edit object and
objects derived from it. They are only used while Blender's snapping toggle (magnet)
is on; otherwise points land on a plane facing the view, like native modeling.
"""

import time
from dataclasses import dataclass

import numpy as np
from bpy_extras.view3d_utils import location_3d_to_region_2d
from mathutils import Vector
from mathutils.geometry import intersect_line_plane, intersect_point_line
from .gn_inputs import input_values


@dataclass
class Hit:
    world: Vector
    obj: object = None      # evaluated source object hit by a ray cast
    face: int = -1          # polygon index on that object's evaluated mesh


def _has_faces(obj):
    """obj is evaluated. Non-mesh objects (curves, text, ...) only count when their
    evaluated geometry really has faces: ray_cast on a plain curve without bevel or
    fill crashes Blender."""
    if obj.type == "MESH":
        return bool(obj.data.polygons)
    # Decide from settings only: this runs inside draw callbacks, where creating and
    # freeing temporary meshes (to_mesh / to_mesh_clear) is not safe.
    if obj.type in {"SURFACE", "META", "FONT"}:
        return True
    if obj.type == "CURVE":
        data = obj.data
        if (getattr(data, "bevel_depth", 0.0) > 0.0 or getattr(data, "extrude", 0.0) > 0.0
                or getattr(data, "bevel_object", None) is not None):
            return True
        return getattr(data, "dimensions", "3D") == "2D" and getattr(data, "fill_mode", "NONE") != "NONE"
    return False


_derived_cache = {"key": None, "time": 0.0, "names": frozenset()}
derived_reasons = {}   # object name -> why it counts as derived (shown in the sidebar)


def _refers_to(value, target):
    try:
        return value is not None and value == target
    except (ReferenceError, TypeError):
        return False


def _node_tree_refers(tree, target, seen):
    if tree is None or tree.name in seen:
        return False
    seen.add(tree.name)
    for node in tree.nodes:
        for socket in node.inputs:
            if _refers_to(getattr(socket, "default_value", None), target):
                return True
        if _node_tree_refers(getattr(node, "node_tree", None), target, seen):
            return True
    return False


def _modifier_refers(mod, target):
    for prop in mod.bl_rna.properties:
        if prop.type == "POINTER" and _refers_to(getattr(mod, prop.identifier, None), target):
            return True
    if mod.type == "NODES":
        for value in input_values(mod):
            if _refers_to(value, target):
                return True
        if _node_tree_refers(getattr(mod, "node_group", None), target, set()):
            return True
    return False


def derived_objects(context):
    """Names of objects generated from the edit object: sharing its mesh data, or with a
    modifier / geometry nodes input pointing at it (e.g. a projected copy that follows
    it). Their geometry sits where the edited mesh is, so snapping onto them looked
    like snapping onto the mesh itself. Cached for a moment (called per redraw)."""
    edit = context.edit_object
    scene = getattr(context.scene, "ppo", None)
    if edit is None or scene is None or not scene.ignore_derived:
        return frozenset()
    key = (edit.as_pointer(), edit.data.as_pointer())
    now = time.monotonic()
    if _derived_cache["key"] == key and now - _derived_cache["time"] < 0.5:
        return _derived_cache["names"]
    names = set()
    derived_reasons.clear()
    for obj in context.scene.objects:
        if obj == edit:
            continue
        try:
            if obj.data is not None and obj.data == edit.data:
                names.add(obj.name)
                derived_reasons[obj.name] = "MESH"
                continue
            mods = [mod.name for mod in obj.modifiers if _modifier_refers(mod, edit)]
            if mods:
                names.add(obj.name)
                derived_reasons[obj.name] = "MODIFIER:" + ", ".join(mods)
        except (ReferenceError, AttributeError):
            continue
    _derived_cache.update(key=key, time=now, names=frozenset(names))
    return _derived_cache["names"]


def sources(context, *, visibility=False):
    ts = context.scene.tool_settings
    if not ts.use_snap and not visibility:
        return []
    edit = context.edit_object
    # The view layer's depsgraph as already evaluated by Blender's main loop. Unlike
    # evaluated_depsgraph_get() it never triggers an evaluation, which inside a draw
    # callback could free evaluated data the viewport is drawing with.
    depsgraph = context.view_layer.depsgraph
    result = []
    seen = set()   # (evaluated mesh, transform): identical copies are queried once
    scene = getattr(context.scene, "ppo", None)
    derived = derived_objects(context)
    for obj in context.visible_objects:
        if obj is edit or obj.mode != "OBJECT" or obj.type not in {"MESH", "CURVE", "SURFACE", "META", "FONT"}:
            continue
        if obj.name in derived:
            continue
        if not visibility and ((ts.use_snap_selectable and obj.hide_select) or
                (scene and scene.snap_only_selected and not obj.select_get()) or
                (scene and scene.source_object and obj != scene.source_object) or
                (scene and scene.source_collection and obj.name not in scene.source_collection.all_objects)):
            continue
        evaluated = obj.evaluated_get(depsgraph)
        if not _invertible(evaluated):
            continue   # a zero-scale axis (S Z 0): no local space to cast a ray or find a point in
        if _has_faces(evaluated):
            # objects sharing one mesh without modifiers (e.g. parts linked to a body)
            # evaluate to the very same geometry: keep one of them
            key = None
            if evaluated.type == "MESH":
                key = (evaluated.data.as_pointer(), tuple(map(tuple, evaluated.matrix_world)))
            if key is not None and key in seen:
                continue
            if key is not None:
                seen.add(key)
            result.append(evaluated)
    return result


def _invertible(obj):
    """matrix_world has an inverse: the same test Matrix.inverted() makes (it raises exactly when this is False)."""
    try:
        obj.matrix_world.inverted()
    except ValueError:
        return False
    return True


_warm = {}   # warmed source keys, oldest first (insertion order)


def prewarm(context):
    """Make Blender build the ray-cast lookup (BVH) of one not yet used source now, from
    an idle timer. Blender builds it on an object's first ray cast: about 0.2 s for a
    500k-face source, which otherwise lands on the first Ctrl preview or stroke after
    opening a file, adding a model or restarting. One source per call; a source whose
    evaluated mesh is unchanged is skipped (Blender keeps its lookup until then)."""
    for obj in sources(context):
        data = obj.data
        key = (obj.name, data.as_pointer() if data is not None else 0,
               len(data.polygons) if obj.type == "MESH" else 0)
        if key in _warm:
            continue
        if len(_warm) >= 4096:
            del _warm[next(iter(_warm))]   # the oldest key; never cleared in the middle of a warm-up pass
        _warm[key] = True
        try:
            obj.ray_cast(Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0)))
        except RuntimeError:
            pass   # no lookup data for this object (see ray_cast_sources)
        return True
    return False


def mouse(event):
    return Vector((event.mouse_region_x, event.mouse_region_y))


PRECISION = 0.1   # Shift: mouse motion counts one tenth (like Blender's transform)


class PreciseMouse:
    """Effective cursor for drags: motion made while Shift is held is scaled down."""

    def __init__(self, xy):
        self.raw = Vector(xy)
        self.xy = Vector(xy)

    def update(self, event):
        raw = mouse(event)
        self.xy += (raw - self.raw) * (PRECISION if event.shift else 1.0)
        self.raw = raw
        return self.xy.copy()


def ray(context, xy):
    return View(context).ray(xy)


ORTHO_REACH_SECONDS = .5   # visible objects' extent is measured again after this
_reach = {}   # RegionView3D pointer -> (view key, time, reach)


def _ortho_reach(context, camera, forward, view_distance, limit):
    """How far in front of the view an orthographic ray starts: just before the nearest
    visible object. bpy_extras starts it at the clipping range instead (500 m back by
    default), where a float is only good to ~30 µm: a drag on a millimetre part shook
    by many pixels and visibility tests failed."""
    rv3d = context.region_data
    key = (tuple(camera), tuple(forward))
    now = time.monotonic()
    cached = _reach.get(rv3d.as_pointer())
    if cached is not None and cached[0] == key and now - cached[1] < ORTHO_REACH_SECONDS:
        return cached[2]
    reach = view_distance
    for obj in getattr(context, "visible_objects", ()):
        try:
            mw = obj.matrix_world
            corners = [mw @ Vector(co) for co in obj.bound_box]
        except (AttributeError, ReferenceError):
            continue
        center = sum(corners, Vector((0, 0, 0))) / 8
        radius = max((co - center).length for co in corners)
        reach = max(reach, radius - (center - camera).dot(forward))
    reach = min(reach * 1.05 + view_distance * .05, limit)
    if len(_reach) > 64:
        _reach.clear()
    _reach[rv3d.as_pointer()] = (key, now, reach)
    return reach


def to_screen(context, world):
    return location_3d_to_region_2d(context.region, context.region_data, world)


class View:
    """The view's matrices, inverted once, for code that converts many points between
    screen and world in one event (a brush move, a multi-vertex drag, brush visibility).

    ray / to_screen / on_view_plane repeat bpy_extras.view3d_utils step by step with the
    same Matrix / Vector operations, so the results are bit-identical; the bpy_extras
    versions re-read and invert the view matrices on every call (3 inversions per
    point). Valid while the view does not change: make one per event.
    """

    def __init__(self, context):
        region, rv3d = context.region, context.region_data
        self.width, self.height = region.width, region.height
        self.persmat = rv3d.perspective_matrix.copy()
        self.persinv = self.persmat.inverted()
        self.viewinv = rv3d.view_matrix.inverted()
        self.is_perspective = rv3d.is_perspective
        self.is_camera = rv3d.view_perspective == "CAMERA"
        self.view_distance = getattr(rv3d, "view_distance", 0.0)
        self._p3 = self.persinv[3].xyz
        self._p33 = self.persinv[3][3]
        self._origin = self.viewinv.translation.copy()
        self._ortho_dir = -self.viewinv.col[2].xyz
        self._plane_normal = self.viewinv.col[2].copy()
        self._ortho = None
        if not self.is_perspective and not self.is_camera:
            # Orthographic rays in double precision from the view's own frame (no
            # inverted projection), starting just before the visible objects
            viewinv = np.linalg.inv(np.array(rv3d.view_matrix, dtype=np.float64))
            winmat = np.array(rv3d.window_matrix, dtype=np.float64)
            forward = -viewinv[:3, 2] / np.linalg.norm(viewinv[:3, 2])
            limit = float(np.linalg.norm(np.array(self.persinv.col[2].xyz)))
            reach = _ortho_reach(context, Vector(viewinv[:3, 3]), Vector(forward),
                                 self.view_distance or 1.0, max(limit, 1e-6))
            self._ortho = (viewinv, np.linalg.inv(winmat[:2, :2]), winmat[:2, 3], forward * reach)

    def vector(self, coord):
        """region_2d_to_vector_3d"""
        if self.is_perspective:
            out = Vector(((2.0 * coord[0] / self.width) - 1.0, (2.0 * coord[1] / self.height) - 1.0, -0.5))
            w = out.dot(self._p3) + self._p33
            view_vector = ((self.persinv @ out) / w) - self.viewinv.translation
        else:
            view_vector = self._ortho_dir.copy()
        view_vector.normalize()
        return view_vector

    def origin(self, coord):
        """region_2d_to_origin_3d (clamp=None)"""
        if self.is_perspective:
            return self._origin.copy()
        if self._ortho is not None:
            viewinv, inverse, shift, back = self._ortho
            dx = (2.0 * coord[0] / self.width) - 1.0
            dy = (2.0 * coord[1] / self.height) - 1.0
            x, y = inverse @ (np.array((dx, dy)) - shift)
            return Vector(viewinv[:3, :3] @ (x, y, 0.0) + viewinv[:3, 3] - back)
        persinv = self.persinv
        dx = (2.0 * coord[0] / self.width) - 1.0
        dy = (2.0 * coord[1] / self.height) - 1.0
        origin_start = (persinv.col[0].xyz * dx) + (persinv.col[1].xyz * dy) + persinv.translation
        if not self.is_camera:
            origin_start -= persinv.col[2].xyz
        return origin_start

    def ray(self, xy):
        return self.origin(xy), self.vector(xy).normalized()

    def to_screen(self, world):
        """location_3d_to_region_2d"""
        prj = self.persmat @ Vector((world[0], world[1], world[2], 1.0))
        if prj.w > 0.0:
            width_half = self.width / 2.0
            height_half = self.height / 2.0
            return Vector((width_half + width_half * (prj.x / prj.w),
                           height_half + height_half * (prj.y / prj.w)))
        return None

    def on_view_plane(self, xy, depth_world):
        """region_2d_to_location_3d"""
        coord_vec = self.vector(xy)
        depth_location = Vector(depth_world)
        origin_start = self.origin(xy)
        origin_end = origin_start + coord_vec
        if self.is_perspective:
            return intersect_line_plane(origin_start, origin_end, depth_location,
                                        self._plane_normal.copy(), 1)
        return intersect_point_line(depth_location, origin_start, origin_end)[0]


def view_forward(context):
    return context.region_data.view_rotation @ Vector((0.0, 0.0, -1.0))


class Prepared(list):
    """Sources with their per-object ray data computed once (inverse matrix, bounding
    sphere), for callers that cast many rays at unchanged sources (brush visibility).
    Same values as computed per call, so the results are identical."""

    def __init__(self, objs):
        super().__init__(objs)
        self.data = [_object_data(obj) for obj in self]


def prepare(objs):
    return objs if isinstance(objs, Prepared) else Prepared(objs)


def _object_data(obj):
    mw = obj.matrix_world.copy()   # a copy, never a live reference into the object
    mi = mw.inverted()
    corners = [Vector(co) for co in obj.bound_box]
    center = sum(corners, Vector((0, 0, 0))) / len(corners)
    radius = max((co - center).length for co in corners)
    return obj, mw, mi, mi.to_3x3(), center, radius


def ray_cast_sources(context, origin, direction, objs=None):
    """Closest hit on sources along a world-space ray, or None."""
    best, best_dist = None, float("inf")
    if isinstance(objs, Prepared):
        data = objs.data
    else:
        data = map(_object_data, sources(context) if objs is None else objs)
    for obj, mw, mi, mi3, center, radius in data:
        o = mi @ origin
        d = (mi3 @ direction).normalized()
        # Orthographic origins can sit at a distant clipping plane. Start
        # just before the source's bounding sphere to reduce hit precision loss.
        advance = max(0.0, (center - o).dot(d) - radius - 0.001)
        o += d * advance
        try:
            ok, co, _normal, face = obj.ray_cast(o, d)
        except RuntimeError:
            # Blender could not build its lookup data for this object (seen with
            # imported SVG curves): skip it rather than abort the tool
            continue
        if not ok:
            continue
        co_w = mw @ co
        dist = (co_w - origin).length
        if dist < best_dist:
            best_dist = dist
            best = Hit(co_w, obj, face)
    return best


def nearest_on_sources(context, world, objs=None):
    best, best_dist = None, float("inf")
    if world is None or not world.length_squared < float("inf"):
        return None   # no usable point (e.g. derived from a vertex with a NaN position)
    if isinstance(objs, Prepared):
        data = ((obj, mw, mi) for obj, mw, mi, _m3, _c, _r in objs.data)
    else:
        data = ((obj, obj.matrix_world, obj.matrix_world.inverted())
                for obj in (sources(context) if objs is None else objs))
    for obj, mw, mi in data:
        try:
            ok, co, _normal, _face = obj.closest_point_on_mesh(mi @ world)
        except RuntimeError:
            continue   # no lookup data for this object (e.g. imported SVG curves): skip it
        if not ok:
            continue
        co_w = mw @ co
        dist = (co_w - world).length
        if dist < best_dist:
            best_dist = dist
            best = Hit(co_w, obj)
    return best


def on_view_plane(context, xy, depth_world):
    """Point under xy on the view-facing plane through depth_world."""
    return View(context).on_view_plane(xy, depth_world)


DEPTH_JUMP = 0.25   # x view distance: a hit this much farther / nearer than the point is ignored


def surface_or_plane(context, xy, depth_world, objs=None, view=None):
    """Hit on a source under xy; otherwise a point on the view plane at depth_world.

    A hit far deeper (or nearer) than depth_world is ignored: a point at the edge of one
    model, moved over a gap, would otherwise jump onto a model far behind the gap (seen
    through it). It then stops at the nearby edge like any point moved off the source.
    """
    origin, direction = view.ray(xy) if view is not None else ray(context, xy)
    hit = ray_cast_sources(context, origin, direction, objs)
    if hit and not _depth_jump(context, origin, direction, hit.world, depth_world, view):
        return hit
    plane = view.on_view_plane(xy, depth_world) if view is not None else on_view_plane(context, xy, depth_world)
    if plane is None or not plane.length_squared < float("inf"):
        # unusable depth (a vertex with a NaN position): the 3D cursor's depth instead
        plane = on_view_plane(context, xy, context.scene.cursor.location)
        if plane is None:
            return Hit(origin + direction * max(_view_distance(context, view), 1e-3))
    if clamp_enabled(context):
        # "Stop at Source Edge": off the source, take the closest source point, so a
        # point dragged past the silhouette stays on the source edge
        nearest = nearest_on_sources(context, plane, objs)
        if nearest is not None:
            return nearest
    return Hit(plane)


def _depth_jump(context, origin, direction, hit_world, depth_world, view=None):
    distance = view.view_distance if view is not None else getattr(context.region_data, "view_distance", 0.0)
    limit = DEPTH_JUMP * max(distance or 0.0, 1e-6)
    return abs((hit_world - origin).dot(direction) - (depth_world - origin).dot(direction)) > limit


def _view_distance(context, view=None):
    return view.view_distance if view is not None else getattr(context.region_data, "view_distance", 1.0) or 1.0


def clamp_enabled(context):
    scene = getattr(context.scene, "ppo", None)
    return bool(scene and scene.clamp_to_source and context.scene.tool_settings.use_snap)


def is_occluded(context, world, objs=None, *, bias=None):
    """True when a source surface sits in front of world as seen from the view."""
    xy = to_screen(context, world)
    if xy is None:
        return True
    origin, direction = ray(context, xy)
    hit = ray_cast_sources(context, origin, direction, objs)
    if not hit:
        return False
    if bias is None:
        bias = context.space_data.overlay.retopology_offset
    return (hit.world - origin).length + bias < (world - origin).length
