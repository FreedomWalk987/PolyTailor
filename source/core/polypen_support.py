# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Build's view of the scene: points on sources, vertex visibility, fallback depth."""
import weakref
from itertools import chain

import bmesh
import numpy as np
from mathutils import Vector
from .. import prefs
from . import raycast, snap


def is_bmvert_hidden(context, vertex):
    # Cache only for the current reference update, cleared by the wrapper.
    view = _views.get(context.edit_object.as_pointer())
    return vertex.hide or (view is not None and not view.visible(vertex))


# Weak: the cache must not keep a MeshView (and the BMesh it wraps) alive after the
# Build operator that owns it has ended - freeing it much later crashed Blender.
_views = weakref.WeakValueDictionary()


def bind_view(view):
    _views.clear()
    _views[view.obj.as_pointer()] = view


_projected = {}   # one projection of the edit mesh, reused while mesh and view are unchanged


def reset_projection_cache():
    """Call whenever the mesh or the selection may have changed (the view, the object
    transform and the vertex count are part of the cache key)."""
    _projected.clear()


def _projection(context):
    obj = context.edit_object
    bm = bmesh.from_edit_mesh(obj.data)
    # vertex count in the key: geometry added within a step (commit) is never missed
    key = (obj.as_pointer(), len(bm.verts), tuple(map(tuple, context.region_data.perspective_matrix)),
           tuple(map(tuple, obj.matrix_world)), context.region.width, context.region.height)
    cached = _projected.get("data")
    if cached is not None and cached[0] == key:
        return cached[1]
    n = len(bm.verts)
    visible = np.fromiter((not v.hide for v in bm.verts), dtype=bool, count=n)
    result = None
    if visible.any():
        co = np.fromiter(chain.from_iterable(v.co for v in bm.verts), dtype=np.float64,
                         count=3 * n).reshape(n, 3)[visible]
        selected = np.fromiter((v.select for v in bm.verts), dtype=bool, count=n)[visible]
        pm = np.array(context.region_data.perspective_matrix, dtype=np.float64) @ np.array(obj.matrix_world)
        h = np.c_[co, np.ones(len(co))] @ pm.T
        front = h[:, 3] > 1e-9
        w = np.where(front, h[:, 3], 1.0)
        hw, hh = context.region.width / 2.0, context.region.height / 2.0
        result = (co, selected, front, hw + hw * h[:, 0] / w, hh + hh * h[:, 1] / w)
    _projected["data"] = (key, result)
    return result


def fallback_depth(context, xy):
    """Depth for a point off every source: the nearest selected vertex on screen, else
    the nearest vertex within 200 px, else the 3D cursor. All vertices are projected in
    one numpy pass (a per-vertex Python loop cost ~10 ms per call at 3,000 vertices)."""
    obj = context.edit_object
    projected = _projection(context)
    if projected is not None:
        co, selected, front, sx, sy = projected
        front = front & np.isfinite(co).all(axis=1) & np.isfinite(sx) & np.isfinite(sy)
        d2 = (sx - xy[0]) ** 2 + (sy - xy[1]) ** 2
        for mask, limit in ((front & selected, None), (front, (200 * prefs.ui_scale(context)) ** 2)):
            if mask.any():
                k = int(np.argmin(np.where(mask, d2, np.inf)))
                if limit is None or d2[k] <= limit:
                    return obj.matrix_world @ Vector(co[k])
                break
    return context.scene.cursor.location.copy()


def raycast_point_valid_sources(context, point):
    if len(point) == 3:
        xy = raycast.to_screen(context, point)
        depth = point
    else:
        xy = Vector(point)
        depth = None   # only needed off the sources: computed lazily below
    if xy is None:
        return None
    sources = raycast.sources(context)
    if sources:
        hit = raycast.ray_cast_sources(context, *raycast.ray(context, xy), sources)
        if hit:
            return hit.world
    if depth is None:
        depth = fallback_depth(context, xy)
    # Preserve PolyPen_Only's view-plane fallback outside a source surface.
    return raycast.surface_or_plane(context, xy, depth, sources).world


def raycast_valid_sources(context, xy):
    # the mouse point: Vertex / Edge / Edge Center snapping applies here (core.snap)
    # with Snap to Itself, the selection Build extends from is not a target
    exclude = None
    if getattr(context.scene.ppo, "snap_self", False):
        exclude = [v for v in bmesh.from_edit_mesh(context.edit_object.data).verts if v.select]
    world = snap.snap(context, Vector(xy), exclude=exclude)
    if world is None:
        world = raycast_point_valid_sources(context, Vector(xy))
    return {"co_local": context.edit_object.matrix_world.inverted() @ world} if world is not None else None


def nearest_point_valid_sources(context, point):
    hit = raycast.nearest_on_sources(context, point)
    return hit.world if hit else point.copy()
