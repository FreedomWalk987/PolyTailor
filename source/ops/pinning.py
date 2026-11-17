# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Pin attributes and overlays without modifying subdivision crease data."""

import time
import traceback

import bmesh
import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .. import prefs, palette
from ..core import draw, raycast
from ..core.glue import VERT_LAYER, EDGE_LAYER
from ..core.pick import VISIBILITY_DEPTH_TOLERANCE, VISIBILITY_PROBE_PX
from ..core.safe_mesh import native_modal_running
from .common import active_tool_id

PIN_LAYER = "ppo_pins"
# pins written by earlier versions (older PolyPen releases, PolyPen_Only 1.x/2.0) are still honored
LEGACY_PIN_LAYERS = ("hpen_pins", "polypen_only_pins")
ALL_PIN_LAYERS = (PIN_LAYER,) + LEGACY_PIN_LAYERS


def pinned(bm, vert):
    for name in ALL_PIN_LAYERS:
        layer = bm.verts.layers.float.get(name)
        if layer is not None and vert[layer] > 0.5:
            return True
    return False


class PPO_OT_pin(bpy.types.Operator):
    bl_idname = "ppo.pin"
    bl_label = "Pin"
    bl_translation_context = "Poly Toolset"
    bl_description = "Protect the selected vertices from Tweak, Relax and Alt+Wheel"
    bl_options = {"REGISTER", "UNDO"}
    unpin: bpy.props.BoolProperty(default=False, options={"SKIP_SAVE"})

    @classmethod
    def description(cls, context, properties):
        from ..i18n import tr
        return tr("Remove the protection from the selected vertices" if properties.unpin else cls.bl_description)

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and context.edit_object is not None

    def execute(self, context):
        me = context.edit_object.data
        bm = bmesh.from_edit_mesh(me)
        if not any(v.select and not v.hide for v in bm.verts):
            # nothing selected on the active edited object (empty selection, or the
            # selection is on another object in multi-object editing): say so instead of a silent FINISHED
            from ..i18n import tr
            self.report({"WARNING"}, tr("Action unavailable for the current selection"))
            return {"CANCELLED"}
        removed = False
        if self.unpin:
            # never create the pin attribute just to clear it
            targets = [v for v in bm.verts if v.select and not v.hide]
            for layer in _pin_layers(bm):
                for vert in targets:
                    vert[layer] = 0.0
            removed = _drop_unused_pin_layers(bm)
        elif any(v.select and not v.hide for v in bm.verts):
            # adding a layer reallocates the vertex data: collect the vertices after it
            layer = bm.verts.layers.float.get(PIN_LAYER) or bm.verts.layers.float.new(PIN_LAYER)
            for vert in bm.verts:
                if vert.select and not vert.hide:
                    vert[layer] = 1.0
        bmesh.update_edit_mesh(me, destructive=removed)
        if removed:
            _clear_pin_cache()
        else:
            _geometry.pop(me.as_pointer(), None)  # pin set changed
        if context.area: context.area.tag_redraw()
        return {"FINISHED"}


# Local coordinates, keyed by Object + Mesh identity. Never cache world positions.
_last_dots = {}
_pin_targets = {}
_live_dots = {}
_live_links = {}


def _mesh_glue_links(mesh):
    attribute = mesh.attributes.get(EDGE_LAYER)
    if attribute is None or attribute.domain != "EDGE" or attribute.data_type != "INT" or len(attribute.data) != len(mesh.edges):
        return []
    return [(mesh.vertices[e.vertices[0]].co.copy(), mesh.vertices[e.vertices[1]].co.copy())
            for e in mesh.edges if attribute.data[e.index].value > 0
            and all(not mesh.vertices[i].hide for i in e.vertices)]


def _update_pin_positions(_scene, depsgraph):
    """Copy evaluated positions on geometry update, not inside viewport draw.

    No BMesh access or forced evaluation; temporary to_mesh() copies are limited
    to native modal fallback here, never viewport draw. Evaluated IDs are not retained.
    Use evaluated pin attributes; index fallback is limited to unmodified meshes.
    """
    for key, (obj, data, count, pins) in list(_pin_targets.items()):
        try:
            if obj.data != data or obj.mode != "EDIT":
                _pin_targets.pop(key, None)
                _live_dots.pop(key, None)
                _live_links.pop(key, None)
                continue
            updated = any((u.id.original or u.id) in (obj, data) and
                          (u.is_updated_geometry or u.is_updated_transform)
                          for u in depsgraph.updates)
            if not updated:
                continue
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.data
            if evaluated.type != "MESH":
                _live_dots.pop(key, None)
                continue
            if not len(mesh.vertices):
                # Without edit-mode modifiers the evaluated edit mesh is a BMesh wrapper
                # that exposes 0 vertices to Python. While a native modal (G, slide...)
                # moves vertices, copy it with to_mesh() here - in the depsgraph handler,
                # never in draw - so pins follow live. The copy is read in C; no Python
                # BMesh wrapper is created, so the edit mesh is not reallocated.
                dots = None
                if pins and native_modal_running(bpy.context):
                    try:
                        copy = evaluated.to_mesh()
                        if len(copy.vertices) == count:
                            dots = [(copy.vertices[i].co.copy(), selected) for i, selected in pins]
                            _live_links[key] = _mesh_glue_links(copy)
                    finally:
                        evaluated.to_mesh_clear()
                if dots is None:
                    # "no data", not "no pins": keep cached coordinates (an empty list
                    # made every pin vanish while orbiting the view)
                    _live_dots.pop(key, None)
                else:
                    _live_dots[key] = dots
                continue
            attributes = [a for name in ALL_PIN_LAYERS
                          if (a := mesh.attributes.get(name)) is not None
                          and a.domain == "POINT" and a.data_type == "FLOAT"
                          and len(a.data) == len(mesh.vertices)]
            glue = mesh.attributes.get(VERT_LAYER)
            if (glue is not None and (glue.domain != "POINT" or glue.data_type != "INT"
                                     or len(glue.data) != len(mesh.vertices))):
                glue = None
            if attributes or glue is not None:
                dots = []
                for i, v in enumerate(mesh.vertices):
                    pin = any(a.data[i].value > .5 for a in attributes)
                    glued = glue is not None and glue.data[i].value > 0
                    if not v.hide and (pin or glued):
                        dots.append((v.co.copy(), int(v.select) | (2 if glued and not pin else 0)))
            elif not any(m.show_viewport and m.show_in_editmode for m in obj.modifiers) and len(mesh.vertices) == count:
                dots = [(mesh.vertices[i].co.copy(), selected) for i, selected in pins]
            else:
                # Do not invent an index mapping through evaluated modifiers.
                _live_dots.pop(key, None)
                continue
            _live_dots[key] = dots
            _live_links[key] = _mesh_glue_links(mesh)
        except (ReferenceError, RuntimeError, AttributeError, IndexError):
            _live_dots.pop(key, None)
            _live_links.pop(key, None)


def _clear_pin_cache(*_args):
    _geometry.clear()
    _screen.clear()
    _pin_source.clear()
    _last_dots.clear()
    _pin_targets.clear()
    _live_dots.clear()
    _live_links.clear()


def _marker_layers(bm):
    layers = _pin_layers(bm)
    integers = getattr(bm.verts.layers, "int", None)
    glue = integers.get(VERT_LAYER) if integers is not None else None
    return layers + ([glue] if glue is not None else [])


def _marker_style(bm, vertex, layers=None):
    # Bit 0: selected outline. Bit 1: Glue (ordinary Pin has priority).
    style = int(vertex.select)
    if layers is None:
        pins = _pin_layers(bm)
        integers = getattr(bm.verts.layers, "int", None)
        glue = integers.get(VERT_LAYER) if integers is not None else None
    else:
        pins, glue = layers
    if not any(vertex[layer] > .5 for layer in pins) and glue is not None and vertex[glue] > 0:
        style |= 2
    return style



def _draw_cached(context, radius):
    obj = context.edit_object
    if obj is None:
        return
    entry = _last_dots.get(context.region.as_pointer())
    if not entry or entry[0] != (obj.as_pointer(), obj.data.as_pointer()):
        return
    dots = _live_dots.get(entry[0], entry[1])
    if not dots:
        return
    # visibility is not re-tested here (no BMesh access during native modals)
    mw = obj.matrix_world
    xys = _project(context, [mw @ local for local, _selected in dots])
    shown = [(xy, selected) for xy, (_l, selected) in zip(xys, dots) if xy is not None]
    cached = _screen.get(context.region.as_pointer(), {})
    local_links = _live_links.get(entry[0], cached.get("links_local", ()))
    projected = _project(context, [mw @ p for pair in local_links for p in pair])
    links = [(a, b) for a, b in zip(projected[::2], projected[1::2]) if a is not None and b is not None]
    _draw_batch({"batch": _build_batch(shown, context, links)})


# ---------------------------------------------------------------- pin draw cache
# Redrawing pins used to rebuild a BVH of the whole edit mesh and scan every vertex
# on every viewport redraw (~100 ms at 40k vertices). Now:
#   - the pinned-vertex list and the self-occlusion BVH are rebuilt only when the
#     mesh changes (depsgraph update generation / element counts / BMesh identity);
#     during a stroke the BVH is refreshed at most every BVH_REFRESH seconds,
#   - screen positions and visibility are recomputed only for pinned vertices, and
#     only when the view or the mesh changed; otherwise cached dots are redrawn.
BVH_REFRESH = 0.25
VISIBILITY_BUDGET = 0.002  # seconds of visibility ray casts per redraw (rest next redraw)
VISIBILITY_STEP = 0.02     # redraw interval while visibility tests are pending
_generation = {}   # Mesh pointer -> bumped on every depsgraph update of that mesh/object
_geometry = {}     # Mesh pointer -> pinned vertices + BVH for the current generation
_pin_source = {}   # (object, mesh) pointers -> geometry that _pin_targets was built from
_screen = {}       # region pointer -> (key, [(xy, selected)])


def _bump_generation(_scene, depsgraph):
    for update in depsgraph.updates:
        id_ = update.id.original or update.id
        if isinstance(id_, bpy.types.Object):
            # Object transforms do not change the local pin geometry or self BVH.
            id_ = id_.data if id_.type == "MESH" and update.is_updated_geometry else None
        if isinstance(id_, bpy.types.Mesh):
            key = id_.as_pointer()
            _generation[key] = _generation.get(key, 0) + 1


def _pin_layers(bm):
    return [layer for name in ALL_PIN_LAYERS if (layer := bm.verts.layers.float.get(name)) is not None]


def _drop_unused_pin_layers(bm):
    """Remove the pin attributes once no vertex is pinned, so the mesh is left as clean
    as before (it may be shared with other objects, e.g. a projection that follows it).
    Returns True when a layer was removed."""
    layers = _pin_layers(bm)
    if not layers or any(v[layer] > 0.5 for layer in layers for v in bm.verts):
        return False
    for name in ALL_PIN_LAYERS:
        layer = bm.verts.layers.float.get(name)
        if layer is not None:
            bm.verts.layers.float.remove(layer)
    return True


def _geometry_for(obj, bm, layers):
    key = obj.data.as_pointer()
    gen = _generation.get(key, 0)
    counts = (len(bm.verts), len(bm.edges), len(bm.faces))
    geo = _geometry.get(key)
    now = time.monotonic()
    if (geo is not None and geo["gen"] == gen and geo["counts"] == counts and geo["bm"] == id(bm)
            and not (geo["stale"] and now - geo["changed"] >= BVH_REFRESH)):
        return geo
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    bm.faces.index_update()
    if same_topology_pins := (geo is not None and geo["counts"] == counts and geo["bm"] == id(bm)
                              and geo["layers"] == len(layers)):
        # same topology and pin layers: the pinned set is unchanged (PPO_OT_pin drops
        # the cache), only coordinates / selection / hide need refreshing
        verts = bm.verts
        pins = [v for v in (verts[p[0]] for p in geo["pins_all"]) if not v.hide]
    else:
        pins = [v for v in bm.verts if any(v[layer] > 0.5 for layer in layers)]
    same_topology = geo is not None and geo["counts"] == counts and geo["bm"] == id(bm)
    if same_topology and geo["gen"] != gen:
        # coordinates changed (a stroke, a drag): keep the self-occlusion BVH until
        # the mesh has been quiet for BVH_REFRESH, then rebuild it once
        bvh, face_rows, bvh_time, stale = geo["bvh"], geo["face_rows"], geo["bvh_time"], True
    else:
        faces = [f for f in bm.faces if not f.hide]
        face_rows = {f.index: row for row, f in enumerate(faces)}
        bvh = BVHTree.FromPolygons([v.co for v in bm.verts], [[v.index for v in f.verts] for f in faces])
        bvh_time, stale = now, False
    integers = getattr(bm.verts.layers, "int", None)
    styles = (_pin_layers(bm), integers.get(VERT_LAYER) if integers is not None else None)
    geo = {
        "gen": gen, "counts": counts, "bm": id(bm), "bvh": bvh, "face_rows": face_rows,
        "bvh_time": bvh_time, "stale": stale, "layers": len(layers),
        "changed": now if geo is None or geo["gen"] != gen else geo["changed"],
        "pins_all": geo["pins_all"] if same_topology_pins else [(v.index,) for v in pins],
        "pins": [(v.index, v.co.copy(), _marker_style(bm, v, styles),
                  frozenset(face_rows.get(f.index, -1) for f in v.link_faces)) for v in pins if not v.hide],
    }
    integers = getattr(bm.edges.layers, "int", None)
    glue_edges = integers.get(EDGE_LAYER) if integers is not None else None
    if glue_edges is not None:
        bm.edges.ensure_lookup_table()
    if glue_edges is None:
        links_all = ()
    elif (same_topology_pins and "glue_edges_all" in _geometry[key]
          and all(bm.edges[edge][glue_edges] > 0 and
                  tuple(v.index for v in bm.edges[edge].verts) == (a, b)
                  for edge, a, b in _geometry[key]["glue_edges_all"])):
        links_all = _geometry[key]["glue_edges_all"]
    else:
        bm.edges.index_update()
        links_all = tuple((e.index, e.verts[0].index, e.verts[1].index)
                          for e in bm.edges if e[glue_edges] > 0)
    geo["glue_edges_all"] = links_all
    geo["glue_edges"] = [(a, b) for edge, a, b in links_all
                         if not bm.edges[edge].hide and not bm.verts[a].hide and not bm.verts[b].hide]
    _geometry[key] = geo
    return geo


def _refresh_views():
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
    except (AttributeError, ReferenceError):
        pass
    return None


def _camera_key(context, obj):
    rv3d = context.region_data
    region = context.region
    offset = getattr(getattr(context.space_data, "overlay", None), "retopology_offset", 0.01)
    return (tuple(map(tuple, rv3d.perspective_matrix)), tuple(map(tuple, obj.matrix_world)),
            region.width, region.height, offset, prefs.ui_scale(context),
            palette.accent(context), palette.main(context))


def _pin_visible(context, mw, mi, geo, world, xy, own_faces, occluders):
    rv3d = context.region_data
    origin, _ = raycast.ray(context, xy)
    dist = (world - origin).length
    depth = dist if getattr(rv3d, "is_perspective", True) else getattr(rv3d, "view_distance", dist)
    offset = getattr(getattr(context.space_data, "overlay", None), "retopology_offset", 0.01)
    bias = offset + VISIBILITY_DEPTH_TOLERANCE * depth
    bvh = geo["bvh"]

    def occluded(point, origin):
        direction = point - origin
        length = direction.length
        if length < 1e-9:
            return False
        direction /= length
        co, _, row, _ = bvh.ray_cast(mi @ origin, (mi.to_3x3() @ direction).normalized())
        if co is not None and row is not None and row not in own_faces:
            if (mw @ co - origin).length < length - bias:
                return True
        hit = raycast.ray_cast_sources(context, origin, direction, occluders)
        return hit is not None and (hit.world - origin).length + bias < length

    if not occluded(world, origin):
        return True
    for dx, dy in ((VISIBILITY_PROBE_PX, 0), (-VISIBILITY_PROBE_PX, 0),
                   (0, VISIBILITY_PROBE_PX), (0, -VISIBILITY_PROBE_PX)):
        probe_xy = xy + Vector((dx, dy))
        probe = raycast.on_view_plane(context, probe_xy, world)
        if probe is not None and not occluded(probe, raycast.ray(context, probe_xy)[0]):
            return True
    return False


_draw_error = None


def _draw_pins():
    """Never let one failure leave a broken cache behind: clear it and report once."""
    global _draw_error
    try:
        _draw_pins_impl()
    except Exception as exc:  # noqa: BLE001 - a draw callback must not keep failing
        _clear_pin_cache()
        message = f"{type(exc).__name__}: {exc}"
        if message != _draw_error:
            _draw_error = message
            print("PolyTailor pin overlay error (cache reset):")
            traceback.print_exc()


def _wheel_cached(context, obj, key):
    """Reuse the last pin batch during this viewport's read-only wheel session.

    Hover/animation must not enter BMesh, BVH or visibility work. Provisional
    visibility resumes on close. Pin/geometry/view changes invalidate this path.
    """
    from ..ui import custom_pie
    wheel = custom_pie._active
    if (wheel is None or wheel.closed or wheel.area != context.area
            or wheel.region != context.region or wheel.obj != obj):
        return False
    cached = _screen.get(key)
    if cached is None:
        return False
    target = (obj.as_pointer(), obj.data.as_pointer())
    previous = _last_dots.get(key)
    if previous is None or previous[0] != target:
        return False
    geo = cached["geo"]
    if (_geometry.get(target[1]) is not geo or geo["gen"] != _generation.get(target[1], 0)
            or cached["camera"] != _camera_key(context, obj)):
        return False
    _draw_batch(cached)
    return True


def _draw_pins_impl():
    context = bpy.context
    tool = active_tool_id(context)
    if context.mode != "EDIT_MESH" or not tool or not tool.startswith("ppo.") or not context.region_data:
        return
    obj = context.edit_object
    if obj is None or context.region is None:
        return
    radius = 3 * prefs.ui_scale(context)
    key = context.region.as_pointer()
    if _wheel_cached(context, obj, key):
        return
    if native_modal_running(context):
        # A native modal operator (G, E, Inset, ...) holds raw pointers into the edit
        # mesh: draw cached coordinates only. PolyTailor's own and other Python modal tools
        # do not block, so pins follow Tweak / Relax / drag / G live.
        _draw_cached(context, radius)
        return
    target = (obj.as_pointer(), obj.data.as_pointer())
    bm = bmesh.from_edit_mesh(obj.data)
    layers = _marker_layers(bm)
    if not layers:
        _last_dots.pop(key, None)
        _screen.pop(key, None)
        _geometry.pop(obj.data.as_pointer(), None)
        _pin_targets.pop(target, None)
        _pin_source.pop(target, None)
        _live_dots.pop(target, None)
        _live_links.pop(target, None)
        return
    geo = _geometry_for(obj, bm, layers)
    camera = _camera_key(context, obj)
    cached = _screen.get(key)
    if cached is None or cached["camera"] != camera or cached["geo"] is not geo or cached["pending"]:
        cached = _update_screen(context, obj, geo, camera, cached)
        _screen[key] = cached
    if (cached["pending"] or geo["stale"]) and not bpy.app.timers.is_registered(_refresh_views):
        # visibility or BVH is provisional: redraw again soon to finish it
        bpy.app.timers.register(_refresh_views,
                                first_interval=VISIBILITY_STEP if cached["pending"] else BVH_REFRESH)
    _draw_batch(cached)
    _last_dots[key] = (target, cached["visible_local"])
    if _pin_source.get(target) is not geo:
        _pin_source[target] = geo
        _pin_targets[target] = (obj, obj.data, geo["counts"][0], [(i, sel) for i, _l, sel, _f in geo["pins"]])
        _live_dots.pop(target, None)
        _live_links.pop(target, None)


def _update_screen(context, obj, geo, camera, cached):
    """Project pins and (re)test visibility within a per-redraw budget."""
    mw = obj.matrix_world
    pins = geo["pins"]
    world = [mw @ local for _i, local, _s, _f in pins]
    xys = _project(context, world)
    same_topology = (cached is not None and cached["geo"]["counts"] == geo["counts"]
                     and cached["geo"]["bm"] == geo["bm"] and cached["geo"]["layers"] == geo["layers"])
    visible = cached["visible"] if same_topology else {}
    # visibility results are valid for one camera + one settled BVH; mid-stroke (stale
    # BVH, same camera) the previous results are kept until the mesh settles
    epoch = (camera, geo["bvh_time"])
    if same_topology and geo["stale"] and cached["camera"] == camera:
        epoch = cached["epoch"]
    todo = [n for n, (index, *_rest) in enumerate(pins)
            if xys[n] is not None and visible.get(index, (None, None))[1] != epoch]
    # untested pins first, then the ones tested longest ago (dict order = test order)
    todo.sort(key=lambda n: pins[n][0] in visible)
    tested = 0
    if todo:
        deadline = time.perf_counter() + VISIBILITY_BUDGET
        mi = mw.inverted()
        occluders = raycast.sources(context, visibility=True)
        for n in todo:
            index, _local, _sel, own_faces = pins[n]
            visible.pop(index, None)
            visible[index] = (_pin_visible(context, mw, mi, geo, world[n], xys[n], own_faces, occluders), epoch)
            tested += 1
            if time.perf_counter() > deadline:
                break
    shown = [n for n in range(len(pins)) if visible.get(pins[n][0], (True, None))[0]]
    dots = [(xys[n], pins[n][2]) for n in shown if xys[n] is not None]
    by_index = {pins[n][0]: xys[n] for n in shown if xys[n] is not None}
    links = [(by_index[a], by_index[b]) for a, b in geo.get("glue_edges", ())
             if a in by_index and b in by_index]
    local_by_index = {pins[n][0]: pins[n][1] for n in shown if xys[n] is not None}
    local_links = [(local_by_index[a], local_by_index[b]) for a, b in geo.get("glue_edges", ())
                   if a in local_by_index and b in local_by_index]
    return {"camera": camera, "geo": geo, "visible": visible, "epoch": epoch,
            "visible_local": [(pins[n][1], pins[n][2]) for n in shown],
            "pending": len(todo) > tested, "batch": _build_batch(dots, context, links),
            "links_local": local_links}


def _project(context, world):
    """location_3d_to_region_2d for many points at once."""
    region, rv3d = context.region, context.region_data
    m = rv3d.perspective_matrix
    hw, hh = region.width / 2, region.height / 2
    out = []
    for p in world:
        x, y, z = p
        w = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3]
        if w <= 0.0:
            out.append(None)
            continue
        out.append(Vector((hw + hw * (m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3]) / w,
                           hh + hh * (m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3]) / w)))
    return out


# one pin dot: a 12-segment triangle fan (centre + rim) of unit radius
_ANGLES = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
_CIRCLE = np.vstack([np.zeros((1, 2)), np.stack([np.cos(_ANGLES), np.sin(_ANGLES)], axis=1)]).astype(np.float32)
_CIRCLE_TRIS = np.stack([np.zeros(12), 1 + np.arange(12), 1 + (np.arange(12) + 1) % 12], axis=1).astype(np.int32)


def _build_batch(dots, context, links=()):
    """All pin dots as one triangle batch: neutral outlines first, orange centres on top.

    Drawing each dot as two separate disc() batches cost ~0.3 ms per pin per
    redraw (1000 pins: ~300 ms), even when nothing changed.
    """
    if not dots:
        return None
    radius = 3 * prefs.ui_scale(context)
    n = len(dots)
    centers = np.array([tuple(xy) for xy, _sel in dots], dtype=np.float32)       # n x 2
    styles = np.array([int(sel) for _xy, sel in dots], dtype=np.int32)
    selected = (styles & 1) != 0
    outline = centers[:, None, :] + (radius + 1) * _CIRCLE[None, :, :]
    inner = centers[:, None, :] + radius * _CIRCLE[None, :, :]
    flat = np.concatenate([outline, inner]).reshape(-1, 2)                       # 2n*13 x 2
    coords = np.zeros((len(flat), 3), dtype=np.float32)   # builtin shaders take vec3 pos
    coords[:, :2] = flat
    edge = np.where(selected[:, None], np.array([palette.main(context)], np.float32),
                    np.array([[0, 0, 0, 0.9]], np.float32))
    centers_color = np.tile(np.array([palette.accent(context)], np.float32), (n, 1))
    colors = np.repeat(np.concatenate([edge, centers_color]), 13, axis=0)
    tris = (_CIRCLE_TRIS[None, :, :] + 13 * np.arange(2 * n, dtype=np.int32)[:, None, None]).reshape(-1, 3)
    # Connected chains share the existing triangle batch and visibility cache.
    pairs = np.asarray(links, dtype=np.float64).reshape(-1, 2, 2)
    delta = pairs[:, 1]-pairs[:, 0]
    lengths = np.linalg.norm(delta, axis=1)
    usable = lengths > 1e-8
    pairs, delta, lengths = pairs[usable], delta[usable], lengths[usable]
    if len(pairs):
        normals = np.stack((-delta[:, 1], delta[:, 0]), axis=1)
        normals *= (1.25*prefs.ui_scale(context)/lengths)[:, None]
        a, b = pairs[:, 0], pairs[:, 1]
        line_coords = np.stack((a+normals, a-normals, b-normals, b+normals), axis=1).reshape(-1, 2)
        lines = np.zeros((len(line_coords), 3), dtype=np.float32)
        lines[:, :2] = line_coords
        line_colors = np.tile(np.array([palette.accent(context,.95)], np.float32), (len(lines), 1))
        line_tris = (np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)[None, :, :]
                     + 4*np.arange(len(pairs), dtype=np.int32)[:, None, None]).reshape(-1, 3)
        tris = np.concatenate((line_tris, tris+len(lines)))
        coords = np.concatenate((lines, coords))
        colors = np.concatenate((line_colors, colors))
    shader = gpu.shader.from_builtin("SMOOTH_COLOR")
    return shader, batch_for_shader(shader, "TRIS", {"pos": coords, "color": colors}, indices=tris)


def _draw_batch(cached):
    batch = cached.get("batch")
    if batch is None:
        return
    shader, batch = batch
    with draw.Blend():
        batch.draw(shader)


_PIN_HANDLERS = (("depsgraph_update_post", _bump_generation),
                 ("depsgraph_update_post", _update_pin_positions),
                 ("undo_pre", _clear_pin_cache), ("redo_pre", _clear_pin_cache),
                 ("load_pre", _clear_pin_cache))


_handle = None


def start():
    """while PolyTailor is in use (session): pins are drawn only with a PolyTailor tool; start clean."""
    global _handle
    _clear_pin_cache()
    for name, callback in _PIN_HANDLERS:
        callback = bpy.app.handlers.persistent(callback)
        handlers = getattr(bpy.app.handlers, name)
        if callback not in handlers:
            handlers.append(callback)
    if _handle is None:
        _handle = bpy.types.SpaceView3D.draw_handler_add(_draw_pins, (), "WINDOW", "POST_PIXEL")


def stop():
    global _handle
    for name, callback in reversed(_PIN_HANDLERS):
        handlers = getattr(bpy.app.handlers, name)
        if callback in handlers:
            handlers.remove(callback)
    _clear_pin_cache()
    if bpy.app.timers.is_registered(_refresh_views):
        bpy.app.timers.unregister(_refresh_views)
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, "WINDOW")
        _handle = None


def register():
    bpy.utils.register_class(PPO_OT_pin)


def unregister():
    stop()
    bpy.utils.unregister_class(PPO_OT_pin)
