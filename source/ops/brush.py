# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Brushes and selection smoothing: Tweak and Relax strokes, F / Shift+F / Ctrl+F brush
adjustment, Reset Brush, Alt+Wheel smoothing, the brush cursor overlay, and the mirror
clipping shared with the other move tools."""

import time
from itertools import chain

import bmesh
import bpy
import numpy as np
from ..i18n import tr as iface_, tr_rpt, friendly_error   # add-on context first; friendly_error
from bpy.props import BoolProperty, EnumProperty
from mathutils import Vector
from mathutils.geometry import intersect_point_line

from .. import palette, prefs, props
from ..core import draw, raycast, safe_mesh
from ..core.pick import MeshView, bvh_from_loops
from ..core.looptools_relax import selected_edge_keys, selected_loop_moves
from ..core.smooth import SmoothSolver
from ..core.brush_index import StrokeIndex
from ..core.brush_bounds import ScreenProjection, local_bvh, plane_visibility
from ..core.brush_grid import MovingGrid
from .common import Hover, active_tool_id, timer_tick
from .pinning import pinned, _pin_layers

BRUSH_TOOLS = {"ppo.tweak": "tweak", "ppo.relax": "relax"}
GRID_STEPS_PER_NOTCH = 6   # Alt+Wheel with Grid Relax: brush ticks per wheel notch
_paused_cursors = set()


def _cursor_key(context):
    """Primitive window/region identity, without retaining UI or mesh handles."""
    return (context.window.as_pointer(), context.region.as_pointer())


def _resume_cursor(context):
    _paused_cursors.discard(_cursor_key(context))


class PPO_OT_brush_hover(bpy.types.Operator):
    bl_idname = "ppo.brush_hover"
    bl_label = "Track Brush Cursor"
    bl_translation_context = "Poly Toolset"
    bl_options = {"INTERNAL"}

    def invoke(self, context, event):
        if event.type == "RIGHTMOUSE":
            # Run before the native context-menu keymap. Popup UI owns subsequent
            # input: only a later viewport event may resume the paint cursor.
            _paused_cursors.add(_cursor_key(context))
        else:
            _resume_cursor(context)
            Hover.region = context.region.as_pointer()
            Hover.xy = raycast.mouse(event)
        context.region.tag_redraw()
        # Do not finish/consume PRESS: Blender must still receive its normal
        # right-click action, including CLICK synthesis and customized keymaps.
        return {"PASS_THROUGH"}


def brush_settings(context, tool_id=None):
    key = BRUSH_TOOLS.get(tool_id or active_tool_id(context))
    return getattr(props.settings(context), key) if key else None


def falloff_weight(t, falloff):
    """Weight for normalized distance t in [0, 1]."""
    if t >= 1.0:
        return 0.0
    return 1.0 - falloff * max(0.0, t) ** 3


def pixels_to_world(context, world, pixels):
    """World length of `pixels` on screen at the depth of `world`."""
    xy = raycast.to_screen(context, world)
    if xy is None:
        return 0.0
    a = raycast.on_view_plane(context, xy, world)
    b = raycast.on_view_plane(context, xy + Vector((pixels, 0.0)), world)
    return (a - b).length


class MirrorClip:
    """Keeps vertices on mirror planes of clipped Mirror modifiers."""

    def __init__(self, obj):
        self.planes = []
        for mod in obj.modifiers:
            if mod.type != "MIRROR" or not mod.use_clip or not mod.show_viewport:
                continue
            transform = mod.mirror_object.matrix_world.inverted_safe() @ obj.matrix_world if mod.mirror_object else None
            for i, used in enumerate(mod.use_axis):
                if used:
                    self.planes.append((i, mod.merge_threshold, transform))

    def on_plane(self, co):
        return any(abs((transform @ co if transform is not None else co)[i]) <= t
                   for i, t, transform in self.planes)

    def apply(self, co_orig, co_new):
        for i, t, transform in self.planes:
            old = transform @ co_orig if transform is not None else co_orig
            new = transform @ co_new if transform is not None else co_new.copy()
            if abs(old[i]) <= t or (old[i] > 0) != (new[i] > 0):
                new[i] = 0.0
                co_new = transform.inverted_safe() @ new if transform is not None else new
        return co_new


def is_boundary(bmv, clip):
    native=getattr(bmv,"is_boundary",None)
    boundary=any(e.is_boundary for e in bmv.link_edges) if native is None else native
    return boundary and not clip.on_plane(bmv.co)


class BrushStroke:
    """Collects the vertices under the brush when the stroke starts."""

    def __init__(self, context, event, settings, selection=False):
        self.context = context
        self.obj = context.edit_object
        self.mw = self.obj.matrix_world
        self.mi = self.mw.inverted()
        self.bm = bmesh.from_edit_mesh(self.obj.data)
        self.srcs = raycast.prepare(raycast.sources(context))   # many rays per stroke
        self.clip = MirrorClip(self.obj)
        self.settings = settings
        self.scene = props.settings(context)
        self.mouse = raycast.mouse(event)
        self.verts = []  # (bmv, original local co, weight)
        self.radius_world = 0.0  # set by _collect once a brush center is found
        self.boundary = []
        self.selection = selection
        self._collect()
        # positions at the start of the stroke (Esc / cancel), from the stroke's arrays
        self._originals = self._cache["co"].copy()
        from ..core.glue import GlueState
        self.glue = GlueState(self)
        from ..core.face_attributes import Correction
        self.attributes = Correction(self.bm, context)

    def _center(self, cache, project=None):
        xy = self.mouse
        origin, direction = raycast.ray(self.context, xy)
        hit = raycast.ray_cast_sources(self.context, origin, direction, self.srcs)
        if hit:
            return hit.world
        if not len(cache["verts"]):
            return None
        sx, sy, front = project() if project is not None else self._project(cache)
        # no source under the cursor: the edit mesh's own surface under the cursor (the
        # front-most face; only faces whose screen box holds the cursor can be hit)
        if len(cache["faces"]):
            loops, starts = cache["loops"], cache["starts"]
            inside = ((np.minimum.reduceat(sx[loops], starts) <= xy[0]) &
                      (np.maximum.reduceat(sx[loops], starts) >= xy[0]) &
                      (np.minimum.reduceat(sy[loops], starts) <= xy[1]) &
                      (np.maximum.reduceat(sy[loops], starts) >= xy[1]))
            inside |= np.logical_or.reduceat(~front[loops], starts)
            if inside.any():
                bvh = bvh_from_loops(cache["co"], loops, starts, inside)
                mi = self.mi
                co, _n, _i, _d = bvh.ray_cast(mi @ origin, (mi.to_3x3() @ direction).normalized())
                if co is not None:
                    return self.mw @ co
        # off the mesh: the vertex nearest the cursor on screen gives the depth
        usable = front & ~cache["hide"]
        if not usable.any():
            return None
        d2 = np.where(usable, (sx - xy[0]) ** 2 + (sy - xy[1]) ** 2, np.inf)
        best = cache["verts"][int(np.argmin(d2))]
        return raycast.on_view_plane(self.context, xy, self.mw @ best.co)

    def _project(self, cache):
        """Screen x, y of every cached vertex and whether it is in front of the view
        (location_3d_to_region_2d for all vertices at once)."""
        region, rv3d = self.context.region, self.context.region_data
        pm = np.array(rv3d.perspective_matrix, dtype=np.float64) @ np.array(self.mw, dtype=np.float64)
        co = cache["co"]
        clip = co @ pm[:3, :3].T + pm[:3, 3]
        w = co @ pm[3, :3] + pm[3, 3]
        front = w > 0.0
        safe_w = np.where(front, w, 1.0)
        return (region.width / 2 * (1 + clip[:, 0] / safe_w),
                region.height / 2 * (1 + clip[:, 1] / safe_w), front)

    def _mesh_cache(self):
        """Per-stroke arrays of the edit mesh (topology does not change during a
        stroke): every vertex, its local position, hidden flags, and the visible faces
        as a flat vertex-index list. Positions of the vertices a stroke step may have
        moved (the ones collected last time) are refreshed on every call."""
        cache = getattr(self, "_cache", None)
        if cache is None or len(cache["verts"]) != len(self.bm.verts):
            self.bm.verts.index_update()   # cache rows = vertex indices for the stroke
            verts = list(self.bm.verts)
            n = len(verts)
            faces = [f for f in self.bm.faces if not f.hide]
            sizes = np.fromiter((len(f.verts) for f in faces), dtype=np.int64, count=len(faces))
            cache = self._cache = {
                "verts": verts,
                "co": np.fromiter(chain.from_iterable(v.co for v in verts), dtype=np.float64,
                                  count=3 * n).reshape(n, 3),
                "hide": np.fromiter((v.hide for v in verts), dtype=bool, count=n),
                "faces": faces,
                "loops": np.fromiter(chain.from_iterable((v.index for v in f.verts) for f in faces),
                                     dtype=np.int64, count=int(sizes.sum())),
                "starts": np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.int64) if len(sizes)
                          else np.zeros(0, dtype=np.int64),
            }
        else:
            co, verts = cache["co"], cache["verts"]
            for bmv, _co, _w in self.verts:
                i = bmv.index
                if not (0 <= i < len(verts)) or verts[i] != bmv:   # indices changed (bounds first)
                    i = verts.index(bmv)
                co[i] = bmv.co[:]
        return cache

    def _occlusion_view(self, cache, candidates, project=None):
        """MeshView whose self-occlusion uses only the faces whose screen box overlaps
        the screen box of the candidate vertices (plus the side probes): a ray toward a
        candidate can only hit those, so the visibility results are the same as with
        the whole mesh, without building a BVH of the whole mesh on every step."""
        if not len(cache["faces"]) or not len(candidates):
            return MeshView(self.context, faces=[])
        sx, sy, front = project() if project is not None else self._project(cache)
        c = candidates[front[candidates]]
        if not len(c):
            return MeshView(self.context, faces=[])
        margin = 3 + 2   # pick.VISIBILITY_PROBE_PX plus rounding slack
        x0, x1 = sx[c].min() - margin, sx[c].max() + margin
        y0, y1 = sy[c].min() - margin, sy[c].max() + margin
        loops, starts = cache["loops"], cache["starts"]
        fx0 = np.minimum.reduceat(sx[loops], starts)
        fx1 = np.maximum.reduceat(sx[loops], starts)
        fy0 = np.minimum.reduceat(sy[loops], starts)
        fy1 = np.maximum.reduceat(sy[loops], starts)
        behind = np.logical_or.reduceat(~front[loops], starts)   # projection unknown: keep
        keep = behind | ((fx1 >= x0) & (fx0 <= x1) & (fy1 >= y0) & (fy0 <= y1))
        faces = cache["faces"]
        kept = [faces[i] for i in np.nonzero(keep)[0]]
        bvh = bvh_from_loops(cache["co"], loops, starts, keep) if kept else None
        return MeshView(self.context, faces=kept, bvh=bvh)

    def _note_spatial_changes(self, cache):
        index = getattr(self, "_spatial_index", None)
        if index is None or getattr(self, "_spatial_cache", None) is not cache:
            return
        verts = cache["verts"]
        rows = []
        for v, _co, _w in self.verts:
            row = v.index
            if not (0 <= row < len(verts)) or verts[row] != v:
                row = verts.index(v)
            rows.append(row)
        index.note(rows)

    def _candidate_indices(self, cache, center, radius):
        if self.selection:
            return np.nonzero(~cache["hide"])[0]
        mw = np.array(self.mw, dtype=np.float64)
        rows = None
        dense = getattr(self, "spatial_candidates", False) and len(cache.get("faces", ())) > 4000
        if dense:
            index = getattr(self, "_spatial_index", None)
            if (index is None or getattr(self, "_spatial_cache", None) is not cache
                    or not np.array_equal(self._spatial_matrix, mw)
                    or not np.array_equal(self._spatial_hidden, cache["hide"])):
                from mathutils.kdtree import KDTree
                world = cache["co"] @ mw[:3, :3].T + mw[:3, 3]
                index = self._spatial_index = StrokeIndex(world, cache["hide"], KDTree)
                self._spatial_cache = cache
                self._spatial_matrix = mw.copy()
                self._spatial_hidden = cache["hide"].copy()
            rows = index.query(center[:], radius * (1 + 1e-5) + 1e-9)
        co = cache["co"] if rows is None else cache["co"][rows]
        hidden = cache["hide"] if rows is None else cache["hide"][rows]
        world = co @ mw[:3, :3].T + mw[:3, 3]
        d = np.sqrt(((world - np.array(center[:], dtype=np.float64)) ** 2).sum(axis=1))
        keep = np.nonzero(~hidden & (d < radius * (1 + 1e-5) + 1e-9))[0]
        return keep if rows is None else rows[keep]

    def _collect(self):
        cache = self._mesh_cache()
        self._note_spatial_changes(cache)
        screen = None
        def project():
            nonlocal screen
            if screen is None:
                screen = self._project(cache)
            return screen
        center = self._center(cache, project)
        self.verts = []
        if center is None:
            return
        radius = pixels_to_world(self.context, center, self.settings.radius * prefs.ui_scale(self.context))
        if radius <= 0.0:
            return
        scene = self.scene
        selected_mode=scene.mask_selected
        boundary_mode=scene.mask_boundary
        include_corners=scene.include_corners
        include_seams=scene.include_seams
        include_sharps=scene.include_sharps
        include_creases=scene.include_creases
        include_occluded=scene.include_occluded
        strength=self.settings.strength
        falloff=self.settings.falloff
        self.radius_world = radius
        # Spatial lookup is only a conservative broad phase. The same world
        # distance and per-vertex masks below still decide actual influence.
        candidates = self._candidate_indices(cache, center, radius)
        all_verts = cache["verts"]
        view = None
        pin_layers = _pin_layers(self.bm)  # handles live only for this collection
        crease_v = self.bm.verts.layers.float.get("crease_vert")
        crease_e = self.bm.edges.layers.float.get("crease_edge")
        for i in candidates:
            bmv = all_verts[i]
            world = self.mw @ bmv.co
            d = (world - center).length
            if self.selection and not bmv.select:
                continue
            if not self.selection and d >= radius:
                continue
            if not self.selection and selected_mode == "EXCLUDE" and bmv.select:
                continue
            if not self.selection and selected_mode == "ONLY" and not bmv.select:
                continue
            if boundary_mode == "EXCLUDE" and is_boundary(bmv, self.clip):
                continue
            if any(bmv[layer] > 0.5 for layer in pin_layers):  # same pin layers
                continue
            if not include_corners and (len(bmv.link_edges) == 2 or
                    (len(bmv.link_edges) == 4 and len(bmv.link_faces) == 3)):
                continue
            if not include_seams and any(e.seam for e in bmv.link_edges):
                continue
            if not include_sharps and any(not e.smooth for e in bmv.link_edges):
                continue
            if not include_creases and ((crease_v is not None and bmv[crease_v] > 0) or
                    (crease_e is not None and any(e[crease_e] > 0 for e in bmv.link_edges))):
                continue
            if not include_occluded:
                if view is None:
                    view = self._occlusion_view(cache, candidates, project)
                if not view.visible(bmv):
                    continue
            w = strength if self.selection else falloff_weight(d / radius, falloff) * strength
            if w > 0.0:
                self.verts.append((bmv, bmv.co.copy(), w))
        if boundary_mode == "SLIDE" and not self.boundary:
            self.boundary = [
                (e.verts[0].co.copy(), e.verts[1].co.copy())
                for e in self.bm.edges
                if e.is_boundary and not (self.clip.on_plane(e.verts[0].co) and self.clip.on_plane(e.verts[1].co))
            ]

    def snap_to_boundary(self, co):
        best, best_d = co, float("inf")
        for a, b in self.boundary:
            p, t = intersect_point_line(co, a, b)
            p = a if t <= 0.0 else b if t >= 1.0 else p
            d = (p - co).length_squared
            if d < best_d:
                best, best_d = p, d
        return best.copy()

    def restore(self):
        """each vertex back to its start only while it still holds what this stroke last wrote (or,
        never written, its start): an outside edit is kept; a stale handle falls back to its row index. Returns the
        number left alone."""
        from ..core.mesh import restore_written
        last = self.__dict__.get("_last", {})
        skipped = restore_written(self.bm, (
            (bmv, i, co, last[bmv][1] if bmv.is_valid and bmv in last else co)
            for i, (bmv, co) in enumerate(zip(self._cache["verts"], self._originals))))
        attributes = getattr(self, "attributes", None)
        if attributes is not None:
            attributes.restore()
        self.update_mesh()
        return skipped

    def update_mesh(self):
        self.bm.normal_update()
        bmesh.update_edit_mesh(self.obj.data, loop_triangles=True, destructive=False)


def drag_vertices(stroke, event, originals=None):
    """Shared Tweak displacement; optional originals journal spans a mixed stroke."""
    xy = raycast.mouse(event)
    delta = xy - stroke.mouse
    if delta.length_squared == 0.0:
        return None
    pressure = event.pressure if event.pressure > 0.0 else 1.0
    slide = stroke.scene.mask_boundary == "SLIDE"
    view = raycast.View(stroke.context)
    written = []
    glue = getattr(stroke, "glue", None)
    before_glue = glue.begin_drag(v for v, _co, _w in stroke.verts) if glue is not None else {}
    for bmv, co_orig, w in stroke.verts:
        world = stroke.mw @ bmv.co
        p = view.to_screen(world)
        if p is None:
            continue
        hit = raycast.surface_or_plane(stroke.context, p + delta * w * pressure, world, stroke.srcs, view)
        co = stroke.mi @ hit.world
        if slide and stroke.boundary and is_boundary(bmv, stroke.clip):
            co = stroke.snap_to_boundary(co)
        if originals is not None and bmv not in originals:
            originals[bmv] = bmv.co.copy()
        before = bmv.co.copy()
        bmv.co = stroke.clip.apply(co_orig, co)
        if bmv.co != before:
            written.append(bmv)
    if before_glue:
        def surface(v, world):
            hit = raycast.nearest_on_sources(stroke.context, world, stroke.srcs) if stroke.srcs else None
            co = stroke.mi @ (hit.world if hit else world)
            if slide and stroke.boundary and is_boundary(v, stroke.clip):
                co = stroke.snap_to_boundary(co)
            original = glue._drag_local[glue.of[v]][glue.row[v]]
            return stroke.mw @ stroke.clip.apply(original, co)
        written = glue.correct_drag(before_glue, written, surface, originals)
    from ..core.face_attributes import apply
    apply(stroke, written)
    stroke.mouse = xy
    return written


class TweakStroke(BrushStroke):
    def drag(self, event):
        journal = self.__dict__.get("journal")
        if journal is None:   # the vertices Tweak writes, by identity
            from ..core.mesh import CoordinateJournal
            journal = self.journal = CoordinateJournal(self.bm)
        written = drag_vertices(self, event, journal)
        if written is not None:
            journal.wrote(written)
            self.update_mesh()

    def stale(self):
        journal = self.__dict__.get("journal")
        return journal is not None and journal.stale()

    def restore(self):
        """Esc puts back only the vertices this Tweak wrote, each only while it still holds what Tweak
        last wrote: an outside edit anywhere is kept (no whole-mesh revert). Returns the number left alone."""
        journal = self.__dict__.get("journal")
        skipped = journal.restore() if journal is not None else 0
        attributes = getattr(self, "attributes", None)
        if attributes is not None:
            attributes.restore()
        self.update_mesh()
        return skipped


class PPO_OT_tweak(bpy.types.Operator):
    bl_idname = "ppo.tweak"
    bl_label = "Tweak"
    bl_translation_context = "Poly Toolset"
    bl_description = "Push vertices around with a soft brush"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and context.region_data is not None

    def invoke(self, context, event):
        self.button = "RIGHTMOUSE" if event.type == "RIGHTMOUSE" else "LEFTMOUSE"
        _resume_cursor(context)
        self.stroke = TweakStroke(context, event, props.settings(context).tweak)
        if not self.stroke.verts:
            return {"CANCELLED"}
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        Hover.xy = raycast.mouse(event)
        button = getattr(self, "button", "LEFTMOUSE")
        cancel = {"ESC", "LEFTMOUSE" if button == "RIGHTMOUSE" else "RIGHTMOUSE"}
        if self.stroke.stale():   # a vertex Tweak wrote changed under it: stop, no rollback
            self.report({"WARNING"}, tr_rpt("The mesh being edited changed; the tool stopped"))
            return {"CANCELLED"}
        if event.type in {"MOUSEMOVE", "INBETWEEN_MOUSEMOVE"}:
            self.stroke.drag(event)
            context.region.tag_redraw()
        elif event.type == button and event.value == "RELEASE":
            return {"FINISHED"}
        elif event.type == "WINDOW_DEACTIVATE" or (event.type in cancel and event.value == "PRESS"):
            if not safe_mesh.restore_allowed(self, context) or self.stroke.restore():   # Rc4
                self.report({"WARNING"}, tr_rpt("The mesh being edited changed; the tool stopped"))
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        stroke = getattr(self, "stroke", None)
        if stroke is not None:
            stroke.restore()


class RelaxStroke(BrushStroke):
    spatial_candidates = True

    def begin_drag(self, event):
        """Freeze the current Relax footprint at the Ctrl transition, without a jump."""
        self.mouse = raycast.mouse(event)
        self._collect()
        self._drag_changed = False

    def drag(self, event):
        written = drag_vertices(self, event, self.solver.prev)
        if not written:
            return
        self._drag_changed = True
        self.note_written(written, np.array([v.co[:] for v in written]))
        faces = {f for v in written for f in v.link_faces}
        for face in faces:
            face.normal_update()
        for vertex in {v for f in faces for v in f.verts}:
            vertex.normal_update()
        if self.feedback is not None:
            self.feedback.update(written)
        else:
            bmesh.update_edit_mesh(self.obj.data, loop_triangles=True, destructive=False)

    def end_drag(self):
        if not getattr(self, "_drag_changed", False):
            return
        # The dragged shape becomes the next Relax reference. Keep the original
        # cancellation journal, but discard numerical targets and L-BFGS history.
        originals = self.solver.prev
        self.solver = SmoothSolver(self)
        self.solver.prev = originals
        if self.scene.pen3d and len(self._cache["faces"]):
            retained = getattr(self, "_retained_entry", None)
            if retained is not None:
                self.solver.self_surface = retained.surface(self)
            else:
                cache = self._cache
                self.solver.self_surface = bvh_from_loops(
                    cache["co"], cache["loops"], cache["starts"],
                    np.ones(len(cache["faces"]), dtype=bool))
        self._drag_changed = False

    def note_written(self,vertices,coordinates):
        last = self.__dict__.setdefault("_last", {})   # what this stroke last wrote, per vertex
        for v in vertices:
            if v.is_valid:
                last[v] = (v.index, v.co.copy())
        if self.selection:
            return
        retained=getattr(self,"_retained_entry",None)
        if retained is not None:
            retained.record(vertices)
        cache=self._cache
        rows=np.fromiter((v.index for v in vertices),dtype=np.int64,count=len(vertices))
        cache["co"][rows]=coordinates
        slab=cache.get("_slab_bounds")
        if slab is not None:
            slab.changed(coordinates)
        projection=getattr(self,"_screen_bounds",None)
        if projection is not None:
            projection.changed(rows)
        grid=getattr(self,"_moving_grid",None)
        if grid is not None:
            matrix=np.asarray(self.mw,dtype=float)
            grid.note(rows,coordinates@matrix[:3,:3].T+matrix[:3,3])
        self._direct_cache=True

    def _note_spatial_changes(self,cache):
        if self.selection:
            return super()._note_spatial_changes(cache)
        if getattr(self,"_direct_cache",False):
            return
        grid=getattr(self,"_moving_grid",None)
        if grid is not None and getattr(self,"_grid_cache",None) is cache and self.verts:
            rows=np.array([v.index for v,_,_ in self.verts],dtype=np.int64)
            mw=np.asarray(self.mw,dtype=float)
            world=cache["co"][rows]@mw[:3,:3].T+mw[:3,3]
            grid.note(rows,world)

    def _candidate_indices(self,cache,center,radius):
        if self.selection or len(cache.get("faces",()))<=4000:
            return super()._candidate_indices(cache,center,radius)
        mw=np.asarray(self.mw,dtype=float)
        grid=getattr(self,"_moving_grid",None)
        if (grid is None or getattr(self,"_grid_cache",None) is not cache
                or radius>grid.width*4 or radius<grid.width*.25
                or not np.array_equal(self._grid_matrix,mw)
                or not np.array_equal(self._grid_hidden,cache["hide"])):
            world=cache["co"]@mw[:3,:3].T+mw[:3,3]
            grid=self._moving_grid=MovingGrid(world,cache["hide"],radius)
            self._grid_cache=cache;self._grid_matrix=mw.copy();self._grid_hidden=cache["hide"].copy()
        rows=grid.query(np.array(center[:]),radius*(1+1e-5)+1e-9)
        co=cache["co"] if rows is None else cache["co"][rows]
        hidden=cache["hide"] if rows is None else cache["hide"][rows]
        world=co@mw[:3,:3].T+mw[:3,3]
        distance=np.sqrt(((world-np.array(center[:],dtype=float))**2).sum(axis=1))
        keep=np.flatnonzero(~hidden&(distance<radius*(1+1e-5)+1e-9))
        return keep if rows is None else rows[keep]

    def close_feedback(self):
        feedback=getattr(self,"feedback",None)
        if feedback is not None:
            feedback.close()
            self.feedback=None

    def seal_cache(self):
        if not getattr(self,"selection",False):
            from ..core.brush_cache import seal
            seal(self)

    def abort_cache(self):
        if not getattr(self,"selection",False):
            from ..core.brush_cache import abort
            abort(self)

    def restore(self):
        if self.selection:
            return super().restore()
        solver=getattr(self,"solver",None)
        if solver is None:
            return super().restore()
        from ..core.mesh import restore_written
        last = self.__dict__.get("_last", {})
        try:
            rows = []   # back only while a vertex still holds what Relax last wrote
            for vertex, coordinate in solver.prev.items():
                index, expected = last.get(vertex, (None, coordinate)) if vertex.is_valid else (None, coordinate)
                rows.append((vertex, index, coordinate, expected))
            self.restore_skipped = restore_written(self.bm, rows)
            if solver.prev:
                vertices=[v for v in solver.prev if v.is_valid]
                self.note_written(vertices,np.array([v.co[:] for v in vertices]))
                faces={f for v in solver.prev if v.is_valid for f in v.link_faces}
                for face in faces:
                    face.normal_update()
                for vertex in {v for f in faces for v in f.verts}:
                    vertex.normal_update()
                attributes = getattr(self, "attributes", None)
                if attributes is not None:
                    attributes.restore()
                bmesh.update_edit_mesh(self.obj.data,loop_triangles=True,destructive=False)
            self.seal_cache()
        finally:
            self.close_feedback()
        return getattr(self, "restore_skipped", 0)

    def _mesh_cache(self):
        old = getattr(self, "_cache", None)
        if old is not None and not self.selection and getattr(self,"_direct_cache",False):
            return old
        if old is None and not self.selection and len(self.bm.faces)>4000:
            from ..core.brush_snapshot import snapshot
            try:
                self._cache=snapshot(self)
                return self._cache
            except (ValueError,RuntimeError,AttributeError):
                # Unsupported host/state: preserve the original BMesh reader.
                self._cache=None
        projection = getattr(self, "_screen_bounds", None)
        rows = None
        if projection is not None and old is projection.cache and self.verts:
            rows = np.array([v.index for v, _, _ in self.verts], dtype=np.int64)
            before = old["co"][rows].copy()
        cache = super()._mesh_cache()
        if rows is not None and cache is old:
            projection.changed(rows[np.any(cache["co"][rows] != before, axis=1)])
        return cache

    def _screen_projection(self, cache):
        if self.selection or len(cache["faces"]) <= 4000:
            return None
        state = getattr(self, "_screen_bounds", None)
        if state is None or state.cache is not cache:
            state = self._screen_bounds = ScreenProjection(self, cache)
        try:
            state.project(self)
        except ValueError:
            self._screen_bounds=None
            return None
        return state

    def _project(self, cache):
        state = self._screen_projection(cache)
        return super()._project(cache) if state is None else state.screen

    def _center(self, cache, project=None):
        if self.selection or len(cache["faces"]) <= 4000:
            return super()._center(cache, project)
        origin, direction = raycast.ray(self.context, self.mouse)
        hit = raycast.ray_cast_sources(self.context, origin, direction, self.srcs)
        if hit:
            return hit.world
        state = self._screen_projection(cache)
        if state is None:
            return super()._center(cache,lambda:BrushStroke._project(self,cache))
        xy = self.mouse
        rows = state.tree.query(xy[0], xy[1], xy[0], xy[1])
        if len(rows):
            bvh = local_bvh(cache["co"], cache["loops"], cache["starts"], state.sizes, rows)
            co, _, _, _ = bvh.ray_cast(self.mi @ origin, (self.mi.to_3x3() @ direction).normalized())
            if co is not None:
                return self.mw @ co
        sx, sy, front = state.screen
        usable = front & ~cache["hide"]
        if not usable.any():
            return None
        d2 = np.where(usable, (sx-xy[0])**2 + (sy-xy[1])**2, np.inf)
        best = cache["verts"][int(np.argmin(d2))]
        return raycast.on_view_plane(self.context, xy, self.mw @ best.co)

    def _occlusion_view(self, cache, candidates, project=None):
        if not self.selection:
            certificate=plane_visibility(self,cache,candidates)
            if certificate is not None:
                return certificate
        state = self._screen_projection(cache)
        if state is None:
            return super()._occlusion_view(cache, candidates, project)
        if not len(candidates):
            return MeshView(self.context, faces=[], indices_valid=True)
        sx, sy, front = state.screen
        rows = candidates[front[candidates]]
        if not len(rows):
            return MeshView(self.context, faces=[], indices_valid=True)
        margin = 5
        face_rows = state.tree.query(sx[rows].min()-margin, sy[rows].min()-margin,
                                     sx[rows].max()+margin, sy[rows].max()+margin)
        faces = [cache["faces"][int(row)] for row in face_rows]
        bvh = local_bvh(cache["co"], cache["loops"], cache["starts"], state.sizes, face_rows) if faces else None
        return MeshView(self.context, faces=faces, bvh=bvh, indices_valid=True)

    def __init__(self, context, event, settings, selection=False, grid=False):
        self.grid = grid    # Alt: Grid Relax instead of Robust Relax
        try:
            self._initialize(context,event,settings,selection)
        except Exception:
            # begin() takes a global owner before the constructor returns. The
            # operator has no stroke attribute yet if construction fails, so its
            # normal teardown cannot reach this partially initialized object.
            attributes=getattr(self,"attributes",None)
            for close in (self.abort_cache,self.close_feedback,
                          getattr(attributes,"close",None)):
                if close is not None:
                    try:
                        close()
                    except Exception:
                        pass  # Keep the constructor's original exception.
            from ..core.brush_cache import release_stroke
            release_stroke(self,context)
            raise

    def _initialize(self,context,event,settings,selection):
        if not selection:
            from ..core.brush_cache import begin
            begin(self,context)
        super().__init__(context, event, settings, selection=selection)
        self.solver = SmoothSolver(self)
        self.feedback=None
        self._relaxable = {}
        if self.scene.pen3d:
            # 3D: no source to keep the vertices on, so they would shrink a curved
            # surface flat. Keep them on the mesh's own shape as it was when the stroke began.
            cache = self._mesh_cache()
            if len(cache["faces"]):
                retained=getattr(self,"_retained_entry",None)
                if retained is not None:
                    self.solver.self_surface=retained.surface(self)
                else:
                    keep = np.ones(len(cache["starts"]), dtype=bool)
                    self.solver.self_surface = bvh_from_loops(cache["co"], cache["loops"], cache["starts"], keep)
        if not selection:
            from ..core.brush_cache import attach
            attach(self,self.solver.self_surface)
        space=context.space_data
        if (not selection and not bpy.app.background and len(self.bm.faces)>250000
                and self.scene.pen3d and not self.srcs
                and prefs.get(context).dense_feedback and space.shading.type=="SOLID"
                and space.overlay.show_retopology):
            from ..core.brush_feedback import LocalFeedback
            self.feedback=LocalFeedback(self)

    def step(self, dt, pressure=1.0):
        if not self.selection:
            self._collect()
        if not self.verts or self.radius_world <= 0.0:
            return  # nothing under the brush (yet)
        # relaxable (has faces; n-gons count too, they only take no part in Square
        # Quads): depends on topology only, remembered for the stroke
        known = self._relaxable
        for v, _, _ in self.verts:
            if v not in known:
                known[v] = bool(v.link_faces)
        verts = {v for v, _, _ in self.verts if known[v]}
        weights = {v: w for v, _, w in self.verts if v in verts}
        self.solver.relax_step(self.context, verts, weights, self.radius_world,
                               dt, self.settings.strength, pressure)



class PPO_OT_relax(bpy.types.Operator):
    bl_idname = "ppo.relax"
    bl_label = "Relax"
    bl_translation_context = "Poly Toolset"
    bl_description = "Relax topology; hold Ctrl to temporarily drag with the same brush, Alt for Grid Relax"
    bl_options = {"REGISTER", "UNDO"}

    grid: BoolProperty(translation_context="Poly Toolset", name="Grid Relax", options={"SKIP_SAVE", "HIDDEN"},
                       description="Even out edge lengths, square up quads and straighten edge lines")

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and context.region_data is not None

    def invoke(self, context, event):
        _resume_cursor(context)
        grid = getattr(self, "grid", False)
        self.stroke = RelaxStroke(context, event, props.settings(context).relax, grid=grid)
        self.dragging = bool(event.ctrl)   # Ctrl+drag: temporary Tweak
        if self.dragging:
            self.stroke.begin_drag(event)
        self.pressure = event.pressure if event.pressure > 0 else 1.0
        self.time = time.perf_counter()
        self.timer = context.window_manager.event_timer_add(1 / 60, window=context.window)
        self.timer_duration = self.timer.time_duration
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _end(self, context, restore=False):
        timer = getattr(self, "timer", None)
        if timer is not None:
            self.timer = None
            context.window_manager.event_timer_remove(timer)
        if restore:
            return self.stroke.restore()   # the number of vertices left alone
        self.stroke.solver.finish()
        return 0

    def cancel(self, context):
        self._end(context, restore=True)

    def modal(self, context, event):
        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            self._end(context)
            return {"FINISHED"}
        if event.type == "WINDOW_DEACTIVATE" or (event.type in {"RIGHTMOUSE", "ESC"} and event.value == "PRESS"):
            if not safe_mesh.restore_allowed(self, context):   # keep outside edits
                self.report({"WARNING"}, tr_rpt("The mesh being edited changed; the tool stopped"))
                return {"CANCELLED"}   # timer and stroke go with the operator teardown; nothing written
            if self._end(context, restore=True):   # an outside edit of a Relaxed vertex was kept
                self.report({"WARNING"}, tr_rpt("The mesh being edited changed; the tool stopped"))
            return {"CANCELLED"}
        if event.type != "TIMER" and bool(event.ctrl) != self.dragging:
            self.dragging = bool(event.ctrl)
            if self.dragging:
                self.stroke.begin_drag(event)
            else:
                self.stroke.end_drag()
                self.stroke.mouse = raycast.mouse(event)
            self.time = time.perf_counter()
        if event.type in {"MOUSEMOVE", "INBETWEEN_MOUSEMOVE"}:
            if self.dragging:
                self.stroke.drag(event)
            else:
                self.stroke.mouse = raycast.mouse(event)
            Hover.xy = raycast.mouse(event)
            self.pressure = event.pressure if event.pressure > 0 else 1.0
        elif event.type == "TIMER":
            if not timer_tick(self):
                return {"PASS_THROUGH"}
            now = time.perf_counter()
            if not self.dragging:
                self.stroke.step(min(0.05, now - self.time), self.pressure)
            self.time = now
        context.region.tag_redraw()
        return {"RUNNING_MODAL"}


class LoopToolsSelectionStroke:
    """Apply the embedded LoopTools core; frames contain coordinates only.

    Only selected vertices ever move, so every per-step cost is kept to the selection
    and its one-ring: the selected vertices and edges are collected once, frames hold
    just their coordinates, and the connectivity check covers the selection's
    neighbourhood. A full-mesh pass per step cost seconds on million-face meshes.
    """

    def __init__(self, context):
        self.context = context
        self.obj = context.edit_object
        self.data = self.obj.data
        self.scene = props.settings(context)
        bm = self._mesh()
        self.edit_bm = bm
        self.counts = (len(bm.verts), len(bm.edges), len(bm.faces))
        self.verts = [v.index for v in bm.verts if v.select and not v.hide]
        self.edge_keys = selected_edge_keys(bm)
        self.topology = self._topology(bm)
        self.originals = self.snapshot()
        self._boundary = None
        self._view = None
        self.bm, self.mw = bm, self.obj.matrix_world.copy()
        self.mi, self.clip, self.selection = self.mw.inverted(), MirrorClip(self.obj), True
        from ..core.glue import GlueState
        self.glue = GlueState(self)
        from ..core.face_attributes import Correction
        self.attributes = Correction(self.bm, context)

    @property
    def boundary(self):
        # Only Slide boundary mode needs it; built on first use, once per operation.
        if self._boundary is None:
            bm, clip = self.edit_bm, MirrorClip(self.obj)
            self._boundary = [(e.verts[0].co.copy(), e.verts[1].co.copy())
                              for e in bm.edges if e.is_boundary and not
                              (clip.on_plane(e.verts[0].co) and clip.on_plane(e.verts[1].co))]
        return self._boundary

    def _topology(self, bm):
        """Connectivity of the selected vertices' one-ring: their edges and faces with
        the vertex indices of each. Together with the element counts this catches an
        outside edit that would make our stored indices point at the wrong vertices."""
        verts = bm.verts
        return tuple((tuple((e.index, e.verts[0].index, e.verts[1].index) for e in verts[i].link_edges),
                      tuple((f.index, tuple(v.index for v in f.verts)) for f in verts[i].link_faces))
                     for i in self.verts)

    def _mesh(self):
        if self.context.mode != "EDIT_MESH":
            raise RuntimeError("The mesh being edited changed; the tool stopped")
        if self.context.edit_object != self.obj or self.obj.data != self.data:
            raise RuntimeError("The mesh being edited changed; the tool stopped")
        bm = bmesh.from_edit_mesh(self.data)
        if hasattr(self, "edit_bm") and bm is not self.edit_bm:
            raise RuntimeError("The mesh being edited changed; the tool stopped")
        for elements in (bm.verts, bm.edges, bm.faces):
            elements.ensure_lookup_table()
            elements.index_update()
        if hasattr(self, "counts") and self.counts != (len(bm.verts), len(bm.edges), len(bm.faces)):
            raise RuntimeError("The mesh being edited changed; the tool stopped")
        if hasattr(self, "topology") and self._topology(bm) != self.topology:
            raise RuntimeError("The mesh being edited changed; the tool stopped")
        return bm

    def snapshot(self):
        verts = self._mesh().verts
        return [verts[i].co.copy() for i in self.verts]

    def restore_frame(self, frame):
        bm = self._mesh()
        if len(frame) != len(self.verts):
            raise RuntimeError("The mesh being edited changed; the tool stopped")
        verts = bm.verts
        for i, co in zip(self.verts, frame):
            verts[i].co = co
        self.attributes.apply([verts[i] for i in self.verts])
        bm.normal_update()
        bmesh.update_edit_mesh(self.data, loop_triangles=True, destructive=False)

    def restore(self):
        self.restore_frame(self.originals)
        self.attributes.restore()

    def _visibility(self):
        # The view cannot change while the operator runs (navigation ends it), so the
        # occlusion BVH is built once; visibility itself is re-tested every step.
        if self._view is None:
            self._view = MeshView(self.context)
        self._view._visible.clear()
        return self._view

    def step(self):
        bm = self._mesh()
        clip = MirrorClip(self.obj)
        view = self._visibility() if not self.scene.include_occluded else None
        crease_v = bm.verts.layers.float.get("crease_vert")
        crease_e = bm.edges.layers.float.get("crease_edge")
        allowed = set()
        for index in self.verts:
            v = bm.verts[index]
            if not v.select or v.hide or pinned(bm, v):
                continue
            if self.scene.mask_boundary == "EXCLUDE" and is_boundary(v, clip):
                continue
            if not self.scene.include_corners and (len(v.link_edges) == 2 or
                    (len(v.link_edges) == 4 and len(v.link_faces) == 3)):
                continue
            if not self.scene.include_seams and any(e.seam for e in v.link_edges):
                continue
            if not self.scene.include_sharps and any(not e.smooth for e in v.link_edges):
                continue
            if not self.scene.include_creases and ((crease_v is not None and v[crease_v] > 0) or
                    (crease_e is not None and any(e[crease_e] > 0 for e in v.link_edges))):
                continue
            if view is not None and not view.visible(v):
                continue
            allowed.add(index)
        if not allowed:
            return
        # Calculate all targets from the same pre-step coordinates. Selection,
        # topology and custom data are not changed by the numerical solver.
        moves = selected_loop_moves(bm, edges=list(self.edge_keys))
        before = self.snapshot()
        start = {index: co for index, co in zip(self.verts, before)}
        glue = getattr(self, "glue", None)
        before_glue = glue.begin_drag(bm.verts[i] for i in allowed) if glue is not None else {}
        written = []
        try:
            for index, target in moves:
                if index in allowed:
                    v = bm.verts[index]
                    if self.scene.mask_boundary == "SLIDE" and is_boundary(v, clip):
                        target = BrushStroke.snap_to_boundary(self, target)
                    v.co = clip.apply(start[index], target)
                    written.append(v)
            if before_glue:
                def surface(v, world):
                    co = self.mi @ world
                    if self.scene.mask_boundary == "SLIDE" and is_boundary(v, clip):
                        co = BrushStroke.snap_to_boundary(self, co)
                    return self.mw @ clip.apply(start.get(v.index, v.co), co)
                written = glue.correct_drag(before_glue, written, surface)
            self.attributes.apply(written)
            bm.normal_update()
            bmesh.update_edit_mesh(self.data, loop_triangles=True, destructive=False)
        except Exception:
            try:
                self.restore_frame(before)
            except Exception:
                pass
            raise


class PPO_OT_smooth_selection(bpy.types.Operator):
    bl_idname = "ppo.smooth_selection"
    bl_label = "Relax Selection"
    bl_translation_context = "Poly Toolset"
    bl_description = "Alt+Wheel relaxes the selection; wheel down steps back"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and context.region_data is not None

    def invoke(self, context, event):
        self.looptools = props.settings(context).smooth_method == "LOOPTOOLS"
        if self.looptools:
            self.stroke = LoopToolsSelectionStroke(context)
        else:
            self.stroke = RelaxStroke(context, event, props.settings(context).relax, selection=True,
                                      grid=props.settings(context).smooth_method == "GRID")
        if not self.stroke.verts:
            return {"CANCELLED"}
        self.frames = []
        try:
            self._wheel(event.type)
        except Exception as exc:
            self.cancel(context)
            self.report({"ERROR"}, tr_rpt(friendly_error(exc, "Relax could not continue here")))
            return {"CANCELLED"}
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        stroke = getattr(self, "stroke", None)
        if stroke is not None:
            try:
                stroke.restore()
            except (RuntimeError, ReferenceError):
                # An invalid target must never receive an old coordinate frame.
                pass

    def _wheel(self, type):
        if type == "WHEELUPMOUSE":
            if self.looptools:
                frame = self.stroke.snapshot()
                self.stroke.step()
                self.frames.append(frame)
                return
            self.frames.append({v: v.co.copy() for v, _co, _w in self.stroke.verts})   # stroke verts only
            if self.stroke.scene.smooth_method == "NATIVE":
                bmesh.ops.smooth_vert(self.stroke.bm, verts=[v for v, _, _ in self.stroke.verts],
                                      factor=self.stroke.settings.strength, use_axis_x=True,
                                      use_axis_y=True, use_axis_z=True)
                for v, original, _ in self.stroke.verts:
                    hit = raycast.nearest_on_sources(self.stroke.context, self.stroke.mw @ v.co, self.stroke.srcs)
                    co = self.stroke.mi @ hit.world if hit else v.co.copy()
                    if self.stroke.scene.mask_boundary == "SLIDE" and is_boundary(v, self.stroke.clip):
                        co = self.stroke.snap_to_boundary(co)
                    v.co = self.stroke.clip.apply(original, co)
                written = [v for v, _co, _w in self.stroke.verts]
                # what Esc compares against (restore_written), as the solver steps record it
                self.stroke.note_written(written, np.array([v.co[:] for v in written]))
                from ..core.face_attributes import apply
                apply(self.stroke, written)
                self.stroke.update_mesh()
            elif self.stroke.scene.smooth_method == "GRID":
                # Grid Relax takes small capped steps: several per wheel notch
                for _ in range(GRID_STEPS_PER_NOTCH):
                    self.stroke.step(1 / 30)
            else:
                self.stroke.step(1 / 30)
        elif type == "WHEELDOWNMOUSE" and self.frames:
            if self.looptools:
                self.stroke.restore_frame(self.frames[-1])
                self.frames.pop()
                return
            frame = self.frames.pop()
            moved = [v for v, co in frame.items() if v.co != co]
            for v, co in frame.items():
                v.co = co
            written = list(frame)
            # what Esc compares against (restore_written) is now this frame
            self.stroke.note_written(written, np.array([v.co[:] for v in written]))
            from ..core.face_attributes import apply
            apply(self.stroke, moved)
            self.stroke.update_mesh()

    def modal(self, context, event):
        if (event.value == "PRESS" and (event.type in {"ESC", "RIGHTMOUSE"} or (event.type == "WHEELDOWNMOUSE" and self.frames))
                and not safe_mesh.restore_allowed(self, context)):   # keep outside edits
            self.report({"WARNING"}, tr_rpt("The mesh being edited changed; the tool stopped"))
            return {"CANCELLED"}
        down = self.__dict__.get("_alt_lmb")
        if down is not None and event.type == "MOUSEMOVE":   # an Alt+drag ends the smoothing
            limit = context.preferences.inputs.drag_threshold_mouse
            if abs(event.mouse_x - down[0]) > limit or abs(event.mouse_y - down[1]) > limit:
                self.__dict__["_alt_lmb"] = None
                return ({"FINISHED"} if self.frames else {"CANCELLED"}) | {"PASS_THROUGH"}
        if event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"} and event.value == "PRESS":
            try:
                self._wheel(event.type)
            except Exception as exc:
                self.cancel(context)
                self.report({"ERROR"}, tr_rpt(friendly_error(exc, "Relax could not continue here")))
                return {"CANCELLED"}
        elif event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            self.cancel(context)
            return {"CANCELLED"}
        elif event.type == "WINDOW_DEACTIVATE":   # Alt+Tab away: keep the smoothing done so far (as releasing Alt)
            return {"FINISHED"} if self.frames else {"CANCELLED"}
        elif event.type in {"LEFT_ALT", "RIGHT_ALT"} and event.value == "RELEASE":
            return {"FINISHED"} if self.frames else {"CANCELLED"}
        elif event.type == "LEFTMOUSE" and event.alt:
            # Alt+click (also with Shift / Ctrl) while Alt is still held: confirm this
            # loop's smoothing as its own undo step and let Blender's native loop / ring
            # select handle the click. The next Alt+Wheel starts on the new selection.
            #
            # With left-click select, native loop select listens for CLICK, which
            # Blender only synthesizes on release if the PRESS was *not handled*.
            # Ending here on PRESS (FINISHED|PASS_THROUGH counts as handled) swallowed
            # the click, so the first Alt+click did nothing. Pass PRESS / RELEASE through
            # untouched and finish on the synthesized CLICK, which then continues to the
            # native keymap. A drag produces no CLICK: smoothing simply continues.
            if event.value in {"CLICK", "CLICK_DRAG"}:   # a drag may change the selection too
                return ({"FINISHED"} if self.frames else {"CANCELLED"}) | {"PASS_THROUGH"}
            # Blender does not always send CLICK_DRAG to a modal: track the press ourselves
            self.__dict__["_alt_lmb"] = (event.mouse_x, event.mouse_y) if event.value == "PRESS" else None
            return {"PASS_THROUGH"}
        elif event.value == "PRESS" and not event.alt:
            return {"FINISHED", "PASS_THROUGH"}
        return {"RUNNING_MODAL"}


class PPO_OT_brush_adjust(bpy.types.Operator):
    bl_idname = "ppo.brush_adjust"
    bl_label = "Adjust Brush"
    bl_translation_context = "Poly Toolset"
    bl_description = "Drag to change the brush radius, strength or falloff"
    bl_options = {"INTERNAL"}

    mode: EnumProperty(translation_context="Poly Toolset", items=(("RADIUS", "Radius", ""), ("STRENGTH", "Strength", ""), ("FALLOFF", "Falloff", "")), options={"SKIP_SAVE"})

    def invoke(self, context, event):
        _resume_cursor(context)
        self.settings = brush_settings(context)
        if self.settings is None:
            return {"CANCELLED"}
        self.start_x = event.mouse_region_x
        self.field = self.mode.lower()
        self.orig = getattr(self.settings, self.field)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _show(self, context):
        value = getattr(self.settings, self.field)
        # translated like Blender's own header text (Radius / Strength / Falloff use
        # Blender's translations)
        label = iface_(self.mode.title())
        text = f"{label}: {value} px" if self.mode == "RADIUS" else f"{label}: {value:.2f}"
        if context.area is not None:
            context.area.header_text_set(text)
            # the whole area, not only the viewport: the tool header and the sidebar
            # show the same value and must follow while F drags it
            context.area.tag_redraw()

    def _finish(self, context):
        if context.area is not None:
            context.area.header_text_set(None)
            context.area.tag_redraw()

    def modal(self, context, event):
        dx = event.mouse_region_x - self.start_x
        if event.type == "MOUSEMOVE":
            if self.mode == "RADIUS":
                self.settings.radius = max(5, int(self.orig + dx))
            else:
                setattr(self.settings, self.field, min(1.0, max(0.01 if self.mode == "STRENGTH" else 0.0, self.orig + dx / 300.0)))
            self._show(context)
        elif event.type in {"LEFTMOUSE", "RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            self._finish(context)
            return {"FINISHED"}
        elif event.type in {"RIGHTMOUSE", "ESC"} and event.value == "PRESS":
            setattr(self.settings, self.field, self.orig)
            self._finish(context)
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}


RELAX_ALGORITHM = ("smooth_iterations", "relax_even_spacing")


class PPO_OT_brush_reset(bpy.types.Operator):
    bl_idname = "ppo.brush_reset"
    bl_label = "Reset Brush"
    bl_translation_context = "Poly Toolset"
    bl_description = ("Restore the default brush radius, strength and falloff "
                      "(Relax also restores its algorithm settings)")
    bl_options = {"REGISTER", "UNDO"}

    tool: EnumProperty(translation_context="Poly Toolset", items=(("ACTIVE", "Active Tool", ""), ("TWEAK", "Tweak", ""), ("RELAX", "Relax", "")),
                       options={"SKIP_SAVE"})

    def execute(self, context):
        tool_id = {"TWEAK": "ppo.tweak", "RELAX": "ppo.relax"}.get(self.tool)
        settings = brush_settings(context, tool_id)
        if settings is None:
            return {"CANCELLED"}
        for name in ("radius", "strength", "falloff"):
            settings.property_unset(name)
        if settings == props.settings(context).relax:
            scene = props.settings(context)
            for name in RELAX_ALGORITHM:
                scene.property_unset(name)
        if context.area is not None:
            context.area.tag_redraw()
        return {"FINISHED"}


def _adjusting(context):
    """F / Shift+F / Ctrl+F radius-strength-falloff drag is running."""
    window = getattr(context, "window", None)
    try:
        return window is not None and any(
            getattr(op, "bl_idname", "") in {"PPO_OT_brush_adjust", "ppo.brush_adjust"}
            for op in window.modal_operators)
    except (AttributeError, ReferenceError):
        return False


def _draw_cursor(xy):
    """Brush ring, drawn the way Blender draws its own sculpt / paint brush cursor.

    Registered with WindowManager.draw_cursor_add (a paint cursor): Blender redraws it on
    every mouse move and passes the live cursor position, also while another operator
    (orbit, pan, zoom, a stroke) owns the events. The previous POST_PIXEL overlay used the
    last position PolyTailor itself had seen, so the ring stayed behind while orbiting and jumped
    to the cursor afterwards. While F adjusts the brush the ring stays where it started.
    """
    context = bpy.context
    region = context.region
    if context.mode != "EDIT_MESH" or region is None or region.type != "WINDOW":
        return
    if _cursor_key(context) in _paused_cursors:
        return  # No settings queries, GPU work or redraw scheduling behind a popup.
    from ..ui import custom_pie
    wheel = custom_pie._active
    if wheel is not None and wheel.area == context.area:
        return
    settings = brush_settings(context)
    if settings is None:
        return
    # Paint cursors are drawn in WINDOW pixel space (Blender's own brush cursor
    # translates by region.winrct before drawing), and xy is the window cursor
    # position: draw at xy as is. Region-space anchors are shifted into window space.
    if _adjusting(context) and Hover.xy is not None:
        center = Hover.xy + Vector((region.x, region.y))
    else:
        center = Vector((xy[0], xy[1]))
    accent = palette.accent(context)
    r = settings.radius * prefs.ui_scale(context)
    with draw.Blend():
        draw.ring(center, r, (*accent[:3], 0.9), width=2.0)
        inner = r * (1.0 - settings.falloff * 0.5)
        draw.ring(center, inner, (*accent[:3], 0.35), width=1.0)
        draw.disc(center, max(2.0, r * settings.strength * 0.08), (*accent[:3], 0.8))


_classes = (PPO_OT_brush_hover, PPO_OT_tweak, PPO_OT_relax, PPO_OT_smooth_selection, PPO_OT_brush_adjust, PPO_OT_brush_reset)
_handle = None


def _install_cursor():
    """(Re)attach the brush ring paint cursor to the current window manager.

    Paint cursors live on the window manager, which Blender replaces when a .blend is
    loaded with its UI - including the startup file right after add-ons register at
    launch, File > New / Open / Revert and recovery. The old cursor is freed with it,
    so the ring vanished after restarting Blender or opening a file. Re-adding on every
    load_post fixes that. Removing a handle Blender already freed is a documented no-op
    (bpy_rna_callback: "Blender has already freed the data ... don't decrement twice").
    """
    global _handle
    if _handle is not None:
        try:
            bpy.types.WindowManager.draw_cursor_remove(_handle)
        except (ValueError, TypeError):
            pass
    _handle = bpy.types.WindowManager.draw_cursor_add(_draw_cursor, (), "VIEW_3D", "WINDOW")


@bpy.app.handlers.persistent
def _on_load_post(*_args):
    _paused_cursors.clear()
    _install_cursor()


def start():
    """while PolyTailor is in use (session): the brush ring paint cursor, attached again after a file load."""
    _install_cursor()
    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)


def stop():
    global _handle
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    if _handle is not None:
        try:
            bpy.types.WindowManager.draw_cursor_remove(_handle)
        except (ValueError, TypeError):
            pass
        _handle = None


def register():
    _paused_cursors.clear()
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    _paused_cursors.clear()
    stop()
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
