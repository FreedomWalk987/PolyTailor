# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Build: hold Ctrl to preview, Ctrl+click to insert, keep dragging to place.

What gets inserted depends on the selection, the hovered element and the
face type (Scene > ppo > merge3_faces):

  nothing selected        -> new vertex, or split the hovered edge
  one vertex selected     -> curve stroke: new vertex connected to it (Ctrl+drag: freehand)
  edge selected (Quad)    -> quad strip: new edge parallel to the selected one
  edge selected (Tri/Quad)-> triangle; click again with the triangle selected
                             to turn it into a quad
  hovering a second edge  -> bridge both edges with a quad
  Ctrl+drag near a selected edge or open edge chain -> row strip along the stroke
"""

from dataclasses import dataclass
from types import SimpleNamespace
from math import cos, sin, hypot

import bmesh
import bpy
from mathutils import Vector
from ..i18n import tr as iface_, tr_rpt, friendly_error   # add-on context first; friendly_error

from .. import prefs, props, palette
from ..core import draw, raycast, safe_mesh, snap, units
from ..core.pick import MeshView
from ..core.mesh import MeshSnapshot, merge_moved, adaptive_radius
from .brush import MirrorClip
from .pinning import pinned
from ..core.meshpen_logic import (MeshPenLogic, BuildStroke, strip_rows, relieve_strip,
                                  strip_cell_normal, segment_closest, flip_toward_view,
                                  chain_sides_with_faces,
                                  chain_profile, chain_rows, relieve_chain, chain_cells_valid,
                                  chain_epsilon, strip_overlaps, CELL_BUDGET)

# Header text for chains that cannot start a row strip (shown while hovering them).
STROKE_XFORM_GAIN = .6     # Ctrl+move default; Preferences > Stroke Transform Sensitivity
STROKE_XFORM_FLOOR = 48.    # px x UI scale, minimum reference length for rotate / scale
MOVED_PX = 2.0     # px x UI scale: an existing vertex moved at least this far by a click's drag may be merged
CHAIN_REASONS = {
    "BOTH": "The chain already has faces on both sides: select a chain of open edges",
    "LONG": "Edge chain too long for a quad strip",
    "DEGENERATE": "Edge chain has overlapping points",
}
from ..core.polypen_support import bind_view, reset_projection_cache
from ..core.meshpen_logic import HOVER_PX

NONE = "NONE"
VERT_EDGE = "VERT_EDGE"
EDGE_QUAD = "EDGE_QUAD"
EDGE_TRI = "EDGE_TRI"
EDGE_BRIDGE = "EDGE_BRIDGE"
TRI_QUAD = "TRI_QUAD"


def _collapsed(points, relative=1e-9):
    """Any two consecutive points coincide, relative to the polyline's own length
    (no absolute world distance: the same at 2 cm and at 20 cm)."""
    lengths = [(b-a).length for a, b in zip(points, points[1:])]
    tolerance = sum(lengths) * relative
    return any(length <= tolerance for length in lengths)


def _wheel_delta(event):
    """Wheel direction is a pulse; ignore optional button-style releases."""
    if event.value not in {"PRESS", "NOTHING"}:
        return 0
    return {"WHEELUPMOUSE": 1, "WHEELDOWNMOUSE": -1}.get(event.type, 0)


@dataclass
class Plan:
    action: str = NONE


class PolyPenState:

    def __init__(self, context):
        self.context = context
        self.scene = props.settings(context)
        self.pending_selection = []
        self.slide_constraints = {}
        self.refresh()

    def insert_mode(self):
        """The face type: Tri/Quad, or quads only (the "4" face type)."""
        return "QUAD-ONLY" if self.scene.merge3_faces == "QUAD" else "TRI/QUAD"

    def refresh(self):
        reset_projection_cache()
        self.view = MeshView(self.context)
        self.bm, self.mw = self.view.bm, self.view.mw
        self.bm.edges.index_update()   # stable order for the edge-strip pick
        self.bm.verts.index_update()   # stable chain start for the row strip
        self.mi = self.mw.inverted()
        self.clip = MirrorClip(self.view.obj)
        self.view_key = self._view_key()
        bind_view(self.view)
        self.logic = MeshPenLogic(self.context, SimpleNamespace(mouse_region_x=0, mouse_region_y=0))

    def _view_key(self):
        # the object transform too (highlights are cached in screen space)
        region = self.context.region
        return (tuple(tuple(row) for row in self.context.region_data.perspective_matrix),
                region.width, region.height,
                tuple(tuple(row) for row in self.context.edit_object.matrix_world))

    def selection(self):
        return tuple([e for e in sequence if e.select and not e.hide]
                     for sequence in (self.bm.verts, self.bm.edges, self.bm.faces))

    def screen(self, item):
        return self.view.screen(item) if isinstance(item, bmesh.types.BMVert) else raycast.to_screen(self.context, item)

    def place(self, xy, depth_world, moving=()):
        """Vertex / Edge / Edge Center snap if one is in range, else the surface (or the
        view plane at depth_world) under xy."""
        snapped = snap.snap(self.context, xy, self.view.srcs, exclude=moving)
        if snapped is not None:
            return snapped
        snap.finish()   # no marker for plain surface placement (the common case)
        return raycast.surface_or_plane(self.context, xy, depth_world, self.view.srcs).world

    def plan(self, xy):
        # The selection sets and the vertex projection are kept between mouse moves;
        # they are dropped where the mesh or the selection changes (refresh, commit,
        # placement, finish_placement) - re-collecting them from the whole mesh on
        # every move cost ~40 ms per move at 50,000 vertices.
        if not self.bm.is_valid or self._view_key() != self.view_key:
            self.refresh()
        bind_view(self.view)
        self.logic.update(self.context, SimpleNamespace(mouse_region_x=xy.x, mouse_region_y=xy.y),
            self.insert_mode(), self.scene.quad_stability,
            self.scene.knife_junctions, self.scene.knife_constrain)
        return Plan(action=self.logic.state.name)

    def close_attributes(self):
        attributes = getattr(self, "attributes", None)
        if attributes is not None:
            attributes.close()
            self.attributes = None

    def commit(self, plan):
        reset_projection_cache()
        if plan.action == NONE:
            return []
        self.close_attributes()
        if plan.action in {EDGE_TRI, EDGE_QUAD, EDGE_BRIDGE, TRI_QUAD}:
            from ..core.face_attributes import BuildExtension
            self.attributes = BuildExtension(
                self.bm, self.context, self.logic.bme,
                face=self.logic.bmf if plan.action == TRI_QUAD else None,
                far=getattr(self.logic, "bridge_far", ()) or ())
        moving, later, free_move = self.logic.commit(self.context, None)
        if getattr(self, "attributes", None) is not None:
            self.attributes.bind(later)
        self.logic.selected = None   # the commit changed the mesh and the selection
        self.pending_selection = later
        self.slide_constraints = {}
        if not free_move:
            for vertex in moving:
                if isinstance(vertex, bmesh.types.BMVert) and len(vertex.link_edges) == 2:
                    neighbors = [e.other_vert(vertex) for e in vertex.link_edges]
                    self.slide_constraints[vertex] = neighbors
        self.bm.normal_update()
        bmesh.update_edit_mesh(self.view.obj.data, loop_triangles=True, destructive=True)
        return [v for v in moving if isinstance(v, bmesh.types.BMVert) and v.is_valid]

    def finish_placement(self):
        self.close_attributes()
        reset_projection_cache()
        self.logic.selected = None
        if self.pending_selection:
            for sequence in (self.bm.verts, self.bm.edges, self.bm.faces):
                for elem in sequence:
                    elem.select_set(False)
            self.bm.select_history.clear()
            for elem in self.pending_selection:
                if elem.is_valid:
                    elem.select_set(True)
                    self.bm.select_history.add(elem)
            self.bm.select_flush(True)
            self.bm.select_flush(False)
        self.pending_selection = []
        self.slide_constraints = {}

    def draw(self, context):
        bind_view(self.view)
        self.logic.draw(context)


class PPO_OT_polypen(bpy.types.Operator):
    bl_idname = "ppo.polypen"
    bl_label = "Build"
    bl_translation_context = "Poly Toolset"
    bl_description = "Hold Ctrl to preview, Ctrl+LMB to add"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and context.region_data is not None

    def invoke(self, context, event):
        self.state = PolyPenState(context)
        self.region = context.region.as_pointer()
        self.mouse = raycast.mouse(event)
        self.plan = self.state.plan(self.mouse)
        self.placing = None  # (verts, original world positions, press xy)
        self.snapshot = None
        self.stroke_pending = None
        self.stroke_curve = None
        self.stroke_edge = None
        self.stroke_chain = None      # ordered BMVerts of a K >= 3 chain row strip
        self.stroke_profile = None
        self.stroke_overlap = []      # crossing grid segments ((i, j), (i2, j2))
        self.stroke_reject = None     # why the hovered chain cannot start a strip
        self.stroke_reject_shown = False
        self.stroke_warning = None    # the stroke / strip warning shown at the cursor
        self.stroke_rows = []
        self.stroke_folded = False
        self.stroke_drawing = False
        self.stroke_world = []
        self.stroke_cursor = None
        self.stroke_xform, self.stroke_press = (1.0, 0.0), None
        self.line_segments = 1
        self.line_world = []
        self.line_placing = None
        self.ctrl_held = event.ctrl
        self.stroke_hover = None
        self._stroke_hover_update(context)
        self.handle = bpy.types.SpaceView3D.draw_handler_add(self._draw, (), "WINDOW", "POST_PIXEL")
        context.window_manager.modal_handler_add(self)
        context.region.tag_redraw()
        return {"RUNNING_MODAL"}

    def _end(self, context):
        self.state.close_attributes()
        snap.finish()
        if context.area is not None:
            context.area.header_text_set(None)
        if self.handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self.handle, "WINDOW")
            self.handle = None
        if self.snapshot is not None:
            self.snapshot.close()
            self.snapshot = None
        if context.region is not None:   # none when the 3D View was closed
            context.region.tag_redraw()

    def _rollback(self):
        self.state.close_attributes()
        if self.snapshot is not None:
            if self.snapshot.restore() is False:   # shape keys changed outside; kept
                self.report({"WARNING"}, tr_rpt("The mesh being edited changed; the tool stopped"))
            else:
                self._restore_select_mode()
            self.snapshot.close()
            self.snapshot = None
            self.placing = None
            self.line_placing = None
            self.state.refresh()

    def _restore_select_mode(self):
        """The select mode recorded with this snapshot (the click turned Vertex mode on)."""
        saved = getattr(self, "select_mode_saved", None)
        self.select_mode_saved = None
        if saved is None or saved[0] is not self.snapshot:
            return
        try:
            tool_settings = bpy.context.tool_settings
            if tuple(tool_settings.mesh_select_mode) != saved[1]:
                tool_settings.mesh_select_mode = saved[1]
        except (AttributeError, ReferenceError, TypeError, RuntimeError):
            pass

    def _outside_edit(self):
        """before Build's full rollback while placing: ONE O(n) comparison of the edit mesh with
        what the pen expects (its commit, plus its own moves of the placed vertices). Anything else differs: an outside
        edit, kept (no rollback). A rollback inside our own event (an error there) is not checked."""
        placing = getattr(self, "placing", None)
        if placing is None or self.snapshot is None or getattr(self, "_ppo_internal_rollback", False):
            return False
        return self.snapshot.outside_changed(placing[0])

    def _ppo_refresh(self, context):
        # The shared guard calls this when the mesh changed outside the pen between
        # events (never while placing): drop the cached view and selection sets.
        self.stroke_pending = None
        self.stroke_curve = None
        self.stroke_edge = None
        self.stroke_chain = None
        self.stroke_overlap = []
        self.stroke_reject_shown = False
        self.stroke_rows = []
        self.stroke_world = []
        self.stroke_cursor = None
        self.stroke_xform, self.stroke_press = (1.0, 0.0), None
        self.stroke_drawing = False
        if context.area is not None:
            context.area.header_text_set(None)
        self.line_world = []
        self.line_placing = None
        self.stroke_hover = None
        self.state.refresh()
        self._replan(context)

    def cancel(self, context):
        if self.snapshot is not None and self._outside_edit():   # keep the outside edit
            self.snapshot.close()
            self.snapshot = None
        self._rollback()
        self._end(context)

    def _replan(self, context):
        try:
            self.plan = self.state.plan(self.mouse)
        except ReferenceError:
            self.state.refresh()
            self.plan = self.state.plan(self.mouse)
        self._line_update(context)
        self._stroke_hover_update(context)
        context.region.tag_redraw()

    def _follow_view(self, context):
        """Orbit / pan / zoom or an object move without a mouse event
        (another modal owns the input, smooth view, a driver...): re-plan, so the
        drag-able highlight and the preview follow instead of staying where they were."""
        if self.stroke_curve is not None or self.stroke_pending is not None:
            self._follow_stroke(context)
            return
        try:
            if self.state._view_key() == self.state.view_key or safe_mesh.native_modal_running(context):
                return   # (never read the BMesh while a native modal holds it)
            self.state.refresh()
            self._replan(context)
        except (ReferenceError, AttributeError):
            return

    def _follow_stroke(self, context):
        """a stroke / strip / row-strip preview is held in world space, so an
        orbit / pan redraws it in place; an object move carries it with the object."""
        st = self.state
        try:
            key = st._view_key()
            if key == st.view_key:
                return
            if key[3] != st.view_key[3]:
                if self.stroke_drawing:
                    return   # mid-draw: re-check after release
                delta = st.view.obj.matrix_world @ st.mi
                self.stroke_world = [delta @ p for p in self.stroke_world]
                self.stroke_rows = [tuple(delta @ p for p in row) for row in self.stroke_rows]
                if self.stroke_profile is not None and "pivot" in self.stroke_profile:
                    self.stroke_profile["pivot"] = delta @ self.stroke_profile["pivot"]
                st.mw = st.view.obj.matrix_world
                st.mi = st.mw.inverted()
            st.view_key = key
        except (ReferenceError, AttributeError):
            return

    def modal(self, context, event):
        if context.region is None or context.area is None:   # the 3D View was closed: Build ends
            if self.placing:
                self._rollback()   # a click still being placed is undone, as with Esc
            self._end(context)
            return {"CANCELLED"}
        if (event.type == "LEFTMOUSE" and event.value == "PRESS" and not self.placing
                and self.stroke_pending is None and not self.stroke_drawing
                and not self._over_own_region(context, event)):
            # a click in another area or region never builds in this view (off-screen);
            # the click goes to where it was made; a held stroke is kept (Build stays), otherwise the pen ends
            if getattr(self, 'stroke_curve', None) is not None:
                return {"PASS_THROUGH"}
            self._end(context)
            return {"FINISHED", "PASS_THROUGH"}
        stroke_result = self._stroke_event(context, event)
        if stroke_result is not None:
            return stroke_result
        delta = _wheel_delta(event)
        if delta and not self.placing:
            # A native wheel pulse can arrive before its final mouse move.
            # Resolve ownership at the wheel's position before navigation gets it.
            self.mouse = raycast.mouse(event)
            self._replan(context)
            if self._line_anchor() is not None:
                self.line_segments = min(BuildStroke.MAX_SEGMENTS,
                                         max(1, self.line_segments + delta))
                self._line_update(context)
                context.region.tag_redraw()
                return {"RUNNING_MODAL"}
        if (event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"}
                and event.value == "RELEASE" and not self.placing
                and self._line_anchor() is not None):
            return {"RUNNING_MODAL"}
        if event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"} and not self.placing:
            self._end(context)
            return {"FINISHED", "PASS_THROUGH"}
        if event.type in {"MOUSEMOVE", "INBETWEEN_MOUSEMOVE"}:
            if self.placing:
                self.mouse = self.precise.update(event)   # Shift: finer placement
                self._place(context)
            else:
                self.mouse = raycast.mouse(event)
                self._replan(context)
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE":
            if event.value == "PRESS" and not self.placing:
                self.mouse = raycast.mouse(event)
                self._insert(context)
            elif event.value == "RELEASE" and self.placing:
                weld = self._weld_candidates(context)
                keep = [v for v in self.placing[0] if v.is_valid and v not in weld]
                if context.scene.tool_settings.use_mesh_automerge:
                    targets = merge_moved(context, self.state.bm, weld,
                                props.settings(context).merge_radius * prefs.ui_scale(context),
                                adaptive_min=3 * prefs.ui_scale(context) if self.line_placing is not None else None,
                                keep=keep) or {}
                else:   # a drag release acts as a click there: a dragged end joins the vertex under it
                    ends = self._dragged(context, weld)
                    if self.line_placing is not None:   # a Ctrl+Wheel line: only its end vertex, never a middle point
                        ends = [v for v in ends if v is self.placing[0][-1]]
                    targets = self._release_weld(context, ends,
                                                 [v for v in self.placing[0] if v.is_valid and v not in ends])
                    if targets:
                        self.state.pending_selection = [targets.get(e, e) if isinstance(e, bmesh.types.BMVert)
                                                        else e for e in self.state.pending_selection]
                if self.line_placing is not None:
                    endpoint = self.placing[0][-1]
                    endpoint = next((target for source, target in targets.items() if source is endpoint), endpoint)
                    self.state.pending_selection = [endpoint]
                    self.line_placing = None
                self.state.finish_placement()
                self.snapshot.close()
                self.snapshot = None
                self.placing = None
                self.placing_existing = None
                context.area.header_text_set(None)
                self._push_undo()
                if not event.ctrl:
                    self._end(context)
                    return {"FINISHED"}
                # Sync and record the completed click before rebuilding the
                # view for the next preview; no preview is needed on exit.
                self.state.refresh()
                self._replan(context)
            return {"RUNNING_MODAL"}

        if event.type in {"LEFT_CTRL", "RIGHT_CTRL"}:
            if event.value == "RELEASE" and not self.placing:
                self._end(context)
                return {"FINISHED"}
            return {"RUNNING_MODAL"}

        if event.type == "WINDOW_DEACTIVATE" or (event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS"):
            if self.snapshot is not None and (not safe_mesh.restore_allowed(self, context) or self._outside_edit()):
                # an outside edit since our last event: keep it, roll nothing back
                self.snapshot.close()
                self.snapshot = None
                self.report({"WARNING"}, tr_rpt("The mesh being edited changed; the tool stopped"))
            self._rollback()
            self._end(context)
            return {"CANCELLED"}

        if self.placing or event.type in {"TIMER", "NONE", "WINDOW_DEACTIVATE"} or event.value != "PRESS":
            return {"RUNNING_MODAL"}
        # any other key (Ctrl+Z, navigation...) ends the pen and runs normally
        self._end(context)
        return {"FINISHED", "PASS_THROUGH"}

    def _over_own_region(self, context, event):
        """the mouse is over the 3D region this Build runs in (as Fill's _over_own_region)."""
        try:
            for area in context.window.screen.areas:
                for region in area.regions:
                    if region.as_pointer() == self.region:
                        return (region.x <= event.mouse_x < region.x + region.width
                                and region.y <= event.mouse_y < region.y + region.height)
        except (AttributeError, ReferenceError):
            pass
        return False

    def _insert(self, context):
        """The existing Ctrl+click path, also used after a sub-threshold drag."""
        self._replan(context)
        if self.plan.action == NONE:
            return
        self.snapshot = MeshSnapshot(self.state.view.obj)
        self.select_mode_saved = (self.snapshot, tuple(context.tool_settings.mesh_select_mode))
        try:
            if self.line_world:
                created = self._insert_line(context)
            else:
                created = self.state.commit(self.plan)
        except (ValueError, ReferenceError, ZeroDivisionError) as exc:   # the last: geometry seen edge-on
            self._rollback()
            self.report({"WARNING"}, tr_rpt(friendly_error(exc, "This path cannot split the face")))
            self._replan(context)
            return
        if created:
            self.placing = (created, [self.state.mw @ v.co for v in created], self.mouse.copy())
            self.placing_existing = self._existing(created)
            self.snapshot.expect(created)   # what an Esc rollback expects (only these move)
            self.precise = raycast.PreciseMouse(self.mouse)
        else:
            self.state.finish_placement()
            self.snapshot.close()
            self.snapshot = None
            self._push_undo()
        self.plan = Plan()
        self.line_world = []
        context.region.tag_redraw()

    def _existing(self, verts):
        """the vertices of `verts` the mesh had before this click: the snapshot indexed every vertex and keeps a
        copy, so an existing vertex has an index and still sits where the copy has it; a new one has no index yet."""
        backup = getattr(self.snapshot, "backup", None)
        if backup is None:
            return set()
        backup.verts.ensure_lookup_table()
        count = len(backup.verts)
        return {v for v in verts if v.is_valid and 0 <= v.index < count and backup.verts[v.index].co == v.co}

    def _weld_candidates(self, context):
        """Auto Merge on release: the click's new vertices, and an existing one only when the drag moved it more
        than MOVED_PX on screen. An existing vertex the click only reused is never merged."""
        verts, origins, _start = self.placing
        existing = getattr(self, "placing_existing", None) or set()
        limit = MOVED_PX * prefs.ui_scale(context)
        out = []
        for v, world in zip(verts, origins):
            if not v.is_valid:
                continue
            if v in existing:
                a = raycast.to_screen(context, world)
                b = raycast.to_screen(context, self.state.mw @ v.co)
                if a is None or b is None or (b - a).length <= limit:
                    continue
            out.append(v)
        return out

    def _dragged(self, context, weld):
        """the weld candidates the drag moved more than MOVED_PX on screen from where the click placed them (a
        plain click welds nothing more than today)."""
        verts, origins, _start = self.placing
        limit = MOVED_PX * prefs.ui_scale(context)
        out = []
        for v, world in zip(verts, origins):
            if not v.is_valid or v not in weld:
                continue
            a = raycast.to_screen(context, world)
            b = raycast.to_screen(context, self.state.mw @ v.co)
            if a is not None and b is not None and (b - a).length > limit:
                out.append(v)
        return out

    def _release_weld(self, context, verts, keep):
        """Auto Merge off: a Build drag release acts as a click at that point. Each end vertex welds to an
        existing vertex within the click's hover radius (HOVER_PX x UI scale), as a click there would reuse it.
        Never a pinned vertex; never onto keep (the stroke's start and its own vertices) or a vertex of a face or
        edge the end vertex is part of (the face being made never collapses); a mirror-plane vertex stays on the
        plane. Returns the weld map."""
        st = self.state
        verts = [v for v in verts if v.is_valid and not pinned(st.bm, v)]
        if not verts:
            return {}
        mates = {u for v in verts for f in v.link_faces for u in f.verts}
        mates.update(u for v in verts for e in v.link_edges for u in e.verts)
        return merge_moved(context, st.bm, verts, HOVER_PX * prefs.ui_scale(context),
                           keep=[v for v in list(keep) + list(mates) if v.is_valid], force=True,
                           clip=st.clip) or {}

    def _line_anchor(self):
        """Only the ordinary one-vertex curve stroke extension owns segment input."""
        st = self.state
        if (self.plan.action == VERT_EDGE and st.logic.selected is not None
                and len(st.logic.selected[bmesh.types.BMVert]) == 1):
            return st.logic.bmv
        return None

    def _line_points(self, context, vertex, end):
        st = self.state
        start = st.mw @ vertex.co
        a, b = raycast.to_screen(context, start), raycast.to_screen(context, end)
        if a is None or b is None:
            return []
        view, sources = raycast.View(context), raycast.prepare(st.view.srcs)
        points = [start]
        for i in range(1, self.line_segments):
            world = raycast.surface_or_plane(context, a.lerp(b, i / self.line_segments),
                                            start, sources, view=view).world
            points.append(st.mw @ st.clip.apply(vertex.co, st.mi @ world))
        points.append(end)
        return points

    def _line_update(self, context):
        self.line_world = []
        vertex = self._line_anchor() if self.line_segments > 1 else None
        if vertex is not None:
            self.line_world = self._line_points(context, vertex, self.state.mw @ self.state.logic.hit)

    def _insert_line(self, context):
        """Stage the subdivided wire in the existing click/drag transaction."""
        st = self.state
        if _collapsed(self.line_world):
            raise ValueError("Stroke segments overlap here")
        anchor = self._line_anchor()
        previous = anchor
        created = []
        target = st.logic.vert_lookup.bmv
        for i, world in enumerate(self.line_world[1:], 1):
            vertex = target if i == self.line_segments and target is not None else st.bm.verts.new(st.mi @ world)
            st.bm.edges.new((previous, vertex))
            created.append(vertex)
            previous = vertex
        st.pending_selection = [previous]
        st.slide_constraints = {}
        for v in st.bm.verts:
            v.select_set(False)
        st.bm.select_history.clear()
        previous.select_set(True)
        st.bm.select_history.add(previous)
        st.bm.select_flush(True)
        st.bm.select_flush(False)
        self.line_placing = anchor
        st.bm.normal_update()
        bmesh.update_edit_mesh(st.view.obj.data, loop_triangles=True, destructive=True)
        return created

    def _stroke_damp(self, context, fine):
        """Viscous Ctrl+move: rotate / scale relative to where Ctrl was pressed.

        The cursor offset is damped (Shift: fine) and split along / across the
        reference direction; both act on a screen-space reference length with a
        floor, so short, nearly closed and zoomed-out strokes stay calm.
        """
        cursor = self.stroke_cursor
        if self.stroke_press is None or fine != getattr(self, "stroke_fine", fine):
            self.stroke_press, self.stroke_base = cursor.copy(), self.stroke_xform   # new Ctrl press or Shift toggled: re-anchor
        self.stroke_fine = fine
        _, ref, diag = self.stroke_curve.reference()
        br, bi = self.stroke_base
        mag = hypot(br, bi)
        u = Vector((br*ref.x - bi*ref.y, bi*ref.x + br*ref.y))
        if u.length < 1e-9:
            return
        u.normalize()
        ui = prefs.ui_scale(context)
        length = max(STROKE_XFORM_FLOOR * ui, mag * ref.length, .5 * mag * diag)
        gain = float(getattr(prefs.get(context), "stroke_xform_sensitivity", STROKE_XFORM_GAIN))
        d = (cursor - self.stroke_press) * (gain * (.1 if fine else 1.))
        scale = max(.05, 1. + d.dot(u) / length)
        turn = (u.x * d.y - u.y * d.x) / length
        dr, di = scale * cos(turn), scale * sin(turn)
        self.stroke_xform = (dr*br - di*bi, dr*bi + di*br)

    def _stroke_update(self, context, reshape=False):
        curve, st = self.stroke_curve, self.state
        depth = self._stroke_depth()
        if reshape:
            curve.prepare(st.scene.build_stroke_corner_angle)
            base = curve.density_segments(st.scene.build_stroke_density)
            curve.set_segments(base + curve.wheel_offset)
        screen = curve.sample() if self.stroke_drawing or self.stroke_cursor is None else curve.transformed_by(self.stroke_xform)
        view, sources = raycast.View(context), raycast.prepare(st.view.srcs)
        strip = self.stroke_edge is not None or self.stroke_chain is not None
        anchor = st.mi @ depth if strip else self.stroke_vertex.co
        self.stroke_world = [depth] + [st.mw @ st.clip.apply(anchor,
            st.mi @ raycast.surface_or_plane(context, p, depth, sources, view=view).world)
            for p in screen[1:]] if screen else []
        self.stroke_rows = []
        self.stroke_folded = False
        self.stroke_overlap = []
        narrowed = 0
        along = False
        if self.stroke_chain is not None and self.stroke_world:
            narrowed, along = self._chain_update(context, view, sources)
        elif self.stroke_edge is not None and self.stroke_world:
            ends = [st.mw @ v.co for v in self.stroke_edge.verts]
            self.stroke_rows = strip_rows(self.stroke_world, ends,
                context.region_data.view_rotation @ Vector((0, 0, 1)))
            # Project both rails, then solve their chord width on the surface.
            # Topo takes precedence when a source boundary makes that impossible.
            for i in range(1, len(self.stroke_rows)):
                center = self.stroke_world[i]
                offset = (self.stroke_rows[i][1] - self.stroke_rows[i][0]) / 2
                width = (ends[1] - ends[0]).length
                factor = 1.0
                for _ in range(6 if sources else 1):
                    row = []
                    for sign in (-1, 1):
                        candidate = center + offset * (factor * sign)
                        p = raycast.to_screen(context, candidate)
                        world = raycast.surface_or_plane(context, p, candidate, sources, view=view).world if p is not None else candidate
                        row.append(st.mw @ st.clip.apply(st.mi @ depth, st.mi @ world))
                    distance = (row[1] - row[0]).length
                    if distance <= width * 1e-9 or abs(distance - width) < width * 1e-5:
                        break
                    factor *= min(2.0, max(.5, width / distance))
                self.stroke_rows[i] = tuple(row)

            def project(candidate):
                p = raycast.to_screen(context, candidate)
                world = raycast.surface_or_plane(context, p, candidate, sources, view=view).world if p is not None else candidate
                return st.mw @ st.clip.apply(st.mi @ depth, st.mi @ world)
            # Tight bends: narrow the inner side instead of refusing the stroke.
            self.stroke_rows, narrowed, self.stroke_folded = relieve_strip(
                self.stroke_rows, self.stroke_world, project if sources else None)
            self.stroke_overlap = self._overlaps(context)
        # the header shows the value only; the keys are in the status bar
        text = f"{iface_('Stroke')}: {curve.segments} {iface_('segments')}"
        # at most one message per stroke, at the cursor (warning label), never repeated reports.
        warning = None
        if self.stroke_folded and along:
            warning = 'A chain edge runs along the stroke; draw away from the chain'
        elif self.stroke_folded:
            warning = 'Strip folds at a bend; reduce width or redraw the curve'
        elif self.stroke_overlap:
            # a self-overlapping strip is only a warning; the crossing parts are
            # highlighted and the strip can still be committed.
            warning = 'Strip overlaps itself (highlighted); you can still add it'
        elif narrowed:
            warning = 'Tight bend: inner side narrowed'
        self.stroke_warning = iface_(warning) if warning else None
        context.area.header_text_set(text)
        context.region.tag_redraw()

    def _overlaps(self, context):
        rows = [[raycast.to_screen(context, p) for p in row] for row in self.stroke_rows]
        return strip_overlaps(rows) if len(rows) > 2 else []

    def _chain_update(self, context, view, sources):
        """Row strip of a K >= 3 chain: rigid profile per row, per-side arc-length width
        on the surface, per-column mirror clip, fold relief and the overlap test.
        Returns (narrowed rows, along) and fills stroke_rows / stroke_folded."""
        st = self.state
        chain = self.stroke_chain
        profile = self.stroke_profile
        normal = context.region_data.view_rotation @ Vector((0, 0, 1))
        rows = chain_rows(self.stroke_world, [st.mw @ v.co for v in chain], profile, normal)
        sides, half = profile["sides"], profile["half"]
        # Columns of each side, ordered from the pivot outward.
        halves = {s: sorted((j for j in range(len(chain)) if profile["sides"][j] == s),
                            key=lambda j: abs(profile["offsets"][j])) for s in (-1, 1)}

        def project(candidate, j):
            p = raycast.to_screen(context, candidate)
            world = raycast.surface_or_plane(context, p, candidate, sources, view=view).world if p is not None else candidate
            return st.mw @ st.clip.apply(chain[j].co, st.mi @ world)

        for i in range(1, len(rows)):
            center = self.stroke_world[i]
            offsets = [q - center for q in rows[i]]
            factor = {-1: 1.0, 0: 1.0, 1: 1.0}
            row = list(rows[i])
            for _ in range(6 if sources else 1):
                row = [project(center + offsets[j] * factor[sides[j]], j) for j in range(len(chain))]
                done = True
                for s, columns in halves.items():
                    if not columns:
                        continue
                    # constant width as arc length, pivot to chain end, per side.
                    length = (row[columns[0]] - center).length + sum(
                        (row[b] - row[a]).length for a, b in zip(columns, columns[1:]))
                    if length <= half * 1e-9 or abs(length - half) < half * 1e-5:
                        continue
                    done = False
                    factor[s] *= min(2.0, max(.5, half / length))
                if done:
                    break
            rows[i] = tuple(row)
        rows, narrowed, folded, along = relieve_chain(rows, self.stroke_world, profile,
                                                      project if sources else None)
        self.stroke_rows = rows
        self.stroke_folded = folded
        self.stroke_overlap = [] if folded else self._overlaps(context)
        return narrowed, along

    def _stroke_depth(self):
        st = self.state
        if self.stroke_chain is not None:
            return self.stroke_profile["pivot"].copy()
        if self.stroke_edge is not None:
            return sum((st.mw @ v.co for v in self.stroke_edge.verts), Vector()) / 2
        return st.mw @ self.stroke_vertex.co

    def _stroke_commit(self, context):
        st = self.state
        if self.stroke_chain is not None:
            self._chain_commit(context)
            return
        if self.stroke_edge is not None:
            self._strip_commit(context)
            return
        # At a collapsed cursor position keep the preview rather than creating
        # coincident vertices. The surface/clamp policy may also collapse points.
        if len(self.stroke_world) < 2 or _collapsed(self.stroke_world):
            return
        self.snapshot = MeshSnapshot(st.view.obj)
        try:
            previous = self.stroke_vertex
            created = []
            for world in self.stroke_world[1:]:
                vertex = st.bm.verts.new(st.mi @ world)
                st.bm.edges.new((previous, vertex))
                created.append(vertex)
                previous = vertex
            st.bm.normal_update()
            bmesh.update_edit_mesh(st.view.obj.data, loop_triangles=True, destructive=True)
            if context.scene.tool_settings.use_mesh_automerge:
                targets = merge_moved(context, st.bm, created,
                                     st.scene.merge_radius * prefs.ui_scale(context),
                                     adaptive_min=3 * prefs.ui_scale(context)) or {}
            else:   # only the stroke's last vertex, as a click at the release point
                targets = self._release_weld(context, created[-1:], [self.stroke_vertex] + created[:-1])
            previous = next((target for source, target in targets.items() if source is previous), previous)
            for seq in (st.bm.faces, st.bm.edges, st.bm.verts):
                for elem in seq:
                    elem.select_set(False)
            st.bm.select_history.clear()
            previous.select_set(True)
            st.bm.select_history.add(previous)
            st.bm.normal_update()
            bmesh.update_edit_mesh(st.view.obj.data, loop_triangles=True, destructive=True)
        except Exception:
            self._rollback()
            raise
        self.snapshot.close()
        self.snapshot = None
        self.stroke_curve = None
        self.stroke_world = []
        self.stroke_drawing = False
        context.area.header_text_set(None)
        self._push_undo()
        st.refresh()
        self._replan(context)

    def _strip_commit(self, context):
        st = self.state
        rows = self.stroke_rows
        if len(rows) < 2:
            return
        # Refuse collapsed/folded cells without editing the mesh. Tight bends were
        # already narrowed; what is left cannot be built, and the header says so
        # once for this stroke (no repeated reports).
        reference = None
        area_epsilon = max((rows[0][1] - rows[0][0]).length_squared * 1e-10, 1e-20)
        for near, far in zip(rows, rows[1:]):
            reference = strip_cell_normal(near, far, reference, area_epsilon)
            if reference is None:
                return
        # BMesh areas are object-local; world-width thresholds cannot be used
        # for the post-merge area check on an unapplied object transform.
        local_area_epsilon = max((self.stroke_edge.verts[1].co -
                                  self.stroke_edge.verts[0].co).length_squared * 1e-10, 1e-30)
        self.snapshot = MeshSnapshot(st.view.obj)
        try:
            previous = list(self.stroke_edge.verts)
            made = []
            for row in rows[1:]:
                far = [st.bm.verts.new(st.mi @ world) for world in row]
                face = st.bm.faces.new((previous[0], previous[1], far[1], far[0]))
                if not made:
                    linked = [loop for loop in self.stroke_edge.link_loops if loop.face != face]
                    if linked:
                        face.material_index = linked[0].face.material_index
                        face.smooth = linked[0].face.smooth
                        if linked[0].vert == previous[0]:
                            face.normal_flip()
                    else:
                        flip_toward_view(context, st.mi, face)
                    reverse = any(loop.edge == self.stroke_edge and loop.vert != previous[0] for loop in face.loops)
                else:
                    face.material_index = made[0].material_index
                    face.smooth = made[0].smooth
                    if reverse:
                        face.normal_flip()
                made.append(face)
                previous = far
            st.bm.normal_update()
            bmesh.update_edit_mesh(st.view.obj.data, loop_triangles=True, destructive=True)
            # Only the far pair can weld; internal strip rows remain intact.
            if context.scene.tool_settings.use_mesh_automerge:
                targets = merge_moved(context, st.bm, previous,
                    st.scene.merge_radius * prefs.ui_scale(context),
                    adaptive_min=3 * prefs.ui_scale(context)) or {}
            else:
                targets = self._release_weld(context, previous,
                                             [v for f in made for v in f.verts if v not in previous])
            previous = [targets.get(v, v) for v in previous]
            if (len(set(previous)) != 2 or any(not f.is_valid or len(f.verts) != 4 or f.calc_area() < local_area_epsilon for f in made)):
                raise ValueError("Merging the far end would collapse the quad strip")
            for seq in (st.bm.faces, st.bm.edges, st.bm.verts):
                for elem in seq:
                    elem.select_set(False)
            st.bm.select_history.clear()
            far_edge = st.bm.edges.get(previous)
            far_edge.select_set(True)
            st.bm.select_history.add(far_edge)
            st.bm.normal_update()
            bmesh.update_edit_mesh(st.view.obj.data, loop_triangles=True, destructive=True)
        except ValueError as exc:
            self._rollback()
            self.stroke_curve = None
            self.stroke_world = []
            self.stroke_rows = []
            self.stroke_edge = None
            self.stroke_drawing = False
            self.stroke_cursor = None
            self.stroke_xform, self.stroke_press = (1.0, 0.0), None
            context.area.header_text_set(None)
            self.report({'WARNING'}, tr_rpt(friendly_error(exc, "This stroke could not be built here")))
            st.refresh()
            self._replan(context)
            return
        except Exception:
            self._rollback()
            raise
        self.snapshot.close()
        self.snapshot = None
        self.stroke_curve = None
        self.stroke_world = []
        self.stroke_rows = []
        self.stroke_edge = None
        self.stroke_drawing = False
        context.area.header_text_set(None)
        self._push_undo()
        st.refresh()
        self._replan(context)

    def _chain_commit(self, context):
        """Commit a K >= 3 chain row strip: (rows-1) x (K-1) quads in one winding, far
        row auto-merged, far chain selected so a held-Ctrl stroke continues from it.
        A self-overlap does not block the commit; folded cells do."""
        st = self.state
        rows, chain = self.stroke_rows, self.stroke_chain
        width = len(chain)
        if len(rows) < 2 or self.stroke_folded or not chain_cells_valid(rows, chain_epsilon(rows[0])):
            return
        # BMesh areas are object-local: the post-merge check uses local chain lengths.
        local_area_epsilon = chain_epsilon([v.co for v in chain])
        edges = [st.bm.edges.get((a, b)) for a, b in zip(chain, chain[1:])]
        linked = [next(iter(e.link_faces), None) if e is not None else None for e in edges]
        fallback = next((f for f in linked if f is not None), None)
        vote = chain_sides_with_faces(chain)
        self.snapshot = MeshSnapshot(st.view.obj)
        try:
            previous = list(chain)
            made = []
            flip = None if vote == "BOTH" or vote == 0 else vote > 0
            for row in rows[1:]:
                far = [st.bm.verts.new(st.mi @ world) for world in row]
                for j in range(width - 1):
                    face = st.bm.faces.new((previous[j], previous[j+1], far[j+1], far[j]))
                    if flip is None:
                        # No existing faces: the first face looks at the viewer and
                        # every other face follows its winding.
                        flip_toward_view(context, st.mi, face)
                        flip = any(loop.vert is previous[1] and loop.edge.other_vert(previous[1]) is previous[0]
                                   for loop in face.loops)
                    elif flip:
                        face.normal_flip()
                    source = linked[j] or fallback
                    if source is not None:
                        face.material_index = source.material_index
                        face.smooth = source.smooth
                    made.append(face)
                previous = far
            st.bm.normal_update()
            bmesh.update_edit_mesh(st.view.obj.data, loop_triangles=True, destructive=True)
            # Only the far row can weld; internal strip rows remain intact.
            if context.scene.tool_settings.use_mesh_automerge:
                targets = merge_moved(context, st.bm, previous,
                    st.scene.merge_radius * prefs.ui_scale(context),
                    adaptive_min=3 * prefs.ui_scale(context)) or {}
            else:
                targets = self._release_weld(context, previous,
                                             [v for f in made for v in f.verts if v not in previous])
            previous = [targets.get(v, v) for v in previous]
            far_edges = [st.bm.edges.get((a, b)) for a, b in zip(previous, previous[1:])]
            if (len(set(previous)) != width or any(e is None for e in far_edges)
                    or any(not f.is_valid or len(f.verts) != 4 or f.calc_area() < local_area_epsilon for f in made)):
                raise ValueError("Merging the far row would collapse the quad strip")
            for seq in (st.bm.faces, st.bm.edges, st.bm.verts):
                for elem in seq:
                    elem.select_set(False)
            st.bm.select_history.clear()
            for edge in far_edges:
                edge.select_set(True)
            st.bm.select_history.add(far_edges[-1])
            st.bm.normal_update()
            bmesh.update_edit_mesh(st.view.obj.data, loop_triangles=True, destructive=True)
        except ValueError as exc:
            self._rollback()
            self._clear_stroke(context)
            self.report({'WARNING'}, tr_rpt(friendly_error(exc, "This stroke could not be built here")))
            st.refresh()
            self._replan(context)
            return
        except Exception:
            self._rollback()
            raise
        self.snapshot.close()
        self.snapshot = None
        self._clear_stroke(context)
        self._push_undo()
        st.refresh()
        self._replan(context)

    def _clear_stroke(self, context):
        self.stroke_curve = None
        self.stroke_world = []
        self.stroke_rows = []
        self.stroke_edge = None
        self.stroke_chain = None
        self.stroke_profile = None
        self.stroke_overlap = []
        self.stroke_drawing = False
        self.stroke_cursor = None
        self.stroke_xform, self.stroke_press = (1.0, 0.0), None
        context.area.header_text_set(None)

    def _stroke_event(self, context, event):
        """Intercept only an eligible near-anchor gesture or an owned stroke."""
        self.ctrl_held = event.ctrl
        if event.type in {"LEFT_CTRL", "RIGHT_CTRL"}:
            self.stroke_press = None   # each Ctrl press starts a new relative drag
            self._stroke_hover_update(context)
            context.region.tag_redraw()
        motion = event.type in {"MOUSEMOVE", "INBETWEEN_MOUSEMOVE"}
        cancel = event.type == "WINDOW_DEACTIVATE" or (event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS")
        if cancel:
            return None  # normal operator cancellation owns all cleanup
        if self.stroke_curve is not None:
            if motion:
                self.mouse = raycast.mouse(event)
                if self.stroke_drawing:
                    if not self.stroke_curve.append(self.mouse + self.stroke_input_offset):
                        return {"RUNNING_MODAL"}
                elif event.ctrl:
                    self.stroke_cursor = self.mouse + self.stroke_input_offset
                    self._stroke_damp(context, event.shift)
                else:
                    self.stroke_press = None
                if self.stroke_drawing or event.ctrl:
                    self._stroke_update(context, reshape=self.stroke_drawing)
            elif event.type == "LEFTMOUSE" and event.value == "RELEASE" and self.stroke_drawing:
                self.mouse = raycast.mouse(event)
                self.stroke_curve.append(self.mouse + self.stroke_input_offset, final=True)
                self._stroke_update(context, reshape=True)
                if (self.stroke_curve.points[-1] - self.stroke_curve.points[0]).length < 2 * prefs.ui_scale(context):
                    self.stroke_curve = None
                    self.stroke_world = []
                    self.stroke_rows = []
                    self.stroke_edge = None
                    self.stroke_chain = None
                    self.stroke_overlap = []
                    context.area.header_text_set(None)
                    self._replan(context)
                self.stroke_drawing = False
                self.stroke_cursor = None
                self.stroke_xform, self.stroke_press = (1.0, 0.0), None
                if self.stroke_curve is not None:
                    self._stroke_update(context)
            elif _wheel_delta(event):
                curve = self.stroke_curve
                old = curve.segments
                curve.set_segments(old + _wheel_delta(event))
                curve.wheel_offset += curve.segments - old
                self._stroke_update(context)
            elif (event.type == "LEFTMOUSE" and event.value == "PRESS" and event.shift and event.ctrl
                  and not self.stroke_drawing and not self.placing):
                return {"PASS_THROUGH"}   # Shift+Ctrl+LMB is the box deselect drag; the stroke stays held
            elif event.type in {"LEFTMOUSE", "RET", "NUMPAD_ENTER"} and event.value == "PRESS" and not self.stroke_drawing:
                if event.type == "LEFTMOUSE":
                    self.mouse = raycast.mouse(event)
                self._stroke_commit(context)
                if self.stroke_curve is None and not event.ctrl:
                    self._end(context)
                    return {"FINISHED"}
            # Ctrl release/re-press keeps the released stroke available to commit.
            return {"RUNNING_MODAL"}
        if self.stroke_pending is not None:
            vertex, anchor, press = self.stroke_pending
            if motion:
                self.mouse = raycast.mouse(event)
                if (self.mouse - press).length >= 4 * prefs.ui_scale(context):
                    self.stroke_pending = None
                    self.stroke_vertex = vertex
                    self.stroke_edge = vertex if isinstance(vertex, bmesh.types.BMEdge) else None
                    self.stroke_chain = None
                    if isinstance(vertex, tuple):   # (chain verts, profile): K >= 3
                        self.stroke_chain, self.stroke_profile = vertex
                    strip = self.stroke_edge is not None or self.stroke_chain is not None
                    self.stroke_input_offset = anchor - press if strip else Vector((0, 0))
                    self.stroke_cursor = None
                    self.stroke_xform, self.stroke_press = (1.0, 0.0), None
                    self.stroke_curve = BuildStroke(anchor, 1.25 * prefs.ui_scale(context))
                    self.stroke_curve.width = self._stroke_screen_width(context)   # retract capture distance
                    if self.stroke_chain is not None:
                        # Bound the preview cost: rows x (K-1) cells.
                        self.stroke_curve.MAX_SEGMENTS = max(1, min(BuildStroke.MAX_SEGMENTS,
                            CELL_BUDGET // (len(self.stroke_chain) - 1)))
                    self.stroke_curve.append(self.mouse + self.stroke_input_offset, final=True)
                    self.stroke_drawing = True
                    self._stroke_update(context, reshape=True)
                return {"RUNNING_MODAL"}
            if event.type == "LEFTMOUSE" and event.value == "RELEASE":
                self.stroke_pending = None
                self.mouse = press
                self._insert(context)
                return None  # finish ordinary click placement on this release
            if event.type in {"LEFT_CTRL", "RIGHT_CTRL"} and event.value == "RELEASE":
                self.stroke_pending = None
                return None
            return {"RUNNING_MODAL"}
        if event.type == "LEFTMOUSE" and event.value == "PRESS" and event.ctrl and not self.placing:
            self.mouse = raycast.mouse(event)
            self._replan(context)
            start = self._stroke_start(context)
            if start is not None:
                vertex, anchor, _radius = start
                self.stroke_pending = (vertex, anchor.copy(), self.mouse.copy())
                self.stroke_hover = None
                self._show_reject(context, None)
                return {"RUNNING_MODAL"}
        return None

    def _stroke_screen_width(self, context):
        """the strip's width on screen (px): the selected edge, or the chain's end-to-end span; 0 when unknown."""
        st = self.state
        ends = None
        if self.stroke_chain is not None and len(self.stroke_chain) >= 2:
            ends = (self.stroke_chain[0], self.stroke_chain[-1])
        elif self.stroke_edge is not None:
            ends = tuple(self.stroke_edge.verts)
        if ends is None:
            return 0.0
        a, b = (raycast.to_screen(context, st.mw @ v.co) for v in ends)
        return float((b - a).length) if a is not None and b is not None else 0.0

    def _stroke_start(self, context):
        st = self.state
        self.stroke_reject = None
        if not self.ctrl_held or self.placing or st.logic.selected is None:   # None: re-read on the next move
            return None
        edges = st.logic.selected[bmesh.types.BMEdge]
        if edges:
            candidates = []
            # only the edges that can pass the 8 px test below (the view's own projection); the ray-cast
            # visibility then runs on these few instead of on every selected edge
            view = st.view._view
            near = st.logic.near_selected_edges(view.persmat, st.view.mw, view.width, view.height,
                                                self.mouse, 8 * prefs.ui_scale(context))
            for edge in sorted(near, key=lambda e: e.index):
                if not all(st.view.visible(v) for v in edge.verts):
                    continue
                a, b = [st.screen(v) for v in edge.verts]
                if a is None or b is None or (b-a).length < 1:
                    continue
                closest = segment_closest(self.mouse, a, b)
                distance = (self.mouse-closest).length
                if distance > 8 * prefs.ui_scale(context):
                    continue
                radius = min(adaptive_radius(st.view, v, 8 * prefs.ui_scale(context),
                    3 * prefs.ui_scale(context), excluded=edge.verts) for v in edge.verts)
                if distance <= radius:
                    candidates.append((distance, edge.index, edge, (a+b)/2, radius))
            if not candidates:
                return None
            _, _, edge, anchor, radius = min(candidates, key=lambda item: item[:2])
            # several separate chains selected -> the chain under the cursor.
            # the connected selected edges + chain_from_edges, kept per selection snapshot and stopped at the first branching
            # vertex (see selected_chain), so a hover near a large selection does not walk all of it on every move
            size, found = st.logic.selected_chain(edge)
            if size == 1:
                return edge, anchor, radius     # a single edge: exactly as before
            chain, reason = found
            if chain is None:
                if reason not in ("CLOSED", "BRANCHED"):   # the built area's selection is not a warning
                    self.stroke_reject = reason
                return None
            if not all(st.view.visible(v) and st.screen(v) is not None for v in chain):
                return None
            if chain_sides_with_faces(chain) == "BOTH":
                self.stroke_reject = "BOTH"
                return None
            normal = context.region_data.view_rotation @ Vector((0, 0, 1))
            profile = chain_profile([st.mw @ v.co for v in chain], normal)
            pivot = raycast.to_screen(context, profile["pivot"]) if profile is not None else None
            if pivot is None:
                return None
            # the stroke starts at the chain's arc-length midpoint.
            return (chain, profile), pivot, radius
        selected = st.logic.selected[bmesh.types.BMVert]
        if len(selected) != 1:
            return None
        vertex = next(iter(selected))
        anchor = st.screen(vertex)
        scale = prefs.ui_scale(context)
        if anchor is None or (self.mouse-anchor).length > 8 * scale:
            return None
        radius = adaptive_radius(st.view, vertex, 8 * scale, 3 * scale)
        if (self.mouse-anchor).length <= radius and st.view.visible(vertex):
            return vertex, anchor, radius
        return None

    def _stroke_hover_update(self, context):
        self.stroke_hover = None
        self.stroke_hover_edge = None
        self.stroke_hover_chain = None
        if self.stroke_curve is None and self.stroke_pending is None:
            start = self._stroke_start(context)
            if start is not None:
                element, anchor, radius = start
                self.stroke_hover = (anchor.copy(), radius)
                self.stroke_hover_edge = element if isinstance(element, bmesh.types.BMEdge) else None
                self.stroke_hover_chain = element[0] if isinstance(element, tuple) else None
            self._show_reject(context, self.stroke_reject if start is None else None)

    def _show_reject(self, context, reason):
        """Say why the hovered chain cannot start a row strip: a warning label at the cursor."""
        if reason is not None:
            text = iface_(CHAIN_REASONS[reason])
            if not self.stroke_reject_shown or getattr(self, "stroke_reject_text", None) != text:
                context.region.tag_redraw()
            self.stroke_reject_text = text
            self.stroke_reject_shown = True
        elif self.stroke_reject_shown:
            self.stroke_reject_text = None
            self.stroke_reject_shown = False
            context.region.tag_redraw()

    def _place(self, context):
        reset_projection_cache()   # the placed vertices move
        verts, origins, start = self.placing
        delta = self.mouse - start
        st = self.state
        if self.line_placing is not None:
            if verts and verts[-1] in (getattr(self, "placing_existing", None) or ()):
                # an existing end vertex moves by the drag (as the one-segment placement), not to the cursor
                xy = raycast.to_screen(context, origins[-1])
                end = st.place(xy + delta, origins[-1], verts) if xy is not None else origins[-1].copy()
            else:
                end = st.place(self.mouse, origins[-1], verts)
            end = st.mw @ st.clip.apply(st.mi @ origins[-1], st.mi @ end)
            points = self._line_points(context, self.line_placing, end)
            for v, world in zip(verts, points[1:]):
                v.co = st.mi @ world
            st.bm.normal_update()
            bmesh.update_edit_mesh(st.view.obj.data, loop_triangles=True, destructive=False)
            context.region.tag_redraw()
            return
        for v, world in zip(verts, origins):
            if not v.is_valid:
                continue
            neighbors = st.slide_constraints.get(v)
            if neighbors and all(n.is_valid for n in neighbors):
                a, b = [raycast.to_screen(context, st.mw @ n.co) for n in neighbors]
                if a is not None and b is not None and (b-a).length_squared > 1e-8:
                    cursor = self.mouse
                    if v in (getattr(self, "placing_existing", None) or ()):   # an existing vertex slides by the drag
                        here = raycast.to_screen(context, world)
                        cursor = here + delta if here is not None else cursor
                    factor = min(1.0, max(0.0, (cursor-a).dot(b-a) / (b-a).length_squared))
                    v.co = neighbors[0].co.lerp(neighbors[1].co, factor)
                    continue
            xy = raycast.to_screen(context, world)
            if xy is None:
                continue
            new_world = st.place(xy + delta, world, verts)
            co = st.mi @ new_world
            v.co = st.clip.apply(st.mi @ world, co)
        if getattr(st, "attributes", None) is not None:
            st.attributes.apply()
        st.bm.normal_update()
        bmesh.update_edit_mesh(st.view.obj.data, loop_triangles=True, destructive=False)
        moved = [(v, st.mi @ world) for v, world in zip(verts, origins)]
        units.show(context, units.offset_text(context, units.mean_offset(st.mw, moved)))
        context.region.tag_redraw()

    def _push_undo(self):
        # Commit, placement and optional weld have already synchronized geometry.
        # Only the final selection changed here; keep that visible without
        # tessellating the whole mesh and invalidating topology again.
        bmesh.update_edit_mesh(self.state.view.obj.data, loop_triangles=False, destructive=False)
        bpy.ops.ed.undo_push(message="Build")

    def preview_segments(self):
        """the segment count the wheel is changing (stroke / strip, or a straight line), else None."""
        if self.stroke_curve is not None:
            return self.stroke_curve.segments
        try:   # the curve stroke line: also at 1 segment, where no multi-point preview is drawn
            if self.line_world or self._line_anchor() is not None:
                return self.line_segments
        except (AttributeError, ReferenceError):
            pass
        return None

    def _draw_segments(self, context):
        # same style and place as the Fill HUD (one line at the cursor's right).
        count = self.preview_segments()
        if count is None or self.mouse is None:
            return
        scale = max(.5, min(3., prefs.ui_scale(context)))
        text = f"{count} {iface_('segment' if count == 1 else 'segments')}"
        xy = (self.mouse.x + 18*scale, self.mouse.y - 6*scale)
        return draw.cursor_label(context, xy, text)   # the shared cursor label (with shadow); its text size

    def _draw_warning(self, context, above=None):
        """The stroke / strip warning or the strip refusal, at the cursor; on line 2 (one line
        height + 4 px lower) when the segment count (text size `above`) is on line 1, left-aligned with it."""
        if self.mouse is None:
            return
        if getattr(self, "stroke_curve", None) is not None:
            text = getattr(self, "stroke_warning", None)
        else:
            text = getattr(self, "stroke_reject_text", None) if getattr(self, "stroke_reject_shown", False) else None
        if not text:
            return
        scale = max(.5, min(3., prefs.ui_scale(context)))
        y = self.mouse.y - 6*scale
        if above:
            y -= above + 4*scale
        draw.cursor_label(context, (self.mouse.x + 18*scale, y), text, kind='warning')

    def _draw(self):
        context = bpy.context
        if context.region is None or context.region.as_pointer() != self.region:
            return
        if safe_mesh.native_modal_running(context):   # no BMesh reads under a native modal
            return
        if self.placing:
            return
        self._follow_view(context)
        try:
            with draw.Blend():
                self._draw_warning(context, self._draw_segments(context))   # stacked under the count
                if self.stroke_curve is not None or self.line_world:
                    if self.stroke_curve is not None and (self.stroke_edge is not None or self.stroke_chain is not None):
                        color = palette.warning(1.) if self.stroke_folded else palette.preview_border(context)
                        scale = prefs.ui_scale(context)
                        rows =[[raycast.to_screen(context, p) for p in row] for row in self.stroke_rows]
                        if rows and all(p is not None for row in rows for p in row):
                            last = len(rows[0]) - 1
                            for side in range(last + 1):
                                # Outer rails thick, inner chain columns thin.
                                draw.polyline([row[side] for row in rows], color,
                                              (2 if side in (0, last) else 1) * scale)
                            for row in rows:
                                draw.polyline(row, color, scale)
                                for p in row:
                                    draw.disc(p, 3 * scale, color, segments=12)
                            # highlight the crossing parts of a self-overlapping strip.
                            warn = palette.warning(1.)
                            for a, b in self.stroke_overlap:
                                pa, pb = rows[a[0]][a[1]], rows[b[0]][b[1]]
                                draw.polyline([pa, pb], warn, 4 * scale)
                                for p in (pa, pb):
                                    draw.disc(p, 5 * scale, warn, segments=12)
                        return
                    points = [raycast.to_screen(context, p) for p in
                              (self.stroke_world if self.stroke_curve is not None else self.line_world)]
                    if points and all(p is not None for p in points):
                        color = palette.preview_border(context)
                        scale = prefs.ui_scale(context)
                        draw.polyline(points, color, 2 * scale)
                        for p in points:
                            draw.disc(p, 3 * scale, color, segments=12)
                        draw.ring(points[0], 6 * scale, color, 1.5 * scale, segments=16)
                else:
                    self.state.draw(context)
                if self.stroke_hover is not None and self.ctrl_held and self.stroke_curve is None and self.stroke_pending is None:
                    point, radius = self.stroke_hover
                    scale = prefs.ui_scale(context)
                    element = self.stroke_hover_edge
                    if element is not None and element.is_valid:
                        ends = [self.state.screen(v) for v in element.verts]
                        if all(p is not None for p in ends):
                            draw.polyline(ends, palette.main(context), 5 * scale)
                    chain = self.stroke_hover_chain
                    if chain is not None and all(v.is_valid for v in chain):
                        points = [self.state.screen(v) for v in chain]
                        if all(p is not None for p in points):
                            draw.polyline(points, palette.main(context), 5 * scale)
                    draw.disc(point, 5 * scale, palette.main(context), segments=24)
                    draw.ring(point, max(6 * scale, radius), palette.preview_border(context),
                              1.5 * scale, segments=24)
        except ReferenceError:
            return


def register():
    bpy.utils.register_class(PPO_OT_polypen)


def unregister():
    bpy.utils.unregister_class(PPO_OT_polypen)
