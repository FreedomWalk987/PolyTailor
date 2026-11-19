# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""LMB drag: move the selection, move a hovered element, or box select.

Where the drag starts decides what happens:
  selected element   -> move the whole selection
  unselected element -> select only it, then move it
  empty space        -> box select (replace)
Shift+drag anywhere is a box select that adds to the selection.
Shift+Ctrl+drag anywhere is a box select that removes from the selection.
"""

import bmesh
import bpy
from mathutils import Vector

from .. import prefs, props
from ..core import axis as axis_utils, pick, raycast, safe_mesh, snap, units
from ..i18n import tr_rpt
from ..core.mesh import merge_moved
from .brush import MirrorClip


def _press_xy(context, event):
    region = context.region
    return Vector((event.mouse_prev_press_x - region.x, event.mouse_prev_press_y - region.y))


def _is_selected(elem):
    if isinstance(elem, bmesh.types.BMFace):
        return elem.select
    if isinstance(elem, bmesh.types.BMEdge):
        return elem.select or all(v.select for v in elem.verts)
    return elem.select


def select_only(view, elem):
    bm = view.bm
    for seq in (bm.verts, bm.edges, bm.faces):
        for e in seq:
            e.select_set(False)
    elem.select_set(True)
    bm.select_history.clear()
    bm.select_history.add(elem)
    bm.select_flush_mode()
    bmesh.update_edit_mesh(view.obj.data, loop_triangles=False, destructive=False)


class PPO_OT_drag(bpy.types.Operator):
    bl_idname = "ppo.drag"
    bl_label = "Drag"
    bl_translation_context = "Poly Toolset"
    bl_description = "Move the selection or the hovered element; box select from empty space"
    bl_options = {"REGISTER", "UNDO"}

    extend: bpy.props.BoolProperty(translation_context="Poly Toolset", name="Extend", options={"SKIP_SAVE", "HIDDEN"},
                                   description="Box select, adding to the selection")
    deselect: bpy.props.BoolProperty(translation_context="Poly Toolset", name="Deselect", options={"SKIP_SAVE", "HIDDEN"},
                                     description="Box select, removing from the selection")

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and context.region_data is not None

    def invoke(self, context, event):
        if getattr(self, "extend", False):
            bpy.ops.view3d.select_box("INVOKE_DEFAULT", mode="ADD")
            return {"CANCELLED"}   # box select pushes its own undo step
        if getattr(self, "deselect", False):
            bpy.ops.view3d.select_box("INVOKE_DEFAULT", mode="SUB")
            return {"CANCELLED"}   # box select pushes its own undo step
        view, elem = pick.element_under(context, _press_xy(context, event))
        if elem is None:
            bpy.ops.view3d.select_box("INVOKE_DEFAULT", mode="SET")
            return {"CANCELLED"}   # box select pushes its own undo step
        if not _is_selected(elem):
            bm_ = view.bm   # Esc gives this selection back
            self._sel_before = tuple([e.index for e in seq if e.select] for seq in (bm_.verts, bm_.edges, bm_.faces))
            self._active_before = bm_.select_history.active
            select_only(view, elem)
        if snap.native_needed(context):
            # Increment / Grid / Face / ... snapping: Blender's own move, confirmed on release
            bmesh.update_edit_mesh(view.obj.data, loop_triangles=False, destructive=False)
            snap.native_translate(context, release_confirm=True)
            return {"CANCELLED"}   # translate pushes its own undo step
        self.view = view
        self.orig = {v: v.co.copy() for v in view.bm.verts if v.select and not v.hide}
        from ..core.mesh import CoordinateJournal
        self.journal = CoordinateJournal(view.bm)   # Esc puts back only what Drag wrote
        for v, co in self.orig.items():
            self.journal.note(v, co)
        self.axis = None
        self.pivot = sum((view.mw @ co for co in self.orig.values()), Vector((0, 0, 0)))/len(self.orig)
        self.start = _press_xy(context, event)
        self.precise = raycast.PreciseMouse(raycast.mouse(event))   # Shift: finer movement
        self.header_area = context.area   # header cleared by the operator teardown (ops.common)
        self.clip = MirrorClip(view.obj)
        from ..core.face_attributes import Correction
        self.attributes = Correction(view.bm, context)
        self._move(context, event)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _move(self, context, event):
        mouse = self.precise.update(event)
        delta = mouse - self.start
        mi = self.view.mw.inverted()
        active = self.view.bm.select_history.active
        snapped = snap.for_move(context, mouse, delta, self.view.mw, self.orig, self.view.srcs,
                                active if isinstance(active, bmesh.types.BMVert) else None)
        exact = {}
        if snapped is not None:
            delta, exact, _offset = snapped
        axis_delta = None
        if self.axis:
            direction = axis_utils.direction(self.axis)
            amount = (_offset.dot(direction) if snapped is not None else
                      axis_utils.distance(context, self.start, mouse, self.pivot, self.axis))
            axis_delta = direction*amount
        view = raycast.View(context)   # one matrix inversion per move, not per vertex
        for v, original in self.orig.items():
            if axis_delta is not None:
                world = self.view.mw @ original + axis_delta
                hit = raycast.nearest_on_sources(context, world, self.view.srcs)
                v.co = self.clip.apply(original, mi @ (hit.world if hit else world))
                continue
            if v in exact:
                v.co = self.clip.apply(original, mi @ exact[v])
                continue
            world = self.view.mw @ original
            xy = view.to_screen(world)
            if xy is None:
                continue
            # depth reference: where the point is now (follows gradual depth changes)
            hit = raycast.surface_or_plane(context, xy + delta, self.view.mw @ v.co, self.view.srcs, view)
            v.co = self.clip.apply(original, mi @ hit.world)
        if snapped is None:
            snap.finish()   # no marker for plain surface placement
        self.journal.wrote(self.orig)
        self.attributes.apply(self.orig)
        self.view.bm.normal_update()
        bmesh.update_edit_mesh(self.view.obj.data, destructive=False)
        units.show(context, units.offset_text(context, units.mean_offset(self.view.mw, self.orig.items())))
        context.area.tag_redraw()

    def _restore(self):
        """each vertex Drag moved goes back only while it still holds what Drag wrote (an outside edit
        of it is kept). Returns the number left alone."""
        skipped = self.journal.restore()
        sel = getattr(self, '_sel_before', None)
        if sel is not None:   # the selection the drag changed
            bm_ = self.view.bm
            for seq, keep in reversed(list(zip((bm_.verts, bm_.edges, bm_.faces), sel))):   # faces, edges, verts
                seq.ensure_lookup_table()
                keep = set(keep)
                for e in seq:
                    e.select = e.index in keep
            bm_.select_history.clear()
            act = getattr(self, '_active_before', None)
            if act is not None and act.is_valid:
                bm_.select_history.add(act)
        self.attributes.restore()
        self.view.bm.normal_update()
        bmesh.update_edit_mesh(self.view.obj.data, destructive=False)
        return skipped

    def _end(self, context):
        snap.finish()
        if context.area is not None:
            context.area.header_text_set(None)

    def cancel(self, context):
        self._restore()
        self._end(context)

    def modal(self, context, event):
        if self.journal.stale():   # a vertex Drag wrote changed under it: stop, no rollback
            self._end(context)
            self.report({"WARNING"}, tr_rpt("The mesh being edited changed; the tool stopped"))
            return {"CANCELLED"}
        if event.type in {"X", "Y", "Z"} and event.value == "PRESS":
            self.axis = None if self.axis == event.type else event.type
            self._move(context, event)
        elif event.type in {"MOUSEMOVE", "INBETWEEN_MOUSEMOVE"}:
            self._move(context, event)
        elif event.type == "LEFTMOUSE" and event.value == "RELEASE":
            merge_moved(context, self.view.bm, self.orig,
                        props.settings(context).merge_radius * prefs.ui_scale(context))
            self._end(context)
            return {"FINISHED"}
        elif event.type == "WINDOW_DEACTIVATE" or (event.value == "PRESS" and event.type in {
                "ESC", "RIGHTMOUSE"}):
            # stale handles never scramble outside edits; per-vertex check in _restore
            if not safe_mesh.restore_allowed(self, context) or self._restore():
                self.report({"WARNING"}, tr_rpt("The mesh being edited changed; the tool stopped"))
            self._end(context)
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}


def register():
    bpy.utils.register_class(PPO_OT_drag)


def unregister():
    bpy.utils.unregister_class(PPO_OT_drag)
