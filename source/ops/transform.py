# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Keyboard surface transforms and live edge extrusion."""

import bmesh
import bpy
from ..i18n import tr as iface_   # add-on context first
from mathutils import Vector

from .. import props, prefs
from ..core import axis as axis_utils, raycast, safe_mesh, snap, units
from ..core.mesh import MeshSnapshot, merge_moved
from ..core.proportional import Influence, FALLOFFS
from ..core.extrude import compute_boundary_offsets, find_extrude_pairs, ExtrusionLadder
from .brush import MirrorClip, pixels_to_world
from .common import timer_tick


def _finite(bm):
    """every vertex coordinate is finite (one NaN / Inf vertex makes the surface move degenerate)."""
    import numpy as np
    from itertools import chain
    n = len(bm.verts)
    return bool(np.isfinite(np.fromiter(chain.from_iterable(v.co for v in bm.verts), dtype=np.float64,
                                        count=3 * n)).all())


class PPO_OT_move(bpy.types.Operator):
    bl_idname = "ppo.move"
    bl_label = "Move on Surface"
    bl_translation_context = "Poly Toolset"
    bl_description = "Move selection on the surface; G again slides; E again offsets an extrusion"
    bl_options = {"REGISTER", "UNDO"}
    extrude: bpy.props.BoolProperty(default=False, options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and context.region_data is not None

    def invoke(self, context, event):
        self.header_area = context.area
        self.obj = context.edit_object
        self.bm = bmesh.from_edit_mesh(self.obj.data)
        if not any(v.select and not v.hide for v in self.bm.verts):
            return {"CANCELLED"}
        if self.extrude and any(f.select for f in self.bm.faces):
            bpy.ops.view3d.edit_mesh_extrude_move_normal("INVOKE_DEFAULT")
            return {"CANCELLED"}
        native = snap.native_needed(context) or not _finite(self.bm)   # NaN / Inf: Blender's move
        if native and not self.extrude:
            # Increment / Grid / Face / ... snapping: Blender's own move
            snap.native_translate(context)
            return {"CANCELLED"}
        self.snapshot = MeshSnapshot(self.obj)
        if self.extrude:
            edges = [e for e in self.bm.edges if e.select and not e.hide]
            vertices = [v for v in self.bm.verts if v.select and not v.hide]
            result = (bmesh.ops.extrude_edge_only(self.bm, edges=edges) if edges else
                      bmesh.ops.extrude_vert_indiv(self.bm, verts=vertices))
            for seq in (self.bm.faces, self.bm.edges, self.bm.verts):
                for elem in seq:
                    elem.select_set(False)
            if edges:
                geom = result.get("geom", [])
            else:
                # Vertex extrusion returns the new vertices AND the edges joining them to
                # the originals; selecting those edges also selected the original vertex,
                # so E moved the original point instead of the new one. Select only the
                # new vertices.
                geom = list(result.get("verts", []))
            for elem in geom:
                if isinstance(elem, (bmesh.types.BMVert, bmesh.types.BMEdge)):
                    elem.select_set(True)
            self.bm.select_flush_mode()
            if native:
                bmesh.update_edit_mesh(self.obj.data)
                self.snapshot.close()
                self.snapshot = None
                try:   # the extrusion is its own undo step, even if the move is cancelled
                    bpy.ops.ed.undo_push(message="Extrude")
                except RuntimeError:
                    pass
                snap.native_translate(context)
                return {"CANCELLED"}   # one undo step: extrusion + move, pushed by translate
        self.mw = self.obj.matrix_world.copy()
        self.mi = self.mw.inverted()
        self.orig = {v: v.co.copy() for v in self.bm.verts if v.select and not v.hide}
        from ..core.mesh import CoordinateJournal
        self.journal = CoordinateJournal(self.bm)   # Esc puts back only what Move wrote
        for v, co in self.orig.items():
            self.journal.note(v, co)
        from ..core.face_attributes import Correction
        self.attributes = None if self.extrude else Correction(self.bm, context)
        self.clip = MirrorClip(self.obj)
        self.srcs = raycast.sources(context)
        self.start = raycast.mouse(event)
        self.mouse = self.start.copy()
        self.precise = raycast.PreciseMouse(self.start)
        self.axis = None
        self.offset_data = None
        self.ladder = ExtrusionLadder(self.bm, find_extrude_pairs(self.bm)) if self.extrude else None
        self.base_center = sum((self.mw @ co for co in self.orig.values()), Vector((0, 0, 0))) / len(self.orig)
        self.center_xy = raycast.to_screen(context, self.base_center)
        self.world_per_pixel = pixels_to_world(context, self.base_center, 1)
        ts = getattr(context.scene, "tool_settings", None)
        self.proportional = not self.extrude and bool(getattr(ts, "use_proportional_edit", False))
        self.proportional_radius = max(1e-5, min(5000.0, getattr(ts, "proportional_size", 1.0)))
        self.influence = None
        self.affected = {}
        self.proportional_guide = ()
        if self.proportional:
            self._influence_refresh(context)
        self._update(context)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _influence_refresh(self, context):
        ts = context.scene.tool_settings
        if self.influence is None:
            self.influence = Influence(self.bm, self.mw, self.orig, raycast.view_forward(context),
                                       ts.use_proportional_connected, ts.use_proportional_projected)
        self.influence.refresh(self.proportional_radius, ts.proportional_edit_falloff)
        self.proportional_exclude = set(self.orig) | set(self.influence.weights)

    def _update(self, context):
        # Always evaluate from the same coordinates. Shrinking the radius or
        # disabling O must put vertices leaving the influence back exactly.
        restored_vertices = set(self.affected)
        for v, co in self.affected.items():
            v.co = co.copy()
        self.affected = {}
        delta = self.mouse - self.start
        if self.offset_data:
            _, bases, offsets, _, _, _ = self.offset_data
            width = self._offset_width()
            for v, base in bases.items():
                world = base + offsets[v] * width
                hit = raycast.nearest_on_sources(context, world, self.srcs)
                v.co = self.clip.apply(self.orig[v], self.mi @ (hit.world if hit else world))
        else:
            # Vertex / Edge / Edge Center snap targets: PolyTailor's own snapping (see core.snap)
            active = self.bm.select_history.active
            snapped = snap.for_move(context, self.mouse, delta, self.mw, self.orig, self.srcs,
                                    active if isinstance(active, bmesh.types.BMVert) else None,
                                    exclude=self.proportional_exclude if self.proportional else None)
            exact = {}
            if snapped is not None and self.axis is None:
                delta, exact, _offset = snapped
            view = raycast.View(context)   # one matrix inversion per move, not per vertex
            axis_delta = None
            if self.axis is not None:
                axis = axis_utils.direction(self.axis)
                amount = (snapped[2].dot(axis) if snapped is not None else
                          axis_utils.distance(context, self.start, self.mouse, self.base_center, self.axis))
                axis_delta = axis*amount
            for v, co in self.orig.items():
                world = self.mw @ co
                if v in exact:
                    v.co = self.clip.apply(co, self.mi @ exact[v])
                    continue
                xy = view.to_screen(world)
                if xy is None:
                    continue
                if self.axis is None:
                    # depth reference: where the point is now (follows gradual depth changes)
                    hit = raycast.surface_or_plane(context, xy + delta, self.mw @ v.co, self.srcs, view)
                else:
                    point = world + axis_delta
                    hit = raycast.nearest_on_sources(context, point, self.srcs)
                    if hit is None:
                        hit = raycast.Hit(point)
                v.co = self.clip.apply(co, self.mi @ hit.world)
            if self.proportional:
                for v, weight in self.influence.weights.items():
                    if weight <= 0.0:
                        continue
                    co = self.influence.originals[v]
                    world = self.mw @ co
                    if self.axis is None:
                        xy = view.to_screen(world)
                        if xy is None:
                            continue
                        hit = raycast.surface_or_plane(context, xy+delta*weight,
                                                       self.mw @ v.co, self.srcs, view)
                    else:
                        point = world+axis_delta*weight
                        hit = raycast.nearest_on_sources(context, point, self.srcs)
                        if hit is None:
                            hit = raycast.Hit(point)
                    v.co = self.clip.apply(co, self.mi @ hit.world)
                    self.affected[v] = co
                    self.journal.note(v, co)
            if snapped is None:
                snap.finish()   # no marker for plain surface placement
        self.journal.wrote(set(self.orig) | set(self.affected) | restored_vertices)
        from ..core.face_attributes import apply
        apply(self, set(self.orig) | set(self.affected) | restored_vertices)
        self.proportional_guide = ()
        if self.proportional:
            center = sum((self.mw @ v.co for v in self.orig), Vector((0, 0, 0))) / len(self.orig)
            self.proportional_guide = (tuple(center), self.proportional_radius)
        if self.ladder:
            self.ladder.update_positions(context, self.mw)
        self.bm.normal_update()
        bmesh.update_edit_mesh(self.obj.data, destructive=self.extrude)
        if self.offset_data:
            text = f"{iface_('Offset')}: {units.length(context, width)}"
        else:
            text = units.offset_text(context, units.mean_offset(self.mw, self.orig.items()))
            if self.axis:
                text += f"   {iface_('along global')} {self.axis}"
            if self.proportional:
                text += f"   {iface_('Proportional size')}: {units.length(context, self.proportional_radius)}"
        units.show(context, text)
        context.area.tag_redraw()

    def _offset_begin(self, context):
        """EE: remember where the offset starts and how the mouse maps to its width.

        The width is signed: positive along the boundary's outward direction, negative
        against it (both sides of the normal). An open boundary (one side) reads the
        mouse along its outward direction on screen; a closed loop, whose outward
        directions point all around, reads the distance from its centre (pulling toward
        the centre goes inward)."""
        self.axis = None
        _, bases, offsets, _, _, _ = self.offset_data
        self.offset_origin = self.mouse.copy()
        self.offset_axis = None
        if offsets:
            mean = sum(offsets.values(), Vector()) / len(offsets)
            if mean.length > 0.3:        # mostly one way: an open boundary
                tip = raycast.to_screen(context, self.base_center + mean.normalized() * self.world_per_pixel * 50)
                if tip is not None and self.center_xy is not None and (tip - self.center_xy).length > 1e-3:
                    self.offset_axis = (tip - self.center_xy).normalized()
        self.offset_start = (self.mouse - self.center_xy).length if self.center_xy is not None else 0.0

    def _offset_width(self):
        if self.offset_axis is not None:
            pixels = (self.mouse - self.offset_origin).dot(self.offset_axis)
        else:
            pixels = (self.mouse - self.center_xy).length - self.offset_start
        return pixels * self.world_per_pixel

    def _end(self, context, restore=False):
        """Returns False when a requested restore was refused (outside edits kept)."""
        snap.finish()
        kept = False
        if restore:
            # never revert outside coordinate / UV / shape-key edits
            if not safe_mesh.restore_allowed(self, context):
                kept = True
            elif not self.extrude:
                kept = self._local_restore() != 0   # never a full-snapshot rollback for Move
            elif self._outside_edit():
                kept = True   # an outside edit the sampled fingerprint missed is kept
            else:
                kept = self.snapshot.restore() is False
        self.snapshot.close()
        context.area.header_text_set(None)
        context.area.tag_redraw()
        if kept:
            self.report({"WARNING"}, iface_("The mesh being edited changed; the tool stopped"))
        return not kept

    def _outside_edit(self):
        """ONE O(n) comparison before the full restore: every vertex Move did not move or create
        (the extrusion and its ladder cuts) must still be where the snapshot has it; Move without an extrusion also
        keeps the edge / face counts. A rollback inside our own event (an error there) is not checked."""
        if getattr(self, "_ppo_internal_rollback", False):
            return False
        touched = set(getattr(self, "orig", {})) | set(getattr(self, "affected", {}))
        ladder = getattr(self, "ladder", None)
        if ladder:
            touched.update(m for mids in ladder.mids for m in mids)
        return self.snapshot.outside_changed(touched, topology=not self.extrude)

    def _local_restore(self):
        """put back only the vertices Move moved and their face corners. each vertex only while
        it still holds what Move last wrote (CoordinateJournal; an outside edit of it is kept), also after a topology
        change elsewhere. Returns the number left alone (-1: failed, nothing more written)."""
        try:
            skipped = self.journal.restore()
            if self.attributes is not None:   # refuses by itself if faces were rebuilt outside
                self.attributes.restore([v for v in self.journal.rows if v.is_valid])
            self.bm.normal_update()
            bmesh.update_edit_mesh(self.obj.data)
            return skipped
        except (ReferenceError, AttributeError, RuntimeError):
            return -1

    def cancel(self, context):
        self._end(context, restore=True)

    def modal(self, context, event):
        if self.journal.stale():   # a vertex Move wrote changed under it: stop, no rollback
            self._end(context)
            self.report({"WARNING"}, iface_("The mesh being edited changed; the tool stopped"))
            return {"CANCELLED"}
        if event.type in {"MOUSEMOVE", "INBETWEEN_MOUSEMOVE"}:
            self.mouse = self.precise.update(event)   # Shift: finer movement
            self._update(context)
        elif event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"} and event.value == "PRESS" and self.ladder:
            self.ladder.set_cuts(self.ladder.cuts + (1 if event.type == "WHEELUPMOUSE" else -1))
            self._update(context)
        elif event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"} and event.value == "PRESS" and self.proportional:
            ts = context.scene.tool_settings
            radius = self.proportional_radius*(1/1.1 if event.type == "WHEELUPMOUSE" else 1.1)
            ts.proportional_size = max(1e-5, min(5000.0, radius))
            self.proportional_radius = ts.proportional_size
            self._influence_refresh(context)
            self._update(context)
        elif event.type == "WINDOW_DEACTIVATE" or (event.type in {"RIGHTMOUSE", "ESC"} and event.value == "PRESS"):
            self._end(context, restore=True)
            return {"CANCELLED"}
        elif event.value == "PRESS":
            if event.type in {"LEFTMOUSE", "RET", "NUMPAD_ENTER", "SPACE"}:
                merge_moved(context, self.bm, set(self.orig) | set(self.affected),
                            props.settings(context).merge_radius * prefs.ui_scale(context))
                self._end(context)
                return {"FINISHED"}
            if event.type in {"X", "Y", "Z"} and not self.offset_data:
                self.axis = None if self.axis == event.type else event.type
                self._update(context)
            elif event.type == "O" and not self.extrude:
                ts = context.scene.tool_settings
                if getattr(event, "shift", False):
                    modes = FALLOFFS
                    ts.proportional_edit_falloff = modes[(modes.index(ts.proportional_edit_falloff)+1) % len(modes)]
                else:
                    self.proportional = not self.proportional
                    ts.use_proportional_edit = self.proportional
                if self.proportional:
                    self._influence_refresh(context)
                self._update(context)
            elif event.type == "G" and not self.extrude:
                if self._end(context, restore=True):
                    bpy.ops.ppo.slide("INVOKE_DEFAULT")
                return {"CANCELLED"}
            elif event.type == "E" and self.extrude:
                for v, co in self.orig.items():
                    v.co = co
                if self.ladder:
                    self.ladder.set_cuts(0)
                self.offset_data = compute_boundary_offsets(self.bm, self.mw, raycast.view_forward(context))
                if self.offset_data:
                    self._offset_begin(context)
                    self._update(context)
        return {"RUNNING_MODAL"}


def _slide_topology(bm):
    """Track connectivity independently of the selection-only Relax cache."""
    return (tuple(tuple(v.index for v in edge.verts) for edge in bm.edges),
            tuple(tuple(v.index for v in face.verts) for face in bm.faces))


class PPO_OT_slide(bpy.types.Operator):
    bl_idname = "ppo.slide"
    bl_label = "Slide on Surface"
    bl_translation_context = "Poly Toolset"
    bl_description = "Slide the selection along its edges"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and context.region_data is not None

    def invoke(self, context, event):
        self.obj = context.edit_object
        self.snapshot = MeshSnapshot(self.obj)
        self.cancelled = False
        self.completion_requested = False
        bm = bmesh.from_edit_mesh(self.obj.data)
        for elements in (bm.verts, bm.edges, bm.faces):
            elements.ensure_lookup_table()
            elements.index_update()
        self.edit_bm = bm
        self.topology = _slide_topology(bm)
        self.vertex_count = len(bm.verts)
        self.stamp = safe_mesh.mesh_stamp(bm)   # checked before a cancel restore
        try:
            if any(e.select for e in bm.edges):
                result = bpy.ops.transform.edge_slide("INVOKE_DEFAULT", False, correct_uv=context.scene.tool_settings.use_transform_correct_face_attributes)
            else:
                result = bpy.ops.transform.vert_slide("INVOKE_DEFAULT", False, correct_uv=context.scene.tool_settings.use_transform_correct_face_attributes)
        except RuntimeError:
            self.snapshot.close()
            return {"CANCELLED"}
        if "RUNNING_MODAL" not in result:
            self.snapshot.close()
            return {"CANCELLED"}
        self.timer = context.window_manager.event_timer_add(0.01, window=context.window)
        self.timer_duration = self.timer.time_duration
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        timer = getattr(self, "timer", None)
        if timer is not None:
            self.timer = None
            context.window_manager.event_timer_remove(timer)
        snapshot = getattr(self, "snapshot", None)
        if snapshot is not None:
            # During native slide ownership, do not rebuild its BMesh.
            active = any(op.bl_idname in {"TRANSFORM_OT_edge_slide", "TRANSFORM_OT_vert_slide"}
                         for op in context.window.modal_operators)
            if not active and not safe_mesh.foreign_modal_running(context, self):
                bm = bmesh.from_edit_mesh(self.obj.data)
                if bm is not self.edit_bm:
                    return
                for elements in (bm.verts, bm.edges, bm.faces):
                    elements.ensure_lookup_table()
                    elements.index_update()
                if len(bm.verts) != self.vertex_count or _slide_topology(bm) != self.topology:
                    return
                if (safe_mesh.mesh_stamp(bm) != self.stamp   # + O(n) check
                        or (not getattr(self, "_ppo_internal_rollback", False) and snapshot.outside_changed())
                        or snapshot.restore() is False):
                    self.report({"WARNING"}, iface_("The mesh being edited changed; the tool stopped"))

    @staticmethod
    def _native_running(context):
        return any(op.bl_idname in {"TRANSFORM_OT_edge_slide", "TRANSFORM_OT_vert_slide"}
                   for op in context.window.modal_operators)

    def modal(self, context, event):
        if event.type != "TIMER":
            if self._native_running(context):
                # the native slide owns the input: every event passes through, never swallowed (also not after a
                # focus loss: WINDOW_DEACTIVATE is no cancel, the native slide decides). Only a confirm / cancel key
                # PRESS is noted; the last one wins.
                if event.value == "PRESS":
                    if event.type in {"ESC", "RIGHTMOUSE"}:
                        self.cancelled = True
                        self.completion_requested = True
                    elif event.type in {"LEFTMOUSE", "RET", "NUMPAD_ENTER", "SPACE"}:
                        self.cancelled = False
                        self.completion_requested = True
                return {"PASS_THROUGH"}
            if self.completion_requested and not safe_mesh.foreign_modal_running(context, self):
                return {"RUNNING_MODAL"}   # the slide ended: held until the next TIMER finishes it (one tick)
            return {"PASS_THROUGH"}
        if event.type == "TIMER":
            if not timer_tick(self):
                return {"PASS_THROUGH"}
            if self._native_running(context):
                return {"PASS_THROUGH"}
            if safe_mesh.foreign_modal_running(context, self):
                return {"PASS_THROUGH"}
            # the native slide ended: its outcome is read from the mesh (a key we noted is not always what ended
            # it; also none after a focus loss). A mesh still equal to the invoke snapshot was cancelled, any change
            # was confirmed. The noted key only breaks the tie of an unchanged mesh (a confirm without movement).
            if self.snapshot.outside_changed():
                self.cancelled = False
            elif not self.completion_requested:
                self.cancelled = True
            timer = self.timer
            self.timer = None
            context.window_manager.event_timer_remove(timer)
            bm = bmesh.from_edit_mesh(self.obj.data)
            if bm is not self.edit_bm:
                raise RuntimeError("Edit mesh was replaced during slide")
            for elements in (bm.verts, bm.edges, bm.faces):
                elements.ensure_lookup_table()
                elements.index_update()
            if len(bm.verts) != self.vertex_count or _slide_topology(bm) != self.topology:
                # Auto Merge (on in Build) welded the slid vertex onto a neighbour when
                # the slide was confirmed. That is a valid result, not an error: keep it
                # as is and skip the surface projection, whose vertex mapping no longer holds.
                self.snapshot.close()
                return {"CANCELLED"} if self.cancelled else {"FINISHED"}
            if self.cancelled:
                # the cancelled native slide put the mesh back itself (no restore: nothing to put back); a
                # difference from the invoke stamp is not ours to revert
                if safe_mesh.mesh_stamp(bm) != self.stamp or self.snapshot.outside_changed():
                    self.report({"WARNING"}, iface_("The mesh being edited changed; the tool stopped"))
            else:
                bm = bmesh.from_edit_mesh(self.obj.data)
                # The nested native slide has undo disabled. The outer PolyTailor
                # operator records the final projected result on completion.
                mw = self.obj.matrix_world
                mi = mw.inverted()
                srcs = raycast.sources(context)
                from ..core.face_attributes import Correction
                self.attributes = Correction(bm, context)
                moved = []
                for v in bm.verts:
                    if v.select and not v.hide:
                        hit = raycast.nearest_on_sources(context, mw @ v.co, srcs)
                        if hit:
                            v.co = mi @ hit.world
                            moved.append(v)
                self.attributes.apply(moved)
                self.attributes.close()
                bm.normal_update()
                bmesh.update_edit_mesh(self.obj.data, destructive=False)
            self.snapshot.close()
            return {"CANCELLED"} if self.cancelled else {"FINISHED"}
        return {"PASS_THROUGH"}


_classes = (PPO_OT_move, PPO_OT_slide)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
