# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Object Mode: create an empty retopology mesh and jump straight into Build."""

import bpy
from ..i18n import tr   # reports in the add-on context

from .. import session
from ..ui.pie import icon


def _target_collection(context):
    """Active collection when it is usable, otherwise the scene's root collection (also for a linked or
    library-override collection, which cannot take a new object)."""
    layer_coll = context.view_layer.active_layer_collection
    coll = layer_coll.collection if layer_coll else None
    if (coll is None or layer_coll.exclude or layer_coll.hide_viewport or coll.hide_viewport
            or coll.library is not None or coll.override_library is not None):
        coll = context.scene.collection
    return coll


class PPO_OT_new_target(bpy.types.Operator):
    bl_idname = "ppo.new_target"
    bl_label = "New Retopology Mesh"
    bl_translation_context = "Poly Toolset"
    bl_description = "Create an empty mesh object for retopology, enter Edit Mode and start Build"
    bl_options = {"REGISTER"}   # one explicit undo step (pushed in execute); no redo panel

    location: bpy.props.EnumProperty(translation_context="Poly Toolset", 
        name="Location",
        items=(
            ("CURSOR", "At 3D Cursor", "Place the new object at the 3D Cursor"),
            ("ACTIVE", "From Active Object", "Copy the active object's location, rotation and scale"),
        ),
        default="CURSOR",
    )

    @classmethod
    def poll(cls, context):
        return (context.mode == "OBJECT" and context.area is not None
                and context.area.type == "VIEW_3D")

    def execute(self, context):
        source = context.view_layer.objects.active if self.location == "ACTIVE" else None
        if self.location == "ACTIVE" and source is None:
            self.report({"WARNING"}, tr("No active object"))
            return {"CANCELLED"}

        name = f"{source.name}_retopo" if source else "Retopology"
        target = _target_collection(context)   # validated before any data is created (no orphans)
        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        try:
            target.objects.link(obj)
        except RuntimeError:
            bpy.data.objects.remove(obj)
            bpy.data.meshes.remove(mesh)
            raise
        if source:
            obj.matrix_world = source.matrix_world.copy()
        else:
            obj.location = context.scene.cursor.location.copy()

        for other in context.view_layer.objects:
            if other.select_get():
                other.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj

        # the new mesh is empty; no seed vertex (2.2.12-2.6.3 added one at the 3D Cursor)
        mesh.update()

        bpy.ops.object.mode_set(mode="EDIT")
        # an empty mesh gets an explicit Edit Mode undo step, so the first Ctrl+Z after the first Build vertex
        # returns to the empty mesh (2.6.3 got this from its seed vertex) instead of undoing the object
        try:
            bpy.ops.ed.undo_push(message="New Retopology Mesh")
        except RuntimeError:
            pass
        try:
            bpy.ops.wm.tool_set_by_id(name="ppo.polypen")
        except RuntimeError:
            pass  # tool switch needs a 3D View; the mesh is still created
        session.sync()
        return {"FINISHED"}


def draw_add_menu(self, context):
    """Shift+A > Mesh."""
    layout = self.layout
    layout.separator()
    layout.operator_context = "EXEC_DEFAULT"
    layout.operator("ppo.new_target", text="Retopology at Cursor", text_ctxt="Poly Toolset",
                    icon_value=icon("polypen")).location = "CURSOR"
    layout.operator("ppo.new_target", text="Retopology from Active", text_ctxt="Poly Toolset",
                    icon="PIVOT_ACTIVE").location = "ACTIVE"


def register():
    bpy.utils.register_class(PPO_OT_new_target)
    bpy.types.VIEW3D_MT_mesh_add.append(draw_add_menu)


def unregister():
    bpy.types.VIEW3D_MT_mesh_add.remove(draw_add_menu)
    bpy.utils.unregister_class(PPO_OT_new_target)
