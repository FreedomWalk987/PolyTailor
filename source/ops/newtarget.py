# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Object Mode: create an empty retopology mesh and jump straight into Build."""

import bpy
from ..i18n import tr_rpt   # reports in the add-on context

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
    bl_label = "New PolyTailor Mesh"
    bl_translation_context = "Poly Toolset"
    bl_description = "Add an empty mesh and start Build: in 3D mode at the 3D Cursor, or in Topo mode on the active object"
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
            self.report({"WARNING"}, tr_rpt("No active object"))
            return {"CANCELLED"}

        name = f"{source.name}_retopo" if source else "Mesh"   # New Mesh: Blender's own default name
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

        # the new mesh is empty (no seed vertex)
        mesh.update()

        bpy.ops.object.mode_set(mode="EDIT")
        # an empty mesh gets an explicit Edit Mode undo step, so the first Ctrl+Z after the first Build vertex
        # returns to the empty mesh instead of undoing the object
        try:
            bpy.ops.ed.undo_push(message="New Mesh" if self.location == "CURSOR" else "Retopology Mesh")
        except RuntimeError:
            pass
        try:
            bpy.ops.wm.tool_set_by_id(name="ppo.polypen")
        except RuntimeError:
            pass  # tool switch needs a 3D View; the mesh is still created
        session.sync()   # first: the first entry in a scene applies the saved defaults (they include 3D / Topo)
        from .. import props
        settings = props.settings(context)
        # Sidebar > Sources is left as it is: with none set, Topo snaps to every visible object (core.raycast)
        settings.pen3d = self.location == "CURSOR"   # New Mesh: 3D mode; Retopology Mesh: Topo mode
        session.sync()
        return {"FINISHED"}


def draw_add_menu(self, context):
    """Shift+A > Mesh."""
    layout = self.layout
    layout.separator()
    layout.operator_context = "EXEC_DEFAULT"
    layout.operator("ppo.new_target", text="New Mesh (3D)", text_ctxt="Poly Toolset",
                    icon_value=icon("polypen")).location = "CURSOR"
    layout.operator("ppo.new_target", text="Retopology Mesh (Topo)", text_ctxt="Poly Toolset",
                    icon="PIVOT_ACTIVE").location = "ACTIVE"


def register():
    bpy.utils.register_class(PPO_OT_new_target)
    bpy.types.VIEW3D_MT_mesh_add.append(draw_add_menu)


def unregister():
    bpy.types.VIEW3D_MT_mesh_add.remove(draw_add_menu)
    bpy.utils.unregister_class(PPO_OT_new_target)
