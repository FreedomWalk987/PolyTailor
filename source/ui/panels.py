# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Native sidebar and popovers shared by the toolbar and pie."""

import bpy
from ..i18n import tr as iface_   # add-on context first

from .. import defaults, props
from ..core import raycast
from ..ops.common import active_tool_id
from .pie import icon, draw_modes, draw_build_settings, draw_masking, draw_pin_glue
from ..i18n import CONTEXT


def draw_general(context, layout):
    # only what the header lacks. Face attributes, merge radius, Stop at Edge, sources and
    # snap filters live in the N panel (Snapping / Sources).
    layout.use_property_split = True
    if context.space_data.type == "VIEW_3D":
        overlay = context.space_data.overlay
        layout.prop(overlay, "show_retopology")
        layout.prop(overlay, "retopology_offset")
        layout.prop(overlay, "show_wireframes")
        layout.prop(overlay, "show_fade_inactive")
        layout.prop(overlay, "fade_inactive_alpha")
    layout.separator()
    for mod in context.edit_object.modifiers:
        if mod.type == "MIRROR":
            box = layout.box()
            box.label(text=mod.name, icon="MOD_MIRROR")
            box.row(align=True).prop(mod, "use_axis", text="Axis", text_ctxt="Poly Toolset", toggle=True)
            box.prop(mod, "use_clip")
            box.prop(mod, "use_mirror_merge")
            box.prop(mod, "merge_threshold")
            box.prop(mod, "mirror_object")
    layout.operator("ppo.add_mirror", icon="MOD_MIRROR")


class PPO_OT_add_mirror(bpy.types.Operator):
    bl_idname = "ppo.add_mirror"
    bl_label = "Add Mirror"
    bl_translation_context = "Poly Toolset"
    bl_description = "Add a Mirror modifier with Clipping and Merge on"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and context.edit_object is not None

    def execute(self, context):
        mod = context.edit_object.modifiers.new("PolyTailor Mirror", "MIRROR")
        mod.use_clip = True
        mod.use_mirror_merge = True
        mod.show_in_editmode = True
        mod.show_on_cage = False
        return {"FINISHED"}


class PPO_PT_general(bpy.types.Panel):
    bl_label = "Retopology Settings"
    bl_translation_context = "Poly Toolset"
    bl_space_type = "VIEW_3D"
    bl_region_type = "HEADER"

    def draw(self, context):
        draw_general(context, self.layout)


def draw_relax_algorithm(context, layout):
    """Shared inline controls for the wheel and native toolbar popover."""
    scene = props.settings(context)
    layout.prop(scene, "smooth_iterations")
    layout.prop(scene, "relax_even_spacing")
    layout.label(text="Robust Relax (Alt: Grid Relax)", text_ctxt="Poly Toolset")


class PPO_PT_relax(bpy.types.Panel):
    bl_label = "Relax Algorithm"
    bl_translation_context = "Poly Toolset"
    bl_space_type = "VIEW_3D"
    bl_region_type = "HEADER"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        draw_relax_algorithm(context, layout)


class PPO_PT_sidebar(bpy.types.Panel):
    """Tools and the settings of the current tool."""
    bl_label = "PolyTailor"
    bl_translation_context = "Poly Toolset"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PolyTailor"

    @classmethod
    def poll(cls, context):
        return context.mode in {"EDIT_MESH", "OBJECT"}

    def draw(self, context):
        layout = self.layout
        if context.mode == "OBJECT":
            col = layout.column(align=True)
            col.label(text="New Retopology Mesh", text_ctxt="Poly Toolset")
            col.operator("ppo.new_target", text="At 3D Cursor", text_ctxt="Poly Toolset", icon_value=icon("polypen")).location = "CURSOR"
            row = col.row(align=True)
            row.enabled = context.view_layer.objects.active is not None
            row.operator("ppo.new_target", text="From Active Object", text_ctxt="Poly Toolset", icon="PIVOT_ACTIVE").location = "ACTIVE"
            return
        scene = props.settings(context)
        current = active_tool_id(context)
        row = layout.row(align=True)
        row.scale_y = 1.3
        for name in ("polypen", "fill", "tweak", "relax"):
            op = row.operator("ppo.switch_tool", text="", icon_value=icon(name), depress=current == f"ppo.{name}")
            op.tool = f"ppo.{name}"
        if current == "ppo.polypen":
            draw_build_settings(context, layout)
        elif current == "ppo.fill":
            # no Fill Pattern dropdown (Fill always uses AUTO)
            from ..tools import draw_smoothing
            draw_smoothing(context, layout.box().column(align=True))
        elif current in {"ppo.tweak", "ppo.relax"}:
            brush = scene.tweak if current == "ppo.tweak" else scene.relax
            col = layout.column(align=True)
            col.prop(brush, "radius")
            col.prop(brush, "strength", slider=True)
            col.prop(brush, "falloff", slider=True)
            layout.operator("ppo.brush_reset", text="Reset Brush", text_ctxt="Poly Toolset", icon="LOOP_BACK")
            draw_masking(context, layout.box().column())
            if current == "ppo.relax":
                layout.popover("PPO_PT_relax", text="Relax Algorithm", text_ctxt="Poly Toolset")


class _EditSub:
    """Sub-panel of the PolyTailor sidebar, shown in Edit Mode."""
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PolyTailor"
    bl_parent_id = "PPO_PT_sidebar"

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH"


class PPO_PT_snapping(_EditSub, bpy.types.Panel):
    bl_label = "Snapping"
    bl_translation_context = "Poly Toolset"

    def draw(self, context):
        layout = self.layout
        scene = props.settings(context)
        ts = context.scene.tool_settings
        row = layout.row(align=True)
        row.prop(ts, "use_snap", text="Snap", text_ctxt="Poly Toolset", toggle=True)   # RNA supplies the on/off icon
        row.prop(scene, "clamp_to_source", text="Stop at Edge", text_ctxt="Poly Toolset", toggle=True,
                 icon="CLIPUV_DEHLT" if scene.clamp_to_source else "CLIPUV_HLT")
        col = layout.column(align=True)
        col.active = ts.use_snap
        col.row(align=True).prop(ts, "snap_elements_individual", expand=True)
        row = layout.row(align=True)
        draw_modes(row, scene)
        sub = row.row(align=True)
        sub.active = ts.use_snap
        sub.popover("VIEW3D_PT_snapping", text="Targets", text_ctxt=CONTEXT)
        col = layout.column(align=True)
        col.prop(scene, "snap_self")
        col.prop(scene, "use_native_snap")
        col.prop(scene, "ignore_derived")
        derived = raycast.derived_objects(context)
        if derived:
            box = layout.box().column(align=True)
            box.scale_y = 0.8
            for name in sorted(derived)[:5]:
                reason = raycast.derived_reasons.get(name, "")
                why = (iface_("shared mesh") if reason == "MESH"
                       else reason.replace("MODIFIER:", iface_("modifier") + ": "))
                box.label(text=f"{name}  ({why})" if why else name, icon="HIDE_ON")
            if len(derived) > 5:
                box.label(text=f"+{len(derived) - 5}")
        layout.separator()
        col = layout.column(align=True)
        col.prop(ts, "use_mesh_automerge", text="Auto Merge", text_ctxt="Poly Toolset", toggle=True)
        col.prop(scene, "merge_radius")
        col = layout.column(align=True)   # moved here from the header popover
        col.prop(ts, "use_transform_correct_face_attributes", text="Correct Face Attributes",
                 text_ctxt="Poly Toolset Controls")   # its existing translation
        row = col.row()
        row.active = ts.use_transform_correct_face_attributes
        row.prop(ts, "use_transform_correct_keep_connected")


class PPO_PT_sources(_EditSub, bpy.types.Panel):
    bl_label = "Sources"
    bl_translation_context = "Poly Toolset"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        scene = props.settings(context)
        layout.prop(scene, "source_object")
        layout.prop(scene, "source_collection")
        layout.prop(scene, "snap_only_selected")
        layout.prop(context.scene.tool_settings, "use_snap_selectable", text="Selectable Sources Only", text_ctxt="Poly Toolset")


class PPO_PT_pins(_EditSub, bpy.types.Panel):
    bl_label = "Constraint Tools"
    bl_translation_context = "Poly Toolset"

    def draw(self, context):
        row = self.layout.row()
        draw_pin_glue(row.row(align=True), row.row(align=True))


class PPO_PT_smoothing(bpy.types.Panel):
    """Build's Alt+Wheel Smooth settings, opened from the tool header."""
    bl_label = "Alt+Wheel Smooth"
    bl_translation_context = "Poly Toolset"
    bl_description = "Alt+Wheel Smooth settings (Build and Fill)"
    bl_space_type = "VIEW_3D"
    bl_region_type = "HEADER"

    def draw(self, context):
        from ..tools import draw_smoothing
        draw_smoothing(context, self.layout.column(align=True))


class PPO_PT_display(_EditSub, bpy.types.Panel):
    bl_label = "Display & Mirror"
    bl_translation_context = "Poly Toolset"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        if context.space_data.type == "VIEW_3D":
            overlay = context.space_data.overlay
            layout.prop(overlay, "show_retopology")
            layout.prop(overlay, "retopology_offset")
            layout.prop(overlay, "show_wireframes")
            layout.prop(overlay, "show_fade_inactive")
            layout.prop(overlay, "fade_inactive_alpha")
        layout.separator()
        for mod in context.edit_object.modifiers:
            if mod.type == "MIRROR":
                box = layout.box()
                box.label(text=mod.name, icon="MOD_MIRROR")
                box.row(align=True).prop(mod, "use_axis", text="Axis", text_ctxt="Poly Toolset", toggle=True)
                box.prop(mod, "use_clip")
                box.prop(mod, "use_mirror_merge")
                box.prop(mod, "merge_threshold")
                box.prop(mod, "mirror_object")
        layout.operator("ppo.add_mirror", icon="MOD_MIRROR")


class PPO_PT_defaults(_EditSub, bpy.types.Panel):
    bl_label = "Defaults"
    bl_translation_context = "Poly Toolset"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        saved = defaults.has_saved()
        layout.label(text="Using your saved defaults" if saved else "Using PolyTailor's defaults", text_ctxt="Poly Toolset",
                     icon="CHECKMARK" if saved else "INFO")
        col = layout.column(align=True)
        col.operator("ppo.defaults_save", icon="FILE_TICK")
        col.operator("ppo.defaults_restore", icon="LOOP_BACK")


class PPO_PT_shortcuts(_EditSub, bpy.types.Panel):
    bl_label = "Shortcuts"
    bl_translation_context = "Poly Toolset"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        from .shortcuts import draw_shortcuts   # generated from the status-bar key tables
        draw_shortcuts(self.layout, context)


_classes = (PPO_OT_add_mirror, PPO_PT_general, PPO_PT_relax, PPO_PT_sidebar, PPO_PT_snapping,
            PPO_PT_sources, PPO_PT_pins, PPO_PT_smoothing, PPO_PT_display, PPO_PT_defaults,
            PPO_PT_shortcuts)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
