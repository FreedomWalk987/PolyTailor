# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Native sidebar and popovers shared by the toolbar and pie."""

import bpy
from ..i18n import tr as iface_   # add-on context first

from .. import defaults, props
from ..core import raycast
from ..ops.common import active_tool_id
from .pie import (icon, draw_modes, draw_build_settings, draw_brush_settings, draw_pin_glue,
                  draw_snapping_rows, draw_snap_extras, draw_selection_rows)
from ..i18n import CONTEXT


def draw_general(context, layout, toggle=True):
    """Overlay and Mirror: the tool header's gear and the sidebar's Display & Mirror. toggle=False leaves out the
    overlay on/off row (the sidebar has it in Select Mode as Retopo Shading)."""
    layout.use_property_split = True
    if context.space_data.type == "VIEW_3D":
        overlay = context.space_data.overlay
        if toggle:
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


# the sidebar's tool buttons, in the W wheel's order: (tool, icon, name)
TOOL_BUTTONS = (("ppo.polypen", "polypen", "Build"), ("ppo.fill", "fill", "Fill"),
                ("ppo.relax", "relax", "Relax"), ("ppo.tweak", "tweak", "Tweak"))


def wheel_keys(context):
    """The keys of the W wheel and of Shift+W as the user's keymap has them (W / Shift W if not found)."""
    from .pie import _keymaps
    found = {}   # kind -> key text ('' when the user's keymap has it switched off)
    wm = context.window_manager
    try:
        for kc in (wm.keyconfigs.user, wm.keyconfigs.addon):   # the user's keymap wins; the add-on's is the fallback
            seen = {}
            for km_name in {name for name, _idname in _keymaps} if kc else ():
                km = kc.keymaps.get(km_name)
                for kmi in (km.keymap_items if km else ()):
                    kind = ("wheel" if kmi.idname == "ppo.call_pie" else
                            "panel" if kmi.idname == "wm.call_panel"
                            and getattr(kmi.properties, "name", "") == "PPO_PT_quick_all" else None)
                    if kind is not None and (kind not in seen or kmi.active):
                        seen[kind] = kmi.to_string() if kmi.active else ""
            for kind, text in seen.items():
                found.setdefault(kind, text)
    except (AttributeError, ReferenceError, RuntimeError):
        pass
    return found.get("wheel", "W"), found.get("panel", "Shift W")


def wheel_hint_visible(context):
    """The W hint row: while Show Wheel Hint is on and the wheel was opened fewer than 3 times."""
    from .. import prefs
    p = prefs.get(context)
    return bool(getattr(p, "show_wheel_hint", True)) and getattr(p, "wheel_open_count", 0) < 3


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

    def draw_header_preset(self, context):
        if context.mode == "EDIT_MESH":   # the wheel's key; hovering shows what it opens (it opens in the viewport)
            key = wheel_keys(context)[0]
            if key:
                self.layout.label(text=key, translate=False)

    def draw(self, context):
        layout = self.layout
        if context.mode == "OBJECT":
            col = layout.column(align=True)
            col.operator("ppo.new_target", text="New Mesh (3D)", text_ctxt="Poly Toolset", icon_value=icon("polypen")).location = "CURSOR"
            row = col.row(align=True)
            row.enabled = context.view_layer.objects.active is not None
            row.operator("ppo.new_target", text="Retopology Mesh (Topo)", text_ctxt="Poly Toolset", icon="PIVOT_ACTIVE").location = "ACTIVE"
            return
        scene = props.settings(context)
        current = active_tool_id(context)
        draw_modes(layout, scene)   # 3D / Topo changes how every tool works: on top, not in Snapping
        row = layout.row(align=True)
        row.scale_y = 1.3
        for tool_id, name, title in TOOL_BUTTONS:
            op = row.operator("ppo.switch_tool", text=title, text_ctxt=CONTEXT, icon_value=icon(name),
                              depress=current == tool_id)
            op.tool = tool_id
        wheel_key, panel_key = wheel_keys(context)
        if wheel_key and wheel_hint_visible(context):   # no hint for a switched-off wheel key
            col = layout.column(align=True)
            col.active = False
            col.label(text=f"{wheel_key}   {iface_('PolyTailor Wheel')}", translate=False)
            if panel_key:   # no row for a switched-off Shift+W key
                col.label(text=f"{panel_key}   {iface_('PolyTailor Settings')}", translate=False)
        title = next((t for tool_id, _name, t in TOOL_BUTTONS if tool_id == current), None)
        if title is not None:   # another (Blender) tool: no tool settings
            layout.label(text=title, text_ctxt=CONTEXT)
            if current == "ppo.polypen":
                draw_build_settings(context, layout)
            elif current == "ppo.fill":
                # no Fill Pattern dropdown (Fill always uses AUTO)
                from ..tools import draw_smoothing
                draw_smoothing(context, layout.box().column(align=True))
            else:
                draw_brush_settings(context, layout, current)   # the active brush; Relax Algorithm inline
        layout.label(text="Constraint Tools", text_ctxt=CONTEXT)
        row = layout.row()
        draw_pin_glue(row.row(align=True), row.row(align=True))


class _EditSub:
    """Sub-panel of the PolyTailor sidebar, shown in Edit Mode; closed until the user opens it (Blender remembers)."""
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PolyTailor"
    bl_parent_id = "PPO_PT_sidebar"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH"


class PPO_PT_snapping(_EditSub, bpy.types.Panel):
    bl_label = "Snapping"
    bl_translation_context = "Poly Toolset"
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        draw_snapping_rows(context, layout.column(), modes=False)
        draw_snap_extras(context, layout.column(align=True))
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


class PPO_PT_select_mode(_EditSub, bpy.types.Panel):
    bl_label = "Select Mode"
    bl_translation_context = "Poly Toolset"
    bl_order = 2

    def draw(self, context):
        draw_selection_rows(context, self.layout.column(), n_only=True)


class PPO_PT_sources(_EditSub, bpy.types.Panel):
    bl_label = "Sources"
    bl_translation_context = "Poly Toolset"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 3

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        scene = props.settings(context)
        layout.prop(scene, "source_object")
        layout.prop(scene, "source_collection")
        layout.prop(context.scene.tool_settings, "use_snap_selectable", text="Selectable Sources Only", text_ctxt="Poly Toolset")


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
    bl_order = 4

    def draw(self, context):
        draw_general(context, self.layout, toggle=False)


class PPO_PT_defaults(_EditSub, bpy.types.Panel):
    bl_label = "Defaults"
    bl_translation_context = "Poly Toolset"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 5

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
    bl_order = 6

    def draw(self, context):
        from .shortcuts import draw_shortcuts   # generated from the status-bar key tables
        draw_shortcuts(self.layout, context)


_classes = (PPO_OT_add_mirror, PPO_PT_general, PPO_PT_relax, PPO_PT_sidebar, PPO_PT_snapping,
            PPO_PT_select_mode, PPO_PT_sources, PPO_PT_smoothing, PPO_PT_display, PPO_PT_defaults,
            PPO_PT_shortcuts)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
