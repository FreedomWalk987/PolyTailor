# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""MMB Tool Switch: an MMB click switches Brush <-> Build, a Ctrl+MMB click switches Fill <-> Build; a drag (past
Blender's drag threshold) or a long press stays Blender's own orbit (MMB) / zoom (Ctrl+MMB). The new tool's name shows
at the cursor for a moment. Preferences > MMB Tool Switch turns it off."""

import time

import bpy

from .. import prefs
from ..core import draw

BUILD_TOOL = "ppo.polypen"
BRUSH_TOOLS = ("ppo.tweak", "ppo.relax")
FILL_TOOL = "ppo.fill"
TOOL_NAMES = {BUILD_TOOL: "Build", FILL_TOOL: "Fill", "ppo.relax": "Relax", "ppo.tweak": "Tweak"}
_mode_brush = ["ppo.relax"]   # the brush the MMB switch returns to


def active_tool(context):
    try:
        tool = context.workspace.tools.from_space_view3d_mode(context.mode, create=False)
        return tool.idname if tool else ""
    except (AttributeError, RuntimeError):
        return ""


class PPO_OT_mode_toggle(bpy.types.Operator):
    """MMB click switches Brush <-> Build, Ctrl+MMB click switches Fill <-> Build. A drag (past Blender's drag
    threshold) or a long press stays Blender's: orbit (MMB) / zoom (Ctrl+MMB)."""
    bl_idname = "ppo.mode_toggle"
    bl_label = "Switch Tool"
    bl_translation_context = "Poly Toolset"
    bl_description = "Click MMB: Build \u2194 Brush; Ctrl+MMB: Build \u2194 Fill. Drags still orbit and zoom"
    bl_options = {"INTERNAL"}

    kind: bpy.props.EnumProperty(translation_context="Poly Toolset", items=(("BRUSH", "Brush", ""), ("FILL", "Fill", "")), default="BRUSH",
                                 options={"SKIP_SAVE"})
    CLICK_SECONDS = .45

    @classmethod
    def poll(cls, context):
        return (context.area is not None and context.area.type == "VIEW_3D"
                and context.mode == "EDIT_MESH" and active_tool(context) in (BUILD_TOOL, FILL_TOOL) + BRUSH_TOOLS)

    def invoke(self, context, event):
        if not getattr(prefs.get(context), "mmb_tool_switch", True):
            return {"PASS_THROUGH"}   # MMB / Ctrl+MMB stay Blender's orbit / zoom
        self.start = (event.mouse_x, event.mouse_y)
        self.t0 = time.perf_counter()
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type in {"MOUSEMOVE", "INBETWEEN_MOUSEMOVE"}:
            dx, dy = event.mouse_x - self.start[0], event.mouse_y - self.start[1]
            if (dx * dx + dy * dy) ** .5 >= context.preferences.inputs.drag_threshold_mouse:
                if self.kind == "FILL":
                    bpy.ops.view3d.zoom("INVOKE_DEFAULT")
                else:
                    bpy.ops.view3d.rotate("INVOKE_DEFAULT")
                return {"FINISHED"}
            return {"RUNNING_MODAL"}
        if event.type == "MIDDLEMOUSE" and event.value == "RELEASE":
            if time.perf_counter() - self.t0 <= self.CLICK_SECONDS:
                _toast(context, switch_mode(context, self.kind), event)
            return {"FINISHED"}
        if event.type in {"ESC", "WINDOW_DEACTIVATE"}:
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}


def _toast(context, tool, event=None):
    """the new tool's name at the cursor, fading over 0.8 s (MMB switches are otherwise silent)."""
    try:
        from ..i18n import tr
        xy = (event.mouse_x, event.mouse_y) if event is not None else None
        draw.toast(context, tr(TOOL_NAMES.get(tool, tool)), .8, xy)
    except Exception:
        pass   # feedback only


def switch_mode(context, kind):
    """BRUSH: brush -> Build, Build / Fill -> the last brush. FILL: Fill -> Build, else -> Fill."""
    tool = active_tool(context)
    if kind == "FILL":
        target = BUILD_TOOL if tool == FILL_TOOL else FILL_TOOL
    elif tool in BRUSH_TOOLS:
        _mode_brush[0] = tool
        target = BUILD_TOOL
    else:
        target = _mode_brush[0]   # Relax by default, then the brush last left by MMB
    bpy.ops.ppo.switch_tool(tool=target)
    return target


def register():
    bpy.utils.register_class(PPO_OT_mode_toggle)


def unregister():
    bpy.utils.unregister_class(PPO_OT_mode_toggle)
