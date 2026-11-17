# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Toolbar tools with their keymaps and tool header settings."""

from pathlib import Path

import bpy
from bpy.types import WorkSpaceTool

from . import props
from .ui import pie

ICONS = Path(__file__).parent / "icons"


def _shipped(name, light):
    mapping = ICONS / "toolbar_icons.txt"
    if mapping.exists():
        names = dict(line.split() for line in mapping.read_text().splitlines() if line.strip())
        key = name + ("_light" if light else "")
        name = names.get(key, names.get(name, name))
    return ICONS / name


def _icon_colours():
    """(main, dim, accent) RGBA bytes the toolbar icons should use, or None = the shipped files (Custom with
    the default colours: the default look is unchanged)."""
    from . import palette, prefs
    from .ui import custom_pie
    s = prefs.get()
    mode = getattr(s, "icon_colours", "CUSTOM")
    if mode == "CUSTOM":
        def dflt(c, d):
            return all(abs(a - b) < .5 / 255 for a, b in zip(c, d))
        if (dflt(s.icon_main, [c / 255 for c in palette.ICON_MAIN]) and dflt(s.icon_dim, [c / 255 for c in palette.ICON_DIM])
                and (palette.close(tuple(s.icon_accent), palette.DARK) or palette.close(tuple(s.icon_accent), palette.LEGACY))
                and (palette.close(tuple(s.accent), palette.DARK) or palette.close(tuple(s.accent), palette.LEGACY))):
            return None   # the default look: the shipped files (dark / light)
        cols = custom_pie.icon_colours(None)[:3]
    else:
        style = bpy.context.preferences.themes[0].user_interface.wcol_toolbar_item
        cols = custom_pie.icon_colours(style)[:3]
    return tuple(tuple(min(255, max(0, round(float(x) * 255))) for x in c[:4]) for c in cols)


_wrote = [False]   # a recoloured icon was written since the last prune (_prune_icons)


def _recoloured(name, cols):
    """The shipped dark VCO geometry with its three colours replaced (exact match); written once per colour set."""
    import hashlib
    src = _shipped(name, False).with_suffix(".dat")
    tag = hashlib.md5(repr((src.name, cols)).encode()).hexdigest()[:10]
    folder = Path(bpy.utils.extension_path_user(__package__, path="toolbar_icons", create=True))   # the user folder only
    out = folder / f"{name}_c{tag}"
    dat = out.with_suffix(".dat")
    if not dat.exists():
        import numpy as np
        data = bytearray(src.read_bytes())
        n = (len(data) - 8) // 18
        rgba = np.frombuffer(bytes(data[8 + 6 * n:8 + 18 * n]), np.uint8).reshape(-1, 4).copy()
        new = rgba.copy()
        for old, c in (((229, 229, 229), cols[0]), ((130, 130, 130), cols[1]), ((255, 158, 52), cols[2])):
            new[np.all(rgba[:, :3] == old, axis=1)] = c
        data[8 + 6 * n:8 + 18 * n] = new.tobytes()
        tmp = dat.with_suffix(".tmp")
        tmp.write_bytes(bytes(data)); tmp.replace(dat)
        _wrote[0] = True
    else:
        try:
            dat.touch()   # used again: its age for _prune_icons counts from now (also for another instance)
        except OSError:
            pass
    return out


def _icon(name):
    # the toolbar follows the icon colours; refresh_icons() re-registers the tools for new ones
    try:
        cols = _icon_colours()
        if cols is not None:
            return _recoloured(name, cols).as_posix()
    except Exception as exc:
        print("PolyTailor toolbar icons:", exc)
    return _shipped(name, pie.light_theme()).as_posix()


_icon_sig = [None]


def refresh_icons(force=False):
    """new toolbar icons when the wanted icon colours changed (prefs, Accent Color, theme)."""
    try:
        sig = (_icon_colours(), pie.light_theme())
    except Exception:
        return
    if sig == _icon_sig[0] and not force:
        return
    _icon_sig[0] = sig
    try:
        pie.refresh_tint(sig[0])   # panel / menu icons
    except Exception as exc:
        print("PolyTailor icon tint:", exc)
    for window in getattr(bpy.context.window_manager, "windows", ()):
        for area in window.screen.areas:
            area.tag_redraw()
    if any(tool.bl_icon != _icon(tool.bl_idname.split(".")[-1]) for tool in TOOLS if _registered[0]):
        _unregister_tools()   # Blender's API: a tool's icon is set when it is registered
        _register_tools()


def _km(idname, type, value, props=None, **mods):
    return (idname, {"type": type, "value": value, **mods}, {"properties": list(props.items())} if props else None)


# Ordinary mesh selection belongs to the user's keymaps (including add-ons).
# Only reserve the PTS drag gestures here (drag: move / box select, Shift+drag: add
# box select). Do not bind click / Shift-click selection or double-click to native
# operators in the tool keymap: those bindings shadow user selection operators.
DRAG_KEYS = (
    _km("ppo.drag", "LEFTMOUSE", "CLICK_DRAG"),
    _km("ppo.drag", "LEFTMOUSE", "CLICK_DRAG", {"extend": True}, shift=True),
    # Box deselect: Blender's Ctrl+drag, but held Ctrl is the Build / Fill preview, so Shift+Ctrl+drag.
    # Blender's default (left-click select) keymap has no Shift+Ctrl+LMB drag for an add-on tool.
    _km("ppo.drag", "LEFTMOUSE", "CLICK_DRAG", {"deselect": True}, shift=True, ctrl=True),
)

COMMON_KEYS = (
    _km("ppo.move", "G", "PRESS"),
    _km("ppo.move", "E", "PRESS", {"extrude": True}),
)

# Alt+Wheel relaxes the selection: a Build feature.
SMOOTH_KEYS = (
    _km("ppo.smooth_selection", "WHEELUPMOUSE", "PRESS", alt=True),
    _km("ppo.smooth_selection", "WHEELDOWNMOUSE", "PRESS", alt=True),
)

BRUSH_KEYS = (
    _km("ppo.brush_hover", "MOUSEMOVE", "ANY", any=True),
    _km("ppo.brush_hover", "RIGHTMOUSE", "PRESS", any=True),
    _km("ppo.brush_adjust", "F", "PRESS", {"mode": "RADIUS"}),
    _km("ppo.brush_adjust", "F", "PRESS", {"mode": "STRENGTH"}, shift=True),
    _km("ppo.brush_adjust", "F", "PRESS", {"mode": "FALLOFF"}, ctrl=True),
)


# MMB click = Brush <-> Build, Ctrl+MMB click = Fill <-> Build; drags stay orbit / zoom (ops/tool_switch.py).
MODE_KEYS = (
    _km("ppo.mode_toggle", "MIDDLEMOUSE", "PRESS", {"kind": "BRUSH"}),
    _km("ppo.mode_toggle", "MIDDLEMOUSE", "PRESS", {"kind": "FILL"}, ctrl=True),
)

def _separator(layout):
    layout.separator(type="LINE") if bpy.app.version >= (4, 2, 0) else layout.separator()


def _session_entry():
    """the tool's settings are drawn (tool header / Tool tab): PolyTailor's tool is shown. Entered by a route without
    a notify of its own (Ctrl+PageUp / PageDown, a scene or editor switched back): one look sets it up; nothing
    while PolyTailor is in use."""
    from . import session
    if not session._in_use[0]:
        session.entry_look()


def _draw_masking(context, layout, header):
    scene = props.settings(context)
    if header:
        layout.row(heading="Selected:", heading_ctxt="Poly Toolset", align=True).prop(scene, "mask_selected", expand=True, icon_only=True)
        layout.row(heading="Boundary:", heading_ctxt="Poly Toolset", align=True).prop(scene, "mask_boundary", expand=True, icon_only=True)
        layout.prop(scene, "include_occluded")
    else:
        pie.draw_masking(context, layout.column())


def _draw_common(context, layout):
    ts = context.scene.tool_settings
    row = layout.row(align=True)
    # use_snap / use_mesh_automerge carry their own on/off icon pair in RNA: Blender shows
    # the next icon when True, so passing the "on" icon here showed the wrong one
    row.prop(ts, "use_snap", text="")
    scene = props.settings(context)
    row.prop(scene, "clamp_to_source", text="", icon="CLIPUV_DEHLT" if scene.clamp_to_source else "CLIPUV_HLT")  # Blender's mask convention
    row.prop(ts, "use_mesh_automerge", text="")
    pie.draw_modes(row, scene)
    row.popover("PPO_PT_general", text="", icon="PREFERENCES")


def _draw_brush(context, layout, brush):
    header = context.region.type == "TOOL_HEADER"
    if header:
        layout.label(text="Brush:", text_ctxt="Poly Toolset")
    layout.prop(brush, "radius")
    layout.prop(brush, "strength", slider=True)
    layout.prop(brush, "falloff", slider=True)
    layout.operator("ppo.brush_reset", text="" if header else "Reset Brush", text_ctxt="Poly Toolset", icon="LOOP_BACK")
    if header:
        _separator(layout)
    _draw_masking(context, layout, header)
    if header:
        _separator(layout)
        _draw_common(context, layout)


class PPO_Tool_PolyPen(WorkSpaceTool):
    bl_space_type = "VIEW_3D"
    bl_context_mode = "EDIT_MESH"
    bl_idname = "ppo.polypen"
    bl_label = "Build "   # TOOL_LABEL_IDS
    bl_description = "Hold Ctrl to preview, Ctrl+LMB to add"
    bl_icon = _icon("polypen")
    bl_widget = None
    bl_keymap = (
        _km("ppo.polypen", "LEFT_CTRL", "PRESS", ctrl=-1),
        _km("ppo.polypen", "RIGHT_CTRL", "PRESS", ctrl=-1),
    ) + MODE_KEYS + DRAG_KEYS + SMOOTH_KEYS + COMMON_KEYS

    def draw_settings(context, layout, tool):
        _session_entry()
        scene = props.settings(context)
        header = context.region.type == "TOOL_HEADER"
        if header:
            # 3in1 is the only Build mode; its face type is the one choice left.
            layout.label(text="Faces:", text_ctxt=props.FACES_CONTEXT)
            layout.row(align=True).prop(scene, "merge3_faces", expand=True)
            _separator(layout)
            layout.popover("PPO_PT_smoothing", text="Alt+Wheel Smooth", text_ctxt="Poly Toolset")
            _separator(layout)
            _draw_common(context, layout)
        else:
            pie.draw_build_settings(context, layout.column())
            _draw_common(context, layout)


class PPO_Tool_Fill(WorkSpaceTool):
    bl_space_type = "VIEW_3D"
    bl_context_mode = "EDIT_MESH"
    bl_idname = "ppo.fill"
    bl_label = "Fill "   # TOOL_LABEL_IDS
    bl_description = ("Select one closed outline, or hover a hole, then hold Ctrl to preview.\n"
                      "Ctrl+Wheel: change choice, Ctrl+click: fill")
    bl_icon = _icon("fill")
    bl_widget = None
    bl_keymap = (
        _km("ppo.fill", "LEFT_CTRL", "PRESS"),
        _km("ppo.fill", "RIGHT_CTRL", "PRESS"),
    ) + MODE_KEYS + DRAG_KEYS + SMOOTH_KEYS + COMMON_KEYS

    def draw_settings(context, layout, tool):
        _session_entry()
        # no Fill Pattern dropdown; Fill always uses AUTO (Ctrl+Wheel changes the choice while previewing)
        if context.region.type == "TOOL_HEADER":
            layout.popover("PPO_PT_smoothing", text="Alt+Wheel Smooth", text_ctxt="Poly Toolset")
            _separator(layout)
            _draw_common(context, layout)
        else:
            draw_smoothing(context, layout.box().column(align=True))
            _draw_common(context, layout)


def draw_smoothing(context, layout, maximum=False):
    """Alt+Wheel Smooth settings (Build and Fill): the method and, for Brush Relax, its algorithm."""
    scene = props.settings(context)
    layout.label(text="Alt+Wheel Smooth", text_ctxt="Poly Toolset")
    layout.prop(scene, "smooth_method", text="")
    if scene.smooth_method == "RF" or maximum:
        from .ui.panels import draw_relax_algorithm
        draw_relax_algorithm(context, layout)


class PPO_Tool_Tweak(WorkSpaceTool):
    bl_space_type = "VIEW_3D"
    bl_context_mode = "EDIT_MESH"
    bl_idname = "ppo.tweak"
    bl_label = "Tweak "   # TOOL_LABEL_IDS
    bl_description = "Push vertices around with a soft brush"
    bl_icon = _icon("tweak")
    bl_widget = None
    bl_keymap = (_km("ppo.tweak", "LEFTMOUSE", "PRESS"),) + BRUSH_KEYS + COMMON_KEYS + MODE_KEYS

    def draw_settings(context, layout, tool):
        _session_entry()
        _draw_brush(context, layout, props.settings(context).tweak)


class PPO_Tool_Relax(WorkSpaceTool):
    bl_space_type = "VIEW_3D"
    bl_context_mode = "EDIT_MESH"
    bl_idname = "ppo.relax"
    bl_label = "Relax "   # TOOL_LABEL_IDS
    bl_description = "Relax topology; hold Ctrl to temporarily drag with the same brush, Alt for Grid Relax"
    bl_icon = _icon("relax")
    bl_widget = None
    bl_keymap = (
        _km("ppo.relax", "LEFTMOUSE", "PRESS"),
        _km("ppo.relax", "LEFTMOUSE", "PRESS", ctrl=True),
        _km("ppo.relax", "LEFTMOUSE", "PRESS", {"grid": True}, alt=True),
    ) + BRUSH_KEYS + COMMON_KEYS + MODE_KEYS

    def draw_settings(context, layout, tool):
        _session_entry()
        _draw_brush(context, layout, props.settings(context).relax)
        layout.popover("PPO_PT_relax", text="Relax Algorithm", text_ctxt="Poly Toolset")


TOOLS = (PPO_Tool_PolyPen, PPO_Tool_Fill, PPO_Tool_Tweak, PPO_Tool_Relax)
# TOOL_LABEL_IDS: each tool label is the English word and a trailing space. Blender's toolbar translates tool labels in
# its "Operator" context, where its own "Fill" / "Relax" read 填入 / 鬆開; a label of its own lets i18n.py give the
# add-on's terms (填補 / 放鬆 ...) to these tools only, without changing Blender's own words.


_registered = [False]


def _refresh_later():
    try:
        refresh_icons()
    except Exception as exc:
        print("PolyTailor toolbar icons:", exc)
    return None


def request_refresh(delay=.3):
    """refresh_icons() once, `delay` after the last request: a one-shot armed only by an icon colour edit in the
    preferences (a colour-picker drag sends many) and by PolyTailor coming into use (session)."""
    if bpy.app.timers.is_registered(_refresh_later):
        bpy.app.timers.unregister(_refresh_later)
    bpy.app.timers.register(_refresh_later, first_interval=delay)


_ICON_MAX_AGE = 24 * 3600   # seconds: an unused recoloured icon younger than this is kept (another instance)


def _prune_icons():
    """after a new colour set was written: the recoloured icons (<name>_c<10 hex>.dat, see _recoloured) that no
    registered tool uses and that were not written or used for 24 hours are deleted from the user folder. The age
    keeps the set of another Blender instance sharing the folder."""
    if not _wrote[0]:
        return
    _wrote[0] = False
    try:
        import time
        keep = {Path(tool.bl_icon).name + ".dat" for tool in TOOLS}
        folder = Path(bpy.utils.extension_path_user(__package__, path="toolbar_icons", create=True))
        now = time.time()
        for dat in folder.glob("*_c" + "?" * 10 + ".dat"):
            if dat.name in keep or not all(c in "0123456789abcdef" for c in dat.stem[-10:]):
                continue
            try:
                if now - dat.stat().st_mtime < _ICON_MAX_AGE:
                    continue
                dat.unlink()
            except OSError:
                _wrote[0] = True   # try again after the next write
    except Exception as exc:
        print("PolyTailor toolbar icons:", exc)


def _register_tools():
    previous = {"builtin.poly_build"}
    for i, tool in enumerate(TOOLS):
        tool.bl_icon = _icon(tool.bl_idname.split(".")[-1])
        bpy.utils.register_tool(tool, after=previous, separator=(i == 0), group=False)
        previous = {tool.bl_idname}
    _registered[0] = True
    _prune_icons()


def _unregister_tools():
    _registered[0] = False
    for tool in reversed(TOOLS):
        bpy.utils.unregister_tool(tool)


def register():
    _register_tools()
    pie.AVAILABLE_TOOLS.update(tool.bl_idname for tool in TOOLS)
    try:   # the panel / menu tint for the current colours (again when PolyTailor comes into use: session)
        refresh_icons()
    except Exception as exc:
        print("PolyTailor toolbar icons:", exc)


def unregister():
    if bpy.app.timers.is_registered(_refresh_later):
        bpy.app.timers.unregister(_refresh_later)
    _icon_sig[0] = None
    pie.AVAILABLE_TOOLS.clear()
    _unregister_tools()
