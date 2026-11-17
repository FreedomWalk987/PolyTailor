# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Viewport help, using scalar snapshots from the existing operator lifecycle.

No mesh/source queries, event ownership, retained RNA references or redraw timer.
"""
from functools import lru_cache

import blf
import bpy
from .. import palette
from mathutils import Vector
from bpy.app.handlers import persistent
from ..i18n import tr as iface_   # add-on context first

from .. import prefs, props
from ..core import axis as axis_utils, draw, raycast
from ..ops.common import active_tool_id
from .custom_layout import format_label
from . import keytable
from ..i18n import tr

_handle = None
_states = {}
_TOOLS = {"ppo.polypen", "ppo.fill", "ppo.relax", "ppo.tweak"}
_OPERATORS = {"ppo.polypen", "ppo.fill", "ppo.drag", "ppo.move", "ppo.slide", "ppo.relax", "ppo.tweak",
              "ppo.brush_adjust", "ppo.smooth_selection"}
# the idle key rows live in ui/keytable.py (one source for the status bar, the HUD and Shortcuts)
_ACTIONS = {
    "NONE": "Nothing to add here", "VERT": "Add vertex",
    "VERT_EDGE": "Extend edge", "EDGE_TRI": "Add triangle", "EDGE_QUAD": "Add quad",
    "TRI_QUAD": "Extend triangle to quad", "EDGE_BRIDGE": "Bridge edges",
    "SPLIT_EDGE": "Split edge", "EDGE_SPLIT_EDGE": "Connect and split edge",
    "SPLIT_QUAD": "Cut face", "SPLIT_QUAD_CENTER": "Cut through face center",
    "SPLIT_QUAD_EDGES": "Cut across face edges", "VERT_SPLIT_EDGE": "Connect vertex to edge",
    "WIRE_SPLIT_EDGE_FACE": "Close wire into face", "WIRE_VERT_SPLIT_FACE": "Close wire into face",
}


def _t(message, **values):
    return tr(message).format(**values)


def operator_name(operator):
    """Blender's runtime RNA ID is PPO_OT_move; Python declarations use ppo.move.

    Normalize at the display boundary so live RNA instances and dotted test/dev
    instances reach the same help and axis paths. Do not change operator routing.
    """
    name = getattr(operator, "bl_idname", "") or ""
    if not name:
        name = getattr(getattr(operator, "bl_rna", None), "identifier", "")
    if "_OT_" in name:
        namespace, operation = name.split("_OT_", 1)
        return namespace.lower()+"."+operation
    return name


def _pointer(value):
    return value.as_pointer() if value is not None else 0


def _scope(context):
    key = tuple(_pointer(getattr(context, name, None)) for name in ("window", "area", "region"))
    obj = context.edit_object
    target = (_pointer(context.workspace), _pointer(context.scene), _pointer(obj),
              _pointer(obj.data) if obj is not None else 0)
    return key, target


def finish(operator):
    """Nested completion clears only the ending operator's own snapshot."""
    owner = id(operator)
    for key, state in tuple(_states.items()):
        if state[0] == owner:
            _states.pop(key, None)


def operation_lines(operator, context, name=None):
    """Read existing scalar state only; return translated lines and a mode tag.
    name: read the operator as this one.
    every line is a keytable.Line whose key / state parts are marked in code (keytable.STATE_PARTS)."""
    _t = keytable.line
    name = name or operator_name(operator)
    if name == "ppo.polypen":
        if getattr(operator, "stroke_curve", None) is not None:
            hint = "Release: stroke preview" if operator.stroke_drawing else "Ctrl+move: rotate / scale | Wheel: segments | LMB / Enter: add | Esc / RMB: cancel"
            return (_t(hint),), "STROKE"
        if getattr(operator, "line_world", None) and not operator.placing:
            return (_t("Wheel: segments | Ctrl+LMB: add | Drag: place"),), "PREVIEW"
        if operator.placing:
            return (_t("Drag: place | Shift: precision | Release LMB: confirm | Esc / RMB: cancel"),), "PLACE"
        action = getattr(getattr(operator, "plan", None), "action", "NONE")
        return (_t("Preview: {action}", action=iface_(_ACTIONS.get(action, "Nothing to add here"))),), "PREVIEW"
    if name == "ppo.fill":
        if not getattr(prefs.get(context), "show_debug_info", False):
            return (keytable.marked(operator.hint()),), "PREVIEW"
        return (keytable.marked(operator.status(), state=True),), "PREVIEW"
    if name in {"ppo.tweak", "ppo.relax"}:
        if name == "ppo.relax" and operator.dragging:
            return (_t("Temporary Tweak | Release Ctrl: resume Relax | Uses the Relax brush settings"),
                    _t("Release LMB: finish | Esc / RMB: cancel the entire stroke")), "TEMP_TWEAK"
        return (_t("Release LMB: finish | Esc / RMB: cancel the entire stroke"),), "STROKE"
    if name == "ppo.brush_adjust":
        value = getattr(operator.settings, operator.field)
        value = f"{value:g} px" if operator.field == "radius" else f"{value:.2f}"
        return (_t("{label}: {value} | LMB / Enter: confirm | Esc / RMB: cancel",
                   label=iface_(operator.mode.title()), value=value),), "ADJUST"
    if name == "ppo.drag":
        return (_t("X / Y / Z: global axis (again: unlock) | Shift: precision"),
                _t("Release LMB: confirm | Esc / RMB: cancel")), "MOVE"
    if name == "ppo.move":
        if operator.offset_data:
            return (_t("Boundary offset | LMB / Enter: confirm | Esc / RMB: cancel"),), "OFFSET"
        lines = [_t("X / Y / Z: global axis (again: unlock) | Shift: precision")]
        if operator.axis:
            lines.append(_t("Global axis: {axis}", axis=operator.axis))
        if operator.extrude:
            lines.append(_t("E again: boundary offset"))
            if operator.ladder:
                lines.append(_t("Wheel: extrusion cuts"))
        else:
            lines.append(_t("G again: slide along edges"))
            lines.append(_t("O: proportional editing | Shift+O: falloff"))
            if getattr(operator, "proportional", False):
                falloff = context.scene.tool_settings.proportional_edit_falloff.replace("_", " ").title()
                lines.append(_t("Proportional editing: {falloff} | Wheel: influence radius", falloff=iface_(falloff)))
        lines.append(_t("LMB / Enter: confirm | Esc / RMB: cancel"))
        return tuple(lines), "MOVE"
    if name == "ppo.slide":
        return (_t("Slide along edges | LMB / Enter: confirm | Esc / RMB: cancel"),), "SLIDE"
    if name == "ppo.smooth_selection":
        return (_t("Alt+Wheel up: relax selection | Wheel down: step back | Release Alt: confirm"),), "SMOOTH"
    return (), ""


def observe(operator, context, event):
    if operator_name(operator) not in _OPERATORS or context.mode != "EDIT_MESH":
        return
    tool = active_tool_id(context)
    if tool not in _TOOLS:
        return
    key, target = _scope(context)
    current = _states.get(key)
    # A stationary Relax timer changes geometry, but not the help. Snapshot on
    # invoke and actual input transitions, outside the paint callback.
    # Fill finishes its preview on timer ticks: refresh its line then, too.
    if (event.type == "TIMER" and current is not None and current[0] == id(operator)
            and operator_name(operator) != "ppo.fill"):
        return
    lines, kind = operation_lines(operator, context)
    if lines:
        _states[key] = (id(operator), target, tool, tuple(lines), kind, axis_snapshot(operator),
                        getattr(operator, "proportional_guide", ()))


def axis_snapshot(operator):
    """Retain only the constrained world axis and the original world pivot."""
    axis = getattr(operator, "axis", None)
    if axis not in {"X", "Y", "Z"}:
        return ()
    name = operator_name(operator)
    if name == "ppo.move" and not operator.offset_data:
        origin = operator.base_center
    elif name == "ppo.drag":
        origin = operator.pivot
    else:
        return ()
    return axis, tuple(float(v) for v in origin)


def paint_axis(context, state):
    if state is None or len(state) < 6 or not state[5] or context.region_data is None:
        return
    axis, origin = state[5]
    segment = axis_utils.projected_line(context.region_data.perspective_matrix, origin, axis,
                                        context.region.width, context.region.height)
    if segment is None:
        return
    ui = context.preferences.themes[0].user_interface
    fallback = {"X": (.95, .2, .2), "Y": (.3, .85, .3), "Z": (.3, .5, 1.)}
    color = (*tuple(getattr(ui, "axis_"+axis.lower(), fallback[axis]))[:3], .9)
    scale = max(.5, min(3., prefs.ui_scale(context)))
    a, b = segment
    with draw.Blend():
        if (a[0]-b[0])**2+(a[1]-b[1])**2 < 4:
            draw.ring(a, 5*scale, color, 1.5*scale)
            draw.disc(a, 1.5*scale, color)
        else:
            draw.polyline(segment, color, 1.5*scale)
        label = (max(8*scale, min(context.region.width-22*scale, b[0]+8*scale)),
                 max(26*scale, min(context.region.height-22*scale, b[1]+8*scale)))
        draw.text(label, axis, color, max(10, round(12*scale)))


def paint_proportional(context, state):
    if state is None or len(state) < 7 or not state[6] or context.region_data is None:
        return
    origin, radius = state[6]
    center = Vector(origin)
    xy = raycast.to_screen(context, center)
    right = context.region_data.view_rotation @ Vector((radius, 0, 0))
    edge = raycast.to_screen(context, center+right)
    if xy is not None and edge is not None:
        with draw.Blend():
            draw.ring(xy, (edge-xy).length, palette.secondary(context, .8), 1.0)


def _current(context, tool):
    key, target = _scope(context)
    state = _states.get(key)
    if state is not None and (context.mode != "EDIT_MESH" or state[1] != target or state[2] != tool):
        _states.pop(key, None)
        state = None
    return state


def help_rows(context, tool, wheel_hint=None, details=()):
    """Viewport Help. Full: title, key rows, state rows, operation rows; Compact (default): title and
    state only (keys are in the status bar); Off: nothing."""
    level = getattr(prefs.get(context), "viewport_help", "COMPACT")
    if level == "OFF":
        return []
    full = level == "FULL"
    scene = props.settings(context)
    state = _current(context, tool)
    name = {"ppo.polypen": "Build", "ppo.fill": "Fill",
            "ppo.relax": "Relax / Tweak", "ppo.tweak": "Relax / Tweak"}[tool]
    title = tr(name)
    if tool in {"ppo.relax", "ppo.tweak"}:
        mode = "Temporary Tweak" if state and state[4] == "TEMP_TWEAK" else "Relax" if tool == "ppo.relax" else "Tweak"
        title += " · " + tr(mode)
    title += " · " + tr("3D" if scene.pen3d else "Topo")
    if not full:
        rows = [(title, "title")]
    else:
        title += " | " + _t("W: tools / settings")
        rows = [(title, "title"), *((tr(s), "help") for s in keytable.idle_rows(context, tool))]
        if keytable.mmb_switch(context):   # the MMB / Ctrl+MMB tool switch
            rows.append((tr(keytable.MMB), "muted"))
    if tool == "ppo.polypen":
        # One Build mode: show only its face type, with the tool header's "Faces:" label.
        faces = iface_({"MIXED": "Tri/Quad", "QUAD": "Quad"}[scene.merge3_faces], props.FACES_CONTEXT)
        rows.append((iface_("Faces: {faces}", props.FACES_CONTEXT).format(faces=faces), "muted"))
    elif tool in {"ppo.relax", "ppo.tweak"}:
        brush = scene.relax if tool == "ppo.relax" else scene.tweak
        # the sidebar / wheel names of the Boundary choice: Move / Slide / Lock
        boundary = {"INCLUDE": "Move", "SLIDE": "Slide", "EXCLUDE": "Lock"}[scene.mask_boundary]
        values = keytable.line("Boundary: {boundary} | Radius: {radius} px | Strength: {strength} | Falloff: {falloff}",
                               boundary=tr(boundary), radius=brush.radius,
                               strength=f"{brush.strength:.2f}", falloff=f"{brush.falloff:.2f}")
        rows.append((values if full else keytable.join(values.state), "muted"))   # Compact: Boundary only
    if wheel_hint and full:
        rows.append((wheel_hint, "muted"))
    if state:
        for line in state[3]:
            if not full:   # Compact: the live state only (marked in code); its keys are in the status bar
                line = keytable.join(getattr(line, "state", ()))
            if line:
                rows.append((line, "active"))
    rows.extend(details)
    return rows


@lru_cache(maxsize=192)
def wrap_text(text, width, size):
    """Wrap at operation separators; bound individual long translated clauses."""
    blf.size(0, size)
    lines, current = [], ""
    for part in text.replace("｜", " | ").split(" | "):
        if blf.dimensions(0, part)[0] > width:
            while part and blf.dimensions(0, part + "…")[0] > width:
                part = part[:-1]
            part += "…"
        combined = current + " | " + part if current else part
        if current and blf.dimensions(0, combined)[0] > width:
            lines.append(current)
            current = part
        else:
            current = combined
    return tuple([*lines, current] if current else lines)


def hud_bounds(context):
    scale = max(.5, min(3.0, prefs.ui_scale(context)))
    left, right = 12*scale, context.region.width-12*scale
    for region in context.area.regions:
        if region.type == "TOOLS" and context.space_data.show_region_toolbar:
            left = max(left, region.x+region.width-context.region.x+12*scale)
        elif region.type == "UI" and context.space_data.show_region_ui and region.width:
            edge = region.x-context.region.x-12*scale
            if edge > left:
                right = min(right, edge)
    bottom = 26*scale
    for region in context.area.regions:   # Adjust Last Operation (redo) panel, bottom-left
        if (region.type == "HUD" and region.width > 1 and region.height > 1 and region.x > context.area.x and region.y > context.area.y):   # a hidden redo panel is at (0, 0) with its old size
            if region.x-context.region.x < left+260*scale:
                bottom = max(bottom, region.y-context.region.y+region.height+10*scale)
    return left, right, bottom, max(10, round(12*scale)), 20*scale


def _visible(context, tool):
    if (context.mode != "EDIT_MESH" or context.edit_object is None or tool not in _TOOLS
            or context.area is None or context.area.type != "VIEW_3D"
            or context.region is None or context.region.type != "WINDOW"):
        return False
    from . import custom_pie
    wheel = custom_pie._active
    return wheel is None or wheel.area != context.area


def render(context, rows):
    left, right, bottom, size, step = hud_bounds(context)
    width = int(right-left)
    if width < 60:
        return
    colors = {"title": palette.accent(context), "help": palette.main(context),
              "muted": palette.muted(context), "active": palette.accent(context),   # muted rows
              "info": palette.main(context,.85), "warning": palette.WARNING}
    wrapped = [(part, colors.get(style, colors["info"])) for text, style in rows
               for part in wrap_text(format_label(text), width, size)]
    available = max(0, int((context.region.height-bottom-12)/step))
    wrapped = wrapped[:available]
    if not wrapped:
        return
    with draw.Blend():
        # a translucent dark backdrop behind the block (like Blender's header-text pill), so muted rows
        # stay readable over light meshes
        scale = max(.5, min(3., prefs.ui_scale(context)))
        pad = 6*scale
        blf.size(0, size)
        text_width = max(blf.dimensions(0, text)[0] for text, _color in wrapped)
        top = bottom+(len(wrapped)-1)*step+size
        draw.rounded_rect(left-pad, bottom-pad-.3*size, left+text_width+pad, top+pad, 6*scale, (0., 0., 0., .45))
        for i, (text, color) in enumerate(wrapped):
            y = bottom+(len(wrapped)-1-i)*step
            draw.text((left, y), text, color, size, shadow=True)   # shadowed text




def _draw():
    context = bpy.context
    try:   # the session's safety net for changes without a notify (this handler exists only while in use)
        from .. import session
        session.watch()
    except Exception:
        pass
    if context.region is not None and context.region.type == "WINDOW":
        draw.draw_toast(context)   # the tool-switch toast, in any mode
    tool = active_tool_id(context)
    if context.area and context.region:
        _current(context, tool)  # Drop stale help when switching to native tools/modes.
    if not _visible(context, tool):
        return
    state = _current(context, tool)
    paint_axis(context, state)
    paint_proportional(context, state)
    rows = help_rows(context, tool)
    if rows:
        render(context, rows)


@persistent
def _clear(*_):
    _states.clear()
    wrap_text.cache_clear()


def _show():
    global _handle
    if not bpy.app.background and _handle is None:
        _handle = bpy.types.SpaceView3D.draw_handler_add(_draw, (), "WINDOW", "POST_PIXEL")


def _hide():
    global _handle
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, "WINDOW")
        _handle = None


_watch_handles = []   # (Space type, handle): the session's safety net in every other editor, while in use


def _watch():
    try:
        from .. import session
        session.watch()
    except Exception:
        pass


def _watch_start():
    """the 3D View's HUD handler calls session.watch(); these do it when any other editor draws (the last 3D View
    closed, a workspace without one)."""
    if _watch_handles or bpy.app.background:
        return
    for name in dir(bpy.types):
        space = getattr(bpy.types, name, None)
        if (not name.startswith("Space") or not isinstance(space, type) or space is bpy.types.SpaceView3D
                or not issubclass(space, bpy.types.Space) or space is bpy.types.Space):
            continue
        try:
            _watch_handles.append((space, space.draw_handler_add(_watch, (), "WINDOW", "POST_PIXEL")))
        except (TypeError, ValueError, RuntimeError, AttributeError):
            pass   # an editor without a WINDOW region drawing POST_PIXEL


def _watch_stop():
    while _watch_handles:
        space, handle = _watch_handles.pop()
        try:
            space.draw_handler_remove(handle, "WINDOW")
        except (TypeError, ValueError, RuntimeError):
            pass


def start():
    """while PolyTailor is in use (session): the HUD draw handler and its cache clearing; start clean."""
    _clear()
    _show()
    _watch_start()
    for name in ("load_post", "undo_post", "redo_post"):
        handlers = getattr(bpy.app.handlers, name)
        if _clear not in handlers:
            handlers.append(_clear)


def stop():
    """a toast still fading keeps the draw handler until it has faded (_toast_ended)."""
    _watch_stop()
    for name in ("load_post", "undo_post", "redo_post"):
        handlers = getattr(bpy.app.handlers, name)
        if _clear in handlers:
            handlers.remove(_clear)
    if not draw.toast_showing():
        _hide()


def _toast_ended():
    from .. import session
    if not session._in_use[0]:
        _hide()


def register():
    _clear()
    draw.toast_owner(_show, _toast_ended)   # the toast is drawn by the HUD handler, also when not in use


def unregister():
    draw.toast_owner()
    stop()
    _hide()
    _clear()
    wrap_text.cache_clear()
