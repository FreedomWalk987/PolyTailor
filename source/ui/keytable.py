# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""the key tables, the single source of truth for key help.

Each row is an existing English hint string ("key: action | key: action"), translated
when shown. The status bar lists them (idle tools: session.update_status; running
modals: ops.common._status_update), the viewport HUD shows them only with Viewport
Help = Full, and the Shortcuts panel is generated from them (ui/shortcuts.py).
"""
import re

from .. import prefs
from ..i18n import tr

TOOLS = ("ppo.polypen", "ppo.fill", "ppo.relax", "ppo.tweak")
TOOL_NAMES = {"ppo.polypen": "Build", "ppo.fill": "Fill", "ppo.relax": "Relax", "ppo.tweak": "Tweak"}

IDLE = {
    "ppo.polypen": (
        "Hold Ctrl: preview | Ctrl+LMB: add | Drag: place new geometry",
        "LMB drag: move, or box select from empty space | G: move | E: extrude",
        "Alt+Wheel: relax"),
    "ppo.fill": (
        "Hold Ctrl: preview the selected outline / hovered hole | Ctrl+LMB: fill",
        "LMB drag: move, or box select from empty space | G: move | E: extrude",
        "Alt+Wheel: relax"),   # the wheel key appears in the Ctrl-held row
    "ppo.relax": (
        "Hold LMB: Relax | Hold Alt+LMB: Grid Relax | Hold Ctrl and drag: temporary Tweak",
        "F: radius | Shift+F: strength | Ctrl+F: falloff"),
    "ppo.tweak": (
        "LMB drag: move vertices under the brush",
        "F: radius | Shift+F: strength | Ctrl+F: falloff"),
}
MMB = "MMB: Brush / Build | Ctrl+MMB: Fill / Build"   # the click tool switch; " / " spacing
# Optional rows: set either to None to drop it.
SHIFT_W = "Shift+W: all settings"
ALL_TOOLS = "All Tools"   # Shortcuts group heading (None: the rows follow the last tool group)
GLOBAL = tuple(row for row in ("W: tools / settings", SHIFT_W) if row)
# Running operations: their keys come from the HUD operation rows (ui.hud.operation_lines);
# a row without keys falls back to the tool's first idle row.

_SPLIT = re.compile(r"\s*[|｜]\s*")
# which parts of an operation row are live state (shown by Compact), by the English source row and
# part index; every other part is a key (status bar). "ALL": the whole row is state. Rows not listed: keys only.
ALL = "ALL"
STATE_PARTS = {
    "Preview: {action}": ALL,
    "Global axis: {axis}": ALL,
    "Temporary Tweak | Release Ctrl: resume Relax | Uses the Relax brush settings": (0, 2),
    "{label}: {value} | LMB / Enter: confirm | Esc / RMB: cancel": (0,),
    "Boundary offset | LMB / Enter: confirm | Esc / RMB: cancel": (0,),
    "Slide along edges | LMB / Enter: confirm | Esc / RMB: cancel": (0,),
    "Proportional editing: {falloff} | Wheel: influence radius": (0,),
    "Boundary: {boundary} | Radius: {radius} px | Strength: {strength} | Falloff: {falloff}": (0,),
}


class Line(str):
    """A translated row that carries its key parts and its state parts (marked in code, never parsed)."""
    keys = ()
    state = ()


def marked(text, state=False):
    """A runtime row (e.g. Fill's hint / debug status): all keys, or all state."""
    out = Line(text)
    out.keys, out.state = ((), (text,)) if state else (tuple(parts(text)), ())
    return out


def line(en, state=None, **values):
    """Translate the English row en (formatted with values) and mark its parts by STATE_PARTS (or state)."""
    text = tr(en).format(**values) if values else tr(en)
    marks = STATE_PARTS.get(en, ()) if state is None else state
    if marks == ALL:
        return marked(text, state=True)
    out = Line(text)
    split = parts(text)
    if len(split) == len(parts(en)):
        out.state = tuple(p for i, p in enumerate(split) if i in marks)
        out.keys = tuple(p for i, p in enumerate(split) if i not in marks)
    else:   # a translation that joins parts differently: never show a fragment as state
        out.keys = tuple(split)
    return out


def mmb_switch(context):
    """The MMB / Ctrl+MMB click switch is on."""
    return getattr(prefs.get(context), "mmb_tool_switch", True)


def idle_rows(context, tool):
    """English rows of an idle tool (without the MMB / W rows)."""
    return IDLE.get(tool, ())


def parts(text):
    return [part for part in _SPLIT.split(text) if part]


def join(items):
    items = list(items)
    zh = any(ord(ch) > 0x2e80 for item in items for ch in item)
    return ("｜" if zh else " | ").join(items)


def status_rows(context, tool):
    """English rows the status bar lists for an idle tool."""
    rows = list(idle_rows(context, tool))
    if mmb_switch(context):
        rows.append(MMB)
    rows.extend(GLOBAL)
    return rows


def status_text(context, tool):
    """The status-bar text of an idle PolyTailor tool."""
    return join(part for row in status_rows(context, tool) for part in parts(tr(row)))


def modal_status(op, context):
    """The status-bar text of a running PolyTailor modal: the keys of its HUD operation rows."""
    from . import hud
    name = hud.operator_name(op)
    lines, _kind = hud.operation_lines(op, context, name)
    keys = [part for row in lines for part in getattr(row, "keys", ())]   # marked in code
    if not keys and name in IDLE:
        rows = idle_rows(context, name)
        keys = parts(tr(rows[0])) if rows else []
    return join(keys)


def groups(context=None):
    """(heading, translated rows) per group for the Shortcuts panel: every tool, then the keys of every tool."""
    out = [(tr(TOOL_NAMES[tool]), [tr(row) for row in IDLE[tool]]) for tool in TOOLS]
    mmb = [tr(MMB)] if getattr(prefs.get(context), "mmb_tool_switch", True) else []
    rows = [*mmb, *(tr(row) for row in GLOBAL)]
    if ALL_TOOLS:
        out.append((tr(ALL_TOOLS), rows))
    else:
        out[-1][1].extend(rows)
    return out
