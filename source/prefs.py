# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty
from . import about, palette

ADDON_ID = __package__


def _auto_setup_update(_prefs=None, _context=None):
    from . import session
    session.later()   # the setup is applied or restored at once


def _refresh_icons(_prefs=None, _context=None):
    from . import tools
    try:
        tools.request_refresh()   # once after the last change (a colour-picker drag sends many)
    except Exception as exc:
        print("PolyTailor toolbar icons:", exc)


class PPO_Preferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_ID

    viewport_help: EnumProperty(   # keys are always in the status bar
        name="Viewport Help", default="COMPACT", translation_context="Poly Toolset Viewport Help",
        description="Key help in the viewport; keys are always listed in the status bar",
        items=(("FULL", "Full", ""), ("COMPACT", "Compact", ""), ("OFF", "Off", "")))
    # the sidebar's W / Shift W hint: shown while on, until the W wheel was opened 3 times (counted, never drawn)
    show_wheel_hint: BoolProperty(translation_context="Poly Toolset", name="Show Wheel Hint", default=True,
                                  update=lambda self, _context: setattr(self, "wheel_open_count", 0) if self.show_wheel_hint else None)
    wheel_open_count: IntProperty(default=0, min=0, options={"HIDDEN"})
    stroke_xform_sensitivity: FloatProperty(translation_context="Poly Toolset",
        name="Stroke Transform Sensitivity",
        description="How fast Ctrl+move rotates / scales a stroke or strip preview (Shift: x0.1 finer)",
        default=0.6, min=0.1, max=1.5, step=5, precision=2,
    )
    pick_radius: IntProperty(translation_context="Poly Toolset", 
        name="Pick Radius",
        description="Screen distance in pixels for picking vertices and edges",
        default=12, min=2, max=60, subtype="PIXEL",
    )
    snap_distance: IntProperty(translation_context="Poly Toolset", 
        name="Snap Distance",
        description="Screen distance in pixels for Vertex / Edge / Edge Center snapping "
                    "(Blender uses 30)",
        default=30, min=5, max=100, subtype="PIXEL",
    )
    snap_vertex_limit: IntProperty(translation_context="Poly Toolset", 
        name="Vertex Snap Limit",
        description="Objects with more vertices than this (e.g. 3D scans) are not used for "
                    "Vertex / Edge / Edge Center snapping, which would stall while indexing them. "
                    "Surface snapping still uses them",
        default=300_000, min=1000, soft_max=5_000_000,
    )
    auto_setup: BoolProperty(translation_context="Poly Toolset", name="Automatic Retopology Setup", default=True,
        update=_auto_setup_update,
        description="Set up surface snapping and the overlay while a PolyTailor tool is active; restore on exit")
    dense_feedback: BoolProperty(translation_context="Poly Toolset", name="Dense Relax Feedback", default=False,
        description="Experimental local display for dense 3D Relax in Solid retopology views; full native display synchronizes on release")
    fill_search_mode: EnumProperty(translation_context="Poly Toolset", name="Search", default='FAST', description="How widely Fill searches",
        items=(('FAST', 'Fast search', 'Try the most likely layouts first; search fully only if the result is poor. '
                'Odd outlines always search fully'),
               ('FULL', 'Full search', 'Try every layout every time (slower)')))
    show_debug_info: BoolProperty(translation_context="Poly Toolset", name="Show Debug Info", default=False,
        description="Show the internal quality numbers of each fill choice and keep a debug log "
                    "(newfill_log.jsonl) in the add-on's user folder")
    accent: FloatVectorProperty(update=_refresh_icons, translation_context="Poly Toolset", 
        name="Accent Color",
        subtype="COLOR", size=4, min=0.0, max=1.0,
        default=palette.DARK,
    )
    mmb_tool_switch: BoolProperty(translation_context="Poly Toolset", name="MMB Tool Switch", default=True,
        description="Click MMB: Build \u2194 Brush; Ctrl+MMB: Build \u2194 Fill. Drags still orbit and zoom")
    preview_fill: FloatVectorProperty(translation_context="Poly Toolset", name="Preview Fill", subtype="COLOR", size=4,
        min=0.0, max=1.0, default=palette.PREVIEW_FILL)
    preview_border: FloatVectorProperty(translation_context="Poly Toolset", name="Preview Border", subtype="COLOR", size=4,
        min=0.0, max=1.0, default=palette.PREVIEW_BORDER)
    icon_colours: EnumProperty(update=_refresh_icons, translation_context="Poly Toolset", name="Icon Colors", default="CUSTOM",
        items=(("CUSTOM", "Custom", "PolyTailor's own colors, set below; the theme does not change them"),
               ("NATIVE", "Native", "Follow the Blender theme like its own icons: the theme's text color, "
                "the Accent Color, and the theme's icon saturation, alpha and border")),
        description="Colors of the tool icons in the W wheel")
    icon_main: FloatVectorProperty(update=_refresh_icons, translation_context="Poly Toolset", name="Icon Lines", subtype="COLOR", size=4, min=0.0, max=1.0,
        default=tuple(c / 255 for c in palette.ICON_MAIN))
    icon_dim: FloatVectorProperty(update=_refresh_icons, translation_context="Poly Toolset", name="Icon Secondary", subtype="COLOR", size=4, min=0.0, max=1.0,
        default=tuple(c / 255 for c in palette.ICON_DIM))
    icon_accent: FloatVectorProperty(update=_refresh_icons, translation_context="Poly Toolset", name="Icon Accent", subtype="COLOR", size=4, min=0.0, max=1.0,
        default=palette.DARK)

    def draw(self, context):
        from .i18n import CONTEXT
        layout = self.layout
        layout.label(text=f"{about.NAME} · {about.version_label()}")

        def group(title):
            box = layout.box().column()
            box.label(text=title, text_ctxt=CONTEXT)
            box.use_property_split = True
            return box

        box = group("Input")
        box.prop(self, "pick_radius")
        box.prop(self, "stroke_xform_sensitivity")
        box.prop(self, "snap_distance")
        box.prop(self, "snap_vertex_limit")
        box.prop(self, "auto_setup")
        box = group("Fill")
        box.prop(self, "fill_search_mode")   # the Fill option everyone sees (debug info below: Developer Extras)
        if context.preferences.view.show_developer_ui:   # Blender's convention for debug options
            box.prop(self, "show_debug_info")
        box = group("Display")
        if hasattr(self, "viewport_help"):
            box.prop(self, "viewport_help")
        box.prop(self, "show_wheel_hint")
        if context.preferences.view.show_developer_ui:   # developer option
            box.prop(self, "dense_feedback")
        box = group("Colors")
        box.prop(self, "accent")
        box.prop(self, "preview_fill")
        box.prop(self, "preview_border")
        box.prop(self, "icon_colours")
        if self.icon_colours == "CUSTOM":
            box.prop(self, "icon_main")
            box.prop(self, "icon_dim")
            box.prop(self, "icon_accent")
        _draw_keymap(self, context, layout.box().column())


_HIDDEN_KMI = {"ppo.brush_hover"}   # pointer tracking, not a key

# Distinct labels for the two MMB switch rows (set to None to use the key tables'
# existing text: Brush/Build, Fill/Build).
SWITCH_LABELS = {"BRUSH": "Switch Tool: Brush", "FILL": "Switch Tool: Fill"}

# the compact list: (key as written in the key tables, tool keymap (None = the add-on's W keymaps), operator, properties)
COMPACT_KEYS = (
    ("W", None, "ppo.call_pie", {}),
    ("Shift+W", None, "wm.call_panel", {"name": "PPO_PT_quick_all"}),
    ("MMB", "Build", "ppo.mode_toggle", {"kind": "BRUSH"}),
    ("Ctrl+MMB", "Build", "ppo.mode_toggle", {"kind": "FILL"}),
    ("F", "Relax", "ppo.brush_adjust", {"mode": "RADIUS"}),
    ("Shift+F", "Relax", "ppo.brush_adjust", {"mode": "STRENGTH"}),
    ("Ctrl+F", "Relax", "ppo.brush_adjust", {"mode": "FALLOFF"}),
    ("G", "Build", "ppo.move", {"extrude": False}),
    ("E", "Build", "ppo.move", {"extrude": True}),
)


def _key_label(key):
    """The action text of `key` in the key tables (existing, translated text), e.g. F -> Radius."""
    import re
    from .i18n import tr
    try:
        from .ui import keytable as kt
        rows = [*kt.GLOBAL, kt.MMB, *(r for rs in kt.IDLE.values() for r in rs)]
    except (ImportError, AttributeError):
        return None
    for row in rows:
        en = kt.parts(row)
        for i, part in enumerate(en):
            if part.split(":", 1)[0].strip() == key:
                shown = kt.parts(tr(row))
                text = shown[i] if len(shown) == len(en) else part
                action = re.split(r"[:\uff1a]", text, 1)[-1].strip()
                return action[:1].upper() + action[1:]
    return None


def _tool_keymap(kc, label):
    from . import tools
    tool = next((t for t in tools.TOOLS if t.bl_label.strip() == label), None)
    keymap = getattr(getattr(tool, "_bl_tool", None), "keymap", None)
    name = keymap[0] if keymap and isinstance(keymap[0], str) else f"3D View Tool: Edit Mesh, {label}"
    return kc.keymaps.get(name)


def _compact_rows(kc):
    """[(label, keymap, item)] of the user-facing bindings, one row each, in COMPACT_KEYS order."""
    from .ui import pie
    from .i18n import tr
    out = []
    for key, tool, idname, props in COMPACT_KEYS:
        kms = ([kc.keymaps.get(name) for name, _idname in pie._keymaps] if tool is None
               else [_tool_keymap(kc, tool)])
        found = next(((km, kmi) for km in kms if km for kmi in km.keymap_items if kmi.idname == idname
                      and all(getattr(kmi.properties, k, None) == v for k, v in props.items())), None)
        if found is None:
            continue
        km, kmi = found
        kind = props.get("kind")
        label = (tr(SWITCH_LABELS[kind]) if SWITCH_LABELS and kind in SWITCH_LABELS
                 else _key_label(key) or kmi.name)
        out.append((label, km, kmi))
    return out


def _keymap_items(kc):
    """(title, keymap, items) of the add-on's W / Shift+W and of each tool keymap, from keyconfig kc."""
    from .ui import pie
    from . import tools
    out = []
    for name, idname in pie._keymaps:
        km = kc.keymaps.get(name)
        items = [kmi for kmi in (km.keymap_items if km else ()) if kmi.idname == idname and (
            kmi.idname != "wm.call_panel" or getattr(kmi.properties, "name", "") == "PPO_PT_quick_all")]
        if items:
            out.append((None, km, items[:1]))
    for tool in tools.TOOLS:
        keymap = getattr(getattr(tool, "_bl_tool", None), "keymap", None)
        name = keymap[0] if keymap and isinstance(keymap[0], str) else f"3D View Tool: Edit Mesh, {tool.bl_label}"
        km = kc.keymaps.get(name)
        items = [kmi for kmi in (km.keymap_items if km else ())   # no RIGHT_CTRL twin, no pointer helper
                 if kmi.type not in {"MOUSEMOVE", "RIGHT_CTRL"} and kmi.idname not in _HIDDEN_KMI]
        if items:
            out.append((tool.bl_label.strip(), km, items))
    return out


def _draw_keymap(prefs, context, col):
    """The add-on's keys, remappable with Blender's own keymap rows, and the MMB Tool Switch option."""
    from .i18n import CONTEXT
    col.label(text="Keymap", text_ctxt="Poly Toolset")   # Blender's own word and translation
    sub = col.column()
    sub.use_property_split = True
    sub.prop(prefs, "mmb_tool_switch")
    kc = context.window_manager.keyconfigs.user
    if kc is None:
        return
    try:
        compact = _compact_rows(kc)
        shown = set()
        for label, km, kmi in compact:   # the user-facing keys, one compact row each
            shown.add((km.name, kmi.id))
            row = col.split(factor=.4)
            row.label(text=label, translate=False)
            row.label(text=kmi.to_string(), translate=False)
        header, body = col.panel("PPO_PT_keymap_tools", default_closed=True)   # everything else, closed
        header.label(text="Tools")   # Blender's own word and translation
        if body:
            for title, km, items in _keymap_items(kc):
                items = [kmi for kmi in items if (km.name, kmi.id) not in shown]
                if not title or not items:
                    continue
                body.label(text=title, text_ctxt=CONTEXT)
                for kmi in items:
                    row = body.split(factor=.4)
                    row.label(text=kmi.name)   # translated as in Blender's own keymap rows
                    row.label(text=kmi.to_string(), translate=False)
    except Exception as exc:   # the preferences must always draw
        col.label(text=str(exc), icon="ERROR")


class _Defaults:
    """Stand-in when the add-on preferences are unavailable."""
    pick_radius = 12
    stroke_xform_sensitivity = 0.6
    snap_distance = 30
    snap_vertex_limit = 300_000
    auto_setup = True
    dense_feedback = False
    show_debug_info = False
    fill_search_mode = 'FAST'
    mmb_tool_switch = True
    accent = palette.DARK
    preview_fill = palette.PREVIEW_FILL
    preview_border = palette.PREVIEW_BORDER
    icon_colours = "CUSTOM"
    icon_main = tuple(c / 255 for c in palette.ICON_MAIN)
    icon_dim = tuple(c / 255 for c in palette.ICON_DIM)
    icon_accent = palette.DARK


def get(context=None):
    context = context or bpy.context
    addon = context.preferences.addons.get(ADDON_ID)
    return addon.preferences if addon and addon.preferences else _Defaults


def ui_scale(context=None):
    context = context or bpy.context
    return context.preferences.system.ui_scale


def _prepare_auto_partition():
    from .core import fill_progressive, fill_field, network_validate  # noqa: F401  (pre-import before tool input)
    fill_progressive.prepare()   # auto partition is always on


def register():
    bpy.utils.register_class(PPO_Preferences)
    prefs = get()
    if prefs is not _Defaults:
        _prepare_auto_partition()


def unregister():
    bpy.utils.unregister_class(PPO_Preferences)
