# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Tool/settings descriptions and the W entry point for the persistent wheel."""

from pathlib import Path

import bpy
import bpy.utils.previews

from .. import props, session, palette
from ..ops.common import active_tool_id
from ..i18n import CONTEXT

ICONS = Path(__file__).parent.parent / "icons"
_previews = None
_keymaps = []
TOOL_ICONS = ("polypen", "fill", "tweak", "relax",
              "pin", "unpin", "glue", "unglue")


def light_theme(context=None):
    """Pick the artwork contrast from the actual widget background."""
    return palette.light_theme(context)


_tint = dict(sig=None, ids={})


def _tint_pixels(name, cols, size=32):
    """the tool's vector geometry recoloured (229 grey -> main, 130 grey -> secondary, orange -> accent),
    rasterised to straight-alpha float pixels, row 0 at the bottom (the preview / ImBuf order)."""
    import numpy as np
    from . import custom_pie
    xy, rgba = custom_pie._read_vco(ICONS / "vector" / f"{name}.dat")
    src = np.round(rgba[:, :3] * 255).astype(int)
    out = rgba.copy()
    for old, c in (((229, 229, 229), cols[0]), ((130, 130, 130), cols[1]), ((255, 158, 52), cols[2])):
        out[np.all(src == old, axis=1)] = np.asarray(c, float) / 255
    n = size * 4
    try:
        image = custom_pie._raster_gpu(xy, out, n)
    except Exception:
        image = custom_pie._raster_cpu(xy, out, n)
    image = image.reshape(size, 4, size, 4, 4).mean((1, 3))
    alpha = image[..., 3:]
    image[..., :3] = np.divide(image[..., :3], alpha, out=np.zeros_like(image[..., :3]), where=alpha > 0)
    return image.astype(np.float32).ravel().tolist()


def refresh_tint(cols):
    """panel / menu icons in the current icon colours; None = the shipped PNGs."""
    if cols == _tint["sig"] or _previews is None:
        return
    import hashlib
    ids = {}
    if cols is not None:
        tag = hashlib.md5(repr(cols).encode()).hexdigest()[:10]
        for name in TOOL_ICONS:
            key = f"{name}#t{tag}"
            try:
                if key not in _previews:
                    pixels = _tint_pixels(name, cols)
                    prev = _previews.new(key)
                    prev.icon_size = (32, 32); prev.icon_pixels_float = pixels
                    prev.image_size = (32, 32); prev.image_pixels_float = pixels
                ids[name] = _previews[key].icon_id
            except Exception as exc:
                print("PolyTailor icon tint:", name, exc)
    _tint.update(sig=cols, ids=ids)


def icon(name):
    tinted = _tint["ids"].get(name)
    if tinted:
        return tinted
    key = name + ("_light" if light_theme() else "")
    return _previews[key].icon_id if _previews and key in _previews else 0


class PPO_OT_set_mode(bpy.types.Operator):
    bl_idname = "ppo.set_mode"
    bl_label = "Modeling Mode"
    bl_translation_context = "Poly Toolset"
    bl_description = "Switch 3D/Topo; Topo from a Blender tool switches to Build"
    bl_options = {"INTERNAL"}

    mode: bpy.props.EnumProperty(translation_context="Poly Toolset", items=(("PEN3D", "3D", ""), ("TOPO", "Topo", "")))

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH"

    def execute(self, context):
        if self.mode == "TOPO" and not (active_tool_id(context) or "").startswith("ppo."):
            # Retopology happens in PolyTailor: enter Build, so its session takes the
            # scene over and the Topo settings (snapping, Stop at Edge) are restored.
            bpy.ops.wm.tool_set_by_id(name="ppo.polypen")
        session.activate_mode(context)
        # Keep the existing RNA setter: it owns the snapping/merge backup.
        props.settings(context).pen_mode = self.mode
        if context.scene.session_uid in session._scenes:
            if self.mode == "TOPO":
                session.apply_topo_defaults(context, reset_view=True)
            else:
                props.begin_pen3d(context.scene)
        session.apply_now()
        return {"FINISHED"}


def draw_modes(layout, scene):
    row = layout.row(align=True)
    for value, label in (("PEN3D", "3D"), ("TOPO", "Topo")):
        row.operator("ppo.set_mode", text=label, text_ctxt="Poly Toolset", depress=scene.pen_mode == value).mode = value


class PPO_OT_switch_tool(bpy.types.Operator):
    bl_idname = "ppo.switch_tool"
    bl_label = "Switch Tool"
    bl_translation_context = "Poly Toolset"
    bl_options = {"INTERNAL"}

    tool: bpy.props.StringProperty()

    @classmethod
    def description(cls, context, properties):
        from ..i18n import tr   # the tool's existing name instead of "Switch Tool" on every icon
        name = {"ppo.polypen": "Build", "ppo.fill": "Fill", "ppo.tweak": "Tweak", "ppo.relax": "Relax"}.get(properties.tool)
        return tr(name) if name else ""

    def execute(self, context):
        tool = self.tool
        if tool.startswith("ppo.") and tool not in AVAILABLE_TOOLS:
            tool = "ppo.polypen"  # Saved retired tool IDs fall back without touching scene data.
        bpy.ops.wm.tool_set_by_id(name=tool)
        session.apply_now()   # the new tool's look and selection at once
        return {"FINISHED"}


class PPO_OT_call_pie(bpy.types.Operator):
    bl_idname = "ppo.call_pie"
    bl_label = "PolyTailor Wheel"
    bl_translation_context = "Poly Toolset"
    bl_description = "Open the PolyTailor wheel: tools and settings"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return (context.mode == "EDIT_MESH"
                and context.area is not None and context.area.type == "VIEW_3D"
                and context.region is not None and context.region.type == "WINDOW")

    def invoke(self, context, event):
        # The custom wheel owns settings and nested menus until explicit dismissal.
        bpy.ops.ppo.persistent_pie("INVOKE_DEFAULT")
        return {"FINISHED"}


PIE_BOX_UNITS = 11   # N (snapping + pins) and S (tool settings) boxes share one width


def _section(pie, title, tool_id=None, current=None):
    back = pie.box().column(align=True)
    if tool_id is None:
        back.label(text=title, text_ctxt=CONTEXT)
    else:
        back.operator("ppo.switch_tool", text=title, text_ctxt=CONTEXT, depress=current == tool_id).tool = tool_id
    return back.box().column()


def draw_build_settings(context, layout, maximum=False):
    """Build settings: one layout for the sidebar, the W wheel, Shift+W and the tool panel."""
    scene = props.settings(context)
    layout.row(align=True).prop(scene, "merge3_faces", expand=True)
    if scene.merge3_faces == "QUAD" or maximum:
        layout.prop(scene, "quad_stability", slider=True)
    row = layout.row(align=True)
    row.prop(scene, "knife_constrain")
    row.prop(scene, "knife_junctions")
    stroke = layout.box().column(align=True)
    stroke.label(text="Stroke", text_ctxt=CONTEXT)
    stroke.prop(scene, "build_stroke_density")
    stroke.prop(scene, "build_stroke_corner_angle")
    from .. import prefs
    p = getattr(prefs.get(context), "_prefs", prefs.get(context))   # the real preferences struct for layout.prop
    if hasattr(p, "bl_rna"):
        stroke.prop(p, "stroke_xform_sensitivity")
    from ..tools import draw_smoothing
    draw_smoothing(context, layout.box().column(align=True), maximum=maximum)


def draw_masking(context, layout):
    """Brush masking: one layout for the sidebar, the W wheel and Shift+W."""
    scene = props.settings(context)
    for label, name in (("Selected", "mask_selected"), ("Boundary", "mask_boundary")):
        row = layout.row(align=True)
        row.label(text=label, text_ctxt="Poly Toolset")
        row.prop(scene, name, expand=True, icon_only=True)
    layout.prop(scene, "include_occluded")
    for first, second in (("include_corners", "include_seams"), ("include_sharps", "include_creases")):
        row = layout.row(align=True)
        row.prop(scene, first)
        row.prop(scene, second)


def _settings_section(context, pie, current, maximum=False):
    """Reserve the tool-settings section now; return a function that fills it later (PPO_PT_quick_all.draw)."""
    scene = props.settings(context)
    if current == "ppo.polypen":
        sec = _section(pie, "Build")

        def fill():
            draw_build_settings(context, sec, maximum=maximum)
    elif current in ("ppo.tweak", "ppo.relax"):
        row = pie.row()
        row.ui_units_x = PIE_BOX_UNITS * 2 + 1
        sections = []
        for tool_id, title, key in (("ppo.relax", "Relax", "relax"),
                                    ("ppo.tweak", "Tweak", "tweak")):
            sec = _section(row, title, tool_id=tool_id, current=current)
            sec.ui_units_x = PIE_BOX_UNITS
            sections.append((tool_id, sec, getattr(scene, key)))

        def fill():
            for tool_id, sec, brush in sections:
                col = sec.column()
                col.prop(brush, "radius")
                col.prop(brush, "strength", slider=True)
                col.prop(brush, "falloff", slider=True)
                reset = col.operator("ppo.brush_reset", text="Reset Brush", text_ctxt="Poly Toolset", icon="LOOP_BACK")
                reset.tool = "RELAX" if tool_id == "ppo.relax" else "TWEAK"
                draw_masking(context, col)
                if tool_id == "ppo.relax":
                    from .panels import draw_relax_algorithm
                    algorithm = col.box().column(align=True)
                    algorithm.label(text="Relax Algorithm", text_ctxt="Poly Toolset")
                    draw_relax_algorithm(context, algorithm)
    elif current == "ppo.fill":
        sec = _section(pie, "Fill")

        def fill():
            # no Fill Pattern dropdown (Fill always uses AUTO)
            from ..tools import draw_smoothing
            draw_smoothing(context, sec.box().column(align=True), maximum=maximum)
    else:
        pie.separator()
        fill = None
    return fill


def draw_selection(context, layout):
    """Selection is independent from snapping and constraint actions."""
    ts = context.scene.tool_settings
    mode_sec = _section(layout, "Select Mode")
    # Explicit labels in both native panels and the custom wheel. The native
    # select-mode operator and the wheel keep their existing Shift extension.
    row = mode_sec.row(align=True)
    modes = tuple(getattr(ts, "mesh_select_mode", (False, False, False)))
    for index, (kind, glyph, label) in enumerate((("VERT", "VERTEXSEL", "Vertex"),
                                               ("EDGE", "EDGESEL", "Edge"),
                                               ("FACE", "FACESEL", "Face"))):
        row.operator("mesh.select_mode", text=label, text_ctxt="Poly Toolset Select Mode",
                     icon=glyph, depress=modes[index]).type = kind
    # Merge and Retopology shading side by side: one row more would crowd the box
    row = mode_sec.row(align=True)
    row.prop(ts, "use_mesh_automerge", text="Auto Merge", text_ctxt="Poly Toolset", toggle=True)
    overlay = getattr(getattr(context, "space_data", None), "overlay", None)
    if overlay is not None and hasattr(overlay, "show_retopology"):
        row.prop(overlay, "show_retopology", text="Retopo Shading", text_ctxt="Poly Toolset", toggle=True)
    mode_sec.prop(ts, "use_transform_correct_face_attributes", text="Correct Face Attributes",
                  toggle=True, text_ctxt="Poly Toolset Controls")


def draw_snapping(context, layout):
    ts = context.scene.tool_settings
    scene = props.settings(context)
    snap_sec = _section(layout, "Snapping")
    row = snap_sec.row(align=True)
    row.prop(ts, "use_snap", text="Snap", text_ctxt="Poly Toolset", toggle=True)   # RNA supplies the on/off icon
    row.prop(scene, "clamp_to_source", text="Stop at Edge", text_ctxt="Poly Toolset", toggle=True,
             icon="CLIPUV_DEHLT" if scene.clamp_to_source else "CLIPUV_HLT")
    row = snap_sec.row(align=True)
    row.active = ts.use_snap
    row.prop(ts, "snap_elements_individual", expand=True)
    row = snap_sec.row(align=True)
    draw_modes(row, scene)
    sub = row.row(align=True)
    sub.active = ts.use_snap
    sub.popover("VIEW3D_PT_snapping", text="Targets", text_ctxt=CONTEXT)


def draw_constraints(context, layout):
    pin_sec = _section(layout, "Constraint Tools")
    row = pin_sec.row()
    pins = row.row(align=True)
    draw_pin_glue(pins, row.row(align=True))


def draw_pin_glue(pins, glue):
    """Pin / Off and Glue / Off with the add-on terms (sidebar and wheel)."""
    pins.operator("ppo.pin", text="Pin", text_ctxt=CONTEXT, icon_value=icon("pin")).unpin = False
    pins.operator("ppo.pin", text="Off", text_ctxt=CONTEXT, icon_value=icon("unpin")).unpin = True
    glue.operator("ppo.glue", text="Glue", text_ctxt=CONTEXT, icon_value=icon("glue")).unglue = False
    glue.operator("ppo.glue", text="Off", text_ctxt=CONTEXT, icon_value=icon("unglue")).unglue = True


def draw_snap_pin(context, layout):
    """Combined settings entry retained for Shift+W."""
    draw_constraints(context, layout)
    draw_snapping(context, layout)
    draw_selection(context, layout)


class _QuickPanel:
    # Opened with wm.call_panel(keep_open=True): unlike a pie, it stays up while the
    # cursor is over it, so several settings can be changed in a row.
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    bl_ui_units_x = PIE_BOX_UNITS + 3   # at least the wheel width: no cut-off labels

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH"


class PPO_PT_quick_all(_QuickPanel, bpy.types.Panel):
    bl_label = "PolyTailor Settings"
    bl_translation_context = "Poly Toolset"

    def draw(self, context):
        current = active_tool_id(context)
        self.layout.ui_units_x = PIE_BOX_UNITS * 2 + 1 if current in {"ppo.relax", "ppo.tweak"} else PIE_BOX_UNITS + 3
        draw_snap_pin(context, self.layout)
        fill_slot = _settings_section(context, self.layout, current)
        if fill_slot is not None:
            fill_slot()


AVAILABLE_TOOLS = set()  # filled by tools.register()


def register():
    global _previews
    _previews = bpy.utils.previews.new()
    for name in TOOL_ICONS:
        for suffix in ("", "_light"):
            key = name + suffix
            path = ICONS / f"{key}-icon.png"
            if path.exists():
                _previews.load(key, str(path), "IMAGE")
    bpy.utils.register_class(PPO_OT_set_mode)
    bpy.utils.register_class(PPO_OT_switch_tool)
    bpy.utils.register_class(PPO_PT_quick_all)
    bpy.utils.register_class(PPO_OT_call_pie)

    # W and Shift+W in the Mesh keymap: Mesh Edit Mode only, with any tool. Shift+W: every setting at once, in a
    # panel that stays open; it wins over Blender's Shift+W Bend there (Bend stays in its menu).
    keyconfig = bpy.context.window_manager.keyconfigs.addon
    if keyconfig:
        km = keyconfig.keymaps.new(name="Mesh", space_type="EMPTY")
        kmi = km.keymap_items.new("ppo.call_pie", "W", "PRESS",
                                 ctrl=False, shift=False, alt=False)
        _keymaps.append((km, kmi))
        kmi = km.keymap_items.new("wm.call_panel", "W", "PRESS", shift=True)
        kmi.properties.name, kmi.properties.keep_open = "PPO_PT_quick_all", True
        _keymaps.append((km, kmi))


def unregister():
    global _previews
    for km, kmi in _keymaps:
        km.keymap_items.remove(kmi)
    _keymaps.clear()
    bpy.utils.unregister_class(PPO_OT_call_pie)
    bpy.utils.unregister_class(PPO_PT_quick_all)
    bpy.utils.unregister_class(PPO_OT_switch_tool)
    bpy.utils.unregister_class(PPO_OT_set_mode)
    bpy.utils.previews.remove(_previews)
    _previews = None
    _tint.update(sig=None, ids={})
