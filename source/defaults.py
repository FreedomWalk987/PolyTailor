# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""User defaults: "Save as Default" / "Restore Factory Defaults" (sidebar > Defaults).

Saved per user (not per .blend) in the extension's user folder as defaults.json:

  scene    PolyTailor settings of the scene (tool settings, brushes, masks, snapping options),
           except the per-file source object / collection.
  session  what PolyTailor sets in Blender while a PolyTailor tool is active: snapping on/off, the
           snap targets, and Auto Merge in Build / in the other PolyTailor tools.

The session values are used every time PolyTailor starts in a scene. The scene values are
copied once into every scene PolyTailor starts in for the first time (new files, new scenes),
so a file keeps the settings it was saved with.
"""

import json
import os
from pathlib import Path

import bpy
from .i18n import tr   # reports in the add-on context

FILE_NAME = "defaults.json"
SKIP_SCENE = {"pen_mode", "rna_type", "name", "source_object", "source_collection", "defaults_applied", "face_attributes_initialized", "face_attributes_default_off",
              "merge_manual_3d", "merge_manual_topo", "retopo_manual_3d", "retopo_manual_topo",
              "pen3d_prev_snap", "pen3d_prev_merge", "pen3d_prev_clamp", "pen3d_backup_ready"}

# PolyTailor's own session setup (also what Restore Factory Defaults goes back to)
FACTORY_SESSION = {
    "use_snap": True,
    "snap_elements": ["FACE_NEAREST"],
    "automerge_polypen": True,
    "automerge_other": True,     # Topo merges with every tool (3D always turns merging off)
}


# --- file ---------------------------------------------------------------------------

def _path():
    try:
        folder = bpy.utils.extension_path_user(__package__, create=True)
    except (ValueError, AttributeError, OSError):
        return None
    return Path(folder) / FILE_NAME


def load():
    path = _path()
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path and path.exists() else {}
    except (OSError, ValueError):
        data = {}
    return data if isinstance(data, dict) else {}


def has_saved():
    return bool(load())


def _write(data):
    path = _path()
    if path is None:
        return False
    tmp = path.with_name(path.name + ".tmp")
    try:   # written in full first, then moved over the old file: a failed write keeps the old defaults
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return False
    return True


# --- property groups <-> plain values ------------------------------------------------

def _group_to_dict(group):
    out = {}
    for prop in group.bl_rna.properties:
        key = prop.identifier
        if key in SKIP_SCENE or prop.is_readonly and prop.type != "POINTER":
            continue
        value = getattr(group, key)
        if prop.type == "POINTER":
            if isinstance(value, bpy.types.PropertyGroup):
                out[key] = _group_to_dict(value)
            continue
        if prop.type == "ENUM" and prop.is_enum_flag:
            value = sorted(value)
        elif hasattr(value, "__len__") and not isinstance(value, str):
            value = list(value)
        out[key] = value
    return out


def _dict_to_group(group, values):
    for key, value in values.items():
        if key in SKIP_SCENE or key not in group.bl_rna.properties:
            continue
        prop = group.bl_rna.properties[key]
        try:
            if prop.type == "POINTER":
                sub = getattr(group, key)
                if isinstance(sub, bpy.types.PropertyGroup) and isinstance(value, dict):
                    _dict_to_group(sub, value)
            elif prop.type == "ENUM" and prop.is_enum_flag:
                setattr(group, key, set(value))
            else:
                setattr(group, key, value)
        except (TypeError, ValueError, AttributeError):
            pass   # a setting renamed or removed in a later version


def _reset_group(group):
    for prop in group.bl_rna.properties:
        key = prop.identifier
        if key in SKIP_SCENE or prop.identifier == "rna_type":
            continue
        if prop.type == "POINTER":
            sub = getattr(group, key)
            if isinstance(sub, bpy.types.PropertyGroup):
                _reset_group(sub)
            continue
        if not prop.is_readonly:
            group.property_unset(key)


# --- session values -------------------------------------------------------------------

def session_values():
    values = dict(FACTORY_SESSION)
    saved = load().get("session", {})
    if isinstance(saved, dict):
        values.update({k: v for k, v in saved.items() if k in FACTORY_SESSION})
    return values


def automerge_for(tool_id):
    values = session_values()
    return bool(values["automerge_polypen"] if tool_id == "ppo.polypen" else values["automerge_other"])


def apply_session(ts):
    """PolyTailor's snapping setup on entering a PolyTailor tool (session.sync)."""
    values = session_values()
    try:
        ts.snap_elements = set(values["snap_elements"]) or {"FACE_NEAREST"}
    except (TypeError, ValueError):
        ts.snap_elements = {"FACE_NEAREST"}
    ts.use_snap = bool(values["use_snap"])


def apply_scene(scene):
    """Copy the saved scene defaults into a scene PolyTailor starts in for the first time."""
    from .core.face_attributes import initialize
    initialize(scene)
    settings = scene.ppo
    if settings.defaults_applied:
        return
    settings.defaults_applied = True
    saved = load().get("scene")
    if isinstance(saved, dict):
        # Defaults saved before 2.2.4 may name a removed Build mode: all are 3in1 now,
        # and the old Quad mode keeps quads-only faces.
        if saved.get("polypen_mode") == "QUAD" and "merge3_faces" not in saved:
            saved = dict(saved, merge3_faces="QUAD")
        _dict_to_group(settings, saved)


# --- operators ----------------------------------------------------------------------

def _current_tool(context):
    from .ops.common import active_tool_id
    return active_tool_id(context) or ""


class PPO_OT_defaults_save(bpy.types.Operator):
    bl_idname = "ppo.defaults_save"
    bl_label = "Save as Default"
    bl_translation_context = "Poly Toolset"
    bl_description = ("Use the current PolyTailor settings - tool and brush settings, snapping, "
                      "Stop at Edge, Auto Merge - as the defaults for new files and scenes")
    bl_options = {"INTERNAL"}

    def execute(self, context):
        ts = context.scene.tool_settings
        session = session_values()
        settings = context.scene.ppo
        pen3d = settings.pen3d   # 3D turned snapping / Auto Merge off: keep the real choice
        session["use_snap"] = bool(settings.pen3d_prev_snap if pen3d else ts.use_snap)
        session["snap_elements"] = sorted(ts.snap_elements)
        # Auto Merge belongs to the tool it was set in
        key = "automerge_polypen" if _current_tool(context) == "ppo.polypen" else "automerge_other"
        session[key] = bool(settings.pen3d_prev_merge if pen3d else ts.use_mesh_automerge)
        data = {"scene": _group_to_dict(context.scene.ppo), "session": session}
        if pen3d:   # 3D turned Stop at Edge off too: keep the real choice
            data["scene"]["clamp_to_source"] = bool(settings.pen3d_prev_clamp)
        if not _write(data):
            self.report({"ERROR"}, tr("Could not save the defaults"))
            return {"CANCELLED"}
        context.scene.ppo.defaults_applied = True
        self.report({"INFO"}, tr("PolyTailor defaults saved"))
        return {"FINISHED"}


class PPO_OT_defaults_restore(bpy.types.Operator):
    bl_idname = "ppo.defaults_restore"
    bl_label = "Restore Factory Defaults"
    bl_translation_context = "Poly Toolset"
    bl_description = "Forget the saved defaults and put this scene's PolyTailor settings back to PolyTailor's own defaults"
    bl_options = {"INTERNAL", "UNDO"}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        path = _path()
        try:
            if path is not None and path.exists():
                path.unlink()
        except OSError:
            self.report({"ERROR"}, tr("Could not remove the saved defaults"))
            return {"CANCELLED"}
        settings = context.scene.ppo
        _reset_group(settings)
        settings.pen3d_backup_ready = False
        settings.defaults_applied = True
        from . import props, session
        if context.scene.session_uid in session._scenes:
            ts = context.scene.tool_settings
            apply_session(ts)
            session.set_merge(context.scene, session.wanted_merge(context.scene, _current_tool(context)))
            if settings.pen3d:
                props.begin_pen3d(context.scene)
            props.sync_native_clamp(context)
        self.report({"INFO"}, tr("PolyTailor factory defaults restored"))
        return {"FINISHED"}


_classes = (PPO_OT_defaults_save, PPO_OT_defaults_restore)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
