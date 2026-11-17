# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Temporary retopology setup, restored when the last tool view leaves.

Only *keys* are stored (pointer integers / scene session_uid), never RNA objects.
View3DOverlay and ToolSettings are not ID data: a Python reference to them is NOT
invalidated when the area, window or scene is freed, so writing to a stored one
after the user closes / joins an area or deletes a scene writes into freed memory
(heap corruption, crashing later e.g. in UI_block_free). Every restore therefore
looks the target up again among the views / scenes that still exist.
"""

import math
import time

import bpy
from . import defaults, prefs, props
from .core import raycast
from .ops.common import active_tool_id

_views = {}    # SpaceView3D pointer -> overlay values to restore
_view_look = {}  # SpaceView3D pointer -> "retopo" / "opaque": the overlay look applied
_offsets = {}    # SpaceView3D pointer -> Retopology Offset last set here (None: set by the user)
_entered = {}     # SpaceView3D pointer -> when it was first seen in Edit Mode with a PolyTailor tool
WARM_DELAY = 0.15  # seconds after that (a frame of its own: a one-shot look) before the sources' ray-cast lookup is built
_scenes = {}   # Scene session_uid -> tool settings values to restore
_tools = {}    # Scene session_uid -> PolyTailor tool active on the last tick
_merge_set = {}  # Scene session_uid -> Auto Merge as PolyTailor last set it; any other live value was set by hand
_pending = {}  # Scene session_uid -> the user's values backed up before a write, the scene not managed (_scenes) yet
_selection_modes = {}  # Selection filters belong to PolyTailor tools, not native mode control
_status = {}   # WorkSpace pointer -> idle key text PolyTailor put in that workspace's status bar
_view_ids = {}  # SpaceView3D pointer -> (area pointer, screen pointer) that held it (a reused address is a new view)

# Auto Merge follows the tool: by default on while Build is active and off for every
# other PolyTailor tool (changeable with Save as Default), always off when PolyTailor is left. Applied
# only when the tool changes, so a manual toggle inside a tool is kept until the next
# tool switch.
AUTOMERGE_TOOL = "ppo.polypen"
MERGE_THRESHOLD_M = 0.0001  # Auto Merge distance while PolyTailor is active (0.1 mm); restored on leave
# Retopology Offset: 1 cm, but never more than 1% of the view distance. A fixed 1 cm
# drew (and let the tools pick) the far side of a model a few millimetres thick.
RETOPO_OFFSET = 0.01
RETOPO_OFFSET_VIEW = 0.01

_OVERLAY = ("show_retopology", "retopology_offset", "show_fade_inactive", "fade_inactive_alpha")
_LOOK_ALPHA = 0.35   # Fade Inactive alpha of PolyTailor's Retopology look
_TOOL = ("use_snap", "snap_elements", "use_snap_self", "use_snap_edit",
         "use_snap_nonedit", "double_threshold", "use_snap_translate", "use_snap_rotate",
         "use_snap_scale", "use_snap_to_same_target",
         "use_transform_correct_keep_connected")  # Topo sets it; restored on leave
# use_mesh_automerge: the user's own value is backed up with these (_SCENE_BACKUP) and restored on leaving;
# inside PolyTailor it is managed per tool and mode, see AUTOMERGE_TOOL / wanted_merge
_SCENE_BACKUP = _TOOL + ("use_mesh_automerge",)   # what entering backs up and leaving / a save puts back
# The snap target filters PolyTailor sets while it is active. Found exactly like
# this on entry, they were left behind (a file saved by an earlier version while a
# PolyTailor tool was active): restore Blender's defaults instead of keeping them.
_OWN_TARGETS = {"use_snap_self": False, "use_snap_edit": False, "use_snap_nonedit": True,
                "use_snap_to_same_target": True}
_BLENDER_TARGETS = {"use_snap_self": True, "use_snap_edit": True, "use_snap_nonedit": True,
                    "use_snap_to_same_target": False}


def _left_behind(values):
    return all(values.get(name) == value for name, value in _OWN_TARGETS.items())


def ensure_backup(scene):
    """the user's own values (_SCENE_BACKUP) are backed up before PolyTailor's first write to the scene's tool
    settings: every writer calls this first. The entry setup uses the same backup (kept in _pending until the scene
    is managed), so a write before it, or an entry that failed midway, never makes PolyTailor's value the one
    restored on leaving or written by a save."""
    key = scene.session_uid
    if key in _scenes or key in _pending:
        return
    values = _snapshot(scene.tool_settings, _SCENE_BACKUP)
    if _left_behind(values):
        values.update(_BLENDER_TARGETS)
    _pending[key] = values
    _track_file_handlers()   # a save from now on writes these values into the file


def _snapshot(target, names):
    return {name: set(value) if isinstance(value := getattr(target, name), set) else
            tuple(value) if hasattr(value, "__len__") and not isinstance(value, str) else value
            for name in names if hasattr(target, name)}


def _apply(target, values):
    for name, value in values.items():
        try:
            setattr(target, name, value)
        except (ReferenceError, AttributeError, TypeError, ValueError):
            pass


def _shown_screens():
    return [window.screen for window in bpy.context.window_manager.windows if window.screen is not None]


def _live_spaces():
    """SpaceView3D pointer -> space, for every 3D View that still exists: in every screen, also the screen of a
    workspace no window shows now (a workspace tab switched away from keeps its views)."""
    try:
        screens = list(bpy.data.screens)
    except AttributeError:   # restricted data (add-on enable / script reload): the shown screens
        screens = _shown_screens()
    spaces = {}
    for screen in screens:
        for area in screen.areas:
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    spaces[space.as_pointer()] = space
    return spaces


def _shown_spaces():
    """pointers of the 3D Views in the screens the windows show now (also an area's inactive spaces)."""
    return {space.as_pointer() for screen in _shown_screens() for area in screen.areas
            for space in area.spaces if space.type == "VIEW_3D"}


def _scene_by_uid(uid):
    return next((scene for scene in bpy.data.scenes if scene.session_uid == uid), None)


def _first_backup(view_key, overlay, screen=None):
    """The overlay values to restore for a view seen for the first time. A view copied while it showed
    PolyTailor's Retopology look (area split, area or workspace duplicated) carries that look: the backup of the
    view it was copied from is used instead, or Blender's defaults when that view is gone.
    The source is looked for among the managed views with exactly that overlay: first in `screen` (the new view's
    screen: an area split), then in screens no window shows (a workspace duplicate), then anywhere. The first group
    with a match decides; if its views' backups differ, Blender's defaults (no guess)."""
    values = _snapshot(overlay, _OVERLAY)
    if not _carries_look(values):
        return values
    found = _copy_sources(view_key, values, screen)
    if found and all(backup == found[0] for backup in found):
        return dict(found[0])
    try:
        rna = overlay.bl_rna.properties
        return {name: rna[name].default for name in values}
    except (AttributeError, KeyError, TypeError):
        return values


def _carries_look(values):
    return bool(values.get("show_retopology") and values.get("show_fade_inactive")
                and abs(values.get("fade_inactive_alpha", 0.0) - _LOOK_ALPHA) < 1e-6)


def _all_screens():
    try:
        return list(bpy.data.screens)
    except AttributeError:   # restricted data (add-on enable / script reload): the shown screens
        return _shown_screens()


def _copy_sources(view_key, values, screen=None):
    """the backups of the managed views a view with these overlay values may be a copy of: the first group with a
    match (in `screen`, then in screens no window shows, then anywhere); empty: none."""
    screens = _all_screens()
    shown = {s.as_pointer() for s in _shown_screens()}
    here = screen.as_pointer() if screen is not None else None
    groups = ([], [], [])   # same screen, a screen no window shows, any other
    for scr in screens:
        scr_key = scr.as_pointer()
        for area in scr.areas:
            for space in area.spaces:
                if space.type != "VIEW_3D":
                    continue
                key = space.as_pointer()
                backup = _views.get(key)
                if key == view_key or backup is None or _snapshot(space.overlay, _OVERLAY) != values:
                    continue
                groups[0 if scr_key == here else 1 if scr_key not in shown else 2].append(backup)
    return next((group for group in groups if group), [])


def _area_alive(area_key):
    return any(area.as_pointer() == area_key for screen in _all_screens() for area in screen.areas)


def _screen_alive(screen_key):
    return any(screen.as_pointer() == screen_key for screen in _all_screens())


def _forget_view(key):
    """a view that no longer exists (or another view at its address): its entries go, nothing is written."""
    for table in (_views, _view_look, _offsets, _entered, _view_ids):
        table.pop(key, None)


def _purge_dead_views():
    """SpaceView3D keys whose view no longer exists (an area closed: no event) are dropped at once."""
    live = set(_live_spaces())
    for key in (set(_views) | set(_view_look) | set(_offsets) | set(_entered) | set(_view_ids)) - live:
        _forget_view(key)


def _shows_applied_look(view_key, overlay):
    """the view still shows exactly the Retopology look PolyTailor applied to it (and its offset)."""
    if overlay is None or _view_look.get(view_key) != "retopo":
        return False
    values = _snapshot(overlay, _OVERLAY)
    offset = _offsets.get(view_key)
    return _carries_look(values) and (offset is None or offset < 0.0
                                      or abs(values.get("retopology_offset", -1.0) - offset) <= 1e-9 * max(1.0, offset))


def _check_identity(view_key, area_key, screen_key, overlay=None):
    """another area at a known view's pointer, while the recorded area is gone and its screen still exists (an area
    closed, a new one at the same address): a new view, the old entries are dropped. A view moved to another area
    (maximize, area swap, area docking: it still shows the look applied to it) keeps its backup."""
    old = _view_ids.get(view_key)
    if (old is not None and old[0] != area_key and not _area_alive(old[0]) and _screen_alive(old[1])
            and not _shows_applied_look(view_key, overlay)):
        _forget_view(view_key)
    _view_ids[view_key] = (area_key, screen_key)


def _restore_view(key, values, spaces=None):
    space = (spaces if spaces is not None else _live_spaces()).get(key)
    if space is not None:
        _apply(space.overlay, values)


def _restore_scene(uid, values):
    scene = _scene_by_uid(uid)
    if scene is not None:
        _apply(scene.tool_settings, values)   # snapping and the user's own Auto Merge
    _merge_set.pop(uid, None)


def wanted_merge(scene, tool=None, pen3d=None):
    """Auto Merge for the scene's mode: the value last chosen by hand in that mode,
    otherwise off in 3D and the tool's default in Topo."""
    settings = scene.ppo
    pen3d = settings.pen3d if pen3d is None else pen3d
    manual = settings.merge_manual_3d if pen3d else settings.merge_manual_topo
    if manual >= 0:
        return bool(manual)
    return False if pen3d else defaults.automerge_for(tool or AUTOMERGE_TOOL)


def wanted_retopo(scene):
    """Retopology overlay shading for the scene's mode: the value last chosen by hand
    in that mode, otherwise on in Topo and off in 3D."""
    settings = scene.ppo
    manual = settings.retopo_manual_3d if settings.pen3d else settings.retopo_manual_topo
    return bool(manual) if manual >= 0 else not settings.pen3d


def _note_manual_retopo(scene, view_key, overlay, tool):
    """Retopology shading that differs from the look PolyTailor applied to this view
    was switched by hand (W wheel, sidebar, Overlays popover): remember it for the mode.
    Only registered retopology tools count."""
    applied = _view_look.get(view_key)
    if applied is None:
        return
    live = bool(overlay.show_retopology)
    if live != (applied == "retopo"):
        settings = scene.ppo
        setattr(settings, "retopo_manual_3d" if settings.pen3d else "retopo_manual_topo", int(live))
        # the hand-made look is now the applied one; fade follows it like PolyTailor's own look
        _view_look[view_key] = "retopo" if live else "opaque"
        overlay.show_fade_inactive = live
        if live:
            _offsets[view_key] = -1.0


def set_merge(scene, value):
    """Auto Merge set by PolyTailor (not a choice by hand)."""
    ts = scene.tool_settings
    value = bool(value)
    if ts.use_mesh_automerge != value:
        ensure_backup(scene)
        ts.use_mesh_automerge = value
    if scene.session_uid in _scenes:
        _merge_set[scene.session_uid] = value


def note_manual_merge(scene):
    """A live Auto Merge that differs from the one PolyTailor set was chosen by hand
    (header, sidebar, W wheel, a key): remember it for the current mode."""
    key = scene.session_uid
    if key not in _scenes:
        return
    live = bool(scene.tool_settings.use_mesh_automerge)
    if _merge_set.get(key, live) != live:
        settings = scene.ppo
        setattr(settings, "merge_manual_3d" if settings.pen3d else "merge_manual_topo", int(live))
    _merge_set[key] = live


def retopo_offset(rv3d):
    distance = getattr(rv3d, "view_distance", 0.0) or 0.0
    return min(RETOPO_OFFSET, max(distance * RETOPO_OFFSET_VIEW, 1e-7))


def _follow_offset(view_key, overlay, rv3d):
    """Keep the Retopology Offset in scale with the view while it is ours: zoomed in
    on a millimetre part it shrinks with the view. A value the user typed is kept."""
    last = _offsets.get(view_key)
    if last is None:
        return
    current = overlay.retopology_offset
    if last >= 0.0 and abs(current - last) > 1e-9 * max(1.0, last):
        _offsets[view_key] = None   # changed by hand
        return
    wanted = retopo_offset(rv3d)
    if last < 0.0 or abs(wanted - current) > .2 * wanted:   # no rewrite (redraw) per zoom step
        overlay.retopology_offset = wanted
        _offsets[view_key] = overlay.retopology_offset


_sync_errors = set()


def _log_once(where, exc):
    key = (where + ": " + repr(exc))[:200]
    if key not in _sync_errors:
        _sync_errors.add(key)
        print("PolyTailor session:", key)


_in_use = [False]   # a PolyTailor tool is active in Edit Mesh in some view, or a PolyTailor modal runs
_fill_seen = [None]   # the last look found a view with the Fill tool (None: the look ended before the views)
_registered = [False]
_warm_left = [None]   # Topo warm-up calls left in this entry's one pass (None: not started)
_after_load = [False]   # a file was loaded since the last entry
_fingerprint = [None]   # what watch() last saw
_entry_fingerprint = [None]   # what the last entry look asked for (entry_look)
_due = [None]       # when the pending one-shot look (later) runs
_ticking = [False]  # the one-shot look runs now
_again = [None]     # a later look asked for while it runs: its delay


def sync():
    """one look at the views and scenes. Run by events only (msgbus notifies, PolyTailor operators, save / load,
    later()); never a timer of its own. It never raises."""
    try:
        _sync()
    except Exception as exc:
        _log_once("sync", exc)
    _track_file_handlers()


def _tick():
    """the one-shot look of later(). A later look asked for while it ran (a warm-up coming due) arms it once more;
    otherwise it ends (None)."""
    _due[0], _ticking[0] = None, True
    try:
        apply_now()
    finally:
        _ticking[0] = False
    again, _again[0] = _again[0], None
    if again is None or not _registered[0]:
        return None
    _due[0] = time.monotonic() + again
    return max(0.0, again)


_ZOOM_STEP = math.log(1.2)   # zoom buckets of 20 %: the Retopology Offset follows (_follow_offset)


def _fingerprint_now():
    out = [bool(prefs.get().auto_setup)]
    for window in bpy.context.window_manager.windows:
        screen, workspace, layer = window.screen, window.workspace, window.view_layer
        obj = layer.objects.active if layer is not None else None
        mode = "" if obj is None else "EDIT_MESH" if (obj.mode == "EDIT" and obj.type == "MESH") else obj.mode
        tool = ""
        if mode == "EDIT_MESH" and workspace is not None:
            item = workspace.tools.from_space_view3d_mode(mode, create=False)
            tool = item.idname if item else ""
        views = []
        for area in (screen.areas if screen is not None else ()):
            if area.type == "VIEW_3D":
                space = area.spaces.active
                distance = getattr(space.region_3d, "view_distance", 0.0) or 0.0
                views.append((area.as_pointer(), space.as_pointer(),
                              round(math.log(distance) / _ZOOM_STEP) if distance > 0.0 else 0))
        out.append((window.as_pointer(), screen.as_pointer() if screen else 0, workspace.as_pointer() if workspace else 0,
                    window.scene.as_pointer(), layer.as_pointer() if layer else 0, mode, tool, tuple(views)))
    return tuple(out)


def watch():
    """the safety net for leaving routes without a notify, run by the HUD draw handler (it exists only while
    PolyTailor is in use; a draw follows every change on screen): a changed fingerprint (a window, screen,
    workspace, scene, view layer, mode, tool, 3D View or zoom step) asks for one look; dead views go at once."""
    try:
        now = _fingerprint_now()
    except Exception as exc:
        _log_once("watch", exc)
        return
    if now != _fingerprint[0]:
        _fingerprint[0] = now
        try:
            _purge_dead_views()
        except Exception as exc:
            _log_once("watch", exc)
        later()


def entry_look():
    """a PolyTailor tool's settings are drawn while PolyTailor is not in use (an entry without a notify): one look,
    asked once per change of the fingerprint (a header redraws often)."""
    try:
        now = _fingerprint_now()
    except Exception as exc:
        _log_once("entry", exc)
        return
    if now != _entry_fingerprint[0]:
        _entry_fingerprint[0] = now
        later()


def later(delay=0.0):
    """one look a moment later, out of a notify / handler or once an operator has ended: a one-shot timer, armed only
    by such an event (or a warm-up coming due); an earlier request moves it forward."""
    if not _registered[0]:
        return
    if _ticking[0]:
        _again[0] = delay if _again[0] is None else min(_again[0], delay)
        return
    due = time.monotonic() + delay
    try:
        if bpy.app.timers.is_registered(_tick):
            if _due[0] is not None and _due[0] <= due:
                return
            bpy.app.timers.unregister(_tick)
        _due[0] = due
        bpy.app.timers.register(_tick, first_interval=max(0.0, delay))
    except Exception as exc:
        _log_once("later", exc)


def _ppo_modal():
    try:
        return any((op.bl_idname or "").startswith("PPO_OT_")
                   for window in bpy.context.window_manager.windows for op in window.modal_operators)
    except (AttributeError, ReferenceError):
        return False


def _set_in_use(on):
    """the depsgraph / undo handlers, the draw handlers (pins, snap marker, HUD), the brush cursor and Edit Mode
    Auto-Save's timer PolyTailor needs only while in use."""
    if on == _in_use[0]:
        return
    _in_use[0] = on
    _entry_fingerprint[0] = None   # the next entry by a route without a notify is asked for again (entry_look)
    if on:
        _warm_left[0] = None   # one warm-up pass per entry
        if _after_load[0]:   # the first entry after a file load: its sources are warmed again
            _after_load[0] = False
            raycast._warm.clear()
    from . import edit_autosave
    from .core import brush_cache, safe_mesh, snap
    from .ops import brush, pinning
    from .ui import hud
    for module in (brush_cache, safe_mesh, pinning, snap, hud, brush, edit_autosave):
        try:
            (module.start if on else module.stop)()
        except Exception as exc:
            _log_once("in use", exc)
    if on:
        from . import tools
        from .ui import custom_pie
        # icons follow the colours / theme of now: refreshed by a one-shot, out of this notify / invoke
        for start in (tools.request_refresh, custom_pie.warm_up_once):
            try:
                start()
            except Exception as exc:
                _log_once("in use", exc)


def ensure(every=True):
    """a PolyTailor operator starts: one look now (the session follows events only: an undo back into Edit Mode, an
    area split, the Retopology Offset for the current zoom). every=False: an operator invoked on every mouse move
    (the brush cursor tracker) looks only while PolyTailor is not in use yet."""
    props.migrate_scene(bpy.context.scene)   # an old Build mode, before the operator reads it (see props.settings)
    if every or not _in_use[0]:
        apply_now()
    if every:
        try:
            update_status(force=True)   # a native modal may have cleared the status bar meanwhile
        except Exception as exc:
            _log_once("status", exc)


def _sync():
    """state and status each in their own try: an error in one never skips the other."""
    seen_views = set()
    try:
        _sync_state(seen_views)
    except Exception as exc:
        _log_once("state", exc)
    try:   # the Fill tool was left: its worker processes stop (only if Fill was ever used); also when the look fails after the views
        if _fill_seen[0] is False:
            from .core import newfill
            worker = getattr(newfill, "worker", None)   # set once the worker was imported (a Fill was used)
            if worker is not None:
                worker.leave()
    except Exception as exc:
        _log_once("fill worker", exc)
    try:
        update_status()
    except Exception as exc:   # advisory text must never stop the session timer
        _log_once("status", exc)


def _sync_state(seen_views):
    """The managed views / scenes (fills seen_views with the PolyTailor tool views)."""
    seen_scenes = set()
    seen_tool_scenes = set()
    current_tools = {}
    in_use = False
    _fill_seen[0] = None   # stays None when the look ends before every view was looked at (no leave then)
    _purge_dead_views()
    fill_view = False   # some view has the Fill tool active (its worker processes are kept)
    area_failed = False   # some view could not be looked at (then no leave is taken from this look)
    for window in bpy.context.window_manager.windows:
        if window.screen is None:
            continue
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            area_known = False   # this view's tool and mode were read
            try:
                with bpy.context.temp_override(window=window, area=area):
                    context = bpy.context
                    tool = active_tool_id(context)
                    if context.mode != "EDIT_MESH":
                        continue
                    area_known = True
                    if tool and tool.startswith("ppo."):
                        in_use = True
                        if tool == "ppo.fill":   # the Fill workers start with the Fill tool
                            fill_view = True
                            from .core.newfill import worker
                            worker.prewarm()
                        from .ui.pie import AVAILABLE_TOOLS
                        if tool not in AVAILABLE_TOOLS:
                            # Saved workspaces can retain a tool from an older release.
                            bpy.ops.wm.tool_set_by_id(name=AUTOMERGE_TOOL, space_type="VIEW_3D")
                            tool = AUTOMERGE_TOOL
                    scene = context.scene
                    ppo_view = bool(tool and tool.startswith("ppo.") and prefs.get(context).auto_setup)
                    if not ppo_view:
                        # Native tools keep Blender's own settings: the 3D / Topo choice is
                        # remembered and applied the next time a PolyTailor tool is used.
                        continue
                    scene_key = scene.session_uid
                    seen_scenes.add(scene_key)
                    if ppo_view:
                        seen_tool_scenes.add(scene_key)
                        if scene_key not in _selection_modes:
                            _selection_modes[scene_key] = tuple(scene.tool_settings.mesh_select_mode)
                            scene.tool_settings.mesh_select_mode = (True, True, False)
                        space = area.spaces.active
                        view_key = space.as_pointer()
                        _check_identity(view_key, area.as_pointer(), window.screen.as_pointer(), space.overlay)
                        seen_views.add(view_key)
                        _entered.setdefault(view_key, time.monotonic())
                        overlay = space.overlay
                        if view_key not in _views:
                            _views[view_key] = _first_backup(view_key, overlay, window.screen)
                        _note_manual_retopo(scene, view_key, overlay, tool)
                        look = "retopo" if wanted_retopo(scene) else "opaque"
                        if _view_look.get(view_key) != look:
                            _view_look[view_key] = look
                            overlay.show_retopology = look == "retopo"
                            overlay.show_fade_inactive = look == "retopo"
                            if look == "retopo":
                                overlay.fade_inactive_alpha = _LOOK_ALPHA
                                _offsets[view_key] = -1.0   # set just below
                            area.tag_redraw()
                        if look == "retopo":
                            _follow_offset(view_key, overlay, space.region_3d)
                    if current_tools.get(scene_key) != AUTOMERGE_TOOL:
                        # Native tool switches do not reset a manual Auto Merge choice.
                        current_tools[scene_key] = tool if tool and tool.startswith("ppo.") else ""
                    if scene_key not in _scenes:
                        ts = scene.tool_settings
                        # the user's values, backed up before the first write below (or kept from a write or a
                        # failed entry before it); pending until the setup completes and the scene is managed
                        ensure_backup(scene)
                        props.apply_snap_targets(scene)
                        # Blender's S / R snap each vertex to the nearest surface of ANY object;
                        # scaled past the edge of its model, a vertex jumped to another model.
                        # Keep each vertex on the model it started on (Face Nearest only).
                        ts.use_snap_to_same_target = True
                        # snapping on/off and targets: PolyTailor's or the user's saved defaults
                        defaults.apply_scene(scene)
                        defaults.apply_session(ts)
                        # Mode callbacks stay inactive while loading defaults. This
                        # avoids clearing the Topo clamp before its backup is taken.
                        _scenes[scene_key] = _pending.pop(scene_key)
                        if scene.ppo.pen3d:
                            # First backup taken while already in 3D: the scene holds 3D's values
                            # (Stop at Edge off), so Topo starts from its own default instead.
                            props.begin_pen3d(scene, merge=defaults.automerge_for(tool), clamp=True)
                        else:
                            apply_topo_defaults(context, scene)
                        # Blender's own S / R snap each vertex to the source only while
                        # Stop at Edge is on (props.sync_native_clamp keeps them tied)
                        ts.use_snap_translate = True
                        props.sync_native_clamp(context, scene)
                        # 0.1 mm in real units: Blender units are metres times the unit scale
                        scale = scene.unit_settings.scale_length or 1.0
                        ts.double_threshold = MERGE_THRESHOLD_M / scale
                    if ppo_view and not scene.ppo.pen3d:
                        # Topo: build the sources' ray-cast lookup a moment after the view is
                        # entered, not on the first stroke and not on top of entering Edit Mode
                        wait = WARM_DELAY - (time.monotonic() - _entered[view_key])
                        if wait >= 0.0:
                            later(wait + 0.01)
                        elif _warm_left[0] != 0:   # one pass per entry: at most one call per source
                            if _warm_left[0] is None:
                                _warm_left[0] = len(raycast.sources(context))
                            if _warm_left[0] > 0 and raycast.prewarm(context):
                                _warm_left[0] -= 1
                                if _warm_left[0] > 0:
                                    later(WARM_DELAY)   # the next source a frame later
                            else:
                                _warm_left[0] = 0
            except Exception as exc:   # one area's error never skips the other areas or the restores below
                area_failed = area_failed or not area_known
                _log_once("area", exc)
    _fill_seen[0] = True if fill_view else (None if area_failed else False)   # every view was looked at
    for scene_key, tool in current_tools.items():
        scene = _scene_by_uid(scene_key)
        if scene is not None:
            note_manual_merge(scene)
        if _tools.get(scene_key) != tool:
            if scene is not None:
                set_merge(scene, wanted_merge(scene, tool))
            _tools[scene_key] = tool
    for scene_key in set(_tools) - set(current_tools):
        del _tools[scene_key]
    in_use = in_use or _ppo_modal()
    gone_views = set(_views) - seen_views
    if gone_views:
        _adopt_copies(seen_views)
        gone_views = set(_views) - seen_views
        spaces = _live_spaces()
        shown = _shown_spaces() if in_use else set()
        for key in gone_views:
            if in_use and key in spaces and key not in shown:
                # a view of a workspace no window shows (another workspace tab): it keeps its backup and look
                # while PolyTailor is in use, and is restored once it is not (or for a save, _hold_for_save)
                continue
            _restore_view(key, _views.pop(key), spaces)
            _view_look.pop(key, None)
            _offsets.pop(key, None)
            _entered.pop(key, None)
            _view_ids.pop(key, None)
    for key in set(_scenes) - seen_scenes:
        # leaving keeps the user's 3D / Topo choice (only their toggle changes it); entering again applies it
        _restore_scene(key, _scenes.pop(key))
    for key in set(_pending) - seen_scenes:   # backed up before a write, never managed (ensure_backup)
        _restore_scene(key, _pending.pop(key))
    for key in set(_selection_modes) - seen_tool_scenes:
        scene = _scene_by_uid(key)
        previous = _selection_modes.pop(key)
        if scene is not None:
            scene.tool_settings.mesh_select_mode = previous
    _set_in_use(in_use)


def _adopt_copies(seen_views):
    """a copy of a managed view that no look saw while PolyTailor was in use (the session follows events only:
    area split, new main window, duplicated area / workspace, a split area switched to another editor) carries the
    Retopology look: when the managed views go, it is restored with them. Every screen, every 3D space; a copy
    needs a managed source (_copy_sources); several differing sources: Blender's defaults, as on entry."""
    managed = set(_views)
    for screen in _all_screens():
        for area in screen.areas:
            for space in area.spaces:
                if space.type != "VIEW_3D":
                    continue
                key = space.as_pointer()
                if key in managed or key in seen_views:
                    continue
                values = _snapshot(space.overlay, _OVERLAY)
                if not _carries_look(values) or not _copy_sources(key, values, screen):
                    continue
                _views[key] = _first_backup(key, space.overlay, screen)
                _view_ids[key] = (area.as_pointer(), screen.as_pointer())


def _foreign_modal(window):
    """A modal operator that is not PolyTailor's may own the status bar (Blender's own tools set it)."""
    try:
        return any(not op.bl_idname.startswith("PPO_OT_") for op in window.modal_operators)
    except (AttributeError, ReferenceError):
        return False


def _own_modal(window):
    try:
        return any(op.bl_idname.startswith("PPO_OT_") for op in window.modal_operators)
    except (AttributeError, ReferenceError):
        return True


def _workspace(key):
    return next((ws for ws in bpy.data.workspaces if ws.as_pointer() == key), None)


def update_status(force=False):
    """While a PolyTailor tool is active in Edit Mode, Blender's status bar lists its keys
    (ui/keytable.py); cleared when the tool or the mode is left. A running PolyTailor modal shows its own
    keys (ops.common.STATUS_OWNERS) and a foreign modal keeps Blender's: both are left alone, then set again."""
    from .ops.common import STATUS_OWNERS
    from .ui import keytable
    wanted = {}
    for window in bpy.context.window_manager.windows:
        workspace, screen = window.workspace, window.screen
        if workspace is None or screen is None:
            continue
        key = workspace.as_pointer()
        area = next((a for a in screen.areas if a.type == "VIEW_3D"), None)
        text = ""
        if area is not None:
            with bpy.context.temp_override(window=window, area=area):
                context = bpy.context
                tool = active_tool_id(context)
                if context.mode == "EDIT_MESH" and tool in keytable.TOOLS:
                    text = keytable.status_text(context, tool)
        if text or key not in wanted:
            wanted[key] = (workspace, text, window)
    for key, (workspace, text, window) in wanted.items():
        if key in STATUS_OWNERS and not _own_modal(window):
            STATUS_OWNERS.pop(key, None)   # its modal ended without teardown: never block the idle keys
        if key in STATUS_OWNERS or (text and _foreign_modal(window)):
            _status.pop(key, None)   # not ours now: set again once it ends
            continue
        if text:
            if force or _status.get(key) != text:
                workspace.status_text_set(text)
                _status[key] = text
        elif _status.pop(key, None) is not None:
            workspace.status_text_set(None)
    for key in set(_status) - set(wanted):   # a workspace no window shows any more
        _status.pop(key, None)
        workspace = _workspace(key)
        if workspace is not None:
            workspace.status_text_set(None)


def _clear_status():
    for key in list(_status):
        workspace = _workspace(key)
        if workspace is not None:
            try:
                workspace.status_text_set(None)
            except (AttributeError, ReferenceError, RuntimeError):
                pass
    _status.clear()


def apply_topo_defaults(context, scene=None, reset_view=False):
    """Explicit Topo entry loads its preset, never another mode's live switches."""
    scene = scene or context.scene
    ensure_backup(scene)   # before the writes below
    settings, ts = scene.ppo, scene.tool_settings
    defaults.apply_session(ts)
    set_merge(scene, wanted_merge(scene, active_tool_id(context), pen3d=False))
    settings.clamp_to_source = True
    settings.pen3d_backup_ready = False
    props.apply_snap_targets(scene)
    ts.use_snap_to_same_target = True
    ts.use_snap_translate = True
    ts.double_threshold = MERGE_THRESHOLD_M/(scene.unit_settings.scale_length or 1.0)
    # Correct Face Attributes is left as it is: off by default in 3D and Topo alike
    # (core.face_attributes.initialize), and a manual choice survives mode switches.
    ts.use_transform_correct_keep_connected = True   # session-managed (_TOOL), restored on leave
    props.sync_native_clamp(context, scene)
    if reset_view:
        space = getattr(context, "space_data", None)
        if space is not None and space.type == "VIEW_3D":
            _view_look.pop(space.as_pointer(), None)


def activate_mode(context):
    """Initialize the scene/defaults and Topo backup BEFORE the RNA mode setter runs
    (in a PolyTailor tool; in native tools the choice is only stored)."""
    sync()


_save_hold = [None]   # the live session values held across a save (save_pre -> save_post)


def _any_edit_mode():
    """any object in Edit Mode (any scene). An error counts as yes (never touch the mode)."""
    try:
        return any(obj.mode == "EDIT" for obj in bpy.data.objects)
    except Exception:
        return True


def _hold_for_save():
    """the file gets the user's own settings (as before), the session keeps its live ones:
    snapshot the LIVE values PolyTailor manages (overlay, snapping and Auto Merge, selection mode), then write the
    user's values. Nothing is forgotten, so save_post puts the live values back exactly.
    while ANY object is in Edit Mode the selection mode is never touched (save_pre or
    save_post): its RNA write runs Blender's select-mode change on the edit mesh (EDBM_selectmode_set), which drops the
    vertex / edge selection of a Face- or Edge-mode user, live and in the saved file. The file then keeps the live mode
    and every element's selection exactly, in all three modes."""
    spaces = _live_spaces()
    views, scenes, modes = {}, {}, {}
    _save_hold[0] = (views, scenes, modes)   # filled in place: a failure midway still restores what was changed
    for key, values in _views.items():
        space = spaces.get(key)
        if space is not None:
            views[key] = _snapshot(space.overlay, _OVERLAY)
            _apply(space.overlay, values)
    for uid, values in list(_scenes.items()) + list(_pending.items()):
        scene = _scene_by_uid(uid)
        if scene is not None:
            ts = scene.tool_settings
            scenes[uid] = _snapshot(ts, _SCENE_BACKUP)
            _apply(ts, values)   # the saved file: the user's own snapping and Auto Merge
    for uid, value in (() if _any_edit_mode() else _selection_modes.items()):
        scene = _scene_by_uid(uid)
        if scene is not None:
            modes[uid] = tuple(scene.tool_settings.mesh_select_mode)
            scene.tool_settings.mesh_select_mode = value


def _release_hold():
    """after the save, the live session values back verbatim (no re-forced selection mode).
    a selection mode is put back only where save_pre swapped it (never in Edit Mode)."""
    held, _save_hold[0] = _save_hold[0], None
    if held is None:
        return
    views, scenes, modes = held
    spaces = _live_spaces()
    for key, values in views.items():
        _restore_view(key, values, spaces)
    for uid, values in scenes.items():
        scene = _scene_by_uid(uid)
        if scene is not None:
            _apply(scene.tool_settings, values)
    for uid, value in modes.items():
        scene = _scene_by_uid(uid)
        if scene is not None and tuple(scene.tool_settings.mesh_select_mode) != value:
            scene.tool_settings.mesh_select_mode = value


def _release_file_state():
    """Put the closing file's viewport / tool settings back and forget them."""
    _save_hold[0] = None
    spaces = _live_spaces()
    for key, values in list(_views.items()):
        _restore_view(key, values, spaces)
    for key, values in list(_scenes.items()) + list(_pending.items()):
        _restore_scene(key, values)
    for key, value in list(_selection_modes.items()):
        scene = _scene_by_uid(key)
        if scene is not None:
            scene.tool_settings.mesh_select_mode = value
    _selection_modes.clear()
    _views.clear()
    _view_look.clear()
    _offsets.clear()
    _entered.clear()
    _view_ids.clear()
    raycast._warm.clear()
    _warm_left[0] = None   # a load during a session: the new file's sources are warmed in a new pass
    _scenes.clear()
    _pending.clear()
    _merge_set.clear()
    _tools.clear()
    _status.clear()   # the closing file's workspaces


@bpy.app.handlers.persistent
def _on_load_pre(*_args):
    _release_file_state()


@bpy.app.handlers.persistent
def _on_save_pre(*_args):
    # Save the user's own snapping / overlay / selection-mode settings, not PolyTailor's temporary
    # ones; the session's live values are held and put back verbatim at save_post.
    _hold_for_save()


@bpy.app.handlers.persistent
def _on_save_post(*_args):
    try:   # a save of the user's own file starts Edit Mode Auto-Save's interval again (not its own copies)
        from . import edit_autosave
        edit_autosave.saved(*_args)
    except Exception as exc:
        _log_once("save", exc)
    _on_save_post_fail(*_args)


@bpy.app.handlers.persistent
def _on_save_post_fail(*_args):
    """after a save (also a failed one: save_post_fail): the live values back, one look. A failed save leaves Edit
    Mode Auto-Save's interval and change mark alone (its unsaved changes still need a copy)."""
    _in_save_post[0] = True
    try:
        try:
            _release_hold()
        finally:
            apply_now()   # re-apply at once after the save
    finally:
        _in_save_post[0] = False


# The save / load handlers above exist only while a session is open (PolyTailor in use, or something still to
# restore): a save then writes the user's own values into the file, a load restores the closing file's values,
# an undo / redo is followed by one look (_on_undo).
_in_save_post = [False]   # Blender is running the save_post / save_post_fail list: its entries are not removed now


def _session_open():
    return bool(_in_use[0] or _views or _scenes or _pending or _selection_modes or _status)


def _file_handlers():
    handlers = bpy.app.handlers
    pairs = [(handlers.load_pre, _on_load_pre), (handlers.save_pre, _on_save_pre), (handlers.save_post, _on_save_post),
             (handlers.undo_post, _on_undo), (handlers.redo_post, _on_undo)]
    if hasattr(handlers, "save_post_fail"):
        pairs.append((handlers.save_post_fail, _on_save_post_fail))
    return pairs


@bpy.app.handlers.persistent
def _on_undo(*_args):
    later()   # an undo / redo can leave or enter Edit Mode without a notify: one look after it


def _track_file_handlers(off=False):
    """added while a session is open, removed when nothing is left (off: always removed, unregister). Never removed
    while Blender runs save_post: a handler removing itself from the list being run makes Blender skip the next
    handler (another add-on's); a one-shot look right after the save removes them."""
    on = not off and _session_open()
    if not on and _in_save_post[0]:
        later()
        return
    for handlers, callback in _file_handlers():
        try:
            if on and callback not in handlers:
                handlers.append(callback)
            elif not on and callback in handlers:
                handlers.remove(callback)
        except Exception as exc:
            _log_once("file handlers", exc)


_msgbus_owner = object()
_exiting = [False]


@bpy.app.handlers.persistent
def _on_exit_pre(*_args):
    _exiting[0] = True   # Blender is quitting: unregister leaves the status bar alone


def apply_now(*_args):
    """At once (not on the next timer tick) when the tool or mode changes, so the view does not show the previous
    look for a moment."""
    try:
        sync()
    except Exception as exc:   # a notifier must never raise
        print("PolyTailor session:", exc)


def _subscribe():
    bpy.msgbus.clear_by_owner(_msgbus_owner)
    # Tab in / out of Edit Mode, and tool changes (toolbar, shortcuts): Blender
    # publishes these before the next redraw. Retopology shading and Auto Merge switched by hand: noted per mode.
    # An editor-type change, another scene or view layer in a window: PolyTailor may be left.
    for key in ((bpy.types.Object, "mode"), (bpy.types.WorkSpace, "tools"), (bpy.types.Window, "workspace"),
                (bpy.types.View3DOverlay, "show_retopology"), (bpy.types.ToolSettings, "use_mesh_automerge"),
                (bpy.types.Area, "ui_type"), (bpy.types.Area, "type"),   # Shift+F-keys publish Area.type
                (bpy.types.Window, "scene"), (bpy.types.Window, "view_layer")):
        bpy.msgbus.subscribe_rna(key=key, owner=_msgbus_owner, args=(), notify=apply_now)


@bpy.app.handlers.persistent
def _on_load_post(*_args):
    _after_load[0] = True
    _entry_fingerprint[0] = None
    try:   # the closing file's queued Build-mode migrations (props._pending) are dropped
        props.forget_migrations()
    except Exception as exc:
        _log_once("load", exc)
    try:   # an opened Edit Mode Auto-Save copy: one message naming its original file
        from . import edit_autosave
        edit_autosave.loaded()
    except Exception as exc:
        _log_once("load", exc)
    _subscribe()
    apply_now()


def register():
    _registered[0] = True
    _subscribe()
    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)
    later()   # one look: enabled (or reloaded) inside a PolyTailor tool
    # the save / load handlers: added by the session while it is open (_track_file_handlers)
    if hasattr(bpy.app.handlers, "exit_pre") and _on_exit_pre not in bpy.app.handlers.exit_pre:
        bpy.app.handlers.exit_pre.append(_on_exit_pre)


def unregister():
    _registered[0] = False
    bpy.msgbus.clear_by_owner(_msgbus_owner)
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    _track_file_handlers(off=True)
    if hasattr(bpy.app.handlers, "exit_pre") and _on_exit_pre in bpy.app.handlers.exit_pre:
        bpy.app.handlers.exit_pre.remove(_on_exit_pre)
    if _exiting[0]:
        _status.clear()
    else:
        _clear_status()
    _release_file_state()
    _set_in_use(False)
