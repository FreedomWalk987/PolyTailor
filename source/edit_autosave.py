# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Edit Mode Auto-Save. Blender's own Auto Save does not write while the user stays in Edit Mode, and retopology is
long Edit Mode work. While a PolyTailor tool is in use, and only then, this module saves a copy of the open file as
Blender's Auto Save would:

- off by default (preference). It follows Blender's own Auto Save settings (Preferences > Save & Load): its
  "Auto Save Temporary Files" toggle and its interval in minutes;
- one one-shot timer at the end of the interval: registered when a PolyTailor tool is entered, removed when it is
  left (start / stop, from the session's in-use change). A short leave does
  not start the interval again from zero (the time of the last save is kept);
- only when something changed: not when the file has no unsaved changes (bpy.data.is_dirty), and not when no
  edit, undo or redo came since the last copy or the last save. The change mark is the mesh guard's
  (core.safe_mesh): its update count and its undo / load count, kept by its own depsgraph / undo handlers, which
  exist exactly while PolyTailor is in use. Without them every interval counts as changed;
- a manual save starts the interval again (the session's save_post calls saved(); a copy of this module does not
  count, nor a save to another file, nor a failed save: save_post_fail has its own session handler);
- bpy.ops.wm.save_as_mainfile(copy=True, compress=True) to <temp dir>/<name>_<pid>_edit_autosave.blend (<name>:
  the file's name, or "untitled"): a name of its own, so Blender's own auto-save file (<name>_<pid>_autosave.blend)
  is never overwritten. It works on every supported Blender version, writes the edit mesh as Ctrl+S does
  and changes neither the user's own file, its path, the unsaved-changes state nor the undo history. The session's
  save handlers put the user's own settings into the copy, as for Ctrl+S. Temp dir: the one Blender's own Auto
  Save uses (File > Recover > Auto Save opens it and lists the copy), else the preference's, else the system's;
- the copy records the original file's path (ORIGIN_KEY on each scene, only while it is written). Opened later, it
  is not linked to that file (Blender links only its own auto-save files back): a message names the file and asks
  to use File > Save As (loaded(), from the session's load_post);
- never in the middle of a PolyTailor stroke or drag (_stroke_active): it looks again after DEFER_S;
- it also waits while a modal operator of Blender or another add-on runs (a G / R / S, a dialog...); while one of
  Blender's own runs it keeps waiting (as Blender's Auto Save), but a modal of another add-on holds it at most one
  whole interval, then the copy is written (a long-running modal of another add-on never holds it forever);
- a failed write is tried again after RETRY_S; after MAX_FAIL failures in a row: one message, then off for the
  session. A write that fails while another modal runs (e.g. a render) is only tried again one interval later,
  never counted (a missing temp folder always counts);
- no error ever leaves the timer (_tick logs it once and looks again after RETRY_S; the same error again: after
  one interval).
Known limits: a change in Edit Mode that is neither geometry nor transform (e.g. a rename or a
shading-only change) is not a change mark: is_dirty is True, but no copy follows until the next edit. While another
modal keeps running (e.g. a long-lived modal of another add-on), the timer looks again every DEFER_S until the copy
has waited one interval. While one of Blender's own modals runs, the copy waits with no limit (looked at again every
DEFER_S), as Blender's own Auto Save does.
"""
import os
import re
import tempfile
import time
import bpy

DEFER_S = 1.0    # a stroke or another modal operator runs: the next look
RETRY_S = 10.0   # after a failed write; also the shortest wait for a save after entering
MAX_FAIL = 5
SUFFIX = "_edit_autosave"
ORIGIN_KEY = "polytailor_autosave_of"   # scene custom property: the original file of a copy

# last: when the last copy / save was made (or the first entry); held: since when a due copy waits for a modal;
# seen: the change mark at the last copy / save (None: unknown, counts as changed); writing: a copy is being written
_state = dict(last=None, fails=0, disabled=False, held=None, seen=None, writing=False)

# PolyTailor modals that never leave an edit half-done between events: Fill commits in one step, the wheel, the tool
# switch and the brush radius / strength drag do not write the mesh. Build (ppo.polypen) is half-done only while
# it places, draws or holds an uncommitted edit (_build_busy); every other PolyTailor modal (Tweak, Relax, Drag,
# Move, Slide, Relax Selection, and any later one) for its whole run.
_IDLE_MODALS = {"ppo.fill", "ppo.persistent_pie", "ppo.mode_toggle", "ppo.brush_adjust"}


def _enabled():
    if _state["disabled"]:
        return False
    try:
        from . import prefs
        return bool(getattr(prefs.get(), "edit_autosave", False))
    except Exception:
        return False


def _interval():
    return max(1, int(bpy.context.preferences.filepaths.auto_save_time)) * 60.


def _directories():
    found = []
    try:   # bpy.app.tempdir is <temp dir>/blender_<id>/, removed when Blender quits; its parent keeps the file
        found.append(os.path.dirname(os.path.normpath(bpy.app.tempdir)))
    except Exception:
        pass
    try:
        folder = bpy.context.preferences.filepaths.temporary_directory
        if folder:
            found.append(bpy.path.abspath(folder))
    except Exception:
        pass
    found.append(tempfile.gettempdir())
    return found


def path():
    """<temp dir>/<name>_<pid>_edit_autosave.blend, or None when no temp dir can be written."""
    stem = os.path.splitext(os.path.basename(bpy.data.filepath))[0] if bpy.data.filepath else ""
    name = re.sub(r"_\d+(_edit)?_autosave$", "", stem) or "untitled"   # an opened auto-save file: no doubled suffix
    if stem == f"{name}_{os.getpid()}{SUFFIX}":   # this session's own copy is open: never written over
        name = stem
    filename = f"{name}_{os.getpid()}{SUFFIX}.blend"
    for folder in _directories():
        if folder and os.path.isdir(folder) and os.access(folder, os.W_OK):
            return os.path.join(folder, filename)
    return None


def _edit_mode():
    try:
        return any(obj.mode == "EDIT" and obj.type == "MESH" for obj in bpy.data.objects)
    except Exception:
        return False


def _other_modal():
    from . import session
    try:
        return any(session._foreign_modal(window) for window in bpy.context.window_manager.windows)
    except Exception:
        return True


# top modules of Blender's own bundled Python operators (scripts/startup, modules)
_BLENDER_PY = {"bl_operators", "bl_ui", "bpy", "bpy_extras", "bpy_types", "bl_keymap_utils", "bl_app_templates_system",
               "bl_app_override", "bl_i18n_utils", "bl_previews_utils", "bl_rna_utils", "rna_prop_ui", "nodeitems_utils"}


def _blender_op(idname):
    """Blender's own operator (RNA name, e.g. TRANSFORM_OT_translate): no Python class is registered under that name
    (it is defined in C), or its class comes from Blender's bundled Python. Add-ons and extensions register classes
    from their own modules; a Python Macro only in the Macro registry, so both are asked. Checked on 4.5 and 5.2.
    An error: not Blender's (the one-interval cap applies)."""
    try:
        cls = (bpy.types.Operator.bl_rna_get_subclass_py(idname)
               or bpy.types.Macro.bl_rna_get_subclass_py(idname))
    except Exception:
        return False
    if cls is None:
        return True
    return str(getattr(cls, "__module__", "") or "").split(".")[0] in _BLENDER_PY


def _blender_modal():
    """one of Blender's own modal operators runs (a knife, G / R / S ...): the copy waits as long as it runs."""
    try:
        return any(not op.bl_idname.startswith("PPO_OT_") and _blender_op(op.bl_idname)
                   for window in bpy.context.window_manager.windows for op in window.modal_operators)
    except Exception:
        return False


def _build_busy(op):
    """Build places a click, draws or waits on a stroke, or holds an uncommitted edit (it is idle while Ctrl only
    previews)."""
    return (getattr(op, "stroke_drawing", False) or getattr(op, "placing", None) is not None
            or getattr(op, "line_placing", None) is not None or getattr(op, "stroke_pending", None) is not None
            or getattr(op, "snapshot", None) is not None)


def _stroke_active():
    """a PolyTailor stroke / drag is half-done (the running PolyTailor modals: ops.common._LIVE)."""
    from .ops import common
    for op in list(common._LIVE):
        try:
            name = common._opname(op)
            if name in _IDLE_MODALS:
                continue
            if name == "ppo.polypen" and not _build_busy(op):
                continue
            return True
        except Exception:   # an operator already gone
            continue
    return False


def _change_mark():
    """the mesh guard's update and undo / load counts, or None when its handlers are not there (unknown)."""
    from .core import safe_mesh
    try:
        if safe_mesh._on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
            return None
        return safe_mesh._generation, safe_mesh._epoch
    except Exception:
        return None


def _changed():
    if not bpy.data.is_dirty:
        return False   # nothing since the last save
    mark = _change_mark()
    return mark is None or mark != _state["seen"]


def saved(filepath=None, *_args):
    """the session's save_post (a save that succeeded; save_post_fail has its own session handler, which leaves this
    state alone): a save of the user's own file starts the interval again. A copy written here, or a save to another
    file (Save Copy), does not count."""
    if _state["writing"]:
        return
    try:
        own = bpy.data.filepath
        if not own or not isinstance(filepath, str) or (
                os.path.normcase(os.path.abspath(filepath)) != os.path.normcase(os.path.abspath(own))):
            return
    except Exception:
        return
    _state.update(last=time.monotonic(), held=None, seen=_change_mark())


def _write(target):
    """one copy; True when it was written. The live session values held by the session's save_pre are always put
    back (also when the write failed on a Blender without save_post_fail)."""
    from . import session
    _state["writing"] = True
    marked = []
    try:
        origin = bpy.data.filepath
        if origin:   # the copy records its original file (removed from the open file right after)
            for scene in bpy.data.scenes:
                if scene.library is None:   # a linked scene cannot hold it
                    try:
                        scene[ORIGIN_KEY] = origin
                        marked.append(scene)
                    except Exception:   # not recorded on this scene; the copy is still written
                        pass
        result = bpy.ops.wm.save_as_mainfile(filepath=target, copy=True, compress=True, check_existing=False)
        return "FINISHED" in result
    except Exception as exc:
        print("PolyTailor Edit Mode Auto-Save failed:", exc)
        return False
    finally:
        for scene in marked:
            try:
                if ORIGIN_KEY in scene:
                    del scene[ORIGIN_KEY]
            except Exception as exc:   # never out of the timer
                print("PolyTailor Edit Mode Auto-Save:", exc)
        _state["writing"] = False
        try:
            if session._save_hold[0] is not None:
                session._release_hold()
        except Exception as exc:   # never out of the timer (Blender would drop it)
            print("PolyTailor Edit Mode Auto-Save:", exc)


_tick_errors = set()   # errors _tick printed (each once)


def _tick():
    """the one-shot timer: the wait until the next look, or None (stopped). No error ever leaves it (Blender would
    drop the timer): it is printed once and the timer looks again after RETRY_S (a repeat: after one interval)."""
    try:
        return _look()
    except Exception as exc:
        tb = exc.__traceback__
        while tb is not None and tb.tb_next is not None:
            tb = tb.tb_next
        key = (type(exc).__name__, tb.tb_frame.f_code.co_name if tb else "", tb.tb_lineno if tb else 0)
        if key not in _tick_errors:   # the same error again = same type at the same line (its text may vary)
            _tick_errors.add(key)
            print("PolyTailor Edit Mode Auto-Save:", repr(exc)[:200])
            return RETRY_S
        try:
            return _interval()   # the same error again: looked at once per interval
        except Exception:
            return RETRY_S


def _look():
    from . import session
    if not (_enabled() and session._in_use[0]):
        return None
    now = time.monotonic()
    if _state["last"] is None:
        _state["last"] = now
    wait = _interval() - (now - _state["last"])
    if wait > DEFER_S:   # the interval was made longer, or a manual save started it again
        return wait
    if not bpy.context.preferences.filepaths.use_auto_save_temporary_files:
        _state.update(last=now, held=None)
        return None   # Blender's Auto Save is off: no timer (the next tool entry looks again)
    if not _edit_mode():
        _state.update(last=now, held=None)   # no Edit Mode: Blender's own Auto Save writes
        return _interval()
    if not _changed():
        _state.update(last=now, held=None)   # nothing new since the last copy / save: no copy
        return _interval()
    if _state["held"] is None:
        _state["held"] = now
    if _stroke_active():
        return DEFER_S   # never in the middle of a PolyTailor stroke / drag
    other = _other_modal()
    if other and (now - _state["held"] < _interval() or _blender_modal()):
        return DEFER_S   # another modal runs: wait (one of Blender's: as long as it runs; an add-on's: one interval)
    target = path()
    if target is not None and _write(target):
        # what the copy holds: no edit can come while it is written, and the original-file record set and
        # removed around the write is not an edit
        _state.update(last=time.monotonic(), fails=0, held=None, seen=_change_mark())
        return _interval()
    if target is None:
        print("PolyTailor Edit Mode Auto-Save failed: no writable temp directory")
    if other and target is not None:
        return _interval()   # refused while another modal runs (e.g. a render): tried again an interval later,
        # never counted
    _state["fails"] += 1
    if _state["fails"] < MAX_FAIL:
        return RETRY_S
    _state["disabled"] = True
    _notify()
    return None


def _notify():
    from .i18n import tr
    text = tr("Edit Mode Auto-Save failed repeatedly and is off for this session")
    print("PolyTailor:", text)
    try:
        windows = bpy.context.window_manager.windows
        if windows and not bpy.app.background:   # a popup needs a real window (none in background)
            def draw(self, _context):
                self.layout.label(text=text)
            with bpy.context.temp_override(window=windows[0]):
                bpy.context.window_manager.popup_menu(draw, title="PolyTailor", icon="ERROR")
    except Exception:
        pass


_opened_copy = [None]   # the original file of the copy opened last (None: the open file is not a copy)


def loaded():
    """after a file load: an Edit Mode Auto-Save copy (its name and its recorded original file) gets one message
    that names the original file and asks to use File > Save As. The record is removed from the opened file
    (also from a file that is not a copy any more, e.g. one saved under another name)."""
    _opened_copy[0] = None
    origin = None
    for scene in bpy.data.scenes:
        value = scene.get(ORIGIN_KEY)
        if value is not None:
            origin = origin or str(value)
            try:
                del scene[ORIGIN_KEY]
            except Exception:   # a linked scene: left as it is
                pass
    stem = os.path.splitext(os.path.basename(bpy.data.filepath))[0] if bpy.data.filepath else ""
    if not origin or not stem.endswith(SUFFIX):
        return
    _opened_copy[0] = origin
    from .i18n import tr
    text = tr('This file is an Edit Mode Auto-Save copy of {path}. Use File > Save As to save it there').format(path=origin)
    print("PolyTailor:", text)
    if bpy.app.background:
        return

    def show():   # after the load has finished drawing its windows
        try:
            windows = bpy.context.window_manager.windows
            if windows:
                def draw(self, _context):
                    self.layout.label(text=text)
                with bpy.context.temp_override(window=windows[0]):
                    bpy.context.window_manager.popup_menu(draw, title="PolyTailor", icon="INFO")
        except Exception as exc:
            print("PolyTailor:", exc)
    bpy.app.timers.register(show, first_interval=0.5)


def start():
    """a PolyTailor tool is in use: the one-shot timer, at the end of the interval."""
    if not _enabled() or not bpy.context.preferences.filepaths.use_auto_save_temporary_files:
        stop()
        return
    now = time.monotonic()
    if _state["last"] is None:
        _state["last"] = now
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=max(RETRY_S, _interval() - (now - _state["last"])),
                                persistent=True)   # a file load inside the tool keeps it (the session may stay in use)


def stop():
    """PolyTailor is left (or the preference turned off): no timer."""
    _state["held"] = None
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)


def update():
    """the preference changed."""
    from . import session
    (start if session._in_use[0] else stop)()


def register():
    _state.update(last=None, fails=0, disabled=False, held=None, seen=None, writing=False)


def unregister():
    stop()
