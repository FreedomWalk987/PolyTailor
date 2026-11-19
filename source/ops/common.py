# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Shared modal helpers: hover state, mesh guard, release on end."""

import time
import bpy
from ..i18n import tr_rpt   # reports in the add-on context
from ..core import safe_mesh


class Hover:
    """Last known mouse position over a 3D View region."""
    region = 0
    xy = None



# Ended operators must release resources and edit-mesh references promptly.
# This is a lifecycle precaution, not a diagnosis of a particular native crash.
_END = {"FINISHED", "CANCELLED"}
_PER_MOUSE_MOVE = {"ppo.brush_hover"}   # invoked on every mouse move: no session look each time (session.ensure)
_GUARDED = {"ppo.drag", "ppo.move", "ppo.tweak",
            "ppo.relax", "ppo.smooth_selection", "ppo.polypen", "ppo.fill"}


# every PolyTailor modal shows its own keys in the status bar (normalised names)
_STATUS_OPS = {"ppo.polypen", "ppo.fill", "ppo.drag", "ppo.move", "ppo.slide", "ppo.relax", "ppo.tweak",
               "ppo.brush_adjust", "ppo.smooth_selection"}
STATUS_OWNERS = {}   # WorkSpace pointer -> id() of the modal whose keys the status bar shows (session skips it)


def _status_idle():
    """Give the status bar back to the idle tool's key table at once (not on the next sync tick)."""
    try:
        from .. import session
        session.update_status(force=True)
    except Exception:
        pass  # Advisory text only.


def _status_update(op, context):
    """Blender's status bar shows this modal's own keys (ui.keytable.modal_status), not the keys
    of the Blender operators it overrides. Set only on change; cleared in _release."""
    try:
        from ..ui import keytable
        text = keytable.modal_status(op, context)
        state = op.__dict__
        if text == state.get("ppo_status_text", ""):
            return
        ws = context.workspace
        if text:
            ws.status_text_set(text)
            state["ppo_status_text"], state["ppo_status_ws"] = text, ws
            STATUS_OWNERS[ws.as_pointer()] = id(op)
        else:   # no keys of its own now: the idle tool's keys again
            _status_drop(op)
            state["ppo_status_text"] = ""
            _status_idle()
    except Exception:
        pass  # Advisory text only.


def _status_drop(op):
    ws = getattr(op, "__dict__", {}).pop("ppo_status_ws", None)
    if ws is None:
        return False
    try:
        if STATUS_OWNERS.get(ws.as_pointer()) == id(op):
            STATUS_OWNERS.pop(ws.as_pointer(), None)
        ws.status_text_set(None)
    except Exception:
        pass
    return True


_LIVE = []   # running PolyTailor modal operator instances
_END_TEARDOWN = {"ppo.fill", "ppo.polypen"}   # tools whose own _end (header, jobs) _release runs


def release_all():
    """tear down every running PolyTailor modal (draw handlers, timers, BMesh refs) before unregister.
    and cancel every second-search run in progress (its jobs stop at their next check); stop its worker processes."""
    for op in list(_LIVE):
        try:
            _release(op, bpy.context)
        except Exception:
            pass
    _LIVE.clear()
    try:
        from ..core.newfill import pipeline
        pipeline.cancel_all()
        pipeline.shutdown()
    except Exception:
        pass
    clear_caches()


def clear_caches():
    """Empty the exact memo and phase caches of the fill modules (add-on unregister)."""
    try:
        from ..core import memo_jobs, boundary_preview, fill_p1
        from ..core.patterns import chords, solver
        from ..core.newfill import apattern, fallback5
    except Exception:
        return
    for module in (memo_jobs, chords, boundary_preview, solver, apattern, fill_p1, fallback5):
        try:
            module.clear_caches()
        except Exception:
            pass
    try:
        from ..core import polypen_support
        polypen_support.reset_projection_cache()
    except Exception:
        pass


def _session_later():
    """one session look once this event is handled (a modal started or ended; the session follows events only)."""
    try:
        from .. import session
        session.later()
    except Exception:
        pass


def _release(op, context=None):
    """Idempotent teardown; resource failures must not mask the original error."""
    try:
        _LIVE.remove(op)
    except (ValueError, ReferenceError):
        pass
    _session_later()   # the operator is still listed while it ends: the look runs once it is gone
    try:
        from ..ui import hud
        hud.finish(op)
    except Exception:
        pass  # Optional help must never interfere with an operator's teardown.
    if _status_drop(op):   # give the status bar back on every end (to the idle tool's keys)
        _status_idle()
    state = getattr(op, "__dict__", None)
    if not state:
        return
    if not state.get("_ppo_end_done") and _opname(op) in _END_TEARDOWN:
        state["_ppo_end_done"] = True   # the tool's own teardown on every exit path, once
        end = getattr(op, "_end", None)
        if end is not None:
            try:
                end(bpy.context if context is None else context)   # idempotent (Fill, Build)
            except Exception:
                pass
    close_feedback = getattr(state.get("stroke"), "close_feedback", None)
    if close_feedback is not None:
        try:
            close_feedback()
        except Exception:
            pass
    abort_cache=getattr(state.get("stroke"),"abort_cache",None)
    if abort_cache is not None:
        try:
            abort_cache()
        except Exception:
            pass
    for owner in (op, state.get("stroke"), state.get("state")):
        attributes = getattr(owner, "attributes", None)
        if attributes is not None:
            try:
                attributes.close()
            except Exception:
                pass  # Teardown must not hide the operation's original error.
    timer = state.pop("timer", None)
    handle = state.pop("handle", None)
    if timer is not None and context is not None:
        try:
            context.window_manager.event_timer_remove(timer)
        except Exception:
            pass
    if handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(handle, "WINDOW")
        except Exception:
            pass
    area = state.pop("header_area", None)
    if area is not None:
        try:
            area.header_text_set(None)
            area.tag_redraw()
        except Exception:
            pass
    try:
        from ..core import snap
        snap.finish()   # no snap marker left behind by an operator that ended abnormally
    except Exception:
        pass
    snapshot = state.get("snapshot")
    close = getattr(snapshot, "close", None)
    if close is not None:
        try:
            close()
        except Exception:
            pass
    stroke = state.pop("stroke", None)
    if stroke is not None:
        from ..core.brush_cache import release_stroke
        release_stroke(stroke, context, op)
    try:   # never drop the last edit-BMesh reference under a foreign modal
        from ..core import brush_cache
        if not brush_cache._safe_point(bpy.context if context is None else context, op):
            brush_cache._retired.append(dict(state))
            brush_cache._schedule_release()
        else:
            brush_cache.break_cycles(state)   # no wrapper left to the cyclic GC
    except Exception:
        pass
    state.clear()


def _target_valid(op, context):
    target = getattr(op, "_ppo_target", None)
    if target is None:
        return True
    try:
        obj, data = target
        return (context.mode == "EDIT_MESH" and context.edit_object == obj
                and obj.data == data)
    except (ReferenceError, AttributeError):
        return False


def _abort(op, context, restore=True):
    try:
        if restore and _target_valid(op, context):
            op._ppo_internal_rollback = True
            callback = getattr(op, "cancel", None)
            if callback is not None:
                callback(context)
    except Exception:
        # Keep the triggering exception; teardown still runs if rollback fails.
        pass
    finally:
        _release(op, context)


def _opname(op):
    """the dotted id (ppo.fill) of a live operator instance (its bl_idname is PPO_OT_fill)."""
    name = getattr(op, "bl_idname", "") or ""
    if "_OT_" in name:
        namespace, operation = name.split("_OT_", 1)
        return namespace.lower() + "." + operation
    return name


def _refresh_sources(op, context):
    """Swap the evaluated source references a running tool holds for fresh ones.

    Per-source caches are dropped on a depsgraph geometry update (and on undo / redo /
    load); the tool keeps running. Holders: the operator, its MeshView, brush stroke, Relax
    solver, and Build / Fill state. A failure keeps the previous references.
    """
    from ..core import raycast
    holders, seen = [], set()

    def add(holder):
        if holder is not None and id(holder) not in seen:
            seen.add(id(holder))
            holders.append(holder)

    add(op)
    for name in ("view", "stroke", "state"):
        holder = getattr(op, name, None)
        add(holder)
        for inner in ("view", "solver"):
            add(getattr(holder, inner, None))
    try:
        plain = prepared = occluders = None
        for holder in holders:
            state = getattr(holder, "__dict__", None)
            if not isinstance(state, dict):
                continue
            if "srcs" in state:
                if plain is None:
                    plain = raycast.sources(context)
                if isinstance(state["srcs"], raycast.Prepared):
                    if prepared is None:
                        prepared = raycast.prepare(list(plain))
                    holder.srcs = prepared
                else:
                    holder.srcs = list(plain)
            if "occluders" in state:
                if occluders is None:
                    occluders = raycast.prepare(raycast.sources(context, visibility=True))
                holder.occluders = occluders
                if isinstance(state.get("_visible"), dict):
                    holder._visible = {}   # per-vertex visibility against the old occluders
                if "_arrays" in state:
                    holder._arrays = None
                if state.get("_share") is not None:
                    from ..core import pick
                    pick.reset_shared()
            if "_trees" in state:
                holder._trees = None   # Relax: per-stroke source BVH copies
    except (ReferenceError, RuntimeError, AttributeError, TypeError):
        pass


def _guard_check(op, context, guard):
    """Before each event of a guarded tool. Returns the (translated) message when the tool must end.

    Edit mesh replaced / freed / undone / topology or transform changed outside the
    tool: tools that can rebuild their state (Build, Fill: _ppo_refresh) do so; the
    others hold BMesh element references and end without rollback. After a foreign
    mesh modal ran over a tool: refresh, or end a stroke tool without rollback.
    Other objects changing never ends a tool.
    """
    refresh = getattr(op, "_ppo_refresh", None)
    can_refresh = refresh is not None and not getattr(op, "placing", None)
    foreign_ran = op.__dict__.pop("_ppo_foreign_ran", False)
    if guard.mesh_changed(context):
        if not can_refresh:
            return tr_rpt("The mesh being edited changed; the tool stopped")
        refresh(context)
        op._ppo_guard = safe_mesh.MeshStateGuard(context)
        return None
    if foreign_ran and refresh is None:
        return tr_rpt("Another tool is editing this mesh; stopped and kept the changes")
    if foreign_ran and can_refresh:
        refresh(context)
        op._ppo_guard = safe_mesh.MeshStateGuard(context)
        return None
    kind = guard.poll_sources(context, time.perf_counter())
    if kind == "members" and can_refresh:
        refresh(context)   # a source appeared / went away: Build / Fill rebuild their view
        op._ppo_guard = safe_mesh.MeshStateGuard(context)
    elif kind is not None:
        _refresh_sources(op, context)   # cheap swap; a running Fill job keeps going
    return None


def _wrap_event(func):
    def method(self, context, event):
        try:
            opname = _opname(self)
            guarded = opname in _GUARDED
            if func.__name__ == "invoke" and getattr(context, "mode", None) == "EDIT_MESH":
                from .. import session
                session.ensure(opname not in _PER_MOUSE_MOVE)
                obj = context.edit_object
                if obj is not None:
                    self._ppo_target = (obj, obj.data)
                    from ..core import raycast
                    if (guarded or opname == "ppo.slide") and not raycast._invertible(obj):
                        _release(self, context)   # positions cannot be mapped to and from this object
                        self.report({"WARNING"}, tr_rpt("This object's transform has a zero scale; fix the scale of the object or its parents"))
                        return {"CANCELLED"}
                allowed = ()
                if opname == "ppo.slide":   # Move hands over to Slide while still on the stack
                    allowed = tuple(safe_mesh.operator_key(o) for o in getattr(context.window, "modal_operators", ())
                                    if getattr(o, "bl_idname", "") == "PPO_OT_move")
                if (guarded or opname == "ppo.slide") and safe_mesh.foreign_modal_running(context, self, allowed):
                    _release(self, context)
                    self.report({"WARNING"}, tr_rpt("Another tool is editing this mesh; finish it first"))
                    return {"CANCELLED"}
            elif func.__name__ == "modal":
                self.__dict__["_ppo_unchecked"] = True   # until this event's guard checks pass
                if not _target_valid(self, context):
                    _abort(self, context, restore=False)
                    self.report({"WARNING"}, tr_rpt("The mesh being edited changed; the tool stopped"))
                    return {"CANCELLED"}
                guard = getattr(self, "_ppo_guard", None)
                if guard is not None:
                    if safe_mesh.foreign_modal_running(context, self):
                        # An event leaking through a foreign mesh modal must not touch tool
                        # state; reset once it has ended.
                        self.__dict__["_ppo_foreign_ran"] = True
                        if event.type == "WINDOW_DEACTIVATE" or (event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS"):
                            _abort(self, context, restore=False)
                            self.report({"WARNING"}, tr_rpt("Another tool is editing this mesh; stopped and kept the changes"))
                            return {"CANCELLED"}
                        if opname in {"ppo.polypen", "ppo.fill"} and not getattr(self, "placing", None) and event.type in {"LEFT_CTRL", "RIGHT_CTRL"} and event.value == "RELEASE":
                            end = getattr(self, "_end", None)   # the operator's own teardown (header, job)
                            if end is not None:
                                try:
                                    end(context)
                                except Exception:
                                    pass
                            _release(self, context)
                            return {"FINISHED"}
                        return {"PASS_THROUGH"}
                    stop = _guard_check(self, context, guard)   # guard check before each event
                    if stop is not None:
                        _abort(self, context, restore=False)   # never write old positions over outside edits
                        self.report({"WARNING"}, stop)
                        return {"CANCELLED"}
            self.__dict__.pop("_ppo_unchecked", None)
            result = func(self, context, event)
            if isinstance(result, (set, frozenset)) and "RUNNING_MODAL" in result:
                try:
                    from ..ui import hud
                    hud.observe(self, context, event)
                except Exception:
                    pass  # Help snapshots are advisory and never own tool input.
                try:
                    from ..ui import hud as _hud   # runtime bl_idname is PPO_OT_fill: normalise as the HUD does
                    if _hud.operator_name(self) in _STATUS_OPS:
                        now = time.perf_counter()   # TIMER events refresh at most every 0.1 s
                        if event.type != "TIMER" or now - self.__dict__.get("ppo_status_t", 0.) >= .1:
                            self.__dict__["ppo_status_t"] = now
                            _status_update(self, context)
                except Exception:
                    pass  # Advisory text only.
            if func.__name__ == "invoke" and isinstance(result, (set, frozenset)) and "RUNNING_MODAL" in result:
                if self not in _LIVE:   # torn down first if the add-on is disabled meanwhile
                    _LIVE.append(self)
                _session_later()   # PolyTailor is in use while its modal runs
            if guarded and isinstance(result, (set, frozenset)) and "RUNNING_MODAL" in result:
                # Rebaseline only after our own event has completed successfully.
                if getattr(self, "_ppo_guard", None) is None:
                    self._ppo_guard = safe_mesh.MeshStateGuard(context)
                else:
                    self._ppo_guard.accept(context)
        except Exception:
            # an error before this event's guard check passed (e.g. inside the check) never rolls
            # back: the mesh was not verified, so a rollback could overwrite outside edits (fail safe)
            _abort(self, context, restore=not getattr(self, "__dict__", {}).get("_ppo_unchecked", False))
            raise
        if isinstance(result, (set, frozenset)) and result & _END:
            _release(self, context)
        return result
    method.__name__, method.__doc__, method._ppo_released = func.__name__, func.__doc__, True
    return method


def _wrap_execute(func):
    def method(self, context):
        try:
            if _opname(self) in {"ppo.pin", "ppo.glue"} and safe_mesh.foreign_modal_running(context, self):
                _release(self, context)
                self.report({"WARNING"}, tr_rpt("Another tool is editing this mesh; finish it first"))
                return {"CANCELLED"}
            result = func(self, context)
        except Exception:
            _abort(self, context)
            raise
        if isinstance(result, (set, frozenset)) and result & _END:
            _release(self, context)
        return result
    method.__name__, method.__doc__, method._ppo_released = func.__name__, func.__doc__, True
    return method


def _wrap_cancel(func):
    def method(self, context):
        try:
            if _target_valid(self, context):
                guard = getattr(self, "_ppo_guard", None)
                internal = getattr(self, "_ppo_internal_rollback", False)
                if guard is not None and not internal:
                    if (safe_mesh.foreign_modal_running(context, self) or guard.mesh_changed(context)
                            or self.__dict__.get("_ppo_foreign_ran")):
                        return
                return func(self, context)
        finally:
            _release(self, context)
    method.__name__, method.__doc__, method._ppo_released = func.__name__, func.__doc__, True
    return method


def release_state_on_end(cls):
    """Route normal completion, system cancellation and errors through teardown."""
    for name, wrap in (("invoke", _wrap_event), ("modal", _wrap_event),
                       ("execute", _wrap_execute), ("cancel", _wrap_cancel)):
        func = cls.__dict__.get(name)
        if func is not None and not getattr(func, "_ppo_released", False):
            setattr(cls, name, wrap(func))
    return cls


def timer_tick(op):
    """Consume each advance of the owned Timer once, without Event.timer.

    Blender exposes Timer.time_duration but no timer handle on bpy.types.Event.
    TIMER events are shared: an unchanged owned timer must not cause more work.
    This gates work by the owned timer's progress, not by event identity.
    """
    timer = getattr(op, "timer", None)
    if timer is None:
        return False
    duration = timer.time_duration
    previous = getattr(op, "timer_duration", duration)
    if duration <= previous:
        return False
    op.timer_duration = duration
    return True


def active_tool_id(context):
    try:
        tool = context.workspace.tools.from_space_view3d_mode(context.mode, create=False)
    except (AttributeError, RuntimeError):
        return None
    return tool.idname if tool else None


