# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Fill: hold Ctrl over a hole (or with a loop of open edges selected) to preview a
quad patch; Ctrl+Wheel changes the choice, Ctrl+click fills."""

import math
import bmesh
import bpy
import time
from ..i18n import tr as iface_, tr_rpt   # add-on context first

from .. import props, palette, prefs
from ..core import draw, fill, raycast
from ..core.pick import MeshView
from .common import timer_tick

SLICE_SECONDS = .008     # leave room for the final cooperative step and UI guard
BETTER_SECONDS = 2.5   # how long "Better result found" stays in the HUD
# refusals no fill engine can change (the fallback search is skipped for them)
_DEFINITE_REFUSALS = ("Invalid boundary", "This outline encloses no area", "Loop too short", "Loop too long")
PROGRESSIVE_SECONDS = 1.5   # searching longer: candidates appear as they are confirmed
SEARCH_CAP_SECONDS = 10.0   # a main search ends with its best so far after this long
FALLBACK_PRE_SECONDS = 2.0   # no main-search candidate yet: precompute the fallback after this
SURFACE_SECONDS = 3.0   # the main search on the outline laid out on the source finds a first candidate within this
FALLBACK_PRE_SLICE = .006   # seconds of fallback precompute per idle tick (the main search keeps its own slices)
# step 4 (main search idle with a candidate): the second search runs this long per modal tick, plus at most one
# step (a step started before the bound runs to its end). 20 ms keeps the viewport near 50 fps (a tick = this budget
# + at most one step + Blender's own event / redraw work), well under
# the ~100 ms where a preview stops feeling live, while giving the run most of each tick (one 6 ms slice per tick
# left it running 5-20 s longer than its own compute time in the real UI).
STEP4_IDLE_SECONDS = .020
PROVISIONAL_SECONDS = 1.0   # settle-first shows a provisional #1 after this long
FIRST_RESULT_SECONDS = 1.0   # fallback search: A's result is shown at 1 s while S still runs
FAST_MIN_Q = .35       # Fast mode Q: a pick below this minQ makes the search fall back to Full
TIMER_SECONDS = .001     # work resumes promptly; redraw has its own cadence
REDRAW_SECONDS = 1 / 30  # HUD and spinner redraw cadence while busy


def _pattern_label(identifier, debug=False):
    """Translated pattern name; internal IDs only with Preferences > Show Debug Info."""
    if identifier == "partition":
        return iface_("Adaptive Partition")
    if identifier == "grid":
        return iface_("Continue Grid")
    label = next((label for key, label in fill.PATTERNS if key == identifier), None)
    if label is not None:
        return iface_(label)
    if debug:
        return identifier
    parts = identifier.split("_")
    if parts[0] == "takayama" and len(parts) > 1 and parts[1].isdigit():
        return iface_("Extended {sides}-sided Pattern").format(sides=parts[1])
    if identifier == "medial":
        return iface_("Adaptive Partition")
    return iface_("Auto Pattern")


def _rim_count(plan):
    """the number of edges (= vertices) of the closed outline being filled, e.g. "24 edges"."""
    try:
        n = sum(1 for o in plan['order'] if o is not None)
    except (KeyError, TypeError):
        return ''
    return f"{n} " + iface_("edges")   # the existing word, so the number does not read as a score


def _face_count(plan):
    """The face count, e.g. "52 faces" (msgid "{faces} faces"), shown when two entries share a label."""
    n = len(fill.faces_of(plan))
    return iface_("{faces} faces").format(faces=n)   # own msgid


QUALITY_FLOOR = .20   # Fill's quality floor


def _debug(context):
    return bool(getattr(prefs.get(context), "show_debug_info", False))


class FillState:
    def __init__(self, context):
        self.context = context
        self.refresh()

    def refresh(self):
        self.view = MeshView(self.context, lazy=True)
        self.bm, self.mw = self.view.bm, self.view.mw
        self.loops = fill.open_loops(self.bm)
        self.chosen, self.selection_error = fill.selected_outline(self.bm)
        self.panels = []
        from ..core import fill_multi
        self.panels, error = fill_multi.selected_panels(self.bm)
        if error:
            self.selection_error = error
        if self.panels:
            self.chosen, self.selection_error = self.panels[0], ''


_RIM_CACHE = {}   # New pipeline results by exact world-space outline (insertion-ordered LRU)
_RIM_CACHE_MAX = 32


def _rim_key(world):
    import numpy as np
    w = np.ascontiguousarray(np.asarray(world, dtype=np.float64))
    return (w.shape, w.tobytes())


def _budget_bound(log):
    """a wall-clock budget cut this run short (the hidden precompute only gets idle slices)."""
    from ..core.newfill import pipeline
    return (log.get('s_stop') in ('budget', 'early_refusal', 'watchdog') or bool(log.get('a_budget_hit'))
            or any(v in ('budget', 'watchdog', 'overall_deadline') for v in (log.get('stop') or {}).values())
            or (log.get('seconds') or {}).get('total', 0.) >= pipeline.CFG['total'])


def _tag_redraw(context):
    """a redraw request that is safe without a region (e.g. teardown from a timer)."""
    region = getattr(context, 'region', None)
    if region is not None:
        region.tag_redraw()


# an unexpected engine error is a refusal (technical text: the user sees fill.PLAIN_FAILURE)
_ENGINE_ERROR = "Invalid fill engine state (see the system console)"


def _log_engine_error(where, exc):
    import traceback
    print("PolyTailor Fill:", where, "error, treated as a refusal:")
    traceback.print_exception(type(exc), exc, exc.__traceback__)


class PPO_OT_fill(bpy.types.Operator):
    bl_idname = "ppo.fill"
    bl_label = "Fill"
    bl_translation_context = "Poly Toolset"
    bl_description = "Hold Ctrl: preview the selected outline or hovered hole, Ctrl+Wheel: change choice, Ctrl+click: fill"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and context.region_data is not None

    def invoke(self, context, event):
        self.handle = self.timer = self.job = None
        self.background = None
        self.triangle_job = None
        self.triangle_selected = False
        self.wheel_touched = False
        self.legacy_searching = False
        self.newfill_active = False          # this preview runs the fallback search
        self.newfill_entries = []            # fallback-search wheel entries (plans), the pick first
        self.newfill_index = 0
        self.newfill_result = None
        from ..core.fill_layouts import Portfolio
        self.portfolio = Portfolio(ranking='FLOW', quality_floor=QUALITY_FLOOR)
        self.improvement_note = ""
        self.settle_pending = False   # settle #1 early
        self.reveal_seconds = PROVISIONAL_SECONDS   # 0 with Show Debug Info (set per replan)
        self.settled_signature = None
        self.better_until = 0.0
        self.field_searching = False
        self.state = FillState(context)
        import numpy as np
        if not np.isfinite(self.state.view.arrays()[4]).all():   # a NaN / Inf vertex: refuse at once
            self.report({"WARNING"}, tr_rpt(fill.PLAIN_FAILURE))
            return {"CANCELLED"}
        self.region = context.region.as_pointer()
        self.mouse = raycast.mouse(event)
        self.loop = None
        self.plan = None
        self.error = ""
        self.key = None
        self._hover(context)
        self.handle = bpy.types.SpaceView3D.draw_handler_add(self._draw, (), "WINDOW", "POST_PIXEL")
        self.timer = context.window_manager.event_timer_add(TIMER_SECONDS, window=context.window)
        self.timer_interval = TIMER_SECONDS
        self.timer_duration = self.timer.time_duration
        self.header = not _debug(context) and context.area is not None
        if self.header:
            context.area.header_text_set(self.hint())
        if self.job is not None:
            try:
                self._advance(context)
            except Exception as exc:   # no modal handler yet: nothing else would remove what was added above
                _log_engine_error('first slice', exc)
                try:
                    self._end(context)
                except Exception as end_exc:
                    _log_engine_error('first slice cleanup', end_exc)
                    self._release_ui(context)
                self.report({"WARNING"}, tr_rpt(fill.PLAIN_FAILURE))
                return {"CANCELLED"}
        _tag_redraw(context)
        context.window_manager.modal_handler_add(self)   # after the first engine slice (an error there leaves none)
        return {"RUNNING_MODAL"}

    # ---------------------------------------------------------------- planning

    def _legacy_pref(self, context, name, default):
        """The Search setting as the main search sees it; pinned to FAST in the second search's S part (its throwaway
        instance sets newfill_more). Ranking is always FLOW (Portfolio.ranked_keys: the default first, then the R1
        band, then by S)."""
        if getattr(self, 'newfill_more', False):
            return 'FAST'   # the only name asked for: fill_search_mode
        return getattr(prefs.get(context), name, default)

    def _settings(self, context):
        settings = {"pattern": "AUTO", "rotation": -1}
        settings['_ranking'] = 'FLOW'
        settings['_quality_floor'] = QUALITY_FLOOR
        settings["boundary_shift"] = 0
        if self.loop is not None:
            # how the surrounding quads turn at each border vertex (continue the grid)
            settings["_face_turns"] = fill.face_turns(self.loop)
        return settings

    def _hover(self, context):
        st = self.state
        loop = fill.pick(context, st.mw, st.loops, st.chosen, self.mouse)
        if loop is not self.loop:
            self.loop = loop
        self._replan(context)

    def _cancel_job(self):
        token, self._job_cancel = getattr(self, '_job_cancel', None), None
        if token is not None:
            token.set()   # the second search's background jobs stop at their next deadline check
        triangle_job, self.triangle_job = getattr(self, 'triangle_job', None), None
        if triangle_job is not None:
            triangle_job.close()
        background, self.background = getattr(self, "background", None), None
        if background is not None:
            background.close()
        self.improvement_note = ""
        self.field_searching = False
        job, self.job = self.job, None
        if job is not None:
            job.close()
        run, self._crease_run = getattr(self, '_crease_run', None), None
        if run is not None:
            run['gen'].close()

    def _cuts_steps(self, context, world, settings):
        """P1 and classification candidates before canonical refinement."""
        self.cuts_searched = True
        if self.state.panels:
            return
        from ..core import fill_progressive
        jobs = [fill_progressive.p1_cut_steps(world, settings),   # P1 cuts and classification: always on
                fill_progressive.cls_steps(world, settings)]
        for job in jobs:
            for update in job:
                if isinstance(update, dict) and 'candidate' in update:
                    proposed = update['candidate']
                    try:
                        if not props.settings(context).pen3d and self.state.view.srcs:
                            proposed = dict(proposed, positions=proposed['positions'].copy())
                            proposed = yield from fill.project_steps(context, proposed, self.state.view.srcs)
                        fill.validate_plan(proposed)
                        yield {'candidate': proposed}
                    except fill.FillError:
                        yield None
                else:
                    yield None

    def _plan_steps(self, context, world, settings):
        # an outline refused at the initial plan because nothing passes the
        # quality floor never fills later (0/101 corpus cases), so skip refinement.
        # Time-out refusals still refine: several of them fill later.
        self.floor_refusal = False
        try:
            return (yield from PPO_OT_fill._initial_steps(self, context, world, settings))
        except fill.FillError as exc:
            self.floor_refusal = str(exc).startswith('No valid pattern above policy quality floor')
            raise

    def _initial_steps(self, context, world, settings):
        self.cuts_searched = False
        if self.state.panels:
            from ..core import fill_multi
            project = None
            if not props.settings(context).pen3d and self.state.view.srcs:
                project = lambda plan: fill.project_steps(context, plan, self.state.view.srcs)
            return (yield from fill_multi.plan_steps(self.state.mw, self.state.panels, settings, project))
        from ..core import fill_progressive
        automatic = fill_progressive.eligible(settings)
        self.network_seeded = False
        self.legacy_frames = []
        marks = fill.certain_fail(world)
        if marks is not None:   # a provable refusal (fill.certain_fail), at once; its spots are marked on the rim
            self.fail_marks = marks
            raise fill.FillError("Invalid boundary")
        if (automatic and len(world) % 2 == 0
                and not getattr(self, 'newfill_more', False)):   # the instant direct grid (not in the S part)
            f1_started = time.perf_counter()
            direct = yield from fill_progressive.deterministic_steps(fill.direct_steps(world, settings), .00005)
            if direct is not None and not props.settings(context).pen3d and self.state.view.srcs:
                try:
                    direct = yield from fill.project_steps(context, direct, self.state.view.srcs)
                except fill.FillError:
                    direct = None
            if getattr(self, 'replan_started', None) is not None:   # the cap / precompute delay count from here
                self.replan_started += time.perf_counter() - f1_started
            if direct is not None:
                yield {'candidate': direct}
        head_stats = {}   # the head start is measured on the job's work clock, not real time
        legacy_job = (fill_progressive.initial_steps(world, dict(settings, _legacy_frames=self.legacy_frames), head_stats)
                      if automatic else
                      fill_progressive.deterministic_steps(fill.solve_steps(world, settings), .00005))   # work clock
        seed = None
        legacy_error = None
        try:
            # Give common network fills a short head start. Pause only between
            # cooperative steps, with every search deadline already restored.
            if automatic:
                while head_stats.get('work_ticks', 0) * .00005 < .02:   # 0.02 work-seconds
                    try:
                        next(legacy_job)
                    except StopIteration as done:
                        plan = done.value
                        if not props.settings(context).pen3d and self.state.view.srcs:
                            plan = yield from fill.project_steps(context, plan, self.state.view.srcs)
                        return plan
                    except fill.FillError as exc:
                        legacy_error = str(exc)
                        break
                    yield None
                self.network_seeded = True
                job = fill_progressive.seed_steps(world, settings)
                try:
                    while True:
                        try:
                            next(job)
                        except StopIteration as done:
                            seed = done.value
                            break
                        yield None
                except (fill.FillError, ValueError):
                    seed = None
                finally:
                    job.close()
                if seed is not None:
                    try:
                        if not props.settings(context).pen3d and self.state.view.srcs:
                            seed = yield from fill.project_steps(context, seed, self.state.view.srcs)
                    except fill.FillError:
                        seed = None
                    if seed is not None:
                        yield {'preview': seed}
            try:
                if legacy_error is not None:
                    if seed is not None:
                        return seed
                    raise fill.FillError(legacy_error)
                plan = yield from legacy_job
            except fill.FillError:
                if seed is None:
                    raise
                return seed
        finally:
            legacy_job.close()
        if not props.settings(context).pen3d and self.state.view.srcs:
            plan = yield from fill.project_steps(context, plan, self.state.view.srcs)
        if seed is not None:
            yield {'candidate': plan}
            if fill_progressive.utility(seed) > fill_progressive.utility(plan)+1e-12:
                return seed
        return plan

    def _crease_running(self):
        """the crease-fill search for this outline is still working."""
        run = getattr(self, '_crease_run', None)
        return run is not None and not run['done']

    def _crease_tail(self, context):
        """After the main search ends (with or without a candidate; not after a certain refusal), Topo mode on an even
        outline: the crease-fill search (fill_crease.tail_steps: detection, a quick crease fill, the wider search) runs
        in idle slices before the second search, so it never slows or changes the main search. Its fills join the
        candidates. True when it used this tick."""
        if (self.loop is None or self._busy() or self.state.panels or getattr(self, 'newfill_active', False)
                or self._fallback_here() or getattr(self, 'fail_marks', None) is not None
                or str(getattr(self, 'error', '') or '').startswith(_DEFINITE_REFUSALS)):
            return False
        pre = getattr(self, '_fb_pre', None)
        if pre is not None and pre['ids'] != tuple(v.index for v in self.loop):
            return False   # a second-search run of another outline is dropped first (_fallback_precompute)
        run = getattr(self, '_crease_run', None)
        if run is None:
            if getattr(self, '_crease_ids', None) == ('run', self.key):   # once per outline and options
                return False
            self._crease_ids = ('run', self.key)
            from ..core import fill_crease, fill_progressive
            if len(self.loop) % 2 or props.settings(context).pen3d or not self.state.view.srcs:
                return False
            settings = self._settings(context)
            if not fill_progressive.eligible(settings):
                return False
            world = fill.world_points(self.state.mw, self.loop)
            run = self._crease_run = dict(gen=fill_crease.tail_steps(world, self.state.view.srcs, settings,
                                                                     self.portfolio.crease_rank,
                                                                     base=self.plan is not None),
                                          done=False, srcs=self.state.view.srcs)
        if run['done']:
            return False
        deadline = time.perf_counter() + STEP4_IDLE_SECONDS
        try:
            while time.perf_counter() < deadline:
                if run['srcs'] is not self.state.view.srcs:   # the sources were read again: drop this run
                    self._crease_drop()
                    return True
                update = next(run['gen'])
                if isinstance(update, dict) and 'crease' in update:
                    self._crease = update['crease']
                elif isinstance(update, dict) and 'candidate' in update:
                    self._crease_join(update['candidate'])
                    break
        except StopIteration as done:
            run['done'] = True
            if done.value is not None and run['srcs'] is self.state.view.srcs:
                self._crease_join(done.value)
        except Exception as exc:   # noqa: BLE001  (Fill continues without the crease fill)
            run['done'] = True
            _log_engine_error('crease search', exc)
        return True

    def _crease_drop(self):
        """a crease run whose sources were read again: closed, and its fills taken out of the candidates (the shown
        fill follows the default when it was one of them)."""
        run = getattr(self, '_crease_run', None)
        if run is not None:
            run['gen'].close()
            run['done'] = True
        self._crease = None
        pf = self.portfolio
        from ..core import fill_layouts
        keys = [k for k in pf.entries if fill_layouts._is_crease(k)]
        for k in keys:
            pf.entries.pop(k, None)
            pf.placements.pop(k, None)
            pf.plain_seen.pop(k, None)
            if pf.selected == k:
                pf.selected, pf.locked = None, False
            if getattr(pf, 'default', None) == k:
                pf.default = None
        if keys and (self.plan is None or self.plan.get('_crease_candidate')):
            self.plan = pf.plan
            self._shown_plan = self.plan

    def _crease_join(self, plan):
        """a crease fill joins the portfolio (it replaces a weaker crease fill). As for the second search's fill: a new
        #1 is shown when the user is on entry 1, the user's own fill stays shown and selected otherwise, and "Better
        result found" either way."""
        pf = self.portfolio
        before = pf._plan_of(getattr(pf, 'default', None))
        kept = self.plan
        if not pf.add(plan):
            return
        best = pf._plan_of(getattr(pf, 'default', None)) if pf.plan is not None else None
        listed = kept is not None and (kept is pf.legacy or any(p is kept for k, (s, p) in pf.entries.items()))
        if kept is not None and not listed:   # the user's fill was the weaker crease fill it replaced
            if not pf.locked:
                pf.selected = None
            self.plan, self.error = pf.plan, ''
            self._shown_plan = self.plan
            if best is not before:
                self._note_better()
            return
        if best is None or best is before:
            return
        parked = getattr(self, 'wheel_touched', False) and not self.__dict__.get('_wheel_first', False)
        on_first = kept is before and not (pf.locked and parked)
        if on_first:
            pf.selected, pf.locked = None, False
            self.plan, self.error = pf.plan, ''
        elif not pf.locked:   # the user's fill stays shown AND selected
            from ..core import fill_layouts
            key = (fill_layouts.LEGACY_KEY if kept is pf.legacy
                   else next((k for k, (s, p) in pf.entries.items() if p is kept), None))
            if key is not None:
                pf.selected, pf.locked = key, True
        self._shown_plan = self.plan
        self._note_better()

    def _refine_steps(self, context, world, settings):
        return (yield from PPO_OT_fill._refine_body(self, context, world, settings))

    def _refine_body(self, context, world, settings):
        if getattr(self, 'floor_refusal', False) or getattr(self, 'fail_marks', None) is not None:
            return None
        if not getattr(self, "cuts_searched", False):
            yield from self._cuts_steps(context, world, settings)
        search_plan = getattr(getattr(self, 'portfolio', None), 'search_plan', self.plan)
        from ..core import fill_progressive
        settings = dict(settings, _network_seeded=getattr(self, 'network_seeded', False))
        fast = PPO_OT_fill._fast_q(self, context, world)   # Fast mode Q: canonical waits for the fallback
        seed = None
        if fast:
            plan = None
            if not settings.get('_network_seeded'):   # the network seed Full runs at the start of refine_steps
                seed = yield from PPO_OT_fill._canonical_steps(self, context, world, dict(settings, _network_only=True),
                                                               search_plan)
                plan = seed
        else:
            plan = yield from PPO_OT_fill._canonical_steps(self, context, world, settings, search_plan)
        if plan is not None:
            if not props.settings(context).pen3d and self.state.view.srcs:
                plan = dict(plan, positions=plan['positions'].copy())
                plan = yield from fill.project_steps(context, plan, self.state.view.srcs)
            yield {'candidate': plan}
        self.legacy_searching = True
        legacy = yield from fill_progressive.legacy_steps(world, settings, getattr(self, 'legacy_frames', ()))
        if legacy is not None:
            try:
                if not props.settings(context).pen3d and self.state.view.srcs:
                    legacy = dict(legacy, positions=legacy['positions'].copy())
                    legacy = yield from fill.project_steps(context, legacy, self.state.view.srcs)
            except fill.FillError:
                legacy = None
            if legacy is not None:
                yield {'candidate': legacy}
        if fast:
            from ..core import fill_flow
            portfolio = getattr(self, 'portfolio_nd', None)   # without the direct grid (it never steers the search)
            if portfolio is None:
                portfolio = self.portfolio
            portfolio.plan   # refresh the automatic default (decide on it, not on a wheel selection)
            best = portfolio._plan_of(getattr(portfolio, 'default', None)) or portfolio.plan
            self.fast_min_q = fill_flow.r1_min_sj(best) if best is not None else None
            if (best is not None and fill_flow.eligible(best, portfolio.quality_floor)
                    and self.fast_min_q >= FAST_MIN_Q):
                yield {'settled': True}
                return None   # Fast: no canonical or field
            # Full, reusing initial / A / P1 / main search (same incumbent as Full)
            self.legacy_searching = False   # the HUD shows the search continuing
            if seed is not None:   # continue exactly as Full's refine_steps after its seed step
                plan = yield from PPO_OT_fill._canonical_steps(self, context, world, dict(settings, _network_seeded=True),
                                                               seed)
            else:
                plan = yield from PPO_OT_fill._canonical_steps(self, context, world, settings, search_plan)
            if plan is not None:
                if not props.settings(context).pen3d and self.state.view.srcs:
                    plan = dict(plan, positions=plan['positions'].copy())
                    plan = yield from fill.project_steps(context, plan, self.state.view.srcs)
                yield {'candidate': plan}
        return (yield from PPO_OT_fill._refine_tail(self, context, world, settings))

    def _fast_q(self, context, world):
        """Fast mode Q applies to even AUTO rims of a single panel with classification candidates ON."""
        return (PPO_OT_fill._legacy_pref(self, context, 'fill_search_mode', 'FAST') == 'FAST' and len(world) % 2 == 0
                and not self.state.panels)

    def _canonical_steps(self, context, world, settings, incumbent):
        from ..core import fill_progressive
        job = fill_progressive.refine_steps(world, settings, incumbent)
        try:
            while True:
                try:
                    update = next(job)
                except StopIteration as done:
                    plan = done.value
                    break
                if isinstance(update, dict) and ('preview' in update or 'candidate' in update):
                    proposed = update.get('preview', update.get('candidate'))
                    if not props.settings(context).pen3d and self.state.view.srcs:
                        proposed = dict(proposed, positions=proposed['positions'].copy())
                        proposed = yield from fill.project_steps(context, proposed, self.state.view.srcs)
                    yield {'candidate': proposed}
                else:
                    yield None
        finally:
            job.close()
        return plan

    def _refine_tail(self, context, world, settings):
        """plan + P1 + canonical + main search are done; #1 is settled here; then the supplementary families."""
        yield {'settled': True}
        from ..core import fill_field
        self.field_searching = True
        self.legacy_searching = False
        # Defer field setup until all released paths have been published.
        yield None
        job = fill_field.search_steps(world, settings, getattr(getattr(self, 'portfolio', None), 'search_plan', self.plan))
        try:
            while True:
                try:
                    update = next(job)
                except StopIteration:
                    break
                if isinstance(update, dict) and 'candidate' in update:
                    proposed = update['candidate']
                    try:
                        if not props.settings(context).pen3d and self.state.view.srcs:
                            proposed = dict(proposed, positions=proposed['positions'].copy())
                            proposed = yield from fill.project_steps(context, proposed, self.state.view.srcs)
                        yield {'candidate': proposed}
                    except fill.FillError:
                        yield None
                else:
                    yield None
        finally:
            job.close()
            self.field_searching = False
        return None

    def _accept_background_plan(self, plan):
        if plan is None:
            return
        direct = bool(plan.get('_direct'))
        aside = direct or bool(plan.get('_crease_candidate'))   # never steers the main search
        shadow = getattr(self, 'portfolio_nd', None)   # the portfolio without the direct grid (search decisions)
        if plan.get('_legacy_layout'):
            self.portfolio.add_legacy(plan, select_if_empty=getattr(self, '_legacy_view', self.plan) is None)
            if shadow is not None:
                shadow.add_legacy(plan, select_if_empty=getattr(self, '_legacy_view', self.plan) is None)
        else:
            self.portfolio.add(plan)
            if shadow is not None and not aside:
                shadow.add(plan)
        if shadow is not None and not aside and shadow.plan is not None:
            self._legacy_view = shadow.plan   # what self.plan would be without the direct grid
        if self.portfolio.plan is not None:
            self.plan, self.error = self.portfolio.plan, ''
        if direct and self.plan is not None:
            self._show_direct()

    def _show_direct(self):
        """A direct grid shows at once: it is the provisional #1 (no settle-first wait) and the display is
        progressive from here (the shown plan is held; a stronger later #1 is announced)."""
        if getattr(self, 'settle_pending', False) and not getattr(self, 'provisional', False):
            self.provisional = True
            self.settled_signature = self._default_signature()
        self._went_progressive = True

    # settle #1 early. Search order and results are
    # unchanged; only what is shown and when a click may commit.
    def _computing(self):
        """Settle ON and #1 not settled yet: no preview, no wheel, no commit."""
        return (getattr(self, 'settle_pending', False) and not getattr(self, 'provisional', False)
                and self.loop is not None and (self.job is not None or self.background is not None))

    def _unseen(self):
        """A main-search plan exists but the viewport does not draw it yet (_paint draws only in the 'done' and
        'progressive' stages): no wheel, no commit. The fallback search keeps its own first-result rules."""
        return (self.plan is not None and self.loop is not None and not getattr(self, 'newfill_active', False)
                and self._display_stage() not in ('done', 'progressive'))

    def _fallback_drawable(self):
        """The silent fallback holds its first result while its search goes on: the viewport draws it, because a
        click adds exactly this plan."""
        return (getattr(self, 'newfill_active', False) and getattr(self, 'fr_shown', False)
                and self.plan is not None and self.loop is not None)

    def _undrawn(self):
        """A fallback plan that no viewport frame has drawn yet: no commit until it is seen."""
        return (getattr(self, 'newfill_active', False) and self.plan is not None
                and getattr(self, '_drawn_plan', None) is not self.plan)

    def _reveal(self):
        """no blank wait beyond PROVISIONAL_SECONDS (none with Show Debug Info); #1 shows as
        provisional until it settles."""
        if (self.plan is not None and self._computing()
                and time.perf_counter()-getattr(self, 'replan_started', time.perf_counter())
                >= getattr(self, 'reveal_seconds', PROVISIONAL_SECONDS)):
            self.provisional = True
            self.settled_signature = self._default_signature()
            return True
        return False

    def _provisional(self):
        return getattr(self, 'settle_pending', False) and getattr(self, 'provisional', False)

    def _default_signature(self):
        """Topology and placement of entry 1 (a better placement of the same layout counts too)."""
        from ..core import fill_layouts, fill_flow
        portfolio = self.portfolio
        plan = portfolio.plan   # refreshes the guarded default
        default = portfolio._plan_of(getattr(portfolio, 'default', None)) or plan
        if default is None:
            return None
        return fill_layouts.signature(default), fill_flow.fingerprint(default)

    def _settle(self):
        if not getattr(self, 'settle_pending', False):
            return
        self.settle_pending = False
        current = self._default_signature()
        if getattr(self, 'provisional', False) and current is not None and current != self.settled_signature:
            self._note_better()   # the provisional #1 shown so far was replaced
        self.settled_signature = current

    def _display_stage(self):
        """display timing: 'done' (not searching: show everything at once),
        'waiting' (searching, the "… / …" box), 'progressive' (searching, >= T: confirmed candidates as they arrive)."""
        if not self._busy():
            return 'done'
        if self._fallback_here():
            return 'waiting'   # "Searching wider...": busy continuous across the hand-over
        elapsed = time.perf_counter() - getattr(self, 'replan_started', time.perf_counter())
        if elapsed < PROGRESSIVE_SECONDS and not getattr(self, '_went_progressive', False):
            return 'waiting'
        return 'progressive'

    def _hold_shown(self):
        """after T the shown candidate never changes on its own; a stronger late one is announced instead."""
        stage = self._display_stage()
        if stage != 'done':
            self.__dict__['_resort_due'] = True   # work is running; re-sort when it ends
        elif (self.__dict__.pop('_resort_due', False) and self.loop is not None and self.plan is not None
                and not self._fallback_here()):
            self._finish_resort()
            return
        if stage == 'progressive':
            self._went_progressive = True
        if not getattr(self, '_went_progressive', False) or self.loop is None:
            return
        shown = getattr(self, '_shown_plan', None)
        if getattr(self, 'wheel_touched', False) or shown is None:
            self._shown_plan = self.plan   # the user's choice (or the first confirmed one) is the shown one
            return
        if self.plan is not shown and self.plan is not None:
            from ..ui import fill_hud
            listed = fill_hud._more_list(self)[0]
            if any(p is shown for p in listed):   # the shown candidate is still a listed one: keep showing it
                self.plan = shown
                self._note_better()
            else:   # replaced (e.g. its settled version): follow it, no announcement
                self._shown_plan = self.plan

    def _finish_resort(self):
        """the finished search's list in the ranked order; the user's place is kept by identity."""
        from ..ui import fill_hud
        from ..core import fill_layouts
        before, cur = fill_hud.displayed_plans(self)
        first_before = before[0] if before else None
        on_first = (not getattr(self, 'wheel_touched', False)
                    or self.__dict__.get('_wheel_first', cur in (0, None)))   # as the user left it
        kept = self.plan
        self._went_progressive, self._shown_plan = False, None
        self.__dict__['_arrival'] = []
        odd = self.triangle_selected and self.loop is not None and len(self.loop) % 2 and getattr(self, 'odd_wheel', None)
        if odd:
            self.odd_wheel = self._odd_list()
            best = self.odd_wheel[0] if self.odd_wheel else None
            if on_first or not any(p is kept for p in self.odd_wheel):
                self.odd_index = 0
                if best is not None and best is not kept:
                    self.plan = best
                    self._note_better()
            else:
                self.odd_index = next(i for i, p in enumerate(self.odd_wheel) if p is kept)
                if best is not first_before:
                    self._note_better()
            return
        pf = self.portfolio
        parked_key = pf.selected if (pf.locked and not on_first) else None   # the user's CANDIDATE (key)
        pf.selected, pf.locked = None, False   # the ranking rule picks rank 1 again (a wheel lock froze it)
        best = pf.plan
        if on_first:
            if best is not None and best is not kept:
                self.plan = best
                self._note_better()
            return
        plan_of = lambda k: pf.legacy if k == fill_layouts.LEGACY_KEY else pf.entries[k][1]
        if parked_key is not None and (parked_key == fill_layouts.LEGACY_KEY or parked_key in pf.entries):
            # lock the parked CANDIDATE before listing, so a list limited to its top entries keeps it (the new
            # rank 1 stays first: the order puts the rule's default first while locked)
            pf.selected, pf.locked = parked_key, True
        keys = pf.ranked_keys()
        if parked_key is not None and parked_key in keys:   # same candidate, possibly its settled plan
            key = parked_key
            kept = plan_of(key)
        else:
            key = next((k for k in keys if plan_of(k) is kept), None)
        if key is None:   # the parked fill is gone (replaced by its settled version): show rank 1
            if best is not None and best is not kept:
                self.plan = best
                self._note_better()
            return
        pf.selected, pf.locked = key, True   # stays on the user's fill, at its new index (no jump)
        self.plan = kept
        if best is not first_before:
            self._note_better()   # a new #1 appeared: the user can wheel up to it

    def _legacy_busy(self):
        return (self.loop is not None and not self._fallback_here()
                and (self.job is not None or self.background is not None or self.triangle_job is not None))

    def _legacy_cap(self, context):
        """a main search still running SEARCH_CAP_SECONDS after it started ends with its best so far."""
        if not self._legacy_busy() or getattr(self, 'newfill_active', False):
            return
        if time.perf_counter() - getattr(self, 'replan_started', time.perf_counter()) < SEARCH_CAP_SECONDS:
            return
        best = self.plan
        self._cancel_job()
        self.plan = best
        if best is None and self.state.panels and not self.error:   # a panel network has no fallback
            self.error = "Panel search took too long; select fewer panels"
            self.report({"WARNING"}, tr_rpt(fill.user_text(self.error, _debug(None))))   # also on the cursor line
        _tag_redraw(context)

    def _fallback_precompute(self, context):
        """with no main-search candidate after FALLBACK_PRE_SECONDS, step the second search in idle
        slices (never shown). Step 4: also started when the main search ends with a candidate; not dropped when
        the main search finds one, but never stepped while the main search runs with a candidate (it keeps its
        share of Blender's time); dropped when the outline or the options its mesh depends on (3D mode, projection
        sources) change.
        a result a budget cut short is not cached (hidden=True); it is still the fallback shown for this
        outline. its S part runs on an active-time clock (only the time spent in its own steps
        counts, not the gaps between idle slices); the takeover switches it to real time."""
        pre = getattr(self, '_fb_pre', None)
        ids = tuple(v.index for v in self.loop) if self.loop is not None else None
        if pre is not None and (ids != pre['ids'] or self._fallback_here()):   # step 4: kept when a candidate exists
            if not self._fallback_here():
                self._drop_fallback_pre()
            return
        if pre is not None and not self._step4_matches(pre, context):   # 3D mode / projection sources changed
            self._step4_drop_stale()
            pre = None
        if (pre is None and not self.state.panels and not getattr(self, 'newfill_active', False)
                and getattr(self, '_step4_blocked', None) is not self.portfolio   # dropped as stale: next re-plan
                and not getattr(self, '_surface_pending', False)   # the outline's layout is still being made
                and ((self._legacy_busy() and self.plan is None
                      and time.perf_counter() - getattr(self, 'replan_started', time.perf_counter()) >= FALLBACK_PRE_SECONDS)
                     or (self.loop is not None and self.plan is not None and not self._busy()
                         and not self._fallback_here()))):   # step 4: the main search ended with a candidate
            world = self._plan_world()
            from ..core.newfill import pipeline
            token = pipeline.cancel_token()   # this run's own token
            clock = dict(active=True, paused=0.)
            pre = self._fb_pre = dict(ids=ids, geo=self._loop_geo(), cancel=token, clock=clock,
                                      pen3d=bool(props.settings(context).pen3d), srcs=self.state.view.srcs,
                                      surface=getattr(self, '_surface', None),
                                      portfolio=self.portfolio, step4=self.plan is not None,
                                      gen=self._newfill_steps(context, world, self._settings(context), cancel=token,
                                                              hidden=True, clock=clock),
                                      done=False, value=None, exc=None,
                                      state={a: (list(v) if isinstance(v, list) else v) for a, v in self.__dict__.items()
                                             if a in self._HIDDEN_ATTRS})   # starts as a copy of the shown values
        if pre is None or pre['done']:
            return
        if self._legacy_busy() and self.plan is not None:
            return   # step 4: paused while the main search runs with a candidate (resumed when it ends)
        idle = self.plan is not None   # step 4 after the main search's end (a busy one with a candidate returned above)
        deadline = time.perf_counter() + (STEP4_IDLE_SECONDS if idle else FALLBACK_PRE_SLICE)
        visible = self._swap_hidden_state(pre['state'])   # the hidden run never touches the shown one
        try:
            while time.perf_counter() < deadline:
                try:
                    u = next(pre['gen'])
                    if isinstance(u, dict) and u.get('work_wait'):
                        break
                except StopIteration as done:
                    pre['done'], pre['value'] = True, done.value
                    break
                except Exception as exc:   # noqa: BLE001  (the fallback reports it when it is shown)
                    pre['done'], pre['exc'] = True, exc
                    break
        finally:
            pre['state'] = self._swap_hidden_state(visible)

    _HIDDEN_ATTRS = ('newfill_result', 'newfill_entries', 'newfill_index', 'better_until', 'fr_shown', '_fr_prov')

    def _step4_matches(self, pre, context):
        """the hidden second-search run's mesh still fits: the same 3D mode setting and, with 3D mode off, the same
        projection sources (the view's list, by identity: a refresh swaps in a new list) and the same layout on the
        source surface (_start_job). With 3D mode on they are not used."""
        pen3d = bool(props.settings(context).pen3d)
        return (pre.get('pen3d') == pen3d and (pen3d or pre.get('srcs') is self.state.view.srcs)
                and pre.get('surface') is getattr(self, '_surface', None))

    def _step4_drop_stale(self):
        """a step-4 run whose mesh no longer fits (options or outline geometry changed): dropped; when it had joined
        the current portfolio its entry is taken out and the shown fill restored; no new run for this portfolio when
        the dropped run had joined it or was started for it as a step-4 run (the main search already had a
        candidate); the next re-plan starts one. A fallback precompute (started with no main-search candidate)
        restarts as before."""
        pre = getattr(self, '_fb_pre', None)
        pf = getattr(self, 'portfolio', None)
        joined = pre is not None and pf is not None and pre.get('joined') is pf
        if pre is not None and pf is not None and (joined or (pre.get('step4') and pre.get('portfolio') is pf)):
            self._step4_blocked = pf
        self._drop_fallback_pre()
        if joined:
            self._step4_unjoin()

    def _step4_unjoin(self):
        """take the second search's entry out of the current portfolio again; the shown fill and pf.plan without
        it (the user's fill stays shown unless it was the second search's; a "Better result found" note for the
        removed fill is cleared)."""
        pf = self.portfolio
        shown = self.plan
        try:
            first = pf._plan_of(getattr(pf, 'default', None))
        except Exception:
            first = None
        self._step4_remove(pf)
        try:
            best = pf.plan   # the default without the second search's fill (an unlocked portfolio selects it)
            odd = getattr(self, 'odd_wheel', None)
            new_shown = shown is not None and bool(shown.get('_step4_candidate'))
            if odd:
                at_first = new_shown or getattr(self, 'odd_index', 0) == 0
                wheel = self._odd_plans(pf, best)
                if not at_first and not any(p is shown for p in wheel):
                    wheel.append(shown)
                at = 0 if at_first else next(i for i, p in enumerate(wheel) if p is shown)
                self.odd_wheel, self.odd_index, self.plan = wheel, at, wheel[at]
            elif new_shown or not pf.locked:
                self.plan = pf.plan
            self._shown_plan = self.plan
            if self.plan is not shown or pf._plan_of(getattr(pf, 'default', None)) is not first:
                self.better_until = 0.0   # the note announced the fill just taken out
        except Exception as exc:
            _log_engine_error('step 4 cleanup', exc)

    @staticmethod
    def _step4_candidate(plan):
        """step 4: the second search's pick as a regular main-search candidate: the same mesh, without the
        fallback-wheel marker (the main search's wheel and HUD show it), marked _step4_candidate (the portfolio's
        final-pick guard)."""
        out = {k: v for k, v in plan.items() if k != '_newfill_lane'}
        out['_step4_candidate'] = plan.get('_newfill_lane') or True
        return out

    @staticmethod
    def _step4_remove(pf):
        """take the step-4 entries out again (a join that raised, a stale run): the main search's candidates as
        before the join. By the ('step4',) key suffix in entries, placements and plain_seen, and any entry marked as
        the second search's."""
        try:
            keys = {k for d in (pf.entries, pf.placements, pf.plain_seen) for k in d
                    if isinstance(k, tuple) and k and k[-1] == 'step4'}
            keys.update(k for k, (s, p) in pf.entries.items() if p.get('_step4_candidate'))
            for k in keys:
                pf.entries.pop(k, None)
                pf.placements.pop(k, None)
                pf.plain_seen.pop(k, None)
                if pf.selected == k:
                    pf.selected, pf.locked = None, False
                if getattr(pf, 'default', None) == k:
                    pf.default = None
            pf._refresh_legacy()
        except Exception as exc:
            _log_engine_error('step 4 cleanup', exc)

    def _step4_running(self):
        """the hidden second-search run for this outline is still working (the timer stays fast)."""
        pre = getattr(self, '_fb_pre', None)
        return pre is not None and not pre['done'] and not self._fallback_here()

    def _step4_tail(self):
        """step 4, the main search done: the second search still works (the HUD says "Improving fill preview...")."""
        return (self.loop is not None and self.plan is not None and not self._busy() and not self.state.panels
                and not getattr(self, 'newfill_active', False)
                and (self._step4_running() or PPO_OT_fill._crease_running(self)))

    def _step4_join(self, context):
        """step 4: when the hidden second-search run is done and the main search ended with a candidate, the
        second search's pick joins the portfolio (once per portfolio). A finished run whose outline geometry or
        options changed is dropped first (_step4_drop_stale: its entry taken out of this portfolio; the next
        re-plan runs the second search again). A new #1 is shown
        when the user is on entry 1; "Better result found" either way."""
        pre = getattr(self, '_fb_pre', None)
        if (pre is None or not pre['done'] or self.loop is None or getattr(self, 'newfill_active', False)
                or self._fallback_here()):
            return
        if pre.get('joined') is self.portfolio and self.key is not None:
            geo = self.key[2]   # joined: the re-plan key's outline coordinates (no vertex read per tick)
        else:
            geo = self._loop_geo()
        if pre.get('geo') != geo or not self._step4_matches(pre, context):   # moved / options changed
            self._step4_drop_stale()
            return
        if pre.get('joined') is self.portfolio or self.plan is None or self._busy() or self.state.panels:
            return
        pre['joined'] = pf = self.portfolio
        if pre['exc'] is not None or pre['value'] is None:
            return
        before = pf._plan_of(getattr(pf, 'default', None))
        kept = self.plan
        try:
            if not pf.add(self._step4_candidate(pre['value'])):
                return
            pf.plan   # refreshes the default: rule (a) and the step-4 guard
            best = pf._plan_of(getattr(pf, 'default', None))
        except Exception as exc:   # never reaches the modal handler: the main search's candidates as before
            _log_engine_error('step 4', exc)
            self._step4_remove(pf)
            return
        if best is None or best is before:
            return   # the second search's fill is a wheel entry; #1 is unchanged
        odd = getattr(self, 'odd_wheel', None)
        # a fill the user wheeled to (and did not wheel back to entry 1 from) is never replaced by a later New #1
        parked = getattr(self, 'wheel_touched', False) and not self.__dict__.get('_wheel_first', False)
        if odd:   # the odd wheel never locks the portfolio; after a re-sort the shown plan is always its entry 1
            on_first = getattr(self, 'odd_index', 0) == 0 and not parked
        else:
            on_first = kept is before and not (pf.locked and parked)   # what is shown, unless a locked wheel choice
        if odd:
            wheel = self._odd_plans(pf, best)
            if not on_first and not any(p is kept for p in wheel):
                wheel.append(kept)   # the user's fill is never lost (wheel limit or the one-per-layout dedupe)
            at = 0 if on_first else next(i for i, p in enumerate(wheel) if p is kept)
            self.odd_wheel, self.odd_index, self.plan = wheel, at, wheel[at]
        elif on_first:
            pf.selected, pf.locked = None, False
            self.plan = pf.plan
        elif not pf.locked:   # the user's fill stays shown AND selected
            from ..core import fill_layouts
            key = (fill_layouts.LEGACY_KEY if kept is pf.legacy
                   else next((k for k, (s, p) in pf.entries.items() if p is kept), None))
            if key is not None:
                pf.selected, pf.locked = key, True
        self._shown_plan = self.plan
        self._note_better()

    def _swap_hidden_state(self, state):
        """install `state` for the attributes the New steps write; return the ones replaced."""
        old = {a: self.__dict__[a] for a in self._HIDDEN_ATTRS if a in self.__dict__}
        for a in self._HIDDEN_ATTRS:
            self.__dict__.pop(a, None)
        self.__dict__.update(state)
        return old

    def _drop_fallback_pre(self):
        """drop the hidden fallback precompute; cancel its run's background jobs."""
        pre, self._fb_pre = getattr(self, '_fb_pre', None), None
        if pre is None:
            return
        token = pre.get('cancel')
        if token is not None:
            token.set()
        try:
            pre['gen'].close()
        except Exception:
            pass

    def _loop_geo(self):
        """the outline's vertex coordinates (the precompute takeover matches geometry, not indices)."""
        return tuple(tuple(v.co) for v in self.loop) if self.loop is not None else None

    @staticmethod
    def _finished_job(pre):
        """A job that ends at once with the precomputed fallback's result (or raises its error)."""
        if pre['exc'] is not None:
            raise pre['exc']
        return pre['value']
        yield None

    def _note_better(self):
        self.better_until = time.perf_counter() + BETTER_SECONDS

    def _check_better(self):
        if not getattr(self, 'settle_on', False) or (self.settle_pending and not getattr(self, 'provisional', False)):
            return
        current = self._default_signature()
        if getattr(self, 'settled_signature', None) is None:
            self.settled_signature = current   # first result after settling with none
            return
        if current is not None and current != self.settled_signature:
            self.settled_signature = current
            self._note_better()

    def _better_note(self):
        return time.perf_counter() < getattr(self, 'better_until', 0.0)

    def _start_background(self, context):
        if self.state.panels:
            return
        settings = self._settings(context)   # always AUTO; no layout / anchors / rotation / shift
        self.background = self._refine_steps(context, self._plan_world(), settings)

    def _replan(self, context):
        if self.loop is None:
            self._cancel_job()
            self.plan, self.error, self.key = None, self.state.selection_error, None
            self.fail_marks = None
            return
        settings = self._settings(context)
        key = (tuple(v.index for v in self.loop),settings["pattern"],
               tuple(tuple(v.co) for v in self.loop),props.settings(context).pen3d,
               getattr(prefs.get(context), 'fill_search_mode', 'FAST'),
               self._fallback_here())   # New only as the fallback
        if key == self.key:
            return
        self._cancel_job()
        self.triangle_selected = False
        self.odd_wheel = None
        self.legacy_searching = False
        self.wheel_touched = False   # per preview
        self.key = key
        self._refused_ids = None   # a new search; a commit refusal of the old one no longer applies
        from ..core import fill_progressive
        # odd rims skip settle-first: their triangle stage runs after the settle
        # mark and swaps #1 in most cases, so settling early only added a blank wait.
        self.settle_pending = bool(not self.state.panels   # settle first / auto partition fixed on
                                   and fill_progressive.eligible(settings) and len(self.loop) % 2 == 0)
        self.provisional = False
        self.settle_on = self.settle_pending
        self.settled_signature = None
        self.better_until = 0.0
        from ..core.fill_layouts import Portfolio
        self.portfolio = Portfolio(ranking='FLOW', quality_floor=QUALITY_FLOOR)
        self.portfolio.rank_r1 = True   # default pick and order by rule R1 delta 3
        self._step4_blocked = None   # a new portfolio: step 4 may run again (the old one is not kept alive)
        self.portfolio_nd = Portfolio(ranking='FLOW', quality_floor=QUALITY_FLOOR)   # the same, without the direct grid
        self.portfolio_nd.rank_r1 = True
        self._legacy_view = None
        self._crease = self._crease_ids = None   # the source crease of this outline (the crease-fill search runs again)
        self.plan, self.error = None, ""
        self.fail_marks = None   # certain_fail markers of this outline
        world = fill.world_points(self.state.mw, self.loop)
        self.newfill_entries, self.newfill_index, self.newfill_result = [], 0, None
        self._fr_prov, self.fr_shown, self.fr_touched = None, False, False   # first-result deadline
        self.newfill_active = (self._fallback_here()
                               and not self.state.panels)   # the second search runs only as the silent fallback
        if self.newfill_active:
            self.settle_pending = self.settle_on = False
            from ..core.newfill import pipeline
            self._job_cancel = pipeline.cancel_token()   # set by _cancel_job
            self._start_job(context, world, lambda w: self._newfill_steps(context, w, settings, cancel=self._job_cancel))
        else:
            self._start_job(context, world, lambda w: self._plan_steps(context, w, settings))
        self.replan_started = time.perf_counter()
        self._went_progressive, self._shown_plan = False, None   # display timing: per search
        self.reveal_seconds = 0.0 if _debug(context) else PROVISIONAL_SECONDS   # debug shows #1 at once
        _tag_redraw(context)

    def _start_job(self, context, world, make):
        """the search job of this outline: make(outline). Topo mode, one even outline: the outline at its flat
        positions on the source surface when there is a patch under it (core/fill_surface; fill.project_steps maps
        each candidate back onto the source), else the outline itself as before."""
        self._surface, self._surface_pending = None, False
        if props.settings(context).pen3d or not self.state.view.srcs or self.state.panels or len(world) % 2:
            self.job = make(world)
            return
        from ..core import fill_surface
        known, surface = fill_surface.known(world, self.state.view.srcs)
        if known:
            self._surface = surface
            self.job = make(world if surface is None or surface.plain else surface.flat)
        else:
            self._surface_pending = True
            self.job = self._surface_steps(world, make)

    def _surface_steps(self, world, make):
        """the patch under the outline is flattened cooperatively, then the search runs."""
        from ..core import fill_surface
        surface = yield from fill_surface.prepare_steps(world, self.state.view.srcs)
        self._surface, self._surface_pending = surface, False
        return (yield from make(world if surface is None or surface.plain else surface.flat))

    def _plan_world(self):
        """the outline as this outline's planners see it (see _start_job)."""
        world = fill.world_points(self.state.mw, self.loop)
        surface = getattr(self, '_surface', None)
        return world if surface is None or surface.plain else surface.flat

    def _surface_fallback(self, context):
        """the main search on the flat outline ended with no fill (every candidate refused), or has none after
        SURFACE_SECONDS: the outline is planned again as before (on its plane, then moved onto the source) with its
        own time cap, as today. A second-search run already planned on the flat outline stays valid (its candidates
        are mapped)."""
        surface = getattr(self, '_surface', None)
        if (surface is None or surface.plain or self.loop is None or getattr(self, 'newfill_active', False)
                or self.plan is not None or self.triangle_job is not None):
            return
        if self.job is not None or self.background is not None:
            if time.perf_counter() - getattr(self, 'replan_started', time.perf_counter()) < SURFACE_SECONDS:
                return
            self._cancel_job()   # no candidate on the source in time: the plane search below
        surface.plain = True
        self.key = None
        self._replan(context)

    def _total_entries(self):
        """the number of New (fallback) entries."""
        return len(self.newfill_entries)

    def _fallback_here(self):
        """the second search runs as the silent fallback for this outline (the main search ended with no
        candidate for it)."""
        loop = getattr(self, 'loop', None)
        return loop is not None and getattr(self, '_fallback_ids', None) == tuple(v.index for v in loop)


    def _odd_list(self):
        """odd wheel: entry 1 = the auto site plan, then the pattern entries."""
        return self._odd_plans(self.portfolio, self.plan)

    @staticmethod
    def _odd_plans(pf, first):
        from ..core import fill_layouts
        out = [first] + [pf.legacy if k == fill_layouts.LEGACY_KEY else pf.entries[k][1] for k in pf.ranked_keys()]
        kept, seen = [], set()   # the auto site plan is also a pattern entry; one entry per layout
        for p in out:
            try:
                sig = fill_layouts.signature(p)
            except Exception:
                sig = ('id', id(p))
            if sig not in seen:
                seen.add(sig)
                kept.append(p)
        return kept

    def _newfill_steps(self, context, world, settings, cancel=None, hidden=False, clock=None):
        """the second search: one pipeline generator; A/E/dopt as worker jobs, S (the main search) stepped here.
        cancel: the run's token (pipeline.cancel_token). hidden: the fallback precompute. clock: its
        S-part active-time clock."""
        from ..core import fill_layouts, fill_triangle
        from ..core.newfill import pipeline
        from . import fill as fill_op   # this module (the S part steps PPO_OT_fill's planner methods)
        s_ctx = dict(fill=fill, layouts=fill_layouts, op=fill_op, triangle=fill_triangle, context=context)
        if clock is not None:
            s_ctx['clock'] = clock
        import copy
        key_ = _rim_key(world)   # the options S reads are fixed; the outline is the key
        hit_ = _RIM_CACHE.pop(key_, None)   # a repeat fill of the same outline reuses its result
        if hit_ is not None:
            _RIM_CACHE[key_] = hit_   # most recently used last
        gen = None if hit_ is not None else pipeline.steps(world, s_ctx, cancel=cancel)
        result = copy.deepcopy(hit_) if hit_ is not None else None
        try:
            while gen is not None:
                try:
                    u = next(gen)
                except StopIteration as done:
                    result = done.value
                    break
                if isinstance(u, dict) and 'provisional' in u:   # first-result deadline
                    try:
                        prov = pipeline.to_plan(u['provisional'], u.get('lane', 'A'), lift=True)
                        fill.validate_plan(prov)
                        if not props.settings(context).pen3d and self.state.view.srcs:
                            prov = yield from fill.project_steps(context, prov, self.state.view.srcs)
                        self._fr_prov = prov
                    except fill.FillError:
                        pass
                    yield None
                    continue
                yield ({'work_wait': True} if u == 'wait' else None)
        finally:
            if gen is not None:
                gen.close()
        if (hit_ is None and result is not None and result.get('cands')
                and not result.get('log', {}).get('cancelled')   # never a cancelled run
                and not result.get('log', {}).get('degraded')   # E / dopt / flow cut by a budget
                and not result.get('log', {}).get('exc')   # a search part ended with an error (e.g. a lost process)
                and not (hidden and _budget_bound(result.get('log', {})))):
            log_ = {k: v for k, v in result.get('log', {}).items() if k != 's_plans'}
            _RIM_CACHE[key_] = copy.deepcopy(dict(result, log=log_))
            while len(_RIM_CACHE) > _RIM_CACHE_MAX:
                _RIM_CACHE.pop(next(iter(_RIM_CACHE)))
        self.newfill_result = result
        log = result['log']
        entries = []
        for lane, q in result['cands']:
            plan = pipeline.to_plan(q['mesh'], lane, lift=True)
            try:
                fill.validate_plan(plan)
                if not props.settings(context).pen3d and self.state.view.srcs:
                    plan = yield from fill.project_steps(context, plan, self.state.view.srcs)
            except fill.FillError as exc:
                log.setdefault('rejected', []).append((lane, str(exc)))   # next candidate
                continue
            entries.append(plan)
        self._newfill_log(context, world, result, len(entries))
        if getattr(self, 'fr_shown', False):   # first-result deadline: A was shown at 1 s
            prov = getattr(self, '_fr_prov', None)
            if entries and entries[0].get('_newfill_lane') != 'A':
                if getattr(self, 'fr_touched', False) and prov is not None:
                    entries = [prov] + [e for e in entries if e.get('_newfill_lane') != 'A']   # wheel touched: A stays, S/E appended
                else:
                    self.better_until = time.perf_counter() + 1.5   # auto swap: "Better result found"
            self.fr_shown = False
        self._fr_prov = None
        self.newfill_entries, self.newfill_index = entries, 0
        if not entries:
            raise fill.FillError("No pattern found in time")
        return entries[0]

    def _newfill_log(self, context, world, result, kept):
        if not _debug(context):
            return
        try:
            import json
            from pathlib import Path
            lg = dict(result['log'], n=len(world), odd=len(world) % 2 == 1, lane=result['lane'], kept=kept,
                      cands=[k for k, _q in result['cands']])
            lg.pop('s_plans', None)
            print('[PolyTailor newfill]', json.dumps(lg, default=str))
            folder = Path(bpy.utils.extension_path_user(__package__.rpartition('.')[0], create=True))
            path = folder / 'newfill_log.jsonl'
            lines = path.read_text(encoding='utf-8').splitlines()[-199:] if path.exists() else []
            lines.append(json.dumps(lg, default=str))
            path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        except Exception:
            pass

    def _triangle_project(self, context):
        if not props.settings(context).pen3d and self.state.view.srcs:
            return lambda plan: fill.project_steps(context, plan, self.state.view.srcs)
        return None

    def _start_triangles(self, context):
        from ..core import fill_triangle, fill_progressive
        if (self.plan is not None and not self.state.panels and len(self.loop) >= 5
                and len(self.loop) % 2
                and fill_progressive.eligible(self._settings(context))):
            self.triangle_job = fill_triangle.candidates_steps(
                fill.world_points(self.state.mw, self.loop), self._settings(context),
                self.plan, self._triangle_project(context),
                reference=self.portfolio.search_plan)

    def _advance(self, context):
        previous, was_busy = self.plan, self._busy()
        self._advance_work(context)
        self._surface_fallback(context)   # no fill on the flat outline: planned again on the plane
        self._legacy_cap(context)   # 10 s total search cap
        if not self._crease_tail(context):   # the wider crease-fill search first, then the second search
            self._fallback_precompute(context)
        self._step4_join(context)   # step 4: the second search's pick joins the main-search candidates
        self._hold_shown()   # display timing
        if (self.loop is not None and self.plan is None and self.job is None and self.background is None
                and self.triangle_job is None and not getattr(self, 'newfill_active', False) and not self.state.panels
                and getattr(self, '_fallback_ids', None) != tuple(v.index for v in self.loop)
                and not str(getattr(self, 'error', '') or '').startswith(_DEFINITE_REFUSALS)
                and getattr(self, '_refused_ids', None) != tuple(v.index for v in self.loop)
                and not self._crease_running()):   # a crease fill may still come
            # the main search ended with no candidate for this outline: the second search runs once as the silent
            # fallback
            self._fallback_ids = tuple(v.index for v in self.loop)
            pre, self._fb_pre = getattr(self, '_fb_pre', None), None
            handed = pre is None   # pre is owned again (shown job / _drop_fallback_pre) once this is True
            try:
                self.key = None
                self._replan(context)
                if (pre is not None and pre['ids'] == self._fallback_ids and pre.get('geo') == self._loop_geo()
                        and self._step4_matches(pre, context)
                        and self.job is not None):   # same indices AND rim geometry (and 3D mode / sources)
                    self.job.close()   # take over the precomputed fallback: it has already run
                    old = getattr(self, '_job_cancel', None)
                    if old is not None:
                        old.set()
                    self.job = pre['gen'] if not pre['done'] else self._finished_job(pre)
                    self._job_cancel = pre.get('cancel')   # the taken-over run is cancelled through it
                    handed = True
                    if pre.get('clock') is not None:
                        pre['clock']['active'] = False   # shown now: its S part runs on real time
                    self._swap_hidden_state(pre['state'])   # its entries become the shown ones
                elif pre is not None:
                    self._fb_pre = pre
                    handed = True
                    self._drop_fallback_pre()
            except BaseException:
                if not handed:   # detached above and owned by nobody: stop its background jobs now
                    token = pre.get('cancel')
                    if token is not None:
                        token.set()
                    if getattr(self, '_fb_pre', None) is None:
                        self._fb_pre = pre   # _end (_drop_fallback_pre) also closes its steps
                raise
            _tag_redraw(context)
        revealed = self._reveal()
        now = time.perf_counter()
        if (revealed or self.plan is not previous or was_busy != self._busy()
                or now >= getattr(self, '_next_work_redraw', 0.)):
            _tag_redraw(context)
            self._next_work_redraw = now + REDRAW_SECONDS
        if getattr(self, 'timer', None) is not None:
            if (self.job is None and self.background is None and self.triangle_job is None
                    and not self._step4_running() and not self._crease_running()):
                interval = 1 / 30
            else:
                interval = TIMER_SECONDS
            if interval != self.timer_interval:
                context.window_manager.event_timer_remove(self.timer)
                self.timer = context.window_manager.event_timer_add(interval, window=context.window)
                self.timer_duration = self.timer.time_duration
                self.timer_interval = interval

    def _advance_work(self, context):
        slice_seconds = SLICE_SECONDS
        if self.triangle_job is not None:
            deadline = time.perf_counter() + slice_seconds
            while self.triangle_job is not None and time.perf_counter() < deadline:
                try:
                    update = next(self.triangle_job)
                    if isinstance(update, dict) and update.get('work_wait'):
                        break
                except StopIteration as done:
                    self.triangle_job = None
                    entries = list(done.value)
                    first = next((i for i, row in enumerate(entries) if row['default']), 0)
                    auto = (bool(entries) and not self.portfolio.locked   # Ranking is FLOW
                            and not getattr(self, 'wheel_touched', False))
                    if auto:
                        first = next((i for i, row in enumerate(entries) if row.get('_auto_selected')), first)
                    # the starting entry is always 1/n.
                    if entries:
                        entries.insert(0, entries.pop(first))
                    if auto and getattr(self.portfolio, 'rank_r1', False):   # R1 over patterns AND site plans
                        for row in entries:
                            sp = row['plan']
                            if sp.get('_legacy_layout'):
                                self.portfolio.add_legacy(sp, select_if_empty=False)
                            else:
                                self.portfolio.add(sp)
                        self.triangle_selected = True
                        self.odd_wheel = self._odd_plans(self.portfolio, self.portfolio.plan)
                        self.plan = self.odd_wheel[0]
                        self.odd_index = 0
                    elif auto:
                        self.plan = entries[0]['plan']
                        self.triangle_selected = True   # the odd wheel exists from entry 1 (HUD 1/n)
                        self.odd_wheel = self._odd_list()
                        self.plan = self.odd_wheel[0]
                        self.odd_index = 0
                        if getattr(self, 'settle_on', False) and not entries[0]['default']:
                            self._note_better()
                except (fill.FillError, ValueError) as exc:
                    self.triangle_job = None
                    self.improvement_note = str(exc)
                except Exception as exc:   # a logged refusal of the triangle stage; Fill keeps running
                    _log_engine_error('triangle', exc)
                    self.triangle_job = None
            return
        deadline = time.perf_counter() + slice_seconds
        while self.job is not None and time.perf_counter() < deadline:
            try:
                update = next(self.job)
                if isinstance(update, dict) and update.get('work_wait'):
                    break
                if isinstance(update, dict) and ('preview' in update or 'candidate' in update):
                    self._accept_background_plan(update.get('preview', update.get('candidate')))
                    # Give Blender a chance to draw the accepted preview before
                    # resuming this generator (which may start the site search).
                    return
            except StopIteration as done:
                self.job = None
                self.plan, self.error = done.value, ""
                self._legacy_view = done.value
                if getattr(self, 'newfill_active', False):
                    _tag_redraw(context)
                    return
                if self.triangle_selected:
                    _tag_redraw(context)
                    return
                if not self.state.panels:
                    self._accept_background_plan(self.plan)
                self._start_background(context)
                _tag_redraw(context)
                return
            except fill.FillError as exc:
                self.job = None
                kept = None if self.state.panels else self.portfolio.plan   # a direct grid shown stays
                self.plan, self.error = kept, ('' if kept is not None else str(exc))
                self._legacy_view = None
                if getattr(self, 'newfill_active', False):
                    _tag_redraw(context)
                    return
                if self.triangle_selected:
                    _tag_redraw(context)
                    return
                self._start_background(context)   # always AUTO (no pattern cycle)
                if self.background is not None:
                    _tag_redraw(context)
                    return
                break
            except Exception as exc:   # a logged refusal; Fill keeps running (the fallback search may run)
                _log_engine_error('search', exc)
                self.job = None
                self.plan, self.error = None, _ENGINE_ERROR
                self._legacy_view = None
                _tag_redraw(context)
                return
        # The incumbent has already been published before any refinement work.
        if self.job is None and self.background is not None:
            deadline = min(deadline, time.perf_counter() + slice_seconds)
            while self.background is not None and time.perf_counter() < deadline:
                try:
                    update = next(self.background)
                    if isinstance(update, dict) and update.get('work_wait'):
                        break
                    if isinstance(update, dict) and update.get('settled'):
                        self._settle()
                    if isinstance(update, dict) and ('preview' in update or 'candidate' in update):
                        self._accept_background_plan(update.get('preview', update.get('candidate')))
                        self._check_better()
                except StopIteration as done:
                    self.background = None
                    self._accept_background_plan(done.value)
                    self._check_better()
                    self._settle()   # refinement ended without a settle mark
                    self._start_triangles(context)
                except (fill.FillError, ValueError) as exc:
                    self.background = None
                    self._settle()
                    if self.plan is None:
                        self.error = str(exc)
                except Exception as exc:   # a logged refusal; Fill keeps running
                    _log_engine_error('background', exc)
                    self.background = None
                    self._settle()
                    if self.plan is None:
                        self.error = _ENGINE_ERROR

    def _ppo_refresh(self, context):
        # The shared guard calls this when the mesh changed outside the tool between events.
        self._drop_fallback_pre()   # a precompute of the old mesh is stale
        self._fallback_ids = None
        self._cancel_job()
        self.state.refresh()
        self.loop = self.plan = self.key = None
        self._hover(context)

    # ---------------------------------------------------------------- events

    def _end(self, context):
        self._drop_fallback_pre()
        self._cancel_job()
        self._release_ui(context)

    def _release_ui(self, context):
        """the event timer, the preview draw handler and the header text this Fill added."""
        if self.timer is not None:
            context.window_manager.event_timer_remove(self.timer)
            self.timer = None
        if self.handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self.handle, "WINDOW")
            self.handle = None
        if getattr(self, "header", False) and context.area is not None:
            context.area.header_text_set(None)
            self.header = False
        _tag_redraw(context)

    def cancel(self, context):
        self._end(context)

    def modal(self, context, event):
        # the preview is solved in world space; after an object move it is
        # no longer valid, so Fill ends cleanly (Ctrl again previews anew).
        try:
            key = tuple(map(tuple, self.state.view.obj.matrix_world))
        except (ReferenceError, AttributeError):
            key = None
        if getattr(self, "_mw_key", None) is None:
            self._mw_key = key
        elif key != self._mw_key:
            self._end(context)
            return {"CANCELLED"}
        if event.type == "TIMER":
            if not timer_tick(self):
                return {"PASS_THROUGH"}
            self._advance(context)
            if (getattr(self, "newfill_active", False) and self.job is not None and self.plan is None
                    and getattr(self, "_fr_prov", None) is not None and not getattr(self, "fr_shown", False)
                    and time.perf_counter() - getattr(self, "replan_started", 0.0) >= FIRST_RESULT_SECONDS):
                self.newfill_entries, self.newfill_index = [self._fr_prov], 0   # first-result deadline: A at 1 s
                self.plan, self.error, self.fr_shown, self.fr_touched = self._fr_prov, "", True, False
                _tag_redraw(context)
            return {"RUNNING_MODAL"}
        if event.type in {"MOUSEMOVE", "INBETWEEN_MOUSEMOVE"}:
            self.mouse = raycast.mouse(event)
            self._hover(context)
            _tag_redraw(context)
            return {"RUNNING_MODAL"}

        if event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"} and event.value == "PRESS":
            from ..core import fill_progressive
            if self.state.panels:
                return {"RUNNING_MODAL"}
            if self.loop is None:   # after a commit (Ctrl still held) there is no outline to change
                return {"RUNNING_MODAL"}
            if getattr(self, 'newfill_active', False):   # the fallback search's wheel
                step = 1 if event.type == "WHEELDOWNMOUSE" else -1
                if self.job is not None and getattr(self, 'fr_shown', False):   # first-result deadline
                    self.fr_touched = True   # S's result will be appended, not swapped in
                    self.report({'INFO'}, tr_rpt('Improving fill preview...'))
                    return {"RUNNING_MODAL"}
                if self.job is not None:
                    self.report({'INFO'}, tr_rpt('Computing; wheel when the first result is shown'))
                    return {"RUNNING_MODAL"}
                nxt = self.newfill_index + step
                if 0 <= nxt < len(self.newfill_entries):
                    self.newfill_index = nxt
                    self.plan, self.error = self.newfill_entries[nxt], ''
                # else clamped: wheel up at entry 1 stays; past the last entry stays (the fallback's
                # entries are all there is, the main search found none for this outline)
                _tag_redraw(context)
                return {"RUNNING_MODAL"}
            if fill_progressive.eligible(self._settings(context)):
                step = 1 if event.type == "WHEELDOWNMOUSE" else -1   # wheel down: next entry
                if self._computing() or self._unseen():
                    self.report({'INFO'}, tr_rpt('Computing; wheel when the first result is shown'))
                    return {"RUNNING_MODAL"}
                if self.triangle_job is not None:
                    self.report({'INFO'}, tr_rpt('Computing; wheel when the first result is shown'))
                    return {"RUNNING_MODAL"}
                if self.triangle_selected and len(self.loop) % 2:
                    # no triangle-site wheel; entry 1 is the auto site plan, then the pattern entries
                    if getattr(self, 'odd_wheel', None) is None:
                        self.odd_wheel = self._odd_list()
                        self.odd_index = 0
                    if not 0 <= self.odd_index + step < len(self.odd_wheel):   # clamped at both ends
                        return {"RUNNING_MODAL"}
                    self.odd_index += step
                    self.__dict__['_wheel_first'] = (self.odd_index == 0)
                    self.plan, self.error = self.odd_wheel[self.odd_index], ''
                    self.wheel_touched = True
                    _tag_redraw(context)
                    return {"RUNNING_MODAL"}
                keys = self.portfolio.ranked_keys()
                if getattr(self, '_went_progressive', False):   # wheel the list the box shows (arrival order)
                    from ..ui import fill_hud
                    from ..core import fill_layouts
                    shown, cur = fill_hud.displayed_plans(self)
                    at = cur if cur is not None else 0
                    if not shown or not 0 <= at + step < len(shown):   # clamped at both ends (no wrap)
                        return {"RUNNING_MODAL"}
                    target = shown[at + step]
                    key = next((k for k in keys if (self.portfolio.legacy if k == fill_layouts.LEGACY_KEY
                                                    else self.portfolio.entries[k][1]) is target), None)
                    if key is None:
                        return {"RUNNING_MODAL"}
                    self.wheel_touched = True
                    self.portfolio.selected, self.portfolio.locked = key, True
                    self.plan, self.error = target, ''
                    self._shown_plan = target
                    self.__dict__['_wheel_first'] = (at + step == 0)   # where the user parked
                    _tag_redraw(context)
                    return {"RUNNING_MODAL"}
                at = keys.index(self.portfolio.selected) if self.portfolio.selected in keys else 0
                if keys and not 0 <= at + step < len(keys):   # clamped at both ends (no wrap)
                    return {"RUNNING_MODAL"}
                self.wheel_touched = True
                chosen = self.portfolio.cycle(step)
                if chosen is not None:
                    self.plan, self.error = chosen, ''
                try:   # where the user parked (index in the list they see)
                    from ..ui import fill_hud as _fh
                    self.__dict__['_wheel_first'] = _fh.displayed_plans(self)[1] in (0, None)
                except Exception:
                    pass
                _tag_redraw(context)
                return {"RUNNING_MODAL"}
            return {"RUNNING_MODAL"}   # Fill is always AUTO (no pattern cycle)

        if event.type == "RIGHTMOUSE":
            # no start drag in the preview (start adjustment removed); RMB is ignored.
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE":
            if event.value == "PRESS" and not self._over_own_region(context, event):
                return {"PASS_THROUGH"}   # a click in another area or region: never swallowed, never commits
            if event.value == "PRESS" and (self._computing() or self._unseen() or self._undrawn()):
                self.report({"INFO"}, tr_rpt("Computing; click when the first result is shown"))
                return {"RUNNING_MODAL"}
            if event.value == "PRESS" and self.job is not None and self.plan is None:
                self.report({"INFO"}, tr_rpt("Computing; click when the first result is shown"))
            if event.value == "PRESS" and self.plan is not None:
                self._commit(context)
                if not event.ctrl:
                    self._end(context)
                    return {"FINISHED"}
            return {"RUNNING_MODAL"}

        if event.type in {"LEFT_CTRL", "RIGHT_CTRL"}:
            if event.value == "RELEASE":
                self._end(context)
                return {"FINISHED"}
            return {"RUNNING_MODAL"}

        if event.type == "WINDOW_DEACTIVATE" or (event.type == "ESC" and event.value == "PRESS"):
            self._end(context)
            return {"CANCELLED"}

        if event.type in {"TIMER", "NONE"} or event.value != "PRESS":
            return {"RUNNING_MODAL"}
        # any other key (Ctrl+Z, navigation...) ends Fill and runs normally
        self._end(context)
        return {"FINISHED", "PASS_THROUGH"}

    def _commit(self, context):
        self._cancel_job()
        st = self.state
        view_dir = raycast.view_forward(context)
        try:
            if self.plan.get('panel_count'):
                from ..core import fill_multi
                created = fill_multi.commit(st.bm, st.mw, st.panels, self.plan, view_dir,
                                           context.scene.tool_settings.use_transform_correct_face_attributes)
            else:
                created = fill.commit(st.bm, st.mw, self.loop, self.plan, view_dir,
                                      context.scene.tool_settings.use_transform_correct_face_attributes)
        except (fill.FillError, ValueError, RuntimeError) as exc:
            self.plan, self.error = None, str(exc)
            # a commit-time refusal is final for this outline: no New fallback for it
            self._refused_ids = tuple(v.index for v in self.loop) if self.loop is not None else None
            self._drop_fallback_pre()   # its background run is not needed for a refused outline
            bmesh.update_edit_mesh(st.view.obj.data, loop_triangles=True, destructive=True)
            self.report({"WARNING"}, tr_rpt(fill.user_text(self.error, _debug(None))))
            return
        warning = self.plan.get("uv_warning")
        if warning == "seam":
            self.report({"WARNING"}, tr_rpt("The outline crosses a UV seam; unwrap the new faces"))
        elif warning == "missing":
            self.report({"WARNING"}, tr_rpt("Part of the outline has no faces next to it; unwrap the new faces"))
        for seq in (st.bm.verts, st.bm.edges, st.bm.faces):
            for elem in seq:
                elem.select_set(False)
        for f in created:
            f.select_set(True)
        st.bm.select_flush(True)
        st.bm.normal_update()
        bmesh.update_edit_mesh(st.view.obj.data, loop_triangles=True, destructive=True)
        bpy.ops.ed.undo_push(message="Fill")
        st.refresh()
        # The newly selected patch includes interior edges. That is the result,
        # not an invalid outline request for the next hole.
        st.selection_error = ""
        self._fr_prov, self.fr_shown = None, False
        self.loop = self.plan = self.key = None
        self._hover(context)

    # ---------------------------------------------------------------- drawing

    def _searching(self):
        return (self.loop is not None and self.plan is None and
                (self.job is not None or self.background is not None))

    def _busy(self):
        """Any background work for the current outline: first plan, refinement
        (network, main search, field, structure candidates) or triangle sites."""
        return (self.loop is not None and
                (self.job is not None or self.background is not None
                 or self.triangle_job is not None))

    def status(self, cursor=False):
        if self._computing():
            return iface_("Computing...")
        if self._searching():
            return iface_("Searching for a fill...") if cursor else ""
        if self.loop is not None and self.plan is None and self._crease_running():
            return iface_("Searching for a fill...") if cursor else ""
        if self.loop is None:
            return iface_(fill.user_text(self.error, _debug(None)) or "Hover a hole, or select one closed outline")
        if self.job is not None:
            from ..core import fill_progressive
            if fill_progressive.eligible(self._settings(self.state.context)):
                return iface_("Searching for a fill...") if self.plan is None else iface_("Improving fill preview...")
            return iface_("Computing...")
        if self.plan is None:
            text = iface_("Searching for a fill...") if self.background is not None else (iface_(fill.user_text(self.error, _debug(None))) if self.error else "")
            return text
        debug = _debug(self.state.context)
        if self.plan.get('_newfill_lane'):
            text = f"{self.newfill_index + 1}/{self._total_entries()} | {_rim_count(self.plan)}"   # rim edge count
            if any(len(face) == 3 for face in fill.faces_of(self.plan)):
                text += " | " + iface_("Odd edge count: one triangle")
            if debug:
                lg = (self.newfill_result or {}).get('log', {}); lane = self.plan['_newfill_lane']
                text += (f" | {lane} minSJ {lg.get('minSJ', {}).get(lane, 0.):.2f} | S {'ran' if lg.get('s_ran') else 'off'}"
                         f" {lg.get('s_stop') or ''} | A budget {'hit' if lg.get('a_budget_hit') else 'ok'}"
                         f" | {lg.get('seconds', {}).get('total', 0.):.2f}s")
            return text
        if self.plan.get('panel_count'):
            return iface_("Fill {panels} panels | {seams} shared seams | {faces} faces").format(
                panels=self.plan['panel_count'], seams=self.plan['shared_seams'], faces=len(self.plan['quads']))
        label = _pattern_label(self.plan["pattern"].rstrip("~"), debug)
        text = iface_("Pattern: {name}").format(name=label) + f" | {_rim_count(self.plan)}"   # rim edge count
        if self.portfolio.plan is self.plan or getattr(self, 'odd_wheel', None):
            if debug:
                text += " | " + self.portfolio.label()
            else:
                keys = self.portfolio.ranked_keys()
                if getattr(self, 'odd_wheel', None):   # the odd-rim wheel numbers its own entries
                    text += f" | {self.odd_index + 1}/{len(self.odd_wheel)}"
                elif self.portfolio.selected in keys:
                    text += f" | {keys.index(self.portfolio.selected) + 1}/{len(keys)}"
        if any(len(face) == 3 for face in fill.faces_of(self.plan)):
            text += " | " + iface_("Odd edge count: one triangle")
        if self._better_note():
            text += " | " + iface_("Better result found")
        if self._provisional():
            text += " | " + iface_("Provisional")   # stays until #1 settles
        if self.improvement_note:
            text += " | " + self.improvement_note
        elif getattr(self, 'field_searching', False):
            text += " | " + iface_("Searching field candidates...")
        elif (self.background is not None and not self.legacy_searching) or self._step4_tail():
            text += " | " + iface_("Improving fill preview...")
        return text

    def hint(self):
        """Control hints for the header / viewport help (simple HUD only)."""
        return iface_("Ctrl+Wheel: change choice | Ctrl+LMB: fill | Esc: cancel")   # only keys that work

    def _character(self, plan, plans):
        """Short plain label of a candidate's character (simple HUD)."""
        pattern = str(plan.get("pattern", "")).rstrip("~")
        if pattern == "grid" or pattern.startswith("takayama") or plan is getattr(self.portfolio, "legacy", None):
            return iface_("Follows Outline")
        try:
            from ..core import fill_flow
            poles = [fill_flow.metrics(p)["poles"] for p in plans if p is not None]
            if len(set(poles)) > 1 and fill_flow.metrics(plan)["poles"] <= min(poles):
                return iface_("Fewer Poles")
        except Exception:
            pass
        return iface_("Even")

    def _label_of(self, plan, plans):
        if plan.get('_p1_candidate') and plan.get('auto_cut_count'):
            return iface_("Cut")
        return self._character(plan, plans)

    def compact(self):
        """One line next to the cursor: "2/5 · Even" (debug info off)."""
        if self.loop is None:
            return iface_(fill.user_text(self.error, _debug(None))) if self.error else ""
        if self._computing():
            return iface_("Computing...")
        if self.plan is None and self._crease_running():   # the crease-fill search may still find one
            return iface_("Searching for a fill...")
        if self.plan is None and self.error and not self._busy():   # a refusal of this outline: say why
            return iface_(fill.user_text(self.error, _debug(None)))
        text = self._compact()
        if text and ((getattr(self, 'fr_shown', False) and self.job is not None)   # first-result deadline indicator
                     or self._step4_tail()):   # step 4: the second search's run after the main search is done
            text += " · " + iface_("Improving fill preview...")
        if text and self._better_note():
            text += " · " + iface_("Better result found")
        if text and self._provisional():
            text += " · " + iface_("Provisional")   # stays until #1 settles
        return text

    def _compact(self):
        if self.plan is None:
            return ""
        if self.plan.get('_newfill_lane'):
            return f"{self.newfill_index + 1}/{self._total_entries()} · {_rim_count(self.plan)}"
        if self.plan.get('panel_count'):
            return iface_("Fill {panels} panels | {seams} shared seams | {faces} faces").format(
                panels=self.plan['panel_count'], seams=self.plan['shared_seams'], faces=len(self.plan['quads']))
        if self.portfolio.plan is self.plan or getattr(self, 'odd_wheel', None):
            from ..core import fill_layouts
            keys = self.portfolio.ranked_keys()
            plans = [self.portfolio.legacy if k == fill_layouts.LEGACY_KEY else self.portfolio.entries[k][1]
                     for k in keys]
            label = self._label_of(self.plan, plans)
            shown = getattr(self, 'odd_wheel', None) or plans
            twins = [p for p in shown if p is not None and self._label_of(p, plans) == label]
            if len(twins) > 1:
                faces = len(fill.faces_of(self.plan))
                label += " · " + _face_count(self.plan)   # another entry of this list has the same label
                same = [p for p in twins if len(fill.faces_of(p)) == faces]
                if len(same) > 1:   # same label and face count: a variant number, list order
                    label += f" #{next(i for i, p in enumerate(same) if p is self.plan) + 1}" if any(p is self.plan for p in same) else ""
            if getattr(self, 'odd_wheel', None):
                return f"{self.odd_index + 1}/{len(self.odd_wheel)} · {label} · {_rim_count(self.plan)}"
            if self.portfolio.selected in keys:
                return f"{keys.index(self.portfolio.selected) + 1}/{len(keys)} · {label} · {_rim_count(self.plan)}"
            return label
        return _pattern_label(self.plan["pattern"].rstrip("~"))

    def _label_kind(self):
        """A refusal / error at the cursor (no preview, an error text) is a warning; the rest is info."""
        return 'warning' if self.error and self.plan is None and not self._computing() else 'info'

    def _draw(self):
        context = bpy.context
        if context.region is None or context.region.as_pointer() != self.region:
            return
        from ..core import safe_mesh as _sm
        if _sm.native_modal_running(context):   # no BMesh reads under a native modal
            return
        try:
            with draw.Blend():
                self._paint(context)
        except ReferenceError:
            return
        except Exception as exc:   # a draw callback must never raise (it would repeat every frame)
            print("PolyTailor Fill preview:", type(exc).__name__, exc)

    def _over_own_region(self, context, event):
        """the mouse is over the 3D region this Fill runs in."""
        try:
            for area in context.window.screen.areas:
                for region in area.regions:
                    if region.as_pointer() == self.region:
                        return (region.x <= event.mouse_x < region.x + region.width
                                and region.y <= event.mouse_y < region.y + region.height)
        except (AttributeError, ReferenceError):
            pass
        return False

    def _preview_geometry(self, context):
        """Pixel geometry changes only with the accepted plan or viewport matrix."""
        view = (tuple(float(v) for row in context.region_data.perspective_matrix for v in row),
                context.region.width,context.region.height)
        if self.plan is getattr(self,"_paint_plan",None) and view == getattr(self,"_paint_view",None):
            return self._paint_geometry
        self._paint_plan,self._paint_view = self.plan,view
        self._paint_geometry = None
        screen = [raycast.to_screen(context,p) for p in map(tuple,self.plan["positions"])]
        if any(p is None for p in screen):
            return None
        tris,tri_yellow,lines = [],[],[]
        for face in fill.faces_of(self.plan):
            pts = [(screen[i].x,screen[i].y) for i in face]
            target = tri_yellow if len(face)==3 else tris
            for k in range(1,len(pts)-1):
                target += [pts[0],pts[k],pts[k+1]]
            for a,b in zip(pts,pts[1:]+pts[:1]):
                lines += [a,b]
        self._paint_geometry = (tris,tri_yellow,lines,(screen[0].x,screen[0].y))
        return self._paint_geometry

    def _paint(self, context):
        if self.loop is not None:
            whole = (self.plan is None and not self.state.panels   # an edge-on / too-long loop
                     and str(getattr(self, 'error', '') or '').startswith(("This outline encloses no area", "Loop too long")))
            for loop in self.state.panels or [self.loop]:
                poly = fill.screen_loop(context, self.state.mw, loop)
                if poly:
                    draw.polyline(poly, palette.warning() if whole else palette.accent(context), 2.0, closed=True)
            marks = getattr(self, 'fail_marks', None)
            if marks is not None and self.plan is None:   # where the outline fails (fill.certain_fail)
                lines = []
                for a, b in marks['segs']:
                    pa, pb = raycast.to_screen(context, a), raycast.to_screen(context, b)
                    if pa is not None and pb is not None:
                        lines += [(pa.x, pa.y), (pb.x, pb.y)]
                draw.segments(lines, palette.warning(), 4.0)
                for p in marks['dots']:
                    s = raycast.to_screen(context, p)
                    if s is not None:
                        draw.disc((s.x, s.y), 5.0, (0., 0., 0., .9))
                        draw.disc((s.x, s.y), 3.5, palette.warning())
        if self.plan is not None and not self._computing() and (self._display_stage() in ('done', 'progressive')
                                                                 or self._fallback_drawable()):
            self._drawn_plan = self.plan   # this frame shows the plan a click adds
            geometry = self._preview_geometry(context)
            if geometry is not None:
                tris,tri_yellow,lines,start = geometry
                # the faces Ctrl+click adds: Preview Fill / Preview Border (preferences)
                if tris:
                    draw._draw("TRIS", tris, palette.preview_faces(context))
                if tri_yellow:
                    draw._draw("TRIS", tri_yellow, palette.warning(.35))
                draw.segments(lines, palette.preview_lines(context), 1.5)
                draw.disc(start, 5, palette.accent(context))
                if _debug(context):
                    draw.text((start[0] + 8, start[1] + 8), iface_("Start"), palette.accent(context))
        debug = _debug(context)
        if not debug:
            # Simple HUD: one line at the cursor's right, spinner below while computing.
            scale = max(.5, min(3., prefs.ui_scale(context)))
            size = max(11, round(13*scale))
            xy = (self.mouse.x + 18*scale, self.mouse.y - 6*scale)
            from ..ui import fill_hud   # the Fill HUD box
            if fill_hud.applicable(self) and fill_hud.draw_hud(self, context, xy, scale, size):
                return   # drawn; on any HUD error the text label below is the fallback
            text = self.compact()
            if text:
                draw.cursor_label(context, xy, text, kind=self._label_kind(), size=size)
            if self._busy():
                center = (xy[0] + 6*scale, xy[1] - 12*scale)
                phase = (time.perf_counter()*1.25 % 1.)*math.tau   # wall clock: steady across timer changes
                draw.ring(center, 5*scale, palette.secondary(context, .4), 1.2*scale)
                draw.polyline(draw.circle_points(center, 5*scale, 24, phase, phase+math.tau*.72),
                              palette.accent(context), 2.*scale)
            return
        text = self.status(cursor=True)
        if debug and getattr(self, 'newfill_active', False) and self.newfill_entries and not self._computing():
            for i, plan in enumerate(self.newfill_entries):
                marker = '> ' if i == self.newfill_index else '  '
                draw.text((self.mouse.x + 18, self.mouse.y - 44 - i*18), marker + f"{i+1} | {plan['_newfill_lane']}", palette.main(context))
        elif debug and not getattr(self, 'newfill_active', False) and self.plan is not None and not self._computing():
            from ..core import fill_flow
            from ..ui import fill_hud
            if self.triangle_selected and self.loop is not None and len(self.loop) % 2 and getattr(self, 'odd_wheel', None):
                shown_list = list(self.odd_wheel)
            else:
                shown_list = fill_hud.displayed_plans(self)[0]   # exactly the box / wheel list
            for i, plan in enumerate(shown_list):
                marker = '> ' if plan is self.plan else '  '
                draw.text((self.mouse.x + 18, self.mouse.y - 44 - i*18),
                          marker + f'{i+1} | ' + fill_flow.short_label(plan, self.portfolio.quality_floor), palette.main(context))
        if text:
            if self._busy():
                scale = max(.5, min(3., prefs.ui_scale(context)))
                center = (self.mouse.x+20*scale, self.mouse.y-20*scale)
                phase = (time.perf_counter()*1.25 % 1.)*math.tau   # wall clock: steady across timer changes
                draw.ring(center, 8*scale, palette.secondary(context, .4), 1.5*scale)
                arc = draw.circle_points(center, 8*scale, 32, phase, phase+math.tau*.72)
                draw.polyline(arc, palette.accent(context), 2.5*scale)
                xy = (center[0]+14*scale, center[1]-5*scale)
                size = max(11, round(13*scale))
                draw.cursor_label(context, xy, text, kind=self._label_kind(), size=size)
            else:
                draw.cursor_label(context, (self.mouse.x + 18, self.mouse.y - 22), text, kind=self._label_kind())


def register():
    bpy.utils.register_class(PPO_OT_fill)


def unregister():
    bpy.utils.unregister_class(PPO_OT_fill)
    _RIM_CACHE.clear()
