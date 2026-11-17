# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Cross-phase memo of leaf piece/template searches (exact under the work-unit clock).

A search is boundary_preview.job(points, refs, settings, site). Its path depends on: its inputs
(boundary_preview.signature, the existing cache identity), the clock quantum, the remaining budget of every enclosing
time_budget at entry, and the code site that advances it (its loop re-derives the job's slice between steps). The caller
names that site explicitly: `site` is a tuple of the advancing loop's name and the values its slices are derived from
(see fill._template_steps, fill_partition._capped_site). A search without a site is not memoised.
With all of these equal, a second run is step for step the same as the first. This memo stores the first run as a
trace:
    per step: the work units charged inside it, the yielded value, and the side effects (fill_selection.checkpoint,
    boundary_preview.remember) in order; at the end: the return value or the exception.
A replay yields the same values, charges the same units per step, repeats the side effects on the current settings and
phase cache, and returns / raises the same object. The surrounding loops therefore see exactly what they saw the first
time: the same yields, the same clock, the same results.
Only leaf searches are stored (no nested boundary_preview.job inside), and only complete runs (not closed early by the
caller). A search already in the phase cache (boundary_preview._cache) runs directly, which answers at once.

Relaxed replay (band-stable traces). Inside a search the budget enters only through time_left() compared with 0
(check_time / the untangle's stop) and with POOL_RESERVE. So every read falls in a band: <= 0, (0, POOL_RESERVE),
>= POOL_RESERVE, and two runs whose reads fall in the same bands take the same path. A trace in which every step's
reads stay in one band is stored by signature alone. A relaxed replay checks, before each step, that the first and the
last read of that step (the values fall monotonically within a step) would land in the recorded band under the current
deadline. If not, it fast-forwards a fresh search over the steps already replayed, each step under the recorded
deadline and quantum (so it sees the recorded values and takes the recorded path), with no charges and no side
effects, and continues it live from there.
A time_budget opened inside a recorded search makes its trace sensitive (strict replay only).

Solver cache. patterns.solver.solve_steps keeps a per-phase cache (swapped by deterministic_steps): a hit returns at
once, a miss yields many times and then inserts. So a search's steps depend on which solver results its phase already
holds. A trace records its solver lookups in order (key, hit) and its inserts (key, value). Searches run interleaved
with other searches that share the cache, so the check is made per step: before replaying a step, each of its lookups
must have the recorded hit/miss in the current cache (and no eviction may be possible); the step's cache updates are
then re-applied. On a mismatch the replay falls back at that step. The fast-forward then runs each earlier step against
private caches that hold exactly that step's recorded hits (values recorded too), so it takes the recorded path without
touching the real caches; the search then continues live on the real caches.

Stores: _MEMO (strict, by full key) and _RELAXED (band-stable, by signature), each an LRU of MAX traces.

The functions this memo observes carry its decorators at their definitions:
    patterns.geom.time_left      @recorded_read          (budget reads of a search being recorded)
    patterns.geom.time_budget    @recorded_budget        (the budget stack of each running clock)
    patterns.solver.solve_steps  @recorded_solver        (solver-cache lookups and inserts)
    fill_selection.checkpoint    @recorded_checkpoint    (side effect, repeated on replay)
    boundary_preview.remember    @recorded_remember      (side effect, repeated on replay)
    boundary_preview.job         @memoised_search        (the memo itself)"""
import contextlib
import copy
import functools
from collections import OrderedDict

_MEMO = OrderedDict()
_STACKS = {}         # id(running clock) -> [[deadline]] of its enclosing time_budget blocks (outermost first)
_REC = []            # active recordings (outermost first)
_RELAXED = OrderedDict()
_SUPPRESS = [0]
MAX = 256

# The modules whose functions this library decorates. They import this module while they load, so they are
# imported here on first use (ordinary relative imports), once every one of them is fully loaded.
_HOME = []


def _home():
    """(geom, solver, fill_selection, boundary_preview)."""
    if not _HOME:
        from .patterns import geom, solver
        from . import fill_selection, boundary_preview
        _HOME.append((geom, solver, fill_selection, boundary_preview))
    return _HOME[0]


def clear_caches():
    """Empty both trace stores (add-on unregister)."""
    _MEMO.clear()
    _RELAXED.clear()


def _put(store, key, value):
    store[key] = value
    store.move_to_end(key)
    while len(store) > MAX:
        store.popitem(last=False)


def _bare(e):
    """A copy of the exception without its traceback (a stored traceback would keep the search's frames alive)."""
    try:
        c = copy.copy(e)
    except Exception:
        return e
    c.__traceback__ = None
    return c


def _clock(geom):
    return geom.clock()


def _band(v):
    return 0 if v <= 0 else (1 if v < float(_home()[3].POOL_RESERVE) else 2)


def _step_ok(ops):
    SV = _home()[1]
    keys = None; n = len(SV.current_cache())
    for op in ops:
        if keys is None:
            keys = set(SV.current_cache().keys())
        if op[0] == 'q':
            if (op[1] in keys) != op[2]:
                return False
        elif op[1] not in keys:
            keys.add(op[1]); n += 1
            if n > 512:
                return False
    return True


def _solver_apply(ops):
    SV = _home()[1]
    for op in ops:
        if op[0] == 'q':
            if op[2]:
                SV.cache_touch(op[1])
        else:
            SV.cache_insert(op[1], copy.deepcopy(op[2]))


def recorded_solver(o_ss):
    """patterns.solver.solve_steps: a search being recorded notes its solver-cache lookups (key, hit) and inserts."""

    @functools.wraps(o_ss)
    def solve_steps(pattern, counts):
        if getattr(pattern, 'takayama', False):
            return (yield from o_ss(pattern, counts))
        SV = _home()[1]
        key = (pattern.name, tuple(counts))
        hit = key in SV.current_cache()
        if _REC:
            val = copy.deepcopy(SV.current_cache()[key]) if hit else None
            for r in _REC:
                r['sops'][-1].append(('q', key, hit, val))
        res = yield from o_ss(pattern, counts)
        if not hit and key in SV.current_cache():             # a miss inserts only when the solve completes normally
            for r in _REC:
                r['sops'][-1].append(('ins', key, copy.deepcopy(res)))
        return res
    return solve_steps


def recorded_read(o_tl):
    """patterns.geom.time_left: a search being recorded notes the offset and band of every budget read."""

    @functools.wraps(o_tl)
    def time_left():
        v = o_tl()
        if _REC:
            clk = _clock(_home()[0]); t = clk.ticks
            for r in _REC:
                b = _band(v); off = t - r['t0']
                if r['firstoff'] is None:
                    r['firstoff'] = off; r['band'] = b
                elif b != r['band']:
                    r['crossed'] = True
                r['lastoff'] = off
        return v
    return time_left


def recorded_budget(o_tb):
    """patterns.geom.time_budget: keeps the stack of enclosing deadlines per running clock (part of the memo key); a
    budget opened inside a recorded search makes its trace sensitive."""

    @contextlib.contextmanager
    @functools.wraps(o_tb)
    def time_budget(seconds):
        for r in _REC:
            r['sensitive'] = True
        with o_tb(seconds):
            geom = _home()[0]
            key = id(_clock(geom))   # the clock running this job's step: interleaved jobs never share a stack
            entry = [geom.deadline_at()]
            _STACKS.setdefault(key, []).append(entry)
            try:
                yield
            finally:
                stack = _STACKS.get(key, [])
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i] is entry:   # popped by identity (another job's block may have closed in between)
                        del stack[i]
                        break
                if not stack:
                    _STACKS.pop(key, None)
    return time_budget


def recorded_checkpoint(o_cp):
    """fill_selection.checkpoint: a side effect recorded per step (silent while a fallback fast-forwards)."""

    @functools.wraps(o_cp)
    def checkpoint(settings, result):
        if _SUPPRESS[0]:
            return None
        for r in _REC:
            r['fx'][-1].append(('cp', copy.deepcopy(result, r['cpm'])))
        return o_cp(settings, result)
    return checkpoint


def recorded_remember(o_rem):
    """boundary_preview.remember: a side effect recorded per step (silent while a fallback fast-forwards)."""

    @functools.wraps(o_rem)
    def remember(key, result):
        if _SUPPRESS[0]:
            return None
        for r in _REC:
            r['fx'][-1].append(('rem', key, copy.deepcopy(result, r['cpm'])))
        return o_rem(key, result)
    return remember


def memoised_search(o_job):
    """boundary_preview.job: the memo. job(points, refs, settings, site=None); `site` names the code site that advances
    the search (with the values its loop derives the search's slices from); None = not memoised."""

    @functools.wraps(o_job)
    def job(points, refs, settings, site=None):
        geom, _, _, BP = _home()
        for r in _REC:
            r['leaf'] = False                       # the enclosing recordings are not leaves
        try:
            sig = BP.signature(points, settings, refs)
        except Exception:
            sig = None
        if sig is None or sig in BP.current_cache():
            return (yield from o_job(points, refs, settings))
        clk = _clock(geom); q = getattr(clk, 'quantum', None)
        if q is None or site is None:                # not on the work clock, or no site given: no memo
            return (yield from o_job(points, refs, settings))
        now = clk.ticks * q
        stack = tuple(round((d - now) / q, 6) for d in (e[0] for e in _STACKS.get(id(clk), ())) if d is not None)
        key = (sig, q, stack, site)
        tr = _MEMO.get(key)
        if tr is None and sig in _RELAXED:
                return (yield from _replay_relaxed(_RELAXED[sig], points, refs, settings, clk, q))
        if tr is not None:
                return (yield from _replay(tr, points, refs, settings, clk, q))
        return (yield from _record(o_job(points, refs, settings), key, settings, clk, sig, q))

    def _record(g, key, settings, clk, sig, q):
        geom = _home()[0]
        rec = dict(steps=[], fx=[[]], leaf=True, done=False, t0=clk.ticks, sensitive=False, crossed=False,
                   firstoff=None, lastoff=None, band=None, cpm={}, sops=[[]])
        try:
            while True:
                t0 = clk.ticks; rec['t0'] = t0; rec['firstoff'] = rec['lastoff'] = rec['band'] = None
                d0 = geom.deadline_at()
                ctx = (None if d0 is None else d0 - t0 * clk.quantum, clk.quantum)     # remaining at step start, quantum
                _REC.append(rec)             # active only while the search itself runs, never across its yields
                try:
                    try:
                        v = next(g)
                    finally:
                        _REC.remove(rec)
                except StopIteration as d:
                    rec['end'] = ('ret', copy.deepcopy(d.value, rec['cpm']), clk.ticks - t0, rec['fx'][-1], (rec['firstoff'], rec['lastoff'], rec['band'], ctx)); rec['done'] = True
                    return d.value
                except BaseException as e:
                    rec['end'] = ('exc', _bare(e), clk.ticks - t0, rec['fx'][-1], (rec['firstoff'], rec['lastoff'], rec['band'], ctx)); rec['done'] = True
                    raise
                rec['sops'].append([])
                rec['steps'].append((clk.ticks - t0, copy.deepcopy(v, rec['cpm']), rec['fx'][-1], (rec['firstoff'], rec['lastoff'], rec['band'], ctx))); rec['fx'].append([])
                yield v
        finally:
            g.close()
            rec['sopsteps'] = rec['sops']
            if rec['done'] and rec['leaf']:
                del rec['cpm'], rec['sops'], rec['t0'], rec['firstoff'], rec['lastoff'], rec['band']   # recording-only state
                _put(_MEMO, key, rec)
                if not rec['sensitive'] and not rec['crossed'] and rec['end'][0] == 'ret':
                    _put(_RELAXED, sig, rec)

    def _apply(fx, settings, m):
        for e in fx:
            if e[0] == 'cp':
                _home()[2].checkpoint(settings, copy.deepcopy(e[1], m))
            else:
                _home()[3].remember(e[1], copy.deepcopy(e[2], m))

    def _ok(info, clk, q):
        first, last, b, _ctx = info
        if first is None:                       # no budget read in this step
            return True
        d = _home()[0].deadline_at()
        if d is None:
            return b == 2
        return _band(d - (clk.ticks + first) * q) == b and _band(d - (clk.ticks + last) * q) == b

    def _replay_relaxed(tr, points, refs, settings, clk, q):
        m = {}
        for j, (units, v, fx, lastoff) in enumerate(tr['steps']):
            if not _ok(lastoff, clk, q) or not _step_ok(tr['sopsteps'][j]):
                return (yield from _fallback(j, tr, points, refs, settings, clk))
            clk.charge(units)
            _apply(fx, settings, m); _solver_apply(tr['sopsteps'][j])
            yield copy.deepcopy(v, m)
        kind, val, units, fx, lastoff = tr['end']
        if not _ok(lastoff, clk, q) or not _step_ok(tr['sopsteps'][-1]):
            return (yield from _fallback(len(tr['steps']), tr, points, refs, settings, clk))
        clk.charge(units)
        _apply(fx, settings, m); _solver_apply(tr['sopsteps'][-1])
        return copy.deepcopy(val, m)

    def _fallback(j, tr, points, refs, settings, clk):
        """Fast-forward a fresh search over the j steps already replayed, each under its recorded remaining budget and
        quantum (so it takes the recorded path), with no charges and no side effects; then continue it live."""
        geom, SV, _, BP = _home()
        g = o_job(points, refs, settings)
        saved, q0 = geom.deadline_at(), clk.quantum
        _SUPPRESS[0] += 1
        try:
            for k in range(j):
                rem, q = tr['steps'][k][3][3]
                t = clk.ticks; clk.set_state(t, q)
                geom.set_deadline_at(None if rem is None else t * q + rem)
                sv_token = SV.push_cache(OrderedDict((op[1], copy.deepcopy(op[3])) for op in tr['sopsteps'][k] if op[0] == 'q' and op[2]))
                bp_token = BP.push_cache(OrderedDict())
                try:
                    next(g)
                finally:
                    clk.set_state(t, q0); geom.set_deadline_at(saved)
                    SV.pop_cache(sv_token); BP.pop_cache(bp_token)
        finally:
            _SUPPRESS[0] -= 1
        try:
            while True:
                try:
                    v = next(g)
                except StopIteration as d:
                    return d.value
                yield v
        finally:
            g.close()

    def _replay(tr, points, refs, settings, clk, q):
        m = {}
        for j, (units, v, fx, _lo) in enumerate(tr['steps']):
            if not _step_ok(tr['sopsteps'][j]):
                return (yield from _fallback(j, tr, points, refs, settings, clk))
            clk.charge(units)
            _apply(fx, settings, m); _solver_apply(tr['sopsteps'][j])
            yield copy.deepcopy(v, m)
        kind, val, units, fx, _lo = tr['end']
        if not _step_ok(tr['sopsteps'][-1]):
            return (yield from _fallback(len(tr['steps']), tr, points, refs, settings, clk))
        clk.charge(units)
        _apply(fx, settings, m); _solver_apply(tr['sopsteps'][-1])
        if kind == 'exc':
            raise _bare(val)
        return copy.deepcopy(val, m)

    return job
