# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""The second Fill search (also the fallback search): the fill pipeline, as ONE steppable
generator stepped by the modal operator (one step per timer tick).

steps(world, s_ctx=None) yields None while working and returns a Result dict:
  mesh   picked mesh {positions, faces, boundary} with positions[:n] == world (None when nothing valid)
  lane   'A' | 'S' | 'E' | None
  cands  [(part, quality dict)] every valid candidate, best first (wheel order)
  log    {pick, s_ran, s_stop, a_budget_hit, exc:{part: class}, stop:{part: class}, seconds:{part: s}, minSJ:{part: v}}
A, E, the post-S dopt polish, the flow stage and the quality checks run as jobs in local worker processes
(worker.py); the caller reads and applies the results in Blender.
Each run has one cancel token (Token). The background jobs check it between their steps; a step in progress
finishes first (e.g. matching.max_weight_matching checks once per stage). CFG total is a hard cap: no quality check or
flow stage starts past it. steps() sets it on every exit, so a job of a run stops at its next check after the run
ends; cancel_all() cancels every run in progress and shutdown() stops the workers (add-on disable). S (the main
search) runs in Blender's own process, step by step, under the early-refusal / cap / per-step watchdog rules of E's
deadline harness."""
import time

import numpy as np

from . import quality

CFG = dict(a_budget=5.0, sq=0.6, smargin=0.0, s_cap=6.0, t_first=3.0,
           s_grace=0.5, e_budget=5.0, e_min=1.0, dopt_budget=1.5, total=20.0, tie=0.02,
           q_band=0.6, sq_band=0.75,   # band rule: a band-like outline (Q < q_band) uses sq_band
           flow_deadline=0.3)   # the flow stage on the picked mesh
_LIVE = set()   # the cancel tokens of the runs in progress


class Token:
    """a run's cancel token: a flag (set / is_set); a background job in a worker gets it as a message."""
    __slots__ = ('_set',)

    def __init__(self):
        self._set = False

    def set(self):
        self._set = True

    def is_set(self):
        return self._set


def cancel_token():
    """a run's cancel token. steps() sets it on every exit; the caller may set it earlier (cancel)."""
    return Token()


def cancel_all():
    """cancel every run in progress (add-on disable: ops.common.release_all). Each background job stops
    at its next cancel check; a step in progress finishes first."""
    for token in list(_LIVE):
        token.set()


def shutdown():
    """add-on disable: stop the worker processes."""
    from . import worker
    worker.shutdown()


class _Job:
    """One job in a worker process (worker.py); the caller reads the
    result in Blender. cancel() sets the run's token, which the job polls at its deadline checks."""
    def __init__(self, fn, *args, token=None):
        from . import worker
        self._done, self.result, self.exc, self.token, self.fn = False, None, None, token, fn
        self.args = [worker.TOKEN if token is not None and a is token else a for a in args]
        worker.submit(self)

    @property
    def done(self):
        if not self._done:
            from . import worker
            worker.pump()
        return self._done

    def finish(self, result, exc):
        self.result, self.exc, self._done = result, exc, True

    def cancel(self):
        if self.token is not None:
            self.token.set()


def _sq(qiso, c):
    """The SJ threshold of the shipped S trigger (band rule)."""
    sq = c['sq']
    if qiso < c['q_band']:   # a band-like outline also runs the S part while A's minSJ < 0.8
        sq = max(sq, c['sq_band'])
    return sq


def _trigger(qa, sq):
    """speed #1: ONE trigger for the shipped order and the overlap path (no divergence)."""
    return qa is None or qa['strip'] or qa['val6'] or qa['minSJ'] < sq


def _start_a(world, budget, token=None, stamp=None):
    """A as a worker job (one A at a time: worker.py). stamp: receives 't', when A starts."""
    return _Job(_run_a, world, budget, token, stamp, token=token)


def _wait(job):
    while not job.done:
        yield 'wait'   # the caller may sleep / return to its event loop; S steps yield None (dense)


def _run_quick(mesh, world, cancel=None, t_end=None):
    """quality.quick; stops (quality.Stopped) between its chunks when the run is cancelled or past t_end."""
    def check():
        if cancel is not None and cancel.is_set():
            raise quality.Stopped('cancelled')
        if t_end is not None and time.perf_counter() > t_end:
            raise quality.Stopped('total')
    check()
    return quality.quick(mesh, world, check)


def _quick(mesh, world, token=None, log=None, lane=None, left=None):
    """quality.quick as a worker job; the generator yields meanwhile. left: seconds to the run's hard cap (CFG total);
    at or past it the check is skipped. A skipped, stopped or failed check gives None (that candidate is not used);
    it never ends the whole run."""
    if left is not None and left <= 0:
        if log is not None:
            log['stop']['quick_' + str(lane)] = 'total'
        return None
    j = _Job(_run_quick, mesh, world, token, None if left is None else time.perf_counter() + left, token=token)
    yield from _wait(j)
    if j.exc is not None:
        if log is not None and not (token is not None and token.is_set()):
            if type(j.exc).__name__ == 'Stopped':   # quality.Stopped, rebuilt from its class name
                log['stop']['quick_' + str(lane)] = 'total'
            else:
                log['exc']['quick_' + str(lane)] = type(j.exc).__name__
        return None
    return j.result


def _run_a(world, budget, cancel=None, stamp=None):
    """A running job of a cancelled run returns its best so far or refuses at its next DP step; a waiting one refuses
    without running (the run's token, set before or after A starts). stamp['t'] is when A
    starts; waiting for another run's A is not A's time (its own budget already starts inside fill_a). One A at a
    time: worker.py dispatches the next A when the previous one has ended."""
    from . import fill_a
    if cancel is not None and cancel.is_set():
        raise fill_a.FillRefused('cancelled before start')
    if stamp is not None:
        stamp['t'] = time.perf_counter()
    return fill_a.fill(world, budget_s=budget, cancel=cancel)


def _run_e(world, budget, cancel=None, flags=None):
    """flags['E'] when the budget cut E (fallback5's flag: its plans or its refusal) or left no
    time for its polish (the polish is skipped below 15 % of the budget)."""
    from . import fallback5, fallback5_dopt
    t = time.perf_counter()
    try:
        return fallback5_dopt.fill(world, budget_s=budget, cancel=cancel)
    finally:
        if flags is not None and (getattr(fallback5._CTX, 'budget_hit', False) or time.perf_counter() - t >= .85 * budget):
            flags['E'] = True


def _run_dopt(mesh, budget, cancel=None, flags=None):
    from . import dopt_v4
    P0 = np.asarray(mesh['positions'], float)
    pos, info = dopt_v4.optimize(P0, mesh['faces'], mesh['boundary'], time_budget=budget,
                                 cancel=None if cancel is None else cancel.is_set)
    if flags is not None and float((info or {}).get('seconds', 0.)) >= budget - .02:
        flags['D'] = True   # its time budget ended the polish
    pos = np.asarray(pos, float)
    pos[:mesh['boundary']] = P0[:mesh['boundary']]
    return dict(mesh, positions=pos.tolist())


def _run_flow(mesh, deadline, cancel=None, flags=None):
    from . import flow_stage2   # keeps the min-SJ floor without freezing band zig-zags
    t = time.perf_counter()
    out = flow_stage2.smooth(mesh, deadline_s=deadline, cancel=cancel)
    if flags is not None and time.perf_counter() - t >= deadline:
        flags['flow'] = True   # its deadline ended the flow stage
    return out


def _flowed(mesh, c, token=None, flags=None, left=None):
    """the flow stage on a mesh that is SHOWN (the 1 s provisional, the other New entries), in a job (the caller
    keeps yielding); the mesh unchanged if flow is off or fails, or the run is at its hard cap (left <= 0)."""
    if mesh is None or (left is not None and left <= 0):
        return mesh
    fj = _Job(_run_flow, mesh, c['flow_deadline'] if left is None else min(c['flow_deadline'], left), token, flags, token=token)
    yield from _wait(fj)
    return fj.result if fj.exc is None and fj.result is not None else mesh


def _stop_class(exc):
    if exc is None:
        return None
    name = type(exc).__name__
    return {'BudgetExceeded': 'budget', 'FillRefused': 'refused', 'DeadlineExceeded': 'watchdog'}.get(name, 'exception:' + name)


def s_steps(ctx, world, budget_s, t_first_s, grace_s, info, cancel=None):
    """The S part (the main search) as a generator: an isolated throwaway instance,
    pinned settings (FLOW, floor 0.20, settle off for the instance, AUTO pattern), early refusal and step cap. The
    S deadline (and the run's cancel token) is checked cooperatively at every step boundary of the main search.
    with ctx['clock'] (the hidden precompute) the deadlines run on active time: the time the caller
    leaves this generator suspended is not counted while clock['active'] (the takeover turns it off: real time again).
    no preference is read during the run (Search pinned FAST; the other Fill options are fixed).
    Returns (mesh | None)."""
    import types
    from mathutils import Matrix
    class DeadlineExceeded(Exception):   # checked cooperatively at step boundaries
        pass
    fill262, layouts, op, tri = ctx['fill'], ctx['layouts'], ctx['op'], ctx['triangle']
    context = ctx['context']
    clock = ctx.get('clock')

    def now():
        return time.perf_counter() - (clock['paused'] if clock is not None else 0.)

    t0 = now(); t_end = t0 + budget_s
    info.update(stop=None, first=None, steps=0)
    inst = types.SimpleNamespace(state=types.SimpleNamespace(panels=[], mw=Matrix.Identity(4), view=types.SimpleNamespace(srcs=[])), plan=None,
                                 portfolio=layouts.Portfolio(ranking='FLOW', quality_floor=.20),
                                 settle_on=True)
    if hasattr(op.PPO_OT_fill, '_cuts_steps'):
        inst._cuts_steps = types.MethodType(op.PPO_OT_fill._cuts_steps, inst)
    inst._settle = lambda: None
    inst.newfill_more = True   # pins Search FAST / Ranking FLOW in the main search (ops._legacy_pref)
    st = dict(pattern='AUTO', rotation=-1, boundary_shift=0, _ranking='FLOW', _quality_floor=.20)

    class _Stop(Exception):
        pass

    def accept(p):
        if p is None:
            return
        if p.get('_legacy_layout'):
            inst.portfolio.add_legacy(p, select_if_empty=inst.plan is None)
        else:
            inst.portfolio.add(p)
        inst.plan = inst.portfolio.plan or p
        if info['first'] is None:
            info['first'] = now() - t0

    def check():
        t = now(); info['steps'] += 1
        if cancel is not None and cancel.is_set():
            info['stop'] = 'cancelled'; raise _Stop
        if inst.plan is None and t - t0 > t_first_s:
            info['stop'] = 'early_refusal'; raise _Stop
        if t > t_end:
            info['stop'] = 'budget'; raise _Stop

    def step(job):
        left = t_end + grace_s - now()
        if left <= 0:
            raise DeadlineExceeded
        return next(job)   # a step is short; the check runs at every step boundary

    def consume(job, acc=True):
        try:
            while True:
                try:
                    u = step(job)
                except StopIteration as d:
                    if d.value is not None and acc:
                        accept(d.value)
                    return d.value
                if isinstance(u, dict):
                    if 'preview' in u or 'candidate' in u:
                        accept(u.get('preview', u.get('candidate')))
                check()
                paused = time.perf_counter()
                yield None
                if clock is not None and clock.get('active'):
                    clock['paused'] += time.perf_counter() - paused   # idle, not S's time
        finally:
            job.close()

    try:
        try:
            inst.replan_started = time.perf_counter()
            try:
                yield from consume(op.PPO_OT_fill._plan_steps(inst, context, world, st))
            except fill262.FillError:
                pass
            yield from consume(op.PPO_OT_fill._refine_steps(inst, context, world, st))
            if len(world) % 2 and inst.plan is not None:
                ts = dict(st)
                entries = yield from consume(tri.candidates_steps(world, ts, inst.plan, None, reference=inst.portfolio.search_plan), acc=False)
                sel = next((r['plan'] for r in entries or () if r.get('_auto_selected')), None)
                for e in entries or ():
                    accept(e['plan'])
                if sel is not None:
                    inst.triangle_selected_plan = sel
        except _Stop:
            pass
    except DeadlineExceeded:
        info['stop'] = 'watchdog'
    info['seconds'] = now() - t0
    if info['stop'] == 'early_refusal':
        return None
    s = getattr(inst, 'triangle_selected_plan', None) or inst.portfolio.plan or inst.plan
    if s is None:
        return None
    info['plans'] = []
    return _plan_to_mesh(fill262, s)


def _qiso(world):
    """Isoperimetric quotient Q = 4*pi*A / P^2 of the rim polygon in its best-fit plane (rotation/scale invariant)."""
    w = np.asarray(world, float); c = w.mean(0)
    nrm = quality._plane(w, len(w))
    u = np.cross(nrm, [1., 0, 0]) if abs(nrm[0]) < .9 else np.cross(nrm, [0, 1., 0]); u /= np.linalg.norm(u); v = np.cross(nrm, u)
    w2 = np.stack([(w - c) @ u, (w - c) @ v], 1)
    area = .5 * abs(float(np.sum(w2[:, 0] * np.roll(w2[:, 1], -1) - np.roll(w2[:, 0], -1) * w2[:, 1])))
    perim = float(np.linalg.norm(np.roll(w2, -1, 0) - w2, axis=1).sum())
    return 4 * np.pi * area / max(perim ** 2, 1e-30)


def _flatten(mesh):
    """Project the interior onto the rim's best plane (through the rim centre); the rim is unchanged. The main
    search lifts its interior harmonically; this search's quality check works in the rim plane and to_plan lifts."""
    P = np.asarray(mesh['positions'], float); n = int(mesh['boundary'])
    if len(P) <= n:
        return mesh
    nrm = quality._plane(P, n); c = P[:n].mean(0)
    P = P.copy(); P[n:] = P[n:] - np.outer((P[n:] - c) @ nrm, nrm)
    return dict(mesh, positions=P.tolist())


def _plan_to_mesh(fill262, s):
    order = s['order']; Pp = np.asarray(s['positions']); faces = [list(q) for q in fill262.faces_of(s)]
    n = sum(o is not None for o in order); used = {v for q in faces for v in q}; new = [None] * len(Pp); nxt = n
    for i in range(len(Pp)):
        o = order[i] if i < len(order) else None
        if o is None and i not in used:
            continue
        if o is None:
            new[i] = nxt; nxt += 1
        else:
            new[i] = int(o)
    pos = [None] * nxt
    for i, j in enumerate(new):
        if j is not None:
            pos[j] = Pp[i].tolist()
    return dict(positions=pos, faces=[[new[v] for v in q] for q in faces], boundary=n)


def _pick(cands, tie):
    ok = [(k, q) for k, q in cands if quality.good(q)]
    if not ok:
        return None
    best = max(q['minSJ'] for _, q in ok)
    near = [(k, q) for k, q in ok if q['minSJ'] >= best - tie]
    return min(near, key=lambda kq: (kq[1]['irr'], -kq[1]['minSJ'], 'ASE'.index(kq[0])))   # equal poles -> higher minSJ


def steps(world, s_ctx=None, cancel=None):
    """The pipeline generator (module docstring). cancel: this run's token (cancel_token(); made when None). The finally
    sets it on every exit (finished, closed by the caller, or failed): no background job of the run keeps working.
    """
    token = cancel if cancel is not None else cancel_token()
    _LIVE.add(token)   # cancel_all reaches it
    try:
        return (yield from _steps(world, s_ctx, token))
    finally:
        token.set()
        _LIVE.discard(token)
        try:   # forward the cancel now: no pump may follow when the tool is left during a wait
            from . import worker
            worker.pump()
        except Exception:
            pass


def _steps(world, s_ctx, token):
    c = CFG
    world = np.asarray(world, float); n = len(world)
    t0 = time.perf_counter(); log = dict(pick=None, s_ran=False, s_stop=None, a_budget_hit=False, exc={}, stop={}, seconds={}, minSJ={})
    flags = {}   # search parts whose own time budget cut their result (E, dopt 'D', flow)
    with np.errstate(all='ignore'):   # NaN / Inf / duplicate / collinear rims refuse at once (no search parts)
        degenerate = (n < 3 or not np.isfinite(world).all() or len(np.unique(world, axis=0)) < n
                      or not _qiso(world) >= 1e-9)
    if degenerate:
        log['stop']['input'] = 'degenerate'; log['seconds']['total'] = 0.; log['degraded'] = []
        log['cancelled'] = token.is_set()
        return dict(mesh=None, lane=None, cands=[], log=log)
    clock = (s_ctx or {}).get('clock')   # the hidden precompute's S-part active-time clock

    def elapsed():
        """The run's time; with an active clock, the S part's idle gaps between its slices are not counted."""
        return time.perf_counter() - t0 - (clock['paused'] if clock is not None else 0.)

    def left():
        """seconds to the run's hard cap (CFG total)."""
        return c['total'] - elapsed()

    def lane_done(name, job, t):
        log['seconds'][name] = time.perf_counter() - t
        if job.exc is not None:
            log['stop'][name] = _stop_class(job.exc)
            if log['stop'][name].startswith('exception:'):
                log['exc'][name] = type(job.exc).__name__
        return job.result if job.exc is None else None

    # A (a worker job)
    log['qiso'] = _qiso(world)
    sq = _sq(log['qiso'], c); log['sq'] = sq
    t = time.perf_counter(); stamp = {}; job = _start_a(world, c['a_budget'], token, stamp)
    try:
        yield from _wait(job)
    finally:
        if not job.done:
            job.cancel()   # a closed preview cancels its A request
    ra = lane_done('A', job, stamp.get('t', t))   # from when A holds the A lock
    log['a_budget_hit'] = type(job.exc).__name__ == 'BudgetExceeded' or bool((ra or {}).get('info', {}).get('budget_hit'))
    if log['seconds'].get('A', 0.) >= c['a_budget'] - .02:
        log['a_budget_hit'] = True   # also from A's elapsed time
    qa = (yield from _quick(quality.as_mesh(ra, n), world, token, log, 'A', left())) if ra is not None else None
    trig = _trigger(qa, sq)
    cands = [('A', qa)] if qa is not None else []
    if qa is not None:
        log['minSJ']['A'] = qa['minSJ']
    # S (only on trigger)
    if trig and s_ctx is not None and qa is not None:
        pm = yield from _flowed(qa['mesh'], c, token, left=left())   # shown -> flowed (never cached: no flags)
        yield {'provisional': pm, 'lane': 'A'}   # first-result deadline: A's result while S runs
    if trig and s_ctx is not None and elapsed() < c['total']:
        log['s_ran'] = True; info = {}; t = time.perf_counter(); ms = None
        try:
            ms = yield from s_steps(s_ctx, world, min(c['s_cap'], c['total'] - elapsed()), c['t_first'], c['s_grace'], info,
                                   cancel=token)
        except Exception as e:   # noqa: BLE001
            log['exc']['S'] = type(e).__name__; info['stop'] = 'exception:' + type(e).__name__
        log['s_stop'] = info.get('stop'); log['seconds']['S'] = info.get('seconds', time.perf_counter() - t)
    if log['s_ran']:
        if ms is not None:
            ms = _flatten(ms)   # : curved / Topo outlines keep S as a candidate
        if ms is not None and elapsed() < c['total'] - .25:
            t = time.perf_counter(); dj = _Job(_run_dopt, ms, min(c['dopt_budget'], max(.1, c['total'] - elapsed())), token, flags, token=token)
            yield from _wait(dj)
            md = lane_done('D', dj, t)
            if md is not None:
                ms = _flatten(md)   # dopt_v4 works in its own frame; back onto the rim plane
        qs = (yield from _quick(ms, world, token, log, 'S', left())) if ms is not None else None
        if qs is not None:
            log['minSJ']['S'] = qs['minSJ']
            if not quality.good(qa) or qs['minSJ'] >= qa['minSJ'] + c['smargin']:
                cands.append(('S', qs))
        log['s_plans'] = info.get('plans', [])
    pick = _pick(cands, c['tie'])
    # E (only if nothing is good), budget capped by the overall deadline
    if pick is None:
        e_left = c['total'] - elapsed()
        if e_left >= c['e_min']:
            t = time.perf_counter(); ej = _Job(_run_e, world, min(c['e_budget'], e_left), token, flags, token=token)
            yield from _wait(ej)
            re_ = lane_done('E', ej, t)
            qe = (yield from _quick(quality.as_mesh(re_, n), world, token, log, 'E', left())) if re_ is not None else None
            if qe is not None:
                log['minSJ']['E'] = qe['minSJ']; cands.append(('E', qe))
                if quality.good(qe):
                    pick = ('E', qe)
        else:
            log['stop']['E'] = 'overall_deadline'
        if pick is None and cands:   # nothing good: best valid (fewest defects, then minSJ, then A > S > E)
            pick = min(cands, key=lambda kq: (int(kq[1]['coarse']) + int(kq[1]['val6']), -kq[1]['minSJ'], 'ASE'.index(kq[0])))
    log['pick'] = pick[0] if pick else None
    order = sorted(cands, key=lambda kq: (pick is None or kq[1] is not pick[1], not quality.good(kq[1]), -kq[1]['minSJ']))
    mesh = pick[1]['mesh'] if pick else None
    if mesh is not None and left() > 0:   # flow stage v1, once, on the final picked mesh
        t = time.perf_counter(); fj = _Job(_run_flow, mesh, min(c['flow_deadline'], left()), token, flags, token=token)
        yield from _wait(fj)
        log['seconds']['flow'] = time.perf_counter() - t
        if fj.exc is None and fj.result is not None:
            mesh = fj.result
        else:
            log['exc']['flow'] = type(fj.exc).__name__ if fj.exc is not None else 'None'
        for i, (lane_i, q_i) in enumerate(order):   # ops builds the wheel entries from cands: every New entry shown
            if q_i is pick[1]:                     # the pick: its flowed mesh (= result['mesh'])
                order[i] = (lane_i, dict(q_i, mesh=mesh))
            elif q_i.get('mesh') is not None:      # the other New entries are flowed too
                order[i] = (lane_i, dict(q_i, mesh=(yield from _flowed(q_i['mesh'], c, token, flags, left()))))
    log['seconds']['total'] = elapsed()
    log['degraded'] = sorted(flags)   # the caller never caches a budget-degraded result
    log['cancelled'] = token.is_set()   # the caller never caches a cancelled run's result
    return dict(mesh=mesh, lane=log['pick'], cands=order, log=log)


def to_plan(mesh, lane, lift=True):
    """adapter: a second-search mesh (positions[:n] == world rim, in loop order) -> the plan dict that
    validate_plan / faces_of / commit / project_steps accept. Faces come from the scorer's normalised (CCW) mesh.
    lift: interior heights above the rim's plane are harmonic from the loop, as the main search's solve_steps does (pen3d)."""
    from ..patterns import geom
    P = np.asarray(mesh['positions'], float); n = int(mesh['boundary']); F = [tuple(int(i) for i in f) for f in mesh['faces']]
    nrm = quality._plane(P, n)
    if lift and len(P) > n:
        centre = P[:n].mean(0)
        heights = (P[:n] - centre) @ nrm
        try:
            h = geom.laplace(np.column_stack((heights, np.zeros(n))), len(P), F, iterations=240)[:, 0]
            P = P.copy(); P[n:] = P[n:] + h[n:, None] * nrm
        except Exception:   # noqa: BLE001  (the flat patch stays; logged by the caller via validate_plan)
            pass
    u = np.cross(nrm, [1., 0, 0]) if abs(nrm[0]) < .9 else np.cross(nrm, [0, 1., 0]); u /= np.linalg.norm(u); v = np.cross(nrm, u)
    return dict(positions=P, quads=F, boundary=n, virtual=None, collapse=None, order=list(range(n)), normal_ccw=np.cross(u, v),
                pattern='newfill', rotation=0, corners=(), anchors=[], shift=0, layout=None, _newfill_lane=lane)
