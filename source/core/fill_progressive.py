# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Cooperative AUTO seeds and refinement with deterministic network-plan work.

Search quotas count deterministic clock observations rather than elapsed time.
The clock/deadline overrides exist only while advancing this generator, never
while Blender processes events. OFF/manual planning uses released real time;
AUTO network planning and continuation use explicit deterministic work quotas.
"""
from collections import OrderedDict
import numpy as np
from . import fill, network_fill
from .patterns import geom

_libraries = None


def prepare():
    """Compile immutable pattern tables when opting in, before tool input."""
    global _libraries
    if _libraries is None:
        np.median(np.array([0., 1.]))
        from .patterns import chords
        _libraries = {key: chords.build_library(key) for key in (False, True)}


class WUClock(geom.WorkUnitClock):
    """Work-unit clock of the deterministic steps. A budget is still read as seconds (ticks * quantum), but the ticks
    are work units:
      * 1 unit per clock read made outside an untangle solve (these reads sit at the planner's cooperative step
        boundaries, so they count steps of work);
      * the untangle: reads made inside a solve cost nothing; instead each solve charges 1 unit per call
        (geom.untangle_steps) + 1 unit per energy+gradient evaluation (geom._lbfgs_steps)."""
    def __init__(self, quantum=.00005):
        self.quantum = quantum
        self.ticks = 0.0

    def monotonic(self):
        if not geom._UNTANGLE[0]:
            self.ticks += 1
        return self.ticks * self.quantum

    perf_counter = monotonic


def deterministic_steps(job, quantum=.00005, preserve_updates=False, max_ticks=None, stats=None):
    prepare()
    clock = WUClock(quantum)   # every module's budget reads go through geom.clock(), P1's too
    deadline = [None]
    preview = fill.boundary_preview
    solver = preview.solver
    chords = preview.chords
    caches = (OrderedDict(), OrderedDict())
    while True:
        if max_ticks is not None and clock.ticks >= max_ticks:
            old_deadline = geom.push_deadline(deadline)
            try:
                job.close()
            finally:
                geom.pop_deadline(old_deadline)
            if stats is not None:
                stats.update(work_ticks=clock.ticks, budget_exhausted=True)
            return None
        old_clock = geom.push_clock(clock)
        old_deadline = geom.push_deadline(deadline)
        old_preview_cache = preview.push_cache(caches[0])
        old_solver_cache = solver.push_cache(caches[1])
        old_library = chords.push_library(_libraries)
        try:
            try:
                update = next(job)
            except StopIteration as done:
                return done.value
        finally:
            geom.pop_clock(old_clock)
            geom.pop_deadline(old_deadline)
            preview.pop_cache(old_preview_cache)
            solver.pop_cache(old_solver_cache)
            chords.pop_library(old_library)
        if stats is not None:
            stats['work_ticks'] = clock.ticks
        try:
            yield update if preserve_updates else None
        except GeneratorExit:
            # Close nested time_budget contexts against their own deadline.
            try:
                geom.push_deadline(deadline)
                job.close()
            finally:
                geom.pop_deadline(old_deadline)
            raise


def utility(plan):
    if plan is None:
        return -float('inf')
    positions = np.asarray(plan['positions'])
    faces = fill.faces_of(plan)
    # Quality in 3D, including source projection.
    minimum = 1.
    for count in sorted({len(q) for q in faces}):
        p = positions[np.asarray([q for q in faces if len(q) == count])]
        d = np.roll(p, -1, axis=1) - p
        q = np.linalg.norm(np.cross(d, np.roll(d, -1, axis=1)), axis=2)
        q /= np.maximum(np.linalg.norm(d, axis=2) * np.linalg.norm(np.roll(d, -1, axis=1), axis=2), 1e-30)
        minimum = min(minimum, float(q.min()))
    poles = fill._interior_poles(dict(positions=positions, quads=faces))
    return minimum * .7 ** poles if minimum + 1e-12 >= .25 else -float('inf')


def eligible(settings):
    return not settings.get('anchors')


def canonical_steps(world, settings, incumbent):
    """Complete deterministic network-plan seeds while keeping the displayed incumbent."""
    if frozen_quality_conflict(world):
        return incumbent
    best = incumbent
    legacy = None
    for quantum in (.00001, .00005):
        # A network-plan result with several singularities also gets a fixed shorter
        # work schedule. The trigger uses only independently solved network-plan data,
        # so the immediate timed preview cannot change the final candidate set.
        if quantum == .00005 and (legacy is None or fill._interior_poles(legacy) <= 2):
            break
        job = _canonical_work_steps(world, settings, legacy, quantum)
        try:
            while True:
                try:
                    update = next(job)
                except StopIteration as done:
                    legacy = done.value
                    break
                if isinstance(update, dict) and 'preview' in update:
                    proposed = update['preview']
                    if utility(proposed) > utility(best) + 1e-12:
                        best = proposed
                        yield {'preview': best}
                    else:
                        yield {'candidate': proposed}
                elif isinstance(update, dict) and 'candidate' in update:
                    yield update
                else:
                    yield None
        finally:
            job.close()
    return best


def _canonical_work_steps(world, settings, incumbent, quantum=.00001):
    best = incumbent
    # Complete both network-plan scheduling branches. A timed legacy slice can finish
    # or defer; running both with fixed work quotas makes that choice explicit.
    for legacy_seconds in (.2, 0.):
        capture = []
        try:
            options = dict(settings, _legacy_seconds=legacy_seconds, _legacy_capture=capture)
            canonical = yield from deterministic_steps(fill.solve_steps(world, options), quantum)
        except fill.FillError:
            canonical = None
        if canonical is not None:
            canonical['network_method'] = '1.5.0_deterministic'
            if utility(canonical) > utility(best) + 1e-12:
                best = canonical
                yield {'preview': best}
            else:
                yield {'candidate': canonical}
        # Released selection precedes final Winslow polish. Compare the fully
        # polished root checkpoints too, so deadline-selected seeds cannot win
        # solely because their final quality improves by a different amount.
        for frame in capture:
            seen = set()
            for result in frame['pool']:
                key = (tuple(tuple(q) for q in result['quads']),
                       np.asarray(result['positions']).tobytes())
                if key in seen:
                    continue
                seen.add(key)
                try:
                    proposed = yield from _polish_legacy_steps(frame, result)
                except fill.FillError:
                    continue
                if utility(proposed) > utility(best) + 1e-12:
                    best = proposed
                    yield {'preview': best}
                else:
                    yield {'candidate': proposed}
    return best


def frozen_quality_conflict(world):
    """A flat fixed convex corner bounds the sine quality of every incident quad.

    Only use the certificate on essentially exact planar data, with a generous
    margin for projection/rounding. Nonplanar and degenerate cages keep searching.
    """
    points = np.asarray(world, float)
    if len(points) < 4 or not np.isfinite(points).all():
        return False
    try:
        centre, u, v, normal = fill._frame(points)
    except (ValueError, np.linalg.LinAlgError):
        return False
    shortest = float(np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1).min())
    if shortest <= 1e-15 or np.max(np.abs((points-centre) @ normal)) > shortest*1e-9:
        return False
    xy = np.column_stack(((points-centre) @ u, (points-centre) @ v))
    area = float(np.sum(xy[:, 0]*np.roll(xy[:, 1], -1)-xy[:, 1]*np.roll(xy[:, 0], -1)))
    if abs(area) <= max(float(np.ptp(xy, axis=0).max()), 1e-30)**2*1e-12:
        return False
    if area < 0:
        xy = xy[::-1]
    angles = np.pi-geom.turning(xy)
    acute = angles[(angles > 0) & (angles < np.pi/2)]
    return bool(len(acute) and float(np.sin(acute).min()) < .25-1e-6)


def _polish_legacy_steps(frame, result):
    pts = frame['pts'];n = len(pts)
    plane = yield from geom.winslow_steps(result['positions'], n, result['quads'])
    heights = (pts-frame['centre']) @ frame['normal']
    h = geom.laplace(np.column_stack((heights, np.zeros(n))), len(plane), result['quads'], iterations=240)[:, 0]
    pos = frame['centre'] + plane[:, :1]*frame['u'] + plane[:, 1:2]*frame['v'] + h[:, None]*frame['normal']
    pos[:n] = pts
    layout = dict(result, positions=plane, total=len(plane))
    plan = dict(positions=pos, quads=[tuple(q) for q in result['quads']], boundary=n,
                virtual=frame['virtual'], collapse=frame['collapse'], order=frame['order'],
                normal_ccw=frame['normal'], pattern=result['pattern'], rotation=result['rotation'],
                corners=tuple(result['corners']), anchors=result['anchors'], shift=frame['shift'],
                layout=layout, network_method='1.5.0_deterministic')
    fill._accept_plan(plan)
    return plan


def initial_steps(world, settings, stats=None):
    """Use a fixed network-plan work quota for AUTO; OFF keeps its released clock. stats -> work_ticks (work clock)."""
    frames = settings.get('_legacy_frames')
    capture = []
    options = dict(settings, _legacy_capture=capture, _legacy_mark=True) if frames is not None else settings
    try:
        return (yield from deterministic_steps(fill.solve_steps(world, options), .00005, stats=stats))
    finally:
        if frames is not None:
            frames.extend(capture)


def legacy_steps(world, settings, frames=()):
    """Retain the exact completed legacy incumbent, or finish its fixed schedule."""
    if not eligible(settings):
        return None
    for frame in frames:
        for result in frame['pool']:
            if result.get('_legacy_seed'):
                try:
                    plan = yield from _polish_legacy_steps(frame, result)
                except fill.FillError:
                    continue
                return dict(plan, _legacy_layout=True)
    options = dict(settings, _policy='legacy', _takayama=False)
    try:
        plan = yield from deterministic_steps(fill.solve_steps(world, options), .00001)
    except fill.FillError:
        return None
    return dict(plan, _legacy_layout=True)


def seed_steps(world, settings, incumbent=None):
    """Find and expose the first validated cheap candidate before the network-plan search."""
    points = np.asarray(world, float)
    if (not eligible(settings) or len(points) < 8 or len(points) % 2
            or len(points) > fill.MAX_POINTS or points.shape != (len(points), 3)
            or not np.isfinite(points).all()):
        return incumbent
    result = yield from _network_steps(points, dict(settings, _lattice_only=True), None)
    if result is None:
        result = yield from _network_steps(world, settings, None)
    return result if utility(result) > utility(incumbent)+1e-12 else incumbent


def refine_steps(world, settings, incumbent):
    if not eligible(settings):
        return incumbent
    # Search cheap pole-free children before network child templates. A failed
    # proposal is followed by the complete existing network schedule.
    result = incumbent
    if not settings.get('_network_seeded'):
        result = yield from seed_steps(world, settings, incumbent)
    if settings.get('_network_only'):
        return result
    if result is not None and result is not incumbent:
        yield {'preview': result}
    # A validated pole-free network has reached the primary topology goal.
    # Network-plan continuation is reserved for unresolved or singular layouts.
    if result is not None and fill._interior_poles(result) == 0:
        return result
    return (yield from canonical_steps(world, settings, result))


def _network_steps(world, settings, incumbent):
    """Return a validated world-space Fill plan, retaining the incumbent on ties."""
    if not eligible(settings) or len(world) % 2 or len(world) < 8:
        return incumbent
    points = np.asarray(world, float)
    centre, u, v, _ = fill._frame(points)
    for axis in (u, v):
        if float(np.sum(((points-centre) @ axis)**3)) < 0:
            axis *= -1
    normal = np.cross(u, v)
    xy = np.column_stack(((points-centre) @ u, (points-centre) @ v))
    order = list(range(len(points)))
    if float(np.sum(xy[:, 0]*np.roll(xy[:, 1], -1)-xy[:, 1]*np.roll(xy[:, 0], -1))) < 0:
        order.reverse()
    xy = xy[order]
    legacy_seed = dict(status='refused')   # both callers pass incumbent=None
    pool = []
    options = dict(settings, partition_all=True, _candidate_pool=pool)
    result = yield from deterministic_steps(network_fill.panel_steps(xy, settings=options, incumbent=legacy_seed))
    for candidate in pool:
        if candidate.get('method') != '1.5.0':
            plan = _network_plan(points, centre, u, v, normal, order, candidate)
            yield {'candidate': plan}
    if result.get('status') != 'filled' or result.get('method') == '1.5.0':
        return incumbent
    plan = _network_plan(points, centre, u, v, normal, order, result)
    if utility(plan) > utility(incumbent) + 1e-12:
        return plan
    return incumbent


def _network_plan(points, centre, u, v, normal, order, result):
    plane, quads = result['positions'], result['quads']
    heights = (points[order]-centre) @ normal
    h = geom.laplace(np.column_stack((heights, np.zeros(len(points)))), len(plane), quads, iterations=240)[:, 0]
    positions = centre + plane[:, :1]*u + plane[:, 1:2]*v + h[:, None]*normal
    positions[:len(points)] = points[order]
    layout = dict(positions=plane, quads=quads, total=len(plane), pattern='partition',
                  rotation=0, corners=(), anchors=[], score=result['qmean'])
    plan = dict(positions=positions, quads=quads, boundary=len(points), virtual=None,
                collapse=None, order=order, normal_ccw=normal, pattern='partition',
                rotation=0, corners=(), anchors=[], shift=0, layout=layout,
                network_method=result['method'])
    fill.validate_plan(plan)
    return plan


def cls_steps(world, settings):
    """classification candidates (fill_cls, PLAN_A v2): cheapest corner set per side count, Takayama."""
    if not eligible(settings):return
    from . import fill_cls
    points=np.asarray(world,float)
    if len(points)<4 or len(points)%2:return
    centre,u,v,_=fill._frame(points)
    normal=np.cross(u,v);xy=np.column_stack(((points-centre)@u,(points-centre)@v))
    order=list(range(len(points)))
    if np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1))<0:order.reverse()
    xy=xy[order]
    job=fill_cls.candidate_steps(xy,dict(settings))
    try:
        for update in job:
            if isinstance(update,dict) and 'candidate' in update:
                result=update['candidate']
                plan=_network_plan(points,centre,u,v,normal,order,result)
                plan['_corner_cut_candidate']=True   # supplementary: kept out of the reference (search) portfolio
                plan['_cls_candidate']=True
                plan['cls_sides']=result['cls_sides']
                yield {'candidate':plan}
            else:yield None
    finally:job.close()


def p1_cut_steps(world, settings, audit=None):
    """P1 cut candidates (fill_p1); ranking unchanged."""
    if not eligible(settings):return
    from . import fill_p1
    points=np.asarray(world,float)
    if len(points)<8 or len(points)%2:return
    centre,u,v,_=fill._frame(points)
    for axis in (u,v):
        if float(np.sum(((points-centre)@axis)**3))<0:axis*=-1
    normal=np.cross(u,v);xy=np.column_stack(((points-centre)@u,(points-centre)@v))
    order=list(range(len(points)))
    if np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1))<0:order.reverse()
    xy=xy[order]
    audit=audit if audit is not None else {}
    job=deterministic_steps(fill_p1.candidate_steps(xy,dict(settings),audit),
        preserve_updates=True,max_ticks=40000,stats=audit)
    try:
        for update in job:
            if isinstance(update,dict) and 'candidate' in update:
                result=update['candidate']
                plan=_network_plan(points,centre,u,v,normal,order,result)
                plan['_corner_cut_candidate']=True
                plan['_p1_candidate']=True
                plan['p1_kind']=result['p1_kind']
                plan['auto_cut_paths']=result['internal_paths']
                plan['auto_cut_count']=len(result['internal_seams'])
                yield {'candidate':plan}
            else:yield None
    finally:job.close()
