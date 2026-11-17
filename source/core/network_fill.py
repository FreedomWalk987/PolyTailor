# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Standalone network prototype. OFF never alters released Fill or its settings."""
from collections import Counter
import numpy as np
from . import fill, boundary_preview, network_medial as medial
from . import network_counts, panel_network
from .patterns import geom
from .patterns.validate import validate_steps, InvalidFill

QUALITY_FLOOR = .25


def run(job):
    try:
        while True:
            next(job)
    except StopIteration as done:
        return done.value
    finally:
        job.close()


def metrics(positions, quads):
    edges = Counter(tuple(sorted((a, b))) for q in quads for a, b in zip(q, q[1:]+q[:1]))
    boundary = {v for edge, count in edges.items() if count == 1 for v in edge}
    degree = Counter(v for edge in edges for v in edge)
    quality = geom.quad_quality(np.asarray(positions)[:, :2], quads)
    return dict(faces=len(quads), poles=sum(degree[v] != 4 for v in degree if v not in boundary),
                qmin=float(quality.min()), qmean=float(quality.mean()))


def legacy_fill_steps(points, settings=None):
    """Released 1.5.0 invocation, remapped to the original CCW boundary indices."""
    p = np.asarray(points, float)
    if p.shape[1] == 2:
        p = np.column_stack((p, np.zeros(len(p))))
    begin = geom.clock().perf_counter()
    try:
        plan = yield from fill.solve_steps(p, dict(settings or {}))
        fill.validate_plan(plan)
        qs = fill.faces_of(plan)
        if plan['virtual'] is not None or any(len(q) != 4 for q in qs):
            return dict(status='mixed_faces', reason='Odd frozen boundary uses released midpoint collapse',
                        legacy_plan=plan, elapsed_ms=(geom.clock().perf_counter()-begin)*1000)
        n = len(p)
        order = plan['order']
        remap = list(order)+list(range(n, len(plan['positions'])))
        positions = np.asarray(plan['positions']).copy()
        positions[:n] = p
        positions[n:] = np.asarray(plan['positions'])[n:]
        quads = tuple(tuple(remap[v] for v in q) for q in qs)
        if geom.min_corner_area(positions[:, :2], quads) < 0:
            quads = tuple(q[::-1] for q in quads)
        run(validate_steps(p[:, :2], positions[:, :2], quads))
        return dict(status='filled', positions=positions[:, :2], quads=quads,
                    method='1.5.0', legacy_pattern=plan['pattern'],
                    corners=tuple(order[i] for i in plan['corners']),
                    elapsed_ms=(geom.clock().perf_counter()-begin)*1000, **metrics(positions, quads))
    except (fill.FillError, InvalidFill) as exc:
        return dict(status='refused', reason=str(exc), elapsed_ms=(geom.clock().perf_counter()-begin)*1000)


def _expand(p, pieces, seams, counts):
    points = list(p); shared = {}
    for (a, b), count in zip(seams, counts):
        ids = []
        for k in range(1, count):
            ids.append(len(points)); points.append(p[a]+(p[b]-p[a])*k/count)
        shared[a, b] = ids
    rings = []
    for poly in pieces:
        ring = []
        for a, b in zip(poly, poly[1:]+poly[:1]):
            ring.append(a)
            ids = shared.get(tuple(sorted((a, b))), [])
            ring.extend(ids if a < b else ids[::-1])
        rings.append(ring)
    return np.asarray(points), rings


def flexible_partition_steps(p, seconds=4., floor=QUALITY_FLOOR, lattice_only=False):
    """Medial child grids with turning seams and exact shared parity counts.

    A child is not assumed to have four geometric corners. The same sampled
    seam is supplied to both child solvers; only the root exterior stays locked
    during the final merged polish. Search limits are explicit, never proofs.
    """
    started=geom.clock().perf_counter()
    deadline=started+seconds
    try:
        with geom.time_budget(seconds):
            graph=yield from medial.skeleton_steps(p)
            cuts=yield from medial.cuts_steps(p,graph)
            layouts=yield from medial.layout_steps(p,cuts)
        # Include every single neck before the much larger combination family.
        singles=[v for v in layouts if len(v[1])==1]
        layouts=singles+[v for v in layouts if len(v[1])>1]
        h=max(float(np.median(np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1))),1e-12)
        for _,seams,pieces in layouts[:40]:
            names=['seam:'+str(i) for i in range(len(seams))]
            eq=[];bounds={};targets={}
            for name,(a,b) in zip(names,seams):
                target=max(1,min(24,round(np.linalg.norm(p[a]-p[b])/h)))
                targets[name]=target;bounds[name]=(max(1,target-4),min(24,target+4))
            halves=[]
            for i,poly in enumerate(pieces):
                half='piece:'+str(i)+':half';halves.append(half);bounds[half]=(2,len(p));targets[half]=len(poly)//2
                row={half:-2};constant=0
                for a,b in zip(poly,poly[1:]+poly[:1]):
                    edge=tuple(sorted((a,b)))
                    if edge in seams:row[names[seams.index(edge)]]=1
                    else:constant+=1
                eq.append(dict(coefficients=row,rhs=-constant,label='piece:'+str(i)+':even_boundary'))
            seen=set()
            for offset in (0,2,-2,4,-4):
                if geom.clock().perf_counter()>=deadline:raise geom.TimeUp()
                wanted=dict(targets)
                for name in names:wanted[name]+=offset
                certificate=yield from network_counts.solve_steps(halves+names,eq,bounds,wanted,seconds=.03)
                # Small infeasible models can finish without yielding. Do not
                # batch an entire cut family into one atomic UI step.
                yield None
                if certificate['status']!='feasible':continue
                counts=tuple(certificate['values'][name] for name in names)
                if counts in seen:continue
                seen.add(counts)
                parent,rings=_expand(p,pieces,seams,counts)
                positions=list(parent);quads=[];methods=[];failed=False
                for ring in rings:
                    if geom.clock().perf_counter()>=deadline:raise geom.TimeUp()
                    local=parent[ring]
                    result=None
                    with geom.time_budget(min(.25,max(.001,deadline-geom.clock().perf_counter()))):
                        try:result=yield from medial.coupled_steps(local,dict(_quality_floor=floor))
                        except geom.TimeUp:pass
                    if result is None and not lattice_only:
                        # Use the released template scheduler directly in the
                        # same normalized XY frame, with a small bounded slice.
                        from .fill_partition import _capped_steps, _capped_site, PartitionError
                        try:
                            cap_seconds=min(.12,max(.001,deadline-geom.clock().perf_counter()))
                            result=yield from _capped_steps(boundary_preview.job(local,[('p',j) for j in range(len(local))],dict(pattern='AUTO',_policy='poles_first',_takayama=True),site=_capped_site(cap_seconds)),cap_seconds)
                        except (PartitionError,boundary_preview.PreviewError,geom.TimeUp):pass
                    if result is None:
                        failed=True;break
                    remap=ring+list(range(len(positions),len(positions)+len(result['positions'])-len(ring)))
                    positions.extend(result['positions'][len(ring):]);quads.extend(tuple(remap[v] for v in q) for q in result['quads'])
                    methods.append(result.get('medial_method',result.get('pattern')))
                if failed:continue
                positions=np.array(positions)
                with geom.time_budget(max(.001,deadline-geom.clock().perf_counter())):
                    if geom.min_corner_area(positions,quads)<=1e-12:
                        positions,ok=yield from geom.untangle_steps(positions,len(p),quads)
                        if not ok:continue
                    positions=yield from geom.winslow_steps(positions,len(p),quads)
                    try:yield from validate_steps(p,positions,quads)
                    except InvalidFill:continue
                if geom.quad_quality(positions,quads).min()<floor:continue
                paths=[];next_id=len(p)
                for (a,b),count in zip(seams,counts):
                    paths.append((a,)+tuple(range(next_id,next_id+count-1))+(b,));next_id+=count-1
                return dict(positions=positions,quads=tuple(quads),internal_seams=seams,internal_counts=counts,
                            internal_paths=paths,count_certificate=certificate,child_methods=methods,method='medial_child_network')
    except geom.TimeUp:pass
    return None


def panel_steps(points, settings=None, incumbent=None):
    settings = dict(settings or {})
    start = geom.clock().perf_counter(); p = np.asarray(points, float)[:, :2]
    if len(p) % 2:
        return dict(incumbent, engine_status='odd_boundary_conflict',
                    engine_ms=(geom.clock().perf_counter()-start)*1000)
    angles=np.pi-geom.turning(p)
    acute=np.flatnonzero(angles<np.pi/2)
    if len(acute):
        k=int(acute[np.argmin(angles[acute])]);upper=float(np.sin(angles[k]))
        if upper<float(settings.get('_quality_floor',QUALITY_FLOOR)):
            return dict(incumbent,engine_status='frozen_corner_quality_conflict',
                        quality_certificate=dict(vertex=k,interior_angle_radians=float(angles[k]),q_upper_bound=upper,
                                                 required=float(settings.get('_quality_floor',QUALITY_FLOOR))),
                        engine_ms=(geom.clock().perf_counter()-start)*1000)
    origin = p.mean(0); scale = float(np.ptp(p, axis=0).max())
    if scale<=1e-15:return dict(status='refused',reason='Degenerate panel',engine_ms=(geom.clock().perf_counter()-start)*1000)
    unit = (p-origin)/scale
    candidate = None
    try:
        with geom.time_budget(float(settings.get('lattice_seconds', .5))):
            candidate = yield from medial.coupled_steps(unit, settings)
            if candidate is not None:
                candidate['method'] = 'study8_lattice'
    except geom.TimeUp:
        pass
    # Only hard panels pay for the extra medial network. The released result
    # remains the incumbent when this bounded search cannot improve it.
    if candidate is None and (incumbent['status'] != 'filled' or settings.get('partition_all', False)):
        candidate = yield from flexible_partition_steps(unit, float(settings.get('partition_seconds', 4.)), float(settings.get('_quality_floor', QUALITY_FLOOR)), bool(settings.get('_lattice_only')))
    candidates = []
    if incumbent['status'] == 'filled':
        candidates.append(incumbent)
    if candidate is not None:
        positions = candidate['positions']*scale+origin
        positions[:len(p)] = p
        try:
            yield from validate_steps(p, positions, candidate['quads'])
            mm = metrics(positions, candidate['quads'])
            if mm['qmin'] >= float(settings.get('_quality_floor', QUALITY_FLOOR)):
                candidates.append(dict(candidate, positions=positions, status='filled', **mm))
        except InvalidFill:
            pass
    if settings.get('_candidate_pool') is not None:
        settings['_candidate_pool'].extend(candidates)
    from . import fill_selection
    best = fill_selection.choose(candidates, settings) or incumbent
    return dict(best, engine_ms=(geom.clock().perf_counter()-start)*1000,
                internal_seam_needed=bool(best.get('internal_seams')))


def solve_steps(settings):
    """Cooperative network1 adapter: one exact count system, then all panels."""
    from .network_validate import validate_network_steps
    settings = dict(settings)
    begin = geom.clock().perf_counter()
    network = settings['_network_data']   # fill_multi.plan_steps (the only caller) passes the network it built
    yield None
    model = panel_network.count_model(network)
    counts = yield from network_counts.solve_steps(model['variables'], model['equations'], model['bounds'], model['targets'])
    results = {}
    if counts['status'] == 'feasible' and not network['conflicts']:
        for p in network['panels']:
            boundary = np.asarray([network['nodes'][i]['point'] for i in p['global_ids']])   # mode: always 'coincident'
            incumbent = yield from legacy_fill_steps(boundary, settings)
            results[p['id']] = yield from panel_steps(boundary, settings, incumbent)
            yield None
    validation = yield from validate_network_steps(network, results)
    return dict(status='filled' if validation['valid'] else 'conflict', panels=results,
                counts=counts, validation=validation, network=network,
                elapsed_ms=(geom.clock().perf_counter()-begin)*1000)
