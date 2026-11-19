# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Poles-first selection (the poles_first policy). The main search's AUTO never enters this module's jobs."""
from itertools import combinations
import math
import numpy as np
from .patterns import chords, geom
from . import memo_jobs

SELECTION_POLICY = 'poles_first'
QUALITY_FLOOR = .25
QUALITY_FRACTION = .7
PROMOTION_BAND = (140., 160.)
LEGACY_SECONDS = .2      # the main search runs first as the incumbent
LEGACY_RESERVE = .4      # CPU kept back to finish it when the policy finds nothing


def enabled(settings):
    return settings.get('_policy', SELECTION_POLICY) == 'poles_first'


def recovery_enabled(settings):
    """the poles-first selection or the Takayama pattern set is on for these settings."""
    return enabled(settings) or chords.takayama_enabled(settings)


def cache_key(settings):
    if not enabled(settings):
        return ('legacy',)
    return ('poles_first', float(settings.get('_quality_floor', QUALITY_FLOOR)),
            float(settings.get('_quality_fraction', QUALITY_FRACTION)),
            tuple(settings.get('_promotion_band', PROMOTION_BAND)))


def child_settings(settings):
    return dict(_policy='poles_first',
                _quality_floor=settings.get('_quality_floor', QUALITY_FLOOR),
                _quality_fraction=settings.get('_quality_fraction', QUALITY_FRACTION),
                _promotion_band=settings.get('_promotion_band', PROMOTION_BAND))


def topology_poles(quads, total, n):
    # In a validated quad disk, each interior vertex has one face per edge.
    degree=np.bincount(np.asarray(quads,dtype=np.int64).ravel(),minlength=total)
    return int(np.count_nonzero(degree[n:] != 4))


def metrics(result):
    quality=geom.quad_quality(np.asarray(result['positions'],float),result['quads'])
    edges={tuple(sorted((a,b))) for q in result['quads'] for a,b in zip(q,q[1:]+q[:1])}
    from collections import Counter
    incidence=Counter(tuple(sorted((a,b))) for q in result['quads'] for a,b in zip(q,q[1:]+q[:1]))
    boundary={v for e,c in incidence.items() if c==1 for v in e}
    degree=Counter(v for e in edges for v in e)
    return (sum(d!=4 for v,d in degree.items() if v not in boundary),
            float(quality.min()),float(quality.mean()),len(result['quads']))


@memo_jobs.recorded_checkpoint
def checkpoint(settings, result):
    """Only complete, validated placements enter this solve-local registry.

    Partition children do not inherit it: their coordinates are not the root cage.
    It is neither cached nor part of cache identity.
    """
    if not recovery_enabled(settings):
        return
    pool = settings.get('_validated_pool')
    if pool is not None and not any(r is result for r in pool):
        pool.append(result)


def best_checkpoint(settings):
    if not recovery_enabled(settings):
        return None
    pool = settings.get('_validated_pool', [])
    if enabled(settings):
        return choose(pool, settings) or settings.get('_incumbent')
    if not pool:
        return None
    from .boundary_preview import _layout_score
    from collections import Counter
    edges = Counter(tuple(sorted(e)) for q in pool[0]['quads']
                    for e in zip(q, q[1:]+q[:1]))
    n = len({v for e, count in edges.items() if count == 1 for v in e})
    return max(enumerate(pool), key=lambda item:
               (round(_layout_score(item[1], n), 9), -item[0]))[1]


def choose(pool,settings):
    if not pool:
        return None
    values=[metrics(r) for r in pool]
    floor=float(settings.get('_quality_floor',QUALITY_FLOOR))
    ratio=float(settings.get('_quality_fraction',QUALITY_FRACTION))
    eligible=[i for i,v in enumerate(values) if v[1]+1e-12>=floor]
    # Maximise minQ * ratio**poles. Removing d poles permits at most the
    # multiplicative loss ratio**d; at equal poles the higher minQ wins.
    # Padding breaks utility ties only, so it cannot bypass the exchange bound.
    def key(i):
        poles,qmin,qmean,faces=values[i]
        utility=math.log(max(qmin,1e-300))+poles*math.log(ratio)
        return (-round(utility,12),poles,-pool[i].get('_padding',0),
                -round(qmin,9),-round(qmean,9),faces,i)
    return pool[min(eligible,key=key)] if eligible else None


def policy_corners(unit,refs,settings):
    from .boundary_preview import corner_sets
    seen=set()
    if not settings.get('anchors') and settings.get('_corners') is None:
        turns=geom.turning(unit)
        lo,hi=settings.get('_promotion_band',PROMOTION_BAND)
        interior=180.-np.degrees(turns)
        sharp=[i for i,t in enumerate(turns) if t>.6]
        soft=[i for i,a in enumerate(interior) if lo<=a<=hi and i not in sharp]
        limit=6 if chords.takayama_enabled(settings) else 5
        promoted=[]
        for extra in range(1,min(len(soft),limit)+1):
            for group in combinations(soft,extra):
                geom.check_time()
                cs=tuple(sorted(sharp+list(group)))
                if len(cs)<=limit:
                    counts=[(cs[(i+1)%len(cs)]-cs[i])%len(unit) for i in range(len(cs))]
                    grid=len(cs)==4 and counts[0]==counts[2] and counts[1]==counts[3]
                    promoted.append((not grid,cs))
                yield None
        for _,cs in sorted(promoted):
            if cs not in seen:
                seen.add(cs)
                yield list(cs),0.
    for candidate in corner_sets(unit,refs,settings):
        if candidate is None:
            yield None
        elif tuple(candidate[0]) not in seen:
            seen.add(tuple(candidate[0]))
            yield candidate


def legacy_steps(xy,refs,settings,turns):
    """Start the main search and run it for a short CPU slice.

    Returns (result, job): job is the paused search when it did not finish in its
    slice, so it can resume if the policy search finds nothing. It is paused only
    between steps (no nested deadline), so a resumed search is the same search.
    """
    from . import fill, fill_partition, boundary_preview
    legacy={k:v for k,v in settings.items() if k not in ('_validated_pool','_incumbent')}
    legacy.update(_policy='legacy',_takayama=False)
    cap=float(settings.get('_legacy_seconds',LEGACY_SECONDS))
    job=fill._planar_steps(xy,refs,legacy,turns)
    spent=0.
    while spent<cap:
        started=geom.clock().monotonic()
        try:
            next(job)
        except StopIteration as done:
            return done.value,None
        except (geom.TimeUp,boundary_preview.PreviewError,fill_partition.PartitionError):
            return None,None
        spent+=geom.clock().monotonic()-started
        yield None
    return None,job


def _policy_steps(xy,refs,settings,turns,pool):
    from . import fill, fill_lattice, boundary_preview
    template={k:v for k,v in settings.items() if k!='_face_turns'}
    try:
        if fill_lattice.eligible(settings):
            grid=yield from fill_lattice.plan_steps(xy,turns)
            if grid is not None:
                pool.append(grid)
                checkpoint(settings,grid)
            grid,spare=yield from fill._balanced_steps(xy,refs,template)
            if grid is not None or spare is not None:
                pool.append(grid if grid is not None else spare)
            template['_best_of']=fill.BEST_OF
        result=yield from fill._template_steps(xy,refs,settings,template)
        pool.extend(result.get('_policy_pool',[result]))
    except (geom.TimeUp,boundary_preview.PreviewError):
        pass


def planar_steps(xy,refs,settings,turns=None):
    from . import fill_partition, boundary_preview
    pool=[]
    incumbent,legacy=yield from legacy_steps(xy,refs,settings,turns)
    try:
        geom.check_time()
        if incumbent is not None:
            if settings.get('_legacy_mark'):
                incumbent=dict(incumbent,_legacy_seed=True)
            settings['_incumbent']=incumbent
            pool.append(incumbent)
            checkpoint(settings,incumbent)
        policy=_policy_steps(xy,refs,settings,turns,pool)
        if legacy is None:
            yield from policy
        else:
            # Leave the unfinished main search time to resume when the policy
            # search ends with nothing usable.
            cap=geom.time_left()-float(LEGACY_RESERVE)
            spent=0.
            try:
                while spent<cap:
                    started=geom.clock().monotonic()
                    try:
                        with geom.time_budget(min(cap-spent,geom.time_left())):
                            next(policy)
                    except StopIteration:
                        break
                    spent+=geom.clock().monotonic()-started
                    yield None
            finally:
                policy.close()
        # Include previews from capped quick searches and balanced spares that
        # were validated before a later candidate exhausted the budget.
        pool.extend(r for r in settings.get('_validated_pool',[])
                    if not any(r is existing for existing in pool))
        if legacy is not None and choose(pool,settings) is None:
            try:
                incumbent=yield from legacy
            except (boundary_preview.PreviewError,fill_partition.PartitionError):
                incumbent=None
            if incumbent is not None:
                if settings.get('_legacy_mark'):
                    incumbent=dict(incumbent,_legacy_seed=True)
                settings['_incumbent']=incumbent
                pool.append(incumbent)
    finally:
        if legacy is not None:
            legacy.close()
    # Below the policy floor everywhere: keep the main search's result.
    chosen=choose(pool,settings) or incumbent
    if chosen is None:
        raise boundary_preview.PreviewError('No valid pattern above policy quality floor')
    return chosen


def job(points,refs,settings):

    from .boundary_preview import (signature, _cache, PreviewError, MAX_BOUNDARY,
        MAX_QUADS, POOL_RESERVE, canonical_order, place_steps, remember)
    from .patterns import solver
    from .patterns.validate import validate_steps, InvalidFill

    """Plan deterministically, then retain exact connectivity while geometry changes."""
    key=signature(points,settings,refs)
    if key in _cache:
        _cache.move_to_end(key)
        checkpoint(settings,_cache[key])
        return _cache[key]
    boundary=np.asarray(points,dtype=np.float64)
    n=len(boundary)
    if n%2:
        raise PreviewError('Odd count')
    if n<4 or n>MAX_BOUNDARY:
        raise PreviewError('Preview boundary limit (256 points)')
    origin=boundary.mean(axis=0)
    scale=float(np.ptp(boundary,axis=0).max())
    if not np.isfinite(scale) or scale<=1e-15:
        raise PreviewError('Invalid boundary')
    unit=(boundary-origin)/scale
    if np.any(np.linalg.norm(np.roll(unit,-1,axis=0)-unit,axis=1)<1e-10):
        raise PreviewError('Invalid boundary')
    canonical=not settings.get('anchors')
    library=list(chords.library(chords.takayama_enabled(settings)))
    best_of=max(4,int(settings.get('_best_of',4)))
    pool=[]
    try:
        for candidate in policy_corners(unit,refs,settings):
            if candidate is None:
                yield None
                continue
            cs,_demoted=candidate
            count=len(cs)
            if canonical:
                _,order,mapped,direction=canonical_order(unit,cs)
                ordered=unit[order]
                target=mapped[1] if len(mapped)>1 else max(1,n//4)
                axis=ordered[target]-ordered[0]
                axis=axis/max(float(np.linalg.norm(axis)),1e-15)
                other=np.array([-axis[1],axis[0]])*direction
                frame=np.column_stack((axis,other))
                local=ordered@frame
            else:
                order=list(range(n));mapped=cs;direction=1
                frame=np.eye(2);local=unit
            # One logical corner wraps back to itself through the entire boundary.
            sides=[[(mapped[s]+i)%n for i in range(((mapped[(s+1)%count]-mapped[s])%n or n)+1)] for s in range(count)] if count else [list(range(n))+[0]]
            counts=[len(side)-1 for side in sides]
            candidates=[]
            for p in library:
                if count==6 and not getattr(p,'takayama',False):
                    continue
                if len(p.loops)!=1 or len(p.loops[0])!=max(count,1) or (count==0)!=(p.loops[0][0][0] not in p.corner):
                    continue
                for rot in range(max(count,1)):
                    shifted = counts[rot:]+counts[:rot]
                    if getattr(p, 'takayama', False):
                        variants = yield from p.policy_parameters_steps(shifted)
                    else:
                        x = yield from solver.solve_steps(p, shifted)
                        variants = [] if x is None else [x]
                    for x in variants:
                        # Exact interior pole count and face count are cheap topology
                        # quantities; place the cheapest pole-free candidates first.
                        geom.check_time()
                        qs, total = chords.refine(p,x,sides[rot:]+sides[:rot],[],n)
                        if len(qs) <= MAX_QUADS:
                            poles = topology_poles(qs, total, n)
                            padding = sum(x[:p.n]) if getattr(p,'takayama',False) else 0
                            candidates.append((poles, -padding, p.name, rot, p, x, qs, total))
                        yield None
                    yield None
            # Geometry positions may change, but must not silently decide among
            # equally suitable topologies. Validate candidates in a stable order.
            candidates.sort(key=lambda c:(c[0],c[1],len(c[6]),c[2],c[3],c[5]))
            seen=set()
            from_this_set=0
            for irregular,_balance,_name,rot,p,x,quads,total in candidates[:12]:
                if len(quads)>MAX_QUADS:
                    continue
                topology=tuple(sorted(tuple(sorted(q)) for q in quads))
                if topology in seen:
                    continue
                seen.add(topology)
                try:
                    placed=yield from place_steps(local,total,quads)
                except geom.TimeUp:
                    if not pool:
                        raise
                    break                           # keep the layouts already found
                if placed is None:
                    continue
                positions,metrics,score=placed
                positions=positions@frame.T
                positions[:n]=unit
                remap=lambda i:order[i] if i<n else i
                quads=[tuple(remap(i) for i in (q if direction==1 else q[::-1])) for q in quads]
                # Interior indices stay fixed; boundary vertices return to the original
                # graph walk so adjacent patches still share exact sample identities.
                if canonical:
                    converted=np.empty_like(positions)
                    converted[:n]=unit
                    converted[n:]=positions[n:]
                    positions=converted
                    try:
                        metrics=yield from validate_steps(unit,positions,quads)
                    except InvalidFill:
                        continue
                name=p.name
                if direction==-1:
                    name=name[:-1] if name.endswith('~') else name+'~'
                actual_rotation=cs.index(order[mapped[rot]]) if count else 0
                positions=positions*scale+origin
                positions[:n]=boundary
                result={'positions':positions,'quads':tuple(quads),'pattern':name,
                        'rotation':actual_rotation,'corners':tuple(cs),'anchors':[refs[i] for i in cs],
                        'metrics':metrics,'score':score,
                        '_family': (tuple(cs),p.n,p.index) if getattr(p,'takayama',False) else None,
                        '_padding':sum(x[:p.n]) if getattr(p,'takayama',False) else 0}
                checkpoint(settings,result)
                pool.append(result)
                from_this_set+=1
                # at most two per corner set, so different corner families get compared
                if len(pool)>=best_of or from_this_set>=2 or geom.time_left()<POOL_RESERVE:
                    break
            if pool and (len(pool)>=best_of or geom.time_left()<POOL_RESERVE):
                break
    except geom.TimeUp:
        if not pool:
            raise
    if pool:
        # Fill AUTO: the geometrically best of the first valid layouts, not just the
        # first one (deterministic: ties keep the earlier candidate).
        result=choose(pool,settings)
        if result is None:
            raise PreviewError('No valid pattern above policy quality floor')
        result=dict(result,_policy_pool=pool)
        remember(key,result)
        return result
    raise PreviewError('No valid pattern for this point plan')
