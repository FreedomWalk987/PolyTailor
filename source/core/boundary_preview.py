# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Cooperative pattern-only preview jobs. Never touches Blender mesh data."""
from collections import OrderedDict
from itertools import combinations
import numpy as np
from .patterns import chords,geom,solver
from . import fill_selection, memo_jobs
from .patterns.validate import validate_steps,InvalidFill

PATTERNS = (('AUTO','Auto Pattern'),('quad','Four-sided Grid'),('triangle','Triangle Pattern'),
            ('pentagon','Pentagon Pattern'),('quad_diag35','Diagonal Transition'),
            ('quad_aligned35','Aligned Transition'),('quad_uturn','U-turn Transition'),
            ('two_corners','Two-corner Pattern'),('one_corner','One-corner Pattern'),('disk','Disk Pattern'))
MAX_BOUNDARY = 256
MAX_QUADS = 1024
POOL_RESERVE = .2      # seconds left at which Fill AUTO stops comparing more layouts
POLE_COST = .1         # layout score lost per interior pole (Fill AUTO comparison)
GOOD_ENOUGH = .6       # worst corner sine at which Fill AUTO keeps its first layout
_cache = OrderedDict()
_CURRENT = [_cache]    # [the phase cache in use]: _cache, or a deterministic step's own (push_cache)


def current_cache():
    """The phase cache in use."""
    return _CURRENT[0]


def push_cache(cache):
    """Make `cache` the phase cache until pop_cache(token); returns the token."""
    old = _CURRENT[0]
    _CURRENT[0] = cache
    return old


def pop_cache(token):
    """Put back the phase cache that push_cache returned."""
    _CURRENT[0] = token


def clear_caches():
    """Empty the phase cache (add-on unregister)."""
    _CURRENT[0].clear()
    _cache.clear()


class PreviewError(ValueError):
    pass


def resolve_refs(refs, anchors):
    indices=[]
    for anchor in anchors:
        matches=[(i,abs(ref[2]-anchor[2]) if ref[0]=='span' else 0)
                 for i,ref in enumerate(refs) if ref[:2]==anchor[:2]]
        if not matches:
            return None
        indices.append(min(matches,key=lambda p:p[1])[0])
    return sorted(set(indices)) if len(set(indices))==len(anchors) else None


def signature(points,settings,refs):
    return (tuple(map(tuple,points)),settings.get('pattern','AUTO'),settings.get('rotation',-1),
            settings.get('mirror','AUTO'),repr(settings.get('anchors',[])),tuple(map(tuple,refs)),
            repr(settings.get('_corners')),repr(settings.get('_layout')),int(settings.get('_best_of',1)),
            chords.takayama_enabled(settings), fill_selection.cache_key(settings))


@memo_jobs.recorded_remember
def remember(key,result):
    cache=_CURRENT[0]
    cache[key]=result
    cache.move_to_end(key)
    while len(cache)>64:
        cache.popitem(last=False)


def canonical_order(points,corners):
    """Match cyclic/mirrored side counts before using invariant geometric tie-breaks."""
    n=len(points)
    lengths=np.linalg.norm(np.roll(points,-1,axis=0)-points,axis=1)
    total=max(float(lengths.sum()),1e-15)
    cs=list(corners)
    starts=cs or list(range(n))
    choices=[]
    for start in starts:
        for direction in (1,-1):
            order=[(start+direction*i)%n for i in range(n)]
            inverse={old:new for new,old in enumerate(order)}
            mapped=sorted(inverse[c] for c in cs)
            counts=tuple((b-a)%n for a,b in zip(mapped,mapped[1:]+mapped[:1])) if mapped else (n,)
            edge_lengths=[lengths[i] if direction==1 else lengths[(i-1)%n] for i in order]
            side_lengths=tuple(round(sum(edge_lengths[a:a+count])/total,6) for a,count in zip(mapped,counts)) if mapped else (1.0,)
            diagonals=tuple(round(float(np.linalg.norm(points[order[mapped[i]]]-points[order[mapped[(i+2)%len(mapped)]]]))/total,6)
                            for i in range(len(mapped)))
            descriptor=(counts,side_lengths,diagonals,tuple(round(float(v)/total,6) for v in edge_lengths))
            choices.append((descriptor,order,mapped,direction))
    return min(choices,key=lambda item:item[0])


def corner_sets(unit,refs,settings):
    """Yield candidate corner plans lazily, with cooperative sorting batches."""
    max_sides=6 if chords.takayama_enabled(settings) else 5
    anchors=settings.get('anchors',[])
    if anchors:
        locked=resolve_refs(refs,anchors)
        if locked is None:
            raise PreviewError('Fill corners no longer available')
        yield locked,0
        return
    planned=settings.get('_corners')
    seen=set()
    if planned is not None:
        indices=resolve_refs(refs,planned)
        if indices is None:
            raise PreviewError('Fill corners no longer available')
        if len(indices)<=8:
            for keep in range(min(max_sides,len(indices)),-1,-1):
                if keep==len(indices):
                    group=[(None,list(indices))]
                else:
                    group=[]
                    for cs in combinations(indices,keep):
                        group.append((canonical_order(unit,cs)[0],list(cs)))
                        yield None
                    group.sort(key=lambda item:item[0])
                for _,cs in group[:16]:
                    seen.add(tuple(cs))
                    yield cs,len(indices)-keep
    for cs,demoted in chords.corner_candidates(geom.turning(unit),max_sides=max_sides):
        if tuple(cs) not in seen:
            seen.add(tuple(cs))
            yield cs,demoted


def place_steps(unit,total,quads,initial=None):
    positions=geom.laplace(unit,total,quads,iterations=240) if initial is None else initial.copy()
    positions[:len(unit)]=unit
    yield None
    initial=geom.quad_quality(positions,quads)
    if initial.min()>=.35 and initial.mean()>=.85:
        ok=True
    else:
        positions,ok=yield from geom.untangle_steps(positions,len(unit),quads)
    yield None
    if not ok:
        return None
    quality=geom.quad_quality(positions,quads)
    if quality.min()<chords.MIN_QUALITY or quality.mean()<chords.MEAN_QUALITY:
        return None
    try:
        metrics=yield from validate_steps(unit,positions,quads)
    except InvalidFill:
        return None
    score=float(quality.mean())-.025*np.log(max(1,metrics['aspect_p95']))
    return positions,metrics,score


def _layout_score(result,n):
    """Quality of a placed layout: mean and worst corner shape, fewer interior poles."""
    positions=np.asarray(result['positions'],dtype=np.float64)
    quads=result['quads']
    quality=geom.quad_quality(positions,quads)
    valence=np.bincount(np.asarray([v for q in quads for v in q],dtype=np.int64),minlength=len(positions))
    poles=int(np.count_nonzero(valence[n:]!=4))
    return float(result['score'])+.5*float(quality.min())-POLE_COST*poles


def _good_enough(result):
    """The first valid layout needs no comparison when its worst corner is fine."""
    quality=geom.quad_quality(np.asarray(result['positions'],dtype=np.float64),result['quads'])
    return float(quality.min())>=GOOD_ENOUGH


# memoised (memo_jobs): job(points, refs, settings, site=None); callers name the loop that advances the search
@memo_jobs.memoised_search
def job(points,refs,settings):
    """Plan deterministically, then retain exact connectivity while geometry changes."""
    if fill_selection.enabled(settings):
        return (yield from fill_selection.job(points,refs,settings))
    key=signature(points,settings,refs)
    cache=_CURRENT[0]
    if key in cache:
        cache.move_to_end(key)
        fill_selection.checkpoint(settings,cache[key])
        return _CURRENT[0][key]
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
    best_of=int(settings.get('_best_of',1))
    pool=[]
    try:
        for candidate in corner_sets(unit,refs,settings):
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
                    x=yield from solver.solve_steps(p,counts[rot:]+counts[:rot])
                    if x is not None:
                        candidates.append((p.irregular,round(-float(sum(np.sqrt(x))),10),p.name,rot,p,x))
                    yield None
            # Geometry positions may change, but must not silently decide among
            # equally suitable topologies. Validate candidates in a stable order.
            candidates.sort(key=lambda c:c[:4])
            seen=set()
            from_this_set=0
            for irregular,_balance,_name,rot,p,x in candidates[:12]:
                quads,total=chords.refine(p,x,sides[rot:]+sides[:rot],[],n)
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
                        'metrics':metrics,'score':score}
                fill_selection.checkpoint(settings,result)
                if best_of<=1 or (not pool and _good_enough(result)):
                    remember(key,result)
                    return result
                pool.append(result)
                from_this_set+=1
                # at most two per corner set, so different corner families get compared
                if len(pool)>=best_of or from_this_set>=2 or geom.time_left()<POOL_RESERVE:
                    break
            if pool and (len(pool)>=best_of or geom.time_left()<POOL_RESERVE):
                break
    except geom.TimeUp:
        if not pool or not fill_selection.recovery_enabled(settings):
            raise
    if pool:
        # Fill AUTO: the geometrically best of the first valid layouts, not just the
        # first one (deterministic: ties keep the earlier candidate).
        result=max(enumerate(pool),key=lambda item:(round(_layout_score(item[1],n),9),-item[0]))[1]
        fill_selection.checkpoint(settings,result)
        remember(key,result)
        return result
    raise PreviewError('No valid pattern for this point plan')
