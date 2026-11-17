# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Bounded AUTO candidate search: trace a few cross-field cuts, fill both children.

Only existing outline vertices anchor cuts. No boundary resampling or outside
code is used. Every merged proposal uses the old validator.
"""
import numpy as np
from . import fill, network_medial as medial, network_counts, network_fill
from .patterns import geom
from .patterns.validate import validate_steps, InvalidFill
from .fill_guidance import _visible


MAX_WORK_TICKS = 1000
MAX_STEPS = 2600


def search_steps(world, settings, incumbent=None):
    """After the network plan: deterministic bounded search. No work precedes first preview.

    High-quality pole-free defaults, odd rims, nonflat outlines and manual modes
    keep their released paths. Limits count work, never wall time, so CPU load
    cannot decide which topology is offered.
    """
    from . import fill_progressive as progressive
    from .fill_guidance import Guide
    if incumbent is None:
        return
    if (not progressive.eligible(settings) or len(world) % 2 or len(world) < 6
            or len(world) > 128 or progressive.frozen_quality_conflict(world)):
        return
    if (incumbent is not None and fill._interior_poles(incumbent) == 0
            and progressive.utility(incumbent) >= .5):
        return
    yield None
    guide = Guide(world)
    yield None
    if not guide.available:
        return
    job = progressive.deterministic_steps(candidate_steps(world, guide),
        preserve_updates=True, max_ticks=MAX_WORK_TICKS)
    try:
        for _ in range(MAX_STEPS):
            try:
                update = next(job)
            except StopIteration:
                break
            yield update
    finally:
        job.close()


def _cross(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def cuts_steps(guide):
    """Euler traces with continuous cross-arm choice, then frozen-rim snapping."""
    p=guide.rim; n=len(p); h=guide.spacing
    turn=geom.turning(p)
    orientation=1 if float(_cross(p,np.roll(p,-1,axis=0)).sum())>0 else -1
    turn=turn*orientation
    # Reflex vertices first; six strongest remaining corners bound the work.
    corners=sorted((i for i in range(n) if abs(turn[i])>.15),key=lambda i:(turn[i]>=0,-abs(turn[i]),i))[:8]
    step=min(h*.25,float(np.ptp(p,axis=0).max())/16)
    found={}
    for source in corners:
        yield None
        incoming=p[source]-p[source-1];outgoing=p[(source+1)%n]-p[source]
        inward=orientation*np.array([-incoming[1],incoming[0]])/max(np.linalg.norm(incoming),1e-30)
        inward+=orientation*np.array([-outgoing[1],outgoing[0]])/max(np.linalg.norm(outgoing),1e-30)
        inward/=max(np.linalg.norm(inward),1e-30)
        probe=p[source]+inward*step*.3
        z=guide.sample(np.array([probe]))[0]
        if abs(z)<.15:continue
        angles=np.angle(z)/4+np.arange(4)*np.pi/2
        arms=np.column_stack((np.cos(angles),np.sin(angles)))
        for direction in arms:
            yield None
            current=p[source]+direction*step*.2
            if not _visible(p[source][None],current[None],p)[0]:continue
            path=[p[source],current];confidence=[]
            for k in range(96):
                value=guide.sample(current[None])[0]
                confidence.append(float(abs(value)))
                if abs(value)>=.15:
                    theta=np.angle(value)/4+np.arange(4)*np.pi/2
                    options=np.column_stack((np.cos(theta),np.sin(theta)))
                    direction=options[np.argmax(options@direction)]
                nxt=current+step*direction
                if not _visible(current[None],nxt[None],p)[0]:
                    edge=np.roll(p,-1,axis=0)-p
                    den=_cross(nxt-current,edge)
                    valid=np.flatnonzero(abs(den)>1e-14)
                    t=_cross(p[valid]-current,edge[valid])/den[valid]
                    u=_cross(p[valid]-current,nxt-current)/den[valid]
                    hits=[(float(v),int(i)) for v,w,i in zip(t,u,valid) if 0<=v<=1 and -1e-9<=w<=1+1e-9]
                    if hits:
                        _,edge_id=min(hits)
                        for target in (edge_id,(edge_id+1)%n):
                            a,b=sorted((source,target))
                            if min(b-a,n-(b-a))<2:continue
                            if not _visible(current[None],p[target][None],p)[0]:continue
                            traced=np.asarray(path+[p[target]])
                            if source!=a:traced=traced[::-1]
                            length=float(np.linalg.norm(np.diff(traced,axis=0),axis=1).sum())
                            score=length/h+.5*(1-np.mean(confidence))
                            entry=dict(seam=(a,b),path=traced,length=length,score=score,source=source,
                                       confidence_min=min(confidence),pole_seed=traced[1:-1][int(np.argmin(confidence[:len(traced)-2]))].tolist())
                            if (a,b) not in found or score<found[a,b]['score']:found[a,b]=entry
                    break
                path.append(nxt);current=nxt
                yield None
            yield None
    # One cut at a time keeps the candidate set small.
    return sorted(found.values(),key=lambda c:(round(c['score'],9),c['seam']))[:6]


def _sample_path(path, count):
    length=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    t=np.linspace(0,length[-1],count+1)
    return np.column_stack([np.interp(t,length,path[:,k]) for k in range(2)])


def candidate_steps(world, guide):
    """Reuse integer parity assignment, coupled lattice and network templates."""
    if not guide.available or len(world)%2 or len(world)<6:return
    from .fill_progressive import _network_plan
    cuts=yield from cuts_steps(guide)
    p=guide.rim;n=len(p)
    origin=p.mean(0);scale=max(float(np.ptp(p,axis=0).max()),1e-30)
    unit=(p-origin)/scale
    # Existing lattice children require counterclockwise input: the root indices stay, each child ring is
    # reversed below for a clockwise rim.
    orientation=1 if float(_cross(p,np.roll(p,-1,axis=0)).sum())>0 else -1
    for ci,cut in enumerate(cuts):
        a,b=cut['seam']
        polys=[list(range(a,b+1)),list(range(b,n))+list(range(a+1))]
        target=int(np.clip(np.floor(cut['length']/guide.spacing+.5),1,24))
        seen=set()
        for offset in (0,2):
            wanted=max(1,min(24,target+offset))
            certificate=yield from network_counts.solve_steps(['half0','half1','seam'],
                [dict(coefficients={f'half{i}':-2,'seam':1},rhs=-(len(poly)-1),label=f'child{i}') for i,poly in enumerate(polys)],
                dict(half0=(2,n+24),half1=(2,n+24),seam=(max(1,wanted-1),min(24,wanted+1))),
                dict(half0=(len(polys[0])+wanted)//2,half1=(len(polys[1])+wanted)//2,seam=wanted),seconds=.02)
            yield None
            if certificate['status']!='feasible':continue
            count=certificate['values']['seam']
            if count in seen:continue
            seen.add(count)
            path=(_sample_path(cut['path'],count)-origin)/scale
            parent=np.vstack((unit,path[1:-1]));seam_ids=list(range(n,len(parent)))
            rings=[polys[0]+seam_ids[::-1],polys[1]+seam_ids]
            if orientation<0:rings=[ring[::-1] for ring in rings]
            positions=list(parent);quads=[];failed=False
            for ring in rings:
                local=parent[ring]
                result=None
                try:
                    with geom.time_budget(.12):
                        result=yield from medial.coupled_steps(local,dict(_quality_floor=.25,lattice_beam=600,lattice_paths_per_state=4))
                except geom.TimeUp:pass
                if result is None:
                    from . import boundary_preview
                    try:
                        with geom.time_budget(.06):
                            result=yield from boundary_preview.job(local,[('p',j) for j in range(len(local))],dict(pattern='AUTO',_policy='poles_first',_takayama=True),site=('fill_field.candidate_steps',))
                    except (geom.TimeUp,boundary_preview.PreviewError):pass
                if result is None:failed=True;break
                mapping=ring+list(range(len(positions),len(positions)+len(result['positions'])-len(ring)))
                positions.extend(result['positions'][len(ring):])
                quads.extend(tuple(mapping[v] for v in q) for q in result['quads'])
            if failed:continue
            positions=np.asarray(positions)
            try:
                with geom.time_budget(.15):
                    positions=yield from geom.winslow_steps(positions,n,quads)
                    yield from validate_steps(unit,positions,quads)
            except (geom.TimeUp,InvalidFill):continue
            metrics=network_fill.metrics(positions,quads)
            if metrics['qmin']<.25:continue
            plane=positions*scale+origin;plane[:n]=p
            result=dict(positions=plane,quads=tuple(quads),method='field_trace',**metrics)
            plan=_network_plan(np.asarray(world),guide.centre,guide.u,guide.v,np.cross(guide.u,guide.v),list(range(n)),result)
            plan['_field_candidate']=True
            plan['field_trace']=dict(cut=ci+1,seam=cut['seam'],count=count,target=target,spacing=guide.spacing,
                length=cut['length'],pole_seed=cut['pole_seed'],path=cut['path'].tolist(),count_certificate=certificate)
            fill.validate_plan(plan)
            yield {'candidate':plan}
