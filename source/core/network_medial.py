# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Boundary-sampled Voronoi blocking (numpy only).

The sampled Voronoi skeleton approximates the boundary-segment medial axis.
It proposes layouts; mesh seams end at existing exterior samples.
Coupled lattice proposals allow turns along shared seams. Rectangular fallback
pieces use exact global integer equations; one widest piece may use the existing
template engine. No reference geometry participates here.
"""
import math
import numpy as np
from .patterns import geom

MAX_CUTS = 40


def _cross(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def _inside_many(p, q):
    a=p[None,:,:];b=np.roll(p,-1,axis=0)[None,:,:]
    y=q[:,None,1];x=q[:,None,0];dy=b[:,:,1]-a[:,:,1]
    hits=(a[:,:,1]>y)!=(b[:,:,1]>y)
    cut=a[:,:,0]+(y-a[:,:,1])*(b[:,:,0]-a[:,:,0])/np.where(abs(dy)>1e-30,dy,1.)
    return np.count_nonzero(hits&(x<cut),axis=1)%2==1


def _visible(p, i, j):
    n = len(p)
    if (j-i) % n < 2 or (i-j) % n < 2:
        return False
    a, b = p[i], p[j]
    ids = np.arange(n)
    mask = (ids != i) & (ids != j) & ((ids+1) % n != i) & ((ids+1) % n != j)
    c, d = p[mask], np.roll(p, -1, axis=0)[mask]
    boxes = np.all(np.maximum(np.minimum(a,b),np.minimum(c,d)) <=
                   np.minimum(np.maximum(a,b),np.maximum(c,d))+1e-12,axis=1)
    c, d = c[boxes], d[boxes]
    if len(c):
        x, y = _cross(b-a,c-a), _cross(b-a,d-a)
        z, w = _cross(d-c,a-c), _cross(d-c,b-c)
        if np.any((np.minimum(x,y)<=1e-12)&(np.maximum(x,y)>=-1e-12)&
                  (np.minimum(z,w)<=1e-12)&(np.maximum(z,w)>=-1e-12)):
            return False
    return bool(_inside_many(p,a+np.array([.02,.25,.5,.75,.98])[:,None]*(b-a)).all())


def skeleton_steps(p):
    """Dual of a dense boundary-sampled Delaunay triangulation (numpy only).

    Bowyer-Watson uses deterministic site order and no coordinate perturbation.
    Voronoi nodes are true circumcentres, never raster cell centres. Sampling
    approximates segment sites; pixel_size is only the sampling error scale.
    """
    delta=np.roll(p,-1,axis=0)-p
    lengths=np.linalg.norm(delta,axis=1)
    step=max(float(lengths.sum())/256,float(lengths.max())/96,1e-12)
    counts=np.maximum(1,np.ceil(lengths/step).astype(int))
    sites=np.array([a+d*k/count for a,d,count in zip(p,delta,counts) for k in range(count)])
    # A containing supertriangle is removed from the completed triangulation.
    center=sites.mean(0);span=max(float(np.ptp(sites,axis=0).max()),1e-12)
    pts=np.vstack((sites,center+span*np.array([[-32,-16],[32,-16],[0,32]])))
    n=len(sites);tri=np.array([[n,n+1,n+2]],dtype=int)
    def circles(t):
        a,b,c=pts[t[:,0]],pts[t[:,1]],pts[t[:,2]]
        u=b-a;v=c-a;det=2*_cross(u,v)
        good=abs(det)>1e-18
        t=t[good];a=a[good];u=u[good];v=v[good];det=det[good]
        uu=(u*u).sum(1);vv=(v*v).sum(1)
        q=a+np.column_stack((v[:,1]*uu-u[:,1]*vv,u[:,0]*vv-v[:,0]*uu))/det[:,None]
        return t,q,((q-a)**2).sum(1)
    tri,centers,radii=circles(tri)
    for i,site in enumerate(sites):
        bad=((centers-site)**2).sum(1)<=radii+1e-14
        if not bad.any():continue
        bt=tri[bad]
        edges=np.sort(np.concatenate((bt[:,[0,1]],bt[:,[1,2]],bt[:,[2,0]])),axis=1)
        edges,num=np.unique(edges,axis=0,return_counts=True);edges=edges[num==1]
        added=np.column_stack((edges,np.full(len(edges),i)))
        added,cc,rr=circles(added)
        tri=np.concatenate((tri[~bad],added));centers=np.concatenate((centers[~bad],cc));radii=np.concatenate((radii[~bad],rr))
        if i%8==0:geom.check_time();yield None
    valid=np.all(tri<n,axis=1)
    valid &= _inside_many(p,centers)
    # Discard the tiny teeth between sites on the same wall. Retain corner
    # branches when their radius is larger than the site approximation error.
    valid &= radii>step*step*.36
    ids=np.flatnonzero(valid);lookup={int(old):i for i,old in enumerate(ids)}
    edge_owner={};neighbors=[[] for _ in ids]
    for old in ids:
        t=tri[old]
        for edge in (t[[0,1]],t[[1,2]],t[[2,0]]):
            key=tuple(sorted(map(int,edge)))
            if key in edge_owner:
                j=edge_owner[key];a=lookup[int(old)];b=lookup[j]
                if np.linalg.norm(centers[old]-centers[j])>step*1e-7:
                    neighbors[a].append(b);neighbors[b].append(a)
            else:edge_owner[key]=int(old)
    # Compute width from boundary segments, not the sampled point radii.
    q=centers[ids];distance=np.full(len(q),np.inf);owner=np.zeros(len(q),int)
    for i,(a,d) in enumerate(zip(p,delta)):
        t=np.clip((q-a)@d/max(float(d@d),1e-30),0,1)
        ds=((q-a-t[:,None]*d)**2).sum(1);better=ds<distance
        distance[better]=ds[better];owner[better]=i
        if i%8==0:geom.check_time();yield None
    return dict(points=q,width=2*np.sqrt(distance),owner=owner,neighbors=neighbors,
                pixel_size=step,sampling_step=step,sites=len(sites),method='sampled_boundary_voronoi')


def cuts_steps(p, graph):
    """Cross sections at skeleton width minima and branch junctions."""
    from .fill_partition import _inside
    points, width, neighbors = graph['points'],graph['width'],graph['neighbors']
    stations = []
    h = max(float(np.median(np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1))),1e-12)
    def along(i,j):
        previous=i;current=j;distance=float(np.linalg.norm(points[j]-points[i]))
        while distance<max(h*1.5,graph['pixel_size']*4) and len(neighbors[current])==2:
            nxt=next(k for k in neighbors[current] if k!=previous)
            distance+=float(np.linalg.norm(points[nxt]-points[current]))
            previous,current=current,nxt
        return current
    for i, ns in enumerate(neighbors):
        if i % 8 == 0:
            yield None
        if len(ns)>=3:
            stations.append((0,i,'branch'))
        elif len(ns)==2:
            # A sampling-sized neighborhood suppresses one-pixel width noise.
            near = set(ns)
            for j in list(near):near.update(neighbors[j])
            if width[i]<=min(width[list(near)])+graph['pixel_size']*.12:
                stations.append((1,i,'neck'))
            else:
                a,b=(along(i,j) for j in ns)
                incoming=points[i]-points[a];outgoing=points[b]-points[i]
                cosine=float(incoming@outgoing)/max(float(np.linalg.norm(incoming)*np.linalg.norm(outgoing)),1e-30)
                if cosine<.85:stations.append((2,i,'bend'))
    if not stations and len(points):
        stations = [(2,i,'ridge') for i in range(0,len(points),max(1,len(points)//8))]
    ranked = {};chosen=[];visibility={}
    def visible(a,b):
        key=tuple(sorted((a,b)))
        if key not in visibility:visibility[key]=_visible(p,*key)
        return visibility[key]
    for station_index, station in enumerate(sorted(stations,key=lambda s:(s[0],round(float(width[s[1]]),8),s[1]))):
        if station_index % 16 == 0:
            yield None
        i=station[1]
        if all(np.linalg.norm(points[i]-points[s[1]])>max(h*.9,graph['pixel_size']*3) for s in chosen):
            chosen.append(station)
    for _,i,kind in chosen[:32]:
        center = points[i]; ns = neighbors[i]
        directions = []
        if len(ns)==2:
            tangent = points[along(i,ns[1])]-points[along(i,ns[0])]
            directions.append(np.array([-tangent[1],tangent[0]]))
        else:
            for j in ns:
                tangent = points[along(i,j)]-center
                directions.append(np.array([-tangent[1],tangent[0]]))
        for direction in directions:
            direction /= max(float(np.linalg.norm(direction)),1e-30)
            delta = np.roll(p,-1,axis=0)-p
            den = _cross(direction,delta)
            ids = np.flatnonzero(abs(den)>1e-12)
            t = _cross(p[ids]-center,delta[ids])/den[ids]
            u = _cross(p[ids]-center,direction)/den[ids]
            hits = sorted((float(ti),int(k)) for ti,ui,k in zip(t,u,ids) if -1e-10<=ui<=1+1e-10)
            neg = [hit for hit in hits if hit[0]<0]; pos = [hit for hit in hits if hit[0]>0]
            if not neg or not pos:continue
            lo,hi = neg[-1],pos[0]
            if not _inside(p,center):continue
            for a in (lo[1],(lo[1]+1)%len(p)):
                for b in (hi[1],(hi[1]+1)%len(p)):
                    a,b = sorted((a,b))
                    if not visible(a,b):continue
                    length = float(np.linalg.norm(p[b]-p[a]))
                    deviation = abs(length-(hi[0]-lo[0]))/h
                    # Local width, orientation, and endpoint rounding, no reflex rays.
                    score = length/h + deviation + (0 if kind=='branch' else .15)
                    entry = (round(score,9),(a,b),kind,float(width[i]))
                    if (a,b) not in ranked or entry<ranked[a,b]:ranked[a,b]=entry
        geom.check_time(); yield None
    delta=np.roll(p,-1,axis=0)-p
    for a in np.flatnonzero(geom.turning(p)<-.35):
        t=np.clip(((p[a]-p)*delta).sum(1)/np.maximum((delta*delta).sum(1),1e-30),0,1)
        feet=p+t[:,None]*delta
        order=np.argsort(np.linalg.norm(feet-p[a],axis=1),kind='stable')
        kept=0
        for edge in order:
            if min((int(edge)-int(a))%len(p),(int(a)-int(edge))%len(p))<3:continue
            for b in (int(edge),(int(edge)+1)%len(p)):
                aa,bb=sorted((int(a),b))
                if not visible(aa,bb):continue
                length=float(np.linalg.norm(p[aa]-p[bb]))
                d=p[bb]-p[aa];wall=delta[edge]
                skew=abs(float(d@wall))/max(float(np.linalg.norm(d)*np.linalg.norm(wall)),1e-30)
                entry=(round(length/h+skew,9),(aa,bb),'wall_normal',length)
                if (aa,bb) not in ranked or entry<ranked[aa,bb]:ranked[aa,bb]=entry
                kept+=1
            if kept>=4:break
        geom.check_time();yield None
    # Keep spatially distinct throats before adjacent endpoint alternatives.
    selected=[];deferred=[];mids=[]
    for item in sorted(ranked.values()):
        a,b=item[1];mid=(p[a]+p[b])/2
        nearby=np.count_nonzero(np.linalg.norm(np.asarray(mids)-mid,axis=1)<max(h*1.2,item[3]*.7)) if mids else 0
        if nearby<2:selected.append(item);mids.append(mid)
        else:deferred.append(item)
    return (selected+deferred)[:MAX_CUTS]


def _split(pieces, seam):
    a,b = seam
    for k,poly in enumerate(pieces):
        if a not in poly or b not in poly:continue
        i,j = sorted((poly.index(a),poly.index(b)))
        if j-i<2 or len(poly)-(j-i)<2:continue
        return pieces[:k]+(poly[i:j+1],poly[j:]+poly[:i+1])+pieces[k+1:]
    return None


def layout_steps(p,cuts):
    """Keep two-cut combinations before extending a small three-cut beam.

    A cut with a weak single-cut score may complete a useful neck pair. Test
    every pair before pruning, instead of requiring that first cut to survive
    the single-cut beam. The embedding order interleaves search depths.
    """
    whole=(tuple(range(len(p))),);singles=[];pairs=[];triples=[]
    base_reflex=float(-np.minimum(geom.turning(p),0).sum());changes={}
    def key(item):
        cost,seams,pieces=item
        ends=[v for seam in seams for v in seam]
        if len(set(ends))==len(ends):
            reflex=base_reflex+sum(changes[seam] for seam in seams)
        else:
            reflex=sum(float(-np.minimum(geom.turning(p[list(poly)]),0).sum()) for poly in pieces)
        return round(reflex,7),cost,seams
    for score,seam,_,_ in cuts:
        pieces=_split(whole,seam)
        if pieces is not None:
            singles.append((score,(seam,),pieces))
            changes[seam]=sum(float(-np.minimum(geom.turning(p[list(poly)]),0).sum()) for poly in pieces)-base_reflex
    def rank_steps(items):
        # Evaluating reflex scores can touch hundreds of polygons. Decorate in
        # small groups before sorting; the index preserves the stable old order.
        ranked=[]
        for index,item in enumerate(items):
            ranked.append((key(item),index,item))
            if index % 8 == 7:
                yield None
        ranked.sort(key=lambda item:(item[0],item[1]))
        items[:]=[item[2] for item in ranked]
    seen=set()
    for cost,seams,pieces in singles:
        for score,seam,_,_ in cuts:
            identity=tuple(sorted(seams+(seam,)))
            if seam in seams or identity in seen:continue
            split=_split(pieces,seam)
            if split is None:continue
            seen.add(identity);pairs.append((cost+score,identity,split))
        geom.check_time();yield None
    yield from rank_steps(singles)
    yield from rank_steps(pairs)
    seen=set()
    for cost,seams,pieces in pairs[:4]:
        for score,seam,_,_ in cuts:
            identity=tuple(sorted(seams+(seam,)))
            if seam in seams or identity in seen:continue
            split=_split(pieces,seam)
            if split is not None:
                seen.add(identity);triples.append((cost+score,identity,split))
        geom.check_time();yield None
    yield from rank_steps(triples)
    return pairs[:48]+singles[:8]+triples[:8]


def coupled_steps(p, settings):
    """Infer a closed integer grid: choose exterior quarter turns jointly, fill the closed simple lattice walk,
    and solve all free positions together. The beam is a bounded proposal search, not a feasibility proof.
    """
    from . import network_lattice as fill_lattice
    from .patterns.validate import validate_steps,InvalidFill
    n=len(p)
    if n<8 or n>int(settings.get('lattice_max_boundary',64)) or n%2:return None
    floor=float(settings.get('_quality_floor',.25))
    turns=geom.turning(p);steps=np.array(fill_lattice.STEPS)
    # An exterior vertex assigned +1 belongs to one convex quad. Its fixed
    # angle therefore bounds quality; impossible corners need no search work.
    allowed=(turns>0)&(np.sin(turns)>=floor)
    states=np.array([[0,1,0]],int);cost=np.zeros(1)
    paths=np.zeros((1,n),np.int8);width=2*n+1
    for i in range(1,n):
        opts=np.array([-1,0,1] if allowed[i] else [-1,0])
        rep=np.repeat(np.arange(len(states)),len(opts));q=np.tile(opts,len(states))
        st=states[rep].copy();st[:,0]+=q;st[:,1:]+=steps[st[:,0]%4]
        co=cost[rep]+(turns[i]-q*math.pi/2)**2
        ok=(np.abs(st[:,1:]).sum(1)<=n-i-1)&(st[:,0]>=-2)&(st[:,0]<=6)
        st=st[ok];co=co[ok];rep=rep[ok];q=q[ok]
        if not len(st):return None
        key=(st[:,0]*width+st[:,1])*width+st[:,2]
        order=np.lexsort((co,key));ks=key[order]
        starts=np.maximum.accumulate(np.where(np.r_[True,ks[1:]!=ks[:-1]],np.arange(len(ks)),0))
        order=order[np.arange(len(ks))-starts<int(settings.get('lattice_paths_per_state',8))]
        order=order[np.argsort(co[order],kind='stable')[:int(settings.get('lattice_beam',2000))]]
        states=st[order];cost=co[order]
        paths=paths[rep[order]].copy();paths[:,i]=q[order]
        geom.check_time();yield None
    q0=4-states[:,0]
    good=(q0<=1)&(q0>=-1)&((q0<=0)|allowed[0])
    paths=paths[good];cost=cost[good]+(turns[0]-q0[good]*math.pi/2)**2
    paths[:,0]=q0[good];attempts=0
    for j in np.argsort(cost,kind='stable'):
        coords=fill_lattice.lattice_walk(paths[j])
        if coords is None:continue
        built=yield from fill_lattice.lattice_quad_steps(coords)
        if built is None:continue
        quads,total=built;attempts+=1
        positions=geom.laplace(p,total,quads,iterations=240)
        geom.check_time();yield None
        if geom.min_corner_area(positions,quads)<=1e-12:
            positions,ok=yield from geom.untangle_steps(positions,n,quads)
            if not ok:
                if attempts>=int(settings.get('lattice_attempts',8)):return None
                continue
        positions=yield from geom.winslow_steps(positions,n,quads)
        quality=geom.quad_quality(positions,quads)
        if quality.min()<floor:
            if attempts>=int(settings.get('lattice_attempts',8)):return None
            continue
        try:metrics=yield from validate_steps(p,positions,quads)
        except InvalidFill:
            if attempts>=int(settings.get('lattice_attempts',8)):return None
            continue
        return dict(positions=positions,quads=tuple(quads),pattern='medial',rotation=0,
                    corners=(),anchors=[],metrics=metrics,score=float(quality.mean()),
                    medial_seams=(),medial_counts=(),
                    medial_paths=(),medial_turns=tuple(map(int,paths[j])),
                    medial_method='coupled_lattice',medial_lattice_attempts=attempts,
                    medial_singular_piece=None)
    return None

