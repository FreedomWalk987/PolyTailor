# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Cooperative, independent checks for planar fixed-boundary quad previews."""
from collections import Counter
import numpy as np


class InvalidFill(ValueError):
    pass


def area(points):
    return float(np.sum(points[:,0]*np.roll(points[:,1],-1)-points[:,1]*np.roll(points[:,0],-1))*.5)


def cross(a,b):
    return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def crossing_steps(positions, edge_ids):
    """Exact old contact predicates on conservative sweep/AABB candidates only.

    Keep every threshold and the original segment order. Sorting only prunes
    pairs which cannot satisfy the previous expanded-box overlap predicate.
    Bounded row groups yield even when there are no candidate contacts.
    """
    if len(edge_ids)<2:
        return
    a, b = positions[edge_ids[:,0]], positions[edge_ids[:,1]]
    lower, upper = np.minimum(a,b), np.maximum(a,b)
    extent = np.ptp(np.concatenate((lower,upper)),axis=0)
    score = np.sum(upper-lower,axis=0)/np.maximum(extent,1e-30)
    score[extent<=1e-12] = np.inf
    axis = int(np.argmin(score))
    order = np.argsort(lower[:,axis],kind='stable')
    starts = lower[order,axis]
    for start in range(0,len(order),16):
        end = min(start+16,len(order))
        rows = order[start:end]
        # Add in the coordinate dtype, just as the old overlap test does; a
        # float32 box must not be pruned using a narrower float64-rounded bound.
        stop = int(np.searchsorted(starts,np.max(upper[rows,axis]+1e-12),side='right'))
        columns = order[start+1:stop]
        if len(columns):
            overlap = np.all(np.maximum(lower[rows,None],lower[columns]) <=
                             np.minimum(upper[rows,None],upper[columns])+1e-12,axis=2)
            overlap &= np.arange(start+1,stop)[None,:] > np.arange(start,end)[:,None]
            ri, ci = np.nonzero(overlap)
            # Restore insertion order so arithmetic is identical even for pairs
            # that the sweep visits in the opposite order.
            ii = np.minimum(rows[ri],columns[ci])
            jj = np.maximum(rows[ri],columns[ci])
            independent = ((edge_ids[ii,0]!=edge_ids[jj,0]) &
                           (edge_ids[ii,0]!=edge_ids[jj,1]) &
                           (edge_ids[ii,1]!=edge_ids[jj,0]) &
                           (edge_ids[ii,1]!=edge_ids[jj,1]))
            ii, jj = ii[independent], jj[independent]
            if len(ii):
                aa,bb,cc,dd = a[ii],b[ii],a[jj],b[jj]
                c1,c2 = cross(bb-aa,cc-aa),cross(bb-aa,dd-aa)
                c3,c4 = cross(dd-cc,aa-cc),cross(dd-cc,bb-cc)
                contacts = ((c1*c2 < -1e-24)&(c3*c4 < -1e-24)) | ((abs(c1)<=1e-12)&(abs(c2)<=1e-12))
                contacts |= (abs(c1)<=1e-12)&(np.sum((cc-aa)*(cc-bb),axis=1)<=1e-12)
                contacts |= (abs(c2)<=1e-12)&(np.sum((dd-aa)*(dd-bb),axis=1)<=1e-12)
                contacts |= (abs(c3)<=1e-12)&(np.sum((aa-cc)*(aa-dd),axis=1)<=1e-12)
                contacts |= (abs(c4)<=1e-12)&(np.sum((bb-cc)*(bb-dd),axis=1)<=1e-12)
                if np.any(contacts):
                    raise InvalidFill('Crossing or unjoined edges')
        yield None


def validate_steps(boundary,positions,quads):
    n = len(boundary)
    q = np.asarray(quads,dtype=np.int64)
    if q.ndim!=2 or q.shape[1]!=4 or q.size==0 or q.min()<0 or q.max()>=len(positions):
        raise InvalidFill('Invalid quad indices')
    if not np.array_equal(positions[:n],boundary) or not np.all(np.isfinite(positions)):
        raise InvalidFill('Boundary changed')
    p = positions[q]
    sides = np.roll(p,-1,axis=1)-p
    turns = cross(sides,np.roll(sides,-1,axis=1))
    if np.min(turns)<=1e-12:
        raise InvalidFill('Folded or degenerate quads')
    areas = cross(p,np.roll(p,-1,axis=1)).sum(axis=1)*.5
    if abs(float(areas.sum())-area(boundary))>1e-8:
        raise InvalidFill('Area coverage mismatch')
    edges = Counter(tuple(sorted((a,b))) for face in quads for a,b in zip(face,face[1:]+face[:1]))
    expected = {tuple(sorted((i,(i+1)%n))) for i in range(n)}
    if {e for e,c in edges.items() if c==1} != expected or any(c not in (1,2) for c in edges.values()):
        raise InvalidFill('Boundary or edge incidence mismatch')
    if len(positions)-len(edges)+len(q)!=1 or len(set(q.ravel()))!=len(positions):
        raise InvalidFill('Disconnected or unused topology')
    edge_ids = np.asarray(list(edges),dtype=np.int64)
    yield from crossing_steps(positions,edge_ids)
    lengths = np.linalg.norm(sides,axis=2)
    return {'quads':len(q),'aspect_p95':float(np.percentile(lengths.max(axis=1)/lengths.min(axis=1),95))}
