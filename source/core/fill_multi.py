# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Selected sampled panel networks, preserving actual BMesh seam identities."""
from collections import Counter
import numpy as np
from . import fill, panel_network, network_fill, fill_progressive
from .patterns import geom


def selected_panels(bm):
    edges = [e for e in bm.edges if e.select and not e.hide]
    if not edges or not all(fill._open(e) for e in edges):
        return [], ''
    degree = Counter(v for e in edges for v in e.verts)
    if any(d < 2 for d in degree.values()):
        return [], ''
    loops = []
    for block in fill._cycle_blocks(edges):
        counts = Counter(v for e in block for v in e.verts)
        loops.extend(fill._cycles(block) if all(d==2 for d in counts.values()) else fill._planar_faces(block))
    if len(loops) < 2:
        return [], ''
    used = {bm.edges.get((a,b)) for loop in loops for a,b in zip(loop,loop[1:]+loop[:1])}
    if used != set(edges) or any(fill._filled_loop(loop) for loop in loops):
        return [], 'Select closed empty panels without dangling connecting edges'
    bm.verts.index_update()
    def ordered(loop):
        i = min(range(len(loop)), key=lambda k:loop[k].index)
        return loop[i:]+loop[:i]
    return sorted((ordered(loop) for loop in loops), key=lambda ring:tuple(v.index for v in ring)), ''


def combine(plans, sources, world, counts, validation):
    positions = list(np.asarray(world));faces=[]
    for plan, ids in zip(plans, sources):
        n=plan['boundary']
        remap=list(ids)+list(range(len(positions),len(positions)+len(plan['positions'])-n))
        positions.extend(plan['positions'][n:])
        faces.extend(tuple(remap[i] for i in q) for q in fill.faces_of(plan))
    return dict(positions=np.asarray(positions), quads=faces, boundary=len(world), order=list(range(len(world))),
                virtual=None, collapse=None, normal_ccw=plans[0]['normal_ccw'], pattern='network', shift=0,
                rotation=0, corners=(), anchors=[], layout=None, panel_count=len(plans),
                shared_seams=len(validation['seams']), counts=counts, network_validation=validation,
                panel_plans=plans, panel_sources=sources)


DISJOINT = "Select one hole, or panels that share edges"   # the refusal of a selection of separate loops


def _groups(loops):
    """the selected loops' groups: loop index -> group id, loops sharing a vertex are one group."""
    group = list(range(len(loops)))
    def root(i):
        while group[i] != i:
            i = group[i]
        return i
    owner = {}
    for i, loop in enumerate(loops):
        for v in loop:
            group[root(i)] = root(owner.setdefault(v, i))
    return [root(i) for i in range(len(loops))]


def _own_plane_steps(plan, pts, plane, quads, shared_normal):
    """One panel placed on its own best-fit plane instead of the shared one: the shared-plane layout is mapped by
    the affine fit of its outline, the outline's remaining offsets spread harmonically inside, then smoothed and
    lifted by harmonic heights. The shared-plane plan stays when the own-plane one folds."""
    n=len(pts)
    centre,u,v,normal=fill._frame(pts)
    if float(normal@shared_normal)<0:   # keep the panel's winding
        v,normal=-v,-normal
    own=np.column_stack(((pts-centre)@u,(pts-centre)@v))
    flat=np.asarray(plane,float)
    qs=[tuple(q) for q in quads]
    fit,*_=np.linalg.lstsq(np.column_stack((flat[:n],np.ones(n))),own,rcond=None)
    placed=np.column_stack((flat,np.ones(len(flat))))@fit
    placed=placed+geom.laplace(own-placed[:n],len(flat),qs,iterations=240)
    placed[:n]=own
    placed=yield from fill_progressive.deterministic_steps(geom.winslow_steps(placed,n,qs))
    h=geom.laplace(np.column_stack(((pts-centre)@normal,np.zeros(n))),len(flat),qs,iterations=240)[:,0]
    positions=centre+placed[:,:1]*u+placed[:,1:2]*v+h[:,None]*normal
    positions[:n]=pts
    out=dict(plan,positions=positions,normal_ccw=normal)
    try:
        fill.validate_plan(out)
    except fill.FillError:
        return plan
    return out


def plan_steps(mw, loops, settings, project=None, own_planes=False):
    """own_planes: each panel's interior is placed on its own best-fit plane (the network counts stay on the shared
    plane); otherwise every panel uses the shared plane."""
    vertices=sorted({v for loop in loops for v in loop}, key=lambda v:v.index)
    source={v:i for i,v in enumerate(vertices)}
    world=fill.world_points(mw,vertices)
    centre,u,v,_=fill._frame(world)
    normal=np.cross(u,v)
    xy=np.column_stack(((world-centre)@u,(world-centre)@v))
    panels=[];sources=[]
    for index,loop in enumerate(loops):
        ids=[source[v] for v in loop]
        local=xy[ids]
        if np.sum(local[:,0]*np.roll(local[:,1],-1)-local[:,1]*np.roll(local[:,0],-1))<0:
            ids.reverse();local=local[::-1]
        sources.append(ids)
        panels.append(dict(id='panel:'+str(index+1),points=local,source_ids=ids))
    group=_groups(loops)
    disjoint=len(set(group))>1   # separate loops on one best-fit plane: a refusal there is not about seams
    try:
        network=yield from panel_network.build_steps(panels)
    except ValueError as exc:   # e.g. both ends of a tube project to degenerate loops
        raise fill.FillError(DISJOINT if disjoint else 'Panel network: '+str(exc)) from None
    # Coincidence may diagnose a seam, but only an already shared mesh vertex
    # authorizes a shared identity. Never silently weld separate source vertices.
    for node in network['nodes']:
        identities={panels[int(pid.split(':')[1])-1]['source_ids'][local] for pid,local in node['uses']}
        if len(identities)!=1:
            if len({group[int(pid.split(':')[1])-1] for pid,local in node['uses']})>1:
                raise fill.FillError(DISJOINT)   # separate loops that only meet in the projection
            raise fill.FillError('Merge the overlapping seam vertices first')
    yield None
    options=dict(settings,_network_data=network)
    result=yield from fill_progressive.deterministic_steps(network_fill.solve_steps(options))
    if result['status']!='filled':
        conflict=result['counts'].get('reason') or next((e['kind'] for e in result['validation']['errors']), 'No conforming panel fill')
        raise fill.FillError(DISJOINT if disjoint else 'Panel network: '+conflict)
    plans=[]
    for index,(panel,ids) in enumerate(zip(panels,sources)):
        out=result['panels'][panel['id']];plane=out['positions'];qs=out['quads'];n=len(ids)
        heights=(world[ids]-centre)@normal
        h=geom.laplace(np.column_stack((heights,np.zeros(n))),len(plane),qs,iterations=240)[:,0]
        positions=centre+plane[:,:1]*u+plane[:,1:2]*v+h[:,None]*normal
        positions[:n]=world[ids]
        # Each per-panel order maps back to the actual selected loop, which may
        # have the opposite winding to the common chart.
        loop=loops[index]
        local_order=[loop.index(vertices[i]) for i in ids]
        plan=dict(positions=positions,quads=qs,boundary=n,order=local_order,virtual=None,collapse=None,
                  normal_ccw=normal,pattern='network',shift=0,rotation=0,corners=(),anchors=[],layout=None)
        fill.validate_plan(plan)
        if own_planes:
            plan=yield from _own_plane_steps(plan,world[ids],plane,qs,normal)
        if project is not None:
            plan=yield from project(plan)
        plans.append(plan)
        yield None
    return combine(plans,sources,world,result['counts'],result['validation'])


def commit(bm,mw,loops,plan,view_dir,correct_attributes=True):
    """All panels commit or all additions roll back; the selected seam stays put."""
    if len(loops)!=plan['panel_count']:
        raise fill.FillError('Panel selection changed; preview again')
    uses=Counter(bm.edges.get((a,b)) for loop in loops for a,b in zip(loop,loop[1:]+loop[:1]))
    if any(edge is None or not edge.is_valid or len(edge.link_faces)+count>2 for edge,count in uses.items()):
        raise fill.FillError('Panel network would create a non-manifold seam')
    for loop,p in zip(loops,plan['panel_plans']):
        fill.validate_plan(p)
        if not np.array_equal(fill.world_points(mw,[loop[i] for i in p['order']]),p['positions'][:p['boundary']]):
            raise fill.FillError('The outline changed; preview again')
    boundary={v for loop in loops for v in loop}
    existing_edges={edge for vertex in boundary for edge in vertex.link_edges}
    rings=[];new_vertices=set();new_edges={}
    try:
        for loop,p in zip(loops,plan['panel_plans']):
            added=fill.commit(bm,mw,loop,p,view_dir,correct_attributes)
            # Native attribute transfer can invalidate face wrappers. Keep
            # vertex rings and rebind them, as the single-panel transaction does.
            for face in added:
                rings.append(tuple(face.verts))
                new_vertices.update(v for v in face.verts if v not in boundary)
                for edge in face.edges:
                    if edge not in existing_edges:
                        new_edges[frozenset(edge.verts)]=tuple(edge.verts)
    except Exception:
        for ring in rings:
            face=bm.faces.get(ring)
            if face is not None:bm.faces.remove(face)
        for endpoints in new_edges.values():
            edge=bm.edges.get(endpoints)
            if edge is not None and not edge.link_faces:bm.edges.remove(edge)
        for vertex in new_vertices:
            if vertex.is_valid and not vertex.link_edges:bm.verts.remove(vertex)
        bm.normal_update()
        raise
    warnings=[p.get('uv_warning') for p in plan['panel_plans']]
    plan['uv_warning']='seam' if 'seam' in warnings else 'missing' if 'missing' in warnings else None
    return [bm.faces.get(ring) for ring in rings]
