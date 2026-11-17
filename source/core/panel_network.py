# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Frozen sampled panel loops, coincident seams and explicit geometry diagnostics."""
from collections import defaultdict
from itertools import combinations, product
import time
import numpy as np


def build_steps(panels, tolerance=None):
    """Input: id, points (CCW XY/XYZ), optional logical_sides of loop indices.

    No resampling or near-miss welding. Coordinate representatives are accepted
    only inside the coincidence tolerance, and every original array is retained.
    Loops lie in one common XY plane; other embeddings are refused explicitly.
    """
    start = time.perf_counter()
    records = []
    for panel in panels:
        p = np.asarray(panel['points'], dtype=float)
        if p.ndim != 2 or p.shape[1] not in (2, 3) or len(p) < 3 or not np.isfinite(p).all():
            raise ValueError('Invalid panel points: ' + panel['id'])
        if p.shape[1] == 2:
            p = np.column_stack((p, np.zeros(len(p))))
        area = np.sum(p[:, 0]*np.roll(p[:, 1], -1)-p[:, 1]*np.roll(p[:, 0], -1))/2
        if area <= 0:
            raise ValueError('Expected a counterclockwise nondegenerate loop: ' + panel['id'])
        records.append(dict(id=panel['id'], points=p.copy(), logical_sides=panel.get('logical_sides'),
                            source_ids=panel.get('source_ids', list(range(len(p))))))
    if not records or len({v['id'] for v in records}) != len(records):
        raise ValueError('Empty network or duplicate panel id')
    allp = np.concatenate([v['points'] for v in records])
    diagonal = float(np.linalg.norm(np.ptp(allp, axis=0)))
    tolerance = max(diagonal*1e-7, 1e-12) if tolerance is None else float(tolerance)
    if tolerance <= 0:
        raise ValueError('Positive tolerance required')
    if np.ptp(allp[:, 2]) > tolerance:
        raise ValueError('Prototype supports a common XY plane only')
    nodes = []
    buckets = defaultdict(list)
    conflicts = []
    for panel in records:
        ids = []
        for local, p in enumerate(panel['points']):
            if local % 16 == 0:
                yield None
            key = tuple(np.floor(p/tolerance).astype(np.int64))
            candidates = sorted({i for offset in product((-1, 0, 1), repeat=3)
                                 for i in buckets[tuple(a+b for a, b in zip(key, offset))]})
            matches = [i for i in candidates if all(np.linalg.norm(p-q) <= tolerance for q in nodes[i]['source_points'])]
            if len(matches) > 1:
                conflicts.append(dict(kind='ambiguous_coincidence', panel=panel['id'], vertex=local, nodes=matches))
                matches = []
            if matches and any(owner == panel['id'] for owner, _ in nodes[matches[0]]['uses']):
                raise ValueError('Repeated or coincident samples in panel: ' + panel['id'])
            if matches:
                gid = matches[0]
                nodes[gid]['uses'].append((panel['id'], local))
                nodes[gid]['source_points'].append(p.copy())
            else:
                gid = len(nodes)
                nodes.append(dict(point=p.copy(), uses=[(panel['id'], local)], source_points=[p.copy()]))
                buckets[key].append(gid)
            ids.append(gid)
        panel['global_ids'] = ids
    edges = defaultdict(list)
    for panel in records:
        ids = panel['global_ids']
        for k, (a, b) in enumerate(zip(ids, ids[1:]+ids[:1])):
            edges[tuple(sorted((a, b)))].append((panel['id'], k, (a, b)))
    by_owners = defaultdict(set)
    for edge, uses in edges.items():
        owners = tuple(sorted(u[0] for u in uses))
        if len(uses) > 2 or len(owners) != len(set(owners)):
            conflicts.append(dict(kind='nonmanifold_edge', edge=edge, panels=owners))
        if len(uses) == 2 and uses[0][2] != uses[1][2][::-1]:
            conflicts.append(dict(kind='same_orientation_overlap', edge=edge, panels=owners))
        by_owners[owners].add(edge)
    arcs = []
    edge_arc = {}
    for owners, unused0 in sorted(by_owners.items()):
        unused = set(unused0)
        links = defaultdict(list)
        for a, b in unused:
            links[a].append(b); links[b].append(a)
        while unused:
            endpoints = sorted(v for v in links if len(links[v]) != 2 and any(tuple(sorted((v, w))) in unused for w in links[v]))
            chain = [endpoints[0] if endpoints else min(min(e) for e in unused)]
            while True:
                choices = sorted(w for w in links[chain[-1]] if tuple(sorted((chain[-1], w))) in unused)
                if not choices:
                    break
                a = chain[-1]; b = choices[0]
                unused.remove(tuple(sorted((a, b)))); chain.append(b)
                if b == chain[0] or len(links[b]) != 2:
                    break
            name = ('S' if len(owners) == 2 else 'B') + str(len(arcs)+1).zfill(2)
            arc = dict(id=name, panels=list(owners), nodes=chain, count=len(chain)-1, uses=[])
            for owner in owners:
                source = next(p for p in records if p['id'] == owner)
                local = [source['global_ids'].index(v) for v in chain]
                orientation = 1 if (local[1]-local[0]) % len(source['points']) == 1 else -1
                arc['uses'].append(dict(panel=owner, indices=local, orientation=orientation))
            arcs.append(arc)
            for a, b in zip(chain, chain[1:]):
                edge_arc[tuple(sorted((a, b)))] = name
    # Unequal sampled collinear overlaps are geometry conflicts. Do not infer
    # semantic seams or diagnose near misses between independent source panels.
    for pa, pb in combinations(records, 2):
        aa, bb = pa['global_ids'], pb['global_ids']
        # Catch unequal sampled collinear overlaps, including unjoined T nodes.
        for i, (a, b) in enumerate(zip(pa['points'], np.roll(pa['points'], -1, axis=0))):
            if i % 4 == 0:
                yield None
            d = b-a; length = float(np.linalg.norm(d)); u = d/length
            for j, (c, e) in enumerate(zip(pb['points'], np.roll(pb['points'], -1, axis=0))):
                if {aa[i], aa[(i+1) % len(aa)]} == {bb[j], bb[(j+1) % len(bb)]}:
                    continue
                t0, t1 = float((c-a)@u), float((e-a)@u)
                residual = max(np.linalg.norm(c-a-t0*u), np.linalg.norm(e-a-t1*u))
                overlap = min(length, max(t0, t1))-max(0., min(t0, t1))
                if residual <= tolerance and overlap > tolerance:
                    conflicts.append(dict(kind='unmatched_collinear_overlap', panels=[pa['id'], pb['id']],
                        edges=[[pa['source_ids'][i], pa['source_ids'][(i+1) % len(aa)]],
                               [pb['source_ids'][j], pb['source_ids'][(j+1) % len(bb)]]], overlap=float(overlap), welded=False))
    for panel in records:
        ids = panel['global_ids']
        panel['edge_arcs'] = [edge_arc[tuple(sorted((a, b)))] for a, b in zip(ids, ids[1:]+ids[:1])]
    return dict(mode='coincident',panels=records, nodes=nodes, arcs=arcs, conflicts=conflicts, tolerance=tolerance,
                exterior_intervals=sum(a['count'] for a in arcs if len(a['panels']) == 1),
                build_ms=(time.perf_counter()-start)*1000)


def count_model(network):
    """Necessary boundary parity per panel (no authored side layouts: no caller supplies logical_sides).

    Geometry is frozen for filling.
    """
    names = [a['id'] for a in network['arcs']]
    bounds = {a['id']: (a['count'], a['count']) for a in network['arcs']}
    targets = {a['id']: a['count'] for a in network['arcs']}
    equations = []
    for panel in network['panels']:
        pid = panel['id']
        half = pid+':half_boundary'
        names.append(half); bounds[half] = (2, 4096); targets[half] = len(panel['points'])//2
        coefficients = defaultdict(int)
        for name in panel['edge_arcs']:   # (every name is an arc: the count lookup below finds it)
            coefficients[name] += 1
        # Every occurrence contributes 1/count of an arc, but arcs can wrap a
        # panel. Summing their full counts once is exact for the registry.
        coefficients = {name: count//next(a['count'] for a in network['arcs'] if a['id']==name)
                        for name,count in coefficients.items()}
        coefficients[half] = -2
        equations.append(dict(coefficients=coefficients, rhs=0, label=pid+':even_boundary'))
    return dict(variables=names, equations=equations, bounds=bounds, targets=targets)
