# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Per-panel disks and welded-network geometry/manifold checks, allowing holes."""
from collections import Counter, defaultdict
import numpy as np
from .patterns import geom
from .patterns.validate import validate_steps, crossing_steps, InvalidFill


def validate_network_steps(network, results):
    errors = list(network['conflicts'])
    positions = [v['point'][:2].copy() for v in network['nodes']]
    quads = []; panel_faces = {}; panel_checks = {}
    for panel in network['panels']:
        pid = panel['id']; result = results.get(pid)
        if result is None or result['status'] != 'filled':
            errors.append(dict(kind='panel_not_filled', panel=pid, reason=None if result is None else result.get('reason')))
            continue
        n = len(panel['points']); pos = np.asarray(result['positions'])[:, :2]
        expected_boundary=np.asarray([network['nodes'][i]['point'][:2] for i in panel['global_ids']])
        shift=float(np.linalg.norm(expected_boundary-panel['points'][:,:2],axis=1).max())
        if shift>network['tolerance']:
            errors.append(dict(kind='canonical_boundary_outside_tolerance',panel=pid,max_shift=shift));continue
        try:
            (yield from validate_steps(expected_boundary, pos, result['quads']))
        except InvalidFill as exc:
            errors.append(dict(kind='panel_invalid', panel=pid, reason=str(exc)))
            continue
        remap = panel['global_ids']+list(range(len(positions), len(positions)+len(pos)-n))
        positions.extend(pos[n:])
        canonical = np.asarray([positions[v] for v in remap])
        try:
            (yield from validate_steps(canonical[:n], canonical, result['quads']))
        except InvalidFill as exc:
            errors.append(dict(kind='canonical_panel_invalid', panel=pid, reason=str(exc)))
        panel_checks[pid] = dict(original_boundary_exact=np.array_equal(panel['points'][:,:2],pos[:n]),
                                canonical_max_shift=shift)
        panel_faces[pid] = [tuple(remap[v] for v in q) for q in result['quads']]
        quads.extend(panel_faces[pid])
        yield None
    pos = np.asarray(positions)
    edges = Counter(tuple(sorted((a, b))) for q in quads for a, b in zip(q, q[1:]+q[:1]))
    seam_checks = []
    for arc in network['arcs']:
        if len(arc['panels']) != 2:
            continue
        expected = list(zip(arc['nodes'], arc['nodes'][1:]))
        valid = all(edges[tuple(sorted(e))] == 2 for e in expected)
        directions = []
        for owner in arc['panels']:
            directed = Counter((a, b) for q in panel_faces.get(owner, []) for a, b in zip(q, q[1:]+q[:1]))
            directions.append([1 if directed[e] else -1 if directed[e[::-1]] else 0 for e in expected])
        valid &= all(a == -b and a != 0 for a, b in zip(*directions))
        seam_checks.append(dict(seam=arc['id'], count=arc['count'], valid=bool(valid), identical_global_ids=True))
        if not valid:
            errors.append(dict(kind='nonconforming_seam', seam=arc['id'], panels=arc['panels']))
    if quads:
        if any(c not in (1, 2) for c in edges.values()):
            errors.append(dict(kind='nonmanifold_edges'))
        if len({tuple(sorted(q)) for q in quads}) != len(quads):
            errors.append(dict(kind='duplicate_faces'))
        try:
            (yield from crossing_steps(pos, np.asarray(list(edges), int)))
        except InvalidFill as exc:
            errors.append(dict(kind='network_intersection', reason=str(exc)))
        if geom.min_corner_area(pos, quads) <= 1e-12:
            errors.append(dict(kind='network_fold'))
        # Edge incidence alone misses vertex bow ties. Every incident face fan
        # must be connected through edges at that vertex, with 0 or 2 border edges.
        incident = defaultdict(set); edge_faces = defaultdict(list)
        for i, q in enumerate(quads):
            for v in q:
                incident[v].add(i)
            for a, b in zip(q, q[1:]+q[:1]):
                edge_faces[tuple(sorted((a, b)))].append(i)
        for index, (v, faces) in enumerate(incident.items()):
            if index % 16 == 0:
                yield None
            links = defaultdict(set); boundary_degree = 0
            for e, fs in edge_faces.items():
                if v not in e:
                    continue
                boundary_degree += len(fs) == 1
                for a in fs:
                    links[a].update(fs)
            visited = set(); todo = [min(faces)]
            while todo:
                a = todo.pop()
                if a not in visited:
                    visited.add(a); todo.extend(links[a]-visited)
            if visited != faces or boundary_degree not in (0, 2):
                errors.append(dict(kind='nonmanifold_vertex', vertex=v, boundary_degree=boundary_degree))
        # Detect one panel wholly contained in another: edge crossing alone does
        # not detect containment. Strict point-in-polygon excludes seam contacts.
        from .network_medial import _inside_many
        for owner, faces in panel_faces.items():
            centers = pos[np.asarray(faces)].mean(axis=1)
            for panel in network['panels']:
                if panel['id'] != owner and _inside_many(panel['points'][:, :2], centers).any():
                    errors.append(dict(kind='panel_interior_overlap', panels=[owner, panel['id']]))
    boundary = {v for e, c in edges.items() if c == 1 for v in e}
    degree = Counter(v for e in edges for v in e)
    used = {v for q in quads for v in q}
    return dict(valid=not errors, errors=errors, seams=seam_checks, panels=panel_checks,
                faces=len(quads), interior_poles=sum(degree[v] != 4 for v in used-boundary),
                euler_characteristic=len(used)-len(edges)+len(quads),
                positions=pos, quads=tuple(quads), frozen_exterior_intervals=network['exterior_intervals'])
