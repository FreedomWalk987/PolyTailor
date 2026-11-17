# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Explicit odd-loop sites. The search never replaces the current AUTO pick."""
import numpy as np
from . import fill, fill_progressive


def metrics(plan):
    faces = fill.faces_of(plan)
    positions = np.asarray(plan['positions'])
    minimum = 1.
    for size in sorted({len(f) for f in faces}):
        p = positions[np.asarray([f for f in faces if len(f) == size])]
        d = np.roll(p, -1, axis=1) - p
        q = np.linalg.norm(np.cross(d, np.roll(d, -1, axis=1)), axis=2)
        q /= np.maximum(np.linalg.norm(d, axis=2) * np.linalg.norm(np.roll(d, -1, axis=1), axis=2), 1e-30)
        minimum = min(minimum, float(q.min()))
    return dict(poles=fill._interior_poles(dict(positions=positions, quads=faces)),
                minQ=minimum, faces=len(faces))


def site(plan):
    virtual = plan.get('virtual')
    if virtual is None:
        return None
    order, n = plan['order'], plan['boundary']
    a, b = order[(virtual-1) % n], order[(virtual+1) % n]
    count = n-1
    edge = a if (a+1) % count == b else b
    return dict(edge=int(edge), vertex=int(order[plan['collapse']]))


def entry(plan, default=False):
    return dict(plan=plan, site=site(plan), default=default, **metrics(plan))


def solve_steps(world, settings, edge, project=None):
    options = {k: v for k, v in settings.items()
               if k in ('_face_turns',)}
    options.update(pattern='AUTO', rotation=-1, _triangle_edge=int(edge))
    plan = yield from fill_progressive.deterministic_steps(fill.solve_steps(world, options))
    if project is not None:
        plan = yield from project(plan)
    fill.validate_plan(plan)
    if site(plan)['edge'] != edge:
        raise fill.FillError('Triangle site changed; preview again')
    return plan


def candidates_steps(world, settings, default, project=None, reference=None):
    """Compare up to eight distributed/long edges, retain default plus top three.

    Ranking uses final 3D poles, minimum corner sine, then face count. This is a
    bounded candidate search, not a claim of the global optimum over all sites.
    """
    n = len(world)
    if n < 5 or n % 2 == 0 or site(default) is None:
        return []
    original = entry(default, True)
    original['_ranking'] = settings.get('_ranking', 'FLOW')
    original['_quality_floor'] = settings.get('_quality_floor', .20)
    edges = candidate_edges(world, default)
    found = []
    pending = [edge for edge in edges if edge != original['site']['edge']]
    for edge in pending:
        try:
            plan = yield from solve_steps(world, settings, edge, project)
            row = entry(plan)
            row['_ranking'] = settings.get('_ranking', 'FLOW')
            row['_quality_floor'] = settings.get('_quality_floor', .20)
            found.append(row)
        except (fill.FillError, ValueError):
            pass
        yield None
    from .fill_flow import guarded_choice, fingerprint
    from .fill_flow import score as value
    reference = reference if reference is not None else default
    retained = original
    if reference is not default:
        retained = entry(reference, True)
        retained['_ranking'] = 'FLOW'
        retained['_quality_floor'] = settings.get('_quality_floor', .20)
        found.append(retained)
    # Distinct plan objects can contain the exact same final mesh. Keep
    # the fixed reference object for fallback without duplicating its row.
    unique = {}
    for row in [original]+found:
        key = fingerprint(row['plan'])
        if key not in unique or row['plan'] is reference:
            unique[key] = row
    options = list(unique.values())
    original = unique[fingerprint(default)]
    retained = unique[fingerprint(reference)]
    chosen = guarded_choice([row['plan'] for row in options], reference, settings.get('_quality_floor', .20))
    selected = next(row for row in options if row['plan'] is chosen)
    options.sort(key=lambda row: (-round(value(row['plan']), 12), row['site']['edge']))
    shown = options[:4]
    for required in (original, retained, selected):
        if not any(row is required for row in shown):
            drop = next((i for i in range(len(shown)-1, -1, -1)
                         if all(shown[i] is not row for row in (original, retained, selected))), None)
            if drop is not None:
                shown[drop] = required
            else:
                shown.append(required)
    # the guarded pick is entry 1; the rest follow by S (2.6.1; before: U),
    # best first, with entries below the quality floor last; ties by U; non-finite last.
    from .fill_flow import metrics, eligible
    import math
    floor = settings.get('_quality_floor', .20)
    fin = lambda x: x if math.isfinite(x) else -math.inf
    shown.sort(key=lambda row: (row is not selected, not eligible(row['plan'], floor),
                                -fin(round(value(row['plan']), 12)),
                                -fin(round(metrics(row['plan'])['raw_U'], 12)), row['site']['edge']))
    for row in shown:
        row['_auto_selected'] = row is selected
    return shown


def candidate_edges(world, default):
    points = np.asarray(world)
    lengths = np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1)
    n = len(world)
    return list(dict.fromkeys([site(default)['edge']] +
                [int(i) for i in np.argsort(-lengths, kind='stable')[:4]] +
                [i*n//4 for i in range(4)]))[:8]
