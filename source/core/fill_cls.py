# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""classification candidates (direction A).

Each rim vertex is either a logical corner (End, one quad) or a side vertex (two quads). The cost of making
vertex i a corner instead of a side is |theta_i - 90| - |theta_i/2 - 90| (theta = interior rim angle, deg).
For every patch side count N = 2..6 the single cheapest corner set (K = 1) is handed to the existing Takayama
engine through its locked-anchor path (`anchors`: exactly these corners, no subsets, no heuristic fallback).
At most five engine calls per outline; a call that finds no valid pattern is skipped. Even rims only.
"""
import heapq

import numpy as np

from .patterns import geom

N_SIDES = (2, 3, 4, 5, 6)
K_SETS = 1            # cheapest corner set per N
STEP_CAP = 5000       # cooperative engine steps per call (B 240 max 528); failsafe only, counted (not timed) so results never depend on load


def corner_costs(unit):
    """Per-vertex cost (deg) of a one-quad corner instead of a two-quad side; `unit` is a ccw loop."""
    theta = np.degrees(np.pi-geom.turning(unit))
    return [abs(t-90.)-abs(t/2.-90.) for t in theta]


def cheapest_sets(cost, n_corners, k=K_SETS):
    """The k cheapest corner sets of size n_corners (exact, best-first over sorted costs); ties by index."""
    order = sorted(range(len(cost)), key=lambda i: (cost[i], i))
    d = [cost[i] for i in order]
    if n_corners > len(d):
        return []
    start = tuple(range(n_corners))
    heap, seen, out = [(sum(d[i] for i in start), start)], {start}, []
    while heap and len(out) < k:
        total, t = heapq.heappop(heap)
        out.append((total, sorted(order[i] for i in t)))
        for j in range(n_corners):
            if t[j]+1 < len(d) and (j == n_corners-1 or t[j]+1 < t[j+1]):
                u = t[:j]+(t[j]+1,)+t[j+1:]
                if u not in seen:
                    seen.add(u)
                    heapq.heappush(heap, (total-d[t[j]]+d[t[j]+1], u))
    return out


def candidate_steps(xy, settings=None):
    """Cooperative generator over a ccw 2D loop; yields None or {'candidate': {...}} like fill_p1."""
    from . import network_fill, boundary_preview
    from .patterns.validate import validate_steps, InvalidFill
    settings = settings or {}
    floor = float(settings.get('_quality_floor', .25))
    xy = np.asarray(xy, float)
    n = len(xy)
    if n < 4 or n % 2:
        return
    scale = float(np.ptp(xy, axis=0).max())
    if not scale > 0:
        return
    cost = corner_costs((xy-xy.mean(0))/scale)
    refs = [('p', i) for i in range(n)]
    seen = set()
    for sides in N_SIDES:
        for _, corners in cheapest_sets(cost, sides):
            yield None
            result, steps = None, 0
            job = boundary_preview.job(xy, refs, dict(pattern='AUTO', _policy='legacy', _best_of=1, _takayama=True,
                                                      anchors=[('p', i) for i in corners]),
                                       site=('fill_cls.candidate_steps',))
            try:
                while True:
                    try:
                        next(job)
                    except StopIteration as done:
                        result = done.value
                        break
                    steps += 1
                    if steps > STEP_CAP:
                        break
                    yield None
            except (boundary_preview.PreviewError, geom.TimeUp):
                result = None
            finally:
                job.close()
            if result is None:
                continue
            positions = np.asarray(result['positions'], float)
            quads = tuple(tuple(int(v) for v in q) for q in result['quads'])
            if quads in seen:   # two corner sets can give the same layout
                continue
            seen.add(quads)
            try:
                yield from validate_steps(xy, positions, quads)
            except InvalidFill:
                continue
            mm = network_fill.metrics(positions, quads)
            if mm['qmin'] < floor:
                continue
            yield dict(candidate=dict(
                positions=positions, quads=quads, status='filled', method='cls_takayama',
                internal_seams=(), internal_counts=(), internal_paths=[], cls_sides=sides,
                cls_corners=tuple(corners), **mm))
