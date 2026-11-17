# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Cooperative integer chord search derived from ShoeQuad's Fraction solver."""
import itertools
import math
from collections import OrderedDict
from fractions import Fraction
import numpy as np
from .. import memo_jobs

_cache = OrderedDict()
_CURRENT = [_cache]    # [the solver cache in use]: _cache, or a deterministic step's own (push_cache)


def current_cache():
    """The solver cache in use."""
    return _CURRENT[0]


def push_cache(cache):
    """Make `cache` the solver cache until pop_cache(token); returns the token."""
    old = _CURRENT[0]
    _CURRENT[0] = cache
    return old


def pop_cache(token):
    """Put back the solver cache that push_cache returned."""
    _CURRENT[0] = token


def cache_touch(key):
    """Mark a cached solve as recently used."""
    _CURRENT[0].move_to_end(key)


def cache_insert(key, value):
    """Store a solve (the cache keeps the 512 most recent)."""
    cache = _CURRENT[0]
    cache[key] = value
    while len(cache) > 512:
        cache.popitem(last=False)


def clear_caches():
    """Empty the solver cache (add-on unregister)."""
    _CURRENT[0].clear()
    _cache.clear()


@memo_jobs.recorded_solver
def solve_steps(pattern, counts):
    if getattr(pattern, 'takayama', False):
        return (yield from pattern.solve_steps(counts))
    key = (pattern.name, tuple(counts))
    cache = _CURRENT[0]
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    sides = pattern.loops[0]
    m = pattern.nchords
    rows = []
    for side, count in zip(sides, counts):
        row = [0] * m
        for c in pattern.side_chords(side):
            row[c] += 1
        if sum(row) > count:
            return None
        rows.append([Fraction(v) for v in row] + [Fraction(count)])
    # A closed interior chord does not appear on any boundary side. The discarded
    # implementation maximized it to the largest boundary count, causing hundreds
    # of unnecessary cells in otherwise small disk patches. Use one transition
    # row unless additional interior layers are explicitly requested in a future
    # interface; the boundary already determines the central grid resolution.
    interior_count=1
    for col in range(m):
        if not any(row[col] for row in rows):
            rows.append([Fraction(int(c==col)) for c in range(m)]+[Fraction(interior_count)])
    pivots, r = [], 0
    for col in range(m):
        p = next((i for i in range(r, len(rows)) if rows[i][col]), None)
        if p is None:
            continue
        rows[r], rows[p] = rows[p], rows[r]
        divisor = rows[r][col]
        rows[r] = [v / divisor for v in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][col]:
                factor = rows[i][col]
                rows[i] = [a - factor*b for a,b in zip(rows[i],rows[r])]
        pivots.append(col)
        r += 1
        if r == len(rows):
            break
    if any(rows[i][m] for i in range(r,len(rows))):
        return None
    free = [c for c in range(m) if c not in pivots]
    hi = max(counts)
    ranges = [range(1, hi+1)] * len(free)
    if hi ** len(free) > 300000:
        guess = max(1, hi // max(1,len(free)+1))
        span = max(3, int(round(300000 ** (1/len(free))/2)))
        ranges = [range(max(1,guess-span),min(hi,guess+span)+1)] * len(free)
    coefficients, constants, denominators = [], [], []
    for row in rows[:r]:
        denominator = math.lcm(*(v.denominator for v in row))
        coefficients.append([int(row[c]*denominator) for c in free])
        constants.append(int(row[m]*denominator))
        denominators.append(denominator)
    coefficients = np.asarray(coefficients,dtype=np.int64).reshape(r,len(free))
    constants = np.asarray(constants,dtype=np.int64)
    denominators = np.asarray(denominators,dtype=np.int64)
    combinations = itertools.product(*ranges)
    best, best_score = None, -float('inf')
    while True:
        batch = list(itertools.islice(combinations,1024))
        if not batch:
            break
        values = np.asarray(batch,dtype=np.int64).reshape(len(batch),len(free))
        numerators = constants - values @ coefficients.T
        valid = np.all((numerators >= denominators) & (numerators % denominators == 0),axis=1)
        if np.any(valid):
            x = np.ones((int(valid.sum()),m),dtype=np.int64)
            x[:,free] = values[valid]
            x[:,pivots] = numerators[valid] // denominators
            scores = np.sqrt(x).sum(axis=1)
            index = int(np.argmax(scores))
            if scores[index] > best_score:
                best_score = scores[index]
                best = tuple(int(v) for v in x[index])
        yield None
    cache_insert(key, best)
    return best
