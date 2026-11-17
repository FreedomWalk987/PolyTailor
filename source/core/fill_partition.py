# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Concavity-guided, cooperative quad decomposition for Fill's AUTO mode.

Only array planning happens here. Outer vertices never move; both children own
identical cut vertex IDs. Every merge passes the same whole-disk validator as
a template fill. Manual template patterns do not enter this module.
"""

import numpy as np
from . import boundary_preview, fill_lattice
from .patterns import geom, chords
from .patterns.validate import validate_steps, InvalidFill

MAX_DEPTH = (3, 6)     # shallow search first; deeper only when that finds nothing
MAX_VISITS = (64, 128)  # pieces visited per pass
MAX_CANDIDATES = 12
REFLEX = -.15
PIECE_SECONDS = .04    # CPU for one piece's template search before trying other cuts


class PartitionError(ValueError):
    pass


def eligible(points, settings):
    if settings.get('anchors') or settings.get('_corners') is not None:
        return False
    p = np.asarray(points, float)
    if len(p) < 8:
        return False
    turns = geom.turning(p)
    return bool(np.min(turns) < -.35 and -np.minimum(turns, 0).sum() > .55)


def _cross(a, b):
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]


def _inside(p, point):
    b = np.roll(p, -1, axis=0)
    mask = (p[:, 1] > point[1]) != (b[:, 1] > point[1])
    a, b = p[mask], b[mask]
    if not len(a):
        return False
    x = a[:, 0] + (b[:, 0]-a[:, 0])*(point[1]-a[:, 1])/(b[:, 1]-a[:, 1])
    return bool(np.count_nonzero(point[0] < x) % 2)


def _visible_row(p, i):
    """bool array over j: the chord a=p[i] -> b=p[j] is visible inside the outline, for every j in one array pass (the
    values for j == i or adjacent j are computed too; callers never use them). Visible = (1) no proper or touching
    crossing with any polygon edge not incident to i or j (c1*c2 <= 1e-14 and c3*c4 <= 1e-14), and (2) the 5 points
    a + t*(b-a), t in (.01, .25, .5, .75, .99), all inside (even-odd ray test)."""
    n = len(p)
    a = p[i]
    B = p                                   # b for every j
    ab = B - a                              # (n, 2)
    c = p; d = np.roll(p, -1, axis=0)       # edge k: c[k] -> d[k]
    ca = c - a; da = d - a                  # (n_edges, 2), the same for every j
    # c1 = cross(ab, c-a), c2 = cross(ab, d-a): (n_j, n_edges)
    c1 = ab[:, None, 0]*ca[None, :, 1] - ab[:, None, 1]*ca[None, :, 0]
    c2 = ab[:, None, 0]*da[None, :, 1] - ab[:, None, 1]*da[None, :, 0]
    dc = d - c
    c3 = _cross(dc, a - c)                  # (n_edges,), independent of j
    bc = B[:, None, :] - c[None, :, :]      # b - c: (n_j, n_edges, 2)
    c4 = dc[None, :, 0]*bc[:, :, 1] - dc[None, :, 1]*bc[:, :, 0]
    hit = (c1*c2 <= 1e-14) & ((c3[None, :]*c4) <= 1e-14)
    ids = np.arange(n)
    excl = ((ids[None, :] == i) | (ids[None, :] == ids[:, None]) | (((ids + 1) % n)[None, :] == i)
            | (((ids + 1) % n)[None, :] == ids[:, None]))
    crossing = np.any(hit & ~excl, axis=1)
    # inside tests of the 5 sample points, for every j
    ok = ~crossing
    pb = np.roll(p, -1, axis=0)
    for t in (.01, .25, .5, .75, .99):
        js = np.flatnonzero(ok)
        if not len(js):
            break
        pt = a + t*ab[js]                   # (k, 2): a + t*ab, row by row
        py = pt[:, 1][:, None]
        mask = (p[None, :, 1] > py) != (pb[None, :, 1] > py)                      # (k, n_edges)
        with np.errstate(divide='ignore', invalid='ignore'):
            x = p[None, :, 0] + (pb[None, :, 0]-p[None, :, 0])*(py - p[None, :, 1])/(pb[None, :, 1]-p[None, :, 1])
        cnt = np.count_nonzero(mask & (pt[:, 0][:, None] < x), axis=1)
        ok[js] = (cnt % 2).astype(bool)
    return ok


def _seam(p, i, j, count):
    return [(1-t)*p[i]+t*p[j] for t in np.arange(1, count)/count]


def _lattice_count(p, i, j, distance, count):
    """Seam point count near `count` (same parity) for which a piece of the cut closes
    on the lattice, so that piece becomes a pole-free grid; both pieces if possible."""
    n = len(p)
    best, best_closed = count, 0
    for c in sorted(range(max(1, count-4), count+5, 2), key=lambda c: abs(c-count)):
        seam = _seam(p, i, j, c)
        left = np.vstack([p[[(i+t) % n for t in range(distance+1)]]] + ([np.asarray(seam[::-1])] if seam else []))
        right = np.vstack([p[[(j+t) % n for t in range(n-distance+1)]]] + ([np.asarray(seam)] if seam else []))
        closed = sum(fill_lattice.closes(piece) for piece in (left, right))
        if closed > best_closed:
            best, best_closed = c, closed
            if closed == 2:
                break
    return best


def _split_steps(p):
    """Candidate cuts from each reflex vertex i, best first. The visibility of every j from i is computed in one array
    pass (_visible_row) at the first j that needs it."""
    n = len(p)
    reflex = np.flatnonzero(geom.turning(p) < REFLEX)
    h = max(float(np.median(np.linalg.norm(np.roll(p, -1, axis=0)-p, axis=1))), 1e-12)
    # seam counts that close a piece on the lattice only matter for right-angled outlines
    rectilinear = fill_lattice.quarter_turns(p) is not None
    candidates = []
    for i in reflex:
        row = None
        for j in range(n):
            if j % 8 == 0:
                geom.check_time()
                yield None
            distance = (j-i) % n
            if distance < 2 or n-distance < 2:
                continue
            if row is None:
                row = _visible_row(p, i)
            if not row[j]:
                continue
            length = float(np.linalg.norm(p[i]-p[j]))
            count = max(1, int(round(length/h)))
            if (distance+count) % 2:
                count += 1
            if rectilinear:
                count = _lattice_count(p, i, j, distance, count)
            if n+count-1 > boundary_preview.MAX_BOUNDARY:
                continue
            tangent = (p[j]-p[i])/max(length, 1e-12)
            ea, eb = p[i]-p[(i-1) % n], p[(i+1) % n]-p[i]
            alignment = max(abs(float(tangent@ea))/max(np.linalg.norm(ea), 1e-12),
                            abs(float(tangent@eb))/max(np.linalg.norm(eb), 1e-12))
            left = p[[(i+t) % n for t in range(distance+1)]]
            right = p[[(j+t) % n for t in range(n-distance+1)]]
            remaining = (np.count_nonzero(geom.turning(left) < REFLEX)
                         + np.count_nonzero(geom.turning(right) < REFLEX))
            if remaining >= len(reflex):
                continue
            score = length*(1+4*(1-alignment))*(1+.25*abs(distance-n/2)/n)
            candidates.append((score, int(i), int(j), count))
    return sorted(candidates)[:MAX_CANDIDATES]


def _capped_steps(job, seconds, settings=None):
    """Run a piece's search with its own CPU allowance (the outer budget still
    applies). Running out only fails this piece, so other cuts get a chance."""
    spent = 0.0
    try:
        while True:
            started = geom.clock().monotonic()
            try:
                with geom.time_budget(min(seconds-spent, geom.time_left())):
                    next(job)
            except StopIteration as done:
                return done.value
            except geom.TimeUp:
                saved = boundary_preview.fill_selection.best_checkpoint(settings or {})
                if saved is not None:
                    return saved
                geom.check_time()
                raise PartitionError('Piece search took too long')
            spent += geom.clock().monotonic()-started
            if spent > seconds:
                saved = boundary_preview.fill_selection.best_checkpoint(settings or {})
                if saved is not None:
                    return saved
                raise PartitionError('Piece search took too long')
            yield None
    finally:
        job.close()


def _capped_site(seconds):
    """The memo site of a search advanced by _capped_steps (boundary_preview.job(..., site=...)): this loop derives
    each step's slice from `seconds` and `spent` (0.0 at the first step)."""
    return ('fill_partition._capped_steps', repr(seconds), repr(0.0))


def _single_quad(p):
    """A four-point piece is one quad or nothing: no template search needed."""
    quads = [(0, 1, 2, 3)]
    quality = geom.quad_quality(p, quads)
    if quality.min() < chords.MIN_QUALITY or quality.mean() < chords.MEAN_QUALITY:
        raise PartitionError('Piece is not a usable quad')
    return np.asarray(p, float), quads, 0


def _validate_steps(boundary, positions, quads):
    if len(quads) > boundary_preview.MAX_QUADS:
        raise PartitionError('Partition is too dense')
    try:
        metrics = yield from validate_steps(boundary, positions, quads)
    except InvalidFill as exc:
        raise PartitionError(str(exc)) from exc
    quality = geom.quad_quality(positions, quads)
    if quality.min() < chords.MIN_QUALITY or quality.mean() < chords.MEAN_QUALITY:
        raise PartitionError('Partition quality is too low')
    return metrics


def solve_steps(points, takayama=False, policy_settings=None):
    boundary = np.asarray(points, float)
    n = len(boundary)
    origin = boundary.mean(axis=0)
    scale = float(np.ptp(boundary, axis=0).max())
    if n < 4 or n % 2 or not np.isfinite(boundary).all() or scale <= 1e-15:
        raise PartitionError('Invalid partition boundary')
    unit = (boundary-origin)/scale
    if np.min(np.linalg.norm(np.roll(unit, -1, axis=0)-unit, axis=1)) <= 1e-10:
        raise PartitionError('Invalid partition boundary')
    visits = [0]
    limit = [MAX_DEPTH[0]]
    k_pass = [0]
    failed = set()      # pieces already known not to fill, by their exact points

    def rec(p, depth):
        key = np.asarray(p, float).tobytes()
        if key in failed:
            raise PartitionError('Piece already failed')
        try:
            return (yield from piece(p, depth))
        except PartitionError:
            failed.add(key)
            raise

    def piece(p, depth):
        geom.check_time()
        visits[0] += 1
        if visits[0] > MAX_VISITS[k_pass[0]]:
            raise PartitionError('Partition search limit reached')
        yield None
        if len(p) == 4:
            return _single_quad(p)
        # a piece whose lattice walk closes is a pole-free grid; no further cuts
        grid = yield from fill_lattice.plan_steps(p)
        if grid is not None:
            return grid['positions'], [tuple(q) for q in grid['quads']], 0
        if np.min(geom.turning(p)) >= REFLEX or depth >= limit[0]:
            settings = dict({'_takayama': True} if takayama else {}, **(policy_settings or {}))
            if boundary_preview.fill_selection.recovery_enabled(settings):
                settings['_validated_pool'] = []
            try:
                result = yield from _capped_steps(
                    boundary_preview.job(p, [('p', i) for i in range(len(p))],
                                         settings, site=_capped_site(PIECE_SECONDS)), PIECE_SECONDS, settings)
            except boundary_preview.PreviewError as exc:
                raise PartitionError(str(exc)) from exc
            return result['positions'], [tuple(q) for q in result['quads']], 0
        candidates = yield from _split_steps(p)
        for _score, i, j, count in candidates:
            geom.check_time()
            seam = [(1-t)*p[i]+t*p[j] for t in np.arange(1, count)/count]
            parent = np.vstack((p, np.asarray(seam).reshape(-1, 2)))
            seam_ids = list(range(len(p), len(parent)))
            left = [(i+t) % len(p) for t in range((j-i) % len(p)+1)] + seam_ids[::-1]
            right = [(j+t) % len(p) for t in range((i-j) % len(p)+1)] + seam_ids
            try:
                lp, lq, lc = yield from rec(parent[left], depth+1)
                rp, rq, rc = yield from rec(parent[right], depth+1)
                positions, quads = list(parent), []
                for ids, pos, qs in ((left, lp, lq), (right, rp, rq)):
                    remap = ids + list(range(len(positions), len(positions)+len(pos)-len(ids)))
                    positions.extend(pos[len(ids):])
                    quads.extend(tuple(remap[v] for v in q) for q in qs)
                positions = np.asarray(positions)
                yield from _validate_steps(p, positions, quads)
            except PartitionError:
                if visits[0] >= MAX_VISITS[k_pass[0]]:
                    break
                continue
            return positions, quads, 1+lc+rc
        raise PartitionError('No valid concavity partition')

    for k, limit[0] in enumerate(MAX_DEPTH):
        visits[0] = 0
        k_pass[0] = k
        failed.clear()          # a piece that failed at a shallower limit may split deeper
        try:
            positions, quads, cuts = yield from rec(unit, 0)
            break
        except PartitionError:
            if k == len(MAX_DEPTH)-1:
                raise
    metrics = yield from _validate_steps(unit, positions, quads)
    positions = positions*scale+origin
    positions[:n] = boundary
    return dict(positions=positions, quads=tuple(quads), pattern='partition',
                rotation=0, corners=(), anchors=[], metrics=metrics,
                partition_cuts=cuts)
