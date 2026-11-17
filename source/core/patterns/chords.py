# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
# GMSH pattern tables: vertex layouts follow Gmsh contrib/QuadMeshingTools/qmtQuadCavityRemeshing.cpp
# (M. Reberol). Gmsh: Copyright (C) 1997-2025 C. Geuzaine and J.-F. Remacle, GNU GPL-2.0-or-later;
# see licenses/THIRD_PARTY_NOTICES.md.
"""Pattern + chord quad meshing for a fixed boundary (no bpy).

Idea learned from Gmsh's quasi-structured quad pipeline (M. Reberol,
contrib/QuadMeshingTools/qmtQuadCavityRemeshing.cpp, GPL) and Takayama et al.
2014; own implementation:

* a pattern is a coarse quad mesh. Boundary vertices used by exactly one coarse
  quad are corners; corners split each boundary loop into sides. A loop without
  corners (a hole, a disk) is a single side.
* a chord is a strip of coarse quads linked through opposite edges. Each chord
  gets an integer x >= 1: every coarse edge it crosses is cut into x segments.
* for a region with the same sides, every side's point count must equal the sum
  of the x of the chords crossing it: a small integer linear system.
* the solution refines the pattern: each coarse quad becomes an x_a by x_b grid.
  Chords become whole rows of parallel quads; size transitions are chords that
  turn back (U-turn) inside, as a modeller would do.
"""

from collections import OrderedDict
from functools import lru_cache


def takayama_enabled(settings=None):
    # An explicit '_takayama' wins over the module switch (the legacy incumbent
    # search turns it off).
    value = (settings or {}).get('_takayama')
    return True if value is None else bool(value)


class PatternError(Exception):
    pass


class Pattern:
    def __init__(self, name, quads, corners=None):
        self.name = name
        self.quads = [tuple(q) for q in quads]
        self.nv = 1 + max(v for q in self.quads for v in q)
        # edges and the quads using them (directed as in each quad)
        self.e2q = {}
        for qi, q in enumerate(self.quads):
            for k in range(4):
                a, b = q[k], q[(k + 1) % 4]
                self.e2q.setdefault(frozenset((a, b)), []).append((qi, k))
        if any(len(v) > 2 for v in self.e2q.values()):
            raise PatternError("%s: non-manifold" % name)
        self.edges = list(self.e2q)
        self.eid = {e: i for i, e in enumerate(self.edges)}
        # chords: union of opposite edges in every quad
        parent = list(range(len(self.edges)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for q in self.quads:
            for k in (0, 1):
                a = self.eid[frozenset((q[k], q[(k + 1) % 4]))]
                b = self.eid[frozenset((q[k + 2], q[(k + 3) % 4]))]
                parent[find(a)] = find(b)
        roots = sorted({find(i) for i in range(len(self.edges))})
        rid = {r: i for i, r in enumerate(roots)}
        self.chord = [rid[find(i)] for i in range(len(self.edges))]
        self.nchords = len(roots)
        # boundary: directed edges as they appear in their single quad
        nxt = {}
        for e, uses in self.e2q.items():
            if len(uses) == 1:
                qi, k = uses[0]
                q = self.quads[qi]
                nxt[q[k]] = q[(k + 1) % 4]
        valence = {}
        for q in self.quads:
            for v in q:
                valence[v] = valence.get(v, 0) + 1
        self.corner = {v for v in nxt if valence[v] == 1}
        if corners is not None:
            self.corner=set(corners)
        self.loops = []          # each: list of sides; side = list of coarse vertices
        seen = set()
        for start in sorted(nxt):
            if start in seen:
                continue
            loop = [start]
            seen.add(start)
            v = nxt[start]
            while v != start:
                loop.append(v)
                seen.add(v)
                v = nxt[v]
            cs = [i for i, v in enumerate(loop) if v in self.corner]
            if not cs:
                self.loops.append([loop + [loop[0]]])         # one closed side
                continue
            loop = loop[cs[0]:] + loop[:cs[0]]
            cs = [i for i, v in enumerate(loop) if v in self.corner]
            sides = []
            for a, b in zip(cs, cs[1:] + [len(loop)]):
                sides.append(loop[a:b + 1] if b < len(loop) else loop[a:] + [loop[0]])
            self.loops.append(sides)
        # outer loop = the one with corners (or the longest), holes after it
        self.loops.sort(key=lambda L: (-(len(L) > 1 or L[0][0] in self.corner), -sum(len(s) for s in L)))
        self.irregular = sum(1 for v, c in valence.items()
                             if (v not in nxt and c != 4) or (v in nxt and v not in self.corner and c != 2))
        if corners is not None:
            self.irregular+=sum(valence[v]!=1 for v in self.corner)
            # Preserve the explicit side-zero convention, including mirrored bases.
            start=corners[0]
            self.loops[0]=self.loops[0][next(i for i,s in enumerate(self.loops[0]) if s[0]==start):]+self.loops[0][:next(i for i,s in enumerate(self.loops[0]) if s[0]==start)]

    def side_chords(self, side):
        return [self.chord[self.eid[frozenset((a, b))]] for a, b in zip(side, side[1:])]


REFINE_MEMO_MAX = 8192
_REFINE_MEMO = OrderedDict()


def refine(pattern, x, outer_sides, hole_loops, first_new):
    """Fine quads. outer_sides: actual vertex lists per pattern outer side (each
    from its first corner to the next, len = count + 1). hole_loops: actual closed
    vertex lists per pattern hole (first vertex repeated at the end).
    Returns (quads, count) with new vertex ids from first_new.

    Memoised: refine is a pure function of (pattern, x, side index lists, hole lists, first_new) with integer output,
    and many calls repeat earlier arguments (also across outlines, since side lists are index ranges). Results are kept
    in a bounded insertion-ordered dict keyed by the pattern object identity (library patterns are lru_cached
    singletons), the integer values of x, the side and hole lists and first_new; a hit returns fresh copies of the quad
    lists. Exceptions are not cached. A hit returns ids as stored: the values equal a fresh call's, but an id that came
    in as a NumPy integer in one call and as a Python int in another keeps the first call's type.
    A result is stored as a tuple of int tuples (the garbage collector stops tracking those; as lists the memo held
    millions of tracked objects and every full collection paused the UI for up to ~0.9 s); a hit rebuilds the lists
    from the same id objects."""
    key = (id(pattern), pattern, tuple(int(v) for v in x), tuple(tuple(int(v) for v in s) for s in outer_sides),
           tuple(tuple(int(v) for v in h) for h in hole_loops), int(first_new))
    hit = _REFINE_MEMO.get(key)
    if hit is not None:
        _REFINE_MEMO.move_to_end(key)
        return [list(q) for q in hit[0]], hit[1]
    quads, count = _refine_compute(pattern, x, outer_sides, hole_loops, first_new)
    _REFINE_MEMO[key] = (tuple(tuple(q) for q in quads), count)   # untracked by the GC (see the docstring)
    if len(_REFINE_MEMO) > REFINE_MEMO_MAX:
        _REFINE_MEMO.popitem(last=False)
    return quads, count


def _refine_compute(pattern, x, outer_sides, hole_loops, first_new):
    if getattr(pattern, 'takayama', False):
        return _refine_takayama(pattern, x, outer_sides, first_new)
    return _refine_generic(pattern, x, outer_sides, hole_loops, first_new)


def _refine_generic(pattern, x, outer_sides, hole_loops, first_new):
    """Every vertex id, every quad and their order; ids taken from the caller's side lists are passed through untouched.
    A coarse quad's grid is built row by row: row 0 = L03, row na = L12, interior rows = L01[i], new ids row-major,
    L32[i]. Grids with na == 0 or nb == 0 produce no quads and no new ids."""
    count = [first_new]

    def new():
        count[0] += 1
        return count[0] - 1

    epath = {}
    cmap = {}
    chord, eid = pattern.chord, pattern.eid
    psides = list(pattern.loops[0]) + [L[0] for L in pattern.loops[1:]]
    actual = list(outer_sides) + list(hole_loops)
    for side, verts in zip(psides, actual):
        pos = 0
        for a, b in zip(side, side[1:]):
            n = x[chord[eid[frozenset((a, b))]]]
            seg = list(verts[pos:pos + n + 1])
            if len(seg) != n + 1:
                raise PatternError("side length mismatch")
            epath[(a, b)] = seg
            epath[(b, a)] = seg[::-1]
            cmap.setdefault(a, seg[0])
            cmap.setdefault(b, seg[-1])
            pos += n
    for v in range(pattern.nv):
        if v not in cmap:
            cmap[v] = new()

    def path(a, b):
        seg = epath.get((a, b))
        if seg is None:
            n = x[chord[eid[frozenset((a, b))]]]
            seg = [cmap[a]] + [new() for _ in range(n - 1)] + [cmap[b]]
            epath[(a, b)] = seg
            epath[(b, a)] = seg[::-1]
        return seg

    quads = []
    for c0, c1, c2, c3 in pattern.quads:
        L01, L12, L32, L03 = path(c0, c1), path(c1, c2), path(c3, c2), path(c0, c3)
        na, nb = len(L01) - 1, len(L03) - 1
        if na <= 0 or nb <= 0:
            continue
        start = count[0]; w = nb - 1
        count[0] += (na - 1) * w
        prev = L03
        for i in range(1, na + 1):
            if i == na:
                row = L12
            else:
                s = start + (i - 1) * w
                row = [L01[i], *range(s, s + w), L32[i]]
            quads += [[a0, b0, b1, a1] for a0, b0, b1, a1 in zip(prev, row, row[1:], prev[1:])]
            prev = row
    return quads, count[0]


def _refine_takayama(fam, parameters, outer_sides, first_new):
    """A Takayama family: its base pattern refined with the flow-chord widths, padding rings added side by side, then
    every generated boundary id remapped onto the actual side ids."""
    import numpy as np
    padding = parameters[:fam.n]; flows = parameters[fam.n:]
    widths = [1] * fam.nchords
    for chord, extra in zip(fam.flow_chords, flows):
        widths[chord] += int(extra)
    counts = np.asarray(fam.base_counts) + sum((np.asarray(c) * x for c, x in zip(fam.flow_columns, flows)), start=np.zeros(fam.n, dtype=np.int64))
    sides = []; next_id = 0
    for count in counts:
        sides.append(list(range(next_id, next_id + int(count) + 1))); next_id += int(count)
    sides[-1][-1] = 0
    quads, total = _refine_generic(fam.base, widths, sides, [], next_id)
    nside = fam.n
    for side, amount in enumerate(padding):
        for _ in range(int(amount)):
            inner = sides[side]; outer = list(range(total, total + len(inner))); total += len(inner)
            quads += [[i0, o0, o1, i1] for i0, o0, o1, i1 in zip(inner, outer, outer[1:], inner[1:])]
            sides[(side - 1) % nside] = sides[(side - 1) % nside] + [outer[0]]
            sides[(side + 1) % nside] = [outer[-1]] + sides[(side + 1) % nside]
            sides[side] = outer
    remap = {}
    for actual, generated in zip(outer_sides, sides):
        if len(actual) != len(generated):
            raise ValueError('Takayama padding side mismatch')
        for old, new in zip(generated, actual):
            if old in remap and remap[old] != new:
                raise ValueError('Inconsistent boundary corner')
            remap[old] = new
    next_vertex = first_new
    for vertex in range(total):
        if vertex not in remap:
            remap[vertex] = next_vertex; next_vertex += 1
    g = remap.__getitem__
    return [[g(a), g(b), g(c), g(d)] for a, b, c, d in quads], next_vertex


# ------------------------------------------------------------------ library
# Gmsh's patterns (vertex ids as in qmtQuadCavityRemeshing.cpp), named by shape
GMSH = {
    "quad": [(0, 1, 2, 3)],
    "triangle": [(0, 1, 6, 5), (1, 2, 3, 6), (3, 4, 5, 6)],
    "pentagon": [(0, 1, 2, 3), (0, 5, 4, 1), (0, 7, 6, 5), (0, 9, 8, 7), (0, 3, 10, 9)],
    "quad_diag35": [(0, 1, 2, 3), (0, 5, 4, 1), (0, 7, 6, 5), (0, 9, 8, 7), (0, 3, 10, 9),
                    (8, 9, 10, 11)],
    "quad_aligned35": [(0, 1, 2, 3), (0, 5, 4, 1), (0, 7, 6, 5), (0, 9, 8, 7), (0, 3, 10, 9),
                       (9, 10, 12, 11), (3, 13, 12, 10), (8, 9, 11, 14), (2, 15, 13, 3)],
    "quad_uturn": [(0, 1, 2, 3), (0, 5, 4, 1), (0, 7, 6, 5), (0, 9, 8, 7), (0, 3, 10, 9),
                   (9, 10, 12, 11), (9, 11, 14, 13), (8, 9, 13, 15), (6, 7, 8, 15)],
    "two_corners": [(0, 1, 2, 3), (0, 5, 4, 1), (0, 3, 6, 5), (4, 5, 6, 7)],
    "one_corner": [(0, 1, 6, 5), (1, 2, 7, 6), (2, 3, 8, 7), (3, 4, 9, 8), (4, 5, 6, 9),
                   (6, 7, 8, 9)],
    "disk": [(0, 1, 6, 5), (1, 2, 7, 6), (2, 3, 4, 7), (0, 5, 4, 3), (4, 5, 6, 7)],
}


_LIBRARY_TABLES = [None]   # {False: patterns, True: patterns} returned by library() while set (push_library)


def push_library(tables):
    """library(t) returns tables[bool(t)] until pop_library(token); returns the token."""
    old = _LIBRARY_TABLES[0]
    _LIBRARY_TABLES[0] = tables
    return old


def pop_library(token):
    """Put back the library tables that push_library returned."""
    _LIBRARY_TABLES[0] = token


def library(takayama=False):
    tables = _LIBRARY_TABLES[0]
    if tables is not None:
        return tables[bool(takayama)]
    return _library_cached(takayama)


def build_library(takayama=False):
    pats = []
    for name, quads in GMSH.items():
        for mirror in (False, True):
            qs = [q[::-1] for q in quads] if mirror else quads
            try:
                pats.append(Pattern(name + ("~" if mirror else ""), qs))
            except PatternError:
                pass
    if takayama:
        from .takayama import library as takayama_library
        pats.extend(takayama_library())
    return pats


_library_cached = lru_cache(maxsize=1)(build_library)


def clear_caches():
    """Empty the refine memo and the library cache (add-on unregister)."""
    _REFINE_MEMO.clear()
    _library_cached.cache_clear()


MIN_QUALITY = 0.15      # smallest corner sine allowed in a result
MEAN_QUALITY = 0.6


def corner_candidates(ext, manual=None, sharp=0.6, max_sides=5):
    """Corner sets to try: the sharp vertices (exterior angle > `sharp` rad), then
    with the least sharp ones demoted one by one down to 0 corners. Manual corners
    (indices) are always kept. Yields (corners, demoted_sharpness)."""
    n = len(ext)
    manual = set(manual or ())
    sharp_v = sorted((k for k in range(n) if ext[k] > sharp or k in manual),
                     key=lambda k: (k not in manual, -ext[k]))
    seen = set()
    for keep in range(len(sharp_v), -1, -1):
        cs = tuple(sorted(sharp_v[:keep]))
        if len(cs) > max_sides or cs in seen:
            continue
        if keep < len(manual):
            break
        seen.add(cs)
        demoted = float(sum(max(0.0, ext[k]) for k in sharp_v[keep:]))
        yield list(cs), demoted
