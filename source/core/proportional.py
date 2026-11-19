# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Frozen proportional-edit distances for a single surface move.

Build only when enabled; mouse movement visits the current influence set, not
the full mesh. Radius/falloff changes reuse distances and original coordinates.
"""

from bisect import bisect_left
import heapq
import math

from mathutils.kdtree import KDTree


FALLOFFS = ("SMOOTH", "SPHERE", "ROOT", "INVERSE_SQUARE", "SHARP", "LINEAR", "CONSTANT", "RANDOM")


def falloff(distance, radius, mode, seed=0):
    if distance >= radius or not math.isfinite(distance):
        return 0.0
    t = max(0.0, 1.0-distance/max(radius, 1e-12))
    if mode == "CONSTANT":
        return 1.0
    if mode == "LINEAR":
        return t
    if mode == "SHARP":
        return t*t
    if mode == "ROOT":
        return math.sqrt(t)
    if mode == "SPHERE":
        return math.sqrt(max(0.0, 2*t-t*t))
    if mode == "INVERSE_SQUARE":
        return t*(2-t)
    if mode == "RANDOM":
        # Stable for this vertex throughout movement and radius changes.
        value = ((seed+1)*0x9e3779b1) & 0xffffffff
        value = ((value ^ (value >> 16))*0x85ebca6b) & 0xffffffff
        value ^= value >> 13
        return t*value/0xffffffff
    return t*t*(3-2*t)


class Influence:
    def __init__(self, bm, mw, selected, forward, connected=False, projected=False):
        self.originals = {v: selected[v].copy() if v in selected else v.co.copy()
                          for v in bm.verts if not v.hide}
        verts = list(self.originals)
        index = {v: i for i, v in enumerate(verts)}
        world = [mw @ self.originals[v] for v in verts]
        # Connected distances follow visible mesh edges and never cross islands.
        if connected:
            adj = [[] for _ in verts]
            for edge in bm.edges:
                a, b = edge.verts
                if edge.hide or a not in index or b not in index:
                    continue
                i, j = index[a], index[b]
                length = (world[i]-world[j]).length
                adj[i].append((j, length)); adj[j].append((i, length))
            distances = [math.inf]*len(verts)
            frontier = []
            for v in selected:
                i = index[v]; distances[i] = 0.0
                frontier.append((0.0, i))
            heapq.heapify(frontier)
            while frontier:
                distance, i = heapq.heappop(frontier)
                if distance != distances[i]:
                    continue
                for j, length in adj[i]:
                    candidate = distance+length
                    if candidate < distances[j]:
                        distances[j] = candidate
                        heapq.heappush(frontier, (candidate, j))
        else:
            origin = world[index[next(iter(selected))]]
            points = [p-origin for p in world]
            if projected:
                points = [p-forward*p.dot(forward) for p in points]
            tree = KDTree(len(selected))
            for i, v in enumerate(selected):
                tree.insert(points[index[v]], i)
            tree.balance()
            distances = [tree.find(p)[2] for p in points]
        rows = sorted((distances[i], i, v) for i, v in enumerate(verts)
                      if v not in selected and math.isfinite(distances[i]))
        self.rows = rows
        self.distances = [row[0] for row in rows]
        self.weights = {}

    def refresh(self, radius, mode):
        self.weights = {v: falloff(distance, radius, mode, i)
                        for distance, i, v in self.rows[:bisect_left(self.distances, radius)]}

