# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Per-stroke conservative spatial candidates; contains no BMesh references."""
import numpy as np


class StrokeIndex:
    def __init__(self, world, hidden, tree_factory):
        world = np.asarray(world, dtype=np.float64)
        valid = ~np.asarray(hidden, dtype=bool) & np.isfinite(world).all(axis=1)
        rows = np.flatnonzero(valid)
        self.count = len(world)
        self.touched = set()
        self.scale = float(np.max(np.abs(world[rows]))) if len(rows) else 1.0
        self.tree = tree_factory(len(rows)) if len(rows) else None
        if self.tree is not None:
            for i in rows:
                self.tree.insert(tuple(world[i]), int(i))
            self.tree.balance()

    def note(self, rows):
        # A vertex that has moved may no longer be near its original tree entry.
        # Keep every potentially moved row in the conservative candidate union.
        self.touched.update(int(i) for i in rows)

    def query(self, center, radius):
        center = np.asarray(center, dtype=np.float64)
        if (not np.isfinite(center).all() or not np.isfinite(radius)
                or len(self.touched) > max(64, self.count // 3)):
            return None  # old full-array path, never approximate/truncate a mask
        if self.tree is None:
            return np.array(sorted(self.touched), dtype=np.int64)
        # Blender's KD tree stores floats. Pad the broad phase for insertion,
        # query-centre/radius and distance rounding at the current world scale.
        pad = 32 * np.finfo(np.float32).eps * (1 + self.scale + float(np.max(np.abs(center))) + abs(radius))
        found = {int(hit[1]) for hit in self.tree.find_range(tuple(center), radius + pad)}
        found.update(self.touched)
        return np.array(sorted(found), dtype=np.int64)
