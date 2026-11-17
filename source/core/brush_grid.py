# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Sorted spatial buckets with an exact, cell-local relocation overlay.

No per-point Python insertion during construction, and no ever-growing global
visited-row union during queries. Every broad-phase result is still checked by
the original world-distance and brush-mask code.
"""
import numpy as np


class MovingGrid:
    BIAS = 1 << 20
    LIMIT = 1 << 21

    def __init__(self, world, hidden, width):
        self.width = float(width)
        self.origin = np.min(world, axis=0)
        self.scale = float(np.max(np.abs(world)))
        self.valid = np.isfinite(world).all() and np.isfinite(width) and width > 0
        self.overridden = np.zeros(len(world), dtype=bool)
        self.relocated = {}
        self.buckets = {}
        self.original = np.zeros(len(world), dtype=np.uint64)
        if not self.valid:
            return
        cells = self.cells(world)
        keys = self.keys(cells)
        if keys is None:
            self.valid = False
            return
        self.original[:] = keys
        rows = np.flatnonzero(~np.asarray(hidden, dtype=bool))
        order = np.argsort(keys[rows], kind="stable")
        self.rows = rows[order]
        self.sorted_keys = keys[self.rows]

    def cells(self, world):
        return np.floor((np.asarray(world)-self.origin)/self.width)

    @classmethod
    def keys(cls, cells):
        biased = cells + cls.BIAS
        if not np.isfinite(biased).all() or np.any(biased < 0) or np.any(biased >= cls.LIMIT):
            return None
        xyz = biased.astype(np.uint64)
        return xyz[..., 0] | (xyz[..., 1] << np.uint64(21)) | (xyz[..., 2] << np.uint64(42))

    def note(self, rows, world):
        if not self.valid or not len(rows):
            return
        keys = self.keys(self.cells(world))
        if keys is None:
            self.valid = False
            return
        changed=(keys!=self.original[rows])|self.overridden[rows]
        rows=np.asarray(rows)[changed]
        keys=keys[changed]
        for row, key in zip(rows, keys):
            row, key = int(row), int(key)
            old = self.relocated.get(row)
            if old == key:
                continue
            if old is not None:
                bucket = self.buckets[old]
                bucket.discard(row)
                if not bucket:
                    del self.buckets[old]
            if key == self.original[row]:
                self.relocated.pop(row, None)
                self.overridden[row] = False
            else:
                self.relocated[row] = key
                self.overridden[row] = True
                self.buckets.setdefault(key, set()).add(row)

    def query(self, center, radius):
        if not self.valid or not np.isfinite(center).all() or not np.isfinite(radius) or radius < 0:
            return None
        center = np.asarray(center, dtype=float)
        # Conservative coordinate-transform and quantization slack. Final tests
        # are unchanged; padding may add candidates, never edits by itself.
        pad = 64*np.finfo(float).eps*(1+self.scale+float(np.max(np.abs(center)))+radius)
        low = self.cells(center-radius-pad)
        high = self.cells(center+radius+pad)
        count = high-low+1
        if np.any(count <= 0) or np.prod(count) > 4096:
            return None
        ranges = [np.arange(int(a), int(b)+1) for a,b in zip(low, high)]
        cells = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1,3)
        keys = self.keys(cells)
        if keys is None:
            return None
        lo = np.searchsorted(self.sorted_keys, keys, side="left")
        hi = np.searchsorted(self.sorted_keys, keys, side="right")
        arrays = [self.rows[a:b] for a,b in zip(lo,hi) if b>a]
        rows = np.concatenate(arrays) if arrays else np.empty(0,dtype=np.int64)
        rows = rows[~self.overridden[rows]]
        extra = [row for key in keys for row in self.buckets.get(int(key), ())]
        if extra:
            rows = np.concatenate((rows, np.array(extra,dtype=np.int64)))
        return np.unique(rows)
