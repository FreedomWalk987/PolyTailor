# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Standalone, fixed-connectivity numerical solvers for Relax; no Blender imports.

Angle derivatives and least-squares rectangle proposals are implemented here
directly. They are not a reproduction of GETMe, EAS or a full Shape-Up solve.
"""
import numpy as np


def face_values(points, cells):
    p = points[cells]
    a, b = np.roll(p, 1, axis=1) - p, np.roll(p, -1, axis=1) - p
    la = np.maximum(np.linalg.norm(a, axis=2), 1e-15)
    lb = np.maximum(np.linalg.norm(b, axis=2), 1e-15)
    cosine = np.sum(a * b, axis=2) / (la * lb)
    centered = p - p.mean(axis=1, keepdims=True)
    area_vector = np.sum(np.cross(centered, np.roll(centered, -1, axis=1)), axis=1) * 0.5
    area = np.linalg.norm(area_vector, axis=1)
    lengths = np.linalg.norm(np.roll(p, -1, axis=1) - p, axis=2)
    aspect = np.max(lengths, axis=1) / np.maximum(np.min(lengths, axis=1), 1e-15)
    return p, a, b, la, lb, cosine, area_vector, area, aspect


class PatchSolver:
    def __init__(self, points, faces, fixed=None):
        self.points = np.asarray(points, dtype=float).copy()
        if not np.isfinite(self.points).all():
            raise ValueError("Mesh coordinates must be finite")
        self.fixed = np.zeros(len(points), dtype=bool) if fixed is None else np.asarray(fixed, dtype=bool).copy()
        self.groups = []
        neighbors = [set() for _ in points]
        for face in faces:
            for i, v in enumerate(face):
                u = face[(i + 1) % len(face)]
                neighbors[v].add(u)
                neighbors[u].add(v)
            if len(face) not in (3, 4):
                self.fixed[list(face)] = True
        for size in (3, 4):
            cells = np.array([f for f in faces if len(f) == size], dtype=int).reshape(-1, size)
            if not len(cells):
                continue
            data = face_values(self.points, cells)
            normals = data[6] / np.maximum(data[7][:, None], 1e-15)
            signed = np.sum(np.cross(data[2], data[1]) * normals[:, None], axis=2)
            epsilon = np.maximum(data[7] * 1e-8, 1e-18)
            # A simple concave quad can be repaired without changing connectivity.
            # Two negative corners indicate a crossed quad; do not guess its winding.
            unsafe = (data[7] <= 1e-15) | (data[3].min(axis=1) <= 1e-14)
            unsafe |= (signed < -epsilon[:, None]).sum(axis=1) >= (2 if size == 4 else 1)
            unsafe |= ~np.isfinite(data[7])
            self.fixed[cells[unsafe].ravel()] = True
            self.groups.append(dict(cells=cells, normals=normals, area=np.maximum(data[7], 1e-15), valid=~unsafe))
        self.neighbors = neighbors
        self.fixed |= np.array([not ns for ns in neighbors])
        self.edge_i = np.array([i for i, ns in enumerate(neighbors) for _ in ns], dtype=int)
        self.edge_j = np.array([j for ns in neighbors for j in sorted(ns)], dtype=int)
        self.counts = np.maximum(np.bincount(self.edge_i, minlength=len(points)), 1)
        lengths = np.linalg.norm(self.points[self.edge_i] - self.points[self.edge_j], axis=1)
        self.length = np.bincount(self.edge_i, weights=lengths, minlength=len(points)) / self.counts
