# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Rigid brush groups. Mesh attributes contain IDs, never BMesh pointers.

References live for one stroke, in world space; external editing can intentionally
reshape a chain between strokes. Ordinary Pin/hidden/mask restrictions take
precedence. Surface-incompatible rigid proposals keep their previous pose.
"""
import numpy as np

VERT_LAYER = "ppo_glue"
EDGE_LAYER = "ppo_glue_edges"
PIN_LAYERS = ("ppo_pins", "hpen_pins", "polypen_only_pins")


class GlueState:
    def __init__(self, stroke):
        self.stroke = stroke
        self.groups, self.of, self.row, self.reference, self.blocked = {}, {}, {}, {}, set()
        self._bound_index, self._bound_rows, self._centered = None, (), {}
        layers = getattr(stroke.bm.verts.layers, "int", None)
        layer = layers.get(VERT_LAYER) if layers is not None else None
        if layer is None:
            return
        for v in stroke.bm.verts:
            group = int(v[layer])
            if group > 0:
                self.groups.setdefault(group, []).append(v)
        for group, vertices in list(self.groups.items()):
            if len(vertices) < 2:
                del self.groups[group]
                continue
            self.of.update((v, group) for v in vertices)
            self.row.update((v, i) for i, v in enumerate(vertices))
            self.reference[group] = np.array([tuple(stroke.mw @ v.co) for v in vertices])
            if any(self._masked(v) for v in vertices):
                self.blocked.add(group)

    def _masked(self, v):
        s, scene = self.stroke, self.stroke.scene
        if v.hide or any(f.hide for f in v.link_faces):
            return True
        for name in PIN_LAYERS:
            layer = s.bm.verts.layers.float.get(name)
            if layer is not None and v[layer] > .5:
                return True
        if getattr(s, "selection", False) and not v.select:
            return True
        selected = getattr(scene, "mask_selected", "ALL")
        if not getattr(s, "selection", False) and ((selected == "ONLY" and not v.select)
                                                  or (selected == "EXCLUDE" and v.select)):
            return True
        boundary = getattr(v, "is_boundary", None)
        if boundary is None:
            boundary = any(e.is_boundary for e in v.link_edges)
        if scene.mask_boundary == "EXCLUDE" and boundary and not s.clip.on_plane(v.co):
            return True
        if not getattr(scene, "include_corners", True) and (len(v.link_edges) == 2 or
                (len(v.link_edges) == 4 and len(v.link_faces) == 3)):
            return True
        if not getattr(scene, "include_seams", True) and any(e.seam for e in v.link_edges):
            return True
        if not getattr(scene, "include_sharps", True) and any(not e.smooth for e in v.link_edges):
            return True
        if not getattr(scene, "include_creases", True):
            cv = s.bm.verts.layers.float.get("crease_vert")
            ce = s.bm.edges.layers.float.get("crease_edge")
            if (cv is not None and v[cv] > 0) or (ce is not None and any(e[ce] > 0 for e in v.link_edges)):
                return True
        return False

    def touched(self, vertices):
        return {self.of[v] for v in vertices if v in self.of}

    def _bindings(self, index):
        # SmoothSolver owns this immutable mapping until it rebuilds the patch.
        # Reuse rows during all line-search trials; never cache coordinates.
        if index is not self._bound_index:
            self._bound_rows = tuple((group, np.array([index[v] for v in self.groups[group]
                                                       if v in index], dtype=int))
                                     for group in self.touched(index))
            self._bound_index = index
        return self._bound_rows

    def _fit(self, group, desired):
        reference = self.reference[group]
        if reference.shape != desired.shape or not np.isfinite(desired).all():
            raise ValueError("Invalid Glue proposal")
        cached = self._centered.get(group)
        if cached is None or cached[0] is not reference:
            cached = (reference, reference-reference.mean(axis=0))
            self._centered[group] = cached
        x = cached[1]
        b = desired.mean(axis=0)
        y = desired-b
        u, _singular, vt = np.linalg.svd(x.T @ y)
        correction = np.eye(3)
        correction[2, 2] = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
        return x @ (u @ correction @ vt) + b

    def expand(self, targets):
        targets = set(targets)
        for group in self.touched(targets):
            vertices = self.groups[group]
            if group in self.blocked:
                targets.difference_update(vertices)
            else:
                targets.update(vertices)
        return targets

    def relax_weights(self, index, weights, fixed):
        for group, rows in self._bindings(index):
            vertices = self.groups[group]
            if not len(rows) or not np.any(weights[rows]):
                continue
            if len(rows) != len(vertices) or group in self.blocked or np.any(fixed[rows]):
                weights[rows] = 0
            else:
                weights[rows] = float(weights[rows].max())

    def constrain(self, points, current, index, active, surface):
        """Fit complete active groups, validate constraints, then accept whole poses."""
        bindings = [(group, rows) for group, rows in self._bindings(index)
                    if len(rows) and np.any(active[rows])]
        if not bindings:
            return points
        result = points.copy()
        checks = []
        check_mask = np.zeros(len(points), dtype=bool)
        for group, rows in bindings:
            vertices = self.groups[group]
            if len(rows) != len(vertices) or group in self.blocked or not np.all(active[rows]):
                result[rows] = current[rows]
                continue
            try:
                fitted = self._fit(group, result[rows])
            except (ValueError, np.linalg.LinAlgError):
                result[rows] = current[rows]
                continue
            result[rows] = fitted
            scale = max(float(np.max(np.abs(fitted))),
                        float(np.linalg.norm(np.ptp(fitted, axis=0))), 1e-8)
            tolerance = 16*np.finfo(np.float32).eps*scale
            checks.append((rows, tolerance))
            check_mask[rows] = True
        if checks:
            checked = surface(result.copy(), check_mask)
            for rows, tolerance in checks:
                if (not np.isfinite(checked[rows]).all() or
                        np.max(np.linalg.norm(checked[rows]-result[rows], axis=1)) > tolerance):
                    result[rows] = current[rows]
        return result

    def begin_drag(self, vertices):
        groups = self.touched(vertices)
        self._drag_local = {g: [v.co.copy() for v in self.groups[g]] for g in groups}
        return {g: np.array([tuple(self.stroke.mw @ v.co) for v in self.groups[g]]) for g in groups}

    def correct_drag(self, before, written, surface, journal=None):
        """Whole-chain Tweak/selection moves, including members outside the brush."""
        if not before:
            return written
        from mathutils import Vector
        s = self.stroke
        changed = set(written)
        for group, current in before.items():
            vertices = self.groups[group]
            driven = np.array([v in changed for v in vertices])
            if not driven.any():
                continue
            if group in self.blocked:
                fitted = current
            else:
                desired = np.array([tuple(s.mw @ v.co) for v in vertices])
                # A partial brush still moves the whole chain, rather than dividing
                # its displacement by the total number of unbrushed members.
                desired[~driven] = current[~driven] + (desired[driven]-current[driven]).mean(axis=0)
                try:
                    fitted = self._fit(group, desired)
                    checked = np.array([tuple(surface(v, Vector(p))) for v, p in zip(vertices, fitted)])
                    scale = max(float(np.abs(fitted).max()), float(np.linalg.norm(np.ptp(fitted, axis=0))), 1e-8)
                    if (not np.isfinite(checked).all() or np.max(np.linalg.norm(checked-fitted, axis=1)) >
                            16*np.finfo(np.float32).eps*scale):
                        fitted = current
                except (ValueError, np.linalg.LinAlgError):
                    fitted = current
            keep_current = np.array_equal(fitted, current)
            for v, old_local, old, point in zip(vertices, self._drag_local[group], current, fitted):
                if journal is not None and v not in journal and np.any(point != old):
                    journal[v] = old_local.copy()
                v.co = old_local.copy() if keep_current else s.mi @ Vector(point)
                changed.discard(v)
                if not np.array_equal(v.co[:], old_local[:]):
                    changed.add(v)
        return sorted(changed, key=lambda v: v.index)
