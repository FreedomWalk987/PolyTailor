# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Adapt Relax Lab's bounded corner-Jacobian solver to PolyTailor strokes.

Brush collection owns pins, selection and feature masks. Boundary vertices are
not automatically fixed: PolyTailor's existing Move / Slide / Lock setting owns
that policy. Every projection/clip happens inside the solver's acceptance test.
Only local incident faces are solved; zero-weight support vertices remain fixed.
"""
import bmesh
import numpy as np
from mathutils import Vector

from . import raycast
from .relax_lab.robust import RobustSolver
from .relax_lab.targets import smooth_strip_targets


def unsafe_vertex(vertex):
    """Protect wires, nonmanifold edges and disconnected fans, allowing open borders."""
    manifold=getattr(vertex,"is_manifold",None)
    if manifold is not None:
        return not manifold
    faces = set(vertex.link_faces)
    if not faces or any(len(e.link_faces) != 1 and len(e.link_faces) != 2
                        for e in vertex.link_edges):
        return True
    remaining = faces.copy()
    stack = [remaining.pop()]
    while stack:
        face = stack.pop()
        for edge in face.edges:
            if vertex not in edge.verts:
                continue
            for other in edge.link_faces:
                if other in remaining:
                    remaining.remove(other)
                    stack.append(other)
    return bool(remaining)


class _Points:
    """The part of RobustSolver the brush adapter uses, for Grid Relax."""

    def __init__(self, points, fixed):
        self.points = np.asarray(points, dtype=float)
        if not np.isfinite(self.points).all():
            raise ValueError("Mesh coordinates must be finite")   # as RobustSolver: the caller skips the patch
        self.fixed = np.asarray(fixed, dtype=bool)
        self._last_eval = None

    def reset_stroke(self):
        self._last_eval = None   # coordinates were re-synced: no earlier evaluation is reused

    def drop_last_eval(self):
        self._last_eval = None


class SmoothSolver:
    """Retain the brush interface; use the supplied Robust Relax objective."""

    def __init__(self, stroke):
        self.stroke = stroke
        self.bm = stroke.bm
        self.em = stroke.obj.data
        self.mw, self.mi = stroke.mw, stroke.mi
        self.relax = stroke.scene
        self.clip, self.srcs = stroke.clip, stroke.srcs
        self.prev = {}
        self.self_surface = None
        self._solver = None
        self._verts = []
        self._index = {}
        self._faces = set()
        self._boundary = []
        self._supported = set()
        self._fixed_cache = {}
        self.glue = getattr(stroke, "glue", None)
        self._trees = None      # source surfaces as BVH trees, built on first use in a stroke
        mw, mi = np.asarray(self.mw, dtype=float), np.asarray(self.mi, dtype=float)
        self._world_linear, self._local_linear = mw[:3, :3].T.copy(), mi[:3, :3].T.copy()
        self._world_offset = mw[:3, 3].copy() if mw.shape[1] == 4 else np.zeros(3)
        self._local_offset = mi[:3, 3].copy() if mi.shape[1] == 4 else np.zeros(3)

    def _on_boundary(self, vertex):
        native=getattr(vertex,"is_boundary",None)
        boundary=any(e.is_boundary for e in vertex.link_edges) if native is None else native
        return boundary and not self.clip.on_plane(vertex.co)

    def _coordinates(self):
        local = np.asarray([v.co[:] for v in self._verts], dtype=np.float64).reshape(-1, 3)
        return np.ascontiguousarray(local @ self._world_linear + self._world_offset)

    def _prepare(self, targets):
        # Reuse retained aspect targets and L-BFGS history while the footprint is
        # fully supported. A halo vertex can only activate if all its faces exist.
        if self._solver is not None and targets.issubset(self._supported):
            if getattr(self.stroke,"_direct_cache",False):
                return True
            current = self._coordinates()
            if not np.array_equal(current, self._solver.points):
                # Blender stores float coordinates. Re-sync even their rounding
                # rather than retaining a fictitious accepted double position.
                self._solver.points = current
                self._solver.reset_stroke()
            return True
        touched = {f for v in targets for f in v.link_faces}
        # A one-ring halo supports nearby cursor motion without rebuilding the
        # numerical solver and its retained aspect targets on each footprint edge.
        halo = {v for f in touched for v in f.verts}
        faces = {f for v in halo for f in v.link_faces}
        vertices = {v for f in faces for v in f.verts}
        self._verts = sorted(vertices, key=lambda v: v.index)
        self._index = {v: i for i, v in enumerate(self._verts)}
        self._faces = faces
        cells = [[self._index[v] for v in f.verts] for f in sorted(faces, key=lambda f: f.index)]
        self._cells = cells
        for v in self._verts:
            if v not in self._fixed_cache:
                self._fixed_cache[v] = v.hide or any(f.hide for f in v.link_faces) or unsafe_vertex(v)
        fixed = [self._fixed_cache[v] for v in self._verts]
        try:
            if self._grid():
                # Grid Relax needs only positions and fixed flags, not the Robust model.
                self._solver = _Points(self._coordinates(), fixed)
                self._topology = None
            else:
                self._solver = RobustSolver(self._coordinates(), cells, fixed)
                if getattr(self.relax, "relax_even_spacing", False):
                    smooth_strip_targets(self._solver)
        except ValueError:
            # Entirely collapsed/nonfinite patches have no trustworthy scale.
            self._solver = None
            return False
        # Do not apply solver.boundary: that includes artificial patch borders
        # and would silently turn the Move/Slide settings into Lock.
        self._boundary = np.array([self._on_boundary(v) for v in self._verts], dtype=bool)
        self._supported = {v for v in self._verts if all(f in faces for f in v.link_faces)}
        return True

    def _project(self, context, points, active, mode):
        result = self._surface_project(context, points, active, mode)
        if self.glue is not None and self.glue.groups:
            result = self.glue.constrain(result, self._solver.points, self._index, active,
                                        lambda p, rows: self._surface_project(context, p, rows, mode))
        return result

    def _surface_project(self, context, points, active, mode):
        # Leave truly unchanged proposals untouched; no source lookup or matrix
        # round trip is needed for them. MirrorClip without planes is a no-op.
        changed = active & np.any(points != self._solver.points, axis=1)
        indices = np.flatnonzero(changed)
        has_clip = bool(getattr(self.clip, "planes", True))
        slide = mode == "SLIDE"
        has_source = bool(self.srcs)
        if (not len(indices) or
                (not has_source and self.self_surface is None and not has_clip and not slide)):
            return points
        result = points.copy()
        for i in indices:
            world = Vector(points[i])
            hit = self._nearest_source(context, world) if has_source else None
            if hit:
                world = hit.world
                if not has_clip and not (slide and self._boundary[i]):
                    result[i] = tuple(world)
                    continue
            p = self.mi @ world
            if hit is None and self.self_surface is not None:
                near = self.self_surface.find_nearest(p)[0]
                if near is not None:
                    p = near
            if slide and self._boundary[i]:
                p = self.stroke.snap_to_boundary(p)
            if has_clip:
                p = self.clip.apply(self.prev[self._verts[i]], p)
            result[i] = tuple(self.mw @ p)
        return result

    def _nearest_source(self, context, world):
        """raycast.nearest_on_sources with each source's BVH tree built once per stroke:
        the same closest surface point, without rebuilding the lookup on every query
        (the solver's line search projects every active vertex several times per step)."""
        trees = self._source_trees(context)
        if all(tree is None for _obj, tree, _mw, _mi in trees):
            return raycast.nearest_on_sources(context, world, self.srcs)
        if world is None or not world.length_squared < float("inf"):
            return None
        best, best_dist = None, float("inf")
        for obj, tree, mw, mi in trees:
            if tree is None:       # no tree for this object: the original per-object query
                hit = raycast.nearest_on_sources(context, world, [obj])
                if hit is not None and (hit.world - world).length < best_dist:
                    best, best_dist = hit, (hit.world - world).length
                continue
            co = tree.find_nearest(mi @ world)[0]
            if co is None:
                continue
            co_w = mw @ co
            dist = (co_w - world).length
            if dist < best_dist:
                best, best_dist = raycast.Hit(co_w, obj), dist
        return best

    def _source_trees(self, context):
        """(object, BVH tree in its local space or None, matrix_world, inverse) per source."""
        if self._trees is None:
            try:
                from mathutils.bvhtree import BVHTree
                depsgraph = context.view_layer.depsgraph
            except (ImportError, AttributeError):
                BVHTree = depsgraph = None
            trees = []
            for obj in self.srcs:
                try:
                    tree = BVHTree.FromObject(obj, depsgraph) if BVHTree is not None else None
                    mw = obj.matrix_world.copy()
                    mi = mw.inverted()
                except Exception:    # no lookup data: the original per-object query is used
                    tree = mw = mi = None
                trees.append((obj, tree, mw, mi))
            self._trees = trees
        return self._trees

    def _update_normals(self, moved):
        total = len(self.bm.faces) if hasattr(self.bm, "faces") else 0
        if total > 4000:
            faces = {f for i in moved for f in self._verts[i].link_faces}
            if len(faces) * 4 < total:
                for face in faces:
                    face.normal_update()
                # Adjacent vertices read all their incident face normals, so the
                # unchanged neighboring faces contribute exactly as before.
                for vertex in {v for f in faces for v in f.verts}:
                    vertex.normal_update()
                return
        self.bm.normal_update()

    def finish(self):
        """Lifecycle compatibility: each accepted step already fully synchronizes."""
        try:
            if getattr(self.stroke,"feedback",None) is not None:
                bmesh.update_edit_mesh(self.em,loop_triangles=True,destructive=False)
            seal=getattr(self.stroke,"seal_cache",None)
            if seal is not None:
                seal()
        finally:
            if hasattr(self.stroke,"close_feedback"):
                self.stroke.close_feedback()

    def _weights(self, verts, vert_strength, brush_strength):
        """Falloff weight per patch row (0 = fixed), or None when nothing can move."""
        targets = {v for v in verts if not v.hide and vert_strength.get(v, 0) > 0}
        if self.glue is not None and self.glue.groups:
            targets = self.glue.expand(targets)
        if not targets or not self._prepare(targets):
            return None
        solver = self._solver
        mode = self.relax.mask_boundary
        weights = np.zeros(len(self._verts), dtype=float)
        indices = np.fromiter((self._index[v] for v in targets), dtype=np.int64)
        influence = np.fromiter((vert_strength.get(v, 0) for v in targets), dtype=float)
        # Collection already multiplied by strength; normalize the complete
        # influence vector once rather than dispatching np.clip per vertex.
        weights[indices] = np.clip(influence / brush_strength, 0, 1)
        weights[solver.fixed] = 0
        if mode == "EXCLUDE":
            weights[self._boundary] = 0
        if self.glue is not None and self.glue.groups:
            fixed = solver.fixed | (self._boundary if mode == "EXCLUDE" else False)
            self.glue.relax_weights(self._index, weights, fixed)
        for i in np.flatnonzero(weights):
            v = self._verts[i]
            if v not in self.prev:
                self.prev[v] = v.co.copy()
        return weights if np.any(weights) else None

    def relax_step(self, context, verts, vert_strength, radius3D, time_delta, brush_strength, pressure):
        if brush_strength <= 0 or pressure <= 0 or time_delta <= 0:
            return
        weights = self._weights(verts, vert_strength, brush_strength)
        if weights is None:
            return
        solver = self._solver
        mode = self.relax.mask_boundary
        before = solver.points.copy()
        project = lambda p, active: self._project(context, p, active, mode)
        passes = max(1, int(self.relax.smooth_iterations))
        if isinstance(solver, _Points):
            from . import relax_forces
            if self._topology is None:
                self._topology = relax_forces.Topology(self._cells, len(self._verts), self._boundary)
            # radius3D is in world units like the solver points
            scale = brush_strength * pressure * min(time_delta, 0.05) / max(radius3D, 1e-12)
            solver.points = relax_forces.relax(before, self._topology, weights, scale, radius3D,
                                               project, passes)
        else:
            solver.step(weights, time_delta, strength=brush_strength, pressure=pressure,
                        passes=passes, project=project)
        self._commit(context, np.flatnonzero(np.any(solver.points != before, axis=1) & (weights > 0)))

    def _grid(self):
        return bool(getattr(self.stroke, "grid", False))

    def _commit(self, context, moved):
        """Write accepted solver rows back to the mesh and refresh the display."""
        solver = self._solver
        if not len(moved):
            return
        local = solver.points[moved] @ self._local_linear + self._local_offset
        for i, point in zip(moved, local):
            self._verts[i].co = Vector(point)
        # Re-read only changed coordinates to account for BMesh float storage.
        stored = np.asarray([self._verts[i].co[:] for i in moved], dtype=float)
        note=getattr(self.stroke,"note_written",None)
        if note is not None:
            note([self._verts[i] for i in moved],stored)
        solver.points[moved] = stored @ self._world_linear + self._world_offset
        # Retain the quasi-Newton history, but never reuse gradients evaluated
        # before BMesh rounded the accepted coordinates to its storage precision.
        solver.drop_last_eval()
        from .face_attributes import apply
        apply(self.stroke, [self._verts[i] for i in moved])
        self._update_normals(moved)
        # Keep display/mesh picking current, including concave or warped quads.
        feedback=getattr(self.stroke,"feedback",None)
        if feedback is not None:
            feedback.update([self._verts[i] for i in moved])
        else:
            bmesh.update_edit_mesh(self.em, loop_triangles=True, destructive=False)
        if context.area is not None:
            context.area.tag_redraw()
