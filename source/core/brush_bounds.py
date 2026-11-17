# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Exact dirty-refitted screen bounds for local brush rays.

The hierarchy is only a conservative broad phase. Original polygons and original
vertex coordinates still determine every final native ray/visibility result.
No numerical target or projection constraint is approximated here.
"""
import numpy as np


class PlaneVisibility:
    """A slab-depth certificate, not a heuristic omission of visibility tests."""
    def visible(self, vertex):
        return not vertex.hide


class SlabBounds:
    """Monotonically conservative scalar bounds under owned coordinate writes.

    Any fixed normal defines a slab. Bounds may grow after edits; they never
    exclude an actual point by incorrectly shrinking after an extremum moves.
    The normal need not track changing face normals for the depth proof to hold.
    """
    def __init__(self,cache):
        self.normal=cache["faces"][0].normal.copy()
        self.normal.normalize()
        self.axis=np.array(self.normal,dtype=float)
        values=cache["co"]@self.axis
        self.low=float(values.min());self.high=float(values.max())
        self.scale=float(np.max(np.abs(cache["co"])))

    def changed(self,coordinates):
        if not len(coordinates):
            return
        values=np.asarray(coordinates)@self.axis
        self.low=min(self.low,float(values.min()))
        self.high=max(self.high,float(values.max()))
        self.scale=max(self.scale,float(np.max(np.abs(coordinates))))

    def world_width(self,matrix,normal,inverse_scale):
        # The stored inverse and normalized mathutils vector have float storage.
        # Bound their residual against the actual forward transform instead of
        # assuming an exact inverse, including ill-conditioned/sheared objects.
        actual_axis=matrix[:3,:3].T@np.asarray(normal,dtype=float)
        residual=actual_axis-self.axis/inverse_scale
        return (self.high-self.low)/inverse_scale+2*self.scale*float(np.abs(residual).sum())


def plane_visibility(stroke, cache, candidates):
    """Certify that no own-surface hit can exceed the native visibility bias.

    For a world unit normal n and an orthographic ray d, two points on the
    ray inside a slab of width h are separated by at most h/abs(dot(n,d)).
    Convex combinations inside every tessellated face stay inside that slab.
    This certificate is invalid with external occluders or near-grazing rays.
    """
    if stroke.context.region_data.is_perspective:
        return None
    from . import raycast
    if raycast.sources(stroke.context,visibility=True):
        return None
    if not cache["faces"]:
        return None
    slab=cache.get("_slab_bounds")
    if slab is None:
        if cache["faces"][0].normal.length_squared<1e-16:
            return None
        slab=cache["_slab_bounds"]=SlabBounds(cache)
    normal=stroke.mi.transposed().to_3x3()@slab.normal
    if normal.length_squared<1e-16:
        return None
    inverse_scale=normal.length
    normal.normalize()
    cosine=abs(normal.dot(raycast.view_forward(stroke.context)))
    if cosine<1e-4:
        return None
    matrix=np.asarray(stroke.mw,dtype=float)
    width=slab.world_width(matrix,normal,inverse_scale)
    scale=slab.scale*float(np.linalg.norm(matrix[:3,:3],ord=np.inf))+float(np.max(np.abs(matrix[:3,3])))
    rounding=64*np.finfo(np.float32).eps*(1+scale)
    view=stroke.context.region_data
    offset=getattr(stroke.context.space_data.overlay,"retopology_offset",.01)
    bias=offset+.01*view.view_distance
    return PlaneVisibility() if (width+rounding)/cosine<max(0,bias)*.25 else None


def selected_loops(loops, starts, sizes, rows):
    """Gather selected polygons in row order without scanning untouched loops."""
    counts = sizes[rows]
    offsets = np.concatenate(([0], np.cumsum(counts)))
    addresses = np.repeat(starts[rows] - offsets[:-1], counts) + np.arange(offsets[-1])
    return loops[addresses], offsets


def local_bvh(co, loops, starts, sizes, rows):
    from mathutils.bvhtree import BVHTree
    selected, offsets = selected_loops(loops, starts, sizes, rows)
    if not len(selected):
        return None
    used, inverse = np.unique(selected, return_inverse=True)
    flat = inverse.tolist()
    polygons = [flat[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    return BVHTree.FromPolygons(co[used].tolist(), polygons)


def _spread(value):
    value = value.astype(np.uint32)
    value = (value | (value << 8)) & 0x00FF00FF
    value = (value | (value << 4)) & 0x0F0F0F0F
    value = (value | (value << 2)) & 0x33333333
    return (value | (value << 1)) & 0x55555555


class BoundsHierarchy:
    """Eight-way Morton-local hierarchy; exact sparse refits, stable face order."""
    FANOUT = 8

    def __init__(self, bounds, always):
        bounds = np.asarray(bounds, dtype=np.float64)
        centers = (bounds[:, :2] + bounds[:, 2:]) * .5
        low = centers.min(axis=0)
        span = np.maximum(np.ptp(centers, axis=0), 1e-30)
        uv = np.clip((centers-low)/span*65535, 0, 65535).astype(np.uint32)
        codes = _spread(uv[:, 0]) | (_spread(uv[:, 1]) << 1)
        self.order = np.argsort(codes, kind="stable")
        self.inverse = np.empty(len(bounds), dtype=np.int64)
        self.inverse[self.order] = np.arange(len(bounds))
        self.levels = [bounds[self.order].copy()]
        while len(self.levels[-1]) > 1:
            child = self.levels[-1]
            starts = np.arange(0, len(child), self.FANOUT)
            parent = np.column_stack((np.minimum.reduceat(child[:, 0], starts),
                                      np.minimum.reduceat(child[:, 1], starts),
                                      np.maximum.reduceat(child[:, 2], starts),
                                      np.maximum.reduceat(child[:, 3], starts)))
            self.levels.append(parent)
        self.always_flags = np.asarray(always, dtype=bool).copy()
        self.always = np.flatnonzero(self.always_flags)

    def update(self, rows, bounds, always):
        positions = self.inverse[rows]
        self.levels[0][positions] = bounds
        for level in range(1, len(self.levels)):
            positions = np.unique(positions // self.FANOUT)
            child = self.levels[level-1]
            indices = positions[:, None]*self.FANOUT + np.arange(self.FANOUT)
            blocks = child[np.minimum(indices, len(child)-1)]
            self.levels[level][positions] = np.column_stack((blocks[:, :, 0].min(axis=1),
                                                            blocks[:, :, 1].min(axis=1),
                                                            blocks[:, :, 2].max(axis=1),
                                                            blocks[:, :, 3].max(axis=1)))
        if not np.array_equal(self.always_flags[rows], always):
            self.always_flags[rows] = always
            self.always = np.flatnonzero(self.always_flags)

    def query(self, x0, y0, x1, y1):
        positions = np.array([0], dtype=np.int64)
        for level in range(len(self.levels)-1, -1, -1):
            box = self.levels[level][positions]
            keep = (box[:, 2] >= x0) & (box[:, 0] <= x1) & (box[:, 3] >= y0) & (box[:, 1] <= y1)
            positions = positions[keep]
            if not len(positions):
                break
            if level:
                positions = (positions[:, None]*self.FANOUT + np.arange(self.FANOUT)).ravel()
                positions = positions[positions < len(self.levels[level-1])]
        rows = self.order[positions] if len(positions) else np.empty(0, dtype=np.int64)
        return np.union1d(rows, self.always)

    def refit_all(self,bounds,always):
        self.levels[0][:]=bounds[self.order]
        for level in range(1,len(self.levels)):
            child=self.levels[level-1]
            starts=np.arange(0,len(child),self.FANOUT)
            self.levels[level][:]=np.column_stack((np.minimum.reduceat(child[:,0],starts),
                                                   np.minimum.reduceat(child[:,1],starts),
                                                   np.maximum.reduceat(child[:,2],starts),
                                                   np.maximum.reduceat(child[:,3],starts)))
        self.always_flags[:]=always
        self.always=np.flatnonzero(always)


class ScreenProjection:
    """Retain screen coordinates and refit only faces incident to changed vertices."""
    def __init__(self, stroke, cache):
        self.cache = cache
        self.bm = stroke.bm
        self.matrix = None
        self.dimensions = None
        self.tree = None
        self.face_rows = None
        self.sizes = np.diff(np.append(cache["starts"], len(cache["loops"])))
        self.dirty = set()
        self.rebuilds = self.refits = 0

    def changed(self, rows):
        self.dirty.update(int(row) for row in rows)

    @staticmethod
    def coordinates(co, matrix, dimensions):
        clip = co @ matrix[:3, :3].T + matrix[:3, 3]
        w = co @ matrix[3, :3] + matrix[3, 3]
        front = w > 0.
        safe = np.where(front, w, 1.)
        return (dimensions[0]/2*(1+clip[:, 0]/safe),
                dimensions[1]/2*(1+clip[:, 1]/safe), front)

    def boxes(self, rows=None):
        cache = self.cache
        if rows is None:
            loops, starts = cache["loops"], cache["starts"]
        else:
            loops, offsets = selected_loops(cache["loops"], cache["starts"], self.sizes, rows)
            starts = offsets[:-1]
        sx, sy, front = self.screen
        bounds = np.column_stack((np.minimum.reduceat(sx[loops], starts),
                                  np.minimum.reduceat(sy[loops], starts),
                                  np.maximum.reduceat(sx[loops], starts),
                                  np.maximum.reduceat(sy[loops], starts)))
        return bounds, np.logical_or.reduceat(~front[loops], starts)

    def project(self, stroke):
        region, rv3d = stroke.context.region, stroke.context.region_data
        matrix = np.array(rv3d.perspective_matrix, dtype=float) @ np.array(stroke.mw, dtype=float)
        dimensions = (region.width, region.height)
        if self.tree is None or dimensions != self.dimensions or not np.array_equal(matrix, self.matrix):
            self.matrix, self.dimensions = matrix, dimensions
            self.screen = self.coordinates(self.cache["co"], matrix, dimensions)
            bounds, behind = self.boxes()
            if not np.isfinite(bounds).all():
                raise ValueError("Nonfinite screen bounds")
            self.tree = BoundsHierarchy(bounds, behind)
            self.bm.faces.index_update()
            self.face_rows = np.full(len(self.bm.faces), -1, dtype=np.int64)
            self.face_rows[[f.index for f in self.cache["faces"]]] = np.arange(len(bounds))
            self.dirty.clear()
            self.rebuilds += 1
        elif self.dirty:
            rows = np.fromiter(self.dirty, dtype=np.int64)
            if len(rows)>len(self.cache["verts"])*.2:
                self.screen=self.coordinates(self.cache["co"],matrix,dimensions)
                bounds,behind=self.boxes()
                if not np.isfinite(bounds).all():
                    raise ValueError("Nonfinite screen bounds")
                self.tree.refit_all(bounds,behind)
                self.dirty.clear();self.refits+=1
                return self.screen
            values = self.coordinates(self.cache["co"][rows], matrix, dimensions)
            for old, new in zip(self.screen, values):
                old[rows] = new
            faces = {self.face_rows[f.index] for row in rows
                     for f in self.cache["verts"][row].link_faces if not f.hide}
            face_rows = np.array(sorted(f for f in faces if f >= 0), dtype=np.int64)
            if len(face_rows):
                bounds, behind = self.boxes(face_rows)
                if not np.isfinite(bounds).all():
                    raise ValueError("Nonfinite screen bounds")
                self.tree.update(face_rows, bounds, behind)
            self.dirty.clear()
            self.refits += 1
        return self.screen
