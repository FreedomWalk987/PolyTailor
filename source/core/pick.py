# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Find the mesh element under the mouse in the edit mesh."""

from itertools import chain

import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import intersect_point_line

from . import raycast
from .. import prefs


# Fraction of the view depth a source surface may sit in front of a vertex while the
# vertex still counts as visible (see MeshView._compute_visible).
VISIBILITY_DEPTH_TOLERANCE = 0.01
# Pixel offset of the extra visibility probes around a vertex whose centre is occluded.
VISIBILITY_PROBE_PX = 3


class MeshView:
    """Screen-space snapshot of the edit mesh used for picking."""

    def __init__(self, context, faces=None, lazy=False, share_key=None, bvh=None, indices_valid=False):
        """faces: only these faces occlude (callers pass every visible face whose
        screen area can reach the vertices they will test); default all visible faces.
        lazy: build the occlusion data of the whole mesh only if a test needs it (see
        limit_to); the mesh must not change while the view is used.
        share_key: the whole-mesh arrays (screen positions, face vertex lists) are kept
        for the next view built with the same key (the caller's mesh + view state), so
        hovering over an unchanged mesh does not re-read it. Only numpy arrays are kept,
        never BMesh elements."""
        self.context = context
        self.obj = context.edit_object
        self.mw = self.obj.matrix_world
        self.bm = bmesh.from_edit_mesh(self.obj.data)
        self.bm.verts.ensure_lookup_table()
        self.bm.faces.ensure_lookup_table()
        if not indices_valid:
            self.bm.verts.index_update()
        self._region = None        # (x0, y0, x1, y1): pixels whose rays use _sub
        self._sub = None           # (bvh, faces) of the faces that can cover the region
        self._arrays = None
        self._share = share_key
        if share_key is None or _shared.get("key") != share_key:
            _shared.clear()
            _shared["key"] = share_key
        if faces is not None:
            self.faces = faces
            self.bvh = bvh if bvh is not None else _build_bvh(faces)
        elif not lazy:
            self.faces = [f for f in self.bm.faces if not f.hide]
            self.bvh = BVHTree.FromPolygons([v.co for v in self.bm.verts],
                                           [[v.index for v in f.verts] for f in self.faces])
        else:
            self.faces = None
            self.bvh = None
        self.srcs = raycast.sources(context)
        self.occluders = raycast.prepare(raycast.sources(context, visibility=True))
        self._mi = self.mw.inverted()
        self._view = raycast.View(context)   # the view does not change while a view lives
        self._mi3 = self._mi.to_3x3()
        self._screen = {}
        self._visible = {}

    # --- whole-mesh arrays (large meshes: no per-vertex Python loop over everything) ---

    def arrays(self):
        """Screen x, y, in-front and hidden flags of every vertex (bm.verts order),
        computed once with numpy. Only used to narrow down candidates: the final tests
        use the exact per-vertex values (screen / visible)."""
        if self._arrays is None and self._share is not None and "arrays" in _shared:
            self._arrays = _shared["arrays"]
        if self._arrays is None:
            verts = self.bm.verts
            n = len(verts)
            co = np.fromiter(chain.from_iterable(v.co for v in verts), dtype=np.float64, count=3 * n).reshape(n, 3)
            hide = np.fromiter((v.hide for v in verts), dtype=bool, count=n)
            v = self._view
            pm = np.array(v.persmat, dtype=np.float64) @ np.array(self.mw, dtype=np.float64)
            clip = co @ pm[:3, :3].T + pm[:3, 3]
            w = co @ pm[3, :3] + pm[3, 3]
            front = w > 0.0
            safe = np.where(front, w, 1.0)
            sx = v.width / 2.0 * (1 + clip[:, 0] / safe)
            sy = v.height / 2.0 * (1 + clip[:, 1] / safe)
            self._arrays = (sx, sy, front, hide, co)
            if self._share is not None:
                _shared["arrays"] = self._arrays
        return self._arrays

    def limit_to(self, x0, y0, x1, y1):
        """Rays through pixels inside this box are tested only against the faces whose
        screen box overlaps it (the only faces such a ray can hit, so the result is the
        same as with the whole mesh). Rays elsewhere use the whole mesh."""
        sx, sy, front, _hide, co = self.arrays()
        topo = _shared.get("faces") if self._share is not None else None
        if topo is None:
            self.bm.faces.index_update()
            visible = [f for f in self.bm.faces if not f.hide]
            sizes = np.fromiter((len(f.verts) for f in visible), dtype=np.int64, count=len(visible))
            loops = np.fromiter(chain.from_iterable((v.index for v in f.verts) for f in visible),
                                dtype=np.int64, count=int(sizes.sum()))
            topo = (np.fromiter((f.index for f in visible), dtype=np.int64, count=len(visible)),
                    loops, np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.int64))
            if self._share is not None:
                _shared["faces"] = topo
        face_ids, loops, starts = topo
        if not len(face_ids):
            self._region, self._sub = (x0, y0, x1, y1), (None, [])
            return
        keep = (np.logical_or.reduceat(~front[loops], starts) |
                ((np.maximum.reduceat(sx[loops], starts) >= x0) & (np.minimum.reduceat(sx[loops], starts) <= x1) &
                 (np.maximum.reduceat(sy[loops], starts) >= y0) & (np.minimum.reduceat(sy[loops], starts) <= y1)))
        faces = self.bm.faces
        sub = [faces[int(face_ids[i])] for i in np.nonzero(keep)[0]]
        self._region, self._sub = (x0, y0, x1, y1), (bvh_from_loops(co, loops, starts, keep) if sub else None, sub)

    def _bvh_for(self, xy):
        """(bvh, faces) for a ray through pixel xy."""
        r = self._region
        if r is not None and xy is not None and r[0] <= xy[0] <= r[2] and r[1] <= xy[1] <= r[3]:
            return self._sub
        if self.bvh is None:
            self.faces = [f for f in self.bm.faces if not f.hide]
            self.bvh = BVHTree.FromPolygons([v.co for v in self.bm.verts],
                                           [[v.index for v in f.verts] for f in self.faces])
        return self.bvh, self.faces

    def screen(self, bmv):
        p = self._screen.get(bmv)
        if p is None and bmv not in self._screen:
            p = self._view.to_screen(self.mw @ bmv.co)
            self._screen[bmv] = p
        return p

    def visible(self, bmv):
        vis = self._visible.get(bmv)
        if vis is None:
            vis = self._compute_visible(bmv)
            self._visible[bmv] = vis
        return vis

    def _compute_visible(self, bmv):
        """Is this vertex visible (not hidden behind the edit mesh or a source)?

        Vertex dots must stay visible wherever Blender draws the vertex itself:
        - Rays are aimed at the vertex's exact 3D position, not re-cast from its
          rounded screen pixel: at grazing angles that rounding made the ray hit the
          vertex's own face first, so far-edge pins vanished on a plain plane.
        - A hit on a face that uses this vertex never occludes it.
        - Occluders must be in front by more than a bias that scales with view depth
          (fixed 0.01 hid pins behind neighbouring bumps of detailed sources).
        - If the centre is still blocked, a probe a few pixels beside the vertex (the
          dot is several pixels wide) may show it.
        """
        if bmv.hide:
            return False
        xy = self.screen(bmv)
        if xy is None:
            return False
        world = self.mw @ bmv.co
        origin, _ = self._view.ray(xy)
        rv3d = self.context.region_data
        dist = (world - origin).length
        depth = dist if getattr(rv3d, "is_perspective", True) else getattr(rv3d, "view_distance", dist)
        offset = getattr(getattr(self.context.space_data, "overlay", None), "retopology_offset", 0.01)
        bias = offset + VISIBILITY_DEPTH_TOLERANCE * depth
        if not self._occluded(world, origin, bias, bmv, xy):
            return True
        for dx, dy in ((VISIBILITY_PROBE_PX, 0), (-VISIBILITY_PROBE_PX, 0),
                       (0, VISIBILITY_PROBE_PX), (0, -VISIBILITY_PROBE_PX)):
            probe_xy = xy + Vector((dx, dy))
            probe = self._view.on_view_plane(probe_xy, world)
            if probe is None:
                continue
            probe_origin, _ = self._view.ray(probe_xy)
            if not self._occluded(probe, probe_origin, bias, bmv, probe_xy):
                return True
        return False

    def _occluded(self, point, origin, bias, bmv=None, xy=None):
        """Something (edit mesh or source) lies in front of point by more than bias.
        xy: the pixel the ray goes through (selects the occlusion data, see limit_to)."""
        direction = point - origin
        dist = direction.length
        if dist < 1e-9:
            return False
        direction /= dist
        bvh, faces = self._bvh_for(xy)
        co, idx = None, None
        if bvh is not None:
            co, _, idx, _ = bvh.ray_cast(self._mi @ origin, (self._mi3 @ direction).normalized())
        if co is not None and idx is not None and 0 <= idx < len(faces):
            face = faces[idx]
            if bmv is None or bmv not in face.verts:
                if (self.mw @ co - origin).length < dist - bias:
                    return True
        hit = raycast.ray_cast_sources(self.context, origin, direction, self.occluders)
        return hit is not None and (hit.world - origin).length + bias < dist

    def face_under(self, xy):
        origin, direction = raycast.ray(self.context, xy)
        mi = self.mw.inverted()
        bvh, faces = self._bvh_for(xy)
        if bvh is None:
            return None
        co, _, idx, _ = bvh.ray_cast(mi @ origin, (mi.to_3x3() @ direction).normalized())
        if co is None:
            return None
        bmf = faces[idx]
        if raycast.is_occluded(self.context, self.mw @ co, self.occluders):
            return None
        return bmf

    def vert_under(self, xy, radius):
        best, best_d = None, radius * radius
        # numpy pre-selection with a 1 px margin, then the exact test in vertex order
        sx, sy, front, hide, _co = self.arrays()
        near = front & ~hide & ((sx - xy[0]) ** 2 + (sy - xy[1]) ** 2 < (radius + 1.0) ** 2)
        verts = self.bm.verts
        for i in np.nonzero(near)[0]:
            bmv = verts[int(i)]
            if bmv.hide:
                continue
            p = self.screen(bmv)
            if p is None:
                continue
            d = (p - xy).length_squared
            if d < best_d and self.visible(bmv):
                best, best_d = bmv, d
        return best

    def edge_under(self, xy, radius):
        best, best_d = None, radius
        # numpy pre-selection with a 1 px margin, then the exact test in edge order
        sx, sy, front, _hide, _co = self.arrays()
        edges = self.bm.edges
        n = len(edges)
        if not n:
            return None
        ends = np.fromiter(chain.from_iterable((e.verts[0].index, e.verts[1].index) for e in edges),
                           dtype=np.int64, count=2 * n).reshape(n, 2)
        ax, ay, bx, by = sx[ends[:, 0]], sy[ends[:, 0]], sx[ends[:, 1]], sy[ends[:, 1]]
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy
        t = np.clip(((xy[0] - ax) * dx + (xy[1] - ay) * dy) / np.where(l2 > 0, l2, 1.0), 0.0, 1.0)
        d2 = (ax + dx * t - xy[0]) ** 2 + (ay + dy * t - xy[1]) ** 2
        near = front[ends[:, 0]] & front[ends[:, 1]] & (d2 < (radius + 1.0) ** 2)
        edges.ensure_lookup_table()
        for i in np.nonzero(near)[0]:
            bme = edges[int(i)]
            if bme.hide:
                continue
            a, b = (self.screen(v) for v in bme.verts)
            if a is None or b is None:
                continue
            pt, t = intersect_point_line(xy, a, b)
            t = min(max(t, 0.0), 1.0)
            d = (a.lerp(b, t) - xy).length
            if d < best_d and any(self.visible(v) for v in bme.verts):
                best, best_d = bme, d
        return best


_shared = {"key": None}   # see MeshView(share_key=...): numpy arrays only


def reset_shared():
    """Forget the shared arrays (the evaluated sources were replaced)."""
    _shared.clear()
    _shared["key"] = None


def bvh_from_loops(co, loops, starts, keep):
    """BVH of the faces selected by keep, from flat arrays: co (vertex positions), loops
    (vertex indices of all faces, face after face) and starts (first loop of each face).
    Same polygons and positions as _build_bvh, without per-vertex Python work."""
    sizes = np.diff(np.append(starts, len(loops)))
    sel = loops[np.repeat(keep, sizes)]
    used, local = np.unique(sel, return_inverse=True)
    flat = local.tolist()
    offsets = np.concatenate(([0], np.cumsum(sizes[keep]))).tolist()
    polys = [flat[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    return BVHTree.FromPolygons(co[used].tolist(), polys)


def _build_bvh(faces):
    local = {}
    polys = [[local.setdefault(v, len(local)) for v in f.verts] for f in faces]
    return BVHTree.FromPolygons([v.co for v in local], polys)


def element_under(context, xy):
    """(view, element) under xy honoring the mesh select mode; element may be None."""
    view = MeshView(context, lazy=True)
    radius = prefs.get(context).pick_radius * prefs.ui_scale(context)
    xy = Vector(xy)
    margin = radius + VISIBILITY_PROBE_PX + 2
    view.limit_to(xy.x - margin, xy.y - margin, xy.x + margin, xy.y + margin)
    vmode, emode, fmode = context.tool_settings.mesh_select_mode
    elem = None
    if vmode:
        elem = view.vert_under(xy, radius)
    if elem is None and emode:
        elem = view.edge_under(xy, radius)
    if elem is None:
        face = view.face_under(xy)
        # A face counts in face mode, or when it is part of the selection.
        if face is not None and (fmode or face.select):
            elem = face
    return view, elem
