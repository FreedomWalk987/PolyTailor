# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""PolyPen_Only 2.0.1 extrusion ladders and mitered boundary offsets.

Ported on 2026-09-30; surface queries use the PolyPen_Only core.
"""
import bmesh
from mathutils import Vector
from . import raycast

def compute_boundary_offsets(bm, M, view_dir):
    """For freshly extruded, selected boundary edges compute, in world space,
    each new vert's base position and outward offset (unit width, mitered).
    Returns None when the selection is not a fresh extrusion."""
    Mn = M.to_3x3().inverted_safe().transposed()

    sel_edges = [e for e in bm.edges if e.select]
    new_verts = {v for e in sel_edges for v in e.verts}
    if not sel_edges:
        return None

    # new vert -> old vert (the one it was extruded from; same position, not selected)
    old_of = {}
    for v in new_verts:
        for e in v.link_edges:
            o = e.other_vert(v)
            if o in new_verts: continue
            if (o.co - v.co).length < 1e-5:
                old_of[v] = o
                break
    if len(old_of) != len(new_verts):
        return None

    sel_center = sum((M @ v.co for v in new_verts), Vector()) / len(new_verts)

    # outward direction per selected edge: perpendicular to the edge, in the plane of the inner face
    edge_dirs = {}
    old_loop_edges = set()
    inner_faces = set()
    for e in sel_edges:
        a1, b1 = e.verts
        a0, b0 = old_of[a1], old_of[b1]
        e_old = bm.edges.get((a0, b0))
        pa, pb = M @ a0.co, M @ b0.co
        t = pb - pa
        if t.length < 1e-9: continue
        t.normalize()
        mid = (pa + pb) / 2
        inner = [f for f in e_old.link_faces if a1 not in f.verts and b1 not in f.verts] if e_old else []
        if e_old: old_loop_edges.add(e_old)
        if inner:
            f = inner[0]
            inner_faces.update(inner)
            nf = (Mn @ f.normal).normalized()
            d = nf.cross(t)
            ref = mid - (M @ f.calc_center_median())
        else:
            d = view_dir.cross(t)
            ref = mid - sel_center
        if d.length < 1e-9: continue
        d.normalize()
        if d.dot(ref) < 0: d = -d
        edge_dirs[e] = d

    # per vertex offset with miter so the width stays constant around corners
    base, offset, vert_edge_count = {}, {}, {}
    for v in new_verts:
        dirs = [edge_dirs[e] for e in v.link_edges if e in edge_dirs]
        vert_edge_count[v] = len([e for e in v.link_edges if e.select])
        base[v] = M @ v.co
        if not dirs:
            offset[v] = Vector((0, 0, 0))
            continue
        m = sum(dirs, Vector())
        if m.length < 1e-6: m = dirs[0].copy()
        m.normalize()
        k = min(1.0 / max(min(m.dot(d) for d in dirs), 0.2), 5.0)
        offset[v] = m * k

    return old_of, base, offset, vert_edge_count, old_loop_edges, inner_faces



# ----------------------------------------------------------------------
# Live extrusion cuts (mouse wheel while extruding).
# Like Strokes, the real geometry is rebuilt immediately: each extruded
# vert gets a chain  old -> m1 .. mk -> new  and the strip of quads between
# neighbouring chains is recreated with k cuts.
# ----------------------------------------------------------------------

def find_extrude_pairs(bm):
    """Right after an extrude: [(new_vert, old_vert)] for each selected vert with a coincident unselected neighbour."""
    pairs = []
    for v in bm.verts:
        if not v.select: continue
        for e in v.link_edges:
            o = e.other_vert(v)
            if o.select: continue
            if (o.co - v.co).length < 1e-5:
                pairs.append((v, o)); break
    return pairs


class ExtrusionLadder:
    def __init__(self, bm, pairs):
        self.bm = bm
        self.pairs = [(v, o) for v, o in pairs if v.is_valid and o.is_valid]
        self.cuts = 0
        self.mids = [[] for _ in self.pairs]
        index = {v: i for i, (v, _) in enumerate(self.pairs)}
        # rungs: neighbouring extruded verts (connected new edge) + winding of the original extrude quad
        self.rungs, self.faces = [], []
        for i, (vi, oi) in enumerate(self.pairs):
            for e in vi.link_edges:
                vj = e.other_vert(vi)
                j = index.get(vj)
                if j is None or j < i: continue
                oj = self.pairs[j][1]
                forward = True
                for f in e.link_faces:
                    fv = list(f.verts)
                    if oi in fv and oj in fv:
                        a, b = fv.index(oi), fv.index(oj)
                        forward = (b - a) % len(fv) == 1
                        self.faces.append(f)
                        break
                self.rungs.append((i, j, forward))

    def chain(self, i):
        v, o = self.pairs[i]
        return [o] + self.mids[i] + [v]

    def set_cuts(self, cuts):
        cuts = min(64, max(0, int(cuts)))
        if cuts == self.cuts or not self.pairs: return
        bm = self.bm
        faces = [f for f in self.faces if f.is_valid]
        if faces: bmesh.ops.delete(bm, geom=faces, context='FACES_ONLY')
        old_mids = [m for ms in self.mids for m in ms if m.is_valid]
        if old_mids: bmesh.ops.delete(bm, geom=old_mids, context='VERTS')
        for (v, o) in self.pairs:
            e = bm.edges.get((v, o))
            if cuts > 0 and e: bm.edges.remove(e)
        self.cuts = cuts
        self.mids = [[bm.verts.new(o.co.lerp(v.co, (k + 1) / (cuts + 1))) for k in range(cuts)] for v, o in self.pairs]
        for i in range(len(self.pairs)):
            ch = self.chain(i)
            for a, b in zip(ch, ch[1:]):
                if not bm.edges.get((a, b)): bm.edges.new((a, b))
        self.faces = []
        for (i, j, forward) in self.rungs:
            ci, cj = self.chain(i), self.chain(j)
            for s in range(cuts + 1):
                a0, b0, a1, b1 = ci[s], cj[s], ci[s + 1], cj[s + 1]
                quad = [a0, b0, b1, a1] if forward else [b0, a0, a1, b1]
                try:
                    self.faces.append(bm.faces.new(quad))
                except ValueError:
                    pass
        for m in (m for ms in self.mids for m in ms): m.select_set(False)
        for (v, _) in self.pairs: v.select_set(True)
        bm.select_flush_mode()

    def update_positions(self, context, M=None):
        if not self.cuts: return
        snap = context.scene.tool_settings.use_snap
        if snap:
            M = M or context.edit_object.matrix_world
            Mi = M.inverted()
        for i, (v, o) in enumerate(self.pairs):
            for k, m in enumerate(self.mids[i]):
                if not m.is_valid: continue
                co = o.co.lerp(v.co, (k + 1) / (self.cuts + 1))
                if snap:
                    hit = raycast.nearest_on_sources(context, M @ co)
                    if hit: co = Mi @ hit.world
                m.co = co


