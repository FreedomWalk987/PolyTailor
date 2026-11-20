# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Mesh transactions and screen-distance merging for modal tools."""

import bmesh
import numpy as np
import bpy

from .pick import MeshView



def _shape_signature(bm, data):
    """the edit mesh's shape layers and its Key (identity + block names)."""
    key = data.shape_keys
    return (tuple(bm.verts.layers.shape.keys()),
            (key.as_pointer(), tuple(block.name for block in key.key_blocks)) if key is not None else None)


RESTORE_TOLERANCE = 1e-7   # "still holds what the tool wrote" (per coordinate)


def _close(co, ref):
    return all(abs(a - b) <= RESTORE_TOLERANCE for a, b in zip(co, ref))


def restore_written(bm, rows):
    """the Esc restore of a coordinate-only tool (Drag, Move, Tweak, Relax): never a full-snapshot
    rollback. rows: (handle, index or None, original co, co the tool last wrote). A vertex goes back to its original
    only while it still holds what the tool last wrote (RESTORE_TOLERANCE); a stale handle falls back to the vertex at
    its recorded index when that one holds it. Any other vertex was edited by someone else and is left alone.
    Returns how many were left alone."""
    seq = bm.verts
    table = False
    skipped = 0
    for v, index, original, expected in rows:
        target = v if v.is_valid and _close(v.co, expected) else None
        if target is None and index is not None and index >= 0:
            if not table:
                seq.ensure_lookup_table()
                table = True
            if index < len(seq) and _close(seq[index].co, expected):
                target = seq[index]
        if target is None:
            if not (v.is_valid and _close(v.co, original)):
                skipped += 1
            continue
        if not _close(target.co, original):
            target.co = original
    return skipped


class CoordinateJournal:
    """The vertices a coordinate-only tool changed, keyed by element identity (BMVert handles held
    while the tool runs): [index, original co, co last written]. `v in journal` / `journal[v] = co` record an original
    (drag_vertices, glue). stale(): before an event's writes; restore(): Esc (restore_written)."""
    def __init__(self, bm):
        self.bm = bm
        self.rows = {}

    def __contains__(self, v):
        return v in self.rows

    def __setitem__(self, v, original):
        self.note(v, original)

    def note(self, v, original):
        if v not in self.rows:
            self.rows[v] = [v.index, original.copy(), None]

    def wrote(self, verts):
        for v in verts:
            row = self.rows.get(v)
            if row is not None and v.is_valid:
                row[2] = v.co.copy()

    def stale(self):
        """A vertex this tool wrote no longer holds what it wrote (an outside edit, or a handle that went stale)."""
        try:
            return any(last is not None and (not v.is_valid or not _close(v.co, last))
                       for v, (_index, _original, last) in self.rows.items())
        except ReferenceError:
            return True

    def restore(self):
        return restore_written(self.bm, ((v, index, original, original if last is None else last)
                                         for v, (index, original, last) in self.rows.items()))


class MeshSnapshot:
    def __init__(self, obj):
        self.obj = obj
        self.data = obj.data
        self.expected = None   # (BMesh copy, positions of the vertices the tool keeps moving)
        bm = bmesh.from_edit_mesh(obj.data)
        self.edit_bm = bm
        bm.verts.index_update()
        bm.edges.index_update()
        bm.faces.index_update()
        self.history = [(type(e).__name__, e.index) for e in bm.select_history]
        self.active_face = bm.faces.active.index if bm.faces.active else None
        self.shapes = _shape_signature(bm, obj.data)
        self.shape_key_index = obj.active_shape_key_index   # the key whose cage the backup holds (restore)
        self.backup = bm.copy()

    def restore(self):
        """Put the snapshot back. Returns False (nothing written, the outside edits kept)
        when the shape keys changed since the snapshot (a deleted key was
        resurrected); raises when the target or the edit session changed."""
        if self.backup is None or self.obj.data != self.data or not self.data.is_editmode:
            raise RuntimeError("Snapshot target is no longer the original edit mesh")
        bm = bmesh.from_edit_mesh(self.data)
        if bm is not self.edit_bm:
            raise RuntimeError("Snapshot edit session changed; rollback refused")
        if _shape_signature(bm, self.data) != self.shapes:
            return False
        keyed = len(self.backup.verts.layers.shape) > 0
        # with shape keys the temp mesh is a copy of the object's own mesh, so its Key keeps every
        # block's uid / order / settings and each BMesh shape layer maps back to its own key (also with uid gaps)
        mesh = self.data.copy() if keyed else bpy.data.meshes.new("PolyTailor rollback")
        key = mesh.shape_keys if keyed else None
        try:
            self.backup.to_mesh(mesh)
            load = {}
            if keyed:   # the cage stays on the snapshot's active shape key (edits must not land in Basis)
                index = min(max(self.shape_key_index, 0), len(mesh.shape_keys.key_blocks) - 1)
                load = dict(use_shape_key=True, shape_key_index=index)
            probe = bmesh.new()   # build into a temporary BMesh first (the same call as below); the edit mesh
            try:                  # is cleared only once that worked
                probe.from_mesh(mesh, **load)
            finally:
                probe.free()
            select_mode = bm.select_mode   # bm.clear() empties it (select_flush_mode would flush nothing)
            bm.clear()
            bm.from_mesh(mesh, **load)
            if select_mode:   # an empty set cannot be assigned
                bm.select_mode = select_mode
            seqs = {"BMVert": bm.verts, "BMEdge": bm.edges, "BMFace": bm.faces}
            for seq in seqs.values():
                seq.ensure_lookup_table()
            bm.select_history.clear()
            for kind, index in self.history:
                if index < len(seqs[kind]):
                    bm.select_history.add(seqs[kind][index])
            if self.active_face is not None and self.active_face < len(bm.faces):
                bm.faces.active = bm.faces[self.active_face]
            bm.normal_update()
            bmesh.update_edit_mesh(self.obj.data, destructive=True)
            return True
        finally:
            bpy.data.meshes.remove(mesh)
            try:   # the copied Key normally goes with its mesh; remove it only if it was left behind
                if key is not None and key.users == 0:
                    bpy.data.batch_remove([key])
            except (ReferenceError, RuntimeError):
                pass

    def expect(self, moving=()):
        """the tool's own edit is done (Build's commit): a later rollback expects the edit mesh as it
        is now, except the vertices in `moving`, which the tool itself keeps moving. A C-speed copy, compared once by
        outside_changed right before the rollback."""
        bm = bmesh.from_edit_mesh(self.data)
        self._drop_expected()
        self.expected = (bm.copy(), [tuple(v.co) for v in moving if v.is_valid])

    def _drop_expected(self):
        expected, self.expected = getattr(self, "expected", None), None
        if expected is not None:
            expected[0].free()

    def outside_changed(self, touched=(), topology=True):
        """ONE O(n) comparison, right before a full restore, of the edit mesh with what the tool
        expects: the snapshot (or the state expect() recorded) plus the tool's own changes. Every vertex not in
        `touched` (the vertices the tool moved or created) must sit at a position the expected mesh has, as many times
        (a multiset of positions: independent of element order), and with topology=True the edge and face counts must
        match. True means something else edited the mesh (a script, another add-on's timer): the caller keeps that
        edit and does not restore. An error counts as changed (fail safe)."""
        from collections import Counter
        try:
            if self.backup is None or self.obj.data != self.data or not self.data.is_editmode:
                return True
            bm = bmesh.from_edit_mesh(self.data)
            base, moving = self.expected if getattr(self, "expected", None) is not None else (self.backup, ())
            touched = {v for v in touched if v.is_valid}
            want = Counter(tuple(v.co) for v in base.verts)
            want.subtract(moving)
            if +want != Counter(tuple(v.co) for v in bm.verts if v not in touched):
                return True
            if topology and (len(bm.edges), len(bm.faces)) != (len(base.edges), len(base.faces)):
                return True
            return _attributes_lost(base, bm, touched)
        except (ReferenceError, RuntimeError, AttributeError, ValueError, TypeError, KeyError):
            return True

    def close(self):
        self._drop_expected()
        if self.backup is not None:
            self.backup.free()
            self.backup = None
        self.edit_bm = None


_LAYER_KINDS = ("float", "int", "float_vector", "float_color", "color", "string", "bool", "deform", "uv")


def _attr_key(elem, kind, layer):
    value = elem[layer]
    if kind == "deform":
        return tuple(sorted(value.items()))
    if kind == "uv":
        return tuple(value.uv)
    if kind in ("float_vector", "float_color", "color"):
        return tuple(value)
    return value


def _attributes_lost(base, bm, touched):
    """True when data the snapshot holds for elements the tool did not touch is no longer in the edit mesh: every
    vertex / edge / face / face-corner attribute layer the snapshot restores (UVs, crease, deform weights, colours,
    generic attributes) and the seam / sharp / smooth / material flags. Every such value of the snapshot must still
    be present (a multiset keyed by positions, independent of element order); values the tool added (new elements)
    are allowed. Elements with a vertex the tool moved are skipped on both sides. Selection and hiding are the
    tool's own and not compared."""
    from collections import Counter
    keep = {tuple(v.co) for v in bm.verts if v not in touched}

    def kept(verts):
        return all(tuple(v.co) in keep for v in verts)

    def where(b):
        return (("verts", lambda v: (tuple(v.co),), [v for v in b.verts if tuple(v.co) in keep]),
                ("edges", lambda e: tuple(sorted(tuple(v.co) for v in e.verts)), [e for e in b.edges if kept(e.verts)]),
                ("faces", lambda f: tuple(sorted(tuple(v.co) for v in f.verts)), [f for f in b.faces if kept(f.verts)]),
                ("loops", lambda l: (tuple(l.vert.co), tuple(l.link_loop_next.vert.co)),
                 [l for f in b.faces if kept(f.verts) for l in f.loops]))

    def signature(b):
        out = {}
        for domain, key, elems in where(b):
            seq = getattr(b, domain) if domain != "loops" else b.loops
            for kind in _LAYER_KINDS:
                access = getattr(seq.layers, kind, None)
                if access is None:
                    continue
                for name, layer in access.items():
                    if name.startswith("."):   # internal: UV selection / pin state, not user data
                        continue
                    out[(domain, kind, name)] = Counter((key(e), _attr_key(e, kind, layer)) for e in elems)
            if domain == "edges":
                out[("edges", "flags", "")] = Counter((key(e), e.seam, e.smooth) for e in elems)
            elif domain == "faces":
                out[("faces", "flags", "")] = Counter((key(f), f.smooth, f.material_index) for f in elems)
        return out

    want, have = signature(base), signature(bm)
    if set(want) != set(have):
        return True
    return any(want[k] - have[k] for k in want)


def adaptive_radius(view, vertex, base, minimum, excluded=()):
    """Cap a pixel radius at 40% of the nearest other shown vertex spacing."""
    p = view.screen(vertex)
    if p is None:
        return 0.0
    sx, sy, front, hide, _co = view.arrays()
    usable = front & ~hide
    usable[vertex.index] = False
    for other in excluded:
        usable[other.index] = False
    distances = (sx[usable] - p.x) ** 2 + (sy[usable] - p.y) ** 2
    nearest = float(np.sqrt(distances.min())) if len(distances) else float('inf')
    return min(base, max(minimum, 0.4 * nearest))


def _same_planes(clip, co, target):
    """target lies on every clipped mirror plane co is on (MirrorClip.planes: axis, threshold, transform)."""
    for i, t, transform in clip.planes:
        a = transform @ co if transform is not None else co
        b = transform @ target if transform is not None else target
        if abs(a[i]) <= t and abs(b[i]) > t:
            return False
    return True


def merge_moved(context, bm, moved, radius, *, adaptive_min=None, keep=(), force=False, clip=None):
    """Weld moved vertices to visible stationary targets; return the weld map. `keep`: vertices that are
    neither welded nor weld targets (a click's reused, unmoved vertices). `force`: weld with Auto Merge off too (a
    Build drag release that acts as a click); then no two moved vertices weld onto the same target. `clip`: a
    MirrorClip; a moved vertex on its plane never welds onto a target off the plane."""
    if not force and not context.scene.tool_settings.use_mesh_automerge:
        return
    view = MeshView(context, lazy=True)
    moved = {v for v in moved if v.is_valid}
    targets = {}
    radii = {}
    used = set()
    if not moved:
        return
    # numpy narrows the stationary vertices down to those near a moved one on screen
    # (1 px margin); the exact test below runs on them in vertex order, as before
    sx, sy, front, hide, _co = view.arrays()
    usable = front & ~hide
    for v in moved:
        usable[v.index] = False
    for v in keep:
        if v.is_valid and 0 <= v.index < len(usable):
            usable[v.index] = False
    points = [p for p in (view.screen(v) for v in moved) if p is not None]
    if not points:
        return
    margin = radius + 3 + 2   # + visibility probe + slack
    view.limit_to(min(p.x for p in points) - margin, min(p.y for p in points) - margin,
                  max(p.x for p in points) + margin, max(p.y for p in points) + margin)
    verts = bm.verts
    for v in moved:
        p = view.screen(v)
        if p is None:
            continue
        near = usable & ((sx - p.x) ** 2 + (sy - p.y) ** 2 <= (radius + 1.0) ** 2)
        stationary = [verts[int(i)] for i in np.nonzero(near)[0]]
        nearest, distance = None, radius * radius
        for target in stationary:
            if force and target in used:
                continue   # two end vertices on one target would collapse the edge between them
            if clip is not None and not _same_planes(clip, v.co, target.co):
                continue   # a mirror-plane vertex stays on each of its planes
            q = view.screen(target)
            if q is None:
                continue
            d = (q - p).length_squared
            if adaptive_min is not None and target not in radii:
                radii[target] = adaptive_radius(view, target, radius, adaptive_min, excluded=moved)
            limit = radii[target] ** 2 if adaptive_min is not None else radius * radius
            if d <= min(distance, limit) and view.visible(target):
                nearest, distance = target, d
        if nearest is not None:
            targets[v] = nearest
            used.add(nearest)
    if targets:
        bmesh.ops.weld_verts(bm, targetmap=targets)
        bm.normal_update()
        bmesh.update_edit_mesh(view.obj.data, destructive=True)
    return targets
