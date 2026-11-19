# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Correct custom-tool corner data with Blender's native face interpolation.

One immutable native BMesh reference per interaction prevents incremental UV drift.
No UV unwrap/packing solver is implemented here. Topology-changing extrusion
deliberately keeps its normal texture-follows-geometry behavior.
"""
import bmesh
import bpy


def _has_data(bm):
    layers = bm.loops.layers
    return any(len(getattr(layers, name, ())) for name in
               ('uv', 'color', 'float_color', 'float', 'float_vector'))


class Correction:
    def __init__(self, bm, context):
        self.reference = None
        self.touched = set()
        self.group_cache = {}
        ts = context.scene.tool_settings
        self.connected = ts.use_transform_correct_keep_connected
        if not ts.use_transform_correct_face_attributes or not _has_data(bm):
            return
        bm.verts.index_update()
        bm.faces.index_update()
        self.reference = bm.copy()
        self.face_count = len(bm.faces)   # reference faces past this are the sample face
        sample = [self.reference.verts.new(co) for co in ((0,0,0), (1,0,0), (0,1,0))]
        self.sample_loop = self.reference.faces.new(sample).loops[0]
        self.reference.faces.ensure_lookup_table()
        self.value_layers = []
        for family in ('color', 'float_color', 'float', 'float_vector'):
            source = getattr(self.reference.loops.layers, family)
            for name, layer in getattr(bm.loops.layers, family).items():
                self.value_layers.append((layer, source[name]))
        self.uv_layers = tuple(bm.loops.layers.uv.values())
        self.ref_uv = tuple(self.reference.loops.layers.uv[name] for name in bm.loops.layers.uv.keys())

    def _original(self, loop):
        face = self.reference.faces[loop.face.index]
        return face, face.loops[list(loop.face.loops).index(loop)]

    def _copy(self, loop, source):
        # BMLoop.copy_from invalidates the destination's Python wrapper. Copy only
        # continuous values so geometry handles, UV pins and selection stay intact.
        for layer, ref_layer in zip(self.uv_layers, self.ref_uv):
            loop[layer].uv = source[ref_layer].uv
        for layer, ref_layer in self.value_layers:
            loop[layer] = source[ref_layer]

    def _groups(self, vertex, loops):
        cached = self.group_cache.get(vertex.index)
        if cached is not None:
            return cached
        originals = [self._original(loop)[1] for loop in loops]
        faces = {loop.face: i for i, loop in enumerate(originals)}
        result = []
        for layer in self.ref_uv:
            parent = list(range(len(loops)))
            def root(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i
            for i, loop in enumerate(originals):
                for edge in (loop.edge, loop.link_loop_prev.edge):
                    if edge.seam:
                        continue
                    other = edge.other_vert(loop.vert)
                    own_end = next(item for item in loop.face.loops if item.vert == other)
                    for radial in edge.link_loops:
                        j = faces.get(radial.face)
                        if j is None or i == j:
                            continue
                        candidate = originals[j]
                        end = next(item for item in candidate.face.loops if item.vert == other)
                        if ((loop[layer].uv-candidate[layer].uv).length <= 1e-6
                                and (own_end[layer].uv-end[layer].uv).length <= 1e-6):
                            parent[root(i)] = root(j)
            result.append({loop.face.index: root(i) for i, loop in enumerate(originals)})
        self.group_cache[vertex.index] = result
        return result

    def apply(self, vertices):
        if self.reference is None:
            return
        for vertex in set(vertices):
            if not vertex.is_valid:
                continue
            loops = list(vertex.link_loops)
            groups = [dict() for _ in self.uv_layers] if self.connected else ()
            ids = self._groups(vertex, loops) if self.connected else ()
            for loop in loops:
                face, original = self._original(loop)
                self.touched.add(loop)
                if vertex.co == original.vert.co:
                    self._copy(loop, original)
                else:
                    self.sample_loop.vert.co = vertex.co
                    self.sample_loop.copy_from_face_interp(face, False, False)
                    self._copy(loop, self.sample_loop)
                for index, (layer, ref_layer) in enumerate(zip(self.uv_layers, self.ref_uv)):
                    if self.connected:
                        groups[index].setdefault(ids[index][loop.face.index], []).append(loop)
            for layer, group in zip(self.uv_layers, groups):
                for members in group.values():
                    if len(members) > 1:
                        uv = sum((loop[layer].uv.copy() for loop in members), members[0][layer].uv*0)/len(members)
                        for loop in members:
                            loop[layer].uv = uv

    def restore(self, vertices=None):
        """Put the touched corner data back. returns False and writes nothing
        when a touched face no longer maps onto its reference face (faces reordered or
        rebuilt outside the tool): stale indices would copy another face's UVs."""
        if self.reference is None:
            return True
        loops = self.touched if vertices is None else {loop for v in vertices if v.is_valid for loop in v.link_loops}
        loops = [loop for loop in loops if loop.is_valid]
        for loop in loops:
            index = loop.face.index
            if not 0 <= index < self.face_count or len(self.reference.faces[index].loops) != len(loop.face.loops):
                return False
        for loop in loops:
            _face, original = self._original(loop)
            self._copy(loop, original)
        return True

    def close(self):
        if self.reference is not None:
            self.reference.free()
            self.reference = None
        self.touched.clear()
        self.group_cache.clear()


def apply(owner, vertices):
    correction = getattr(owner, 'attributes', None)
    if correction is not None:
        correction.apply(vertices)


class FillUV:
    """Extend an unambiguous UV rim with Blender's native polygon interpolation.

    Native face_attribute_fill propagates rim data but leaves brand-new interior
    vertices at UV zero. A temporary native n-gon supplies those missing values.
    UV islands which disagree around the rim are reported, never forced together.
    A wire outline (no faces on the rim) has no UV to continue and is not reported.
    issue: None, 'seam' (the rim crosses a UV seam) or 'missing' (part of the rim
    has no faces).
    """
    def __init__(self, bm, boundary, rim_faces=None, along=None):
        self.reference = None
        self.layers = []
        self.issue = None
        # for a loop closed through an existing edge only the faces on its open rim edges
        # are the rim (rim_faces); the faces across the closing edge (chain) never are.
        # along = {middle chain vertex: (a, b, t)}: its UV is blended from a's and b's along the
        # straight chain
        along = along or {}
        links = ((lambda v: v.link_loops) if rim_faces is None else
                 (lambda v: [l for l in v.link_loops if l.face in rim_faces]))
        self.has_rim = any(links(vertex) for vertex in boundary)
        if not len(bm.loops.layers.uv) or not self.has_rim:
            return
        if not all(links(vertex) or vertex in along for vertex in boundary):
            self.issue = 'missing'
            return
        samples = {}
        for name, layer in bm.loops.layers.uv.items():
            values = []
            valid = True
            for vertex in boundary:
                if vertex in along:
                    values.append(None)
                    continue
                candidates = [loop[layer].uv.copy() for loop in links(vertex)]
                if any((uv-candidates[0]).length > 1e-6 for uv in candidates):
                    valid = False
                    break
                values.append(candidates[0])
            if valid and along:
                index = {vertex: i for i, vertex in enumerate(boundary)}
                for vertex, (a, b, t) in along.items():
                    values[index[vertex]] = values[index[a]].lerp(values[index[b]], t)
            if valid:
                samples[name] = (layer, values)
            else:
                self.issue = 'seam'
        if not samples:
            return
        self.reference = bmesh.new()
        vertices = [self.reference.verts.new(v.co) for v in boundary]
        self.face = self.reference.faces.new(vertices)
        self.rim = {v: i for i, v in enumerate(boundary)}
        for name, (layer, values) in samples.items():
            ref_layer = self.reference.loops.layers.uv.new(name)
            for loop, uv in zip(self.face.loops, values):
                loop[ref_layer].uv = uv
            self.layers.append((layer, ref_layer, values))
        sample = [self.reference.verts.new(co) for co in ((0,0,0), (1,0,0), (0,1,0))]
        self.sample_loop = self.reference.faces.new(sample).loops[0]
        self.face.normal_update()

    def apply(self, faces):
        if self.reference is None:
            return
        vertices = {loop.vert for face in faces for loop in face.loops}
        created = set(faces)
        for vertex in vertices:
            index = self.rim.get(vertex)
            if index is None:
                self.sample_loop.vert.co = vertex.co
                self.sample_loop.copy_from_face_interp(self.face, False, False)
            for layer, ref_layer, values in self.layers:
                uv = values[index] if index is not None else self.sample_loop[ref_layer].uv.copy()
                for loop in vertex.link_loops:
                    if loop.face in created:
                        loop[layer].uv = uv

    def close(self):
        if self.reference is not None:
            self.reference.free()
            self.reference = None


class BuildExtension:
    """Continue a neighbouring face's corner data while placing Build geometry.

    Only donor faces are copied, once per click. New faces sample that immutable
    reference with Blender's native interpolation. Unfolding across the shared
    edge avoids collapsing UVs when the strip turns onto a steep surface.
    Existing faces and separate UV islands are never written.
    """
    def __init__(self, bm, context, edge, face=None, far=()):
        self.reference = None
        self.targets = []
        self.donors = []
        if not context.scene.tool_settings.use_transform_correct_face_attributes or not _has_data(bm):
            return
        candidates = list(edge.link_faces) if edge is not None else []
        if face is not None:
            candidates = [face]
        if not candidates:
            # A bridge may start on a wire and end at an existing surface.
            candidates = list({f for v in far if v is not None for f in v.link_faces})
        if not candidates:
            return
        bm.faces.index_update()
        candidates.sort(key=lambda f: f.index)
        self.reference = bmesh.new()
        self.layers = []
        for family in ('uv', 'color', 'float_color', 'float', 'float_vector'):
            for name, layer in getattr(bm.loops.layers, family).items():
                ref = getattr(self.reference.loops.layers, family).new(name)
                self.layers.append((family, layer, ref))
        self.reshape = face is not None
        for source in candidates:
            vertices = [self.reference.verts.new(l.vert.co) for l in source.loops]
            ref = self.reference.faces.new(vertices)
            ref.normal_update()
            for dst, src in zip(ref.loops, source.loops):
                for family, layer, ref_layer in self.layers:
                    if family == 'uv':
                        dst[ref_layer].uv = src[layer].uv
                    else:
                        dst[ref_layer] = src[layer]
            self.donors.append((ref, tuple(source.verts), source.material_index, source.smooth))
        sample = [self.reference.verts.new(co) for co in ((0,0,0), (1,0,0), (0,1,0))]
        self.sample_loop = self.reference.faces.new(sample).loops[0]

    def bind(self, faces):
        if self.reference is None:
            return
        for face in dict.fromkeys(faces):
            if not isinstance(face, bmesh.types.BMFace) or not face.is_valid:
                continue
            ring = set(face.verts)
            donor = max(self.donors, key=lambda d: len(ring.intersection(d[1])))
            ref, original, material, smooth = donor
            shared = ring.intersection(original)
            if len(shared) < 2:
                continue
            hinge = None
            if not self.reshape and len(shared) == 2:
                for i, loop in enumerate(ref.loops):
                    if original[i] in shared and original[(i+1)%len(original)] in shared:
                        a = loop.vert.co.copy()
                        direction = loop.link_loop_next.vert.co-a
                        if direction.length_squared > 1e-20:
                            direction.normalize()
                            inward = ref.calc_center_median()-a
                            inward -= direction*inward.dot(direction)
                            if inward.length_squared > 1e-20:
                                hinge = (a, direction, inward.normalized())
                        break
            face.material_index = material
            face.smooth = smooth
            self.targets.append((face, ref, original, hinge))
        self.apply()

    def apply(self):
        if self.reference is None:
            return
        for face, ref, original, hinge in self.targets:
            if not face.is_valid:
                continue
            outward = None
            if hinge is not None:
                a, direction, inward = hinge
                outward = face.calc_center_median()-a
                outward -= direction*outward.dot(direction)
                if outward.length_squared > 1e-20:
                    outward.normalize()
                else:
                    outward = -inward
            for loop in face.loops:
                # Exact shared-corner values preserve the donor island's rim.
                if loop.vert in original:
                    src = ref.loops[original.index(loop.vert)]
                else:
                    co = loop.vert.co
                    if hinge is not None:
                        delta = co-a
                        co = a + direction*delta.dot(direction) - inward*delta.dot(outward)
                    self.sample_loop.vert.co = co
                    self.sample_loop.copy_from_face_interp(ref, False, False)
                    src = self.sample_loop
                for family, layer, ref_layer in self.layers:
                    if family == 'uv':
                        loop[layer].uv = src[ref_layer].uv
                    else:
                        loop[layer] = src[ref_layer]

    def close(self):
        if self.reference is not None:
            self.reference.free()
            self.reference = None
        self.targets.clear()
        self.donors.clear()
