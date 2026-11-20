# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Persistent connected Glue chains; no mesh edges are created or removed."""
import bmesh
import bpy
from ..i18n import tr_rpt   # reports in the add-on context

from ..core.glue import VERT_LAYER, EDGE_LAYER
from ..core.safe_mesh import native_modal_running
from .pinning import _clear_pin_cache


def glue_selection(bm, remove=False):
    for elements in (bm.verts, bm.edges):
        elements.ensure_lookup_table()
        elements.index_update()
    vl = bm.verts.layers.int.get(VERT_LAYER)
    el = bm.edges.layers.int.get(EDGE_LAYER)
    if remove:
        if vl is None:
            return 0, False
        chosen = {v.index for v in bm.verts if v.select and not v.hide and v[vl] > 0}
        if not chosen:
            return 0, False
        groups = {int(bm.verts[i][vl]) for i in chosen}
        # a link: an edge whose value and both ends carry the same group id
        links = [e for e in bm.edges if el is not None and e[el] > 0
                 and int(e.verts[0][vl]) == int(e.verts[1][vl]) == int(e[el])]
        cut = {e.index for e in links if not e.hide and all(v.index in chosen for v in e.verts)}
        covered = {v.index for e in links if e.index in cut for v in e.verts}
        loose = chosen - covered   # a selected vertex without a fully selected link: released with its links
        cut.update(e.index for e in links if any(v.index in loose for v in e.verts))
        members = [v.index for v in bm.verts if int(v[vl]) in groups]
        parent = {i: i for i in members}
        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for e in links:   # the touched groups split into parts over the links that remain
            if e.index not in cut and int(e[el]) in groups:
                parent[root(e.verts[0].index)] = root(e.verts[1].index)
        parts = {}
        for i in members:
            if i not in loose:
                parts.setdefault(root(i), []).append(i)
        next_id = max([int(v[vl]) for v in bm.verts] + [int(e[el]) for e in bm.edges] if el is not None
                      else [int(v[vl]) for v in bm.verts])+1   # above every id of both layers
        if next_id+len(parts) >= 2**31:
            raise ValueError("Too many Glue groups")
        renamed, offset = {}, 0
        for part in parts.values():
            group = 0
            if len(part) >= 2:   # a 1-vertex part is released
                group, offset = next_id+offset, offset+1
            renamed.update((i, group) for i in part)
        kept = {e.index for e in links if e.index not in cut}
        if el is not None:
            for e in bm.edges:
                if int(e[el]) in groups:
                    a, b = (renamed.get(v.index, 0) for v in e.verts)
                    e[el] = a if e.index in kept and a == b and a > 0 else 0
        for i in members:
            bm.verts[i][vl] = renamed.get(i, 0)
        dropped = False
        if not any(v[vl] > 0 for v in bm.verts):
            bm.verts.layers.int.remove(vl)
            dropped = True
            if el is not None:
                bm.edges.layers.int.remove(el)
        return len(groups), dropped
    selected = [(e.index, e.verts[0].index, e.verts[1].index) for e in bm.edges
                if e.select and not e.hide and all(v.select and not v.hide for v in e.verts)]
    if not selected:
        return 0, False
    old = {v.index: int(v[vl]) for v in bm.verts} if vl is not None else {}
    parent = {}
    def find(i):
        parent.setdefault(i, i)
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def join(a, b):
        parent[find(a)] = find(b)
    for _edge, a, b in selected:
        join(a, b)
    touched = {old.get(i, 0) for i in parent} - {0}
    first = {}
    for i, group in old.items():
        if group in touched:
            if group in first:
                join(first[group], i)
            else:
                first[group] = i
                find(i)
    components = {}
    for i in list(parent):
        components.setdefault(find(i), []).append(i)
    next_id = max(old.values(), default=0)+1
    if next_id+len(components) >= 2**31:
        raise ValueError("Too many Glue groups")
    # Allocate both attributes before acquiring writable element wrappers.
    if vl is None:
        vl = bm.verts.layers.int.new(VERT_LAYER)
    if el is None:
        el = bm.edges.layers.int.new(EDGE_LAYER)
    membership = {}
    for offset, vertices in enumerate(components.values()):
        group = next_id+offset
        for i in vertices:
            bm.verts[i][vl] = group
            membership[i] = group
    selected_edges = {i for i, _a, _b in selected}
    for e in bm.edges:
        a, b = (v.index for v in e.verts)
        if e.index in selected_edges or (e[el] > 0 and (a in membership or b in membership)):
            ga, gb = int(e.verts[0][vl]), int(e.verts[1][vl])
            e[el] = ga if ga == gb else 0
    return len(components), False


class PPO_OT_glue(bpy.types.Operator):
    bl_idname = "ppo.glue"
    bl_label = "Glue"
    bl_translation_context = "Poly Toolset"
    bl_description = "Keep each connected selected chain rigid during Relax, Tweak and Alt+Wheel; the whole chain can still move and rotate"
    bl_options = {"REGISTER", "UNDO"}
    unglue: bpy.props.BoolProperty(default=False, options={"SKIP_SAVE"})

    @classmethod
    def description(cls, context, properties):
        from ..i18n import tr
        return tr("Let the selected chains bend again" if properties.unglue else cls.bl_description)

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and context.edit_object is not None

    def execute(self, context):
        if native_modal_running(context):
            self.report({"WARNING"}, tr_rpt("Finish the active transform first"))
            return {"CANCELLED"}
        bm = bmesh.from_edit_mesh(context.edit_object.data)
        count, removed = glue_selection(bm, self.unglue)
        if not count:
            self.report({"INFO"}, tr_rpt("Select a Glue member to unglue" if self.unglue else "Select a connected edge chain to Glue"))
            return {"CANCELLED"}
        bmesh.update_edit_mesh(context.edit_object.data, loop_triangles=False, destructive=removed)
        _clear_pin_cache()
        if context.area:
            context.area.tag_redraw()
        return {"FINISHED"}


def register():
    bpy.utils.register_class(PPO_OT_glue)


def unregister():
    bpy.utils.unregister_class(PPO_OT_glue)
