# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Bulk public Mesh reads after a native Edit Mode synchronization."""
import numpy as np
from .brush_bounds import selected_loops


def snapshot(stroke):
    bm=stroke.bm
    keys=getattr(stroke.obj.data,"shape_keys",None)
    if keys is not None and len(bm.verts.layers.shape) and stroke.obj.active_shape_key_index!=0:
        # the Mesh holds the Basis positions while the edit mesh holds the active key's: read the edit mesh
        raise ValueError("Native Edit Mode snapshot is not the edited shape")
    bm.verts.index_update();bm.faces.index_update()
    stroke.obj.update_from_editmode()
    mesh=stroke.obj.data
    if len(mesh.vertices)!=len(bm.verts) or len(mesh.polygons)!=len(bm.faces):
        raise ValueError("Native Edit Mode snapshot changed element counts")
    vertices=list(bm.verts)
    faces=list(bm.faces)
    co=np.empty((len(vertices),3),dtype=np.float32)
    mesh.vertices.foreach_get("co",co.ravel())
    if len(vertices):   # the bulk read must be the edit mesh itself (sampled rows)
        for i in np.unique(np.linspace(0,len(vertices)-1,num=min(64,len(vertices))).astype(np.int64)):
            if tuple(co[i])!=tuple(np.float32(c) for c in vertices[int(i)].co):
                raise ValueError("Native Edit Mode snapshot is not the edited shape")
    loops=np.empty(len(mesh.loops),dtype=np.int32)
    mesh.loops.foreach_get("vertex_index",loops)
    starts=np.empty(len(faces),dtype=np.int32);sizes=np.empty(len(faces),dtype=np.int32)
    mesh.polygons.foreach_get("loop_start",starts);mesh.polygons.foreach_get("loop_total",sizes)
    hidden=np.fromiter((v.hide for v in vertices),dtype=bool,count=len(vertices))
    face_hidden=np.fromiter((f.hide for f in faces),dtype=bool,count=len(faces))
    if face_hidden.any():
        rows=np.flatnonzero(~face_hidden)
        loops,offsets=selected_loops(loops,starts,sizes,rows)
        starts=offsets[:-1]
        faces=[faces[int(i)] for i in rows]
    return dict(verts=vertices,co=co.astype(np.float64),hide=hidden,
                faces=faces,loops=loops,starts=starts)
