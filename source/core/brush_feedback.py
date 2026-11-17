# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Optional local dense-stroke feedback over the retained native edit display.

Actual BMesh coordinates change immediately. This display optimization is scoped
to solid retopology views; full native display is synchronized on stroke end.
"""
import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from mathutils.geometry import tessellate_polygon

_live=set()


class LocalFeedback:
    def __init__(self,stroke):
        self.stroke=stroke
        self.area=stroke.context.area
        self.faces=set()
        self.triangles=self.lines=self.points=None
        self.shader=gpu.shader.from_builtin("UNIFORM_COLOR")
        self.handle=bpy.types.SpaceView3D.draw_handler_add(self.draw,(),"WINDOW","POST_VIEW")
        self.draw_calls=0
        _live.add(self)

    def update(self,vertices):
        self.faces.update(f for v in vertices for f in v.link_faces if not f.hide)
        faces=[f for f in self.faces if f.is_valid]
        vertices=sorted({v for f in faces for v in f.verts},key=lambda v:v.index)
        if not vertices:
            return
        ids={v:i for i,v in enumerate(vertices)}
        positions=np.array([(self.stroke.mw@v.co)[:] for v in vertices])
        solver = self.stroke.solver._solver
        edges = getattr(solver, "length", None)   # R-2: Grid Relax's solver (_Points) has no edge lengths
        length = (float(np.median(edges)) if edges is not None
                  else max(self.stroke.radius_world * .02, 1e-8))
        view=self.stroke.context.region_data.view_matrix.inverted().to_3x3()@Vector((0,0,1))
        offset=np.array(view)*length*.002
        fill=np.ascontiguousarray(positions+offset,dtype=np.float32)
        wire=np.ascontiguousarray(positions+offset*2,dtype=np.float32)
        triangles=[[],[]];edges=[set(),set()]
        for face in faces:
            local=list(face.verts)
            group=triangles[int(face.select)]
            if len(local)==3:
                group.append(tuple(ids[v] for v in local))
            else:
                polygon=[Vector(fill[ids[v]]) for v in local]
                for triangle in tessellate_polygon([polygon]):
                    group.append(tuple(ids[local[int(i)]] for i in triangle))
            for edge in face.edges:
                edges[int(edge.select)].add(tuple(sorted(ids[v] for v in edge.verts)))
        theme=self.stroke.context.preferences.themes[0].view_3d
        base=(.21,.38,.37)
        face_colors=[(*base,1.),(*(np.array(base)*.65+np.array(theme.face_select[:3])*.35),1.)]
        self.triangles=[(batch_for_shader(self.shader,"TRIS",{"pos":fill},indices=indices),face_colors[i])
                        for i,indices in enumerate(triangles) if indices]
        edge_colors=[(*theme.wire_edit,1.),(*theme.edge_select,1.)]
        self.lines=[(batch_for_shader(self.shader,"LINES",{"pos":wire},indices=list(indices)),edge_colors[i])
                    for i,indices in enumerate(edges) if indices]
        point_colors=[(*theme.vertex,1.),(*theme.vertex_select,1.)]
        selected=np.array([v.select for v in vertices],dtype=bool)
        self.points=[(batch_for_shader(self.shader,"POINTS",{"pos":wire[selected==bool(i)]}),point_colors[i])
                     for i in (0,1) if np.any(selected==bool(i))]
        self.point_size=float(theme.vertex_size)

    def draw(self):
        if bpy.context.area!=self.area or self.triangles is None:
            return
        self.draw_calls+=1
        depth=gpu.state.depth_test_get();mask=gpu.state.depth_mask_get();width=gpu.state.line_width_get()
        gpu.matrix.push();gpu.matrix.push_projection()
        try:
            gpu.matrix.load_matrix(bpy.context.region_data.view_matrix)
            gpu.matrix.load_projection_matrix(bpy.context.region_data.window_matrix)
            gpu.state.depth_test_set("LESS_EQUAL");gpu.state.depth_mask_set(True)
            self.shader.bind()
            for batch,color in self.triangles:
                self.shader.uniform_float("color",color);batch.draw(self.shader)
            gpu.state.depth_mask_set(False);gpu.state.line_width_set(1.)
            for batch,color in self.lines:
                self.shader.uniform_float("color",color);batch.draw(self.shader)
            gpu.state.point_size_set(self.point_size)
            for batch,color in self.points:
                self.shader.uniform_float("color",color);batch.draw(self.shader)
        finally:
            gpu.state.depth_test_set(depth);gpu.state.depth_mask_set(mask);gpu.state.line_width_set(width)
            gpu.state.point_size_set(1.);gpu.matrix.pop_projection();gpu.matrix.pop()

    def close(self):
        if self.handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self.handle,"WINDOW")
            self.handle=None
        self.triangles=self.lines=self.points=None
        _live.discard(self)
        self.area.tag_redraw()


def unregister():
    for feedback in list(_live):
        feedback.close()
