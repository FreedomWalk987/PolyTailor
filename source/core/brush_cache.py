# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Single-object, owned-stroke reuse with immutable-base/dirty-surface queries.

Handlers only invalidate references. All BMesh access and snapshot construction
happen in the brush's normal synchronous event body.
"""
import bpy
import bmesh
import numpy as np
from .brush_bounds import local_bvh

_entry=None
_owner=None
# Entries dropped by a handler. Freeing the last Python reference to the edit BMesh
# reallocates all of its custom data (Blender removes its Python-pointer layer). Done
# inside a depsgraph handler while a native modal tool (Vertex / Edge Slide, Loop Cut)
# keeps pointers into that data can invalidate its pointers (a possible crash).
# References are kept here and released only where no such tool is running.
_retired=[]
_retired_strokes=[]


def _schedule_release():
    if not bpy.app.timers.is_registered(_release_retired):
        bpy.app.timers.register(_release_retired,first_interval=.25)


def _clear_stroke(stroke):
    # SmoothSolver and GlueState point back to their stroke. Dropping only the
    # operator's reference leaves BMesh wrappers in cyclic garbage until an
    # unrelated event (possibly a native slide) invokes Python's collector.
    # A reported Python exception may keep the solver/glue object alive through
    # its traceback even after the stroke drops it. Clear their BMesh state too.
    for name in ("solver","glue"):
        component=getattr(stroke,name,None)
        if component is not None:
            component.__dict__.clear()
    stroke.__dict__.clear()


def _clear_strokes():
    for stroke in _retired_strokes:
        _clear_stroke(stroke)
    _retired_strokes.clear()


def release_stroke(stroke,context=None,owner=None):
    """Release a completed stroke after rollback/commit and resource cleanup.

    The retained Entry already owns any reusable arrays. Clear the stroke's
    dictionary without modifying those shared arrays or waiting for cyclic GC.
    If another modal owns the mesh during cancellation, defer the whole stroke
    so removing its last BMesh reference cannot reallocate native custom data.
    """
    if not _safe_point(bpy.context if context is None else context,owner):
        _retired_strokes.append(stroke)
        _schedule_release()
        return
    _clear_stroke(stroke)


def _retire():
    global _entry,_owner
    # Between strokes an entry holds arrays only (release_mesh): drop it at once.
    # Only one still wrapping the edit mesh (mid-stroke) waits for a safe point.
    if _entry is not None and _entry.bm is not None:
        _retired.append(_entry)
        _schedule_release()
    _entry=None;_owner=None


def _safe_point(context,owner=None):
    try:
        if context.edit_object is None:
            return True
        from .safe_mesh import foreign_modal_running
        return not foreign_modal_running(context,owner)
    except (AttributeError,ReferenceError,RuntimeError):
        return False


def break_cycles(state):
    """clear the operator state's nested holders (FillState / PolyPenState and its logic) so
    their BMesh wrappers go with the last reference now, at our safe point, not with a later cyclic GC run."""
    try:
        holder = state.get("state") if isinstance(state, dict) else None
        for obj in (getattr(holder, "logic", None), holder):
            d = getattr(obj, "__dict__", None)
            if isinstance(d, dict):
                d.clear()
    except Exception:
        pass


def _release_retired():
    if (_retired or _retired_strokes) and not _safe_point(bpy.context):
        return .25   # a native modal tool is running: try again later
    _clear_strokes()
    for state in _retired:
        break_cycles(state)
    _retired.clear()
    return None


class SurfaceOverlay:
    def __init__(self,entry):
        self.base=entry.base
        self.dirty=frozenset(entry.dirty)
        self.rows=np.array(sorted(self.dirty),dtype=np.int64)
        self.co=entry.cache["co"].copy()
        self.loops=entry.cache["loops"]
        self.starts=entry.cache["starts"]
        self.sizes=np.diff(np.append(self.starts,len(self.loops)))
        self.overlay=local_bvh(self.co,self.loops,self.starts,self.sizes,self.rows)
        from .brush_bounds import selected_loops
        corners,offsets=selected_loops(self.loops,self.starts,self.sizes,self.rows)
        following=np.arange(len(corners))+1
        following[offsets[1:]-1]=offsets[:-1]
        lengths=np.linalg.norm(self.co[corners]-self.co[corners[following]],axis=1)
        positive=lengths[lengths>0]
        self.edge_scale=float(positive.min()) if len(positive) else 0.
        self.scale=float(np.max(np.abs(self.co)))
        self.full=None
        self.fallbacks=0
        self.hits={}
        self.window=None
        self.window_center=None
        self.window_radius=0.
        self.initial_radius=self.edge_scale*4
        self.window_builds=0
        self.window_max_faces=0

    def materialize(self):
        if self.full is None:
            from .pick import bvh_from_loops
            self.full=bvh_from_loops(self.co,self.loops,self.starts,np.ones(len(self.starts),dtype=bool))
            self.fallbacks+=1
            self.hits.clear()
            self.window=None
        return self.full

    @staticmethod
    def equivalent_point(a,b,edge_scale=float("inf")):
        if tuple(a[0])==tuple(b[0]):
            return True
        left=np.asarray(a[0],dtype=float);right=np.asarray(b[0],dtype=float)
        # Equal-distance projections onto a shared edge can differ by one
        # float-storage rounding step after independent native traversals.
        # Permit at most 32 float32 ULPs per coordinate, also capped by
        # 1e-3 of the shortest changed edge. This is a storage-rounding allowance,
        # not a fixed world-space epsilon that grows freely with translation.
        quantum=np.abs(np.spacing(np.maximum(np.abs(left),np.abs(right)).astype(np.float32))).astype(float)
        delta=np.abs(left-right)
        return bool(np.all(delta<=32*quantum) and np.linalg.norm(delta)<=1e-3*edge_scale)

    def find_nearest(self,point):
        if self.full is not None:
            return self.full.find_nearest(point)
        key=tuple(point)
        hit=self.hits.get(key)
        if hit is not None:
            return hit
        hit=self._nearest(point)
        if self.full is None:
            if len(self.hits)>=16384:self.hits.clear()
            self.hits[key]=hit
        return hit

    def _nearest(self,point):
        if self.window_center is None:
            self.window_center=point.copy()
        distance=(point-self.window_center).length
        if self.window is None or distance>self.window_radius:
            radius=max(self.initial_radius,self.window_radius*2,distance*1.5)
            if not np.isfinite(radius) or radius<=0:
                return self.materialize().find_nearest(point)
            anchor=self._nearest_global(self.window_center)
            if self.full is not None:
                return self.full.find_nearest(point)
            # For |q-c|<=r and any known surface anchor a, the true nearest s
            # obeys |s-c|<=|s-q|+|q-c|<=|a-q|+r<=2r+|a-c|.
            # Gather every current polygon intersecting this larger ball once.
            pad=32*np.finfo(np.float32).eps*(1+self.scale+radius+anchor[3])
            coverage=2*radius+anchor[3]+pad
            rows={hit[2] for hit in self.base.find_nearest_range(self.window_center,coverage)
                  if hit[2] not in self.dirty}
            rows.update(int(self.rows[hit[2]]) for hit in self.overlay.find_nearest_range(self.window_center,coverage))
            self.window_rows=np.array(sorted(rows),dtype=np.int64)
            self.window=local_bvh(self.co,self.loops,self.starts,self.sizes,self.window_rows)
            self.window_radius=radius
            self.window_builds+=1
            self.window_max_faces=max(self.window_max_faces,len(self.window_rows))
        nearest=self.window.find_nearest(point)
        if nearest[0] is None:
            return self.materialize().find_nearest(point)
        pad=32*np.finfo(np.float32).eps*(1+self.scale+nearest[3])
        for hit in self.window.find_nearest_range(point,nearest[3]+pad):
            if hit[3]==nearest[3] and not self.equivalent_point(hit,nearest,self.edge_scale):
                self.ambiguity=(tuple(nearest[0]),tuple(hit[0]),nearest[3],hit[3])
                return self.materialize().find_nearest(point)
        return nearest[0],nearest[1],int(self.window_rows[nearest[2]]),nearest[3]

    def _nearest_global(self,point):
        old=self.base.find_nearest(point)
        current=self.overlay.find_nearest(point)
        if current[0] is None:
            return self.materialize().find_nearest(point)
        current=(current[0],current[1],int(self.rows[current[2]]),current[3])
        # Every unchanged winner must be within this known current-surface upper
        # bound. Padding only broadens the search; native distances decide it.
        pad=32*np.finfo(np.float32).eps*(1+self.scale+max(abs(x) for x in point)+current[3])
        bound=current[3]
        if old[0] is not None and old[2] not in self.dirty:
            bound=min(bound,old[3])
        radius=bound+pad
        unchanged=[hit for hit in self.base.find_nearest_range(point,radius) if hit[2] not in self.dirty]
        candidates=[current]+unchanged
        best=min(candidates,key=lambda hit:hit[3])
        # Broad-phase padding only broadens the query. Native reported distances
        # choose the winner; only an actual equal-distance result is ambiguous.
        tie=0.
        # Float-distance ties can depend on the native full-tree traversal order.
        # Geometrically distinct tied sheets retain the original full oracle.
        # Shared-edge float-storage ties have the explicit bounded allowance.
        for hit in candidates:
            if abs(hit[3]-best[3])<=tie and not self.equivalent_point(hit,best,self.edge_scale):
                self.ambiguity=(tuple(best[0]),tuple(hit[0]),best[3],hit[3])
                return self.materialize().find_nearest(point)
        local=self.overlay.find_nearest_range(point,best[3]+pad)
        for hit in local:
            if abs(hit[3]-best[3])<=tie and not self.equivalent_point(hit,best,self.edge_scale):
                self.ambiguity=(tuple(best[0]),tuple(hit[0]),best[3],hit[3])
                return self.materialize().find_nearest(point)
        return best


class Entry:
    def __init__(self,stroke,base):
        self.obj=stroke.obj;self.data=stroke.obj.data;self.bm=stroke.bm
        self.cache=stroke._cache;self.base=base;self.dirty=set()
        self.counts=(len(self.bm.verts),len(self.bm.edges),len(self.bm.faces))
        self.matrix=tuple(map(tuple,stroke.mw))
        self.pen3d=bool(stroke.scene.pen3d)
        self.state={};self.reference=None;self.stroke_dirty=set()
        projection=getattr(stroke,"_screen_bounds",None)
        if projection is not None and projection.face_rows is not None:
            self.face_rows=projection.face_rows
        else:
            self.bm.faces.index_update()
            self.face_rows=np.full(len(self.bm.faces),-1,dtype=np.int64)
            self.face_rows[[f.index for f in self.cache["faces"]]]=np.arange(len(self.cache["faces"]))

    def valid(self,context):
        try:
            obj=context.edit_object
            if not (obj==self.obj and obj.data==self.data
                    and self.pen3d==bool(context.scene.ppo.pen3d)
                    and self.matrix==tuple(map(tuple,obj.matrix_world))):
                return False
            bm=bmesh.from_edit_mesh(obj.data)
            return self.counts==(len(bm.verts),len(bm.edges),len(bm.faces))
        except (ReferenceError,RuntimeError,AttributeError):
            return False

    # Between strokes the entry keeps arrays only, no BMesh or element wrappers
    # (each stroke takes the edit mesh anew). A wrapper alive between
    # strokes kept Blender's Python-pointer layer on the edit mesh while native tools
    # ran, and freeing it reallocated all of the mesh's custom data at an unpredictable
    # moment (a possible crash).
    SAMPLES=48

    def release_mesh(self):
        self.bm=None
        if self.cache is not None:
            self.cache["verts"]=None;self.cache["faces"]=None
        for value in self.state.values():
            if hasattr(value,"bm"):
                value.bm=None

    def attach_mesh(self,bm):
        """Wrap the edit mesh again for a new stroke; False when it no longer matches
        the arrays (rows, coordinates or face corners changed without an update)."""
        cache=self.cache
        bm.verts.index_update();bm.faces.index_update()
        verts=list(bm.verts)
        faces=[f for f in bm.faces if not f.hide]
        if len(verts)!=len(cache["co"]) or len(faces)!=len(cache["starts"]):
            return False
        n=len(verts)
        for i in {int(i) for i in np.linspace(0,n-1,min(n,self.SAMPLES))} if n else ():
            if verts[i].hide!=bool(cache["hide"][i]) or np.any(np.abs(np.array(verts[i].co)-cache["co"][i])>1e-6*(1+np.abs(cache["co"][i]))):
                return False
        loops=cache["loops"];starts=cache["starts"];m=len(faces)
        for i in {int(i) for i in np.linspace(0,m-1,min(m,self.SAMPLES))} if m else ():
            corners=[v.index for v in faces[i].verts]
            if corners!=[int(x) for x in loops[starts[i]:starts[i]+len(corners)]]:
                return False
        self.bm=bm
        cache["verts"]=verts;cache["faces"]=faces
        for value in self.state.values():
            if hasattr(value,"bm"):
                value.bm=bm
        return True

    def record(self,vertices):
        faces={int(self.face_rows[f.index]) for v in vertices for f in v.link_faces if not f.hide}
        faces.discard(-1)
        self.dirty.update(faces);self.stroke_dirty.update(faces)

    def surface(self,stroke):
        if self.base is None or not self.dirty:
            self.reference=None
            return self.base
        # Avoid a large dirty overlay or a long-lived degraded base. Compaction
        # is an exact full rebuild, not truncation of changed geometry.
        if len(self.cache["faces"])<=50000 or len(self.dirty)>min(50000,len(self.cache["faces"])*.2):
            from .pick import bvh_from_loops
            self.base=bvh_from_loops(self.cache["co"],self.cache["loops"],self.cache["starts"],np.ones(len(self.cache["faces"]),dtype=bool))
            self.dirty.clear();self.reference=None
            return self.base
        self.reference=SurfaceOverlay(self)
        self.reference.initial_radius=max(self.reference.initial_radius,
                                         stroke.radius_world*float(np.linalg.norm(np.asarray(stroke.mi)[:3,:3],ord=2))*2)
        return self.reference


def begin(stroke,context):
    global _entry,_owner
    _clear_strokes()
    _retired.clear()   # inside our own operator: a safe point
    if _entry is not None and _owner is None:
        # Flush pending edits made by other tools before taking ownership. Their
        # dependency notification must invalidate us, not be mistaken for ours.
        context.view_layer.update()
    if _entry is not None and not (_entry.valid(context) and _entry.attach_mesh(bmesh.from_edit_mesh(context.edit_object.data))):
        _entry=None
    _owner=stroke
    stroke._cache_closed=False
    if _entry is not None:
        stroke._cache=_entry.cache
        for name,value in _entry.state.items():
            setattr(stroke,name,value)
        stroke._direct_cache=True
        _entry.stroke_dirty.clear()
    stroke._retained_entry=_entry
    stroke.cache_reused=_entry is not None


def attach(stroke,base):
    global _entry
    if getattr(stroke,"_retained_entry",None) is None:
        _entry=Entry(stroke,base)
        stroke._retained_entry=_entry


def seal(stroke):
    global _entry,_owner
    entry=getattr(stroke,"_retained_entry",None)
    if entry is None or _owner is not stroke:
        return
    try:
        # Consume our own pending dependency updates while ownership is explicit.
        # Any later geometry update outside the stroke invalidates the cache.
        stroke.context.view_layer.update()
        if entry.reference is not None and entry.reference.full is not None:
            entry.base=entry.reference.full
            entry.dirty=set(entry.stroke_dirty)
        entry.reference=None
        entry.state={name:getattr(stroke,name) for name in
                     ("_screen_bounds","_moving_grid","_grid_cache","_grid_matrix","_grid_hidden")
                     if hasattr(stroke,name)}
        entry.counts=(len(stroke.bm.verts),len(stroke.bm.edges),len(stroke.bm.faces))
        entry.release_mesh()
        _entry=entry
    except (ReferenceError,RuntimeError,AttributeError):
        _entry=None
    finally:
        _owner=None;stroke._cache_closed=True


def abort(stroke):
    global _entry,_owner
    if _owner is stroke and not getattr(stroke,"_cache_closed",False):
        _entry=None;_owner=None
        stroke._cache_closed=True


def invalidate(*_args):
    _retire()


def graph_update(_scene,graph):
    if _entry is None or _owner is not None:
        return
    try:
        for update in graph.updates:
            if (update.is_updated_geometry or update.is_updated_transform) and update.id.original in {_entry.obj,_entry.data}:
                _retire();return
    except (ReferenceError,RuntimeError,AttributeError):
        _retire()


_HANDLERS=(("depsgraph_update_post",graph_update),("undo_pre",invalidate),
           ("redo_pre",invalidate),("load_pre",invalidate))


def start():
    """while PolyTailor is in use (session); what changed meanwhile is not known: start clean."""
    invalidate()
    for name,callback in _HANDLERS:
        callback=bpy.app.handlers.persistent(callback)
        if callback not in getattr(bpy.app.handlers,name):getattr(bpy.app.handlers,name).append(callback)


def stop():
    for name,callback in _HANDLERS:
        if callback in getattr(bpy.app.handlers,name):getattr(bpy.app.handlers,name).remove(callback)
    invalidate()


def unregister():
    stop()
    # Disabling an add-on can itself happen while a native modal is active: the retired wrappers are then kept
    # (not freed under it); the release timer never outlives the add-on.
    _release_retired()
    if bpy.app.timers.is_registered(_release_retired):
        bpy.app.timers.unregister(_release_retired)
