# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Conservative mesh access coordination and primitive cache fingerprints.

Native modals holding edit-mesh pointers and mesh-writing modals above ours are
avoided; Python background tools beneath ours coexist.
Changes to other objects only refresh source caches; only changes to the edit
mesh itself end or reset a tool. Classification is not a safety proof.
No depsgraph or Undo handler traverses BMesh or performs evaluated queries.
"""

import bmesh
import bpy

_revisions = {}           # original ID pointer -> generation of its last geometry/transform update
_generation = 0
_foreign_generation = 0   # bumped by any depsgraph update outside the watched edit mesh
_epoch = 0                # bumped by undo / redo / file load
_watched = set()          # pointers of the edit object and mesh of the latest guard


def operator_key(op):
    """RNA instance identity; never infer ownership from an operator name."""
    try:
        return op.as_pointer()
    except (AttributeError, ReferenceError):
        return id(op)


def _python_operator_type(op):
    """Use registered operator classes, accepting both RNA and dotted IDs."""
    idname = getattr(op, "bl_idname", "") or ""
    candidates = [idname]
    if "." in idname:
        namespace, name = idname.split(".", 1)
        candidates.append(namespace.upper() + "_OT_" + name)
    rna = getattr(op, "bl_rna", None)
    identifier = getattr(rna, "identifier", "")
    if identifier and identifier != "Operator":
        candidates.append(identifier)
    return next((cls for name in candidates
                 if name and (cls := getattr(bpy.types, name, None)) is not None), None)


def modal_may_write_mesh(op):
    """Compatibility policy, not proof that an arbitrary Python modal is safe.

    Registered Python background tools without mesh-edit signals may coexist.
    Explicit writers, UNDO operators and mesh-edit namespaces still block.
    State fingerprints continue to catch outside mutations between events.
    """
    if getattr(op, "ppo_mesh_read_only", False) is True:
        return False
    if getattr(op, "ppo_mesh_write", False) is True:
        return True
    cls = _python_operator_type(op)
    if cls is None:
        # Native, or a Python modal whose class can no longer be looked up (e.g.
        # another add-on's sidebar session operator, a session that stays running for the
        # whole Blender session). Treating every such modal as a mesh writer made
        # Pin refuse with "pinning deferred" until Blender was restarted. Only
        # mesh-editing namespaces block now.
        return _native_may_hold_mesh(op)
    flags = set(getattr(cls, "bl_options", ())) | set(getattr(op, "bl_options", ()))
    if "UNDO" in flags or "UNDO_GROUPED" in flags:
        return True
    idname = (getattr(op, "bl_idname", "") or "").upper()
    return idname.startswith(("MESH_OT_", "MESH.", "TRANSFORM_OT_", "TRANSFORM.",
                              "OBJECT_OT_", "OBJECT.", "SCULPT_OT_", "SCULPT."))


# Only native namespaces whose modal operators can hold edit-mesh data block the
# BMesh path. Everything else - view navigation, file browser, UI, and other
# add-ons' Python modals whose class can no longer be looked up (e.g. a session
# operator left running after its add-on reloaded, such as another add-on's sidebar session)
# - is ignored. Such a long-lived modal used to freeze pins on cached positions.
_MESH_NAMESPACES = ("MESH", "TRANSFORM", "GIZMOGROUP", "GIZMO", "UV", "SCULPT", "OBJECT",
                    "VIEW3D_OT_edit", "VIEW3D_OT_ruler", "CURVE", "ARMATURE")


def _native_may_hold_mesh(op):
    idname = getattr(op, "bl_idname", "") or ""
    if "." in idname:
        namespace, name = idname.split(".", 1)
        idname = namespace.upper() + "_OT_" + name
    if not idname:
        return True
    return any(idname.startswith(ns if "_OT_" in ns else ns + "_OT_") for ns in _MESH_NAMESPACES)


def native_modal_running(context):
    """True while a native (C) modal operator runs in any window.

    For draw callbacks only. Python modal operators - PolyTailor's own or other add-ons' -
    run their mesh edits inside their own event handlers, never while a redraw is in
    progress, and hold no raw C pointers into the edit mesh, so reading the BMesh
    while only Python modals are active is safe. Native operators (G, E, Inset,
    Knife, slides...) do hold raw pointers / indices, so BMesh must not be touched.
    """
    try:
        windows = list(context.window_manager.windows)
        if context.window is not None and context.window not in windows:
            windows.append(context.window)
        for window in windows:
            for op in window.modal_operators:
                if _python_operator_type(op) is None and _native_may_hold_mesh(op):
                    return True
        return False
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        return True


# Modal families that never rebuild the edit mesh, and always-on overlays that sit
# on top of everything and drive nothing.
SAFE_MODAL_PREFIXES = ("VIEW3D_OT_", "VIEW2D_OT_", "WM_OT_")
IGNORED_MODAL_NAMES = frozenset({"Screencast Keys"})


def _rna_idname(op):
    idname = getattr(op, "bl_idname", "") or ""
    if "." in idname and "_OT_" not in idname:
        namespace, name = idname.split(".", 1)
        idname = namespace.upper() + "_OT_" + name
    return idname


def _is_safe_modal(op):
    if getattr(op, "name", "") in IGNORED_MODAL_NAMES:
        return True
    return _rna_idname(op).startswith(SAFE_MODAL_PREFIXES) and not _native_may_hold_mesh(op)


def _holds_mesh(op):
    """A modal that keeps raw pointers into the edit mesh while it runs (native mesh /
    transform / slide / knife ...), or a Python modal that declares itself a writer."""
    if getattr(op, "ppo_mesh_write", False) is True:
        return True
    return _python_operator_type(op) is None and _native_may_hold_mesh(op)


def foreign_modal_running(context, owner=None, allowed=()):
    """True while another modal may be editing the mesh under us.

    Current window: window.modal_operators is newest first. Only
    the modals ABOVE `owner` drive events; a mesh-writing modal there has control.
    Everything below `owner` started earlier, so an add-on's always-running modal
    (HUD, key display, file watcher) never blocks us. At invoke time (owner not on
    the stack yet) Python modals have just passed our key through and are idle.
    Native modals holding raw edit-mesh pointers block wherever they are, in any
    window: our TIMER events still run while one is active in another window.
    View / WM modals and `allowed` operator keys are ignored.
    """
    try:
        if context.edit_object is None:
            return True
        window = context.window
        windows = list(context.window_manager.windows)
        if window is None and not windows:
            return False   # no window (scripts, background): no modal operator can be running
        own_key = operator_key(owner) if owner is not None else None
        if window is not None:
            stack = list(window.modal_operators)
            keys = [operator_key(op) for op in stack]
            above = keys.index(own_key) if own_key in keys else 0
            for index, (op, key) in enumerate(zip(stack, keys)):
                if key == own_key or key in allowed or _is_safe_modal(op):
                    continue
                if index < above and modal_may_write_mesh(op):
                    return True
                if _holds_mesh(op):
                    return True
        for other in windows:
            if window is not None and other == window:
                continue
            for op in other.modal_operators:
                if _holds_mesh(op):
                    return True
        return False
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        return True


def _pointer(value):
    return value.as_pointer() if value is not None else 0


def _matrix(matrix):
    return tuple(tuple(row) for row in matrix)


def _snap_filters(context):
    ts = context.scene.tool_settings
    settings = getattr(context.scene, "ppo", None)
    return (ts.use_snap, getattr(ts, "use_snap_selectable", False),
            getattr(settings, "snap_only_selected", False),
            getattr(settings, "ignore_derived", False),
            _pointer(getattr(settings, "source_object", None)),
            _pointer(getattr(settings, "source_collection", None)))


def source_key(context):
    """(members, content) of the snap / occlusion sources, read only after the
    depsgraph handler saw an update outside the edit mesh.

    members: original object pointers (an object deleted, hidden, added): evaluated
    references the tools hold must be replaced at once. content: per-source
    geometry / transform revisions, selection and snap filters: cached copies
    (Prepared matrices, Relax source BVHs) are refreshed, nothing is stopped.
    """
    from . import raycast
    members, content = [], []
    for obj in raycast.sources(context, visibility=True):
        original = getattr(obj, "original", None) or obj
        key = _pointer(original)
        members.append(key)
        content.append((_revisions.get(key, 0), _revisions.get(_pointer(original.data), 0),
                        original.select_get(), original.hide_select))
    return tuple(members), (tuple(content), _snap_filters(context))


FINGERPRINT_VERTS = 64   # sampled vertices per fingerprint
FINGERPRINT_ELEMS = 16   # sampled edges and faces (their storage order)


def _rounded(co):
    return (round(co[0], 7), round(co[1], 7), round(co[2], 7))


def _sampled(seq, count):
    """Up to `count` elements evenly spaced in storage order (deterministic)."""
    n = len(seq)
    if not n:
        return ()
    seq.ensure_lookup_table()   # rebuilds only a dirty table; never touches BMElem.index
    if n <= count:
        return [seq[i] for i in range(n)]
    return [seq[i * n // count] for i in range(count)]


def geometry_sample(bm):
    """a cheap same-count edit detector, O(K) per call.

    Sampled vertices: rounded local coordinates, link-edge count and the first link
    edge's other vertex (order / connectivity, without BMElem.index: draw code may
    call index_update between events), first loop UV on the active UV map. Sampled
    edges and faces catch their own reorders (sort_elements). Selection is not part
    of it (box select during a tool is not an edit). Sampled, so a coordinate edit
    away from the samples can pass; reorders and whole-mesh edits are caught.
    """
    uv = bm.loops.layers.uv.active
    verts = []
    for v in _sampled(bm.verts, FINGERPRINT_VERTS):
        edges = v.link_edges
        item = (_rounded(v.co), len(edges), _rounded(edges[0].other_vert(v).co) if len(edges) else None)
        if uv is not None:
            loops = v.link_loops
            item += ((round(loops[0][uv].uv[0], 7), round(loops[0][uv].uv[1], 7)) if len(loops) else None,)
        verts.append(item)
    return (tuple(verts),
            tuple(_rounded(e.verts[0].co) + _rounded(e.verts[1].co) for e in _sampled(bm.edges, FINGERPRINT_ELEMS)),
            tuple((len(f.verts), _rounded(f.verts[0].co)) for f in _sampled(bm.faces, FINGERPRINT_ELEMS)))


def mesh_stamp(bm):
    """Fingerprint: element counts, pin/crease/glue/shape layer presence and a sampled
    geometry fingerprint (geometry_sample, O(K)).

    Hashing every vertex, edge and face twice per mouse event costs ~120 ms per event
    on a 10k-vertex mesh, and counts alone miss same-count outside edits (coordinates / UV by another add-on's timer or a
    Python modal below ours, sort_elements); the sample catches reorders and most
    coordinate / UV edits at a fixed cost.
    """
    return (len(bm.verts), len(bm.edges), len(bm.faces), tuple(bm.verts.layers.shape.keys()), geometry_sample(bm),
            tuple(name for name in ("ppo_pins", "crease_vert")
                  if bm.verts.layers.float.get(name) is not None),
            bool(getattr(bm.verts.layers, "int", None) is not None and bm.verts.layers.int.get("ppo_glue") is not None),
            bool(getattr(getattr(bm.edges, "layers", None), "int", None) is not None
                 and bm.edges.layers.int.get("ppo_glue_edges") is not None))


SOURCE_REFRESH_INTERVAL = 0.1   # s; source content refreshes at most this often (members at once)


class MeshStateGuard:
    """Baseline after our event; compare before the next event.

    Only changes to the mesh being edited stop or reset a
    tool (mesh_changed). Changes to other objects (sources) never do: poll_sources
    tells the caller to refresh its source references and caches instead.
    """
    def __init__(self, context):
        self.obj = context.edit_object
        self.data = self.obj.data
        self.bm = bmesh.from_edit_mesh(self.data)
        _watched.clear()
        _watched.update((_pointer(self.obj), _pointer(self.data)))
        self.source_gen = _foreign_generation
        try:
            self.sources = source_key(context)
        except (ReferenceError, RuntimeError, AttributeError, TypeError):
            self.sources = None
        self.source_time = 0.0
        self.accept(context)

    def accept(self, context):
        self.mesh = mesh_stamp(self.bm)
        self.transform = _matrix(self.obj.matrix_world)
        self.epoch = _epoch
        self.view = self._view(context)

    @staticmethod
    def _view(context):
        rv3d = context.region_data
        return _matrix(rv3d.perspective_matrix) if rv3d is not None else None

    def mesh_changed(self, context):
        """The edit mesh itself: object / data replaced, left Edit Mode, BMesh freed or
        rebuilt, undo / redo / file load, element counts or pin layers, the sampled
        geometry fingerprint (same-count outside edits), its transform.
        Element references a tool holds may be stale: refresh or end the tool."""
        try:
            if (context.mode != "EDIT_MESH" or context.edit_object != self.obj
                    or self.obj.data != self.data or not self.bm.is_valid
                    or _epoch != self.epoch):
                return True
            bm = bmesh.from_edit_mesh(self.data)
            return (bm is not self.bm or mesh_stamp(bm) != self.mesh
                    or _matrix(self.obj.matrix_world) != self.transform)
        except (ReferenceError, RuntimeError, AttributeError):
            return True

    def changed(self, context, include_view=False):
        try:
            return self.mesh_changed(context) or (include_view and self._view(context) != self.view)
        except (ReferenceError, RuntimeError, AttributeError):
            return True

    def poll_sources(self, context, now):
        """None, "members" or "content". O(1) unless the depsgraph handler saw an
        update outside the edit mesh since the last baseline."""
        if _foreign_generation == self.source_gen:
            return None
        try:
            key = source_key(context)
        except (ReferenceError, RuntimeError, AttributeError, TypeError):
            key = None
        old = self.sources
        if key is not None and key == old:
            self.source_gen = _foreign_generation
            return None
        members = key is None or old is None or key[0] != old[0]
        if not members and now - self.source_time < SOURCE_REFRESH_INTERVAL:
            return None   # retried on a later event; no evaluated reference went away
        self.sources, self.source_gen, self.source_time = key, _foreign_generation, now
        return "members" if members else "content"


def restore_allowed(op, context):
    """True when a rollback that spans events may run.

    Checked right before Esc / wheel-back / cancel restores. False when the edit mesh
    no longer matches the baseline taken after our last event (an outside edit: the
    restore would overwrite it, or scramble it through stale element handles): the
    caller keeps the outside edits and stops the tool. A rollback inside our own
    event (an error mid-edit: _ppo_internal_rollback) is always allowed; nothing
    outside runs during our event. Tools without a guard are not checked here.
    """
    if getattr(op, "_ppo_internal_rollback", False):
        # not when the error came before this event's guard check passed (fail safe)
        return not getattr(op, "_ppo_unchecked", False)
    guard = getattr(op, "_ppo_guard", None)
    if guard is None:
        return True
    return not guard.changed(context)


def _drop_source_cache(name):
    """Forget derived data of one changed source."""
    try:
        from . import snap
        snap._cache.pop(name, None)
    except Exception:
        pass   # a cache miss only costs a rebuild


def _on_depsgraph(_scene, depsgraph):
    """Bookkeeping only: revision counters and source cache invalidation. Never stops
    a tool and never touches BMesh or evaluated data beyond the update list."""
    global _generation, _foreign_generation
    try:
        foreign = False
        for update in depsgraph.updates:
            original = update.id.original or update.id
            key = _pointer(original)
            outside = key not in _watched
            foreign = foreign or outside
            geometry = update.is_updated_geometry
            if not (geometry or update.is_updated_transform):
                continue
            _generation += 1
            _revisions[key] = _generation
            if geometry and outside and isinstance(original, bpy.types.Object):
                _drop_source_cache(original.name)
        if len(_revisions) > 8192:
            _revisions.clear()
            foreign = True
        if foreign:
            _foreign_generation += 1
    except (ReferenceError, AttributeError, RuntimeError):
        _foreign_generation += 1


def _invalidate(*_args):
    """Undo / redo / file load: every BMesh wrapper and element reference is stale."""
    global _epoch, _foreign_generation
    _epoch += 1
    _foreign_generation += 1
    _revisions.clear()
    try:
        from . import snap
        snap.clear_cache()
    except Exception:
        pass


_HANDLERS = (("depsgraph_update_post", _on_depsgraph),
             ("undo_pre", _invalidate), ("redo_pre", _invalidate),
             ("load_pre", _invalidate))


def start():
    """while PolyTailor is in use (session); what changed meanwhile is not known: start clean."""
    _invalidate()
    for name, callback in _HANDLERS:
        callback = bpy.app.handlers.persistent(callback)
        handlers = getattr(bpy.app.handlers, name)
        if callback not in handlers:
            handlers.append(callback)


def stop():
    for name, callback in reversed(_HANDLERS):
        handlers = getattr(bpy.app.handlers, name)
        if callback in handlers:
            handlers.remove(callback)
    _invalidate()


def unregister():
    stop()
