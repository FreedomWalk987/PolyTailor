# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty


class PPO_BrushSettings(bpy.types.PropertyGroup):
    radius: IntProperty(translation_context="Poly Toolset", 
        name="Radius", description="Brush radius in pixels",
        default=100, min=5, max=1000, subtype="PIXEL",
    )
    strength: FloatProperty(translation_context="Poly Toolset", 
        name="Strength", description="How strongly the brush acts",
        default=0.75, min=0.01, max=1.0, subtype="FACTOR",
    )
    falloff: FloatProperty(translation_context="Poly Toolset", 
        name="Falloff", description="0 = hard edge, 1 = smooth fade toward the rim",
        default=1.0, min=0.0, max=1.0, subtype="FACTOR",
    )


class PPO_RelaxBrushSettings(bpy.types.PropertyGroup):
    radius: IntProperty(translation_context="Poly Toolset", name="Radius", description="Brush radius in pixels",
                        default=100, min=5, max=1000, subtype="PIXEL")
    strength: FloatProperty(translation_context="Poly Toolset", name="Strength", description="How strongly the brush acts",
                            default=0.5, min=0.01, max=1, subtype="FACTOR")
    falloff: FloatProperty(translation_context="Poly Toolset", name="Falloff", description="0 = hard edge, 1 = smooth fade toward the rim",
                           default=1, min=0, max=1, subtype="FACTOR")


FACES_CONTEXT = "Poly Toolset Faces"


def _pen_mode_get(settings):
    return 0 if settings.pen3d else 1


def _pen_mode_set(settings, value):
    target = value == 0
    if settings.pen3d != target:
        settings.pen3d = target


class PPO_SceneSettings(bpy.types.PropertyGroup):
    tweak: PointerProperty(type=PPO_BrushSettings)
    relax: PointerProperty(type=PPO_RelaxBrushSettings)
    source_object: PointerProperty(translation_context="Poly Toolset", name="Source Object", type=bpy.types.Object)
    source_collection: PointerProperty(translation_context="Poly Toolset", name="Source Collection", type=bpy.types.Collection)
    snap_only_selected: BoolProperty(translation_context="Poly Toolset", name="Selected Sources Only", default=False)
    clamp_to_source: BoolProperty(translation_context="Poly Toolset", 
        name="Stop at Edge",
        description="With surface snapping on, points moved past the edge of the source stop at the "
                    "edge instead of leaving the surface",
        # New scenes start in 3D, where it is off; Topo turns it on from its backup.
        default=False, update=lambda self, context: _clamp_changed(context))
    ignore_derived: BoolProperty(translation_context="Poly Toolset", 
        name="Ignore Derived Objects",
        description="Do not snap to objects generated from the mesh being edited: objects sharing "
                    "its mesh data, or whose modifiers / geometry nodes read it (e.g. a projected copy)",
        default=True)
    snap_self: BoolProperty(translation_context="Poly Toolset", 
        name="Snap to Itself",
        description="Vertex / Edge / Edge Center snapping also uses the mesh being edited "
                    "(not the vertices being moved)",
        default=False)
    use_native_snap: BoolProperty(translation_context="Poly Toolset", 
        name="Use Blender Snapping",
        description="Left-drag, G and E use Blender's own move for Vertex / Edge / Edge Center "
                    "snapping (Ignore Derived Objects does not apply). "
                    "Build placement keeps PolyTailor's snapping",
        default=False)
    pen3d: BoolProperty(translation_context="Poly Toolset", 
        name="3D",
        description="Model instead of retopologize: no snapping to other objects, no Auto Merge "
                    "and no Stop at Edge in any PolyTailor tool; Relax keeps the mesh on its own shape. "
                    "Turn off for the Topo defaults",
        default=True, update=lambda self, context: _pen3d_changed(self, context))
    pen_mode: EnumProperty(translation_context="Poly Toolset", 
        name="Modeling Mode",
        items=(("PEN3D", "3D", "Model freely without snapping to other objects", "NONE", 0),
               ("TOPO", "Topo", "Retopologize using the Topo defaults", "NONE", 1)),
        get=_pen_mode_get, set=_pen_mode_set, options={"SKIP_SAVE"})
    # snapping / Auto Merge before 3D was turned on (restored when it is turned off)
    pen3d_backup_ready: BoolProperty(options={"HIDDEN"}, default=False)
    pen3d_prev_snap: BoolProperty(options={"HIDDEN"}, default=True)
    pen3d_prev_merge: BoolProperty(options={"HIDDEN"}, default=True)
    pen3d_prev_clamp: BoolProperty(options={"HIDDEN"}, default=True)
    # Auto Merge chosen by hand in 3D / Topo (-1: none, follow the tool; 0 off; 1 on).
    # Kept across tool, 3D/Topo and Edit Mode switches until changed by hand again.
    merge_manual_3d: IntProperty(options={"HIDDEN"}, default=-1, min=-1, max=1)
    merge_manual_topo: IntProperty(options={"HIDDEN"}, default=-1, min=-1, max=1)
    # Retopology overlay shading chosen by hand, the same way (default: on in Topo, off in 3D)
    retopo_manual_3d: IntProperty(options={"HIDDEN"}, default=-1, min=-1, max=1)
    retopo_manual_topo: IntProperty(options={"HIDDEN"}, default=-1, min=-1, max=1)
    defaults_applied: BoolProperty(options={"HIDDEN"}, default=False)  # see defaults.apply_scene
    merge_radius: IntProperty(translation_context="Poly Toolset", name="Merge Radius", description="Auto Merge radius in screen pixels",
                              default=10, min=1, max=60, subtype="PIXEL")
    include_corners: BoolProperty(translation_context="Poly Toolset", name="Corners", default=True)
    include_seams: BoolProperty(translation_context="Poly Toolset", name="Seams", default=True)
    include_sharps: BoolProperty(translation_context="Poly Toolset", name="Sharps", default=True)
    include_creases: BoolProperty(translation_context="Poly Toolset", name="Creases", default=True)
    smooth_method: EnumProperty(translation_context="Poly Toolset", name="Alt+Wheel Smooth", items=(
        ("LOOPTOOLS", "LoopTools Relax", "Use the embedded LoopTools Relax algorithm; no extension required"),
        ("RF", "Brush Relax", "Use the same relaxation as the Relax brush"),
        ("GRID", "Grid Relax", "Use the Relax tool's Grid Relax (Alt): even, square quads and straight edge lines"),
        ("NATIVE", "Native", "Use Blender's Smooth Vertices")), default="RF")
    smooth_iterations: IntProperty(translation_context="Poly Toolset", name="Iterations", description="Relax passes per brush step",
                                   default=2, min=1, max=10)
    relax_even_spacing: BoolProperty(translation_context="Poly Toolset", name="Even Out Spacing", description="Even out uneven spacing along each row of quads while keeping row widths (holding loops, density changes); off: keep every quad's own size and proportions",
                                     default=True)

    mask_selected: EnumProperty(translation_context="Poly Toolset", 
        name="Selected",
        description="Which vertices the brushes may affect, based on selection",
        items=(
            ("ALL", "All", "Affect selected and unselected vertices", "RESTRICT_SELECT_OFF", 0),
            ("EXCLUDE", "Unselected", "Leave selected vertices alone", "RESTRICT_SELECT_ON", 1),
            ("ONLY", "Only Selected", "Affect only selected vertices", "CHECKBOX_HLT", 2),
        ),
        default="ALL",
    )
    mask_boundary: EnumProperty(translation_context="Poly Toolset", 
        name="Boundary",
        description="How brushes treat vertices on the mesh border",
        items=(
            ("INCLUDE", "Move", "Boundary vertices move freely", "MOD_MESHDEFORM", 0),
            ("SLIDE", "Slide", "Boundary vertices slide along the border", "SNAP_EDGE", 1),
            ("EXCLUDE", "Lock", "Boundary vertices stay in place", "LOCKED", 2),
        ),
        default="INCLUDE",
    )
    build_stroke_density: FloatProperty(translation_context="Poly Toolset", 
        name="Stroke Density", default=4.0, min=0.01, max=1000.0, precision=1,
        description="Segments per screen length: 4 gives one segment every 80 pixels; bends add more. The model's size does not matter",
    )
    build_stroke_corner_angle: FloatProperty(translation_context="Poly Toolset", 
        name="Corner Sensitivity", subtype="ANGLE", default=0.9599310886,
        min=0.1745329252, max=2.617993878,
        description="Minimum turn angle for a fixed stroke corner; lower angles detect more corners",
    )
    merge3_faces: EnumProperty(
        name="Faces",
        description="Faces built from a selected edge",
        # Own context: "Quad" reads 四邊形 here without overriding Blender's own "Quad".
        translation_context=FACES_CONTEXT,
        items=(
            ("MIXED", "Tri/Quad", "Mixed triangles and quads: build a triangle, click again to turn it into a quad", 0),
            ("QUAD", "Quad", "Quads only: build quad strips from the selected edge", 1),
        ),
        default="MIXED",
    )
    quad_stability: FloatProperty(translation_context="Poly Toolset", name="Quad Stability", default=1.0, min=0.0, max=1.0,
        description="Quad mode: how much new quads fan out when the cursor is off to one side; at 1 the far edge stays square to the cursor direction (least fan-out)")
    knife_constrain: BoolProperty(translation_context="Poly Toolset", name="Constrain", default=True,
        description="Keep split vertices on their original edges while placing")
    knife_junctions: BoolProperty(translation_context="Poly Toolset", name="Junctions", default=True,
        description="Preserve quad junctions when cutting adjacent edges")
    include_occluded: BoolProperty(translation_context="Poly Toolset", 
        name="Affect Hidden Vertices", description="Brushes also affect vertices hidden behind the surface",
        default=False,
    )


def sync_native_clamp(context, scene=None):
    """Blender's own S / R only stay on the source when snapping affects Rotate and
    Scale (each vertex snaps to the nearest source point). Tie that to Stop at Edge,
    so turning it off also lets S / R move points off the source."""
    scene = scene or context.scene
    clamp = scene.ppo.clamp_to_source
    ts = scene.tool_settings
    if ts.use_snap_rotate != clamp or ts.use_snap_scale != clamp:
        from . import session
        session.ensure_backup(scene)   # the user's values before PolyTailor's first write
    if ts.use_snap_rotate != clamp:
        ts.use_snap_rotate = clamp
    if ts.use_snap_scale != clamp:
        ts.use_snap_scale = clamp


def apply_snap_targets(scene):
    """3D snaps to every object, the edited mesh included; Topo only to other objects
    (the source), never to the mesh being built."""
    ts, pen3d = scene.tool_settings, scene.ppo.pen3d
    for name, value in (("use_snap_self", pen3d), ("use_snap_edit", pen3d), ("use_snap_nonedit", True)):
        if getattr(ts, name, value) != value:
            from . import session
            session.ensure_backup(scene)   # the user's values before PolyTailor's first write
            setattr(ts, name, value)


def apply_pen3d(scene):
    """3D on: snapping, Auto Merge and Stop at Edge off (called on toggle and when
    PolyTailor starts)."""
    from . import session
    ts = scene.tool_settings
    if scene.ppo.clamp_to_source:
        scene.ppo.clamp_to_source = False
    if ts.use_snap:
        session.ensure_backup(scene)   # the user's values before PolyTailor's first write
        ts.use_snap = False
    session.set_merge(scene, session.wanted_merge(scene, pen3d=True))


def begin_pen3d(scene, merge=None, clamp=None):
    """Save Topo settings once before entering 3D mode, including initial entry."""
    settings, ts = scene.ppo, scene.tool_settings
    ready = settings.pen3d_backup_ready
    if not ready:
        settings.pen3d_prev_snap = ts.use_snap
        settings.pen3d_prev_merge = ts.use_mesh_automerge if merge is None else merge
        settings.pen3d_prev_clamp = settings.clamp_to_source if clamp is None else clamp
    settings.pen3d_backup_ready = True
    apply_pen3d(scene)


def _pen3d_changed(settings, context):
    # like Stop at Edge: Blender's settings are only touched while PolyTailor manages this scene
    # (session.sync applies 3D when PolyTailor starts in it)
    from . import session
    scene = context.scene
    if scene.session_uid not in session._scenes:
        if not settings.pen3d:
            settings.pen3d_backup_ready = False
        return
    if settings.pen3d:
        begin_pen3d(scene)
    else:
        session.apply_topo_defaults(context, scene)
    apply_snap_targets(scene)
    sync_native_clamp(context)


def _clamp_changed(context):
    # only while PolyTailor's session manages this scene's tool settings (they are restored on
    # leave); outside it, toggling the option must not change Blender's snapping
    from . import session
    if context.scene.session_uid in session._scenes:
        sync_native_clamp(context)


def settings(context=None):
    """The scene's PolyTailor settings: every PolyTailor panel, tool and operator reads them here."""
    return (context or bpy.context).scene.ppo


_classes = (PPO_BrushSettings, PPO_RelaxBrushSettings, PPO_SceneSettings)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.ppo = PointerProperty(type=PPO_SceneSettings)


def unregister():
    del bpy.types.Scene.ppo
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
