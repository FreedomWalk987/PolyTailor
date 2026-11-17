# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""PolyTailor: Build, Fill and brushes for Edit Mode."""

# Every module of the add-on, each after the modules it imports when it loads (generated from the source by the
# build; core/_fill_worker.py is left out: it runs only outside Blender, in Fill's own Python). Reload Scripts,
# or enabling again after a newer version was installed over a disabled copy, runs this file again but keeps the
# modules imported before: they are reloaded in this order.
_RELOAD = (
    "about", "palette", "i18n", "prefs", "props", "defaults", "edit_autosave", "core", "core.gn_inputs",
    "core.face_attributes", "core.raycast", "core.safe_mesh", "core.pick", "core.draw", "core.snap", "core.axis",
    "core.proportional", "core.mesh", "core.extrude", "core.glue", "core.brush_index", "core.brush_bounds",
    "core.brush_grid", "core.brush_snapshot", "core.brush_cache", "core.brush_feedback", "core.relax_forces",
    "core.looptools_relax", "core.polypen_support_base", "core.polypen_support", "core.meshpen_logic",
    "core.fill_flow", "core.fill_guidance", "core.fill_layouts", "core.fill_ps1_features", "core.fill_ps1_rank",
    "core.memo_jobs", "core.newfill", "core.newfill.cleanup", "core.newfill.dopt", "core.newfill.dopt_v4",
    "core.newfill.dp_fast", "core.newfill.apattern", "core.newfill.emesh5", "core.newfill.fill_r2",
    "core.newfill.fill_a", "core.newfill.flow_stage2", "core.newfill.matching", "core.newfill.fallback5",
    "core.newfill.fallback5_dopt", "core.newfill.quality", "core.newfill.pipeline", "core.newfill.topofix",
    "core.newfill.worker", "core.panel_network", "core.patterns", "core.patterns.geom", "core.patterns.chords",
    "core.patterns.solver", "core.patterns.validate", "core.fill_cls", "core.fill_cuts", "core.fill_p1",
    "core.fill_selection", "core.boundary_preview", "core.fill_lattice", "core.fill_partition", "core.fill",
    "core.network_counts", "core.network_lattice", "core.network_medial", "core.network_fill", "core.fill_field",
    "core.fill_progressive", "core.fill_multi", "core.fill_triangle", "core.network_validate",
    "core.patterns.takayama", "core.relax_lab", "core.relax_lab.solver", "core.relax_lab.robust",
    "core.relax_lab.targets", "core.smooth", "core.units", "ops", "ops.common", "ops.tool_switch", "ops.fill",
    "ops.pinning", "ops.brush", "ops.drag", "ops.transform", "ops.polypen", "ops.glue", "session", "ui", "ui.pie",
    "ops.newtarget", "ui.custom_layout", "ui.panels", "tools", "ui.fill_hud", "ui.fill_hud_draw", "ui.glyphs",
    "ui.keytable", "ui.hud", "ui.shortcuts", "ui.wheel_geometry", "ui.custom_pie",
)

if "bpy" in locals() or "_loaded" in locals():   # _loaded: an older __init__ (1.0.1) had no module-level bpy
    import importlib
    for _name in _RELOAD:
        _top, _, _rest = _name.partition(".")
        _module = locals().get(_top)
        for _part in (_rest.split(".") if _rest else ()):
            _module = getattr(_module, _part, None)
        if isinstance(_module, type(importlib)):   # imported before (a lazy module never imported is skipped)
            try:
                importlib.reload(_module)
            except ModuleNotFoundError as _exc:
                if _exc.name != _module.__name__:   # a module it imports is missing: the original error
                    raise
                # this module was removed in this version: skipped
            except Exception:   # stop at the first failure: a half-reloaded add-on must not register
                raise   # the original error
    del _name, _top, _rest, _module

import bpy

from . import about, palette, i18n, prefs, props, core, ops, ui
from .core import (gn_inputs, face_attributes, raycast, snap, safe_mesh, pick, draw, axis, proportional, mesh,
                   extrude)
from .core.relax_lab import solver, robust, targets
from .core import (glue, brush_index, brush_bounds, brush_grid, brush_snapshot, brush_cache, brush_feedback,
                   relax_forces, smooth, looptools_relax, polypen_support_base, polypen_support, meshpen_logic)
from .core.patterns import geom, chords, solver, validate
from .core import boundary_preview, fill_partition, fill
from .ops import common, brush   # brush before drag / transform: they import MirrorClip etc. from it
from .ops import drag, transform, polypen, tool_switch, fill, pinning, glue, newtarget
from .ui import pie, custom_layout, custom_pie, panels, hud
from . import tools, defaults, session, edit_autosave

# registered in this order, unregistered in reverse
_MODULES = (
    about, palette, i18n, prefs, props, core.gn_inputs, core.face_attributes, core.raycast, core.snap,
    core.safe_mesh, core.pick, core.draw, core.axis, core.proportional, core.mesh, core.extrude,
    core.relax_lab.solver, core.relax_lab.robust, core.relax_lab.targets, core.glue, core.brush_index,
    core.brush_bounds, core.brush_grid, core.brush_snapshot, core.brush_cache, core.brush_feedback,
    core.relax_forces, core.smooth, core.looptools_relax, core.polypen_support_base, core.polypen_support,
    core.meshpen_logic, core.patterns.geom, core.patterns.chords, core.patterns.solver, core.patterns.validate,
    core.boundary_preview, core.fill_partition, core.fill, ops.common, ops.brush, ops.drag, ops.transform,
    ops.polypen, ops.tool_switch, ops.fill, ops.pinning, ops.glue, ops.newtarget, ui.pie, ui.custom_layout,
    ui.custom_pie, ui.panels, ui.hud, tools, defaults, session, edit_autosave,
)


def register():
    import bpy
    from .ops.common import release_state_on_end
    for module in _MODULES:
        for value in list(vars(module).values()):
            if (isinstance(value, type) and issubclass(value, bpy.types.Operator)
                    and value.__module__ == module.__name__):
                release_state_on_end(value)
    from . import i18n
    registered = []
    try:
        for module in _MODULES:
            if hasattr(module, "register"):
                module.register()
                registered.append(module)
    except Exception:
        # Roll back what is registered (reverse order), translations last, so a
        # retry after the failure starts clean; the original error is re-raised.
        for module in reversed(registered):
            if module is not i18n and hasattr(module, "unregister"):
                try:
                    module.unregister()
                except Exception:
                    pass
        try:
            i18n.unregister()
        except Exception:
            pass
        raise


def unregister():
    from . import i18n
    try:   # end running PolyTailor modals before their classes go
        from .ops import common
        common.release_all()
    except Exception:
        pass
    first = None
    try:
        for module in reversed(_MODULES):   # every module is unregistered even if one raises
            if module is not i18n and hasattr(module, "unregister"):
                try:
                    module.unregister()
                except Exception as exc:
                    first = first or exc
                    print("PolyTailor unregister:", module.__name__, exc)
        if first is not None:
            raise first
    finally:
        # Translations were registered first and are normally removed last.
        # Always remove them even if another module cannot finish its cleanup.
        i18n.unregister()
