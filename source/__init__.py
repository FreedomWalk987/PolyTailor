# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""PolyTailor: Build, Fill and brushes for Edit Mode."""

from . import about, palette, i18n, prefs, props, core, ops, ui
from .core import (gn_inputs, face_attributes, raycast, snap, safe_mesh, pick, draw, axis, proportional, mesh,
                   extrude)
from .core.relax_lab import solver, robust, targets
from .core import (glue, brush_index, brush_bounds, brush_grid, brush_snapshot, brush_cache, brush_feedback,
                   relax_forces, smooth, looptools_relax, polypen_support_base, polypen_support, meshpen_logic)
from .core.patterns import geom, chords, solver as patterns_solver, validate
from .core import boundary_preview, fill_partition, fill
from .ops import common, brush   # brush before drag / transform: they import MirrorClip etc. from it
from .ops import drag, transform, polypen, tool_switch, fill as fill_op, pinning, glue as glue_op, newtarget
from .ui import pie, custom_layout, custom_pie, panels, hud
from . import tools, defaults, session

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
    ui.custom_pie, ui.panels, ui.hud, tools, defaults, session,
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
