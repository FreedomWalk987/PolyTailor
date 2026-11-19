# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""PolyTailor icon family04: orange actions, neutral linework, semantic warnings."""
DARK = (1.0, 158/255, 52/255, 1.0)
LIGHT = (214/255, 104/255, 0.0, 1.0)
DARK_MAIN = (229/255,)*3 + (1.0,)
LIGHT_MAIN = (45/255,)*3 + (1.0,)
DARK_GREY = (130/255,)*3 + (1.0,)
DARK_MUTED = (176/255,)*3 + (1.0,)   # #B0B0B0, 5.0:1 on the default viewport
LIGHT_GREY = (100/255,)*3 + (1.0,)
WARNING = (1.0, .8, .04, 1.0)
# Fill/Build previews: a faint accent wash with a warm apricot line,
# so previews read apart from the accent outline without a hard white edge.
PREVIEW_FILL = (*DARK[:3], .14)
PREVIEW_BORDER = (1.0, 210/255, 154/255, 1.0)
# Custom tool icon colours (Preferences > Icon Colors > Custom): the family04 dark palette
ICON_MAIN = (229, 229, 229, 255)
ICON_DIM = (130, 130, 130, 255)


def close(a, b):
    return len(a) == len(b) and all(abs(x-y)<1e-5 for x, y in zip(a, b))


def light_theme(context=None):
    import bpy
    try:
        ui = (context or bpy.context).preferences.themes[0].user_interface
        c = ui.wcol_regular.inner
        return .2126*c[0]+.7152*c[1]+.0722*c[2] > .5
    except (AttributeError, IndexError, ReferenceError):
        return False


def light_viewport(context=None):
    """Text is drawn on the 3D viewport, whose background is its own theme colour."""
    import bpy
    try:
        c = (context or bpy.context).preferences.themes[0].view_3d.space.gradients.high_gradient
        return .2126*c[0]+.7152*c[1]+.0722*c[2] > .5
    except (AttributeError, IndexError, ReferenceError, TypeError):
        return light_theme(context)


def main(context=None, alpha=1.0):
    """Viewport text is light under every theme; the blurred dark shadow (core/draw.text)
    carries it on light meshes and light viewports."""
    return (*DARK_MAIN[:3], alpha)


def secondary(context=None, alpha=1.0):
    return (*(LIGHT_GREY if light_viewport(context) else DARK_GREY)[:3], alpha)


def muted(context=None, alpha=1.0):
    """Muted text rows, light under every theme (#B0B0B0)."""
    return (*DARK_MUTED[:3], alpha)


def accent(context=None, alpha=1.0):
    from . import prefs
    color = tuple(prefs.get(context).accent)
    if close(color, DARK):
        color = LIGHT if light_theme(context) else DARK
    return (*color[:3], alpha)


def preview_fill(context=None):
    from . import prefs
    color = tuple(prefs.get(context).preview_fill)
    return (*accent(context)[:3],color[3]) if close(color[:3],DARK[:3]) else color


def preview_border(context=None):
    from . import prefs
    color = tuple(prefs.get(context).preview_border)
    return main(context,color[3]) if close(color[:3],(1,1,1)) or close(color[:3],DARK_MAIN[:3]) else color


def warning(alpha=1.0):
    return (*WARNING[:3], alpha)


def preview_lines(context=None, alpha=1.0):
    """Preview Border colour with its own alpha scaled (dimmer previews)."""
    color = preview_border(context)
    return (*color[:3], color[3]*alpha)


def preview_faces(context=None, alpha=1.0):
    """Preview Fill colour with its own alpha scaled (inactive previews)."""
    color = preview_fill(context)
    return (*color[:3], color[3]*alpha)


def selection_fill(background, context=None, amount=.24):
    color = accent(context)
    return tuple((1-amount)*a+amount*b for a,b in zip(background[:3],color[:3]))+(1.0,)


def icon_colors(rgba, light):
    """Use the supplied light palette at vector-render time; preserve file bytes."""
    if not light:
        return rgba
    import numpy as np
    out = rgba.copy()
    for a,b in ((DARK_MAIN,LIGHT_MAIN),(DARK_GREY,LIGHT_GREY),(DARK,LIGHT)):
        mask = np.all(np.abs(rgba[:,:3]-a[:3])<1e-5,axis=1)
        out[mask,:3] = b[:3]
    return out
