# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Measured wheel geometry in unscaled UI units; shared drawing/hit bounds."""

CARDS = (("Build", "ppo.polypen", -176, 70, 136),
         ("Fill", "ppo.fill", -204, 0, 160),
         ("Relax / Tweak", "ppo.relax", -176, -70, 184))


def arrange(heights, size, origin, preferred_scale, settings_width, reserved_height, insets=(0, 0, 0, 0)):
    """Fit complete bounds, including both brush columns and the status footer.

    The maximum settings tree is measured without changing RNA. All panels keep
    their anchors throughout a session, including mode changes and numeric edits.
    Very small viewports reduce scale instead of clipping interactive controls.
    """
    fixed, snapping, selection, settings = heights
    south_top = min(-128, -4 - selection - 24)
    top = max(154 + fixed, 12 + snapping, 85)
    bottom = min(south_top - max(settings, reserved_height) - 27, -85)
    left = min(-284, -settings_width / 2)
    right = max(393, settings_width / 2)
    margin = 12
    il, ib, ir, it = insets   # overlapping header / toolbar / sidebar regions (visible part only)
    scale = min(preferred_scale, max(1, size[0]-il-ir-2*margin)/(right-left),
                max(1, size[1]-ib-it-2*margin)/(top-bottom))
    cx = max(il+margin-left*scale, min(size[0]-ir-margin-right*scale, origin[0]))
    cy = max(ib+margin-bottom*scale, min(size[1]-it-margin-top*scale, origin[1]))
    anchors = ((-124, 154 + fixed, 248),
               (145, 12 + snapping, 248),
               (145, -4, 248),
               (-settings_width/2, south_top, settings_width))
    placements = [(cx+x*scale, cy+y*scale, width*scale) for x,y,width in anchors]
    cards = [(label, tool, (cx+(x-width/2)*scale, cy+(y-15)*scale,
                            cx+(x+width/2)*scale, cy+(y+15)*scale))
             for label,tool,x,y,width in CARDS]
    return scale, (cx,cy), placements, cards
