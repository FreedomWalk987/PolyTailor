# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""the Shortcuts panel body, generated from the key tables (ui/keytable.py)."""

from . import keytable


def draw_shortcuts(layout, context):
    """One box per group (each tool, all tools); one label per key."""
    for heading, rows in keytable.groups(context):
        box = layout.box()
        col = box.column(align=True)
        col.label(text=heading, translate=False)
        for row in rows:
            for part in keytable.parts(row):
                col.label(text=part, translate=False)
