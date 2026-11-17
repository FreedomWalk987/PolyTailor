# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Concave-vertex gate. the Concave Cuts candidate path was removed; its cut geometry too.
P1 (fill_p1) still uses gate(): True when the outline has no reflex turn below NECK_TURN. numpy only.
"""
import numpy as np
from .patterns import geom

NECK_TURN = -.15       # rad (8.6 deg): shallow dips may still form a neck pair


def gate(p, turns=None):
    """Cheap pre-check; True when no rule can fire."""
    turns = geom.turning(p) if turns is None else turns
    return bool(np.min(turns) >= NECK_TURN)
