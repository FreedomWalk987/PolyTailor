# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Header text for moves, formatted like Blender's own transform header."""

from mathutils import Vector


_METRIC = (("KILOMETERS", 1000.0, "km"), ("METERS", 1.0, "m"), ("CENTIMETERS", 0.01, "cm"),
           ("MILLIMETERS", 0.001, "mm"), ("MICROMETERS", 1e-6, "µm"))
_IMPERIAL = (("MILES", 1609.344, "mi"), ("FEET", 0.3048, "'"), ("INCHES", 0.0254, '"'),
             ("THOU", 0.0000254, "thou"))


def _unit_for(context, magnitude):
    """(factor in metres, symbol): the scene's fixed length unit, or for Adaptive the
    largest unit that keeps the magnitude at or above 1 (Blender's own choice)."""
    units = context.scene.unit_settings
    table = _METRIC if units.system == "METRIC" else _IMPERIAL
    for name, factor, symbol in table:
        if units.length_unit == name:
            return factor, symbol
    for _name, factor, symbol in table:
        if magnitude >= factor:
            return factor, symbol
    return table[-1][1], table[-1][2]


def length(context, value, unit=None):
    """A length in the scene's units, e.g. "3.713 mm" (like the native header)."""
    units = context.scene.unit_settings
    if units.system not in {"METRIC", "IMPERIAL"}:
        return f"{value:.4f}"
    metres = value * units.scale_length
    factor, symbol = unit or _unit_for(context, abs(metres))
    number = f"{metres / factor:.4g}" if abs(metres / factor) >= 1e-4 else ("-0" if metres < 0 else "0")
    return f"{number} {symbol}" if symbol not in {"'", '"'} else f"{number}{symbol}"


def offset_text(context, delta):
    """"Dx: … Dy: … Dz: … (total)" for a world-space displacement."""
    delta = Vector(delta)
    units = context.scene.unit_settings
    unit = None
    if units.system in {"METRIC", "IMPERIAL"}:
        # one unit for all components, chosen from the total (Adaptive) or fixed
        unit = _unit_for(context, delta.length * units.scale_length)
    return (f"Dx: {length(context, delta.x, unit)}   Dy: {length(context, delta.y, unit)}   "
            f"Dz: {length(context, delta.z, unit)} ({length(context, delta.length, unit)})")


def mean_offset(mw, pairs):
    """Average world displacement of (vertex, original local co) pairs."""
    total, count = Vector((0.0, 0.0, 0.0)), 0
    for v, original in pairs:
        if v.is_valid:
            total += mw @ v.co - mw @ original
            count += 1
    return total / count if count else total


def show(context, text):
    area = context.area
    if area is not None:
        area.header_text_set(text)
