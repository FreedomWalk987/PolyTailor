# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Geometry Nodes modifier inputs: public RNA in 5.2, ID properties in older Blender.

Choose by API capability. Never fall back to legacy storage when the new API is
present but not ready: that can silently lose an object input's binding.
"""

_LEGACY = object()
_MISSING = object()


def _inputs(modifier):
    properties = getattr(modifier, "properties", _LEGACY)
    if properties is _LEGACY:
        return _LEGACY
    return getattr(properties, "inputs", None) if properties is not None else None


def input_values(modifier):
    """Values for dependency discovery, including object/collection input sockets."""
    inputs = _inputs(modifier)
    if inputs is not _LEGACY:
        if inputs is None:
            return  # An unassigned modifier has no input bindings to inspect.
        tree = getattr(modifier, "node_group", None)
        interface = getattr(tree, "interface", None)
        identifiers = [item.identifier for item in getattr(interface, "items_tree", ())
                       if getattr(item, "in_out", None) == "INPUT"
                       and getattr(item, "identifier", None)]
        if not identifiers:
            # Public RNA introspection also handles dynamically exposed input
            # structures when no node interface items are available.
            rna = getattr(inputs, "bl_rna", None)
            identifiers = [p.identifier for p in getattr(rna, "properties", ())
                           if p.identifier != "rna_type" and p.type == "POINTER"]
        for identifier in identifiers:
            socket = getattr(inputs, identifier, None)
            value = getattr(socket, "value", _MISSING)
            if value is not _MISSING:
                yield value
        return
    try:
        keys = tuple(modifier.keys())
    except (TypeError, AttributeError):
        return
    for key in keys:
        try:
            value = modifier[key]
        except (KeyError, TypeError, ReferenceError):
            continue
        yield value
