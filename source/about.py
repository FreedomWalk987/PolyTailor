# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Public release identity."""

NAME = "PolyTailor"
VERSION = "1.1.0"


def version_label():
    """Preferences header: "Version" and the version number."""
    from .i18n import tr as iface_   # add-on context first
    return iface_("Version {version}").format(version=VERSION)   # the public version
