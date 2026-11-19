# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Small RNA-aware layout tree shared by the persistent wheel and its submenus.

Existing tool draw functions describe the controls. This adapter records those
descriptions without invoking native popup UI or duplicating tool behavior.
"""
from types import SimpleNamespace
from functools import lru_cache

import bpy
from ..i18n import tr as iface_   # add-on context first


BRUSH_CHOICE_ICONS = {
    "mask_selected": {"ALL": "RESTRICT_SELECT_OFF", "EXCLUDE": "RESTRICT_SELECT_ON", "ONLY": "CHECKBOX_HLT"},
    "mask_boundary": {"INCLUDE": "MOD_MESHDEFORM", "SLIDE": "SNAP_EDGE", "EXCLUDE": "LOCKED"},
}


@lru_cache(maxsize=None)
def icon_name(value):
    if isinstance(value, str):
        return "" if value == "NONE" else value
    icons = bpy.types.UILayout.bl_rna.functions['prop'].parameters['icon'].enum_items
    return next((item.identifier for item in icons if item.value == value and item.identifier != "NONE"), "")


def format_label(value, text_ctxt="*"):
    from ..i18n import tr
    label = tr(str(value), text_ctxt)
    view = bpy.context.preferences.view
    return label.replace(": ", "：").replace(":", "：") if view.use_translate_interface and view.language.startswith("zh") else label


def value_label(value):
    view = bpy.context.preferences.view
    separator = "：" if view.use_translate_interface and view.language.startswith("zh") else ":"
    return format_label(value).rstrip(":： ")+separator if value else ""


def numeric_text(node, value):
    rna = node.rna
    units = bpy.context.scene.unit_settings
    if rna.type == "INT":
        result = str(value)
    elif rna.subtype == "ANGLE":
        if units.system_rotation == "RADIANS":
            return f"{value:.{rna.precision}f}".rstrip("0").rstrip(".")+" rad"
        return bpy.utils.units.to_string(units.system,"ROTATION",value,precision=rna.precision)
    elif rna.unit == "LENGTH" and units.system != "NONE":
        return bpy.utils.units.to_string(units.system,"LENGTH",value*units.scale_length,precision=rna.precision)
    elif rna.subtype == "FACTOR":
        result = f"{value:.2f}"
    else:
        result = f"{value:.{min(6,rna.precision)}f}"
        if result in {"", "-0"}:
            result = "0"
    return result+(" px" if rna.subtype == "PIXEL" else "")


def parse_numeric(node, value):
    units = bpy.context.scene.unit_settings
    if node.rna.subtype == "ANGLE":
        return bpy.utils.units.to_value(units.system,"ROTATION",value,
                                       str_ref_unit=numeric_text(node,getattr(node.owner,node.name)))
    if node.rna.unit == "LENGTH" and units.system != "NONE":
        reference = numeric_text(node,getattr(node.owner,node.name))
        return bpy.utils.units.to_value(units.system,"LENGTH",value,str_ref_unit=reference)/units.scale_length
    return float(value.removesuffix(" px"))


class Layout:
    def __init__(self, kind="column", **kwargs):
        self.kind, self.children = kind, []
        self.enabled = self.active = True
        self.rect = (0, 0, 0, 0)
        self.is_enabled = True
        self.text, self.description = "", ""
        self.parent = None
        self.align = False
        self.__dict__.update(kwargs)

    def add(self, kind, **kwargs):
        node = Layout(kind, **kwargs)
        node.parent = self
        self.children.append(node)
        return node

    def row(self, **kwargs):
        node = self.add("row",align=kwargs.get("align",False))
        if kwargs.get("heading"):
            node.label(text=kwargs["heading"], text_ctxt="Poly Toolset")
        return node

    def column(self, **kwargs):
        return self.add("column",align=kwargs.get("align",False))

    def box(self):
        return self.add("box")

    def separator(self, **kwargs):
        return self.add("separator")

    def label(self, text="", **kwargs):
        ctxt = kwargs.get("text_ctxt", "*")   # a specific context (Faces, Controls, Select Mode) as in native panels
        return self.add("label", text=format_label(text, "*" if ctxt == "Poly Toolset" else ctxt), icon=kwargs.get("icon",""),
                        heading=not self.children and self.parent is not None and self.parent.kind=="box")

    def operator(self, idname, text=None, **kwargs):
        import bpy
        namespace, name = idname.split(".")
        operator = getattr(getattr(bpy.ops, namespace), name)
        rna = operator.get_rna_type()
        options = SimpleNamespace()
        self.add("operator", idname=idname, options=options, enabled=operator.poll(),
                 text=(format_label(rna.name if text is None else text, kwargs.get("text_ctxt", "*"))
                       if kwargs.get("translate", True) else (rna.name if text is None else text)),
                 description=iface_(rna.description), selected=kwargs.get("depress", False),
                 icon=kwargs.get("icon",""),icon_value=kwargs.get("icon_value",0))
        return options

    def prop(self, owner, name, **kwargs):
        rna = owner.bl_rna.properties[name]
        text = kwargs.get("text", rna.name)
        if rna.type == "ENUM" and kwargs.get("expand"):
            group = self.add("row" if self.kind == "row" else "column",align=True)
            from .pie import MASK_CHOICE_TEXT   # the same short words wherever the choices are shown as text
            short = MASK_CHOICE_TEXT.get(name, {})
            for item in rna.enum_items:
                if item.identifier:
                    group.add("choice", owner=owner, name=name, value=item.identifier,
                              text=format_label(short.get(item.identifier,item.name or item.identifier),
                                                getattr(rna, "translation_context", "") or "*"),
                              description=iface_(item.description), flag=rna.is_enum_flag,
                              icon=BRUSH_CHOICE_ICONS.get(name, {}).get(item.identifier, icon_name(item.icon)),
                              icon_only=kwargs.get("icon_only",False))
            return group
        return self.add("prop", owner=owner, name=name, rna=rna,
                        text=format_label(text, kwargs.get("text_ctxt", "*")), description=iface_(rna.description),slider=kwargs.get("slider",False),
                        toggle=kwargs.get("toggle",rna.type=="BOOLEAN" and bool(kwargs.get("icon"))),
                        icon=kwargs.get("icon",icon_name(rna.icon)),
                        icon_only=kwargs.get("icon_only",kwargs.get("text")=="" and rna.type=="BOOLEAN"))

    def menu(self, idname, text=None, **kwargs):
        import bpy
        cls = getattr(bpy.types, idname)
        return self.add("menu", idname=idname, text=format_label(cls.bl_label if text is None else text),icon=kwargs.get("icon",""))

    def popover(self, idname, text=None, **kwargs):
        import bpy
        cls = getattr(bpy.types, idname)
        return self.add("popover", idname=idname, text=format_label(cls.bl_label if text is None else text),icon=kwargs.get("icon",""))

    def height(self, scale):
        gap = (0 if self.align else 2) * scale
        if self.kind == "row":
            return max((c.height(scale) for c in self.children), default=0)
        if self.kind in {"column", "box"}:
            padding = 4 * scale if self.kind == "box" else 0
            return sum(c.height(scale) for c in self.children) + max(0, len(self.children)-1)*gap + padding*2
        return getattr(self,"height_units",20 if getattr(self,"heading",False) else 18 if self.kind=="label" else 7 if self.kind=="separator" else 20)*scale

    def place(self, x, top, width, scale, enabled=True):
        height = self.height(scale)
        self.rect = (x, top-height, x+width, top)
        self.is_enabled = enabled and self.enabled and self.active
        gap = (0 if self.align else 2) * scale
        if self.kind == "row" and self.children:
            # Native heading + icon-only enums occupy one compact 19px slot per
            # choice, right aligned; the heading takes the remaining row width.
            if (len(self.children) == 2 and self.children[0].kind == "label"
                    and self.children[1].kind == "row" and self.children[1].children
                    and all(getattr(c, "icon_only", False) for c in self.children[1].children)):
                icons = self.children[1]
                icon_width = min(width / 2, len(icons.children) * 19 * scale)
                self.children[0].place(x, top, width-icon_width-gap, scale, self.is_enabled)
                icons.place(x+width-icon_width, top, icon_width, scale, self.is_enabled)
                return
            unit = (width-gap*(len(self.children)-1))/len(self.children)
            for i, child in enumerate(self.children):
                child.place(x+i*(unit+gap), top, unit, scale, self.is_enabled)
        elif self.kind in {"column", "box"}:
            padding = 4*scale if self.kind == "box" else 0
            y = top-padding
            for child in self.children:
                child.place(x+padding, y, width-padding*2, scale, self.is_enabled)
                y -= child.height(scale)+gap

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()


def contains(rect, xy):
    return rect[0] <= xy[0] <= rect[2] and rect[1] <= xy[1] <= rect[3]


def hit(layout, xy):
    return next((n for n in reversed(list(layout.walk()))
                 if n.kind in {"operator", "prop", "choice", "menu", "popover"}
                 and contains(n.rect, xy)), None)


def selected(node):
    if node.kind == "choice":
        value = getattr(node.owner, node.name)
        return node.value in value if node.flag else node.value == value
    if node.kind == "prop" and node.rna.type == "BOOLEAN":
        return bool(getattr(node.owner, node.name))
    return getattr(node, "selected", False)


def enum_selection_text(node):
    """Resolve the current RNA choice every redraw, including translated labels."""
    value = getattr(node.owner, node.name)
    identifiers = set(value) if node.rna.is_enum_flag else {value}
    labels = [iface_(item.name or item.identifier) for item in node.rna.enum_items
              if item.identifier in identifiers]
    return ", ".join(labels) or (", ".join(sorted(identifiers)) if identifiers else iface_("None"))


def set_choice(node):
    if node.flag:
        value = set(getattr(node.owner, node.name))
        value.symmetric_difference_update({node.value})
        setattr(node.owner, node.name, value)
    else:
        setattr(node.owner, node.name, node.value)
