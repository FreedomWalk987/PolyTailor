# PolyTailor

Retopology and modeling tools for Blender 4.5 or newer, in Mesh Edit Mode. Free and
open source (GPL-3.0-or-later). Interface in English and 繁體中文 (Traditional Chinese),
following Blender's language setting. 中文使用說明：[GUIDE_zh-TW.md](GUIDE_zh-TW.md).

Release notes: [CHANGELOG.md](CHANGELOG.md).

## Overview

- **Build** - draw new faces, edges and vertices: hold Ctrl to preview, Ctrl+click to add,
  keep dragging to place. One mode; **Faces: Tri/Quad** or **Quad**; with one selected vertex
  Ctrl+drag draws a curve stroke; Ctrl+drag near a selected edge chain sweeps a quad strip.
- **Fill** - fill existing holes or closed wire outlines, reusing their vertices.
  Select one closed outline or hover a hole; hold Ctrl for the preview. Fill looks for several
  layouts (continuing the surrounding grid, balanced corners, patterns; deep concavities are
  split) and shows the choices as they are found; Ctrl+Wheel steps through the choices.
  Ctrl+click applies the preview shown. Select two or more closed empty outlines (panels,
  which may share edges) to fill them together. An open chain must first be closed; Fill does
  not bridge two open chains. Each fill is one undo step. Fill runs part of its search in up to
  three background processes, so Blender stays responsive. They start when you use the Fill tool
  and stop when you leave it.

- **Relax** - even out the layout under the brush while keeping the surface shape:
  Robust Relax (straightens skewed quads and, by default, evens out spacing along rows while keeping holding loops); hold Alt for Grid Relax
  (even, square quads, straight edge lines). Hold Ctrl and drag for a temporary Tweak.
- **Tweak** - push the vertices under the brush.
- **Pin / Glue** - Pin protects vertices from the brushes; Glue makes a chain or loop move
  as one rigid group.
- **Alt+Wheel** in Build and Fill relaxes the selection.
- **MMB Tool Switch** (on by default) - click the middle mouse button to switch between Build
  and the brushes; Ctrl+click it to switch between Build and Fill. Dragging with MMB still
  orbits the view, and Ctrl+MMB drag still zooms.
- **Edit Mode Auto-Save** (off by default) - Blender's own Auto Save does not run while you stay
  in Edit Mode. With this option on, PolyTailor saves an auto-save copy while you use its tools,
  so File > Recover > Auto Save still finds your work. It saves only when Blender's Auto Save is
  turned on, at the same interval, and only after something has changed.

**3D or Topo mode** - **3D** models your own mesh, **Topo** retopologizes over another model (see 3D / Topo).

Press **W** in Edit Mode for the tool wheel. The bottom-left corner of the viewport shows the
current tool, its mode and its state; the keys of the current tool are listed in Blender's
status bar. To list the keys in the viewport as well, set **Viewport Help** to **Full** in the
add-on preferences (default: Compact; Off hides the corner text).

Icon colors are set in the add-on preferences, **Icon Colors**: **Custom** (default) uses
PolyTailor's own colors, which you can edit there; **Native** follows the Blender theme like
Blender's own icons.

## Install

Blender: Edit > Preferences > Get Extensions > Install from Disk, choose the zip, enable
**PolyTailor**. Needs Blender 4.5 or newer. Tested in Blender 5.2.3, on Windows only.

Extension ID: `polytailor`.

## Getting started

- Object Mode: Shift+A > Mesh > **Retopology at Cursor** / **Retopology from Active**
  (also in the sidebar's PolyTailor tab). An empty mesh is created, Edit Mode and Build
  start, and Ctrl+click on the model places the first vertex.
- Any visible mesh (or curve / text with faces) in Object Mode is a *source*: new and
  moved points land on it while snapping (the magnet) is on.

## Tools

| Tool | Use |
|---|---|
| **Build** | Hold Ctrl to preview, Ctrl+click to add, keep dragging to place. One mode (Faces: Tri/Quad or Quad; one selected vertex: curve stroke; Ctrl+drag near a selected edge or edge chain: quad strip). Keep Ctrl held to continue. **Alt+Wheel** relaxes the selection (wheel down steps back, release Alt to confirm); its method (Brush Relax, Grid Relax, LoopTools Relax, Native) is in Build's settings and the tool header's **Alt+Wheel Smooth** menu. |
| **Fill** | Select one closed outline (or several closed panels) or hover a hole, hold Ctrl to preview, Ctrl+Wheel to step through the choices found, Ctrl+click to fill. **Alt+Wheel** relaxes the selection, as in Build. |
| **Tweak** | Drag to push the vertices under the brush. |
| **Relax** | Hold LMB to relax triangles and quads with **Robust Relax**: it fixes skewed quads while keeping each quad's size and proportions; its **Even Out Spacing** (on by default) evens out spacing along each row of quads while keeping row widths. Boundary: Move (default), Slide or Lock. N-gon vertices are protected. Hold **Alt+LMB** for **Grid Relax**: edges even out, each vertex moves toward the center of its neighbors and faces become regular, in small steps along the surface (same brush and mask settings). Hold **Ctrl** and drag for a temporary Tweak. |

## Mesh editing

In the PolyTailor tools:

- Ordinary mesh selection (click, Shift+click, loop selection) follows your own Blender
  keymap, including selection add-ons; PolyTailor does not bind these keys.
- **LMB drag** (Build and Fill) moves the selected or hovered element; from empty space it
  box selects. **Shift+drag** box selects and adds to the selection; **Shift+Ctrl+drag** box
  deselects (Ctrl alone starts the preview). These drags, the Ctrl modeling gestures and the
  brush strokes can conflict with other add-ons that use the same keys.
- **G** moves on the surface, **G G** slides along edges, **X / Y / Z** lock to a
  world axis.
- **E** extrudes edges or vertices and moves them; the wheel adds cuts; **E E** offsets
  an edge boundary along the surface, to either side (pull outward or back past the
  start). Faces use Blender's own extrude.
- **Shift** while moving: one tenth of the mouse speed. The header shows Dx / Dy / Dz.
- **Alt+Wheel** (Build and Fill) relaxes the selection. The method is set in
  **Alt+Wheel Smooth** (tool settings and tool header): Brush Relax (the Relax brush's own
  algorithm; default), Grid Relax, LoopTools Relax or Native. Wheel down steps back; release
  Alt to confirm.
- **MMB click** switches between Build and the brushes: from Build or Fill to Relax (or to the
  brush you last left this way), from Relax or Tweak back to Build. **Ctrl+MMB click** switches
  between Build and Fill: from Fill to Build, from any other PolyTailor tool to Fill. A drag
  keeps Blender's orbit (MMB) and zoom (Ctrl+MMB). Turn this off with **MMB Tool Switch** in
  the add-on preferences.
- **F / Shift+F / Ctrl+F** change brush radius / strength / falloff. **Reset Brush**
  restores the defaults (Tweak 100 px / 0.75 / 1, Relax 100 px / 0.5 / 1).
- **Esc / RMB** cancels the current move, stroke or placement.

## W wheel

**W** opens the wheel at the mouse position in Mesh Edit Mode; Object Mode keeps Blender's W.

| Position | Content |
|---|---|
| Top | Constraint Tools: Pin / Off and Glue / Off |
| Upper right | Snapping: Snap, Stop at Edge, face snap mode (Face Project / Face Nearest), **3D / Topo**, Targets |
| Right | Select Mode (Vertex / Edge / Face; Shift+click adds a mode), Auto Merge, Retopo Shading, Correct Face Attributes |
| Upper left | Build |
| Left | Fill |
| Lower left | Relax / Tweak |
| Bottom | Settings of the current tool; for Relax and Tweak, both brushes side by side (Relax left, Tweak right) |

To choose a tool:

- click it, or move the mouse toward it (going past the button is fine) and click;
- or hold W, move toward a tool and release W;
- or press **7** (Build), **4** (Fill) or **1** (Relax), on the number row or the numpad.

Moving toward a tool only highlights it; nothing switches until you click or release W. A
long press of W without moving leaves the wheel open. Choosing a tool switches to it and
closes the wheel.

The wheel stays open while you change settings:

- Click buttons and toggles. Lists, Targets and other menus open inside the wheel; Esc
  closes such a menu first, and long lists scroll with the mouse wheel.
- Drag a number left or right (hold Shift for finer steps), turn the mouse wheel over it, or
  double-click it to type a value (Enter confirms, Esc cancels). Esc during a drag restores
  the original value.

Esc, right-click or a click outside the wheel closes it.

**Shift+W** opens a panel with all settings (in Edit Mode this replaces Blender's Shift+W
Bend shortcut; Bend remains in Mesh > Transform).

## 3D / Topo

The mode applies while a PolyTailor tool is active. Choosing **Topo** while a Blender
tool (Select, Move, ...) is active switches to Build, since retopology happens in
PolyTailor; choosing it inside a PolyTailor tool keeps that tool. 3D chosen in a
Blender tool is remembered for the next PolyTailor tool. Leaving PolyTailor (a Blender
tool, or out of Edit Mode) returns to 3D and to your own snapping settings; switching
between PolyTailor tools keeps the mode. Saved files keep your own snapping settings.

Snap targets follow the mode: 3D includes the active, edited and non-edited objects;
Topo only non-edited objects (the source), so points never snap to the mesh being built.

Two mutually exclusive text buttons in the W Snapping box, sidebar Snapping panel and tool
header choose modeling or retopology. New scenes start in 3D, unless the file or your saved
defaults say otherwise. While 3D is selected, no PolyTailor tool snaps to other objects, auto merges or stops
at the source edge:
Build places points at the depth of the vertex it extends from, left-drag / G / E /
Tweak move at that depth, and Relax keeps the mesh on its own shape (as it was when
the stroke began) so curved surfaces do not shrink. Selecting Topo restores the
snapping, Auto Merge and Stop at Edge you had before. Snapping can still be turned on by hand.

## Snapping

- **Surface** (Face Project / Face Nearest): points land on the source surface; with
  snapping off, or no source under the cursor, on a plane facing the view. No marker
  is drawn for this everyday case.
- **Stop at Edge** (on in Topo, off in 3D): points moved past the edge of a model stop
  on its edge. A point moved over a gap never jumps onto a model far
  behind the gap; it crosses only to a model at a similar depth.
- **Vertex / Edge / Edge Center** (Blender's Snap Targets): handled by PolyTailor in Build,
  LMB drag, G and E, with Blender's rules - within 30 px ("Snap Distance" in the add-on
  preferences) the closest element wins; on an edge its outer parts snap to the end
  vertex and its middle to the center. Every source near the cursor is searched and
  elements behind a source are skipped. Blender's symbols mark the snap (square =
  vertex, bow tie = edge, triangle = edge center). Moves follow Blender's Snap Base
  (Closest / Center / Median / Active); axis locks snap along the axis.
- **Other targets** (Increment, Grid, Face, Volume, Edge Perpendicular) hand the move
  to Blender's own transform.
- **Blender's S / R** keep each vertex on the source while Stop at Edge is on - on the
  model it started on (Blender's Snap to Same Target, set while PolyTailor is active).
- Sidebar > Snapping options:
  - **Snap to Itself** (off): also snap to the mesh being edited, except moving points.
  - **Use Blender Snapping** (off): LMB drag, G and E use Blender's own move for
    Vertex / Edge / Edge Center (Ignore Derived Objects does not apply).
  - **Ignore Derived Objects** (on): objects generated from the edited mesh - sharing
    its mesh data or reading it through a modifier / geometry nodes (e.g. a projected
    copy) - are never snapped to. The sidebar lists them.
- Sidebar > Sources limits snapping to one object, a collection, or selected /
  selectable objects. Objects above "Vertex Snap Limit" (preferences, default 300,000
  vertices, e.g. scans) are used for surface snapping only.

## Sidebar (N > PolyTailor)

PolyTailor (tools and the current tool's settings, including Build's Alt+Wheel Smooth),
Snapping, Sources, Constraint Tools, Display & Mirror, Defaults, Shortcuts. Less used sections start
folded.

## Automatic setup

While a PolyTailor tool is active, PolyTailor sets vertex + edge select mode and a 0.1 mm Auto Merge
distance; in Topo it also turns on the retopology overlay, surface snapping (other objects only) and
Auto Merge in every tool (it welds dragged, placed and moved vertices; brush strokes never merge); 3D
turns Auto Merge off. PolyTailor leaves
Correct Face Attributes as you set it (Blender's default is off). Everything
is restored when you leave PolyTailor. The setup can be turned off in the add-on preferences.

## Your own defaults

Sidebar > Defaults > **Save as Default** keeps the current PolyTailor settings (tools, brushes,
masks, snapping options and targets, Auto Merge) as your defaults for every file;
**Restore Factory Defaults** goes back to PolyTailor's. Snapping and Auto Merge defaults apply
whenever PolyTailor starts; the other settings fill in each scene the first time PolyTailor is used
in it.

## Fill several panels

Select two or more closed, empty outlines (panels) and hold Ctrl: they are previewed and filled
together, as one undo step. Panels may share edges; along a shared edge both fills use the
same existing vertices, so they meet cleanly. Every selected edge must belong to a panel
outline (no dangling connecting edges), and separate vertices that sit at the same place on a
shared edge must be merged first. The preview shows the number of panels, of shared edges
("shared seams") and of faces. If any panel cannot be filled, nothing is added.

## Pins, masks and mirror

- **Pin / Off** protects the selected vertices from Tweak, Relax and Alt+Wheel. Pins
  are stored in a mesh attribute, which is removed when no vertex is pinned. Pins are drawn
  as dots and Glue chains as lines, in the accent color.
- Brush masks: selection, boundary (move / slide / lock), corners, seams, sharps,
  creases, and vertices occluded behind the surface (**Affect Hidden Vertices**, off
  by default).
- Mirror modifiers with clipping are respected by every move; **Add Mirror**
  (Sidebar > Display & Mirror) adds one with Clipping and Merge on.

## Licensing

Code: GNU GPL v3 or later. Icons: CC0 1.0 (public domain). License texts and notices
are in `licenses/` (see `licenses/README.md` for the full list).

Credits: Alt+Wheel LoopTools Relax uses functions from Blender's LoopTools
(GPL; notice in `licenses/`).
