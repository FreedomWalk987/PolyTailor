## 1.1.0

- Topo Fill: on curved sources the fill is laid out on the source surface, so it follows the shape more closely.
- Topo Fill: an outline across a sharp crease can be filled in two parts that meet at the crease.
- Topo Fill: fewer long waits on creased outlines, and "Better result found" can come sooner.
- Topo Fill: sources with polygons of more than four corners no longer make the fill fold.
- Fill: the search indicator keeps turning while Fill is still looking for a fill, instead of a warning-colored message.
- Build: with Auto Merge off, a drag that ends on a vertex joins it, like a click there.
- Glue: Off releases only the selected part; the rest stays glued.
- Retopology Mesh (Topo) no longer changes Sidebar > Sources.
- Sidebar: a two-line hint for the W wheel and Shift+W settings (Preferences > Show Wheel Hint), and a Select Mode section.
- Slide on Surface (G G) keeps vertices on a clipped mirror center line.
- Stability fixes.

## 1.0.5

- Fill: the HUD shows whether you are in 3D or Topo mode, and the search indicator is on the left.
- Fill: a second search for a better fill now shows in its own strip below the HUD, so the main result looks finished; when it finds one, the strip says "Better result found".
- Shift+A > Mesh: New Mesh (3D) adds an empty mesh named "Mesh" at the 3D Cursor and starts in 3D mode; Retopology Mesh (Topo) adds one on the active object and starts in Topo mode with that object as the source.
- Edit Mode Auto-Save is removed.
- Fixed: when you step through the results of a wider search, the Fill HUD marks the one shown; Build no longer collapses a triangle when its new tip lands on the triangle's own corner.
- Stability fixes.

## 1.0.4

- Fill: a fill is drawn in the viewport as soon as it can be added, and Ctrl+click adds only a fill that has been shown.
- Edit Mode Auto-Save: the copy has its own file name, so it never replaces Blender's own auto-save file; when you open a copy, a message names the original file so you can save it there with File > Save As.
- Build: Auto Merge no longer merges a vertex the click only reused; in line mode (Ctrl+Wheel) an existing end vertex no longer jumps to the cursor; new vertices respect a Mirror modifier's Mirror Object.
- Relax: Grid Relax leaves the brush area alone next to a vertex with an invalid position; on large meshes Relax works on the active shape key, not the Basis.
- The Alt+Wheel operator is now called Relax Selection.
- An object whose transform has a zero scale gets a message instead of an error; saved defaults are kept when saving them fails.
- Stability fixes.

## 1.0.3

- Fill: when an even outline fits a plain grid with no poles, that grid is shown at once while the search continues.
- Fill: an outline that crosses itself, repeats a point or doubles back is refused at once, and the problem spots are marked on it.
- Fill: the ranking now also counts irregular vertices on the outline, so choices with fewer of them come first.
- Fill: after the main search, a second search keeps looking ("Improving fill preview..."). When it finds a better fill, "Better result found" appears and that fill moves to the top of the list. It replaces the preview unless you have moved to another choice.
- Fill: the edge count in the Fill HUD turns amber when the outline has an odd number of edges.
- Fill: the reason for a refusal shows at the cursor; separate outlines and a multi-panel search that runs out of time now get their own messages.
- Fill: no more long freezes on zig-zag outlines; new faces now point the right way on mirrored (negative-scale) objects.
- Fill: its background processes stop when you leave the Fill tool.
- Build: Ctrl+click in another editor goes to that editor; Quad mode no longer welds to hidden vertices; faster Ctrl hover with large selections.
- Edit Mode Auto-Save now also works on Blender 4.5 and 5.0, saves only while a PolyTailor tool is in use, and never in the middle of a stroke or drag.
- PolyTailor no longer switches off Correct Face Attributes, and your overlay, snapping and Auto Merge settings are restored reliably, also in other workspaces.
- Fixed: G G Slide could lock input after Alt+Tab; canceling could drop the active shape key; zero-scale objects broke the brushes; releasing a rebound wheel key on a card chose nothing and left the wheel open.
- Stability fixes.

## 1.0.1

- Fill searches in background processes; Edit Mode Auto-Save is now off by default.

## 1.0.0

- First release.
