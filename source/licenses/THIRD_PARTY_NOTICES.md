# Third-party notices

PolyTailor (formerly Poly Toolset and PolyPen_Only) is maintained by FreedomWalk and licensed under the GNU
GPL v3 or later; the full text is in licenses/LICENSE.

## Code

Relax uses the embedded Relax Lab 0.2.0 numerical core (core/smooth.py)
and Build uses core/meshpen_logic.py; both are written by the author.

PolyPen_Only (the author's earlier add-on) supplies the extrusion ladder algorithm
adapted in core/extrude.py. Its original artwork is not included; PolyTailor's own
icons are covered by licenses/ASSETS_LICENSE.md.

## Gmsh quad cavity patterns (core/patterns/chords.py)

The quad pattern vertex tables (`GMSH` in `core/patterns/chords.py`) follow Gmsh's
`contrib/QuadMeshingTools/qmtQuadCavityRemeshing.cpp` (M. Reberol). Gmsh is
Copyright (C) 1997-2025 C. Geuzaine and J.-F. Remacle and is licensed under the GNU
GPL version 2 or later. The attribution is also in the
header of `core/patterns/chords.py`.

## Maximum-weight matching (core/newfill/matching.py)

`core/newfill/matching.py` implements Edmonds' blossom method for maximum-weight matching in Galil's
O(V^3) form (Galil 1986). It was written from the published algorithm; its data layout (labels, blossom parents
and children, endpoint handling, the dual deltas) is derived from the reference design of Joris van Rantwijk's
`mwmatching.py` (MIT License) and its NetworkX port (BSD 3-Clause License). The file is therefore treated as
derived from that design, and this notice keeps the attribution those licenses require:

- Copyright (C) 2008-2023 Joris van Rantwijk and NetworkX Developers.

The full license texts are in `licenses/mwmatching-MIT.txt` (MIT, Joris van Rantwijk) and
`licenses/NetworkX-BSD-3-Clause.txt` (BSD 3-Clause, NetworkX Developers).

## Embedded LoopTools Relax

- Source: Blender's official `blender/blender-addons` repository,
  `blender-v4.1-release/mesh_looptools.py`.
- URL: https://github.com/blender/blender-addons/blob/blender-v4.1-release/mesh_looptools.py
- Retrieved: 2026-10-01.
- Source SHA-256: 58f886a6322bb60dbea458938161be8ceb0e3c1032a6b17420646ba513a834f0
- SPDX-FileCopyrightText: 2011-2023 Blender Foundation.
- Original addon by Bart Crouch; maintainer Vladimir Spivak (cwolf3d).
- License: GPL-2.0-or-later. This derivative is distributed under
  GPL-3.0-or-later, as permitted by the upstream "or later" grant; see licenses/LICENSE.
- Ten traversal/spline/Relax functions are included unchanged in
  `core/looptools_relax.py`. PolyTailor adds an edit-mesh adapter and applies pins,
  boundary locks and Mirror clipping when committing the result.
- It works on selected edge loops of the edited mesh; it does not use evaluated mirror
  geometry or extend to parallel unselected loops.

## Relax Lab 0.2.0

The numerical solver under core/relax_lab is included from the author's
Relax Lab 0.2.0 (Copyright (C) 2026 FreedomWalk), GPL-3.0-or-later; its license
text is in core/relax_lab/LICENSE. Only the Python/NumPy solver is used: the
RobustSolver objective and PatchSolver support. PolyTailor supplies its own stroke
adapter, projection and configurable boundary policy. Provenance and limitations
are summarized in core/relax_lab/PROVENANCE.md.

Since PolyPen_Only 3.7.4, the add-on also takes the local-collection/incident-normal
approach from the author's Relax Lab SparseStroke design. PolyTailor's KD broad phase, moved-row
tracking and existing mask/display integration are provided in core/brush_index.py
and ops/brush.py. The original standalone tool/preview registration is not bundled.
