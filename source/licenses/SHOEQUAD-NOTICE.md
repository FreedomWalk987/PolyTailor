# Pattern fill core attribution

`core/patterns/chords.py`, `geom.py` and `solver.py` derive from ShoeQuad 0.9.0, the author's
own extension (Copyright (C) 2026 FreedomWalk). Each file carries an `SPDX-License-Identifier: GPL-3.0-or-later`
line and the copyright line `Copyright (C) 2026 FreedomWalk`; ShoeQuad's longer license headers are not repeated in
them. A copy of ShoeQuad's GPL license text is provided as `ShoeQuad-GPL-3.0.txt`.

The source attributes its coarse pattern layouts to Gmsh's
`contrib/QuadMeshingTools/qmtQuadCavityRemeshing.cpp` (M. Reberol), its pattern/chord
approach to Gmsh and Takayama et al. (2014), and its placement energy to Garanzha
et al. (2021). These attributions are kept in the files' docstrings; `core/patterns/chords.py` also names the
Gmsh copyright holders and license in its header.

Unused legacy solve/fill functions and submap dependencies have been removed.
PolyTailor adds cooperative integer search, bounds otherwise unconstrained closed
interior chords, cooperatively schedules placement optimization, adds validation, preview scheduling,
per-region controls and GPU overlays. It also adds planned-corner/cyclic-mirror
correspondence, retained topology with affine/harmonic placement transfer, and
deterministic cooperative optimization steps. Only the pattern core is included.
