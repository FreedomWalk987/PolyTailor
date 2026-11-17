# Licenses / 授權文件

All license and notice files of PolyTailor are collected in this folder.
PolyTailor 的所有授權與聲明文件都集中在這個資料夾。

| File | Content |
|---|---|
| LICENSE | Full text of the GNU General Public License v3. PolyTailor's code is GPL-3.0-or-later. / GPL v3 全文 |
| THIRD_PARTY_NOTICES.md | Blender's LoopTools functions, the Gmsh pattern tables and the blossom matching design, with their notices; where the author's own code comes from (Relax Lab, PolyPen_Only). / 第三方來源與版權聲明；作者自有程式碼的來源 |
| mwmatching-MIT.txt | MIT License text for Joris van Rantwijk's mwmatching.py design (core/newfill/matching.py). / MIT 授權全文 |
| NetworkX-BSD-3-Clause.txt | BSD 3-Clause License text for the NetworkX port of that design (core/newfill/matching.py). / BSD 3-Clause 授權全文 |
| ASSETS_LICENSE.md | Icon artwork: CC0 1.0 (public domain), and how the icons are built. / 圖示授權（CC0 公眾領域）與建置方式 |
| SHOEQUAD-NOTICE.md / ShoeQuad-GPL-3.0.txt | Pattern-fill core provenance and GPL text. / 圖樣填面核心來源與授權 |

The embedded numerical solver's license and provenance are also in
`core/relax_lab/LICENSE` and `core/relax_lab/PROVENANCE.md`. PolyTailor is written in
Python and uses NumPy.

## Copyright holders / 版權所有者

- 2026 FreedomWalk: PolyTailor, Relax Lab 0.2.0 (core/relax_lab), ShoeQuad 0.9.0 (pattern-fill core), icons (CC0).
- 2011-2023 Blender Foundation: LoopTools Relax portions (core/looptools_relax.py; original authors Bart Crouch and
  Vladimir Spivak).
- 2008-2023 Joris van Rantwijk and NetworkX Developers: blossom matching design (core/newfill/matching.py).
- 1997-2025 C. Geuzaine and J.-F. Remacle (Gmsh; tables by contributor M. Reberol): quad cavity pattern tables
  (core/patterns/chords.py).
- The GPL license texts themselves are Copyright (C) 2007 Free Software Foundation, Inc.

PolyTailor is not an official Blender Foundation product.
