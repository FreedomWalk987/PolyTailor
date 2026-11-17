# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Topological cleanup of a fixed-rim quad(-dominant) mesh.

Operators (from the published literature, own code):
  doublet collapse     Remacle et al. 2012 sec. 6; Kinney 1997 "CleanUp"
  quad-vertex-merge    Remacle et al. 2012 sec. 6 (= diagonal collapse, Canann et al. 1998)
  edge rotation        Bunin 2006 / Kinney 1997: re-split the hexagon of two quads
Cost: sum over vertices of (valence - target)^2; interior target 4, rim target from the rim angle
(faces k = round(angle/90), as in CUBIT vertex types end/side/corner/reversal), valence = k + 1.
Each move is kept only if, after a local smoothing of the touched interior vertices, every touched face
keeps scaled Jacobian >= max(SJ_FLOOR, 0.6 * old local min). Rim vertices never move or split.
"""
from collections import defaultdict

import numpy as np



class Mesh:
    def __init__(self, V, F, n):
        self.V = np.array(V, float)
        self.n = n
        self.F = {}
        self.vf = defaultdict(set)
        self.next = 0
        for f in F:
            self.add(list(f))

    def add(self, f):
        k = self.next
        self.next += 1
        self.F[k] = f
        for v in f:
            self.vf[v].add(k)
        return k

    def remove(self, k):
        for v in self.F[k]:
            self.vf[v].discard(k)
        del self.F[k]

    def edge_faces(self, u, v):
        return [k for k in self.vf[u] & self.vf[v] if is_edge(self.F[k], u, v)]


def is_edge(f, u, v):
    i = f.index(u)
    return f[(i + 1) % len(f)] == v or f[i - 1] == v


