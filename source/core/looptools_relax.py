# SPDX-FileCopyrightText: 2011-2023 Blender Foundation
# SPDX-License-Identifier: GPL-2.0-or-later
# Maintainer: Vladimir Spivak (cwolf3d)
# Originally an addon by Bart Crouch
# Original authors: Bart Crouch, Vladimir Spivak (cwolf3d)
# Modified 2026-10-01 by FreedomWalk: extracted the selected-loop Relax core
# for PolyTailor; no external operator, registration, cache or evaluated mirror mesh.
"""LoopTools Relax core, taken from Blender's official 4.1 release branch.

Upstream: https://github.com/blender/blender-addons/blob/blender-v4.1-release/mesh_looptools.py
The ten upstream numerical/traversal functions below are unchanged.
"""

import mathutils


def calculate_cubic_splines(bm_mod, tknots, knots):
    # hack for circular loops
    if knots[0] == knots[-1] and len(knots) > 1:
        circular = True
        k_new1 = []
        for k in range(-1, -5, -1):
            if k - 1 < -len(knots):
                k += len(knots)
            k_new1.append(knots[k - 1])
        k_new2 = []
        for k in range(4):
            if k + 1 > len(knots) - 1:
                k -= len(knots)
            k_new2.append(knots[k + 1])
        for k in k_new1:
            knots.insert(0, k)
        for k in k_new2:
            knots.append(k)
        t_new1 = []
        total1 = 0
        for t in range(-1, -5, -1):
            if t - 1 < -len(tknots):
                t += len(tknots)
            total1 += tknots[t] - tknots[t - 1]
            t_new1.append(tknots[0] - total1)
        t_new2 = []
        total2 = 0
        for t in range(4):
            if t + 1 > len(tknots) - 1:
                t -= len(tknots)
            total2 += tknots[t + 1] - tknots[t]
            t_new2.append(tknots[-1] + total2)
        for t in t_new1:
            tknots.insert(0, t)
        for t in t_new2:
            tknots.append(t)
    else:
        circular = False
    # end of hack

    n = len(knots)
    if n < 2:
        return False
    x = tknots[:]
    locs = [bm_mod.verts[k].co[:] for k in knots]
    result = []
    for j in range(3):
        a = []
        for i in locs:
            a.append(i[j])
        h = []
        for i in range(n - 1):
            if x[i + 1] - x[i] == 0:
                h.append(1e-8)
            else:
                h.append(x[i + 1] - x[i])
        q = [False]
        for i in range(1, n - 1):
            q.append(3 / h[i] * (a[i + 1] - a[i]) - 3 / h[i - 1] * (a[i] - a[i - 1]))
        l = [1.0]
        u = [0.0]
        z = [0.0]
        for i in range(1, n - 1):
            l.append(2 * (x[i + 1] - x[i - 1]) - h[i - 1] * u[i - 1])
            if l[i] == 0:
                l[i] = 1e-8
            u.append(h[i] / l[i])
            z.append((q[i] - h[i - 1] * z[i - 1]) / l[i])
        l.append(1.0)
        z.append(0.0)
        b = [False for i in range(n - 1)]
        c = [False for i in range(n)]
        d = [False for i in range(n - 1)]
        c[n - 1] = 0.0
        for i in range(n - 2, -1, -1):
            c[i] = z[i] - u[i] * c[i + 1]
            b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
            d[i] = (c[i + 1] - c[i]) / (3 * h[i])
        for i in range(n - 1):
            result.append([a[i], b[i], c[i], d[i], x[i]])
    splines = []
    for i in range(len(knots) - 1):
        splines.append([result[i], result[i + n - 1], result[i + (n - 1) * 2]])
    if circular:  # cleaning up after hack
        knots = knots[4:-4]
        tknots = tknots[4:-4]

    return(splines)


def calculate_linear_splines(bm_mod, tknots, knots):
    splines = []
    for i in range(len(knots) - 1):
        a = bm_mod.verts[knots[i]].co
        b = bm_mod.verts[knots[i + 1]].co
        d = b - a
        t = tknots[i]
        u = tknots[i + 1] - t
        splines.append([a, d, t, u])  # [locStart, locDif, tStart, tDif]

    return(splines)


def calculate_splines(interpolation, bm_mod, tknots, knots):
    if interpolation == 'cubic':
        splines = calculate_cubic_splines(bm_mod, tknots, knots[:])
    else:  # interpolations == 'linear'
        splines = calculate_linear_splines(bm_mod, tknots, knots[:])

    return(splines)


def check_loops(loops, mapping, bm_mod):
    valid_loops = []
    for loop, circular in loops:
        # loop needs to have at least 3 vertices
        if len(loop) < 3:
            continue
        # loop needs at least 1 vertex in the original, non-mirrored mesh
        if mapping:
            all_virtual = True
            for vert in loop:
                if mapping[vert] > -1:
                    all_virtual = False
                    break
            if all_virtual:
                continue
        # vertices can not all be at the same location
        stacked = True
        for i in range(len(loop) - 1):
            if (bm_mod.verts[loop[i]].co - bm_mod.verts[loop[i + 1]].co).length > 1e-6:
                stacked = False
                break
        if stacked:
            continue
        # passed all tests, loop is valid
        valid_loops.append([loop, circular])

    return(valid_loops)


def dict_vert_verts(edge_keys):
    # create connection data
    vert_verts = {}
    for ek in edge_keys:
        for i in range(2):
            if ek[i] in vert_verts:
                vert_verts[ek[i]].append(ek[1 - i])
            else:
                vert_verts[ek[i]] = [ek[1 - i]]

    return(vert_verts)


def edgekey(edge):
    return(tuple(sorted([edge.verts[0].index, edge.verts[1].index])))


def get_connected_selections(edge_keys):
    # create connection data
    vert_verts = dict_vert_verts(edge_keys)

    # find loops consisting of connected selected edges
    loops = []
    while len(vert_verts) > 0:
        loop = [iter(vert_verts.keys()).__next__()]
        growing = True
        flipped = False

        # extend loop
        while growing:
            # no more connection data for current vertex
            if loop[-1] not in vert_verts:
                if not flipped:
                    loop.reverse()
                    flipped = True
                else:
                    growing = False
            else:
                extended = False
                for i, next_vert in enumerate(vert_verts[loop[-1]]):
                    if next_vert not in loop:
                        vert_verts[loop[-1]].pop(i)
                        if len(vert_verts[loop[-1]]) == 0:
                            del vert_verts[loop[-1]]
                        # remove connection both ways
                        if next_vert in vert_verts:
                            if len(vert_verts[next_vert]) == 1:
                                del vert_verts[next_vert]
                            else:
                                vert_verts[next_vert].remove(loop[-1])
                        loop.append(next_vert)
                        extended = True
                        break
                if not extended:
                    # found one end of the loop, continue with next
                    if not flipped:
                        loop.reverse()
                        flipped = True
                    # found both ends of the loop, stop growing
                    else:
                        growing = False

        # check if loop is circular
        if loop[0] in vert_verts:
            if loop[-1] in vert_verts[loop[0]]:
                # is circular
                if len(vert_verts[loop[0]]) == 1:
                    del vert_verts[loop[0]]
                else:
                    vert_verts[loop[0]].remove(loop[-1])
                if len(vert_verts[loop[-1]]) == 1:
                    del vert_verts[loop[-1]]
                else:
                    vert_verts[loop[-1]].remove(loop[0])
                loop = [loop, True]
            else:
                # not circular
                loop = [loop, False]
        else:
            # not circular
            loop = [loop, False]

        loops.append(loop)

    return(loops)


def relax_calculate_knots(loops):
    all_knots = []
    all_points = []
    for loop, circular in loops:
        knots = [[], []]
        points = [[], []]
        if circular:
            if len(loop) % 2 == 1:  # odd
                extend = [False, True, 0, 1, 0, 1]
            else:  # even
                extend = [True, False, 0, 1, 1, 2]
        else:
            if len(loop) % 2 == 1:  # odd
                extend = [False, False, 0, 1, 1, 2]
            else:  # even
                extend = [False, False, 0, 1, 1, 2]
        for j in range(2):
            if extend[j]:
                loop = [loop[-1]] + loop + [loop[0]]
            for i in range(extend[2 + 2 * j], len(loop), 2):
                knots[j].append(loop[i])
            for i in range(extend[3 + 2 * j], len(loop), 2):
                if loop[i] == loop[-1] and not circular:
                    continue
                if len(points[j]) == 0:
                    points[j].append(loop[i])
                elif loop[i] != points[j][0]:
                    points[j].append(loop[i])
            if circular:
                if knots[j][0] != knots[j][-1]:
                    knots[j].append(knots[j][0])
        if len(points[1]) == 0:
            knots.pop(1)
            points.pop(1)
        for k in knots:
            all_knots.append(k)
        for p in points:
            all_points.append(p)

    return(all_knots, all_points)


def relax_calculate_t(bm_mod, knots, points, regular):
    all_tknots = []
    all_tpoints = []
    for i in range(len(knots)):
        amount = len(knots[i]) + len(points[i])
        mix = []
        for j in range(amount):
            if j % 2 == 0:
                mix.append([True, knots[i][round(j / 2)]])
            elif j == amount - 1:
                mix.append([True, knots[i][-1]])
            else:
                mix.append([False, points[i][int(j / 2)]])
        len_total = 0
        loc_prev = False
        tknots = []
        tpoints = []
        for m in mix:
            loc = mathutils.Vector(bm_mod.verts[m[1]].co[:])
            if not loc_prev:
                loc_prev = loc
            len_total += (loc - loc_prev).length
            if m[0]:
                tknots.append(len_total)
            else:
                tpoints.append(len_total)
            loc_prev = loc
        if regular:
            tpoints = []
            for p in range(len(points[i])):
                tpoints.append((tknots[p] + tknots[p + 1]) / 2)
        all_tknots.append(tknots)
        all_tpoints.append(tpoints)

    return(all_tknots, all_tpoints)


def relax_calculate_verts(bm_mod, interpolation, tknots, knots, tpoints,
points, splines):
    change = []
    move = []
    for i in range(len(knots)):
        for p in points[i]:
            m = tpoints[i][points[i].index(p)]
            if m in tknots[i]:
                n = tknots[i].index(m)
            else:
                t = tknots[i][:]
                t.append(m)
                t.sort()
                n = t.index(m) - 1
            if n > len(splines[i]) - 1:
                n = len(splines[i]) - 1
            elif n < 0:
                n = 0

            if interpolation == 'cubic':
                ax, bx, cx, dx, tx = splines[i][n][0]
                x = ax + bx * (m - tx) + cx * (m - tx) ** 2 + dx * (m - tx) ** 3
                ay, by, cy, dy, ty = splines[i][n][1]
                y = ay + by * (m - ty) + cy * (m - ty) ** 2 + dy * (m - ty) ** 3
                az, bz, cz, dz, tz = splines[i][n][2]
                z = az + bz * (m - tz) + cz * (m - tz) ** 2 + dz * (m - tz) ** 3
                change.append([p, mathutils.Vector([x, y, z])])
            else:  # interpolation == 'linear'
                a, d, t, u = splines[i][n]
                if u == 0:
                    u = 1e-8
                change.append([p, ((m - t) / u) * d + a])
    for c in change:
        move.append([c[0], (bm_mod.verts[c[0]].co + c[1]) / 2])

    return(move)

def selected_edge_keys(bm):
    return [edgekey(e) for e in bm.edges if e.select and not e.hide
            and all(not v.hide for v in e.verts)]


def selected_loop_moves(bm, interpolation="cubic", regular=True, edges=None):
    """Return a single upstream Relax iteration, without writing mesh data.

    Operate on selected edges in the real edit mesh. PolyTailor applies masks and
    Mirror clipping when committing the returned coordinates. Evaluated Mirror
    geometry and parallel unselected loops are intentionally not evaluated.
    edges: selected edge keys collected once by the caller (see selected_edge_keys);
    rescanning every edge each iteration is the main cost on very large meshes.
    """
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    if edges is None:
        edges = selected_edge_keys(bm)
    loops = check_loops(get_connected_selections(edges), False, bm)
    if not loops:
        raise ValueError("Select a connected loop of at least three non-coincident vertices")
    knots, points = relax_calculate_knots(loops)
    tknots, tpoints = relax_calculate_t(bm, knots, points, regular)
    splines = [calculate_splines(interpolation, bm, tknots[i], knots[i])
               for i in range(len(knots))]
    return relax_calculate_verts(bm, interpolation, tknots, knots,
                                 tpoints, points, splines)
