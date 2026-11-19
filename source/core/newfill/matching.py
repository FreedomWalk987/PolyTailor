# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
# Galil blossom matching; design derived from J. van Rantwijk mwmatching.py (MIT) and the NetworkX port (BSD-3); see licenses/THIRD_PARTY_NOTICES.md.
"""Maximum-weight maximum-cardinality matching in a general graph.

Edmonds' primal-dual blossom method in the O(V^3) form of Galil (1986, "Efficient algorithms
for finding maximum matching in graphs", ACM Comput. Surv. 18).
Written from the published algorithm; its data layout follows the widely used
Galil-based reference design of J. van Rantwijk's mwmatching.py (current repository: MIT licence;
NetworkX's port: BSD-3). The file is treated as derived from that design; the attribution is in
licenses/THIRD_PARTY_NOTICES.md.

max_weight_matching(nv, edges) -> mate list (mate[v] = u or -1)
edges: list of (i, j, w); i != j; at most one edge per pair.
Minimum-cost perfect matching: weight = C - cost.
"""


def max_weight_matching(nv, edges):
    if not edges:
        return [-1] * nv
    ne = len(edges)
    wmax = max(0, max(w for _, _, w in edges))
    endpoint = [0] * (2 * ne)
    for k, (i, j, w) in enumerate(edges):
        endpoint[2 * k] = i
        endpoint[2 * k + 1] = j
    neighb = [[] for _ in range(nv)]          # remote endpoints p (endpoint[p] is the neighbour)
    for k, (i, j, w) in enumerate(edges):
        neighb[i].append(2 * k + 1)
        neighb[j].append(2 * k)
    mate = [-1] * nv                           # remote endpoint of the matched edge
    label = [0] * (2 * nv)                     # 0 free, 1 S, 2 T (for top-level blossoms and vertices)
    labelend = [-1] * (2 * nv)
    inb = list(range(nv))                      # top-level blossom of each vertex
    bparent = [-1] * (2 * nv)
    bchilds = [None] * (2 * nv)
    bbase = list(range(nv)) + [-1] * nv
    bendps = [None] * (2 * nv)
    bestedge = [-1] * (2 * nv)
    bbestedges = [None] * (2 * nv)
    unused = list(range(nv, 2 * nv))
    dual = [wmax] * nv + [0] * nv
    allowed = [False] * ne
    stack = []

    def slack(k):
        i, j, w = edges[k]
        return dual[i] + dual[j] - 2 * w

    def leaves(b):
        if b < nv:
            yield b
        else:
            for t in bchilds[b]:
                if t < nv:
                    yield t
                else:
                    yield from leaves(t)

    def assign(w, t, p):
        b = inb[w]
        label[w] = label[b] = t
        labelend[w] = labelend[b] = p
        bestedge[w] = bestedge[b] = -1
        if t == 1:
            stack.extend(leaves(b))
        else:
            base = bbase[b]
            assign(endpoint[mate[base]], 1, mate[base] ^ 1)

    def scan(v, w):
        """Trace back from v and w; return the base of a new blossom or -1 for an augmenting path."""
        path = []
        base = -1
        while v != -1 or w != -1:
            b = inb[v]
            if label[b] & 4:
                base = bbase[b]
                break
            path.append(b)
            label[b] = 5
            if labelend[b] == -1:
                v = -1
            else:
                v = endpoint[labelend[b]]
                b = inb[v]
                v = endpoint[labelend[b]]
            if w != -1:
                v, w = w, v
        for b in path:
            label[b] = 1
        return base

    def add_blossom(base, k):
        v, w, _ = edges[k]
        bb = inb[base]
        bv = inb[v]
        bw = inb[w]
        b = unused.pop()
        bbase[b] = base
        bparent[b] = -1
        bparent[bb] = b
        path = []
        endps = []
        while bv != bb:
            bparent[bv] = b
            path.append(bv)
            endps.append(labelend[bv])
            v = endpoint[labelend[bv]]
            bv = inb[v]
        path.append(bb)
        path.reverse()
        endps.reverse()
        endps.append(2 * k)
        while bw != bb:
            bparent[bw] = b
            path.append(bw)
            endps.append(labelend[bw] ^ 1)
            w = endpoint[labelend[bw]]
            bw = inb[w]
        bchilds[b] = path
        bendps[b] = endps
        label[b] = 1
        labelend[b] = labelend[bb]
        dual[b] = 0
        for u in leaves(b):
            if label[inb[u]] == 2:
                stack.append(u)
            inb[u] = b
        # best edges to other S-blossoms
        best = [-1] * (2 * nv)
        for c in path:
            if bbestedges[c] is None:
                lists = [[p // 2 for p in neighb[u]] for u in leaves(c)]
            else:
                lists = [bbestedges[c]]
            for lst in lists:
                for kk in lst:
                    i, j, _ = edges[kk]
                    if inb[j] == b:
                        i, j = j, i
                    bj = inb[j]
                    if bj != b and label[bj] == 1 and (best[bj] == -1 or slack(kk) < slack(best[bj])):
                        best[bj] = kk
            bbestedges[c] = None
            bestedge[c] = -1
        bbestedges[b] = [kk for kk in best if kk != -1]
        bestedge[b] = -1
        for kk in bbestedges[b]:
            if bestedge[b] == -1 or slack(kk) < slack(bestedge[b]):
                bestedge[b] = kk

    def expand(b, endstage):
        for s in bchilds[b]:
            bparent[s] = -1
            if s < nv:
                inb[s] = s
            elif endstage and dual[s] == 0:
                expand(s, endstage)
            else:
                for u in leaves(s):
                    inb[u] = s
        if not endstage and label[b] == 2:
            entry = inb[endpoint[labelend[b] ^ 1]]
            j = bchilds[b].index(entry)
            if j & 1:
                j -= len(bchilds[b])
                step, et = 1, 0
            else:
                step, et = -1, 1
            p = labelend[b]
            while j != 0:
                label[endpoint[p ^ 1]] = 0
                label[endpoint[bendps[b][j - et] ^ et ^ 1]] = 0
                assign(endpoint[p ^ 1], 2, p)
                allowed[bendps[b][j - et] // 2] = True
                j += step
                p = bendps[b][j - et] ^ et
                allowed[p // 2] = True
                j += step
            bv = bchilds[b][j]
            label[endpoint[p ^ 1]] = label[bv] = 2
            labelend[endpoint[p ^ 1]] = labelend[bv] = p
            bestedge[bv] = -1
            j += step
            while bchilds[b][j] != entry:
                bv = bchilds[b][j]
                if label[bv] == 1:
                    j += step
                    continue
                u = -1
                for u in leaves(bv):
                    if label[u] != 0:
                        break
                if label[u] != 0:
                    label[u] = 0
                    label[endpoint[mate[bbase[bv]]]] = 0
                    assign(u, 2, labelend[u])
                j += step
        label[b] = labelend[b] = -1
        bchilds[b] = bendps[b] = None
        bbase[b] = -1
        bbestedges[b] = None
        bestedge[b] = -1
        unused.append(b)

    def augment_blossom(b, v):
        t = v
        while bparent[t] != b:
            t = bparent[t]
        if t >= nv:
            augment_blossom(t, v)
        i = j = bchilds[b].index(t)
        if i & 1:
            j -= len(bchilds[b])
            step, et = 1, 0
        else:
            step, et = -1, 1
        while j != 0:
            j += step
            t = bchilds[b][j]
            p = bendps[b][j - et] ^ et
            if t >= nv:
                augment_blossom(t, endpoint[p])
            j += step
            t = bchilds[b][j]
            if t >= nv:
                augment_blossom(t, endpoint[p ^ 1])
            mate[endpoint[p]] = p ^ 1
            mate[endpoint[p ^ 1]] = p
        bchilds[b] = bchilds[b][i:] + bchilds[b][:i]
        bendps[b] = bendps[b][i:] + bendps[b][:i]
        bbase[b] = bbase[bchilds[b][0]]

    def augment(k):
        v, w, _ = edges[k]
        for s, p in ((v, 2 * k + 1), (w, 2 * k)):
            while True:
                bs = inb[s]
                if bs >= nv:
                    augment_blossom(bs, s)
                mate[s] = p
                if labelend[bs] == -1:
                    break
                t = endpoint[labelend[bs]]
                bt = inb[t]
                s = endpoint[labelend[bt]]
                j = endpoint[labelend[bt] ^ 1]
                if bt >= nv:
                    augment_blossom(bt, j)
                mate[j] = labelend[bt]
                p = labelend[bt] ^ 1

    from . import emesh5
    check = emesh5.CHECK   # the fallback job's cancel token / deadline check (fallback5); raises FillRefused
    for _stage in range(nv):
        if check is not None:
            check()
        for i in range(2 * nv):
            label[i] = 0
            bestedge[i] = -1
        for i in range(nv, 2 * nv):
            bbestedges[i] = None
        for i in range(ne):
            allowed[i] = False
        del stack[:]
        for v in range(nv):
            if mate[v] == -1 and label[inb[v]] == 0:
                assign(v, 1, -1)
        augmented = False
        while True:
            while stack and not augmented:
                v = stack.pop()
                for p in neighb[v]:
                    k = p // 2
                    w = endpoint[p]
                    if inb[v] == inb[w]:
                        continue
                    if not allowed[k]:
                        ks = slack(k)
                        if ks <= 0:
                            allowed[k] = True
                    if allowed[k]:
                        if label[inb[w]] == 0:
                            assign(w, 2, p ^ 1)
                        elif label[inb[w]] == 1:
                            base = scan(v, w)
                            if base >= 0:
                                add_blossom(base, k)
                            else:
                                augment(k)
                                augmented = True
                                break
                        elif label[w] == 0:
                            label[w] = 2
                            labelend[w] = p ^ 1
                    elif label[inb[w]] == 1:
                        b = inb[v]
                        if bestedge[b] == -1 or ks < slack(bestedge[b]):
                            bestedge[b] = k
                    elif label[w] == 0:
                        if bestedge[w] == -1 or ks < slack(bestedge[w]):
                            bestedge[w] = k
            if augmented:
                break
            dtype = -1
            delta = dedge = dblos = None
            for v in range(nv):
                if label[inb[v]] == 0 and bestedge[v] != -1:
                    d = slack(bestedge[v])
                    if dtype == -1 or d < delta:
                        delta, dtype, dedge = d, 2, bestedge[v]
            for b in range(2 * nv):
                if bparent[b] == -1 and label[b] == 1 and bestedge[b] != -1:
                    d = slack(bestedge[b]) / 2
                    if dtype == -1 or d < delta:
                        delta, dtype, dedge = d, 3, bestedge[b]
            for b in range(nv, 2 * nv):
                if bbase[b] >= 0 and bparent[b] == -1 and label[b] == 2 and (dtype == -1 or dual[b] < delta):
                    delta, dtype, dblos = dual[b], 4, b
            if dtype == -1:
                dtype = 1
                delta = max(0, min(dual[:nv]))
            for v in range(nv):
                lb = label[inb[v]]
                if lb == 1:
                    dual[v] -= delta
                elif lb == 2:
                    dual[v] += delta
            for b in range(nv, 2 * nv):
                if bbase[b] >= 0 and bparent[b] == -1:
                    if label[b] == 1:
                        dual[b] += delta
                    elif label[b] == 2:
                        dual[b] -= delta
            if dtype == 1:
                break
            elif dtype == 2:
                allowed[dedge] = True
                i, j, _ = edges[dedge]
                if label[inb[i]] == 0:
                    i, j = j, i
                stack.append(i)
            elif dtype == 3:
                allowed[dedge] = True
                i, j, _ = edges[dedge]
                stack.append(i)
            else:
                expand(dblos, False)
        if not augmented:
            break
        for b in range(nv, 2 * nv):
            if bparent[b] == -1 and bbase[b] >= 0 and label[b] == 1 and dual[b] == 0:
                expand(b, True)
    return [endpoint[m] if m != -1 else -1 for m in mate]
