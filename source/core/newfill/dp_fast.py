# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""S2: windowed, variant-batched, bound-pruned labelling DP (pure Python + NumPy).
Exactly the plain cyclic labelling DP's result (once apattern.label_dp): same recurrence, same float32 operations, same tie-breaking (first option wins).

  1. Windows. Per step s and turn count k the arithmetic runs only inside
         window(s, k) = forward-reachable box  ∩  backward (can still close) box,
     both propagated as integer boxes before the DP. Shifts are offset changes, not copies. Cells outside the window
     are never on a closing path, so labels and cost are unchanged.
  2. Batching (label_dp_batch). Variants that share the rim, the start and kband are solved in ONE pass with a leading
     variant axis: per-variant costs are broadcast, options a variant lacks cost +inf. label_dp clips every variant
     to its own grid (half-width Rr); the batch reproduces that by masking each variant outside its grid.
  3. Pruning (label_dp(ub=...)). Given the cost ub of ANY labelling the variant allows (path_cost), cells whose cost
     exceeds ub - (lower bound of the remaining steps) are set to +inf and the tables cropped. Such cells are neither on
     an optimal path nor the best predecessor of a cell that is, so the result is unchanged.

label_dp(th, prm, start, A, ub=None)  -> (labels, cost) | (None, inf)
label_dp_batch(th, prms, start, A)    -> [(labels, cost) | (None, inf)] in the order of prms (no 'wpos' variants)
solve(th, prms, start, A)             -> the same list for any mix of variants (groups, batches, prunes)
A = the apattern module (vertex_costs, DIRS, check_budget)."""
import math
import numpy as np


def _drive(gen):
    """run a generator to its end; its value."""
    try:
        while True:
            next(gen)
    except StopIteration as done:
        return done.value


def _call_steps(fn, *args, **kwargs):
    """call fn; when it returns a generator (a *_steps function), step it here."""
    import inspect
    out = fn(*args, **kwargs)
    if inspect.isgenerator(out):
        out = yield from out
    return out

_LAB = ('F', 'H', 'R', 'Q')
_DK = (0, -1, 1, 2)   # pred k offset of F, H, R, Q


def _clip(b, lo, hi):
    x0, x1, y0, y1 = max(b[0], lo), min(b[1], hi), max(b[2], lo), min(b[3], hi)
    return (x0, x1, y0, y1) if x0 < x1 and y0 < y1 else None


def _sh(b, dx, dy, lo, hi):
    return None if b is None else _clip((b[0] + dx, b[1] + dx, b[2] + dy, b[3] + dy), lo, hi)


def _un(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))


def _it(a, b):
    if a is None or b is None:
        return None
    x0, x1, y0, y1 = max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3])
    return (x0, x1, y0, y1) if x0 < x1 and y0 < y1 else None


def _opts(k, st, D):
    """Options of one step for target k, in label_dp's order: (label, d, pred k, jump x, jump y, cost)."""
    base, soft = st
    ol = [(_LAB[i], 0, k + _DK[i], 0, 0, base[i]) for i in range(4) if base[i] < np.inf]
    if soft:
        a, b = D[(k - 1) % 4], D[k % 4]
        jx, jy = a[0] + b[0], a[1] + b[1]
        ol += [('S', d, k - 1, jx * d, jy * d, cs) for d, cs in soft if cs < np.inf]
    return ol


def _summ(steps):
    """Per step: which of F/H/R/Q exist, and the soft d range (lo, hi) or None."""
    out = []
    for base, soft in steps:
        ds = [d for d, cs in soft if cs < np.inf]
        out.append(((base[0] < np.inf, base[1] < np.inf, base[2] < np.inf, base[3] < np.inf), (min(ds), max(ds)) if ds else None))
    return out


def _boxes(n, acts, summ, D, Rr):
    """Windows W[s][k] (centred boxes, post-move) = forward-reachable box ∩ can-still-close box; None if no closing
    path can exist. The soft options of a step shift one table along one diagonal, so their hull needs only the two
    extreme d."""
    lo, hi = -Rr, Rr + 1
    org = (0, 1, 0, 1)
    Fw = [None] * n
    prev = {0: org}
    for s in range(n):
        cur = {}
        av, sr = summ[s]
        for k in acts[s]:
            pre = None
            for i in range(4):
                if av[i]:
                    p = prev.get(k + _DK[i])
                    if p is not None:
                        pre = p if pre is None else (min(pre[0], p[0]), max(pre[1], p[1]), min(pre[2], p[2]), max(pre[3], p[3]))
            if sr is not None:
                p = prev.get(k - 1)
                if p is not None:
                    a, b = D[(k - 1) % 4], D[k % 4]
                    jx, jy = a[0] + b[0], a[1] + b[1]
                    for d in sr:
                        pre = _un(pre, _sh(p, jx * d, jy * d, lo, hi))
            post = _sh(pre, D[k % 4][0], D[k % 4][1], lo, hi)
            if post is not None:
                cur[k] = post
        Fw[s] = cur; prev = cur
        if not cur:
            return None
    W = [None] * n
    nxt = {4: org}
    for s in range(n - 1, -1, -1):
        Ws = {}
        Fs = Fw[s]
        for k, b in nxt.items():
            w = _it(Fs.get(k), b)
            if w is not None:
                Ws[k] = w
        if not Ws:
            return None
        W[s] = Ws
        back = {}
        av, sr = summ[s]
        for k, w in Ws.items():
            pre = _sh(w, -D[k % 4][0], -D[k % 4][1], lo, hi)
            if pre is None:
                continue
            for i in range(4):
                if av[i]:
                    kp = k + _DK[i]
                    back[kp] = _un(back.get(kp), pre)
            if sr is not None:
                a, b = D[(k - 1) % 4], D[k % 4]
                jx, jy = a[0] + b[0], a[1] + b[1]
                for d in sr:
                    c = _sh(pre, -jx * d, -jy * d, lo, hi)
                    if c is not None:
                        back[k - 1] = _un(back.get(k - 1), c)
        nxt = back
    return W if (0 in nxt and _it(nxt[0], org) is not None) else None


def _prep(th, prm, start, A, boxes=True):
    """Everything label_dp derives before its loop, plus the windows. Coordinates are centred (origin = 0, 0)."""
    n = len(th)
    order = [(start + i) % n for i in range(n)]
    tau = 180.0 - th[order]
    T = np.cumsum(tau) / 90.0
    vc = A.vertex_costs(th[order], prm)
    fl = prm.get('_forbid_lab')
    if fl:
        pos_of = {v: i for i, v in enumerate(order)}
        vc = {k: np.array(c, float) for k, c in vc.items()}
        for (v, lb) in fl:
            if lb in vc:
                vc[lb][pos_of[v]] = np.inf
    dmax = min(prm['dmax'], max(3, n // 4))
    band = prm.get('kband', 1.6)
    if prm['soft']:
        rho = np.asarray(prm['_rho'])
        dcap = int(min(dmax, np.max(np.round(np.maximum(1.0, prm['dfrac'] * rho))) + prm.get('dwin', 2)))
    else:
        dcap = 0
    Rr = n // 2 + 2 * dcap + 2
    D = [(int(d[0]), int(d[1])) for d in A.DIRS]
    acts, steps = [], []
    cFa, cHa, cRa, cQa = (np.asarray(vc[l], float).tolist() for l in ('F', 'H', 'R', 'Q'))
    soft_on = prm['soft']
    fb = prm.get('_forbid', ())
    dw = prm.get('dwin', 2)
    for s in range(n):
        v = order[s]
        Ts = T[s]
        act = [k for k in range(int(math.floor(Ts - band)), int(math.ceil(Ts + band)) + 1) if abs(k - Ts) <= band]
        if s == n - 1:
            act = [4]
        soft_d = []
        if soft_on and prm['s_min'] < th[v] < prm['s_max']:
            cF = vc['F'][s]
            dt = max(1.0, prm['dfrac'] * prm['_rho'][v])
            d0 = int(round(dt))
            soft_d = [(d, cF + prm['wpole'] + prm['wd'] * (d - dt) ** 2) for d in range(max(1, d0 - dw), min(dmax, d0 + dw) + 1) if (v, d) not in fb and (v, 0) not in fb]
        acts.append(act); steps.append(((cFa[s], cHa[s], cRa[s], cQa[s]), soft_d))
    P = dict(n=n, order=order, T=T, Rr=Rr, D=D, acts=acts, steps=steps, W=None, wT=prm['wT'], POS=None)
    if prm.get('wpos') and prm.get('_xy') is not None:
        Pxy = np.asarray(prm['_xy'], float)
        st = order[0]
        e = Pxy[st] - Pxy[st - 1]
        u = e / max(np.linalg.norm(e), 1e-300); vv = np.array([-u[1], u[0]])
        hh = np.mean(np.linalg.norm(np.roll(Pxy, -1, 0) - Pxy, axis=1))
        Q = Pxy[[(st + i + 1) % n for i in range(n)]] - Pxy[st]
        P['POS'] = np.stack([Q @ u, Q @ vv], 1) / hh
        P['wpos'] = prm['wpos']
    if boxes:
        P['W'] = _boxes(n, acts, _summ(steps), D, Rr)
    return P


def path_cost(labels, P):
    """Cost of a given labelling under the model of P (float64), inf if P does not allow it: an upper bound of the
    optimum of P's DP."""
    n, order, T, D, Rr, wT = P['n'], P['order'], P['T'], P['D'], P['Rr'], P['wT']
    POS = P['POS']
    k = x = y = 0
    tot = 0.0
    for s in range(n):
        lab, d = labels[order[s]]
        base, soft = P['steps'][s]
        if lab == 'S':
            c = next((cs for dd, cs in soft if dd == d), np.inf)
            a, b = D[k % 4], D[(k + 1) % 4]
            x += (a[0] + b[0]) * d; y += (a[1] + b[1]) * d
            k += 1
        else:
            i = _LAB.index(lab)
            c = base[i]
            k -= _DK[i]
        if not c < np.inf or k not in P['acts'][s] or abs(x) > Rr or abs(y) > Rr:
            return np.inf
        x += D[k % 4][0]; y += D[k % 4][1]
        if abs(x) > Rr or abs(y) > Rr:
            return np.inf
        tot += c + wT * (k - T[s]) ** 2
        if POS is not None:
            tot += P['wpos'] * ((x - POS[s][0]) ** 2 + (y - POS[s][1]) ** 2)
    return tot if (k, x, y) == (4, 0, 0) else np.inf


def _backtrack(P, choices, j=None):
    n, order, D = P['n'], P['order'], P['D']
    labels = [None] * n
    k, x, y = 4, 0, 0
    for s in range(n - 1, -1, -1):
        dx, dy = D[k % 4]
        x, y = x - dx, y - dy
        ch, names, px0, py0 = choices[s][k]
        lab, d = names[ch[x - px0, y - py0] if j is None else ch[j, x - px0, y - py0]]
        labels[order[s]] = (lab, d)
        if lab == 'H':
            k -= 1
        elif lab == 'R':
            k += 1
        elif lab == 'Q':
            k += 2
        elif lab == 'S':
            k -= 1
            dk = k % 4
            x -= D[dk][0] * d + D[(dk + 1) % 4][0] * d
            y -= D[dk][1] * d + D[(dk + 1) % 4][1] * d
    if (k, x, y) != (0, 0, 0):
        raise RuntimeError('backtrack mismatch')
    return labels


def label_dp_steps(th, prm, start, A, ub=None, P=None):
    if P is None:
        P = _prep(th, prm, start, A)
    W = P['W']
    if W is None:
        return None, np.inf
    n, T, D, steps, Rr, POS = P['n'], P['T'], P['D'], P['steps'], P['Rr'], P['POS']
    if POS is not None:
        g = np.arange(2 * Rr + 1, dtype=np.float32) - Rr
    prune = ub is not None and np.isfinite(ub)
    if prune:
        mins = [min([c for c in st[0] if c < np.inf] + [cs for _, cs in st[1] if cs < np.inf] + [np.inf]) for st in steps]
        lb = np.concatenate([np.cumsum(np.array(mins[::-1]))[::-1][1:], [0.0]])
        ubm = ub + 1e-3 * (1.0 + abs(ub)) + 1e-2
    INF = np.float32(np.inf)
    C = {0: (np.zeros((1, 1), np.float32), 0, 0)}
    choices = []
    wT = prm['wT']
    for s in range(n):
        yield from A.tick()
        newC, chs = {}, {}
        Ts = T[s]
        st = steps[s]
        for k, w in W[s].items():
            yield   # one rotation state per step
            mx, my = D[k % 4]
            px0, px1, py0, py1 = w[0] - mx, w[1] - mx, w[2] - my, w[3] - my
            ol = _opts(k, st, D)
            if prune:      # shrink the target to what the (cropped) sources can reach
                hx0 = hy0 = 1 << 30; hx1 = hy1 = -(1 << 30)
                for (lab, d, kp, jx, jy, cost) in ol:
                    src = C.get(kp)
                    if src is not None:
                        arr, ox, oy = src
                        hx0 = min(hx0, ox + jx); hx1 = max(hx1, ox + jx + arr.shape[0])
                        hy0 = min(hy0, oy + jy); hy1 = max(hy1, oy + jy + arr.shape[1])
                px0, px1, py0, py1 = max(px0, hx0), min(px1, hx1), max(py0, hy0), min(py1, hy1)
                if px0 >= px1 or py0 >= py1:
                    continue
            best = None; ch = None; names = []
            for (lab, d, kp, jx, jy, cost) in ol:
                src = C.get(kp)
                if src is None:
                    continue
                arr, ox, oy = src
                sx0, sy0 = ox + jx, oy + jy
                x0, x1 = max(sx0, px0), min(sx0 + arr.shape[0], px1)
                y0, y1 = max(sy0, py0), min(sy0 + arr.shape[1], py1)
                if x0 >= x1 or y0 >= y1:
                    continue
                sub = arr[x0 - sx0:x1 - sx0, y0 - sy0:y1 - sy0]
                if best is None:
                    first = sub + np.float32(cost)
                    best = np.full((px1 - px0, py1 - py0), INF, first.dtype)
                    best[x0 - px0:x1 - px0, y0 - py0:y1 - py0] = first
                    ch = np.zeros(best.shape, np.int8)
                else:
                    tmp = np.add(sub, np.float32(cost), dtype=np.float32, casting='same_kind')
                    bv = best[x0 - px0:x1 - px0, y0 - py0:y1 - py0]
                    mb = tmp < bv
                    np.copyto(bv, tmp, where=mb)
                    np.copyto(ch[x0 - px0:x1 - px0, y0 - py0:y1 - py0], np.int8(len(names)), where=mb)
                names.append((lab, d))
            if best is None:
                continue
            best = best + np.float32(wT * (k - Ts) ** 2)
            qx0, qy0 = px0 + mx, py0 + my
            if POS is not None:
                ex, ey = POS[s]
                GX = g[qx0 + Rr:qx0 + Rr + best.shape[0], None] * np.ones((1, best.shape[1]), np.float32)
                GY = np.ones((best.shape[0], 1), np.float32) * g[None, qy0 + Rr:qy0 + Rr + best.shape[1]]
                best = best + np.float32(prm['wpos']) * ((GX - ex) ** 2 + (GY - ey) ** 2)
            chs[k] = (ch, names, px0, py0)
            if prune:
                keep = best <= ubm - lb[s]
                rows = keep.any(1)
                if not rows.any():
                    continue
                cols = keep.any(0)
                r0 = int(rows.argmax()); r1 = len(rows) - int(rows[::-1].argmax())
                c0 = int(cols.argmax()); c1 = len(cols) - int(cols[::-1].argmax())
                best = np.where(keep, best, INF)[r0:r1, c0:c1]
                qx0 += r0; qy0 += c0
            newC[k] = (best, qx0, qy0)
        C = newC
        choices.append(chs)
        if not C:
            return None, np.inf
    fin = C.get(4)
    if fin is None:
        return None, np.inf
    arr, ox, oy = fin
    if not (0 <= -ox < arr.shape[0] and 0 <= -oy < arr.shape[1]) or not np.isfinite(arr[-ox, -oy]):
        return None, np.inf
    return _backtrack(P, choices), float(arr[-ox, -oy])


def label_dp(*args, **kwargs):
    """label_dp_steps run to its end (the synchronous form)."""
    return _drive(label_dp_steps(*args, **kwargs))


def label_dp_batch_steps(th, prms, start, A, preps=None):
    """One DP pass for several variants (same rim, start and kband; none with 'wpos')."""
    if len(prms) == 1:
        return [(yield from label_dp_steps(th, prms[0], start, A))]
    PP = preps
    if not PP:   # one variant's preparation per step
        PP = []
        for p in prms:
            PP.append(_prep(th, p, start, A, boxes=False))
            yield
    V = len(PP)
    out = [(None, np.inf)] * V
    P0 = PP[0]
    n, T, D, acts = P0['n'], P0['T'], P0['D'], P0['acts']
    Rv = [P['Rr'] for P in PP]
    Rmin = min(Rv)
    summ = []
    for s in range(n):
        av = [False] * 4; lo = hi = None
        for P in PP:
            base, soft = P['steps'][s]
            for i in range(4):
                av[i] = av[i] or base[i] < np.inf
            for d, cs in soft:
                if cs < np.inf:
                    lo = d if lo is None else min(lo, d); hi = d if hi is None else max(hi, d)
        summ.append((tuple(av), None if lo is None else (lo, hi)))
    yield
    WU = _boxes(n, acts, summ, D, max(Rv))
    yield
    if WU is None:
        return out
    wT = np.array([P['wT'] for P in PP], float)
    INF = np.float32(np.inf)
    C = {0: (np.zeros((V, 1, 1), np.float32), 0, 0)}
    choices = []
    for s in range(n):
        yield from A.tick()
        base = np.array([P['steps'][s][0] for P in PP], float).T.astype(np.float32)     # (4, V)
        sd = {}
        for j, P in enumerate(PP):
            for d, cs in P['steps'][s][1]:
                if cs < np.inf:
                    row = sd.get(d)
                    if row is None:
                        row = sd[d] = np.full(V, INF, np.float32)
                    row[j] = np.float32(cs)
        ds = sorted(sd)
        av = summ[s][0]
        Ts = T[s]
        newC, chs = {}, {}
        for k, w in WU[s].items():
            yield   # one rotation state per step (a slice point)
            mx, my = D[k % 4]
            px0, px1, py0, py1 = w[0] - mx, w[1] - mx, w[2] - my, w[3] - my
            best = np.full((V, px1 - px0, py1 - py0), INF, np.float32)
            ch = np.zeros(best.shape, np.int8)
            names = []
            a, b = D[(k - 1) % 4], D[k % 4]
            jx1, jy1 = a[0] + b[0], a[1] + b[1]
            ol = [(_LAB[i], 0, k + _DK[i], 0, 0, base[i]) for i in range(4) if av[i]]
            ol += [('S', d, k - 1, jx1 * d, jy1 * d, sd[d]) for d in ds]
            for (lab, d, kp, jx, jy, cost) in ol:
                src = C.get(kp)
                if src is None:
                    continue
                arr, ox, oy = src
                sx0, sy0 = ox + jx, oy + jy
                x0, x1 = max(sx0, px0), min(sx0 + arr.shape[1], px1)
                y0, y1 = max(sy0, py0), min(sy0 + arr.shape[2], py1)
                if x0 >= x1 or y0 >= y1:
                    continue
                tmp = arr[:, x0 - sx0:x1 - sx0, y0 - sy0:y1 - sy0] + cost[:, None, None]
                bv = best[:, x0 - px0:x1 - px0, y0 - py0:y1 - py0]
                mb = tmp < bv
                np.copyto(bv, tmp, where=mb)
                np.copyto(ch[:, x0 - px0:x1 - px0, y0 - py0:y1 - py0], np.int8(len(names)), where=mb)
                names.append((lab, d))
            if not names:
                continue
            best += (wT * (k - Ts) ** 2).astype(np.float32)[:, None, None]
            # each variant's own grid: pre cell q and post cell q + move must both lie in [-Rr, Rr]
            if px0 < -Rmin or px1 > Rmin + 1 or py0 < -Rmin or py1 > Rmin + 1 or w[0] < -Rmin or w[1] > Rmin + 1 or w[2] < -Rmin or w[3] > Rmin + 1:
                for j, R in enumerate(Rv):
                    ax0 = max(-R, -R - mx) - px0; ax1 = min(R, R - mx) + 1 - px0
                    ay0 = max(-R, -R - my) - py0; ay1 = min(R, R - my) + 1 - py0
                    bj = best[j]
                    if ax0 > 0: bj[:ax0, :] = INF
                    if ax1 < bj.shape[0]: bj[max(ax1, 0):, :] = INF
                    if ay0 > 0: bj[:, :ay0] = INF
                    if ay1 < bj.shape[1]: bj[:, max(ay1, 0):] = INF
            newC[k] = (best, w[0], w[2])
            chs[k] = (ch, names, px0, py0)
        C = newC
        choices.append(chs)
        if not C:
            return out
    fin = C.get(4)
    if fin is None:
        return out
    arr, ox, oy = fin
    if not (0 <= -ox < arr.shape[1] and 0 <= -oy < arr.shape[2]):
        return out
    for j in range(V):
        c = arr[j, -ox, -oy]
        if np.isfinite(c):
            out[j] = (_backtrack(PP[j], choices, j), float(c))
    return out


def label_dp_batch(*args, **kwargs):
    """label_dp_batch_steps run to its end (the synchronous form)."""
    return _drive(label_dp_batch_steps(*args, **kwargs))


def _wide(prm):
    """Variants whose reachable set is large (R allowed on nearly flat vertices, or every soft depth)."""
    return prm.get('r_min', 190.0) < 185.0 or prm.get('dwin', 2) > 10 or bool(prm.get('wpos'))


PRUNE_MIN_N = 60   # below this the pruning bookkeeping costs more than it saves (measured on the DEV traces)


def solve_steps(th, prms, start, A, known=()):
    """All variants of one rim: narrow variants in batches (per kband), then the wide ones one by one, each pruned
    with the best upper bound among the labellings found so far (and `known`). A.BudgetExceeded (the fill's deadline
    or cancel) propagates: nothing more is built from a partial table past the deadline."""
    out = [None] * len(prms)
    pool = [l for l in known if l is not None]
    prune = len(th) >= PRUNE_MIN_N
    groups = {}
    for i, p in enumerate(prms):
        if not _wide(p):
            groups.setdefault(p.get('kband', 1.6), []).append(i)
    for ix in groups.values():
        for i, r in zip(ix, (yield from label_dp_batch_steps(th, [prms[i] for i in ix], start, A))):
            out[i] = r
            if r[0] is not None:
                pool.append(r[0])
    for i, p in enumerate(prms):
        if out[i] is None:
            P = _prep(th, p, start, A)
            ub = min([path_cost(l, P) for l in pool] + [np.inf]) if (prune and P['W'] is not None) else None
            out[i] = (yield from label_dp_steps(th, p, start, A, ub=ub, P=P))
            if out[i][0] is not None:
                pool.append(out[i][0])
    return out


def solve(*args, **kwargs):
    """solve_steps run to its end (the synchronous form)."""
    return _drive(solve_steps(*args, **kwargs))
