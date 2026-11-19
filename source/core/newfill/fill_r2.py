# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Plan-and-realise quad fill used by the A part (fill_a).

fill(world, budget_s=5.0) -> dict(positions, faces, info)

Pipeline:
  1. Plans = closing labellings from the tau-ladder frontier DP (min poles subject to a rim-SJ floor tau),
     plus a few quality-weighted and triangle-centre variants. Odd rims: one triangle (ear / edge apex) + even solve.
  2. Every plan has an exact a-priori upper bound of the selection key (rim SJ bound, #poles, #tolerant mismatches).
     Plans are realised (Tutte -> Winslow -> hinge SJ) in descending bound order; stop when no remaining bound can win.
  3. Key = sj - LAM * poles - MU * mismatches - terrible/strip penalties (no SJ cap).
  4. One dopt FULL pass (adj_min_cap 1.5, edge_min_cap 2.0) on the winner, under the fill's deadline.
Budget: one deadline checked inside the DP per-vertex loop, CG, smoother, optimiser, polish and dopt; when it expires
after at least one realised candidate, the best one so far is returned (dopt skipped or shortened); with no candidate
it raises apattern.BudgetExceeded (a FillRefused)."""
import time
import numpy as np

from . import apattern as A
from . import dopt


def _drive(gen):
    """run a generator to its end; its value."""
    try:
        while True:
            next(gen)
    except StopIteration as done:
        return done.value


FillRefused = A.FillRefused
BudgetExceeded = A.BudgetExceeded

TB = A.TAU_BASE
VARIANTS_R2 = ([dict()]                                              # the default variant (safety net)
               + [dict(TB, tau=t) for t in (0.85, 0.75, 0.65, 0.55, 0.45, 0.0)]
               + [dict(TB, tau=t, r_min=168.0) for t in (0.6, 0.45)]  # rim dipole (S + flat R) option
               + [dict(TB, tau=t, wq=1.0) for t in (0.7, 0.5)]
               + [dict(TB, tau=0.7, dfrac=1.0, dwin=99, wd=0.02)]
               + [dict(r_min=168.0, wm=0.6, wpole=1.0, wT=0.2),                       # S + flat-R variants
                  dict(r_min=168.0, wm=1.0, wpole=0.8, wT=0.2, s_min=120.0, sj_lo=0.8, wlow=3.0)])
ODD_R2 = [dict(TB, tau=t) for t in (0.85, 0.55, 0.0)]
ODD_PROBE = [dict(TB, tau=0.7)]

# ---------------------------------------------------------------- bands (isoperimetric quotient Q < 0.6)
BAND = dict(q=0.6, lam=0.2, noguard=True, max_real=8, post_top=2)
BAND_EXTRA = ([dict(TB, tau=0.75, wpos=0.02)]                                               # position-follow (density)
              + [dict(TB, tau=0.6, r_min=158.0, wpole=0.6)]                                 # R on 158-200 deg (SJ >= 0.79)
              + [dict(TB, tau=t, r_min=158.0, wpole=0.6, wT=0.1) for t in (0.8, 0.7)])      # weak shape-follow on bands


def is_band(xy):
    xy = np.asarray(xy, float)
    area = 0.5 * abs(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]))
    per = np.linalg.norm(np.roll(xy, -1, 0) - xy, axis=1).sum()
    return bool(4.0 * np.pi * area / max(per * per, 1e-300) < BAND['q'])
RESCUE = [dict(TB, tau=None, r_min=168.0, wpole=0.3), dict(TB, tau=None, r_min=168.0, wpole=-0.5), dict(TB, tau=0.4, r_min=168.0, wpole=0.3)]
# selection: piecewise-concave SJ utility (steep below t1, unit slope to t2, s3 above) minus pole / mismatch prices.
# Fitted on logged candidate lists under the constraint 'no more interior irregular vertices than before'.
# Quality first, poles allowed where they buy quality: a concave SJ key; the prices and caps are in CFG.
CFG = dict(s1=2.0, t1=0.75, t2=1.0, s3=0.0, lam=0.35, mu=0.06, klow=2.0, tlow=0.45, kterr=1.0, kstrip=0.3,
           max_real=4, rescue_sj=0.4)


def util(s, cfg):
    return cfg['s1'] * min(s, cfg['t1']) + (min(max(s, cfg['t1']), cfg['t2']) - cfg['t1']) + cfg['s3'] * max(0.0, s - cfg['t2'])


def key_of(m, cfg):
    return (util(m['minSJ'], cfg) - cfg.get('klow', 0.0) * max(0.0, cfg.get('tlow', 0.45) - m['minSJ'])
            - cfg['lam'] * m['irr'] - cfg['mu'] * m['btol']
            - (cfg['kterr'] if m['minSJ'] < 0.3 else 0.0) - (cfg['kstrip'] if m['strip'] else 0.0))


def _tries(xy):
    n = len(xy)
    th = A.interior_angles(xy)
    sharp = [int(v) for v in np.argsort(th)[:2]]
    tries = [('ear', v) for v in sharp if th[v] < 125.0]
    flat = np.abs(th - 180.0) + np.abs(np.roll(th, -1) - 180.0)
    tries += [('edge', int(i)) for i in np.argsort(flat)[:2]]
    for v in sharp[:2]:
        for e in ((v - 1) % n, v):
            if ('edge', e) not in tries:
                tries.append(('edge', e))
    return tries


def _batch0_steps(xy, prm):
    """Cheap safety batch: the default variant (even) or one probe DP per triangle placement (odd)."""
    if len(xy) % 2 == 0:   # no relaxed fallback and few retries here: this batch must stay cheap
        return (yield from A.plans_even_steps(xy, dict(prm, _relaxed=True, nogood=2), prm['variants'][:1]))
    out = []
    for kind, v in _tries(xy):
        try:
            out += (yield from A._odd_plans_steps(xy, dict(prm, odd_variants=ODD_PROBE, _relaxed=True, nogood=2), kind, v))
        except BudgetExceeded:
            if out:
                break
            raise
    return out


def _batch1_steps(xy, prm, b0):
    """The frontier: remaining variants (even) or the full odd frontier on the two best placements; then fallbacks."""
    n = len(xy)
    out = []
    try:
        if n % 2 == 0:
            out = (yield from A.plans_even_steps(xy, prm, prm['variants'][1:]))
            if not out and not b0:
                out = (yield from A.plans_even_steps(xy, prm, A.VARIANTS))
        else:
            best = {}
            for p in b0:
                k = p['tag']
                best[k] = max(best.get(k, -9.0), key_of(dict(minSJ=p['b'], irr=0, btol=0, strip=False), CFG))
            for kind, v in sorted(best, key=lambda t: -best[t])[:2]:
                out += (yield from A._odd_plans_steps(xy, prm, kind, v))
            if not out and not b0:
                p2 = dict(prm, odd_variants=A.ODD_VARIANTS)
                for kind, v in _tries(xy):
                    out += (yield from A._odd_plans_steps(xy, p2, kind, v))
    except BudgetExceeded:
        pass
    return out


def _rank(plans, cfg):
    th_cache = {}
    for pl in plans:
        k = id(pl['xy'])
        if k not in th_cache:
            th_cache[k] = A.interior_angles(pl['xy'])
        b, irr, btol = A.label_bound(pl['labels'], th_cache[k])
        pl['ub2'] = key_of(dict(minSJ=b, irr=irr, btol=btol, strip=False), cfg)
    plans.sort(key=lambda p: -p['ub2'])
    return plans


def fill_xy_steps(xy, cfg=None, prm=None):
    """Anytime: realise one safety candidate from the cheap batch first, then the frontier within the budget."""
    cfg = dict(CFG, **(cfg or {}))
    prm = dict(A.DEFAULT, **(prm or {}))
    prm.setdefault('variants', VARIANTS_R2)
    prm.setdefault('odd_variants', ODD_R2)
    prm.setdefault('nogood', 6)
    cands, err, done = [], None, set()

    def realise(pl):
        nonlocal err
        try:
            _, pos, F, info = (yield from A.run_plan_steps(pl, prm))
        except BudgetExceeded:
            raise
        except FillRefused as ex:
            err = err or ex
            return False
        cands.append((key_of(info['m'], cfg), pos, F, info))
        return True
    b0 = _rank((yield from _batch0_steps(xy, prm)), cfg)
    for pl in b0:                      # safety candidate: the best realisable cheap plan
        done.add(id(pl))
        try:
            if (yield from realise(pl)):
                break
        except BudgetExceeded:
            break
    b1 = _rank((yield from _batch1_steps(xy, prm, b0)), cfg) if A.time_left() is None or A.time_left() > 0 else []
    plans = _rank([p for p in b0 if id(p) not in done] + b1, cfg)
    if not plans and not cands:
        raise err or FillRefused('no closing labelling')
    best = max([c[0] for c in cands], default=-np.inf)
    nreal = len(cands)
    for pl in plans:
        if pl['ub2'] <= best + 1e-9 or nreal >= cfg['max_real']:
            break
        try:
            ok = (yield from realise(pl))
        except BudgetExceeded:
            break
        if ok:
            nreal += 1
            best = max(best, cands[-1][0])
    # rescue: the rim bound cannot see interior shear of one-row strips; if the pick is terrible or a strip,
    # add labellings that are pushed towards interior structure (cheap / negative pole cost, flat R allowed)
    top = max(cands, key=lambda c: c[0]) if cands else None
    if top is not None and (top[3]['m']['minSJ'] < cfg.get('rescue_sj', 0.4) or top[3]['m']['strip']) and (A.time_left() is None or A.time_left() > 0):
        try:
            if len(xy) % 2 == 0:
                extra = (yield from A.plans_even_steps(xy, dict(prm, nogood=8), RESCUE))
            else:
                extra = []
                for kind, v in _tries(xy)[:3]:
                    extra += (yield from A._odd_plans_steps(xy, dict(prm, odd_variants=RESCUE), kind, v))
        except BudgetExceeded:
            extra = []
        seen = {tuple(c[3]['labels']) for c in cands}
        for pl in _rank(extra, cfg)[:4]:
            if tuple(pl['labels']) in seen:
                continue
            try:
                (yield from realise(pl))
            except BudgetExceeded:
                break
    if not cands:
        raise err or BudgetExceeded('time budget exceeded before any candidate')
    cands.sort(key=lambda c: -c[0])
    kk, pos, F, info = cands[0]
    info = dict(info); info['ncand'] = len(cands); info['nplans'] = len(plans) + len(done)
    info['_alts'] = [(c[1], c[2], c[3]) for c in cands[1:cfg.get('post_top', 1)]]
    return pos, F, info


def _dopt_post_steps(pos, F, n, band=False):
    """One dopt pass (FULL, caps 1.5/2.0) under the fill's own deadline; on expiry dopt returns its best valid stage.
    Bands: FULL without guards (small thin meshes: the input-derived caps block the corner fix)."""
    left = A.time_left()
    if left is not None and left < 0.03:
        return pos, 'skipped'
    P3 = np.c_[pos, np.zeros(len(pos))]
    kw = dict(guards=False) if band and BAND['noguard'] else dict(adj_min_cap=1.5, edge_min_cap=2.0)
    out, inf = (yield from dopt.optimize_steps(P3, F, n, preset='FULL', deadline=A.deadline, cancel=A.cancelled, on_budget='best', **kw))
    if not inf.get('ok'):
        return pos, 'failed'
    return np.asarray(out, float)[:, :2], 'ok'


def _untangle_steps(pos, faces, n):
    P3 = np.c_[pos, np.zeros(len(pos))]
    out, info = (yield from dopt.optimize_steps(P3, [list(f) for f in faces], n, preset='FAST', guards=False,
                              deadline=A.deadline, cancel=A.cancelled, on_budget='best'))
    return np.asarray(out, float)[:, :2] if info.get('ok', False) else pos


def fill_steps(world, budget_s=5.0, cfg=None, cancel=None):
    """cancel: the run's token (set / is_set); polled with the deadline at every budget check."""
    t0 = time.perf_counter()
    A.set_deadline(None if budget_s is None else t0 + float(budget_s), cancel)
    try:
        world = np.asarray(world, float)
        xy, (c, u, v, nrm) = A.plane_frame(world)
        band = is_band(xy)
        cfg2 = dict(cfg or {})
        prm0 = {}
        if band:      # bands: band-only plans and price (other outlines unchanged)
            cfg2.setdefault('lam', BAND['lam'])
            cfg2.setdefault('max_real', BAND['max_real'])
            cfg2.setdefault('post_top', BAND['post_top'])
            prm0['variants'] = VARIANTS_R2 + BAND_EXTRA
        try:
            pos2, faces, info = (yield from fill_xy_steps(xy, cfg2, dict(prm0)))
        except BudgetExceeded:
            raise
        except FillRefused:
            try:
                pos2, faces, info = (yield from fill_xy_steps(xy, cfg2, dict(prm0, untangle=_untangle_steps)))
            except BudgetExceeded:
                raise
            except FillRefused:     # last resort: apattern's own driver (cost variants + relaxed set + untangle)
                pos2, faces, info = (yield from A.fill_xy_steps(xy, dict(untangle=_untangle_steps)))
                info = dict(info, fallback='A_v2')
        n = len(world)
        alts = info.pop('_alts', [])
        pos2, info['dopt'] = (yield from _dopt_post_steps(pos2, faces, n, band))
        if band and alts:      # round 3: on bands choose among the top candidates by their post-dopt key
            kc = dict(CFG, **cfg2)
            best = (key_of(A.quick_metrics(pos2, faces, n), kc), pos2, faces, info)
            for apos, aF, ainfo in alts:
                if A.time_left() is not None and A.time_left() < 0.05:
                    break
                p2, st = (yield from _dopt_post_steps(apos, aF, n, band))
                k2 = key_of(A.quick_metrics(p2, aF, n), kc)
                if k2 > best[0]:
                    best = (k2, p2, aF, dict(ainfo, dopt=st, alt=True))
            _, pos2, faces, info = best
        info['band'] = band
        P = c + pos2[:, :1] * u + pos2[:, 1:2] * v
        P[:n] = world
        info['seconds'] = time.perf_counter() - t0
        return dict(positions=P.tolist(), faces=faces, info=info)
    finally:
        A.set_deadline(None)


def fill(*args, **kwargs):
    """fill_steps run to its end (the synchronous form)."""
    return _drive(fill_steps(*args, **kwargs))
