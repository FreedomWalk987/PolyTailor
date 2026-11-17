# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""PS-1 ranker: plain Python (no numpy), deterministic, order-independent.

    order = ps1_rank(candidates, params=None)

candidate = dict with
    id       str   unique, geometry-derived (fingerprint) -> final deterministic tie-break
    valid    bool  production validator verdict (default True); invalid candidates are never ranked
    minQ     float worst corner sine (existing metric Q)
    poles    int   interior vertices with edge valence != 4
    D        float CV of face areas (population std / mean)
    cdev     float chain_dev_wp90       (0.0 if no open chain)
    stot     float strip_tot_wp90, deg  (0.0 if no open strip)
    T        float|None  turn_p95, deg  (None = no sample -> cannot trigger)
    faces    int
    padding  int   (default 0)  Takayama padding of the template, tie-break only

Result `order` = indices into `candidates` (best first) of every VALID candidate:
    block 1  gate-passing, not Pareto-dominated   (sorted by the PS-1 lexicographic key)
    block 2  gate-passing, Pareto-dominated       (same key)
    block 3  valid but below the minQ floor       (sorted by -round(minQ,6), then poles, D, -padding, faces, id)
If no valid candidate passes the floor: block 3 only (order[0] is a labelled below-floor fallback).

The PS-1 key of a gate-passing, non-dominated candidate c is
    ( band(D_c), P_c, flag(c), -round(Q_c,6), [c is not the reference], -padding_c, faces_c, id_c )
The reference (optional `reference_id`, e.g. the independent legacy/search plan, as in fill_flow.guarded_choice)
wins every exact tie through Q; padding, faces and the geometry id only separate candidates that tie with
each other and with the reference.
    band(D) = floor((D - Dmin)/w + 1e-9)          Dmin = min D over block-1 candidates
    flag    = 1 iff the VOTE-th largest of (cdev/tau1, stot/tau2, T/tau3) >= 1  (default 2nd of 3)
All comparisons are on these tuples only -> result is independent of input order.
"""
import math

DEFAULTS = {
    "floor": 0.25,            # theta_Q, gate on minQ (inclusive, 1e-12 slack as in fill_flow.eligible)
    "floor_tol": 1e-12,
    "w": 0.25,                # D band width, CV units (free parameter 1)
    "flow_scale": 1.0,        # s: multiplies the three thresholds (free parameter 2)
    "tau": (0.25, 120.0, 25.0),   # (cdev, stot [deg], T [deg]) at s = 1
    "flow_vote": 2,           # flagged iff >= vote detectors reach their threshold
    "pareto_margin": {"D": 0.10, "P": 0.0, "Q": 0.05, "F": 0.25},
    "q_round": 6,             # Q compared after rounding to 1e-6 (solver noise is ~1e-9; fill_selection.choose uses 1e-9)
    "band_eps": 1e-9,
}



def _num(x, default=0.0):
    if x is None:
        return default
    try:
        if math.isnan(x):
            return default
    except TypeError:
        return default
    return float(x)


def flow_level(c, p):
    """Fl2 (generalised: vote-th largest) of the three threshold-normalised flow detectors.
    Missing detector values count as 0 (they can never trigger)."""
    t1, t2, t3 = (t * p["flow_scale"] for t in p["tau"])
    comps = sorted(((_num(c.get("cdev")) / t1), (_num(c.get("stot")) / t2), (_num(c.get("T")) / t3)),
                   reverse=True)
    return comps[min(max(int(p["flow_vote"]), 1), 3) - 1]


def passes_gate(c, p):
    return bool(c.get("valid", True)) and _num(c.get("minQ"), -1.0) + p["floor_tol"] >= p["floor"]


def _crit(c, p):
    """Pareto criteria vector, all 'smaller is better': (D, P, -Q, Fl)."""
    return (_num(c.get("D")), float(c.get("poles", 0)), -_num(c.get("minQ"), -1.0), flow_level(c, p))


def dominates(a, b, p):
    """a Pareto-dominates b (with margins): no criterion worse, at least one better by > margin."""
    ca, cb = _crit(a, p), _crit(b, p)
    m = p["pareto_margin"]
    margins = (m["D"], m["P"], m["Q"], m["F"])
    if any(x > y for x, y in zip(ca, cb)):
        return False
    return any(x + g < y for x, y, g in zip(ca, cb, margins))


def pareto_dominated(cs, idx, p):
    """set of indices (subset of idx) dominated by another member of idx."""
    out = set()
    for j in idx:
        for i in idx:
            if i != j and dominates(cs[i], cs[j], p):
                out.add(j)
                break
    return out


def band_index(d, dmin, p):
    w = p["w"]
    if w <= 0:
        return d            # exact comparison (no banding)
    return math.floor((d - dmin) / w + p["band_eps"])


def _key(c, bd, p, reference_id=None):
    pad = int(c.get("padding", 0) or 0)
    fl = flow_level(c, p)
    return (bd, int(c.get("poles", 0)), 1 if fl >= 1.0 - 1e-12 else 0,
            -round(_num(c.get("minQ"), -1.0), p["q_round"]),
            0 if (reference_id is not None and str(c["id"]) == str(reference_id)) else 1,
            -pad, int(c.get("faces", 0)), str(c["id"]))


def _key_below(c, p):
    pad = int(c.get("padding", 0) or 0)
    return (-round(_num(c.get("minQ"), -1.0), p["q_round"]), int(c.get("poles", 0)), _num(c.get("D")),
            -pad, int(c.get("faces", 0)), str(c["id"]))


def ps1_rank(candidates, params=None, reference_id=None):
    p = DEFAULTS if params is None else params
    cs = list(candidates)
    ids = [str(c["id"]) for c in cs]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate ids must be unique")
    valid = [i for i, c in enumerate(cs) if c.get("valid", True)]
    passing = [i for i in valid if passes_gate(cs[i], p)]
    below = [i for i in valid if i not in set(passing)]
    dominated = pareto_dominated(cs, passing, p)
    core = [i for i in passing if i not in dominated]
    dmin = min((_num(cs[i].get("D")) for i in core), default=0.0)
    keyed = {}
    for i in passing:
        bd = band_index(_num(cs[i].get("D")), dmin, p)
        keyed[i] = (1 if i in dominated else 0,) + _key(cs[i], bd, p, reference_id)
    block12 = sorted(passing, key=lambda i: keyed[i])
    block3 = sorted(below, key=lambda i: _key_below(cs[i], p))
    if passing:
        return block12 + block3
    return block3   # nothing passes the floor: the labelled below-floor fallback
