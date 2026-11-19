# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Global quality ranking after an independent minQ feasibility gate."""
from collections import defaultdict
import math
import numpy as np


def metrics(plan):
    """Measure A and all-regular, all-quad primal T in degrees on final positions.

    Missing samples are N/A, never zero.
    Plans are immutable after admission; cache only on that plan instance.
    """
    cached = plan.get('_global_metrics_v1')
    if cached is not None:
        return cached
    from . import fill
    p = np.asarray(plan['positions'], float)
    faces = fill.faces_of(plan)
    incidence = defaultdict(int)
    neighbors = defaultdict(set)
    adjacent = defaultdict(set)
    triangles = set()
    face_quality = np.empty(len(faces))
    face_shape = np.empty(len(faces))
    face_area = np.empty(len(faces))
    face_incidence = defaultdict(list)
    minimum = 1.
    signed = 1.
    for count in sorted({len(f) for f in faces}):
        indices = [i for i, f in enumerate(faces) if len(f) == count]
        rings = np.asarray([faces[i] for i in indices], dtype=int)
        x = p[rings]
        a = np.roll(x, -1, axis=1)-x
        b = np.roll(x, 1, axis=1)-x
        cross = np.cross(a, b)
        normals = cross.sum(axis=1)
        normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-30)
        denom = np.maximum(np.linalg.norm(a, axis=2)*np.linalg.norm(b, axis=2), 1e-30)
        minimum = min(minimum, float((np.linalg.norm(cross, axis=2)/denom).min()))
        projected = np.sum(cross*normals[:, None, :], axis=2)
        signed = min(signed, float((projected/denom).min()))
        face_quality[indices] = (np.linalg.norm(cross, axis=2)/denom).min(axis=1)
        face_shape[indices] = (2*projected/np.maximum(np.sum(a*a+b*b, axis=2), 1e-30)).min(axis=1)
        face_area[indices] = .5*np.linalg.norm(np.cross(x, np.roll(x, -1, axis=1)).sum(axis=1), axis=1)
    for fi, f in enumerate(faces):
        if len(f) != 4:
            triangles.update(f)
        for i, v in enumerate(f):
            w = f[(i+1) % len(f)]
            incidence[tuple(sorted((v, w)))] += 1
            face_incidence[tuple(sorted((v, w)))].append(fi)
            neighbors[v].add(w)
            neighbors[w].add(v)
            adjacent[v].add(tuple(sorted((f[i-1], w))))
    rim = [e for e, count in incidence.items() if count == 1]
    internal = [e for e, count in incidence.items() if count == 2]
    boundary = {v for e in rim for v in e}
    turns = []
    opposite = {}
    for v, ns in neighbors.items():
        if v in boundary or v in triangles or len(ns) != 4:
            continue
        for a in sorted(ns):
            other = [b for b in ns if b != a and tuple(sorted((a, b))) not in adjacent[v]]
            if len(other) == 1:
                opposite[(v, a)] = other[0]
            if len(other) == 1 and a < other[0]:
                b = other[0]
                x, y = p[v]-p[a], p[b]-p[v]
                cosine = float(x@y)/max(float(np.linalg.norm(x)*np.linalg.norm(y)), 1e-30)
                turns.append(math.degrees(math.acos(float(np.clip(cosine, -1., 1.)))))
    alignment = None
    if rim and internal:
        ra = p[[a for a, b in rim]]
        rd = p[[b for a, b in rim]]-ra
        rl2 = np.sum(rd*rd, axis=1)
        # Bound temporary memory for large outlines, preserving first-nearest
        # edge ties in face-incidence order, as in the documented definition.
        angles, lengths = [], []
        for start in range(0, len(internal), 128):
            edges = np.asarray(internal[start:start+128])
            mid = (p[edges[:, 0]]+p[edges[:, 1]])*.5
            d = p[edges[:, 1]]-p[edges[:, 0]]
            delta = mid[:, None, :]-ra
            t = np.clip(np.sum(delta*rd, axis=2)/np.maximum(rl2, 1e-30), 0., 1.)
            k = np.argmin(np.linalg.norm(delta-t[:, :, None]*rd, axis=2), axis=1)
            length = np.linalg.norm(d, axis=1)
            cosine = np.abs(np.sum(d*rd[k], axis=1))/np.maximum(length*np.sqrt(rl2[k]), 1e-30)
            angle = np.degrees(np.arccos(np.clip(cosine, -1., 1.)))
            angles.extend(np.minimum(angle, 90.-angle))
            lengths.extend(length)
        alignment = float(np.average(angles, weights=lengths))
    strip_angles = []
    transitions = []
    for edge, adjacent_faces in face_incidence.items():
        if len(adjacent_faces) != 2:
            continue
        left, right = adjacent_faces
        transitions.append(abs(math.log(max(face_area[left], 1e-30)/max(face_area[right], 1e-30))))
        if len(faces[left]) != 4 or len(faces[right]) != 4:
            continue
        midpoints = []
        for fi in adjacent_faces:
            face = faces[fi]
            j = next(j for j in range(4) if tuple(sorted((face[j], face[(j+1) % 4]))) == edge)
            midpoints.append((p[face[(j+2) % 4]]+p[face[(j+3) % 4]])*.5)
        mid = (p[edge[0]]+p[edge[1]])*.5
        a, b = mid-midpoints[0], midpoints[1]-mid
        cosine = float(a@b)/max(float(np.linalg.norm(a)*np.linalg.norm(b)), 1e-30)
        strip_angles.append(math.degrees(math.acos(float(np.clip(cosine, -1., 1.)))))
    turning = float(np.percentile(turns, 95)) if turns else None
    valid = bool(faces) and bool(np.isfinite(p).all()) and signed > 0 and minimum+1e-12 >= .20
    result = dict(A=alignment, T=turning, minQ=minimum,
                  signed_minQ=signed, valid=valid,
                  poles=sum(v not in boundary and len(ns) != 4 for v, ns in neighbors.items()),
                  faces=len(faces),
                  q10=float(np.percentile(face_quality, 10)) if len(faces) else None,
                  meanQ=float(face_quality.mean()) if len(faces) else None,
                  shape10=float(np.percentile(face_shape, 10)) if len(faces) else None,
                  strip95=float(np.percentile(strip_angles, 95)) if strip_angles else None,
                  adjacent_area_log=float(np.mean(transitions)) if transitions else 0.)
    result.update(_global_parts(p, faces, neighbors, opposite, internal, boundary,
                                face_quality, face_shape, face_area, face_incidence, result))
    plan['_global_metrics_v1'] = result
    return result


def fingerprint(plan):
    """A geometry digest of the plan, used only for deterministic ties."""
    cached = plan.get('_m11_fingerprint')
    if cached is None:
        import hashlib
        import json
        from . import fill
        payload = [np.asarray(plan['positions']).round(10).tolist(),
                   [[int(v) for v in f] for f in fill.faces_of(plan)]]
        cached = hashlib.sha256(json.dumps(payload).encode()).hexdigest()
        plan['_m11_fingerprint'] = cached
    return cached


def _global_parts(p, faces, neighbors, opposite, internal, boundary, q, shape, areas, incidence, m):
    """Fixed physical diagnostics; no candidate-pool normalization or pole count cost."""
    def angle(a, b):
        return math.degrees(math.acos(float(np.clip(float(a@b)/max(float(np.linalg.norm(a)*np.linalg.norm(b)), 1e-30), -1., 1.))))
    unseen = set(internal)
    variation = []
    weights = []
    while unseen:
        start = None
        for edge in sorted(unseen):
            for a, b in (edge, edge[::-1]):
                if (a, b) not in opposite:
                    start = (a, b)
                    break
            if start is not None:
                break
        if start is None:
            start = min(unseen)
        a, b = start
        vertices = [a]
        used = set()
        closed = False
        while True:
            edge = tuple(sorted((a, b)))
            if edge in used:
                closed = True
                break
            used.add(edge)
            unseen.discard(edge)
            vertices.append(b)
            following = opposite.get((b, a))
            if following is None:
                break
            a, b = b, following
        chain = p[vertices]
        if closed:
            chain = chain[:-1]
            directions = np.roll(chain, -1, axis=0)-chain
            turns = [angle(directions[i-1], directions[i]) for i in range(len(directions))]
            if len(turns) >= 2:
                variation.append(float(np.mean(np.abs(np.asarray(turns)-np.roll(turns, 1)))))
                weights.append(float(np.linalg.norm(directions, axis=1).sum()))
        else:
            directions = np.diff(chain, axis=0)
            turns = [angle(a, b) for a, b in zip(directions, directions[1:])]
            if len(turns) >= 2:
                variation.append(float(np.mean(np.abs(np.diff(turns)))))
                weights.append(float(np.linalg.norm(directions, axis=1).sum()))
    fairness = float(np.average(variation, weights=weights)) if variation else None
    log_area = np.log(np.maximum(areas, 1e-30))
    poles = [v for v, ns in neighbors.items() if v not in boundary and len(ns) != 4]
    at = defaultdict(list)
    for fi, face in enumerate(faces):
        for v in face:
            at[v].append(fi)
    badness = []
    for v in poles:
        dispersion = float(log_area[at[v]].std())
        scale = float(np.median([np.linalg.norm(p[v]-p[w]) for w in neighbors[v]]))
        distance = min((float(np.linalg.norm(p[v]-p[w])) for w in poles if w != v), default=float('inf'))
        crowding = max(0., 1.-distance/max(scale, 1e-30))
        badness.append(.5*dispersion/(1.+dispersion)+.5*crowding)
    disturbance = float(np.mean(badness)) if badness else 0.
    def decay(value, scale):
        return math.exp(-max(value, 0.)/scale) if value is not None else .5
    element = float(np.clip(.30*m['meanQ']+.20*m['q10']+.50*m['shape10'], 0., 1.)) if faces else 0.
    distribution = .80*math.exp(-m['adjacent_area_log'])+.20*math.exp(-float(log_area.std())) if faces else 0.
    flow = (.40*decay(m['strip95'], 45.)+.25*decay(m['T'], 45.)
            +.15*decay(m['A'], 30.)+.20*decay(fairness, 30.))
    score = float(np.clip(100*(.30*element+.30*distribution+.40*flow)-2*disturbance, 0., 100.))
    return dict(S=score, raw_U=m['minQ']*.7**m['poles'])


RANK_DELTA = 3.0   # main-search rank R1: the band is every plan within RANK_DELTA of the best S*
RANK_MINQ_GUARD = .005   # rule (a): today's pick stays when the new one's minQ is lower by more than this


def band_key(plan):
    """the in-band order, rule (a) (smaller sorts first): fewest interior poles, then fewest rim poles (the flat rim
    count of singularities()), then higher S*."""
    m = metrics(plan)
    return (int(m['poles']), int(singularities(plan)[1]), -float(m['S']))


def band_key_before(plan):
    """the in-band order before rule (a): fewest interior poles, then higher S* (the minQ guard's reference)."""
    m = metrics(plan)
    return (int(m['poles']), -float(m['S']))


def r1_band(plans, floor=.20):
    """(band, pool): pool = the eligible plans (all if none is); band = those with S* within RANK_DELTA
    of the pool's best S* (the default rule). Plans whose metrics fail are left out."""
    pool = []
    for p in plans:
        try:
            m = metrics(p)
            s, u = float(m['S']), float(m['raw_U'])
        except Exception:
            continue
        if math.isfinite(s) and math.isfinite(u):
            pool.append((p, s, u, eligible(p, floor)))
    ok = [x for x in pool if x[3]] or pool
    if not ok:
        return [], []
    top = max(x[1] for x in ok)
    return [x for x in ok if x[1] >= top - RANK_DELTA - 1e-9], ok


def eligible(plan, floor=.20):
    if plan is None:
        return False
    m = metrics(plan)
    return (m['faces'] > 0 and m['signed_minQ'] > 0 and m['minQ']+1e-12 >= floor
            and math.isfinite(m['S']) and bool(np.isfinite(np.asarray(plan['positions'])).all()))


def score(plan):
    return metrics(plan)['S'] if plan is not None else -float('inf')


def guarded_choice(candidates, reference, floor=.20):
    """The default-rule pick; when it has none (multi-panel plans, nothing eligible), the fallback: feasibility, then
    global S, the density guard, and the feature ranking that may replace that pick."""
    try:
        chosen = r1_choice(candidates, reference, floor)
        if chosen is not None:
            return chosen
    except Exception:   # never lose the released choice
        pass
    chosen = _s_choice(candidates, reference, floor)
    if chosen is not None:   # density guard; the feature ranking then starts from the guarded pick
        try:
            chosen = density_guard(candidates, reference, chosen, floor)
        except Exception:
            pass
    if chosen is None:
        return chosen
    try:
        return ps1_choice(candidates, reference, chosen, floor)
    except Exception:   # never lose the released choice
        return chosen


# density guard: an over-dense S pick (more faces than both
# 1.5*(n/4)^2 and 2.5*area/mean_edge^2, rim n >= 12) is replaced by the best-S eligible candidate
# under both caps whose minQ is at least 0.10 higher; otherwise the pick is kept.
DENSITY_K_RIM, DENSITY_K_AREA, DENSITY_DQ, DENSITY_NMIN = 1.5, 2.5, .10, 12


def density_guard(candidates, reference, chosen, floor=.20):
    n = int(chosen.get('boundary') or 0)
    if chosen.get('panel_count') or n < DENSITY_NMIN:
        return chosen
    rim = np.asarray(chosen['positions'], float)[:n]
    edges = np.linalg.norm(np.roll(rim, -1, 0)-rim, axis=1)
    area = .5*float(np.linalg.norm(np.cross(rim, np.roll(rim, -1, 0)).sum(0)))
    caps = (DENSITY_K_RIM*(n/4.)**2, DENSITY_K_AREA*area/max(float(edges.mean())**2, 1e-30))
    m = metrics(chosen)
    if m['faces'] <= max(caps):   # dense only when above BOTH caps
        return chosen
    cap = min(caps)               # the substitute pick must be under both caps
    options = list(candidates)
    if reference is not None and not any(p is reference for p in options):
        options.append(reference)
    better = [p for p in options if eligible(p, floor) and metrics(p)['faces'] <= cap
              and metrics(p)['minQ'] >= m['minQ']+DENSITY_DQ]
    if not better:
        return chosen
    best = max(round(score(p), 12) for p in better)
    return min((p for p in better if round(score(p), 12) == best), key=fingerprint)


# Feature ranking, with guards (the fallback when the default rule has no eligible plan)
# that keep the current (S) choice.
PS1_FACE_RATIO = 1.5    # its pick may not have more than 1.5x the faces of the current pick
PS1_FACE_FLOOR = .7     # its pick may not have under 0.7x the faces
PS1_FLOW_GUARD = True   # its pick may not be (nearly) flow-flagged when the current pick is not flagged
PS1_FLOW_LIMIT = .95    # Fl2 of its pick that counts as nearly flagged
PS1_MINQ_DROP = .10     # its pick may not have a minQ more than this below the current pick


def ps1_features(plan):
    cached = plan.get('_ps1_features_v1')
    if cached is None:
        from . import fill
        from .fill_ps1_features import mesh_features
        f = mesh_features(np.asarray(plan['positions'], float), fill.faces_of(plan))
        cached = dict(faces=f['faces'], poles=f['poles'], minQ=f['minQ'], signed_minQ=f['signed_minQ'],
                      D=f['D'], T=f['T'], cdev=f['cdev'], stot=f['stot'])
        plan['_ps1_features_v1'] = cached
    return cached


def ps1_choice(candidates, reference, current, floor=.20):
    from . import fill_ps1_rank as R
    options = list(candidates)
    if reference is not None and not any(p is reference for p in options):
        options.append(reference)
    usable = [p for p in options if eligible(p, floor)]
    if not any(p is current for p in usable):
        usable.append(current)
    rows, seen = [], {}
    for p in usable:
        key = fingerprint(p)
        if key in seen:
            continue
        seen[key] = p
        rows.append(dict(ps1_features(p), id=key, valid=True))
    cur_id = fingerprint(current)
    order = R.ps1_rank(rows, R.DEFAULTS, reference_id=cur_id)
    pick = rows[order[0]] if order else None
    cur = next(r for r in rows if r['id'] == cur_id)
    result = current
    if pick is not None and pick['id'] != cur_id:   # the guards, in order: floor, face ratio / floor, flow, minQ
        if not (not R.passes_gate(pick, R.DEFAULTS)
                or pick['faces'] > PS1_FACE_RATIO*cur['faces']
                or pick["faces"] < PS1_FACE_FLOOR*cur["faces"]   # the face floor is always on
                or (PS1_FLOW_GUARD and R.flow_level(pick, R.DEFAULTS) >= PS1_FLOW_LIMIT-1e-12
                    and R.flow_level(cur, R.DEFAULTS) < 1.-1e-12)
                or pick['minQ'] < cur['minQ']-PS1_MINQ_DROP-1e-12):
            result = seen[pick['id']]
    return result


# Default ranking rule.
# The default is the eligible candidate with the
# highest minQ * 0.7^(interior poles + rim singularities). An interior pole is a non-rim vertex whose valence is
# not 4; a rim singularity is a rim vertex whose face count differs from its interior angle / 90 deg by more than
# R1_RIM_TOL (Cubit End/Side/Corner/Reversal classes). It replaces S, the density
# guard and the feature ranking with its guards for the default pick.
# When the default pick has minQ < 0.3, the best default-rule candidate with
# minQ >= 0.3 replaces it, unless it has more than R1_GATE_CAP more singularities.
# ties of U go to the higher S, then to a scale-free position key (the same order at any
# model size); a rim slot without faces (an unused odd-rim split point) is not a singularity.
R1_POLE_BASE = .7
R1_RIM_TOL = .65
R1_SJ_GATE = .3
R1_GATE_CAP = 4


def singularities(plan):
    """(interior poles, rim singularities) of a final mesh; cached on the plan instance."""
    cached = plan.get('_r1_singularities_v1')
    if cached is not None:
        return cached
    from . import fill
    p = np.asarray(plan['positions'], float)
    n = int(plan.get('boundary') or 0)
    neighbors = defaultdict(set)
    count = defaultdict(int)
    angle = defaultdict(float)
    for f in fill.faces_of(plan):
        k = len(f)
        for i, v in enumerate(f):
            a, b = int(f[(i+1) % k]), int(f[i-1])
            neighbors[v].update((a, b))
            count[v] += 1
            if v < n:   # the corner angles at a rim vertex add up to its interior rim angle
                u, w = p[a]-p[v], p[b]-p[v]
                angle[v] += math.atan2(float(np.linalg.norm(np.cross(u, w))), float(u @ w))
    poles = sum(1 for v in neighbors if v >= n and len(neighbors[v]) != 4)
    rim = sum(1 for v in range(n) if count[v] and abs(count[v]-math.degrees(angle[v])/90.) > R1_RIM_TOL)
    plan['_r1_singularities_v1'] = result = (poles, rim)
    return result


def is_strip(plan):
    """hybrid: a degenerate strip has no interior vertex (every vertex of its faces is on the rim)."""
    from . import fill
    n = int(plan.get('boundary') or 0)
    return not any(v >= n for face in fill.faces_of(plan) for v in face)


def r1_min_sj(plan):
    """minQ rounded to 1e-6 (scale-free), so near-identical meshes from different generators tie."""
    return round(metrics(plan)['minQ'], 6)


def r1_value(plan):
    return r1_min_sj(plan)*R1_POLE_BASE**sum(singularities(plan))


def r1_key(plan):
    """Total order for R1: U, then S, then positions relative to the rim bounding box (scale-free)."""
    cached = plan.get('_r1_key_v1')
    if cached is None:
        p = np.asarray(plan['positions'], float)
        rim = p[:max(int(plan.get('boundary') or 0), 1)]
        low = rim.min(axis=0)
        extent = max(float(np.ptp(rim, axis=0).max()), 1e-30)
        cached = tuple(np.round((p-low)/extent, 4).ravel().tolist())
        plan['_r1_key_v1'] = cached
    return (r1_value(plan), round(score(plan), 9), cached)


def r1_choice(candidates, reference, floor=.20):
    """The default rule's pick; None (use the S / feature-ranking stack) for multi-panel plans or when nothing is eligible."""
    options = list(candidates)
    if reference is not None and not any(p is reference for p in options):
        options.append(reference)
    if any(p.get('panel_count') for p in options):
        return None
    usable = [p for p in options if eligible(p, floor)]
    if not usable:
        return None
    chosen = plain = max(usable, key=r1_key)
    if r1_min_sj(plain) < R1_SJ_GATE:   # C
        cap = sum(singularities(plain))+R1_GATE_CAP
        gated = [p for p in usable if r1_min_sj(p) >= R1_SJ_GATE and sum(singularities(p)) <= cap]
        if gated:
            chosen = max(gated, key=r1_key)
    return chosen


def _s_choice(candidates, reference, floor=.20):
    options = list(candidates)
    if reference is not None and not any(p is reference for p in options):
        options.append(reference)
    usable = [p for p in options if eligible(p, floor)]
    if not usable:
        return reference
    best = max(round(score(p), 12) for p in usable)
    tied = [p for p in usable if round(score(p), 12) == best]
    if any(p is reference for p in tied):
        return reference
    return min(tied, key=fingerprint)


def short_label(plan, quality_floor=.20):
    m = metrics(plan)
    # U is the historical raw formula, not a threshold or the active ranking.
    floor = round(float(quality_floor), 4)
    shown = f"{m['S']:.1f}" if eligible(plan, floor) else 'N/A'
    return f"S* {shown} | U {m['raw_U']:.3f} | Q {m['minQ']:.2f} | P {m['poles']} | F {m['faces']}"


