# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Boundary-anchored combinatorial maps and a small ranked AUTO portfolio."""
from collections import defaultdict, deque
import math


def _anchored_map(plan):
    if plan is None:
        return None, {}
    from . import fill
    faces = [tuple(q) for q in fill.faces_of(plan)]
    order = plan.get('order', list(range(plan['boundary'])))
    labels = {i: int(source) for i, source in enumerate(order) if source is not None}
    boundary = sorted(labels, key=labels.get)
    incidence = defaultdict(list)
    for k, face in enumerate(faces):
        for a, b in zip(face, face[1:]+face[:1]):
            incidence[tuple(sorted((a, b)))].append(k)
    # Walking faces from the anchored rim makes interior numbering irrelevant.
    queue = deque(zip(boundary, boundary[1:]+boundary[:1]))
    visited = set()
    next_label = len(boundary)
    while queue:
        a, b = queue.popleft()
        for k in incidence[tuple(sorted((a, b)))]:
            if k in visited:
                continue
            visited.add(k)
            face = faces[k]
            index = face.index(a)
            ring = face[index:]+face[:index]
            if ring[1] != b:
                ring = (ring[0],)+ring[:0:-1]
            for vertex in ring:
                if vertex not in labels:
                    labels[vertex] = next_label
                    next_label += 1
            queue.extend(zip(ring, ring[1:]+ring[:1]))
    if len(visited) != len(faces):
        raise ValueError('Layout is not connected to its boundary')
    def canonical(face):
        ring = tuple(labels[v] for v in face)
        return min(seq[i:]+seq[:i] for seq in (ring, ring[::-1]) for i in range(len(ring)))
    return (len(boundary), tuple(sorted(canonical(face) for face in faces))), labels


def signature(plan):
    return _anchored_map(plan)[0]


def _same_mesh(first, second):
    """Compare anchored vertices, independent of interior numbering."""
    import numpy as np
    left, a = _anchored_map(first)
    right, b = _anchored_map(second)
    if left != right:
        return False
    x = np.asarray(first['positions'])[sorted(a, key=a.get)]
    y = np.asarray(second['positions'])[sorted(b, key=b.get)]
    scale = max(float(np.ptp(x, axis=0).max()), 1e-12)
    return x.shape == y.shape and bool(np.allclose(x, y, rtol=0., atol=scale*1e-10))


LEGACY_KEY = ('legacy',)


def _representative_key(plan):
    """A total order for numerically tied placements of the same anchored map."""
    import numpy as np
    from . import fill
    positions = np.asarray(plan['positions'], dtype='<f8')
    order = tuple(-1 if v is None else int(v) for v in plan.get('order', ()))
    return (positions.shape, positions.tobytes(),
            tuple(tuple(map(int, face)) for face in fill.faces_of(plan)), order,
            str(plan.get('pattern', '')), int(plan.get('rotation', 0)),
            int(plan.get('shift', 0)), str(plan.get('network_method', '')))


def _representative(variants, prefer_plain=False):
    """The placement that represents one topology: R1's or the best S. With `prefer_plain` (2.5.0
    hybrid, full pool) an exact tie goes to a plain plan, and ties between classification plans to the fingerprint;
    ties between plain plans keep the 2.4 order."""
    from .fill_flow import fingerprint, r1_key
    variants = list(variants)
    if not prefer_plain:
        return max(variants, key=r1_key)   # R1 is always on
    cls = lambda p: bool(p.get('_cls_candidate'))
    return max(variants, key=lambda p: (r1_key(p), not cls(p), fingerprint(p) if cls(p) else ''))


class Portfolio:
    """Rank validated topology representatives; keep an artist's choice stable."""
    def __init__(self, limit=4, ranking='FLOW', quality_floor=.20):
        self.ranking = ranking
        self.quality_floor = round(float(quality_floor), 4)
        self.reference = Portfolio(limit, 'LEGACY') if ranking == 'FLOW' else None
        self.limit = limit
        self.entries = {}
        self.placements = {}
        self.selected = None
        self.locked = False
        self.legacy = None
        self.legacy_slot = None
        self.legacy_match = None
        self.has_cls = False     # hybrid: any classification candidate seen
        self.plain_seen = {}     # hybrid: key -> order of the first non-classification arrival (A-OFF order)

    def _normal_keys(self):
        return sorted(self.entries, key=lambda k: (-round(self.entries[k][0], 12), k))

    def _refresh_legacy(self):
        self.legacy_slot = self.legacy_match = None
        if self.legacy is None:
            return
        key = signature(self.legacy)
        if key in self.entries and _same_mesh(self.entries[key][1], self.legacy):
            self.legacy_match = key
        keys = self._normal_keys()
        if self.locked and self.selected == LEGACY_KEY:
            self.legacy_slot = LEGACY_KEY
        elif self.legacy_match is not None:
            if not keys or self.legacy_match != keys[0]:
                self.legacy_slot = self.legacy_match
        else:
            self.legacy_slot = LEGACY_KEY

    def add_legacy(self, plan, select_if_empty=False):
        if plan is None or (self.locked and self.selected == LEGACY_KEY):
            return False
        if self.reference is not None:
            self.reference.add_legacy(plan, select_if_empty)
        self.legacy = plan
        self._refresh_legacy()
        if select_if_empty and not self.entries and self.selected is None:
            self.selected = self.legacy_slot
        return True

    def add(self, plan):
        from .fill_progressive import utility
        if plan is None:
            return False
        if plan.get('_legacy_layout'):
            return self.add_legacy(plan)
        if self.reference is not None:
            # Extra cuts must not suppress any released generator or field search.
            if not plan.get('_corner_cut_candidate') and not plan.get('_direct') and not plan.get('_step4_candidate'):
                self.reference.add(plan)
            from .fill_flow import fingerprint, eligible, score
            key = signature(plan)
            if plan.get('_step4_candidate'):   # step 4: New's fill is its own entry, never a Legacy placement
                key = key + ('step4',)
            if self.locked and key == self.selected:
                return False
            if not eligible(plan, self.quality_floor):
                return False
            if plan.get('_cls_candidate'):
                self.has_cls = True
            else:
                self.plain_seen.setdefault(key, len(self.plain_seen))
            variants = self.placements.setdefault(key, {})
            twin = variants.get(fingerprint(plan))
            # hybrid: a classification plan never replaces an identical plain one (any arrival order)
            if twin is None or not plan.get('_cls_candidate') or twin.get('_cls_candidate'):
                if twin is not None and twin.get('_cls_candidate') and not plan.get('_cls_candidate'):
                    del variants[fingerprint(plan)]   # the plain plan takes its own arrival slot
                variants[fingerprint(plan)] = plan
            candidate = _representative(variants.values())
            self.entries[key] = (score(candidate), candidate)
            self._refresh_legacy()
            self.plan  # Refresh the automatic selected key.
            return True
        value = utility(plan)
        if not math.isfinite(value):
            return False
        current = self.plan
        field = bool(plan.get('_field_candidate'))
        incumbent_value = utility(current)
        key = signature(plan)
        old = self.entries.get(key)
        if self.ranking == 'LEGACY' and field and old is not None and value <= old[0] + 1e-12:
            return False
        # A selected plan must remain the exact mesh the artist chose.
        if self.locked and key == self.selected:
            return False
        if old is not None:
            rank, previous_rank = round(value, 12), round(old[0], 12)
            if rank < previous_rank:
                return False
            if rank == previous_rank and _representative_key(plan) >= _representative_key(old[1]):
                return False
        # The same candidate set must pick the same placement regardless of
        # whether a network-plan result or network seed arrives first.
        self.entries[key] = (value, plan)
        self._refresh_legacy()
        if not self.locked and (not field or value > incumbent_value + 1e-12):
            self.selected = self._normal_keys()[0]
        if not self.locked:
            self.default = self.selected
        elif not field or value > utility(self._plan_of(getattr(self, 'default', None))) + 1e-12:
            self.default = self._normal_keys()[0]   # the default keeps moving after a wheel step
        return True

    def _plan_of(self, key):
        if key == LEGACY_KEY:
            return self.legacy
        return self.entries[key][1] if key in self.entries else None

    @property
    def search_plan(self):
        return self.reference.plan if self.reference is not None else self.plan

    def _choose_guarded(self):
        """The guarded default; it is also the selection until the wheel is touched.
        after a wheel step the default keeps being updated (it stays entry 1)
        while the artist's selection is kept by identity."""
        from .fill_flow import guarded_choice
        reps = self._full_representatives() if self.has_cls else None   # hybrid: computed, not stored
        candidates = list(reps.values()) if reps is not None else [plan for score, plan in self.entries.values()]
        if self.legacy is not None:
            candidates.append(self.legacy)
        if getattr(self, 'rank_r1', False):   # rule R1 delta 3 (study band_max), no guards
            chosen, key = self._r1_choice(candidates), None
        else:
            chosen = guarded_choice(candidates, self.legacy, self.quality_floor)
            chosen, key = self._hybrid(chosen, reps)
        if reps is not None:
            self._apply_representatives(reps)
        if chosen is not None and key is None:
            key = next((k for k, (score, plan) in self.entries.items() if plan is chosen), None)
            if key is None:
                if self.locked:
                    key = LEGACY_KEY if self.legacy_match is None or self.selected == LEGACY_KEY else self.legacy_match
                else:
                    # A released standalone legacy slot can also be the fixed reference.
                    self.legacy = chosen
                    self.legacy_slot = LEGACY_KEY
                    key = LEGACY_KEY
        self.default = key
        if not self.locked:
            self.selected = key

    def _r1_choice(self, candidates):
        """rule (a) (_r1_pick). Step 4: with New's fill among the candidates (_step4_candidate), the pick over all of
        them stays only if it has no more interior poles than the pick over the Legacy candidates alone and its minQ
        is not lower than that one's by more than RANK_MINQ_GUARD; otherwise the Legacy-only pick stays. An ineligible
        Legacy-only pick (below the floor: r1_band used the whole pool) never stays over an eligible pick."""
        pick = self._r1_pick(candidates)
        if pick is None or not any(p is not None and p.get('_step4_candidate') for p in candidates):
            return pick
        base = self._r1_pick([p for p in candidates if p is not None and not p.get('_step4_candidate')])
        if base is None or base is pick:
            return pick
        from .fill_flow import eligible, metrics, RANK_MINQ_GUARD
        try:
            if eligible(pick, self.quality_floor) and not eligible(base, self.quality_floor):
                return pick   # never an ineligible Legacy fill over an eligible one
            m, b = metrics(pick), metrics(base)
            if m['poles'] <= b['poles'] and m['minQ'] >= b['minQ'] - RANK_MINQ_GUARD:
                return pick
        except Exception:
            pass
        return base

    def _r1_pick(self, candidates):
        """rule (a): within RANK_DELTA of the best S* (eligible candidates), the fewest interior poles, then the
        fewest rim poles, then the higher S*; ties keep the candidates' order. Guard: when that is not the pick of
        the order before rule (a) (interior poles, then S*) and its minQ is lower than that pick's by more than
        RANK_MINQ_GUARD, the earlier pick stays."""
        from .fill_flow import r1_band, band_key, band_key_before, metrics, RANK_MINQ_GUARD
        band, _ok = r1_band([p for p in candidates if p is not None], self.quality_floor)
        best, best_k, before, before_k = None, None, None, None
        for p, s, u, e in band:
            try:
                k0 = band_key_before(p)
            except Exception:
                continue
            try:
                k = band_key(p)
            except Exception:   # no rim count: after the fills with the same interior poles
                k = (k0[0], math.inf, k0[1])
            if best is None or k < best_k:
                best, best_k = p, k
            if before is None or k0 < before_k:
                before, before_k = p, k0
        if best is not None and before is not None and best is not before:
            try:
                if metrics(best)['minQ'] < metrics(before)['minQ'] - RANK_MINQ_GUARD:
                    return before
            except Exception:
                return before
        return best

    def _full_representatives(self):
        """hybrid: the representative over all placements per topology (entries order; the locked key keeps
        the artist's mesh). Computed fresh on every call, so no earlier decision is carried over."""
        reps = {}
        for key, (score, plan) in self.entries.items():
            variants = self.placements.get(key)
            reps[key] = (plan if (self.locked and key == self.selected) or not variants
                         else _representative(variants.values(), prefer_plain=True))
        return reps

    def _apply_representatives(self, reps):
        """Write only the entries whose representative changed; refresh the legacy match once."""
        from .fill_flow import score
        changed = False
        for key, rep in reps.items():
            if self.entries[key][1] is not rep:
                self.entries[key] = (score(rep), rep)
                changed = True
        if changed:
            self._refresh_legacy()

    def _hybrid(self, chosen, reps):
        """hybrid: the default without the classification candidates stands unless it
        is a degenerate strip (no interior vertex) and the default over all candidates is not; then that is used. Returns
        (plan, key or None); a plain win also becomes its topology's representative in `reps`."""
        from .fill_flow import fingerprint, guarded_choice, is_strip
        if chosen is None or reps is None:
            return chosen, None
        base = []
        for key in sorted(self.plain_seen, key=self.plain_seen.get):   # A-OFF arrival order (ties)
            plain = [p for p in self.placements.get(key, {}).values() if not p.get('_cls_candidate')]
            if plain:
                base.append((key, _representative(plain)))
        options = [p for _, p in base]+([self.legacy] if self.legacy is not None else [])
        plain = guarded_choice(options, self.legacy, self.quality_floor)
        strip = plain is not None and is_strip(plain)
        # the pick over all candidates replaces the A-OFF pick only when it is not a strip
        if plain is None or (strip and not is_strip(chosen)):
            return chosen, None
        fingerprint(plain), fingerprint(chosen)   # kept: caches both tie digests on the plans, as before
        if plain is self.legacy:
            return plain, None
        key = next(k for k, p in base if p is plain)
        if key in reps and not (self.locked and key == self.selected):
            reps[key] = plain   # the A-OFF placement represents its topology
        return plain, key

    def _flow_keys(self):
        from .fill_flow import score as value
        self._choose_guarded()
        keys = list(self.entries)
        if self.legacy is not None:
            if self.legacy_match is None:
                keys.append(LEGACY_KEY)
            elif self.selected == LEGACY_KEY:
                keys.remove(self.legacy_match)
                keys.append(LEGACY_KEY)
        keys.sort(key=lambda k: (-round(value(self.legacy if k == LEGACY_KEY else self.entries[k][1]), 12), (1, ()) if k == LEGACY_KEY else (0, k)))
        shown = keys[:self.limit]
        # Keep the protected single-grid/template slot available to the wheel.
        protected = LEGACY_KEY if LEGACY_KEY in keys else self.legacy_match
        if protected in keys and protected not in shown:
            shown[-1:] = [protected]
        # The guarded default must remain visible even if lower-J alternatives
        # fail the guards. Explicit wheel choices may override automatic guards.
        for keep in dict.fromkeys((self.selected, getattr(self, 'default', None))):
            if keep in keys and keep not in shown:
                drop = next((i for i in range(len(shown)-1, -1, -1)
                             if shown[i] not in (protected, self.selected, getattr(self, 'default', None))), None)
                if drop is not None:
                    shown[drop] = keep
        # when P1 produced Cut candidates, keep the best-scoring one in
        # the wheel (replaces the last free slot; never the default/protected).
        cuts = [k for k in keys if k != LEGACY_KEY and self.entries[k][1].get('_p1_candidate')
                and self.entries[k][1].get('auto_cut_count')]
        if cuts and not any(k in shown for k in cuts):
            if len(shown) < self.limit:
                shown.append(cuts[0])
            else:
                drop = next((i for i in range(len(shown)-1, -1, -1)
                             if shown[i] not in (protected, self.selected, getattr(self, 'default', None))), None)
                if drop is not None:
                    shown[drop] = cuts[0]
        shown.sort(key=lambda k: keys.index(k))
        return shown

    def ranked_keys(self):
        keys = self._stable(self._u_order(self._ranked_keys()))
        return keys

    def _u_order(self, shown):
        """the default pick is entry 1, so a preview starts at 1/N; then the R1 band by its band key (rule (a)),
        then the rest by S*, guard-failing entries last. Display order only: the default and the wheel members are unchanged.
        A non-finite S sorts last."""
        from .fill_flow import eligible, metrics, r1_band, band_key
        plan_of = lambda k: self.legacy if k == LEGACY_KEY else self.entries[k][1]
        first = getattr(self, 'default', None) if self.locked else self.selected
        band, _ok = r1_band([plan_of(k) for k in shown], self.quality_floor)
        in_band = {id(x[0]) for x in band}

        def r1(k):
            p = plan_of(k)
            try:
                m = metrics(p); s = float(m['S'])
                bk = band_key(p)
            except Exception:
                s, bk = -math.inf, (math.inf,)
            s = s if math.isfinite(s) else -math.inf
            b = id(p) in in_band
            return (k != first, not b, not eligible(p, self.quality_floor), bk if b else (-s,),
                    (1, ()) if k == LEGACY_KEY else (0, k))
        return sorted(shown, key=r1)

    def _stable(self, shown):
        """lock the selection, not the order. The order is always the
        default first, then S (2.6.1; U with Legacy ranking); a wheel choice is kept by identity, so its index
        may move (1/2 -> 2/3) but the view never jumps to another entry."""
        return list(shown)

    def _ranked_keys(self):
        return self._flow_keys()

    @property
    def plan(self):
        if self.ranking == 'FLOW':
            self._choose_guarded()
        if self.selected == LEGACY_KEY:
            return self.legacy
        return self.entries[self.selected][1] if self.selected in self.entries else None

    def cycle(self, step):
        keys = self.ranked_keys()
        if not keys:
            return None
        index = keys.index(self.selected) if self.selected in keys else 0
        self.selected = keys[(index+step) % len(keys)]
        self.locked = True
        return self.plan

    def label(self):
        keys = self.ranked_keys()
        if self.selected not in keys:
            return ''
        plan = self.plan
        from ..i18n import tr as iface_   # add-on context first
        from .fill_flow import short_label
        label = short_label(plan, self.quality_floor)
        if plan.get('_field_candidate'):
            label += ' | '+iface_('Field')
        if plan.get('_p1_candidate'):
            label += ' | '+iface_('Cut' if plan.get('auto_cut_count') else 'Single patch')
        if plan.get('_cls_candidate'):
            label += ' | '+iface_('Classified')
        return f'{label} | {keys.index(self.selected)+1}/{len(keys)}'
