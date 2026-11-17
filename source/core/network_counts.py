# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Small exact interval assignment: eliminate first, search free variables only.

Arithmetic is rational/integer throughout. Exhaustion proves infeasibility only
inside the supplied model and finite bounds; a node/time stop is search_limit.
"""
from fractions import Fraction
from math import gcd, lcm
from functools import reduce
from .patterns import geom


def _reduced_steps(variables, equations, bounds, targets=None, max_nodes=50000,
                seconds=.05):
    started = geom.clock().perf_counter()
    names = list(variables)
    if len(set(names)) != len(names):
        raise ValueError('Duplicate variable')
    index = {name: i for i, name in enumerate(names)}
    limits = [tuple(map(int, bounds[name])) for name in names]
    target = [int((targets or {}).get(name, lo)) for name, (lo, hi) in zip(names, limits)]
    stats = dict(variables=len(names), equations=len(equations), nodes=0,
                 rank=0, free_dimensions=0)

    def result(status, reason=None, sources=(), values=None):
        return dict(status=status, reason=reason, conflicts=sorted(sources),
                    values=None if values is None else dict(zip(names, values)),
                    elapsed_ms=(geom.clock().perf_counter()-started)*1000, **stats)

    for name, (lo, hi) in zip(names, limits):
        if lo > hi:
            return result('proven_infeasible_for_this_model', 'Empty interval bound', [name])
    rows = []
    for eq in equations:
        row = [Fraction(0) for _ in range(len(names)+1)]
        for name, value in eq['coefficients'].items():
            row[index[name]] += Fraction(value)
        row[-1] = Fraction(eq['rhs'])
        rows.append([row, {eq.get('label', 'unlabelled equation')}])
    for i, (lo, hi) in enumerate(limits):
        if lo == hi:
            row = [Fraction(int(j == i)) for j in range(len(names))] + [Fraction(lo)]
            rows.append([row, {'frozen:' + names[i]}])
    pivots = []
    r = 0
    # Prefer frozen variables; this avoids introducing needless template freedoms.
    columns = sorted(range(len(names)), key=lambda i: (limits[i][0] != limits[i][1], i))
    for c in columns:
        k = next((k for k in range(r, len(rows)) if rows[k][0][c]), None)
        if k is None:
            continue
        rows[r], rows[k] = rows[k], rows[r]
        divisor = rows[r][0][c]
        rows[r][0] = [v/divisor for v in rows[r][0]]
        for j in range(len(rows)):
            if j != r and rows[j][0][c]:
                factor = rows[j][0][c]
                rows[j][0] = [a-factor*b for a, b in zip(rows[j][0], rows[r][0])]
                rows[j][1].update(rows[r][1])
        pivots.append(c)
        r += 1
        if r % 8 == 0:
            yield None
    stats['rank'] = r
    free = [i for i in range(len(names)) if i not in pivots]
    stats['free_dimensions'] = len(free)
    for row, source in rows[r:]:
        if not any(row[:-1]) and row[-1]:
            return result('proven_infeasible_for_this_model', 'Inconsistent exact equations', source)
    reduced = []
    for (row, source), pivot in zip(rows[:r], pivots):
        denominator = lcm(*(v.denominator for v in row))
        coeff = {i: int(-row[i]*denominator) for i in free if row[i]}
        reduced.append((pivot, denominator, int(row[-1]*denominator), coeff, source))
    free.sort(key=lambda i: (limits[i][1]-limits[i][0], -sum(i in v[3] for v in reduced), i))
    assigned = {}
    found = None
    stopped = False

    def possible():
        for pivot, den, constant, coeff, source in reduced:
            value = constant + sum(c*assigned[i] for i, c in coeff.items() if i in assigned)
            remaining = [(i, c) for i, c in coeff.items() if i not in assigned]
            low = value + sum(min(c*limits[i][0], c*limits[i][1]) for i, c in remaining)
            high = value + sum(max(c*limits[i][0], c*limits[i][1]) for i, c in remaining)
            if high < den*limits[pivot][0] or low > den*limits[pivot][1]:
                return False
            divisor = reduce(gcd, (abs(c) for _, c in remaining), den)
            if value % divisor:
                return False
        return True

    def visit(depth):
        nonlocal found, stopped
        if stats['nodes'] >= max_nodes or geom.clock().perf_counter()-started >= seconds:
            stopped = True
            return
        stats['nodes'] += 1
        if not possible():
            return
        if depth == len(free):
            values = [assigned.get(i, 0) for i in range(len(names))]
            for pivot, den, constant, coeff, source in reduced:
                value = constant + sum(c*assigned[i] for i, c in coeff.items())
                if value % den:
                    return
                values[pivot] = value//den
            if all(lo <= v <= hi for v, (lo, hi) in zip(values, limits)):
                assert all(sum(int(c)*values[index[k]] for k, c in eq['coefficients'].items()) == int(eq['rhs']) for eq in equations)
                found = values
            return
        i = free[depth]
        lo, hi = limits[i]
        for value in sorted(range(lo, hi+1), key=lambda v: (abs(v-target[i]), v)):
            assigned[i] = value
            yield from visit(depth+1)
            if found is not None or stopped:
                break
            if stats['nodes'] % 128 == 0:
                yield None
        assigned.pop(i, None)

    # No search is needed for a fully determined system; do not turn elimination
    # time alone into a false search failure.
    if not free:
        values = [0]*len(names)
        for pivot, den, constant, coeff, source in reduced:
            if constant % den or not limits[pivot][0]*den <= constant <= limits[pivot][1]*den:
                return result('proven_infeasible_for_this_model', 'Divisibility or bound conflict', source | {names[pivot]})
            values[pivot] = constant//den
        return result('feasible', values=values)
    yield from visit(0)
    if found is not None:
        return result('feasible', values=found)
    if stopped:
        return result('search_limit', 'Node or elapsed budget exhausted; no infeasibility claim')
    return result('proven_infeasible_for_this_model', 'No integer point within supplied bounds',
                  {s for _, source in rows for s in source})


def solve_steps(variables,equations,bounds,targets=None,max_nodes=50000,seconds=.05):
    """Substitute frozen counts before building the rational elimination matrix."""
    begin=geom.clock().perf_counter();names=list(variables)
    if len(set(names))!=len(names):raise ValueError('Duplicate variable')
    for k in names:
        if any(int(v)!=v for v in bounds[k]):raise ValueError('Integer bounds required: '+k)
    for eq in equations:
        if int(eq['rhs'])!=eq['rhs'] or any(int(v)!=v for v in eq['coefficients'].values()):raise ValueError('Integer equations required')
        if any(k not in bounds for k in eq['coefficients']):raise ValueError('Unknown count variable')
    fixed={k:int(bounds[k][0]) for k in names if bounds[k][0]==bounds[k][1]}
    free=[k for k in names if k not in fixed]
    reduced=[];sources={}
    for eq in equations:
        label=eq.get('label','unlabelled equation')
        sources[label]={k for k,c in eq['coefficients'].items() if c and k in fixed}
        reduced.append(dict(coefficients={k:c for k,c in eq['coefficients'].items() if k not in fixed},
                            rhs=eq['rhs']-sum(c*fixed[k] for k,c in eq['coefficients'].items() if k in fixed),label=label))
    result=yield from _reduced_steps(free,reduced,{k:bounds[k] for k in free},targets,max_nodes,seconds)
    result['substituted_fixed']=len(fixed);result['variables_before_substitution']=len(names)
    if result['values'] is not None:
        result['values']={k:(fixed[k] if k in fixed else result['values'][k]) for k in names}
        if not all(sum(c*result['values'][k] for k,c in eq['coefficients'].items())==eq['rhs'] for eq in equations):
            raise AssertionError('Invalid exact count certificate')
    else:
        implicated=set(result['conflicts'])
        for label in result['conflicts']:implicated.update('frozen:'+k for k in sources.get(label,()))
        result['conflicts']=sorted(implicated)
    result['elapsed_ms']=(geom.clock().perf_counter()-begin)*1000
    return result


