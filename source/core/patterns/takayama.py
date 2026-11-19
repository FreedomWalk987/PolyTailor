# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Figure-4 quad skeletons with explicit logical corners and zero-safe padding.

Independent implementation of Takayama, Panozzo, Sorkine-Hornung (2014),
Sections 2.1-2.3.
"""
import itertools
import math
from fractions import Fraction
from functools import lru_cache

import numpy as np


# Boundary vertices are listed counter-clockwise, starting on side zero.
# Each entry is (quads, logical boundary corners, Figure-4 flow columns).
SKELETONS = {
    (3,0): ([(0,1,2,3)], (0,2,3), ()),
    (3,1): ([(0,1,6,5),(1,2,3,6),(3,4,5,6)], (0,4,5), ((2,0,0),)),
    (4,0): ([(0,1,2,3)], (0,1,2,3), ()),
    (4,1): ([(0,1,6,5),(1,2,3,6),(3,4,5,6)], (0,2,4,5), ((1,1,0,0),)),
    (4,2): ([(0,1,4,5),(1,2,3,4)], (0,3,4,5), ((1,1,0,0),(1,0,0,1))),
    (4,3): ([(0,1,6,5),(1,2,7,6),(2,3,4,7),(6,7,4,5)],
            (0,3,4,5), ((2,0,0,0),)),
    (4,4): ([(0,1,8,7),(1,2,9,8),(2,3,10,9),(3,4,5,10),(10,5,6,9),(8,9,6,7)],
            (0,4,6,7), ((2,0,0,0),(1,1,0,0))),
    (5,0): ([(0,1,4,5),(1,2,3,4)], (0,2,3,4,5), ()),
    (5,1): ([(0,1,2,3),(0,3,4,5)], (0,2,3,4,5), ((1,1,0,0,0),)),
    (5,2): ([(0,1,9,8),(1,2,3,9),(3,4,8,9),(4,5,6,8),(6,7,0,8)],
            (0,4,5,6,7), ((2,0,0,0,0),)),
    (5,3): ([(0,1,10,9),(1,2,11,10),(2,3,12,11),(3,4,13,12),
             (4,5,6,13),(13,6,7,12),(9,10,11,8),(11,12,7,8)],
            (0,5,7,8,9), ((2,0,0,0,0),(1,1,0,0,0))),
    (6,0): ([(0,1,2,5),(2,3,4,5)], (0,1,2,3,4,5), ((1,0,0,1,0,0),)),
    (6,1): ([(0,1,9,8),(1,2,3,9),(9,3,4,8),(8,4,5,6),(0,8,6,7)],
            (0,2,4,5,6,7), ((1,1,0,0,0,0),)),
    (6,2): ([(0,1,10,8),(1,2,11,10),(2,3,9,11),(8,10,11,9),
             (8,9,5,6),(0,8,6,7),(3,4,5,9)],
            (0,3,4,5,6,7), ((2,0,0,0,0,0),(1,0,0,1,0,0))),
    (6,3): ([(0,1,12,9),(9,12,10,8),(12,13,11,10),(11,13,6,7),
             (1,2,13,12),(2,3,14,13),(3,4,5,14),(13,14,5,6),(8,10,11,7)],
            (0,4,6,7,8,9), ((2,0,0,0,0,0),(1,1,0,0,0,0))),
}


def flow_values(columns, residual):
    """Solve the independent zero/one/two Figure-4 flow columns exactly."""
    if not columns:return () if not any(residual) else None
    if len(columns)==1:
        a=columns[0];i=next(i for i,v in enumerate(a) if v)
        if residual[i]%a[i]:return None
        x=residual[i]//a[i]
        return (x,) if x>=0 and all(a[j]*x==residual[j] for j in range(len(a))) else None
    a,b=columns
    for i in range(len(a)):
        for j in range(i+1,len(a)):
            determinant=a[i]*b[j]-a[j]*b[i]
            if not determinant:continue
            xn=residual[i]*b[j]-residual[j]*b[i]
            yn=a[i]*residual[j]-a[j]*residual[i]
            if xn%determinant or yn%determinant:return None
            x,y=xn//determinant,yn//determinant
            return (x,y) if min(x,y)>=0 and all(a[k]*x+b[k]*y==residual[k] for k in range(len(a))) else None
    raise ValueError('Dependent Figure-4 flow columns')


def reduce_counts(counts):
    """Constructive trimming certificate; inverse padding is explicitly generated."""
    values=list(map(int,counts));padding=[0]*len(values);n=len(values)
    changed=True
    while changed:
        changed=False
        for side in range(n):
            amount=min(values[(side-1)%n],values[(side+1)%n])-1
            if amount>0:
                values[(side-1)%n]-=amount;values[(side+1)%n]-=amount
                padding[side]+=amount;changed=True
    return tuple(values),tuple(padding)


class Family:
    takayama=True

    def __init__(self,Pattern,n,index,mirror=False):
        quads,corners,flows=SKELETONS[(n,index)]
        if mirror:
            # Reflect the side labels as well as face orientation. Keep side 0
            # first so solver parameters retain the documented matrix convention.
            quads=[q[::-1] for q in quads]
            corners=(corners[1],corners[0])+tuple(reversed(corners[2:]))
            flows=tuple(tuple([c[0]]+list(reversed(c[1:]))) for c in flows)
        self.name=f'takayama_{n}_{index}'+('~' if mirror else '')
        self.base=Pattern(self.name+'_base',quads,corners=corners)
        self.quads=self.base.quads;self.corner=self.base.corner;self.loops=self.base.loops
        self.nv=self.base.nv;self.nchords=self.base.nchords
        self.n=n;self.index=index;self.flow_columns=flows
        self.base_counts=tuple(len(side)-1 for side in self.loops[0])
        a=np.asarray([np.bincount(self.base.side_chords(side),minlength=self.nchords)
                      for side in self.loops[0]],dtype=np.int64)
        self.flow_chords=[]
        for flow in flows:
            choices=[j for j in range(self.nchords) if tuple(a[:,j])==tuple(flow)]
            if not choices:raise ValueError((self.name,'Missing flow chord',flow,a.tolist()))
            self.flow_chords.append(choices[0])
        padding=np.zeros((n,n),dtype=np.int64)
        for i in range(n):padding[(i-1)%n,i]+=1;padding[(i+1)%n,i]+=1
        self.parameter_matrix=np.column_stack((padding,np.asarray(flows,dtype=np.int64).T)) if flows else padding
        self.irregular=self.base.irregular
        self._prepare_exact()

    def _prepare_exact(self):
        a=self.parameter_matrix;n,m=a.shape
        rows=[[Fraction(int(v)) for v in row]+[Fraction(int(i==j)) for j in range(n)] for i,row in enumerate(a)]
        pivots=[];r=0
        for c in range(m):
            pivot=next((i for i in range(r,n) if rows[i][c]),None)
            if pivot is None:continue
            rows[r],rows[pivot]=rows[pivot],rows[r]
            divisor=rows[r][c];rows[r]=[v/divisor for v in rows[r]]
            for i in range(n):
                if i!=r and rows[i][c]:
                    factor=rows[i][c];rows[i]=[u-factor*v for u,v in zip(rows[i],rows[r])]
            pivots.append(c);r+=1
            if r==n:break
        free=[i for i in range(m) if i not in pivots]
        den=[];coeff=[];constant=[]
        for row in rows:
            d=math.lcm(*(v.denominator for v in row));den.append(d)
            coeff.append([int(row[j]*d) for j in free]);constant.append([int(v*d) for v in row[m:]])
        self._pivots=pivots;self._free=free;self._rank=r
        self._den=np.asarray(den,np.int64);self._coeff=np.asarray(coeff,np.int64).reshape(n,len(free))
        self._constant=np.asarray(constant,np.int64)

    def solve_steps(self,counts):
        if len(counts)!=self.n:return None
        b=np.asarray(counts,np.int64)-self.base_counts
        if np.any(b<0):return None
        reduced,padding=reduce_counts(counts)
        flow=flow_values(self.flow_columns,tuple(c-d for c,d in zip(reduced,self.base_counts)))
        if flow is not None:
            return tuple(padding)+flow
        constants=self._constant@b;r=self._rank
        if np.any(constants[r:]):return None
        ranges=[]
        for col in self._free:
            indices=np.flatnonzero(self.parameter_matrix[:,col])
            hi=min(int(b[i])//int(self.parameter_matrix[i,col]) for i in indices)
            ranges.append(range(hi+1))
        combinations=itertools.product(*ranges)
        while True:
            batch=list(itertools.islice(combinations,512))
            if not batch:break
            values=np.asarray(batch,np.int64).reshape(len(batch),len(self._free))
            nums=constants[:r]-values@self._coeff[:r].T
            valid=np.all((nums>=0)&(nums%self._den[:r]==0),axis=1)
            if np.any(valid):
                first=int(np.flatnonzero(valid)[0]);x=np.zeros(self.parameter_matrix.shape[1],np.int64)
                x[self._free]=values[first];x[self._pivots]=nums[first]//self._den[:r]
                return tuple(map(int,x))
            yield None
        return None

    def policy_parameters_steps(self, counts):
        if len(counts) != self.n:
            return []
        b = np.asarray(counts, np.int64) - self.base_counts
        if np.any(b < 0):
            return []
        constants = self._constant @ b
        r = self._rank
        if np.any(constants[r:]):
            return []
        ranges = []
        for col in self._free:
            indices = np.flatnonzero(self.parameter_matrix[:, col])
            hi = min(int(b[i]) // int(self.parameter_matrix[i, col]) for i in indices)
            ranges.append(range(hi + 1))
        combinations = itertools.product(*ranges)
        solutions = []
        from . import geom
        while True:
            geom.check_time()
            batch = list(itertools.islice(combinations, 512))
            if not batch:
                break
            values = np.asarray(batch, np.int64).reshape(len(batch), len(self._free))
            nums = constants[:r] - values @ self._coeff[:r].T
            valid = np.all((nums >= 0) & (nums % self._den[:r] == 0), axis=1)
            for index in np.flatnonzero(valid):
                x = np.zeros(self.parameter_matrix.shape[1], np.int64)
                x[self._free] = values[index]
                x[self._pivots] = nums[index] // self._den[:r]
                solutions.append(tuple(map(int, x)))
            yield None
        return sorted(solutions, key=lambda x: (-sum(x[:self.n]), x))


@lru_cache(maxsize=1)
def library():
    from .chords import Pattern
    return tuple(Family(Pattern,n,index,mirror) for n,index in SKELETONS for mirror in (False,True))
