# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""Bounded planar direction field for additional AUTO Fill candidates.
"""
import numpy as np


def _inside(points, rim):
    p = np.asarray(points)
    a, b = rim, np.roll(rim, -1, axis=0)
    y = p[:, 1, None]
    crossing = (a[:, 1] > y) != (b[:, 1] > y)
    dy = b[:, 1]-a[:, 1]
    x = a[:, 0]+(y-a[:, 1])*(b[:, 0]-a[:, 0])/np.where(dy == 0, 1., dy)
    return np.count_nonzero(crossing & (p[:, 0, None] < x), axis=1) % 2 == 1


def _visible(a, b, rim):
    """Reject graph/query segments crossing any outline edge, including notches."""
    a, b = np.asarray(a), np.asarray(b)
    c, d = rim, np.roll(rim, -1, axis=0)
    def cross(x, y):
        return x[..., 0]*y[..., 1]-x[..., 1]*y[..., 0]
    ab = b-a
    ac, ad = c[None]-a[:, None], d[None]-a[:, None]
    cd = d-c
    ca, cb = a[:, None]-c[None], b[:, None]-c[None]
    first = cross(ab[:, None], ac)*cross(ab[:, None], ad)
    second = cross(cd[None], ca)*cross(cd[None], cb)
    # Strict crossing: touching a boundary endpoint cannot bridge an open notch.
    blocked = np.any((first < -1e-24) & (second < -1e-24), axis=1)
    return ~blocked & _inside((a+b)*.5, rim)


class Guide:
    def __init__(self, world):
        from . import fill
        points = np.asarray(world, float)
        self.available = False
        self.spacing = 0.
        if len(points) < 3 or not np.isfinite(points).all():
            return
        self.centre, self.u, self.v, normal = fill._frame(points)
        lengths = np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1)
        self.spacing = float(np.median(lengths))
        scale = max(float(np.ptp(points, axis=0).max()), 1e-30)
        if self.spacing <= scale*1e-12 or np.max(np.abs((points-self.centre)@normal)) > scale*1e-6:
            return
        self.rim = self.plane(points)
        area = np.sum(self.rim[:, 0]*np.roll(self.rim[:, 1], -1)-self.rim[:, 1]*np.roll(self.rim[:, 0], -1))
        if area < 0:
            self.v = -self.v
            self.rim = self.plane(points)
        lo, hi = self.rim.min(0), self.rim.max(0)
        # At most 48 x 48 nodes; use half the outline spacing where affordable.
        step = max(self.spacing*.5, float((hi-lo).max())/47.)
        dims = np.maximum(2, np.minimum(48, np.ceil((hi-lo)/step).astype(int)+1))
        xs, ys = [np.linspace(lo[i], hi[i], dims[i]) for i in range(2)]
        grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        mask = _inside(grid, self.rim)
        nodes = grid[mask]
        if not len(nodes):
            return
        ids = np.full(len(grid), -1, int)
        ids[mask] = np.arange(len(nodes))
        ids = ids.reshape(len(ys), len(xs))
        links = []
        for left, right in ((ids[:, :-1], ids[:, 1:]), (ids[:-1], ids[1:])):
            good = (left >= 0) & (right >= 0)
            aa, bb = left[good], right[good]
            good = _visible(nodes[aa], nodes[bb], self.rim)
            links.extend(zip(aa[good].tolist(), bb[good].tolist()))
        self.nodes = nodes
        # Closest outline edge supplies its fourfold tangent and soft constraint.
        edge = np.roll(self.rim, -1, axis=0)-self.rim
        relative = nodes[:, None]-self.rim[None]
        t = np.clip(np.sum(relative*edge[None], axis=2)/np.maximum(np.sum(edge*edge, axis=1), 1e-30), 0., 1.)
        distance = np.linalg.norm(relative-t[..., None]*edge[None], axis=2)
        nearest = distance.argmin(1)
        theta = np.arctan2(edge[:, 1], edge[:, 0])
        target = np.exp(4j*theta[nearest])
        weights = np.exp(-distance.min(1)/max(self.spacing*.5, 1e-30))
        if links:
            pair = np.asarray(links, int)
            a, b = np.concatenate((pair[:, 0], pair[:, 1])), np.concatenate((pair[:, 1], pair[:, 0]))
        else:
            a = b = np.array([], int)
        degree = np.bincount(a, minlength=len(nodes))
        field = weights*target
        for _ in range(40):
            real = np.bincount(a, weights=field[b].real, minlength=len(nodes))
            imag = np.bincount(a, weights=field[b].imag, minlength=len(nodes))
            field = (real+1j*imag+2.*weights*target)/(degree+2.*weights)
        self.field = field
        self.available = True

    def plane(self, points):
        relative = np.asarray(points)-self.centre
        return np.column_stack((relative@self.u, relative@self.v))

    def sample(self, points):
        values = np.zeros(len(points), complex)
        # Visibility prevents sampling through concavities, including narrow slots.
        for start in range(0, len(points), 128):
            p = points[start:start+128]
            distance = np.sum((p[:, None]-self.nodes[None])**2, axis=2)
            order = np.argsort(distance, axis=1, kind='stable')[:, :8]
            pending = np.ones(len(p), bool)
            for k in range(order.shape[1]):
                ids = np.flatnonzero(pending)
                if not len(ids):
                    break
                nodes = order[ids, k]
                valid = _visible(p[ids], self.nodes[nodes], self.rim)
                chosen = ids[valid]
                values[start+chosen] = self.field[nodes[valid]]
                pending[chosen] = False
        return values

