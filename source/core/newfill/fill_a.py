# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""The A lane: fill_r2 with band-only plans, pricing and geometry.
fill(world, budget_s=5.0) -> dict(positions, faces, info). Raises fill_r2.FillRefused (BudgetExceeded is a subclass)."""
from . import fill_r2


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
FillRefused = fill_r2.FillRefused


def fill_steps(world, budget_s=5.0, cancel=None):
    return (yield from fill_r2.fill_steps(world, budget_s=budget_s, cancel=cancel))


def fill(*args, **kwargs):
    """fill_steps run to its end (the synchronous form)."""
    return _drive(fill_steps(*args, **kwargs))


WALL_CAP_S = 12.0   # the sliced A is cancelled after this many seconds of its own active time


class _Guard:
    """the sliced run's cancel token: the caller's token, or its active time past the cap."""

    def __init__(self, cancel, cap):
        self.cancel, self.cap, self.active = cancel, cap, 0.0

    def is_set(self):
        return (self.cancel is not None and self.cancel.is_set()) or (self.cap is not None and self.active > self.cap)


def fill_steps_sliced(world, cancel=None, wall_cap=WALL_CAP_S):
    """Sliced A: fill_steps with no time budget, yielding at its slice points; cancelled by `cancel`
    or after `wall_cap` seconds of its own active time. The run's (deadline, cancel) are installed around every
    resume and the previous A context is restored after each step (no other A run sees them).
    Returns fill's dict; raises FillRefused / BudgetExceeded."""
    import time
    from . import apattern as A
    guard = _Guard(cancel, wall_cap)
    gen = fill_steps(world, budget_s=None, cancel=guard)
    mine = None          # this run's (deadline, cancel) between steps

    def ctx():
        return getattr(A._CTX, 'deadline', None), getattr(A._CTX, 'cancel', None)
    try:
        while True:
            outer = ctx()
            if mine is not None:
                A.set_deadline(*mine)
            started = time.perf_counter()
            try:
                next(gen)
            except StopIteration as done:
                return done.value
            finally:
                guard.active += time.perf_counter() - started
                mine = ctx()
                A.set_deadline(*outer)
            yield None
    finally:
        outer = ctx()
        try:
            if mine is not None:
                A.set_deadline(*mine)
            gen.close()   # its own cleanup runs in its own context
        finally:
            A.set_deadline(*outer)
