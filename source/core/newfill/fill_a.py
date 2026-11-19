# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""The A part of the second search: fill_r2 with band-only plans, pricing and geometry.
fill(world, budget_s=5.0) -> dict(positions, faces, info). Raises fill_r2.FillRefused (BudgetExceeded is a subclass)."""
from . import fill_r2


def _drive(gen):
    """run a generator to its end; its value."""
    try:
        while True:
            next(gen)
    except StopIteration as done:
        return done.value


FillRefused = fill_r2.FillRefused


def fill_steps(world, budget_s=5.0, cancel=None):
    return (yield from fill_r2.fill_steps(world, budget_s=budget_s, cancel=cancel))


def fill(*args, **kwargs):
    """fill_steps run to its end (the synchronous form)."""
    return _drive(fill_steps(*args, **kwargs))

