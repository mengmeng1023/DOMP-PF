"""S25 deterministic postdecision cap on retained H position dispersion."""
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from . import candidate_closed_loop as base


@dataclass
class Contraction:
    reason: str
    center: np.ndarray
    scale: float
    max_radius_before: float
    max_radius_after: float
    before_H: base.Branch
    after_H: base.Branch


def contract_history(branch):
    """Cap distance from this branch's weighted mean; preserve all other state.

    Contraction requires float64 particles, as produced by the base closed
    loop, to avoid lossy position assignment or changing other state dtypes.
    Branches already within the cap are exact copies regardless of dtype.
    """
    base._validate_branch(branch)
    out = deepcopy(branch)
    weights = base._weights(out.state.weights)
    positions = out.state.particles[:, :2]
    center = np.array([math.fsum(weights*positions[:, axis]) for axis in (0, 1)])
    delta = positions - center
    before = float(np.hypot(delta[:, 0], delta[:, 1]).max())
    scale = 1. if before <= 25. else 25./before
    if scale < 1.:
        if positions.dtype != np.float64:
            raise ValueError('contraction requires float64 particles')
        positions[:] = center + scale*delta
    delta_after = positions - center
    after = float(np.hypot(delta_after[:, 0], delta_after[:, 1]).max())
    return out, Contraction('contracted' if scale < 1. else 'within_radius',
                            center, scale, before, after, deepcopy(branch), deepcopy(out))


@dataclass
class StepResult:
    output_xy: np.ndarray
    output_source: str
    candidate_result: base.StepResult
    contraction: Contraction | None
    carried: dict[str, base.Branch]


def step(tracker, candidates, frame_id, config=base.Config(), enabled=True):
    """Run old R25 once, then cap retained H without revising this frame's result.

    Forking and T construction finish before contraction. The cap uses the
    retained H's own mean; the original birth gate and candidate snapshots
    remain unchanged. Only the returned tracker carries the capped H forward.
    """
    carried, candidate_result = base.step(tracker, candidates, frame_id, config, branch_radius=25.)
    contraction = None
    if (enabled and carried.H.path_id == tracker.H.path_id
            and candidate_result.matches['H'].status != 'unique'
            and candidate_result.action != 'promote_T'):
        carried.H, contraction = contract_history(carried.H)
    snapshots = {'H': deepcopy(carried.H)}
    if carried.T is not None:
        snapshots['T'] = deepcopy(carried.T)
    return carried, StepResult(candidate_result.output_xy.copy(), candidate_result.output_source,
                               candidate_result, contraction, snapshots)
