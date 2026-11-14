"""S33 candidate-center reporting over the unchanged stable closed loop.

Coordinates remain continuous MATLAB1. Particle observation updates, motion,
branch decisions and history dispersion maintenance all belong to the stable
implementation; this module changes only the selected current-frame output.
"""
import numpy as np

from . import candidate_closed_loop as base
from . import history_dispersion_guard as stable


def step(tracker, candidates, frame_id, config=base.Config(), enabled=True):
    """Run the stable step once and report its accepted output-path candidate.

    Unconfirmed T updates and all unresolved decisions retain stable output.
    ``enabled`` retains its stable meaning: enable history dispersion control.
    The original posterior mean remains available from ``posterior[role]``
    in ``candidate_result``; all diagnostics and carried state are unchanged.
    """
    carried, result = stable.step(tracker, candidates, frame_id, config, enabled=enabled)
    decision = result.candidate_result
    role = {'single_update': 'H', 'confirm_H': 'H', 'promote_T': 'T'}.get(decision.action)
    if role is not None:
        center = np.array(decision.accepted_candidate[role].center_xy, dtype=np.float64)
        result.output_xy = center
        decision.output_xy = center.copy()
    return carried, result
