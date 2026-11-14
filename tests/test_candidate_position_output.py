"""Output-only S33 checks against real synthetic stable tracker transitions."""
import copy
import importlib.util
import inspect
import unittest
from dataclasses import fields, is_dataclass
from unittest.mock import patch

import numpy as np

from pf_tbd import candidate_closed_loop as base
from pf_tbd import history_dispersion_guard as stable
from pf_tbd.core import State

SPEC = importlib.util.find_spec('pf_tbd.candidate_position_output')
if SPEC is not None:
    from pf_tbd import candidate_position_output as variant

CFG = base.Config(K=8, position_sigma=0., velocity_sigma=0., amplitude_sigma=0.)


def branch(n=500, path=0, x=50.):
    particles = np.tile([x, 50., 0., 0., 9.], (n, 1))
    particles[:n//2, 0] -= 1.
    particles[n//2:, 0] += 1.
    weights = np.full(n, 1/n)
    return base.Branch(path, State(particles, weights, np.log(weights), (128, 128), 0))


def dual(count=0):
    return base.Tracker(branch(250), branch(250, 1, 100.), count, 2)


def candidate(x, identifier=0, y=50.):
    return base.Candidate(identifier, (x, y), 1.)


class ModuleExists(unittest.TestCase):
    def test_output_variant_is_available(self):
        self.assertIsNotNone(SPEC, 'S33 output-only wrapper is not implemented')


@unittest.skipIf(SPEC is None, 'S33 output-only wrapper is not implemented')
class OutputVariant(unittest.TestCase):
    def assert_exact(self, left, right):
        self.assertEqual(type(left), type(right))
        if isinstance(left, np.ndarray):
            self.assertEqual(left.dtype, right.dtype)
            self.assertEqual(left.shape, right.shape)
            self.assertEqual(left.tobytes(), right.tobytes())
        elif is_dataclass(left):
            for field in fields(left):
                self.assert_exact(getattr(left, field.name), getattr(right, field.name))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_exact(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_exact(a, b)
        else:
            self.assertEqual(left, right)

    def compare(self, tracker, candidates, action, role=None, frame_id=1, enabled=True, config=CFG):
        original, original_candidates = copy.deepcopy(tracker), copy.deepcopy(candidates)
        old, old_result = stable.step(tracker, candidates, frame_id, config, enabled=enabled)
        new, result = variant.step(tracker, candidates, frame_id, config, enabled=enabled)
        self.assert_exact(tracker, original)
        self.assert_exact(candidates, original_candidates)
        self.assert_exact(old, new)
        self.assertEqual(result.candidate_result.action, action)
        expected = copy.deepcopy(old_result)
        if role is not None:
            center = np.array(old_result.candidate_result.accepted_candidate[role].center_xy)
            self.assertFalse(np.array_equal(center, old_result.output_xy))
            expected.output_xy = center.copy()
            expected.candidate_result.output_xy = center.copy()
            posterior = result.candidate_result.posterior[role].state
            posterior_xy = base._weights(posterior.weights) @ posterior.particles[:, :2]
            self.assert_exact(posterior_xy, old_result.output_xy)
        self.assert_exact(result, expected)
        return new, result

    def test_single_update_uses_accepted_h_center(self):
        self.compare(base.Tracker(branch()), [candidate(53.)], 'single_update', 'H')

    def test_confirm_h_uses_h_even_when_t_has_unique_match(self):
        self.compare(dual(4), [candidate(103., 0), candidate(53., 1)], 'confirm_H', 'H')

    def test_promote_t_uses_predecision_t_accepted_center(self):
        new, result = self.compare(dual(CFG.K-1), [candidate(103.)], 'promote_T', 'T')
        self.assertEqual(result.output_source, 'T')
        self.assertEqual(new.H.path_id, 1)
        self.assertIsNone(new.T)

    def test_unconfirmed_t_keeps_original_h_output(self):
        self.compare(dual(), [candidate(103.)], 'update_T')

    def test_fork_and_reseed_keep_original_h_output(self):
        self.compare(base.Tracker(branch()), [candidate(60.)], 'fork')
        self.compare(dual(4), [candidate(60.)], 'reseed_T')

    def test_no_candidates_and_outside_gate_keep_original_output(self):
        for candidates in ([], [candidate(127., y=127.)]):
            self.compare(base.Tracker(branch()), candidates, 'predict_only')
            self.compare(dual(4), candidates, 'hold')

    def test_ambiguous_h_does_not_use_unique_t_output(self):
        candidates = [candidate(53., 0), candidate(53., 1), candidate(103., 2)]
        _, result = self.compare(dual(4), candidates, 'hold')
        self.assertEqual(result.candidate_result.matches['H'].status, 'ambiguous')
        self.assertEqual(result.candidate_result.matches['T'].status, 'unique')

    def test_ambiguous_t_and_single_h_keep_original_output(self):
        self.compare(dual(4), [candidate(103., 0), candidate(103., 1)], 'hold')
        self.compare(base.Tracker(branch()), [candidate(53., 0), candidate(53., 1)], 'predict_only')

    def test_enabled_preserves_stable_contraction_switch(self):
        for enabled in (False, True):
            self.compare(base.Tracker(branch()), [candidate(53.)], 'single_update', 'H', enabled=enabled)
            tracker = base.Tracker(branch())
            tracker.H.state.particles[:250, 0] = 1.
            tracker.H.state.particles[250:, 0] = 101.
            _, result = self.compare(tracker, [], 'predict_only', enabled=enabled)
            self.assertEqual(result.contraction is not None, enabled)

    def test_multistep_state_and_diagnostics_are_identical(self):
        old = base.Tracker(branch())
        new = copy.deepcopy(old)
        sequence = [[candidate(60.)]] + [[candidate(61.)] for _ in range(CFG.K)]
        sequence += [[], [candidate(62.)], [candidate(62., 0), candidate(62., 1)]]
        actions = []
        for frame_id, candidates in enumerate(sequence, 1):
            old, old_result = stable.step(old, candidates, frame_id, CFG)
            new, result = variant.step(new, candidates, frame_id, CFG)
            self.assert_exact(new, old)
            expected = copy.deepcopy(old_result)
            action = old_result.candidate_result.action
            actions.append(action)
            role = {'single_update': 'H', 'confirm_H': 'H', 'promote_T': 'T'}.get(action)
            if role is not None:
                xy = np.array(old_result.candidate_result.accepted_candidate[role].center_xy)
                expected.output_xy = xy.copy()
                expected.candidate_result.output_xy = xy.copy()
            self.assert_exact(result, expected)
        self.assertEqual(actions[0], 'fork')
        self.assertEqual(actions[1:CFG.K], ['update_T']*(CFG.K-1))
        self.assertEqual(actions[CFG.K], 'promote_T')
        self.assertEqual(actions[-3:], ['predict_only', 'single_update', 'predict_only'])

    def test_one_stable_call_returns_same_tracker_and_adds_no_rng(self):
        tracker, candidates = dual(CFG.K-1), [candidate(103.)]
        with patch.object(np.random, 'default_rng', wraps=np.random.default_rng) as rng:
            stable.step(tracker, candidates, 1, base.Config(K=8))
            expected_rng_calls = rng.call_count
        captured = []
        original_step = stable.step

        def invoke(*args, **kwargs):
            result = original_step(*args, **kwargs)
            captured.append(result)
            return result

        with patch.object(stable, 'step', side_effect=invoke) as call, \
                patch.object(np.random, 'default_rng', wraps=np.random.default_rng) as rng:
            new, result = variant.step(tracker, candidates, 1, base.Config(K=8), enabled=False)
            call.assert_called_once_with(tracker, candidates, 1, base.Config(K=8), enabled=False)
            self.assertEqual(rng.call_count, expected_rng_calls)
        self.assertIs(new, captured[0][0])
        self.assertIs(result, captured[0][1])

    def test_output_arrays_do_not_alias_each_other_or_internal_state(self):
        tracker = base.Tracker(branch())
        new, result = variant.step(tracker, [candidate(53.)], 1, CFG)
        original = copy.deepcopy((tracker, new, result.candidate_result))
        self.assertFalse(np.shares_memory(result.output_xy, result.candidate_result.output_xy))
        result.output_xy[:] = -999.
        self.assert_exact((tracker, new, result.candidate_result), original)

    def test_signature_and_matlab1_coordinates_are_preserved(self):
        self.assertEqual(list(inspect.signature(variant.step).parameters),
                         ['tracker', 'candidates', 'frame_id', 'config', 'enabled'])
        _, result = variant.step(base.Tracker(branch()), [candidate(53.)], 1, CFG)
        np.testing.assert_array_equal(result.output_xy, [53., 50.])


if __name__ == '__main__':
    unittest.main()
