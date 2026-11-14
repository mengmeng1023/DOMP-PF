"""Synthetic in-memory tests for the S25 postdecision H radius cap."""
import copy
import ast
import importlib.util
import inspect
import math
import unittest
from dataclasses import fields, is_dataclass
from unittest.mock import patch

import numpy as np

from pf_tbd import candidate_closed_loop as base
from pf_tbd.core import State

CFG = base.Config(position_sigma=0., velocity_sigma=0., amplitude_sigma=0.)


SPEC = importlib.util.find_spec('pf_tbd.history_dispersion_guard')
if SPEC is not None:
    from pf_tbd import history_dispersion_guard as guard


def branch(n=500, path=0):
    p = np.tile([1., 50., 0., 0., 9.], (n, 1))
    p[n//2:, 0] = 101.
    p[:, 4] = np.arange(n)
    return base.Branch(path, State(p, np.full(n, 1/n), np.full(n, -np.log(n)), (128, 128), 4))


def center(b):
    w = b.state.weights / math.fsum(b.state.weights)
    return np.array([math.fsum(w*b.state.particles[:, k]) for k in (0, 1)])


class ExactAssertions(unittest.TestCase):
    def assert_exact(self, a, b):
        self.assertEqual(type(a), type(b))
        if isinstance(a, np.ndarray):
            self.assertEqual(a.dtype, b.dtype)
            self.assertEqual(a.shape, b.shape)
            self.assertEqual(a.tobytes(), b.tobytes())
        elif is_dataclass(a):
            for field in fields(a):
                self.assert_exact(getattr(a, field.name), getattr(b, field.name))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_exact(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_exact(x, y)
        else:
            self.assertEqual(a, b)

    def assert_branch_equal(self, a, b):
        self.assertEqual(a.path_id, b.path_id)
        self.assertEqual(a.state.frame_size, b.state.frame_size)
        self.assertEqual(a.state.step_index, b.state.step_index)
        for attr in ('particles', 'weights', 'log_weights'):
            x, y = getattr(a.state, attr), getattr(b.state, attr)
            self.assertEqual(x.dtype, y.dtype)
            self.assertEqual(x.shape, y.shape)
            self.assertEqual(x.tobytes(), y.tobytes())


class HelperExists(unittest.TestCase):
    def test_helper_is_available(self):
        self.assertIsNotNone(SPEC, 'history dispersion helper is not implemented')


@unittest.skipIf(SPEC is None, 'helper not implemented')
class Contraction(ExactAssertions):
    def test_weighted_center_and_global_scale(self):
        b = branch(2)
        b.state.weights = np.array([.25, .75])
        b.state.log_weights = np.log(b.state.weights) + 9
        out, d = guard.contract_history(b)
        np.testing.assert_array_equal(d.center, [76., 50.])
        self.assertEqual(d.reason, 'contracted')
        self.assertEqual(d.scale, 1/3)
        self.assertEqual(d.max_radius_before, 75.)
        self.assertLessEqual(d.max_radius_after, 25. + 1e-9)
        np.testing.assert_allclose(out.state.particles[:, 0], [51., 76. + 25/3], atol=1e-12)
        np.testing.assert_allclose(center(out), center(b), atol=1e-9, rtol=0)

    def test_zero_inside_and_exact_boundary_are_bytewise_noops(self):
        for radius in (0., 24., 25.):
            b = branch(4)
            b.state.particles[:, :2] = np.array([[radius, 0], [-radius, 0], [0, radius], [0, -radius]]) + 50
            out, d = guard.contract_history(b)
            self.assert_branch_equal(out, b)
            self.assertEqual(d.scale, 1.)
            self.assertEqual(d.reason, 'within_radius')
            self.assertEqual(d.max_radius_before, radius)
            self.assertEqual(d.max_radius_after, radius)

    def test_four_directions_and_500_250_counts(self):
        for n in (500, 250):
            b = branch(n)
            b.state.particles[:, :2] = np.resize([[100., 0.], [-100., 0.], [0., 100.], [0., -100.]], (n, 2)) + 50
            out, d = guard.contract_history(b)
            self.assertEqual(len(out.state.weights), n)
            np.testing.assert_allclose(center(out), center(b), atol=1e-9, rtol=0)
            self.assertLessEqual(np.linalg.norm(out.state.particles[:, :2]-center(out), axis=1).max(), 25.+1e-9)
            np.testing.assert_allclose(out.state.particles[:, :2], d.center + d.scale*(b.state.particles[:, :2]-d.center), atol=1e-12)

    def test_outside_image_coordinates_are_not_clipped(self):
        b = branch(2)
        b.state.particles[:, :2] = [[-99., -49.], [101., 51.]]
        out, _ = guard.contract_history(b)
        self.assertLess(out.state.particles[0, 0], 1.)
        self.assertLess(out.state.particles[0, 1], 1.)
        np.testing.assert_allclose(center(out), [1., 1.], atol=1e-9)

    def test_valid_near_edge_weighted_particles_remain_in_image(self):
        b = branch(2)
        b.state.particles[:, 1] = 1.
        b.state.weights = np.array([.99, .01])
        b.state.log_weights = np.log(b.state.weights)
        out, d = guard.contract_history(b)
        self.assertEqual(d.reason, 'contracted')
        np.testing.assert_allclose(center(b), [2., 1.], atol=1e-9, rtol=0)
        np.testing.assert_allclose(center(out), center(b), atol=1e-9, rtol=0)
        self.assertTrue(np.all(out.state.particles[:, :2] >= 1.))
        self.assertTrue(np.all(out.state.particles[:, :2] <= 128.))
        self.assertLessEqual(d.max_radius_after, 25.+1e-9)

    def test_only_positions_change_and_normalization_is_internal(self):
        b = branch(2, path=7)
        b.state.weights = np.array([1., 3.])
        b.state.log_weights = np.log(b.state.weights)
        b.state.particles[:, 2:4] = [[2., -3.], [5., 8.]]
        out, _ = guard.contract_history(b)
        self.assertEqual(out.path_id, b.path_id)
        self.assertEqual(out.state.step_index, b.state.step_index)
        self.assertEqual(out.state.frame_size, b.state.frame_size)
        for attr in ('weights', 'log_weights'):
            self.assertEqual(getattr(out.state, attr).tobytes(), getattr(b.state, attr).tobytes())
        self.assertEqual(out.state.particles[:, 2:].tobytes(), b.state.particles[:, 2:].tobytes())

    def test_inputs_and_diagnostic_snapshots_do_not_alias(self):
        b = branch()
        original = copy.deepcopy(b)
        out, d = guard.contract_history(b)
        self.assert_branch_equal(b, original)
        self.assert_branch_equal(d.before_H, original)
        self.assert_branch_equal(d.after_H, out)
        arrays = [x.state.particles for x in (b, out, d.before_H, d.after_H)]
        for i, x in enumerate(arrays):
            for y in arrays[i+1:]:
                self.assertFalse(np.shares_memory(x, y))

    def test_existing_validation_rejects_invalid_states_and_weights(self):
        cases = []
        for attr, value in [('particles', np.zeros((2, 4))), ('particles', np.full((2, 5), np.nan)),
                            ('weights', np.array([-1., 2.])), ('weights', np.zeros(2)),
                            ('weights', np.array([1., np.inf])), ('weights', np.ones(3)),
                            ('log_weights', np.array([0., 0.])), ('log_weights', np.array([np.nan, 0.])),
                            ('frame_size', (0, 128)), ('step_index', -1)]:
            b = branch(2)
            b.state.weights = np.array([.25, .75])
            b.state.log_weights = np.log(b.state.weights)
            setattr(b.state, attr, value)
            cases.append(b)
        b = branch(2)
        b.path_id = -1
        cases.append(b)
        for b in cases:
            with self.subTest(state=b.state), self.assertRaises(ValueError):
                guard.contract_history(b)

    def test_repeated_cap_adds_no_shrink(self):
        once, _ = guard.contract_history(branch())
        twice, d = guard.contract_history(once)
        self.assert_branch_equal(once, twice)
        self.assertEqual(d.scale, 1.)

    def test_contraction_rejects_lossy_particle_dtypes_without_mutation(self):
        for dtype in (np.int64, np.float32):
            b = branch(2)
            b.state.weights = np.array([.25, .75])
            b.state.log_weights = np.log(b.state.weights)
            b.state.particles = b.state.particles.astype(dtype)
            original = copy.deepcopy(b)
            with self.assertRaisesRegex(ValueError, 'float64'):
                guard.contract_history(b)
            self.assert_branch_equal(b, original)

    def test_noop_preserves_integer_and_low_precision_particle_bytes(self):
        for dtype in (np.int64, np.float32):
            b = branch(2)
            b.state.particles[:, 0] = [25., 75.]
            b.state.particles = b.state.particles.astype(dtype)
            out, d = guard.contract_history(b)
            self.assert_branch_equal(out, b)
            self.assertEqual(d.scale, 1.)


class WrapperExists(unittest.TestCase):
    def test_wrapper_is_available(self):
        self.assertTrue(SPEC is not None and hasattr(guard, 'step'), 'S25 wrapper is not implemented')


@unittest.skipIf(SPEC is None or not hasattr(guard, 'step'), 'wrapper not implemented')
class Wrapper(ExactAssertions):
    def compare(self, tracker, candidates, enabled=True):
        original = copy.deepcopy(tracker)
        old, oldr = base.step(tracker, candidates, 1, CFG, branch_radius=25.)
        new, r = guard.step(tracker, candidates, 1, CFG, enabled=enabled)
        self.assert_exact(tracker, original)
        self.assert_exact(r.candidate_result, oldr)
        self.assert_exact(r.output_xy, oldr.output_xy)
        self.assertEqual(r.output_source, oldr.output_source)
        self.assertEqual(new.C_T, old.C_T)
        self.assertEqual(new.next_path_id, old.next_path_id)
        self.assert_exact(new.T, old.T)
        self.assert_exact(r.carried['H'], new.H)
        if new.T is not None:
            self.assert_exact(r.carried['T'], new.T)
        return old, oldr, new, r

    def dual(self, count=0):
        t = branch(250, path=1)
        t.state.particles[:, :2] = [100., 100.]
        return base.Tracker(branch(250), t, C_T=count, next_path_id=2)

    def test_disabled_is_exact_old_r25_passthrough(self):
        for tracker, cs in ((base.Tracker(branch()), []),
                            (base.Tracker(branch()), [base.Candidate(0, (51., 50.), 1.)]),
                            (self.dual(), [base.Candidate(0, (100., 100.), 1.)])):
            old, _, new, r = self.compare(tracker, cs, enabled=False)
            self.assert_exact(new, old)
            self.assertIsNone(r.contraction)

    def test_unique_h_skips_single_update_and_confirm_h(self):
        for tracker in (base.Tracker(branch()), self.dual()):
            old, oldr, new, r = self.compare(tracker, [base.Candidate(0, (1., 50.), 1.)])
            self.assertEqual(oldr.matches['H'].status, 'unique')
            self.assertIn(oldr.action, ('single_update', 'confirm_H'))
            self.assert_exact(new, old)
            self.assertIsNone(r.contraction)

    def test_t_promotion_skips_contraction(self):
        old, oldr, new, r = self.compare(self.dual(CFG.K-1), [base.Candidate(0, (100., 100.), 1.)])
        self.assertEqual(oldr.action, 'promote_T')
        self.assertEqual(oldr.matches['H'].status, 'none')
        self.assert_exact(new, old)
        self.assertIsNone(r.contraction)

    def test_replaced_history_identity_skips_contraction(self):
        tracker = base.Tracker(branch())
        replacement, candidate_result = base.step(tracker, [], 1, CFG, branch_radius=25.)
        replacement.H.path_id = tracker.next_path_id
        replacement.next_path_id += 1
        candidate_result.action = 'replace_H'
        candidate_result.carried['H'] = copy.deepcopy(replacement.H)
        expected = copy.deepcopy(replacement)
        expected_result = copy.deepcopy(candidate_result)
        # Preserve a complete real result while simulating a future replacement action.
        with patch.object(base, 'step', return_value=(replacement, candidate_result)):
            new, r = guard.step(tracker, [], 1, CFG)
        self.assertTrue(r.contraction is None, 'a replacement H must not be contracted')
        self.assert_exact(new, expected)
        self.assert_exact(r.candidate_result, expected_result)
        self.assert_exact(r.carried['H'], expected.H)

    def test_none_ambiguous_empty_retained_history_contracts(self):
        cases = [([], 'none'), ([base.Candidate(0, (127., 127.), 1.)], 'none'),
                 ([base.Candidate(0, (1., 50.), 1.), base.Candidate(1, (1., 50.), 1.)], 'ambiguous')]
        for tracker in (base.Tracker(branch()), self.dual()):
            for cs, status in cases:
                old, oldr, new, r = self.compare(tracker, cs)
                self.assertEqual(oldr.matches['H'].status, status)
                self.assertEqual(r.contraction.reason, 'contracted')
                self.assertEqual(new.H.path_id, old.H.path_id)
                self.assertLessEqual(r.contraction.max_radius_after, 25.+1e-9)
                np.testing.assert_allclose(center(new.H), center(old.H), atol=1e-9)

    def test_fork_uses_retained_250_center_after_original_gate(self):
        tracker = base.Tracker(branch())
        tracker.H.state.particles[:251, 0] = 1.
        original_center = center(tracker.H)
        # At the original 500-particle gate boundary, outside the sampled gate.
        cs = [base.Candidate(0, tuple(original_center + [0., 25.]), 1.)]
        old, oldr, new, r = self.compare(tracker, cs)
        self.assertEqual(oldr.action, 'fork')
        self.assertEqual(len(new.H.state.weights), 250)
        retained_center = center(old.H)
        self.assertGreater(np.linalg.norm(retained_center-original_center), .1)
        self.assertGreater(np.linalg.norm(np.array(cs[0].center_xy)-retained_center), 25.)
        np.testing.assert_array_equal(r.contraction.center, retained_center)
        np.testing.assert_allclose(center(new.H), retained_center, atol=1e-9)
        self.assert_exact(r.contraction.before_H, old.H)
        self.assert_exact(r.contraction.after_H, new.H)
        self.assert_exact(new.T, old.T)

    def test_t_update_without_promotion_keeps_t_and_counters(self):
        old, oldr, new, r = self.compare(self.dual(), [base.Candidate(0, (100., 100.), 1.)])
        self.assertEqual(oldr.action, 'update_T')
        self.assertEqual(new.C_T, 1)
        self.assertEqual(r.contraction.reason, 'contracted')
        self.assert_exact(new.T, old.T)

    def test_one_base_call_no_extra_rng_or_same_frame_scoring(self):
        tracker = base.Tracker(branch())
        cs = [base.Candidate(0, (51., 50.), 1.)]
        with patch.object(base, 'score_support', wraps=base.score_support) as score, \
             patch.object(np.random, 'default_rng', wraps=np.random.default_rng) as rng:
            old, oldr = base.step(tracker, cs, 1, CFG, branch_radius=25.)
            score_calls, rng_calls = score.call_count, rng.call_count
        with patch.object(base, 'step', wraps=base.step) as step, \
             patch.object(base, 'score_support', wraps=base.score_support) as score, \
             patch.object(np.random, 'default_rng', wraps=np.random.default_rng) as rng:
            new, r = guard.step(tracker, cs, 1, CFG)
            step.assert_called_once_with(tracker, cs, 1, CFG, branch_radius=25.)
            self.assertEqual(score.call_count, score_calls)
            self.assertEqual(rng.call_count, rng_calls)
        self.assert_exact(r.candidate_result, oldr)
        self.assert_exact(new.T, old.T)

    def test_next_frame_predicts_from_contracted_h(self):
        tracker = base.Tracker(branch())
        new, r = guard.step(tracker, [], 1, CFG)
        following, next_r = guard.step(new, [], 2, CFG)
        self.assert_exact(next_r.candidate_result.predicted['H'].state.particles, new.H.state.particles)
        self.assertFalse(np.array_equal(next_r.candidate_result.predicted['H'].state.particles,
                                       r.candidate_result.carried['H'].state.particles))
        self.assert_exact(following.H.state.particles, new.H.state.particles)
        self.assertEqual(next_r.contraction.scale, 1.)

    def test_result_snapshots_and_output_are_detached(self):
        new, r = guard.step(base.Tracker(branch()), [], 1, CFG)
        frozen = copy.deepcopy(r)
        new.H.state.particles[:] = -10
        new.H.state.weights[:] = 0
        self.assert_exact(r, frozen)
        r.output_xy[:] = 0
        self.assert_exact(r.candidate_result.output_xy, frozen.candidate_result.output_xy)
        branches = [r.carried['H'], r.contraction.before_H, r.contraction.after_H,
                    r.candidate_result.carried['H'], r.candidate_result.predicted['H'],
                    r.candidate_result.posterior['H']]
        for i, a in enumerate(branches):
            for b in branches[i+1:]:
                for attr in ('particles', 'weights', 'log_weights'):
                    self.assertFalse(np.shares_memory(getattr(a.state, attr), getattr(b.state, attr)))

    def test_api_has_no_image_ground_truth_or_fallback_dependencies(self):
        self.assertEqual(list(inspect.signature(guard.step).parameters),
                         ['tracker', 'candidates', 'frame_id', 'config', 'enabled'])
        tree = ast.parse(inspect.getsource(guard))
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        names = [alias.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for alias in n.names]
        self.assertNotIn('history_pixel_fallback', names)
        self.assertTrue(set(imports) <= {None, 'copy', 'dataclasses'})
        self.assertTrue(set(names) <= {'deepcopy', 'dataclass', 'math', 'numpy', 'candidate_closed_loop'})


if __name__ == '__main__':
    unittest.main()
