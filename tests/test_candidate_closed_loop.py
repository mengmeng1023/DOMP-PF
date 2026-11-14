"""Synthetic S18 arithmetic and complete transition-table regression tests."""
import unittest
from dataclasses import replace

import numpy as np

from pf_tbd.core import State
from pf_tbd.candidate_closed_loop import (
    Candidate, Branch, Tracker, Config, position_posterior, systematic_indices,
    score_support, match_candidates, step,
)


def branch(xy=(50., 50.), n=250, path_id=0, weights=None, velocity=(0., 0.)):
    p = np.tile([*xy, *velocity, 9.], (n, 1))
    w = np.full(n, 1/n) if weights is None else np.asarray(weights, dtype=float)
    lw = np.full(n, -np.inf)
    np.log(w, out=lw, where=w > 0)
    return Branch(path_id, State(p, w, lw, (256, 256), 0))


def candidate(xy=(50., 50.), score=1., cid=0):
    return Candidate(cid, tuple(xy), score)


NOISELESS = Config(position_sigma=0., velocity_sigma=0., amplitude_sigma=0.)


class CandidateMathTests(unittest.TestCase):
    def test_position_likelihood_and_no_state_motion(self):
        p = np.array([[10, 10, 1, 2, 9], [13, 14, 3, 4, 9]], dtype=float)
        original = p.copy()
        w, lw = position_posterior(p, np.log([.25, .75]), [10, 10])
        expected = np.array([.25, .75 * np.exp(-25 / 18)])
        np.testing.assert_allclose(w, expected / expected.sum(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.exp(lw), w, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(p, original)

    def test_zero_prior_and_far_log(self):
        p = np.array([[1., 1., 0, 0, 1], [2., 2., 0, 0, 1]])
        w, lw = position_posterior(p, [-np.inf, 0], [1e5, 1e5])
        np.testing.assert_array_equal(w, [0, 1])
        self.assertEqual(lw[0], -np.inf)
        support = score_support(branch(n=2), candidate((1e5, 1e5)), NOISELESS)
        self.assertTrue(np.isfinite(support.logM))
        self.assertEqual(support.M, 0.)

    def test_systematic_zero_mass_full_state_and_cdf_right(self):
        p = np.arange(15).reshape(3, 5)
        idx = systematic_indices([0, 1, 0], 500, .2)
        np.testing.assert_array_equal(p[idx], np.repeat(p[1:2], 500, axis=0))
        np.testing.assert_array_equal(systematic_indices([.5, .5], 2, 0), [0, 1])
        np.testing.assert_array_equal(systematic_indices([0, .5, .5], 2, 0), [1, 2])

    def test_support_branch_normalization_radius_and_threshold_inclusive(self):
        b = branch(n=2, weights=[2., 98.])
        b.state.particles[0, :2] = [53, 54]
        b.state.particles[1, :2] = [200, 200]
        s = score_support(b, candidate(), replace(NOISELESS, M_min=0.))
        self.assertAlmostEqual(s.Q, .02)
        self.assertAlmostEqual(s.M, .02*np.exp(-25/18))
        m = match_candidates(b, [candidate()], replace(NOISELESS, M_min=s.M, Q_min=s.Q))
        self.assertEqual(m.status, 'unique')
        self.assertEqual(match_candidates(b, [candidate()], NOISELESS).status, 'none')

    def test_same_candidate_dual_gate_and_tie(self):
        b = branch(n=2, weights=[.99, .01])
        b.state.particles[1, :2] = [100, 100]
        cfg = replace(NOISELESS, Q_min=.02, M_min=.01)
        self.assertEqual(match_candidates(b, [candidate((100, 100))], cfg).status, 'none')
        self.assertEqual(match_candidates(b, [candidate(cid=0), candidate(cid=1)], cfg).status, 'ambiguous')

    def test_uniform_250_and_500_two_percent_support_is_inclusive(self):
        for n, supported in ((250, 5), (500, 10)):
            with self.subTest(n=n):
                b = branch(n=n)
                b.state.particles[supported:, :2] = [200, 200]
                row = score_support(b, candidate(), NOISELESS)
                self.assertEqual(row.Q, .02)
                self.assertGreater(row.M, .01)
                self.assertTrue(row.eligible)
                self.assertEqual(match_candidates(b, [candidate()], NOISELESS).status, 'unique')

    def test_invalid_inputs(self):
        p = np.ones((2, 5))
        for sigma in (0, -1, np.nan, np.inf):
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                position_posterior(p, np.log([.5, .5]), [1, 1], sigma)
        for prior in ([np.nan, 0], [np.inf, 0], [-np.inf, -np.inf], [0]):
            with self.subTest(prior=prior), self.assertRaises(ValueError):
                position_posterior(p, prior, [1, 1])
        for w in ([-1, 2], [np.nan, 1], [np.inf, 1], [0, 0], [[1, 1]]):
            with self.subTest(weights=w), self.assertRaises(ValueError):
                systematic_indices(w, 500, 0)
        for count, u in ((0, 0), (True, 0), (2, 1), (2, np.nan)):
            with self.assertRaises(ValueError):
                systematic_indices([1, 1], count, u)
        with self.assertRaises(ValueError):
            position_posterior(np.ones((2, 4)), [0, 0], [1, 1])
        with self.assertRaises(ValueError):
            position_posterior(np.full((2, 5), np.nan), [0, 0], [1, 1])
        for score in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                Candidate(0, (1, 1), score)
        with self.assertRaises(TypeError):
            Candidate(0, (1, 1), 1., gt=(50, 50))


class TransitionTests(unittest.TestCase):
    def test_single_update_and_original_inputs_untouched(self):
        old = Tracker(branch(n=500))
        new, result = step(old, [candidate((52, 50))], 1, NOISELESS)
        self.assertEqual(result.action, 'single_update')
        self.assertEqual(result.update_counts, {'H': 1})
        self.assertEqual(new.H.state.step_index, 1)
        self.assertEqual(old.H.state.step_index, 0)
        self.assertIsNot(new.H.state.particles, old.H.state.particles)
        np.testing.assert_array_equal(result.predicted['H'].state.particles, result.posterior['H'].state.particles)

    def test_single_empty_ambiguous_and_control_none_do_not_fork(self):
        for candidates, cfg in (([], NOISELESS), ([candidate(cid=1), candidate(cid=2)], NOISELESS),
                                ([candidate((100, 100))], replace(NOISELESS, dual_enabled=False))):
            new, result = step(Tracker(branch(n=500)), candidates, 1, cfg)
            self.assertIsNone(new.T)
            self.assertEqual(result.action, 'predict_only')
            self.assertEqual(result.update_counts, {'H': 0})

    def test_fork_weighted_full_parent_copy_top1_and_no_observation(self):
        h = branch(n=500)
        h.state.weights[:] = 0
        h.state.weights[123] = 1
        h.state.log_weights[:] = -np.inf
        h.state.log_weights[123] = 0
        h.state.particles[123, 2:] = [1, 2, 99]
        new, result = step(Tracker(h), [candidate((100, 100), 2, 9), candidate((90, 90), 2, 1)], 1, NOISELESS)
        self.assertEqual(result.action, 'fork')
        self.assertEqual(new.C_T, 0)
        self.assertEqual(len(new.H.state.weights), 250)
        self.assertEqual(len(new.T.state.weights), 250)
        np.testing.assert_array_equal(new.T.state.particles[:, :2], np.tile([90, 90], (250, 1)))
        np.testing.assert_array_equal(new.T.state.particles[:, 2:], np.tile([1, 2, 99], (250, 1)))
        np.testing.assert_allclose(result.output_xy, [51, 52], rtol=0, atol=1e-12)
        self.assertTrue(all(v == 0 for v in result.update_counts.values()))
        self.assertEqual(result.unique_parent_count['H'], 1)

    def test_t_requires_three_consecutive_frames_candidate_id_can_change(self):
        tracker = Tracker(branch(), branch((100, 100), path_id=1), next_path_id=2)
        for frame in (1, 2, 3):
            tracker, r = step(tracker, [candidate((100, 100), cid=frame)], frame, NOISELESS)
            np.testing.assert_allclose(r.output_xy, [50, 50] if frame < 3 else [100, 100], rtol=0, atol=1e-12)
            self.assertEqual(r.update_counts, {'H': 0, 'T': 1})
            self.assertEqual(r.action, 'update_T' if frame < 3 else 'promote_T')
        self.assertIsNone(tracker.T)
        self.assertEqual(tracker.H.path_id, 1)
        self.assertEqual(len(tracker.H.state.weights), 500)
        self.assertEqual(r.parent_path_ids['H'], 1)
        np.testing.assert_array_equal(tracker.H.state.particles, r.posterior['T'].state.particles[r.parent_indices['H']])

    def test_empty_resets_counter_without_compression_or_merge(self):
        tracker = Tracker(branch(), branch((51, 50), path_id=1), C_T=2, next_path_id=2)
        tracker, r = step(tracker, [], 1, NOISELESS)
        self.assertEqual(tracker.C_T, 0)
        self.assertIsNotNone(tracker.T)
        self.assertEqual(r.action, 'hold')
        self.assertEqual(r.parent_indices, {})
        np.testing.assert_array_equal(r.predicted['H'].state.weights, tracker.H.state.weights)

    def test_h_priority_even_low_image_score_and_t_stronger(self):
        h = branch((53, 50))
        tracker = Tracker(h, branch((100, 100), path_id=1), C_T=2, next_path_id=2)
        new, r = step(tracker, [candidate(score=.001), candidate((100, 100), 99, 1)], 1, NOISELESS)
        self.assertEqual(r.action, 'confirm_H')
        self.assertEqual(r.update_counts, {'H': 1, 'T': 0})
        self.assertEqual(r.parent_path_ids, {'H': 0})
        self.assertEqual(list(r.parent_indices), ['H'])
        self.assertIsNone(new.T)
        np.testing.assert_array_equal(new.H.state.particles, r.posterior['H'].state.particles[r.parent_indices['H']])

    def test_low_ess_winner_collapses_directly_once_for_h_and_t(self):
        for winner in ('H', 'T'):
            h, t = branch(), branch((100, 100), path_id=1)
            selected = h if winner == 'H' else t
            selected.state.weights[:] = 0
            selected.state.weights[:2] = [.7, .3]
            selected.state.log_weights[:] = -np.inf
            selected.state.log_weights[:2] = np.log([.7, .3])
            selected.state.particles[1, 0] += 3
            selected.state.particles[:, 4] = np.arange(250)
            tracker = Tracker(h, t, C_T=1, next_path_id=2)
            new, r = step(tracker, [candidate((50, 50) if winner == 'H' else (100, 100))], 9,
                          replace(NOISELESS, K=2))
            self.assertEqual(set(r.parent_indices), {'H'})
            self.assertLess(r.ess[winner], 125)
            self.assertEqual(r.update_counts[winner], 1)
            self.assertEqual(sum(r.update_counts.values()), 1)
            posterior = r.posterior[winner].state
            u = np.random.default_rng(np.random.SeedSequence([7, 9, selected.path_id, 5])).random()
            expected = systematic_indices(posterior.weights, 500, u)
            np.testing.assert_array_equal(r.parent_indices['H'], expected)
            np.testing.assert_array_equal(new.H.state.particles, posterior.particles[expected])
            np.testing.assert_allclose(r.output_xy, posterior.weights @ posterior.particles[:, :2], atol=1e-12)

    def test_unobserved_low_ess_history_keeps_all_particles_and_weights(self):
        h = branch()
        h.state.weights[:] = 0
        h.state.weights[249] = 1
        h.state.log_weights[:] = -np.inf
        h.state.log_weights[249] = 0
        new, r = step(Tracker(h, branch((100, 100), path_id=1), next_path_id=2),
                      [candidate((100, 100))], 1, NOISELESS)
        self.assertEqual(len(new.H.state.weights), 250)
        self.assertNotIn('H', r.parent_indices)
        np.testing.assert_array_equal(new.H.state.weights, h.state.weights)
        np.testing.assert_array_equal(new.H.state.particles, h.state.particles)

    def test_t_reseed_own_velocity_and_h_protected(self):
        tracker = Tracker(branch(velocity=(1, 0)), branch((100, 100), path_id=1, velocity=(3, 4)), C_T=2, next_path_id=2)
        new, r = step(tracker, [candidate((200, 200))], 1, NOISELESS)
        self.assertEqual(r.action, 'reseed_T')
        self.assertEqual(new.T.path_id, 2)
        self.assertEqual(new.next_path_id, 3)
        self.assertEqual(new.C_T, 0)
        self.assertEqual(r.update_counts, {'H': 0, 'T': 0})
        np.testing.assert_array_equal(new.T.state.particles[:, 2:4], np.tile([3, 4], (250, 1)))
        np.testing.assert_array_equal(new.H.state.particles, r.predicted['H'].state.particles)
        np.testing.assert_array_equal(new.H.state.weights, r.predicted['H'].state.weights)
        self.assertEqual(r.parent_path_ids, {'T': 1})

    def test_h_and_t_ambiguity_hold_without_forced_action(self):
        for xy in ((50, 50), (100, 100)):
            tracker = Tracker(branch(), branch((100, 100), path_id=1), C_T=2, next_path_id=2)
            new, r = step(tracker, [candidate(xy, cid=1), candidate(xy, cid=2)], 1, NOISELESS)
            self.assertEqual(r.action, 'hold')
            self.assertEqual(new.C_T, 0)
            self.assertEqual(r.update_counts, {'H': 0, 'T': 0})
            self.assertEqual(r.parent_indices, {})

    def test_ess_equal_125_not_resampled_below_is(self):
        for active in (125, 124):
            w = np.zeros(250)
            w[:active] = 1/active
            tracker = Tracker(branch(), branch((100, 100), path_id=1, weights=w), next_path_id=2)
            new, r = step(tracker, [candidate((100, 100))], 1, NOISELESS)
            self.assertAlmostEqual(r.ess['T'], active)
            self.assertEqual('T' in r.parent_indices, active < 125)
            np.testing.assert_array_equal(new.H.state.particles, r.predicted['H'].state.particles)

    def test_deterministic_random_keys_and_snapshots_detached(self):
        t = Tracker(branch(n=500))
        a, ar = step(t, [candidate((200, 200))], 7, Config())
        b, br = step(t, [candidate((200, 200))], 7, Config(K=2))
        for label in ('H', 'T'):
            np.testing.assert_array_equal(getattr(a, label).state.particles, getattr(b, label).state.particles)
        a.H.state.particles[:] = -1
        np.testing.assert_array_equal(ar.carried['H'].state.particles, br.carried['H'].state.particles)
        np.testing.assert_array_equal(ar.parent_indices['H'], br.parent_indices['H'])

    def test_invalid_branch_sizes_and_weights_rejected(self):
        for h in (branch(n=499), branch(n=500, weights=np.full(500, np.nan))):
            with self.assertRaises(ValueError):
                step(Tracker(h), [], 1, NOISELESS)

    def test_inconsistent_log_prior_rejected(self):
        h = branch(n=500)
        h.state.log_weights[0] += 5
        with self.assertRaisesRegex(ValueError, 'different priors'):
            step(Tracker(h), [], 1, NOISELESS)


if __name__ == '__main__':
    unittest.main()
