"""Birth-only radius gate regression tests; old Config serialization is frozen."""
import unittest
from dataclasses import asdict, replace

import numpy as np

from pf_tbd.candidate_closed_loop import Tracker, step
from test_candidate_closed_loop import branch, candidate, NOISELESS


class RadiusGateTests(unittest.TestCase):
    def run_step(self, tracker, candidates, radius=25., config=NOISELESS):
        return step(tracker, candidates, 11, config, branch_radius=radius)

    def test_distant_only_forbids_fork_and_preserves_prediction(self):
        tracker = Tracker(branch(n=500))
        out, result = self.run_step(tracker, [candidate((100, 50))])
        self.assertEqual(result.action, 'predict_only')
        self.assertIsNone(out.T)
        self.assertEqual(out.next_path_id, 1)
        np.testing.assert_array_equal(out.H.state.particles, result.predicted['H'].state.particles)
        self.assertEqual(result.parent_indices, {})

    def test_lower_score_near_candidate_replaces_far_top1_and_stable_tie(self):
        cs = [candidate((100, 50), 9, 0), candidate((70, 50), 2, 1), candidate((50, 70), 2, 2)]
        out, result = self.run_step(Tracker(branch(n=500)), cs)
        self.assertEqual(result.action, 'fork')
        np.testing.assert_array_equal(out.T.state.particles[:, :2], np.tile([70, 50], (250, 1)))

    def test_boundary_inclusive(self):
        for direction in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            for distance, expected in [(25., 'fork'), (25.001, 'predict_only')]:
                with self.subTest(direction=direction, distance=distance):
                    xy = 50.+distance*np.asarray(direction)
                    _, result = self.run_step(Tracker(branch(n=500)), [candidate(xy)])
                    self.assertEqual(result.action, expected)

    def test_center_is_current_predicted_not_previous(self):
        tracker = Tracker(branch(n=500, velocity=(20., 0.)))
        out, result = self.run_step(tracker, [candidate((90, 50))])
        self.assertEqual(result.action, 'fork')
        np.testing.assert_array_equal(result.predicted['H'].state.particles[:, 0], np.full(500, 70.))

    def test_weighted_center_before_split(self):
        h = branch(n=500, weights=np.r_[.9, np.full(499, .1/499)])
        h.state.particles[1:, 0] = 150.
        # Weighted center x=60, unweighted x~150, split sample center is not exactly60.
        _, result = self.run_step(Tracker(h), [candidate((84.9999, 50))])
        self.assertEqual(result.action, 'fork')
        _, result = self.run_step(Tracker(h), [candidate((85.001, 50))])
        self.assertEqual(result.action, 'predict_only')

    def test_dual_no_match_outside_holds_and_resets_counter(self):
        tracker = Tracker(branch(), branch((100, 100), path_id=1), C_T=2, next_path_id=2)
        out, result = self.run_step(tracker, [candidate((200, 200))])
        self.assertEqual((result.action, out.C_T, out.next_path_id), ('hold', 0, 2))
        for role in ('H', 'T'):
            np.testing.assert_array_equal(getattr(out, role).state.particles, result.predicted[role].state.particles)
        self.assertEqual(result.parent_indices, {})

    def test_existing_t_beyond_h_radius_updates_and_promotes(self):
        for counter, action in [(0, 'update_T'), (2, 'promote_T')]:
            tracker = Tracker(branch(), branch((100, 100), path_id=1), C_T=counter, next_path_id=2)
            out, result = self.run_step(tracker, [candidate((100, 100))])
            self.assertEqual(result.action, action)
            self.assertEqual(result.update_counts['T'], 1)

    def test_reseed_selects_near_candidate_and_advances_path(self):
        tracker = Tracker(branch(), branch((100, 100), path_id=1), C_T=2, next_path_id=2)
        out, result = self.run_step(tracker, [candidate((200, 200), 9, 0), candidate((70, 50), 1, 1)])
        self.assertEqual((result.action, out.C_T, out.next_path_id, out.T.path_id), ('reseed_T', 0, 3, 2))
        np.testing.assert_array_equal(out.T.state.particles[:, :2], np.tile([70, 50], (250, 1)))

    def test_normal_h_match_unchanged_even_with_tiny_gate(self):
        tracker = Tracker(branch(n=500))
        out, result = self.run_step(tracker, [candidate((54, 50))], radius=.1)
        self.assertEqual(result.action, 'single_update')
        self.assertEqual(result.update_counts['H'], 1)

    def test_none_regression_and_config_schema(self):
        tracker = Tracker(branch(n=500))
        cs = [candidate((100, 100))]
        old, a = step(tracker, cs, 11, NOISELESS)
        new, b = self.run_step(tracker, cs, None)
        self.assertEqual(a.action, b.action)
        self.assertEqual(old.next_path_id, new.next_path_id)
        for role in ('H', 'T'):
            np.testing.assert_array_equal(getattr(old, role).state.particles, getattr(new, role).state.particles)
        self.assertNotIn('branch_radius', asdict(NOISELESS))

    def test_invalid_radius(self):
        for radius in (0, -1, float('nan'), float('inf'), True, np.bool_(False), '25', [25]):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                self.run_step(Tracker(branch(n=500)), [], radius)


if __name__ == '__main__':
    unittest.main()
