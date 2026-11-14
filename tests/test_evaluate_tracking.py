import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class EvaluationTests(unittest.TestCase):
    def evaluator(self):
        self.assertIsNotNone(importlib.util.find_spec('evaluate_tracking'), 'offline evaluator is required')
        from evaluate_tracking import evaluate_rows
        return evaluate_rows

    def rows(self, errors):
        truth = [dict(frame_id=i, target_count=1, label_valid=True, x=1., y=2., exclusion_reason=None) for i in range(len(errors))]
        pred = [dict(frame_id=i, initialized=int(i == 0), x=1.+v, y=2., io_seconds=.01, preprocess_seconds=.02, filter_seconds=.03) for i, v in enumerate(errors)]
        return pred, truth

    def test_first_frame_is_not_accuracy_evidence(self):
        pred, truth = self.rows([0, 3, 4])
        result = self.evaluator()(pred, truth)
        self.assertEqual(result['summary']['evaluated_frames'], 2)
        self.assertEqual(result['summary']['excluded_counts'], {'initialization': 1})
        self.assertAlmostEqual(result['summary']['mean_error_px'], 3.5)
        self.assertAlmostEqual(result['summary']['rmse_px'], (25/2)**.5)
        self.assertIsNone(result['summary']['failure_rule'])

    def test_invalid_and_absent_labels_have_distinct_exclusions(self):
        pred, truth = self.rows([0, 3, 4, 5])
        truth[1].update(label_valid=False, exclusion_reason='invalid_coordinates')
        truth[2].update(target_count=0, x=None, y=None)
        result = self.evaluator()(pred, truth, 2., 2)
        self.assertEqual(result['summary']['excluded_counts'], {'initialization': 1, 'invalid_coordinates': 1, 'target_absent': 1})
        self.assertEqual(result['summary']['failure_episodes'], [])

    def test_consecutive_failure_and_recovery(self):
        pred, truth = self.rows([0, 1, 6, 7, 3, 8, 9, 10])
        result = self.evaluator()(pred, truth, 5., 2)
        self.assertEqual(result['summary']['failure_episodes'], [dict(start_frame=2, end_frame=3, length=2), dict(start_frame=5, end_frame=7, length=3)])
        self.assertEqual(result['summary']['sustained_failure_frames'], 5)

    def test_alignment_or_nonfinite_predictions_rejected(self):
        pred, truth = self.rows([0, 1, 2])
        evaluate = self.evaluator()
        with self.assertRaises(ValueError):
            evaluate(pred[:-1], truth)
        pred[1]['x'] = float('nan')
        with self.assertRaises(ValueError):
            evaluate(pred, truth)

    def test_gap_must_not_count_as_consecutive(self):
        pred, truth = self.rows([0, 7, 7])
        pred[2]['frame_id'] = truth[2]['frame_id'] = 3
        with self.assertRaises(ValueError):
            self.evaluator()(pred, truth, 5., 2)


if __name__ == '__main__':
    unittest.main()
