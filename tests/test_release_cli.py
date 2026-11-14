"""Synthetic interface checks; no dataset or pretrained weights are needed."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]


def load(name):
    path = ROOT / 'scripts' / (name + '.py')
    if not path.is_file():
        raise AssertionError('Missing release CLI: ' + str(path))
    spec = importlib.util.spec_from_file_location('release_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseInterfaces(unittest.TestCase):
    def setUp(self):
        # Keep synthetic artifacts; never delete someone else's test directory.
        self.temp = Path(tempfile.mkdtemp(prefix='domp_pf_release_',
            dir=os.environ.get('DOMP_PF_TEST_ARTIFACTS')))

    def initial(self):
        from pf_tbd import core
        cfg = core.make_config(dict(motionModel='legacy_motion', legacyPositionSigma=2,
                                  initialVelocityX=[-2, 2], initialVelocityY=[-1, 1]))
        state = core.initialize(np.zeros((64, 64)), [32, 42], cfg, np.random.default_rng(7))
        path = self.temp / 'initial.npz'
        np.savez(path, initial_state=state.particles, initial_weights=state.weights,
                 initial_log_weights=state.log_weights, frame_size=[64, 64], step_index=1)
        return path, state

    def test_frames_are_numeric_and_reject_gaps(self):
        track = load('track')
        folder = self.temp / 'frames'; folder.mkdir()
        for i in range(12): (folder / (str(i) + '.png')).touch()
        self.assertEqual([p.stem for p in track.image_paths(folder)], [str(i) for i in range(12)])
        (folder / '14.png').touch()
        with self.assertRaises(ValueError): track.image_paths(folder)

    def test_replay_equals_core_with_source_zero_coordinates(self):
        import copy
        from pf_tbd import candidate_closed_loop as base, candidate_position_output
        track = load('track')
        state_path, state = self.initial()
        groups = [dict(frame_id=i, candidates=[dict(candidate_id=1, center_xy=[31., 41.],
                                                image_score=.9)]) for i in range(1, 4)]
        path = self.temp / 'candidates.json'; path.write_text(json.dumps(groups))
        output = self.temp / 'run'
        args = ['--candidates', str(path), '--initial-state', str(state_path),
                '--initial-x', '31', '--initial-y', '41', '--output', str(output)]
        track.main(args)
        with (output / 'predictions.csv').open() as f: rows = list(csv.DictReader(f))
        self.assertEqual([int(r['initialized']) for r in rows], [1, 0, 0, 0])
        self.assertEqual([float(rows[0]['x']), float(rows[0]['y'])], [31, 41])
        cfg = base.Config(**json.loads((ROOT / 'configs/final_candidate_output_v1.json').read_text())['engine'])
        tracker = base.Tracker(base.Branch(0, copy.deepcopy(state)))
        for i, row in enumerate(rows[1:], 1):
            tracker, result = candidate_position_output.step(tracker, [base.Candidate(1, (32, 42), .9)], i, cfg, enabled=True)
            np.testing.assert_array_equal([float(row['x']), float(row['y'])], result.output_xy - 1)
        before = (output / 'predictions.csv').read_bytes()
        with self.assertRaises(FileExistsError): track.main(args)
        self.assertEqual(before, (output / 'predictions.csv').read_bytes())

    def test_candidate_input_rejects_gt_and_reordered_frames(self):
        track = load('track')
        path = self.temp / 'candidates.json'
        path.write_text(json.dumps([dict(frame_id=1, candidates=[], ground_truth=[1, 2])]))
        with self.assertRaises(ValueError): track.read_candidate_groups(path, (64, 64))
        path.write_text(json.dumps([dict(frame_id=2, candidates=[]), dict(frame_id=1, candidates=[])]))
        with self.assertRaises(ValueError): track.read_candidate_groups(path, (64, 64))

    def test_initial_state_center_must_match_declared_center(self):
        track = load('track'); path, _ = self.initial()
        with self.assertRaises(ValueError): track.read_initial_state(path, (64, 64), [1, 2], 500)

    def test_paper_metrics_exclude_initialization_and_use_strict_threshold(self):
        evaluate = load('evaluate')
        labels = self.temp / 'data1.txt'
        labels.write_text('0 data1 8 1\n' + ''.join(f'frame:{i} 1 object:1 0 0\n' for i in range(8)))
        prediction = self.temp / 'predictions.csv'
        with prediction.open('w', newline='') as f:
            writer = csv.DictWriter(f, ['frame_id','initialized','x','y','io_seconds','preprocess_seconds','filter_seconds'])
            writer.writeheader()
            for i, x in enumerate([0, 5, 10, 11, 11, 11, 11, 11]):
                writer.writerow(dict(frame_id=i, initialized=int(i == 0), x=x, y=0,
                                     io_seconds=0, preprocess_seconds=0, filter_seconds=0))
        output = self.temp / 'evaluation'
        evaluate.main(['--predictions', str(prediction), '--labels', str(labels), '--output', str(output)])
        metrics = json.loads((output / 'metrics.json').read_text())
        self.assertEqual(metrics['count'], 7)
        self.assertAlmostEqual(metrics['mean'], 10.)
        self.assertAlmostEqual(metrics['within5_fraction'], 1/7)
        self.assertEqual(metrics['failure_total_frames'], 5)
        self.assertEqual(metrics['failure_intervals'], [dict(start_frame=3, end_frame=7, length=5)])

    def test_macro_averages_sequences_equally(self):
        evaluate = load('evaluate')
        rows = [dict(count=100, mean=1., within5_fraction=1., failure_total_fraction=0.),
                dict(count=10, mean=9., within5_fraction=0., failure_total_fraction=1.)]
        self.assertEqual(evaluate.macro_summary(rows)['mean_error_px'], 5.)
        self.assertEqual(evaluate.macro_summary(rows)['within5_fraction'], .5)

    def test_metadata_clean_checkpoint_loads_strictly(self):
        import torch
        from deep_observation.models.deeper_unet import DeeperUNet
        track = load('track')
        source = DeeperUNet()
        path = self.temp / 'clean_checkpoint.pt'
        torch.save(dict(model='deeper_unet', arm='B', step=5500,
                        state_dict=source.state_dict()), path)
        loaded = track.Frontend(path, 'cpu')
        self.assertFalse(loaded.model.training)
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.model.state_dict()[key]), key)


if __name__ == '__main__':
    unittest.main()
