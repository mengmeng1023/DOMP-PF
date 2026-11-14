"""Offline MovingTarget evaluation; never imported by the tracking entry point."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

from evaluate_tracking import evaluate_rows, read_predictions, write_errors
from movingtarget_data import parse_labels


def macro_summary(metrics):
    """Paper summary: arithmetic mean of per-sequence metrics, not frame pooling."""
    if not metrics: raise ValueError('No sequence metrics')
    return dict(sequence_count=len(metrics), aggregation='equal weight per sequence',
                mean_error_px=mean(m['mean'] for m in metrics),
                within5_fraction=mean(m['within5_fraction'] for m in metrics),
                sustained_failure_fraction=mean(m['failure_total_fraction'] for m in metrics))


def evaluate(predictions, label_path):
    parsed = parse_labels(label_path)
    ids = [int(p['frame_id']) for p in predictions]
    if len(ids) < 2 or ids != list(range(len(ids))):
        raise ValueError('Predictions must contain the complete timeline beginning with frame 0')
    if parsed['header']['frame_count'] != len(ids) or set(parsed['rows']) != set(ids):
        raise ValueError('Prediction/label timelines or header frame count differ')
    reference = []; target_id = None
    for frame in ids:
        row = parsed['rows'][frame]
        # Paper sequences have one valid, continuously identified target.
        # Refuse to silently score a partial/malformed or multi-target sequence.
        if not row['label_valid'] or row['target_count'] != 1 or len(row['objects']) != 1:
            raise ValueError('This evaluator requires one valid target on every frame')
        point = row['objects'][0]
        if target_id is None: target_id = point['object_id']
        if point['object_id'] != target_id: raise ValueError('Target identity changes within sequence')
        reference.append(dict(frame_id=frame, label_valid=True, target_count=1, x=point['x'], y=point['y']))
    if [float(predictions[0][k]) for k in ('x','y')] != [reference[0][k] for k in ('x','y')]:
        raise ValueError('Initial prediction differs from the supplied first-frame center')
    result = evaluate_rows(predictions, reference, failure_threshold_px=10., consecutive_frames=5)
    summary = result['summary']; frames = result['frames'][1:]; episodes = summary['failure_episodes']
    count = summary['evaluated_frames']
    metric = dict(sequence=parsed['header']['sequence'], count=count, mean=summary['mean_error_px'],
                  median=summary['median_error_px'], rmse=summary['rmse_px'], p90=summary['p90_error_px'],
                  p95=summary['p95_error_px'], max=summary['max_error_px'],
                  within5_fraction=mean(row['error_px'] <= 5 for row in frames),
                  failure_total_frames=summary['sustained_failure_frames'],
                  failure_longest_frames=max((e['length'] for e in episodes), default=0),
                  failure_total_fraction=summary['sustained_failure_frames']/count,
                  failure_interval_count=len(episodes), failure_intervals=episodes,
                  initialization_excluded=True, coordinate_origin=0,
                  failure_rule='error > 10 pixels for at least 5 consecutive frames',
                  aggregation='Report each sequence; paper aggregate gives each sequence equal weight')
    return metric, result['frames']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, help='Single-sequence predictions.csv, source0 coordinates')
    parser.add_argument('--labels', type=Path, help='Original MovingTarget data_label/dataN.txt, offline only')
    parser.add_argument('--metrics', nargs='+', type=Path, help='Aggregate already computed per-sequence metrics.json files')
    parser.add_argument('--output', type=Path, required=True, help='New, nonexistent output directory')
    args = parser.parse_args(argv)
    if args.output.exists(): raise FileExistsError('Output already exists; choose a new directory')
    if args.metrics:
        if args.predictions or args.labels: parser.error('--metrics cannot be combined with predictions/labels')
        rows = [json.loads(p.read_text(encoding='utf-8')) for p in args.metrics]
        names = [r['sequence'] for r in rows]
        if len(set(names)) != len(names): raise ValueError('Duplicate sequences in macro aggregate')
        for row in rows:
            if (not isinstance(row['count'], int) or isinstance(row['count'], bool) or row['count'] < 1
                    or not all(math.isfinite(row[k]) for k in ('mean','within5_fraction','failure_total_fraction'))
                    or row['mean'] < 0 or not 0 <= row['within5_fraction'] <= 1
                    or not 0 <= row['failure_total_fraction'] <= 1):
                raise ValueError('Invalid sequence metric')
        result = macro_summary(rows)
        result['sequences'] = names
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'macro_metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    else:
        if args.predictions is None or args.labels is None: parser.error('--predictions and --labels are required')
        result, frames = evaluate(read_predictions(args.predictions), args.labels)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        write_errors(args.output/'frame_errors.csv', frames)
        with (args.output/'failure_intervals.csv').open('x', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, ['start_frame','end_frame','length']); writer.writeheader()
            writer.writerows(result['failure_intervals'])
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
