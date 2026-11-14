"""Offline conditional-center evaluation; never imported by the MATLAB estimator."""
from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median


def quantile(values, probability):
    values = sorted(values)
    index = (len(values) - 1) * probability
    left = math.floor(index)
    right = math.ceil(index)
    return values[left] + (values[right] - values[left]) * (index - left)


def evaluate_rows(predictions, truth, failure_threshold_px=None, consecutive_frames=None):
    if not predictions or len(predictions) != len(truth):
        raise ValueError('Predictions must cover every original clip frame exactly once')
    ids = [int(row['frame_id']) for row in truth]
    if ids != list(range(ids[0], ids[0] + len(ids))):
        raise ValueError('Frame timeline is not contiguous')
    if [int(row['frame_id']) for row in predictions] != ids:
        raise ValueError('Prediction and annotation frame IDs differ')
    if (failure_threshold_px is None) != (consecutive_frames is None):
        raise ValueError('Both failure threshold and consecutive frame count are required')
    if failure_threshold_px is not None and (not math.isfinite(failure_threshold_px) or failure_threshold_px <= 0 or
                                            isinstance(consecutive_frames, bool) or not isinstance(consecutive_frames, int) or consecutive_frames < 1):
        raise ValueError('Invalid failure definition')
    errors, frame_rows, episodes = [], [], []
    exclusions = Counter()
    timing = {key: 0. for key in ('io_seconds', 'preprocess_seconds', 'filter_seconds')}
    streak = []

    def finish_streak():
        if consecutive_frames is not None and len(streak) >= consecutive_frames:
            episodes.append(dict(start_frame=streak[0], end_frame=streak[-1], length=len(streak)))
        streak.clear()

    for index, (pred, label) in enumerate(zip(predictions, truth)):
        xy = [float(pred[k]) for k in ('x', 'y')]
        if not all(math.isfinite(v) for v in xy):
            raise ValueError('Nonfinite prediction')
        initialized = int(pred['initialized'])
        if initialized != int(index == 0):
            raise ValueError('Initialization must occur exactly on the first frame')
        for key in timing:
            value = float(pred[key])
            if not math.isfinite(value) or value < 0:
                raise ValueError('Invalid timing')
            timing[key] += value
        reason = None
        if index == 0:
            reason = 'initialization'
        elif not label['label_valid']:
            reason = label.get('exclusion_reason') or 'invalid_label'
        elif label['target_count'] == 0:
            reason = 'target_absent'
        elif label['target_count'] != 1:
            reason = 'multiple_targets'
        error = None
        if reason:
            exclusions[reason] += 1
        else:
            reference = [float(label[k]) for k in ('x', 'y')]
            if not all(math.isfinite(v) for v in reference):
                raise ValueError('Valid reference must have finite coordinates')
            error = math.hypot(xy[0] - reference[0], xy[1] - reference[1])
            errors.append(error)
        above = error is not None and failure_threshold_px is not None and error > failure_threshold_px
        if above:
            streak.append(ids[index])
        else:
            finish_streak()
        frame_rows.append(dict(frame_id=ids[index], initialized=initialized, x=xy[0], y=xy[1],
                               reference_x=label.get('x'), reference_y=label.get('y'),
                               target_count=label['target_count'], label_valid=label['label_valid'],
                               error_px=error, exclusion_reason=reason, exceeds_threshold=bool(above)))
    finish_streak()
    if not errors:
        raise ValueError('No evaluable post-initialization frames')
    rule = None if failure_threshold_px is None else dict(threshold_px=failure_threshold_px,
                                                        consecutive_frames=consecutive_frames, comparison='strictly_greater')
    summary = dict(raw_clip_frames=len(truth), evaluated_frames=len(errors), excluded_counts=dict(exclusions),
                   mean_error_px=mean(errors), median_error_px=median(errors), rmse_px=math.sqrt(mean(e*e for e in errors)),
                   p90_error_px=quantile(errors, .9), p95_error_px=quantile(errors, .95), max_error_px=max(errors),
                   failure_rule=rule, failure_episodes=episodes if rule else None,
                   sustained_failure_frames=sum(e['length'] for e in episodes) if rule else None,
                   above_threshold_frames=sum(row['exceeds_threshold'] for row in frame_rows) if rule else None,
                   timing_seconds=timing, timing_includes_initialization=True,
                   accuracy_includes_initialization=False, task='conditional_single_target_present_tracking')
    return dict(summary=summary, frames=frame_rows)


def read_predictions(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def write_errors(path, rows):
    with Path(path).open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
