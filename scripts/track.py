"""Portable DOMP-PF inference and frozen-candidate replay, with no GT input.

The initial particle state is an explicit input so historical initialization is
preserved without silently replacing the legacy first-frame observation.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pf_tbd import core, candidate_closed_loop as base, candidate_position_output


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def image_paths(folder):
    """Require a complete, numerically ordered timeline beginning with 0."""
    files = {}
    for path in Path(folder).iterdir():
        if not path.is_file() or path.suffix.lower() not in {'.bmp','.png','.jpg','.jpeg','.tif','.tiff'}:
            continue
        if not path.stem.isdigit(): raise ValueError('Image names must be numeric frame IDs')
        frame = int(path.stem)
        if frame in files: raise ValueError('Duplicate image frame ID')
        files[frame] = path
    if len(files) < 2 or sorted(files) != list(range(len(files))):
        raise ValueError('Images must cover frame IDs 0..N-1, with N >= 2 and no gaps')
    return [files[i] for i in range(len(files))]


def read_initial_state(path, fallback_shape, initial_source0, particles):
    """Load original initial_* arrays; NPZ frame_size is [height,width]."""
    with np.load(path, allow_pickle=False) as data:
        shape = data['frame_size'] if 'frame_size' in data else np.asarray(fallback_shape)
        if np.asarray(shape).shape != (2,) or not np.issubdtype(np.asarray(shape).dtype, np.integer):
            raise ValueError('frame_size must contain integer height,width')
        shape = tuple(int(x) for x in shape)
        index = data['step_index'].item() if 'step_index' in data else 1
        if isinstance(index, bool) or index != 1: raise ValueError('Expected frame-zero initial state, step_index=1')
        state = core.State(data['initial_state'].copy(), data['initial_weights'].copy(),
                           data['initial_log_weights'].copy(), shape, 1)
    base._validate_branch(base.Branch(0, state))
    if state.particles.dtype != np.float64 or state.particles.shape != (particles, 5):
        raise ValueError('Initial particles must be float64 with the configured N x 5 shape')
    center = np.asarray(initial_source0, dtype=np.float64)
    if (center.shape != (2,) or not np.isfinite(center).all() or (center < 0).any()
            or center[0] >= shape[1] or center[1] >= shape[0]):
        raise ValueError('Initial source0 center is outside the frame')
    if not np.array_equal(state.particles[:, :2], np.tile(center + 1, (particles, 1))):
        raise ValueError('Initial particle positions differ from the explicit frame-zero center')
    return state


def read_candidate_groups(path, shape):
    """Read the frozen score-only S27/S28 schema; reject extra/GT fields."""
    groups = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(groups, list) or not groups:
        raise ValueError('Candidate input must contain one group for each post-initialization frame')
    for index, group in enumerate(groups, 1):
        if (not isinstance(group, dict) or not {'frame_id','candidates'} <= set(group)
                or set(group)-{'frame_id','candidates','sequence','key'}):
            raise ValueError('Unexpected candidate group fields; ground truth is not an online input')
        if type(group['frame_id']) is not int or group['frame_id'] != index:
            raise ValueError('Candidate frame IDs must be contiguous and ordered from 1')
        candidates = group['candidates']
        if not isinstance(candidates, list) or len(candidates) > 20:
            raise ValueError('Each frame must contain at most 20 candidates')
        previous = math.inf
        for candidate_id, candidate in enumerate(candidates, 1):
            if not isinstance(candidate, dict) or set(candidate) != {'candidate_id','center_xy','image_score'}:
                raise ValueError('Unexpected candidate fields')
            xy = np.asarray(candidate['center_xy'], dtype=np.float64)
            score = candidate['image_score']
            if (type(candidate['candidate_id']) is not int or candidate['candidate_id'] != candidate_id
                    or xy.shape != (2,) or not np.isfinite(xy).all() or (xy < 0).any()
                    or xy[0] >= shape[1] or xy[1] >= shape[0]
                    or type(score) not in (int, float) or not math.isfinite(score)
                    or not 0 <= score <= 1 or score > previous):
                raise ValueError('Invalid candidate ID, source0 center, score, or rank')
            previous = score
    return groups


class Frontend:
    """The S11 objectness checkpoint and original candidate extractor."""
    def __init__(self, checkpoint, device):
        import torch
        from deep_observation.models.deeper_unet import DeeperUNet
        from deep_observation.dense_objectness import spatial_candidates
        self.torch = torch
        self.extract = spatial_candidates
        self.device = torch.device(device)
        # Exact settings from the frozen S11 configure_torch function.
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if payload.get('model') != 'deeper_unet' or payload.get('step') != 5500 or payload.get('arm') != 'B':
            raise ValueError('Expected the frozen S11 objectness DeeperUNet, arm B, step 5500')
        self.model = DeeperUNet().float().to(self.device)
        self.model.load_state_dict(payload['state_dict'], strict=True)
        self.model.eval()

    def infer(self, gray):
        from torch.nn import functional as F
        torch = self.torch
        x = torch.from_numpy(np.array(gray, copy=True)).float().to(self.device)[None, None] / 255.
        h, w = x.shape[-2:]
        with torch.inference_mode():
            # Frozen run_s11_dense.predict_logits, followed by S27 float32 scores.
            raw = self.model(F.pad(x, (0, (-w) % 16, 0, (-h) % 16), mode='replicate'))
            if raw.shape != (len(x), 1, h + (-h) % 16, w + (-w) % 16) or not torch.isfinite(raw).all():
                raise ValueError('Nonfinite raw output or spatial mismatch')
            z = raw[..., :h, :w]
            score = z.sigmoid()[0, 0].cpu().numpy().astype(np.float32)
        return [dict(candidate_id=i, center_xy=r['center_xy'], image_score=r['raw_score'])
                for i, r in enumerate(self.extract(score), 1)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--frames', type=Path, help='Numeric image files 0..N-1; no label file is read')
    source.add_argument('--candidates', type=Path, help='Frozen post-initialization candidate JSON')
    parser.add_argument('--checkpoint', type=Path, help='S11 best.pt; required only with --frames')
    parser.add_argument('--initial-state', required=True, type=Path, help='Frozen initial particle arrays NPZ')
    parser.add_argument('--initial-x', required=True, type=float, help='Frame-zero x in source0 coordinates')
    parser.add_argument('--initial-y', required=True, type=float, help='Frame-zero y in source0 coordinates')
    parser.add_argument('--frame-size', nargs=2, type=int, metavar=('WIDTH','HEIGHT'), default=[256,256],
                        help='Fallback size only for legacy NPZ without frame_size')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/final_candidate_output_v1.json')
    parser.add_argument('--output', required=True, type=Path, help='New, nonexistent result directory')
    parser.add_argument('--device', default='cpu', help='Torch device for image inference, e.g. cpu or cuda')
    args = parser.parse_args(argv)
    if args.output.exists(): raise FileExistsError('Output already exists; choose a new directory')
    if args.frames is not None and args.checkpoint is None: parser.error('--frames requires --checkpoint')
    if args.candidates is not None and args.checkpoint is not None: parser.error('--checkpoint is only used with --frames')
    config_data = json.loads(args.config.read_text(encoding='utf-8'))
    cfg = base.Config(**config_data['engine'])
    if not config_data.get('history_dispersion_enabled', True):
        raise ValueError('This release requires the adopted history dispersion maintenance')
    initial_xy = [args.initial_x, args.initial_y]
    initial = read_initial_state(args.initial_state, tuple(reversed(args.frame_size)), initial_xy, cfg.total_particles)
    groups = read_candidate_groups(args.candidates, initial.frame_size) if args.candidates is not None else None
    paths = image_paths(args.frames) if args.frames is not None else None
    frontend = Frontend(args.checkpoint, args.device) if paths is not None else None
    if paths is not None:
        from PIL import Image
        from deep_observation.inference import gray_uint8
        # Validate the first frame's geometry; its supplied state is not reinitialized.
        with Image.open(paths[0]) as im: first = gray_uint8(np.asarray(im).copy())
        if first.shape != initial.frame_size: raise ValueError('Image dimensions differ from initial state')
    args.output.mkdir(parents=True, exist_ok=False)
    tracker = base.Tracker(base.Branch(0, initial))
    columns = ['frame_id','initialized','x','y','action','output_source','candidate_id',
               'io_seconds','preprocess_seconds','filter_seconds']
    count = len(paths) if paths is not None else len(groups) + 1
    started = time.perf_counter()
    with (args.output / 'predictions.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, columns); writer.writeheader()
        writer.writerow(dict(frame_id=0, initialized=1, x=initial_xy[0], y=initial_xy[1], action='initialize',
                             output_source='H', candidate_id='', io_seconds=0, preprocess_seconds=0, filter_seconds=0))
        for frame_id in range(1, count):
            io_seconds = preprocess_seconds = 0.
            if paths is not None:
                tick = time.perf_counter()
                with Image.open(paths[frame_id]) as im: raw = np.asarray(im).copy()
                io_seconds = time.perf_counter() - tick
                tick = time.perf_counter(); gray = gray_uint8(raw)
                if gray.shape != initial.frame_size: raise ValueError('Image dimensions changed')
                rows = frontend.infer(gray); preprocess_seconds = time.perf_counter() - tick
            else:
                rows = groups[frame_id - 1]['candidates']
            # This is the original S27 source0-to-MATLAB1 conversion.
            candidates = [base.Candidate(r['candidate_id'], tuple(np.asarray(r['center_xy']) + 1), r['image_score']) for r in rows]
            tick = time.perf_counter()
            tracker, result = candidate_position_output.step(tracker, candidates, frame_id, cfg, enabled=True)
            elapsed = time.perf_counter() - tick
            decision = result.candidate_result
            role = {'single_update':'H', 'confirm_H':'H', 'promote_T':'T'}.get(decision.action)
            candidate_id = decision.accepted_candidate[role].candidate_id if role else ''
            xy = result.output_xy - 1
            if not np.isfinite(xy).all(): raise ValueError('Nonfinite prediction')
            writer.writerow(dict(frame_id=frame_id, initialized=0, x=float(xy[0]), y=float(xy[1]),
                                 action=decision.action, output_source=result.output_source, candidate_id=candidate_id,
                                 io_seconds=io_seconds, preprocess_seconds=preprocess_seconds, filter_seconds=elapsed))
    inputs = dict(config_sha256=sha256(args.config), initial_state_sha256=sha256(args.initial_state))
    if args.candidates is not None: inputs['candidates_sha256'] = sha256(args.candidates)
    if args.checkpoint is not None: inputs['checkpoint_sha256'] = sha256(args.checkpoint)
    metadata = dict(mode='images' if paths is not None else 'candidate_replay', frame_count=count,
                    initial_center_source0=initial_xy, coordinate_origin=0, frame_size=list(initial.frame_size),
                    engine=config_data['engine'], inputs=inputs, seconds=time.perf_counter()-started,
                    initialization='Provided frozen particle state; no replacement amplitude or observation',
                    timing='Per-frame times exclude model loading and initial-state loading',
                    numpy_version=np.__version__, python_version=sys.version.split()[0])
    if frontend is not None: metadata.update(torch_version=frontend.torch.__version__, device=str(frontend.device))
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'Wrote {count} frames to {args.output / "predictions.csv"}')
    return metadata


if __name__ == '__main__':
    main()
