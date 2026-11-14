# DOMP-PF

Code accompanying **DOMP-PF: A Deep Observation–Motion Prior Coupled Particle Filter for Infrared Dim Small Target Tracking**, by Mengchu Tian, Huanyu Yang, Zhimin Chen, and Zhongtan Zhang.

This submission review release provides the dense objectness frontend, motion-prior candidate screening and ranking, dual-path particle filtering, candidate-position output, and offline evaluation. The tracking core is copied unchanged from the implementation used for the manuscript. The portable command-line interfaces replace local experiment-launch infrastructure.

## Permission

This is source-available code for manuscript review and result verification, under [LICENSE.txt](LICENSE.txt). Viewing and running the unmodified software for that purpose are permitted. Source modification, derivative versions, redistribution outside review, and commercial use require prior written permission. Independently installed dependencies retain their own licenses.

Corresponding author: Huanyu Yang, yhy@njust.edu.cn.

## Contents

- `src/pf_tbd/`: particle filtering, candidate support, path decisions, history-dispersion control, and final candidate-position output.
- `src/deep_observation/`: DeeperUNet, dense objectness loss and candidate extraction. The model package imports only the required architecture.
- `scripts/track.py`: image-based tracking or replay of precomputed candidates.
- `scripts/evaluate.py`: offline center-error evaluation, kept separate from tracking.
- `configs/final_candidate_output_v1.json`: explicit manuscript parameters, including `K=8` and 500 particles.
- `configs/paper_sequences.json`: sequence counts, supplied first-frame centers, and annotation checksums.
- `results/paper_metrics.csv`: recorded per-sequence DOMP-PF results, not results newly computed by this release.
- `tests/`: synthetic core and interface checks; no real dataset is included.
- `docs/TRAINING.md`: frozen frontend training details and the scope of this release.
- `SOURCE_MANIFEST.json`: provenance and checksums of unchanged source files.

## Environment

The recorded numerical environment uses Linux/WSL, Python 3.11, NumPy 2.4.6, SciPy 1.17.1, Pillow 12.3.0, and PyTorch 2.13.0. The research GPU environment used the CUDA 13.0 PyTorch build. A CPU device is also supported for inference. Different library versions, devices and floating-point kernels can change numerical results; the recorded versions identify the original environment.

In a separate Python environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-torch.txt
```

Select the PyTorch wheel appropriate for the CPU/CUDA platform. No package installation is performed by the tracking scripts. The scripts resolve `src/` from their own location and do not require an editable package install.

## Data and frozen assets

Obtain the MovingTarget image sequences and original annotation files through an authorized source. This repository does not distribute images or labels. It also does not invent a dataset download URL. Keep the original frame order and pixel encoding. `configs/paper_sequences.json` specifies the four sequence counts and the **zero-based** initial centers used in the manuscript.

The separately prepared `DOMP-PF-assets` directory contains:

- `domp_pf_frontend.pt`: the S11 dense objectness checkpoint selected at step 5500. Parameter tensors are unchanged; internal machine/training metadata is omitted.
- `data5_initial.npz`, `data10_initial.npz`, `data18_initial.npz`, `data20_initial.npz`: frozen initial particles, weights and log weights.
- `ASSETS.json`: exported and original checksums.

These assets are intentionally separate from the source archive. The author can provide them through the journal-approved review channel. The source license does not grant redistribution rights for datasets or binary assets.

The original frontend checkpoint SHA-256 is `0ca7683939d6b07bb9373fbee181d3105fedfca43716c9f30129c6c462148d0f`. Use the exported checksum in `ASSETS.json` for the metadata-cleaned file.

The stored initial particle states matter: the original initialization used an earlier calibrated observation for particle amplitudes. Replacing it with the S11 score map would not reproduce the complete historical state. The supplied initialization files preserve the original values and avoid needing the old baseline models. First-frame target coordinates are an explicit tracking input; later annotations are only read by the offline evaluator.

## Track a sequence

Example for `data5`, run from this repository directory:

```bash
python scripts/track.py \
  --frames /path/to/MovingTarget/data5 \
  --checkpoint ../DOMP-PF-assets/domp_pf_frontend.pt \
  --initial-state ../DOMP-PF-assets/data5_initial.npz \
  --initial-x 71 --initial-y 109 \
  --device cpu \
  --output outputs/data5
```

Use `--device cuda` with a compatible GPU installation. Image filenames must be numeric frame IDs `0..N-1` with no gaps (for example `0.bmp`, `1.bmp`, ...); ordering is numeric, not lexicographic. The output directory must not already exist, so an earlier run cannot be silently overwritten. Run `python scripts/track.py --help` for the full interface. The first input frame is initialization and is included in the output timeline. Frozen-candidate replay can reproduce historical PF outputs exactly in the recorded environment; image inference on other devices can introduce floating-point differences.

For candidate replay, replace `--frames` and `--checkpoint` with `--candidates /path/to/candidates.json` and use the same initial state and position. Candidate centers and output CSV coordinates are zero-based; the core converts to its continuous one-based convention internally. The candidate file contains frames 1 onward; the initial frame is represented by the explicit initialization inputs.

The final output is a selected candidate center only for the accepted actions `single_update`, `confirm_H`, and `promote_T`. Unconfirmed alternative-path updates do not directly replace the output path.

## Evaluate offline

Tracking itself never loads the annotation file. The release evaluator excludes the first initialization frame and requires a complete contiguous timeline with one valid, continuously identified target on every frame, as in the four paper sequences. Malformed, absent-target or multiple-target labels are rejected.

```bash
python scripts/evaluate.py \
  --predictions outputs/data5/predictions.csv \
  --labels /path/to/MovingTarget/data_label/data5.txt \
  --output outputs/data5-eval

python scripts/evaluate.py --metrics \
  outputs/data5-eval/metrics.json outputs/data10-eval/metrics.json \
  outputs/data18-eval/metrics.json outputs/data20-eval/metrics.json \
  --output outputs/macro
```

The manuscript reports mean center error, the fraction of frames within 5 pixels, and sustained deviations. A sustained deviation is an interval of **at least five consecutive evaluable frames with center error strictly greater than 10 pixels**. The main summary averages per-sequence metrics equally; it does not pool all frames across sequences.

The recorded evidence covers `data5`, `data10`, `data18`, and `data20`, seed 7, with 4,297 evaluated frames after excluding the four initialization frames. The per-sequence macro averages are 3.1729545869 pixels mean center error, 84.4402% within 5 pixels, and 8.6778% sustained-deviation frames. These results concern those four development sequences and that seed; they are not a multi-seed or unseen-test-set claim.

## Checks

From the repository root on Linux/WSL:

```bash
PYTHONPATH=src:scripts PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest discover -s tests -v
```

Tests use synthetic inputs, not the original data. Some interface tests retain small temporary fixtures for inspection. Test success checks implementation consistency; it does not replace full-sequence reproduction using the original images, assets and evaluation protocol.

The frontend architecture, loss and documented recipe are included. A portable full training launcher and the third-party comparison trackers are outside this focused inference/evaluation release; see [training details](docs/TRAINING.md) and [source notes](docs/SOURCES.md).
