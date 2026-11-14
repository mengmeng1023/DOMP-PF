# Frontend training provenance

This release supports inference, particle tracking, and offline evaluation. It includes the final network definition and dense-objectness loss for inspection. It does not include a complete, portable retraining entry point. The settings below describe the training run that produced the manuscript's frontend weights; they are not a claim that retraining has been repeated from this release.

## Model and input

The frontend is the project's `DeeperUNet`: three spatial reductions, channel widths 16, 32, 64, and 128, two 3-by-3 convolutions with ReLU per block, full-resolution skip connections, and a one-channel 1-by-1 output head. Upsampling uses bilinear interpolation with `align_corners=False`; the model has no normalization layers. It has 487,009 trainable parameters and returns logits.

Inputs are native-resolution uint8 images converted to grayscale and then float32 values divided by 255. Images are padded on the bottom and right, using replication, to spatial multiples of 16. Outputs are cropped back to the original dimensions. A single sigmoid produces the dense objectness map in [0, 1]. The final objectness path does not multiply that map by 255 before candidate extraction.

## Data split

Training used NUDT-SIRST image-mask pairs. The 663-image training list was split into 530 training and 133 validation images; the separate 664-image test list was not used for optimization or checkpoint selection. Masks were required to have uint8 values in {0, 255} and were converted to binary foreground labels.

The original manifest is `configs/splits/s2_frontend_static.json`, SHA-256 `92878e4cf74a8a32461f4f1ca225f21e6ebc5265b054387d571f3165c5a0d118`. Its split procedure groups identical decoded images and any supplied common-source groups, sorts training groups by SHA-256 of `s2-static-v1|` followed by the group ID, and selects the closest nonempty prefix to 20% for validation. No per-image common-source mapping was supplied. These grouping checks do not establish independence of unknown image sources.

Each pass uses a shuffled permutation of the training IDs. Horizontal flips, vertical flips, and rotations by 0, 90, 180, or 270 degrees are sampled and applied consistently to the input and mask. MovingTarget images do not enter this frontend training or its checkpoint-selection criterion. The tracking sequences were used for method development, so they should not be described as an independently held-out test set.

## Objective and optimization

The final run is the dense-objectness arm, rather than the earlier residual-intensity arm. For each image, binary cross-entropy is averaged separately over foreground and background pixels and the two region means receive equal weight when both regions exist. A per-image SoftIoU loss is added, using epsilon `1e-6`. For an empty foreground, only background BCE is used and the IoU term is zero. For an all-foreground mask, foreground BCE and SoftIoU are used. The batch loss averages these per-image losses.

| Setting | Recorded value |
| --- | --- |
| Parameter/RNG seed | 7 |
| Data-order and augmentation seed | 20260919 |
| Batch size | 8 |
| Parameter and input dtype | float32 |
| Optimizer | AdamW |
| Initial learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Validation interval | 250 updates |
| Recovery-checkpoint interval | 2,000 updates |
| Maximum updates | 20,000 |
| Minimum updates before early stopping | 4,000 |
| LR scheduler | ReduceLROnPlateau on validation loss, mode `min` |
| LR reduction factor / patience | 0.5 / 3 validation checks |
| Relative improvement threshold | 0.005 |
| Minimum learning rate | 0.000003125 |
| Early-stop patience at minimum LR | 8 nonimproving validation checks |
| Plateau window / maximum relative range | 4 validation losses / 0.02 |

Convolution weights were initialized using Kaiming-normal initialization and zero biases. The output-head bias was then set to the previously frozen, training-only value `-1.9757769107818604`. This bias was not recalibrated on validation images, test images, or MovingTarget. cuDNN benchmarking and TF32 were disabled, and cuDNN deterministic behavior was enabled in the recorded training configuration.

Training stopped at update 16,000 under the specified validation-plateau rule. The deployed checkpoint is from update **5,500**, selected by validation target-component recall within 5 pixels: maximize Top-20 recall, then Top-5 recall, then Top-1 recall; choose the earliest update on complete ties. Thus it is not the minimum-validation-loss checkpoint, which occurred at update 7,000.

## Checkpoint identity

The original training artifact was `runs/RD-PFTBD-S11-OBJECTNESS-20261007-001/best.pt`:

- Size: 1,979,209 bytes.
- Training update: 5,500.
- SHA-256: `0ca7683939d6b07bb9373fbee181d3105fedfca43716c9f30129c6c462148d0f`.

The original artifact contains a `state_dict` and training metadata. A separately prepared state-dictionary export has a different file hash even when its tensor values are identical. Use the release's asset metadata to check the exported file; do not compare an exported file's hash against the original-checkpoint hash above. Data and model assets are handled separately from this source-code release, and this document does not grant redistribution rights for them.

## Source revision

The recorded frontend source commit is `aec63f93367bfd2f8255493d0660d8820a9fadf6`. Relevant paths in that source revision are:

- `src/deep_observation/dense_objectness.py`: loss, score smoothing, candidate extraction, validation matching, and checkpoint ranking.
- `src/deep_observation/models/deeper_unet.py` and `shallow_unet.py`: architecture and shared convolution block.
- `scripts/run_s11_dense.py`: final objective, padded/cropped prediction, validation, and training loop.
- `scripts/train_frontend_full.py`: initialization, optimizer, scheduler, early stopping, and reproducible batch-stream state.
- `scripts/train_frontend.py`, `src/deep_observation/data.py`, and `targets.py`: batching, split construction, input conversion, and geometric augmentation.

Recorded environment versions were Python 3.11.16, PyTorch 2.13.0+cu130, NumPy 2.4.6, SciPy 1.17.1, and Pillow 12.3.0. The original training scripts also depended on project-specific experiment-management code. Their historical paths and settings above document provenance and do not constitute runnable commands in this release.
