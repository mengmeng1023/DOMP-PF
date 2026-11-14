# Source and asset provenance

The source package contains the project's implementations of the DOMP-PF tracker, its final U-Net frontend and dense-objectness candidate extraction, plus release-specific inference and evaluation interfaces. Source provenance is distinct from a grant of rights: the project's review-use conditions are described in the repository license, while independently installed third-party dependencies remain subject to their own licenses.

## Tracking implementation

The final candidate-output mechanism is associated with source commit `c0d3dd17010f000399fffc522c3bef75daac8e81` and these original source paths:

- `src/pf_tbd/core.py`
- `src/pf_tbd/candidate_closed_loop.py`
- `src/pf_tbd/history_dispersion_guard.py`
- `src/pf_tbd/candidate_position_output.py`
- `configs/final_candidate_output_v1.json`

The Python particle-filter core was rewritten from the authors' existing MATLAB `tp_*.m` implementation. The project records this migration separately from the subsequent candidate-based tracking changes. The historical MATLAB programs, their old video runners, and their original workspace are not required by the release and are not bundled.

The final output rule reports an accepted candidate center for `single_update`, `confirm_H`, and `promote_T`, while preserving the existing output for unresolved decisions. The frozen configuration sets `K=8` and enables history-dispersion maintenance; historical low-level defaults must not replace that explicit configuration. Internal tracker coordinates are continuous MATLAB1 coordinates. Conversion to source-zero coordinates occurs at the external reporting boundary.

## Frontend implementation

The final frontend and objectness extraction come from source commit `aec63f93367bfd2f8255493d0660d8820a9fadf6`. Relevant original paths are `src/deep_observation/dense_objectness.py`, `src/deep_observation/models/deeper_unet.py`, `src/deep_observation/models/shallow_unet.py`, and `scripts/run_s11_dense.py`. Architecture, training, selection, and checkpoint details are in [TRAINING.md](TRAINING.md).

The network is a project implementation of a U-Net-style architecture. This release does not bundle DNANet, FocusTrack, or SiamDT source code or their pretrained weights. Those methods' implementation and license terms are not covered by the project license. The release's reduced model entry point selects the required frontend without importing the exploratory project model registry.

## Third-party dependencies

PyTorch, NumPy, SciPy, and Pillow are installed separately; their source distributions are not vendored in this package. Any additional development or packaging tools are likewise independent dependencies. The repository's restrictions on modification or redistribution of project-owned material do not restrict rights granted by these dependencies' own licenses.

The historical development repository also contains a DNANet derivative under its upstream MIT license. It is outside this release's selected implementation. Its permissive terms must be retained if that implementation is later redistributed; the project's review-use restrictions cannot replace its upstream license.

## Data and model assets

This source package does not redistribute NUDT-SIRST, MovingTarget, their image files or full annotations, or third-party baseline checkpoints. Source code does not confer permission to redistribute the data used with it. Obtain data through an authorized source and follow the dataset owner's terms.

The NUDT-SIRST copy used for frontend training was obtained from the academic mirror [SCTransNet_Jittor](https://github.com/TianyiHuang-cs/SCTransNet_Jittor), fixed at commit `0c308052020b29e50e476d4aea29ab38270327af`. This identifies the copy used; it does not establish byte-for-byte identity with an independently obtained original-author archive or a right to redistribute the dataset. The original public release source of the local MovingTarget collection has not been verified, so this release supplies no claimed official download link for it.

The final frontend checkpoint was trained within this project rather than downloaded as a third-party pretrained model. Its original hash is recorded in [TRAINING.md](TRAINING.md). Review assets, when supplied separately, should carry the original and exported artifact hashes and their permitted access terms. Extracting only tensor state removes training-host metadata but does not by itself establish data or weight redistribution permission.

All source commits above identify historical development revisions. Release wrappers, relative-path handling, and packaging files are distinct from those frozen algorithm sources. Consult the release manifest for the exact file set and hashes of the assembled package.
