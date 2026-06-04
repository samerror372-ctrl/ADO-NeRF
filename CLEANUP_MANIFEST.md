# Cleanup Manifest

This directory is a non-destructive clean copy prepared from the parent workspace. The parent workspace was not modified.

## Kept

- Training, evaluation, dataset, network, and utility source code.
- ADO export bridge: `run_ado_export.py`.
- Diffusion refinement entry points: `diffusion_nerf/refine_ado_scene_consistent.py` and `diffusion_nerf/refine_ado_frequency_blending.py`.
- DTU split and source-view pairing metadata under `data/ado_metadata/`.
- Project license, requirements, README, and Git ignore rules.

## Removed

- Python caches.
- Training logs and TensorBoard records.
- Rendered results, debug images, exported diffusion tensors, and temporary reports.
- NeRF checkpoints and diffusion checkpoints.
- Local absolute data/checkpoint symlinks.
- Machine-specific run notes.

## Renamed

- `networks/ado_nerf/` is the project network package.
- `evaluators/ado_nerf.py` is the evaluator module.
- `train/losses/ado_nerf.py` is the training wrapper and loss module.
- `data/ado_metadata/` stores lightweight split and pairing metadata.

## Runtime Assets

For local runs, provide datasets and checkpoints outside version control using the paths documented in `README.md`.
