# ADO-NeRF

Adaptive Diffusion Refinement for Occlusion-Aware Generalizable Neural Radiance Fields.

This repository contains a clean research-code release of ADO-NeRF: the renderer, dataset loaders, training and evaluation configs, the ADO export bridge, and the diffusion-based refinement scripts. Datasets, logs, rendered results, exported tensors, model weights, and diffusion checkpoints are intentionally not tracked.

## Highlights

- Generalizable NeRF training and evaluation on DTU, NeRF Synthetic, and LLFF-style data.
- Occlusion-aware target-view export for diffusion repair.
- Two refinement entry points: scene-consistent ADO diffusion repair and frequency-blending repair.
- Optional MVSGaussian predicted-source bridge for cross-model diffusion repair studies.
- Clean repository layout with generated data and checkpoints excluded by `.gitignore`.
- Tested with Python 3.10, PyTorch 2.5.1, and CUDA 11.8 wheels.

## Installation

### Recommended Versions

The pinned requirements were generated from a working Python 3.10 / CUDA 11.8 environment:

```text
Python 3.10.12
torch 2.5.1+cu118
```

### Create a New Environment

```bash
conda create -n ado-nerf python=3.10 -y
conda activate ado-nerf

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

`requirements.txt` pins the Python packages used by this release, including CUDA 11.8 PyTorch wheels and the diffusion, metric, rendering, and data-processing dependencies. If `nvdiffrast` fails to build, first check that `git`, a C++ compiler, and a CUDA-compatible NVIDIA driver are available.

The code reads datasets from `cfg.workspace / data_root`. If `workspace` is not set, it defaults to the repository root:

```bash
export workspace=/path/to/workspace
```

For a single-repo setup, keep `workspace=$(pwd)` and place or symlink datasets under the repository root.

## Datasets

ADO-NeRF does not redistribute datasets. The included files under `data/ado_metadata/` provide split and source-view metadata used by the loaders.

### DTU

The DTU loader expects the MVSNet/ENeRF-style preprocessed DTU layout:

```text
dtu/
  Cameras/train/00000000_cam.txt
  Rectified/scan1_train/rect_001_3_r5000.png
  Depths/scan1_train/depth_map_0000.pfm
```

`Depths_raw/<scene>/`, `Depths/<scene>/`, `Depths_raw/<scene>_train/`, and `Depths/<scene>_train/` are all supported for target-view depth maps. These maps are used only for training supervision and valid-mask evaluation. Source-view geometry for diffusion refinement is predicted by the ADO/GBD MVS network and is never read from these files.

Public data sources:

- Official DTU MVS dataset page: https://roboimagedata.compute.dtu.dk/?page_id=36
- ENeRF dataset preparation reference: https://github.com/zju3dv/ENeRF
- GDB-NeRF dataset preparation reference: https://github.com/KLMAV-CUC/GDB-NeRF

The official DTU page provides the original dataset. This code expects the preprocessed layout shown above, so raw official downloads must be converted or reorganized before training.

Prepared DTU archives used by ENeRF-style code can be downloaded with `gdown`:

```bash
python -m pip install gdown

gdown https://drive.google.com/uc?id=1eDjh8tCzvpI24vD-bJsZ3TcIa_gBUc_K
gdown https://drive.google.com/uc?id=1G-8R1YkReWbX8F_eCkGQ3vXl7vYl8prp
```

After extraction, place or symlink the prepared folder as:

```bash
ln -s /path/to/preprocessed_dtu dtu
```

Verify the dataset path before running training:

```bash
test -f dtu/Cameras/train/00000000_cam.txt
test -f dtu/Rectified/scan1_train/rect_001_3_r5000.png
test -f dtu/Depths/scan1_train/depth_map_0000.pfm
```

### NeRF Synthetic

The NeRF Synthetic loader expects:

```text
nerf_synthetic/
  lego/transforms_train.json
  lego/train/r_0.png
```

Download source, following the ENeRF/GDB-NeRF dataset convention:

```bash
gdown https://drive.google.com/uc?id=128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
```

Extract the archive so the final directory is `nerf_synthetic/`.

### LLFF

The LLFF loader expects:

```text
nerf_llff_data/
  room/poses_bounds.npy
  room/images_4/*.png
```

Download source, following the ENeRF/GDB-NeRF dataset convention:

```bash
gdown https://drive.google.com/uc?id=1boZI9uMuJxK1JHFMrX_WZhFyUbauH7iO
```

Extract the archive so the final directory is `nerf_llff_data/`.

## Diffusion Checkpoints

Diffusion refinement requires two local checkpoint folders:

```text
diffusion_nerf/check_points/
  pretrained_model/
    config.yaml
    ema_unet.pt
    ratio_set.json
  Stable-Diffusion-2-1-base/
    model_index.json
    scheduler/
    text_encoder/
    tokenizer/
    unet/
    vae/
```

The refinement scripts use `local_files_only=True`, so all files must exist locally before running repair.

### Scene-Consistent Diffusion Checkpoint

The scene-consistent diffusion checkpoint follows the GaMO/MVGenMaster release:

- GaMO official repository: https://github.com/yichuanH/GaMO_official
- MVGenMaster checkpoint host: https://huggingface.co/ewrfcas/MVGenMaster

Download and extract:

```bash
mkdir -p diffusion_nerf/check_points

wget "https://huggingface.co/ewrfcas/MVGenMaster/resolve/main/check_points/pretrained_model.zip?download=true" \
  -O diffusion_nerf/check_points/pretrained_model.zip

unzip -q diffusion_nerf/check_points/pretrained_model.zip \
  -d diffusion_nerf/check_points
```

### Stable Diffusion 2.1 Base

Official model page:

- https://huggingface.co/stabilityai/stable-diffusion-2-1-base

Use the Python API instead of relying on `huggingface-cli`; this avoids stale script shebangs in relocated conda environments:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="stabilityai/stable-diffusion-2-1-base",
    local_dir="diffusion_nerf/check_points/Stable-Diffusion-2-1-base",
)
PY
```

If your Hugging Face account does not have access to the official repository, set `HF_TOKEN` after accepting the model license, or use the GaMO-compatible mirror:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Manojb/stable-diffusion-2-1-base",
    local_dir="diffusion_nerf/check_points/Stable-Diffusion-2-1-base",
)
PY
```

## Training

### DTU Pretraining

Train a generalizable DTU model from scratch:

```bash
python train_net.py \
  --cfg_file configs/dtu_pretrain.yaml \
  exp_name ado_dtu_pretrain \
  train.pretrain "" \
  resume False
```

Checkpoints are saved under:

```text
model_result/ado_nerf/ado_dtu_pretrain/
record/ado_nerf/ado_dtu_pretrain/
```

### DTU Single-Scene Fine-Tuning

Fine-tune on one DTU scene, for example `scan1`, from a previously trained renderer checkpoint:

```bash
python train_net.py \
  --cfg_file configs/dtu_pretrain.yaml \
  exp_name ado_ft_scan1 \
  train_dataset.scene scan1 \
  test_dataset.scene scan1 \
  train.pretrain ado_dtu_pretrain \
  train.epoch 233 \
  resume False
```

`train.pretrain ado_dtu_pretrain` loads `model_result/ado_nerf/ado_dtu_pretrain/latest.pth` or the highest numbered `.pth` in that folder.

### NeRF Synthetic and LLFF Fine-Tuning

```bash
python train_net.py --cfg_file configs/nerf/lego.yaml resume False
python train_net.py --cfg_file configs/llff/room.yaml resume False
```

Other scene configs are available under `configs/nerf/` and `configs/llff/`.

## Evaluation and ADO Export

### Evaluation

Evaluate the latest checkpoint of an experiment:

```bash
python run.py \
  --type evaluate \
  --cfg_file configs/dtu_eval.yaml \
  exp_name ado_ft_scan1 \
  test_dataset.scene scan1 \
  test.eval_depth True
```

Rendered images and metric summaries are written under `result/ado_nerf/<exp_name>/`.

### Export for Diffusion Repair

Export rendered target views, source RGBs, source depths, masks, and camera tensors:

```bash
ADO_EXPORT_OUTPUT_DIR=diffusion_input \
ADO_EXPORT_MAX_BATCHES=0 \
python run_ado_export.py \
  --type evaluate \
  --cfg_file configs/dtu_eval.yaml \
  exp_name ado_ft_scan1 \
  test_dataset.scene scan1
```

Each item is written as:

```text
diffusion_input/<scene>_view_<id>_ado_data.pt
```

Every exported source view has its own MVS depth prediction from the same ADO/GBD network. The exporter reruns the depth network with each source camera as the reference and writes:

```text
source_mvs_depths
source_geometry_source = model_predicted_src_mvs_depth
```

`source_depths` is retained only as an identical compatibility alias. Dataset/raw source depth and repeated target-depth fallbacks are not allowed. Evaluation mode always enables per-source depth prediction, independently of the training warm-up threshold.

Useful export environment variables:

```text
ADO_EXPORT_OUTPUT_DIR       Output directory for exported tensors
ADO_EXPORT_MAX_BATCHES      Number of batches to export; 0 means all
ADO_MASK_PROFILE            Mask profile name
ADO_USE_TUNED_MASK          Set to 0 for the pre-tuning mask profile
```

### Mask Setting Guide

The mask is generated during ADO export and saved into each `*_ado_data.pt` file. If you change mask settings, re-run `run_ado_export.py` before running diffusion repair again.

Choose the mask profile according to your target:

| Goal | Recommended setting | Notes |
| --- | --- | --- |
| Better visual quality | `ADO_USE_TUNED_MASK=0` | Uses the original mask profile. It is usually smoother and less aggressive. |
| Better quantitative metrics | `ADO_USE_TUNED_MASK=1` | Uses the default recomposed mask profile. It is usually more metric-oriented. |
| Ablation or custom tuning | `ADO_MASK_PROFILE=<name>` | Explicit profile selection overrides `ADO_USE_TUNED_MASK`. |

For visually smoother and more natural repair results, use the original mask profile:

```bash
ADO_USE_TUNED_MASK=0
```

For stronger quantitative metrics, the default internal recomposed mask is recommended:

```bash
ADO_USE_TUNED_MASK=1
```

You can also choose a profile explicitly:

```bash
ADO_MASK_PROFILE=orig_s0p50
ADO_MASK_PROFILE=grid_l1_0p10_l2_0p25_l3_0p10_s0p50
```

Example visual-first export:

```bash
ADO_USE_TUNED_MASK=0 \
ADO_EXPORT_OUTPUT_DIR=diffusion_input_visual \
python run_ado_export.py \
  --type evaluate \
  --cfg_file configs/dtu_eval.yaml \
  exp_name ado_ft_scan1 \
  test_dataset.scene scan1
```

Example metric-first export:

```bash
ADO_USE_TUNED_MASK=1 \
ADO_EXPORT_OUTPUT_DIR=diffusion_input_metric \
python run_ado_export.py \
  --type evaluate \
  --cfg_file configs/dtu_eval.yaml \
  exp_name ado_ft_scan1 \
  test_dataset.scene scan1
```

For custom masks, edit `MASK_STRENGTH_PROFILES` in `run_ado_export.py`. In the recomposed profile, the `w1`, `w2`, and `w3` values control the contribution of the exported mask layers, and `post_scale` controls the final global strength. If repair is too aggressive or causes color/detail drift, reduce `post_scale` first. If unrepaired artifacts remain, increase `post_scale` gradually, then tune the layer weights. Mask recomposition and post-scaling are model- and dataset-sensitive; for other models in the same family, these values may need to be explored manually.

## Diffusion Repair

Run scene-consistent ADO diffusion repair:

```bash
ADO_REFINE_INPUT_DIR=diffusion_input \
ADO_REFINE_OUTPUT_DIR=diffusion_nerf/ado_result_scene \
ADO_REFINE_DEBUG_DIR=diffusion_nerf/ado_debug_scene \
ADO_REFINE_MAX_FILES=0 \
ADO_DIFFUSION_STEPS=50 \
python diffusion_nerf/refine_ado_scene_consistent.py
```

Run frequency-blending ADO diffusion repair:

```bash
ADO_REFINE_INPUT_DIR=diffusion_input \
ADO_REFINE_OUTPUT_DIR=diffusion_nerf/ado_result_frequency \
ADO_REFINE_DEBUG_DIR=diffusion_nerf/ado_debug_frequency \
ADO_REFINE_MAX_FILES=0 \
ADO_DIFFUSION_STEPS=50 \
python diffusion_nerf/refine_ado_frequency_blending.py
```

Both refinement commands validate source-depth provenance. Legacy `*_ado_data.pt` files that contain only `source_depths`, or do not carry the required `source_geometry_source` tag, are rejected and must be regenerated with `run_ado_export.py`.

Useful refinement environment variables:

```text
ADO_REFINE_INPUT_DIR         Directory containing *_ado_data.pt files
ADO_REFINE_OUTPUT_DIR        Output directory for refined images and reports
ADO_REFINE_DEBUG_DIR         Debug-output directory
ADO_REFINE_MAX_FILES         Number of input files to refine; 0 means all
ADO_DIFFUSION_STEPS          Diffusion sampling steps
ADO_GUIDANCE_SCALE           Classifier-free guidance scale
ADO_REPAIR_MASK_THRESHOLD    Threshold for repair-mask metrics
ADO_ENABLE_PASTE             Enable pixel-prior paste when set to 1
```

## Cross-Model MVSGaussian Bridge

This repository also includes an optional migration example for applying the same scene-consistent diffusion repair idea to MVSGaussian:

```text
optional_projects/mvsgaussian_predicted_source_bridge/
```

The bridge is intentionally based on predicted source-view MVS depths. It does not use raw or ground-truth source-view MVS depth maps as diffusion conditions. See `optional_projects/mvsgaussian_predicted_source_bridge/README.md` for copy commands and MVSGaussian-side usage.

At a high level, copy the bridge files into an MVSGaussian checkout, export predicted-source diffusion tensors, then run the scene-consistent refinement script:

```bash
cd /path/to/MVSGaussian
export workspace=$(pwd)

MVSGS_DIFFUSION_OUTPUT_DIR=diffusion_nerf/mvsgs_predicted_source_inputs \
python export_mvsgs_predicted_source_depth.py \
  --type evaluate \
  --cfg_file configs/mvsgs/dtu_pretrain.yaml \
  exp_name <mvsgs_experiment_name> \
  test.epoch <checkpoint_epoch>

cd diffusion_nerf
python refine_mvsgs_scene_consistent.py \
  --input_dir mvsgs_predicted_source_inputs \
  --experiment_root mvsgs_scene_consistent_results \
  --depth_mode source \
  --mask_profile orig
```

### Required Outputs for MVSGaussian-Style Migration

The checked MVSGaussian network already provides the target-view fields used by the bridge:

```text
rgb_level{level}
depth_mvs_level{level}
std_level{level}
```

The bridge adds source-view predicted depth fields during export by running extra depth-only passes with each source camera as the target:

```text
src_mvs_depths_level{level}
src_mvs_stds_level{level}        optional
source_depth_origin
uses_source_raw_depth=False
```

For another model in the same family, the export side should provide target rendered RGB, target MVS depth, source RGBs, camera intrinsics/extrinsics, a repair mask, and per-source predicted MVS depths. The per-source depth condition should come from the model's own prediction path, not from dataset raw source-depth supervision.

## Repository Layout

```text
configs/                    Training and evaluation configs
data/ado_metadata/          Split files and source-view metadata
datasets/                   DTU, LLFF, and NeRF Synthetic loaders
networks/ado_nerf/          Renderer, depth modules, and bundle sampling
train/                      Losses, optimizer, recorder, and train loop
evaluators/                 Evaluation metrics and output writers
diffusion_nerf/             Diffusion repair scripts and local modules
optional_projects/          Optional cross-model migration examples
run.py                      Dataset, network, and evaluation entry point
train_net.py                Training entry point
run_ado_export.py           Renderer-to-diffusion export bridge
```

## Generated Files

The following paths are local artifacts and are ignored by Git:

```text
dtu/
nerf_synthetic/
nerf_llff_data/
model_result/
trained_model/
record/
result/
diffusion_input/
diffusion_nerf/check_points/
diffusion_nerf/ado_input*/
diffusion_nerf/ado_result*/
diffusion_nerf/ado_debug*/
diffusion_nerf/logs/
diffusion_nerf/run_logs/
optional_projects/**/diffusion_nerf/check_points/
optional_projects/**/diffusion_nerf/*inputs*/
optional_projects/**/diffusion_nerf/*results*/
```

## References

- DTU MVS dataset: https://roboimagedata.compute.dtu.dk/?page_id=36
- ENeRF dataset and setup reference: https://github.com/zju3dv/ENeRF
- GDB-NeRF setup reference: https://github.com/KLMAV-CUC/GDB-NeRF
- GaMO diffusion checkpoint reference: https://github.com/yichuanH/GaMO_official
- MVGenMaster checkpoint host: https://huggingface.co/ewrfcas/MVGenMaster
- Stable Diffusion 2.1 base: https://huggingface.co/stabilityai/stable-diffusion-2-1-base
- Stable Diffusion 2.1 base mirror used by GaMO-style setups: https://huggingface.co/Manojb/stable-diffusion-2-1-base

## License

Unless otherwise noted, the original ADO-NeRF code is released under the Apache License 2.0. See `LICENSE` and `NOTICE`.

Third-party components retain their original licenses and copyright notices. External datasets, pretrained renderer checkpoints, scene-consistent diffusion weights, and Stable Diffusion weights are not included in this repository and are governed by their respective licenses and terms of use.
