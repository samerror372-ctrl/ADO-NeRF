import glob
import os
import argparse
import sys
import csv
import math

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict
from omegaconf import OmegaConf

from diffusers import AutoencoderKL
from my_diffusers.models import UNet2DConditionModel
from my_diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_multiview_inverse_for_nerfv1 import (
    StableDiffusionMultiViewPipeline,
)
from mvsgs_refinement_metrics import evaluate_masked_region, load_metric_models, write_masked_report
from src.modules.schedulers import get_diffusion_scheduler
from depth_completion_utils import select_depth_condition


DATASET_NAMES = ("dtu", "llff", "nerf_synthetic")
DIFFUSION_DATA_SUFFIX = "_diffusion_data.pt"


def list_diffusion_data(input_dir):
    return sorted(glob.glob(os.path.join(input_dir, f"*{DIFFUSION_DATA_SUFFIX}")))


def sample_name_from_path(path):
    name = os.path.basename(path)
    if not name.endswith(DIFFUSION_DATA_SUFFIX):
        raise ValueError(f"Expected MVSGaussian diffusion tensor ending with {DIFFUSION_DATA_SUFFIX}: {path}")
    return name[:-len(DIFFUSION_DATA_SUFFIX)]

MASK_STRENGTH_PROFILES = {
    "orig": {
        "setting": "orig",
        "kind": "orig",
        "scale": 1.0,
    },
    "orig_s0p50": {
        "setting": "orig_s0.50",
        "kind": "orig",
        "scale": 0.50,
    },
    "ado_tuned": {
        "setting": "ado_grid_l1.0.10_l2.0.25_l3.0.10_s0.50",
        "kind": "recompose",
        "w1": 0.10,
        "w2": 0.25,
        "w3": 0.10,
        "post_scale": 0.50,
    },
}

MASK_PROFILE_ALIASES = {
    "orig": "orig",
    "original": "orig",
    "none": "orig",
    "off": "orig",
    "orig_s0.50": "orig_s0p50",
    "orig_s0p50": "orig_s0p50",
    "pre_tuning": "orig_s0p50",
    "ado": "ado_tuned",
    "tuned": "ado_tuned",
    "ado_tuned": "ado_tuned",
    "grid_l1_0p10_l2_0p25_l3_0p10_s0p50": "ado_tuned",
}


def smoothstep(x):
    return x * x * (3.0 - 2.0 * x)


def spatial_mask_blend(refined_preds, pred_rgb, blur_mask, max_mask_strength=0.35):
    original = (pred_rgb.detach().cpu().permute(0, 2, 3, 1).float().numpy() + 1.0) / 2.0
    original = np.clip(original, 0.0, 1.0)
    mask = blur_mask.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    if mask.shape[0] == 1 and original.shape[0] > 1:
        mask = np.repeat(mask, original.shape[0], axis=0)

    alpha = smoothstep(np.clip(mask / max_mask_strength, 0.0, 1.0))
    return np.clip(refined_preds * alpha + original * (1.0 - alpha), 0.0, 1.0)


def resolve_mask_profile(name):
    key = MASK_PROFILE_ALIASES.get(str(name).strip(), str(name).strip())
    if key not in MASK_STRENGTH_PROFILES:
        valid = ", ".join(sorted(MASK_STRENGTH_PROFILES))
        raise ValueError(f"Unknown mask profile {name!r}. Valid profiles: {valid}")
    return key, dict(MASK_STRENGTH_PROFILES[key])


def gaussian_blur_4d(mask, kernel_size, sigma):
    coords = torch.arange(kernel_size, dtype=torch.float32, device=mask.device) - kernel_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = torch.mm(kernel_1d[:, None], kernel_1d[None, :])[None, None]
    pad = kernel_size // 2
    pad_mode = "reflect" if mask.shape[-2] > pad and mask.shape[-1] > pad else "replicate"
    return F.conv2d(F.pad(mask, (pad, pad, pad, pad), mode=pad_mode), kernel).clamp(0.0, 1.0)


def resize_mask_nchw(mask, target_size):
    if mask.shape[-2:] == target_size:
        return mask
    return F.interpolate(mask.float(), size=target_size, mode="bilinear", align_corners=False)


def mask_field_to_nchw(data, key, target_size, device, dtype):
    if key not in data:
        return None
    mask = data[key]
    if not torch.is_tensor(mask):
        return None
    mask = mask.to(device=device, dtype=dtype)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim == 5 and mask.shape[2] == 1:
        mask = mask.flatten(0, 1)
    if mask.ndim != 4:
        return None
    return resize_mask_nchw(torch.nan_to_num(mask.float(), nan=0.0), target_size).clamp(0.0, 1.0)


def recompose_mask_from_layers(layer1_raw, layer2_soft, layer3_raw, profile):
    raw = (
        float(profile["w1"]) * layer1_raw.float()
        + float(profile["w2"]) * layer2_soft.float()
        + float(profile["w3"]) * layer3_raw.float()
    ).clamp_min(0.0)
    if raw.max() <= 1e-7:
        return torch.zeros_like(layer1_raw.float())

    expanded = F.max_pool2d(raw, kernel_size=15, stride=1, padding=7)
    blur_level_1 = gaussian_blur_4d(expanded, kernel_size=15, sigma=5.0)
    blur_level_2 = gaussian_blur_4d(blur_level_1, kernel_size=51, sigma=15.0)
    blur_level_3 = gaussian_blur_4d(blur_level_2, kernel_size=101, sigma=35.0)
    final_gradient = (blur_level_1 * 0.4) + (blur_level_2 * 0.4) + (blur_level_3 * 0.2)
    fg_max = final_gradient.max()
    if fg_max > 1e-7:
        angle = torch.tensor(torch.pi / 2.0, device=final_gradient.device)
        final_gradient = 0.35 * torch.sin(angle * (final_gradient / fg_max))
    return (final_gradient * float(profile.get("post_scale", 1.0))).clamp(0.0, 0.35)


def build_mask_condition(data, blur_mask, target_size, device, dtype, profile_name):
    profile_key, profile = resolve_mask_profile(profile_name)
    blur_mask = resize_mask_nchw(torch.nan_to_num(blur_mask.float(), nan=0.0), target_size)

    if profile["kind"] == "orig":
        mask = blur_mask * float(profile.get("scale", 1.0))
        return mask.clamp(0.0, 0.35).to(dtype=dtype), profile_key

    layer1 = mask_field_to_nchw(data, "mask_layer1_raw", target_size, device, dtype)
    layer2 = mask_field_to_nchw(data, "mask_layer2_soft", target_size, device, dtype)
    layer3 = mask_field_to_nchw(data, "mask_layer3_raw", target_size, device, dtype)
    if layer1 is None or layer2 is None or layer3 is None:
        return blur_mask.clamp(0.0, 0.35).to(dtype=dtype), f"orig_missing_layers_for_{profile_key}"
    return recompose_mask_from_layers(layer1, layer2, layer3, profile).to(dtype=dtype), profile_key


def safe_dataset_name(name):
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(name).strip())
    return cleaned or "dataset"


def infer_dataset_name(input_dir):
    lower = os.path.basename(os.path.abspath(input_dir)).lower()
    if "llff" in lower:
        return "llff"
    if "nerf_synthetic" in lower or "synthetic" in lower:
        return "nerf_synthetic"
    if "dtu" in lower:
        return "dtu"

    first_pt = list_diffusion_data(input_dir)
    if first_pt and os.path.basename(first_pt[0]).startswith("scan"):
        return "dtu"
    return "dataset"


def parse_dataset_input(value):
    if "=" not in value:
        raise ValueError(f"Dataset input must be name=path, got: {value}")
    name, path = value.split("=", 1)
    return safe_dataset_name(name), os.path.abspath(path)


def build_dataset_runs(args):
    runs = []
    if args.dataset_input:
        for item in args.dataset_input:
            runs.append(parse_dataset_input(item))
    elif args.input_root:
        input_root = os.path.abspath(args.input_root)
        for name in [safe_dataset_name(x) for x in args.datasets.split(",") if x.strip()]:
            runs.append((name, os.path.join(input_root, name)))
    else:
        name = safe_dataset_name(args.dataset_name or infer_dataset_name(args.input_dir))
        runs.append((name, os.path.abspath(args.input_dir)))

    if not runs:
        raise ValueError("No dataset inputs were configured.")
    return runs


def make_output_dirs(args, dataset_name, multiple_datasets):
    if args.output_dir and not multiple_datasets:
        dataset_dir = os.path.abspath(args.output_dir)
    else:
        dataset_dir = os.path.join(os.path.abspath(args.experiment_root), safe_dataset_name(dataset_name))
    dirs = {
        "root": dataset_dir,
        "refined": os.path.join(dataset_dir, "refined"),
        "intermediate": os.path.join(dataset_dir, "intermediate"),
        "comparison": os.path.join(dataset_dir, "comparison"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def pred_rgb_to_np(pred_rgb):
    rgb = (pred_rgb.detach().cpu().permute(0, 2, 3, 1).float().numpy() + 1.0) * 0.5
    return np.clip(rgb, 0.0, 1.0)


def mask_to_np(blur_mask, target_count=1):
    mask = blur_mask.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    if mask.shape[0] == 1 and target_count > 1:
        mask = np.repeat(mask, target_count, axis=0)
    return np.clip(mask, 0.0, 0.35)


def to_uint8_rgb(image):
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def write_rgb(path, image):
    cv2.imwrite(path, cv2.cvtColor(to_uint8_rgb(image), cv2.COLOR_RGB2BGR))


def load_gt_rgb(input_dir, base_name, target_shape):
    gt_path = os.path.join(input_dir, f"{base_name}_0_gt_rgb.png")
    gt_bgr = cv2.imread(gt_path, cv2.IMREAD_COLOR)
    if gt_bgr is None:
        h, w = target_shape[:2]
        return np.full((h, w, 3), 0.18, dtype=np.float32), False
    gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if gt_rgb.shape[:2] != target_shape[:2]:
        gt_rgb = cv2.resize(gt_rgb, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
    return np.clip(gt_rgb, 0.0, 1.0), True


def mask_visuals(mask, base_rgb):
    mask_linear = np.clip(mask[..., 0] / 0.35, 0.0, 1.0)
    mask_gray = np.repeat(mask_linear[..., None], 3, axis=-1)
    heatmap_bgr = cv2.applyColorMap((mask_linear * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    alpha = np.clip(mask_linear * 0.70, 0.0, 0.70)[..., None]
    overlay = np.clip(base_rgb * (1.0 - alpha) + heatmap_rgb * alpha, 0.0, 1.0)
    return mask_gray, overlay


def fmt_metric(value):
    try:
        value = float(value)
    except Exception:
        return "nan"
    if not math.isfinite(value):
        return "nan"
    return f"{value:.4f}"


def metric_lines(record, mask_profile):
    if record.get("skipped"):
        return [
            f"metrics skipped: {record.get('reason', '')}",
            f"mask_profile: {mask_profile}",
        ]

    lines = [
        (
            "mask PSNR/SSIM/LPIPS: "
            f"{fmt_metric(record.get('psnr_orig'))}->{fmt_metric(record.get('psnr_diff'))} | "
            f"{fmt_metric(record.get('ssim_orig'))}->{fmt_metric(record.get('ssim_diff'))} | "
            f"{fmt_metric(record.get('lpips_orig'))}->{fmt_metric(record.get('lpips_diff'))}"
        ),
        (
            "global PSNR/SSIM/LPIPS: "
            f"{fmt_metric(record.get('global_psnr_orig'))}->{fmt_metric(record.get('global_psnr_diff'))} | "
            f"{fmt_metric(record.get('global_ssim_orig'))}->{fmt_metric(record.get('global_ssim_diff'))} | "
            f"{fmt_metric(record.get('global_lpips_orig'))}->{fmt_metric(record.get('global_lpips_diff'))}"
        ),
        f"mask_profile: {mask_profile}",
    ]
    if record.get("dtu_valid"):
        lines.append(
            "dtu valid PSNR/SSIM: "
            f"{fmt_metric(record.get('dtu_psnr_orig'))}->{fmt_metric(record.get('dtu_psnr_diff'))} | "
            f"{fmt_metric(record.get('dtu_ssim_orig'))}->{fmt_metric(record.get('dtu_ssim_diff'))}"
        )
    return lines


def labeled_panel(image, title):
    rgb = to_uint8_rgb(image)
    h, w = rgb.shape[:2]
    header = np.zeros((34, w, 3), dtype=np.uint8)
    cv2.putText(header, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([header, rgb])


def make_comparison_image(base_name, gt_rgb, gt_exists, original, refined, mask_gray, mask_overlay, record, mask_profile):
    panels = [
        labeled_panel(gt_rgb, "GT" if gt_exists else "GT missing"),
        labeled_panel(original, "MVSGaussian before"),
        labeled_panel(refined, "Diffusion after"),
        labeled_panel(mask_gray, "Original mask"),
        labeled_panel(mask_overlay, "Mask overlay"),
    ]
    row = np.hstack(panels)
    text_h = 104
    text = np.zeros((text_h, row.shape[1], 3), dtype=np.uint8)
    cv2.putText(text, base_name, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 1, cv2.LINE_AA)
    for idx, line in enumerate(metric_lines(record, mask_profile)):
        cv2.putText(text, line[:220], (10, 50 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([row, text])


def save_sample_visuals(dirs, input_dir, base_name, pred_rgb, blended, blur_mask, record, mask_profile):
    original = pred_rgb_to_np(pred_rgb)[0]
    refined = np.clip(blended[0], 0.0, 1.0)
    mask = mask_to_np(blur_mask, target_count=1)[0]
    gt_rgb, gt_exists = load_gt_rgb(input_dir, base_name, original.shape)
    mask_gray, mask_overlay = mask_visuals(mask, original)

    write_rgb(os.path.join(dirs["intermediate"], f"{base_name}_0_gt_rgb.png"), gt_rgb)
    write_rgb(os.path.join(dirs["intermediate"], f"{base_name}_1_mvsgs_before.png"), original)
    write_rgb(os.path.join(dirs["intermediate"], f"{base_name}_2_diffusion_after.png"), refined)
    write_rgb(os.path.join(dirs["intermediate"], f"{base_name}_3_mask_gray.png"), mask_gray)
    write_rgb(os.path.join(dirs["intermediate"], f"{base_name}_4_mask_overlay.png"), mask_overlay)

    comparison = make_comparison_image(
        base_name,
        gt_rgb,
        gt_exists,
        original,
        refined,
        mask_gray,
        mask_overlay,
        record,
        mask_profile)
    cv2.imwrite(os.path.join(dirs["comparison"], f"{base_name}_compare.png"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))


def write_records_csv(output_dir, records):
    path = os.path.join(output_dir, "metrics.csv")
    fields = [
        "dataset", "name", "skipped", "reason", "mask_profile",
        "psnr_orig", "psnr_diff", "ssim_orig", "ssim_diff", "fsim_orig", "fsim_diff",
        "lpips_orig", "lpips_diff", "niqe_orig", "niqe_diff",
        "global_psnr_orig", "global_psnr_diff", "global_ssim_orig", "global_ssim_diff",
        "global_fsim_orig", "global_fsim_diff", "global_lpips_orig", "global_lpips_diff",
        "global_niqe_orig", "global_niqe_diff",
        "dtu_valid", "dtu_reason", "dtu_psnr_orig", "dtu_psnr_diff", "dtu_ssim_orig", "dtu_ssim_diff",
        "dtu_fsim_orig", "dtu_fsim_diff", "dtu_lpips_orig", "dtu_lpips_diff", "dtu_niqe_orig", "dtu_niqe_diff",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})
    return path


def load_pipeline(device, dtype):
    model_dir = os.path.join(ROOT, "check_points", "pretrained_model")
    sd_base_dir = os.path.join(ROOT, "check_points", "Stable-Diffusion-2-1-base")

    config = EasyDict(OmegaConf.load(os.path.join(model_dir, "config.yaml")))
    config.pretrained_model_name_or_path = sd_base_dir

    vae = AutoencoderKL.from_pretrained(
            sd_base_dir,
            subfolder="vae",
            local_files_only=True).to(device, dtype=dtype)
    vae.requires_grad_(False)

    unet = UNet2DConditionModel.from_pretrained(
            sd_base_dir,
            subfolder="unet",
            rank=0,
            model_cfg=config.model_cfg,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
            local_files_only=True).to(device, dtype=dtype)
    weights = torch.load(os.path.join(model_dir, "ema_unet.pt"), map_location="cpu")
    unet.load_state_dict(weights, strict=False)
    unet.requires_grad_(False)
    unet.eval()

    scheduler = get_diffusion_scheduler(config, name="DDIM")
    scheduler.config.prediction_type = "v_prediction"

    pipeline = StableDiffusionMultiViewPipeline.from_pretrained(
            sd_base_dir,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            torch_dtype=dtype,
            local_files_only=True,
            zoom_scale=1.0).to(device)
    return pipeline, config


def prepare_data(pt_file, device, dtype, config, depth_mode="source", mask_profile="orig"):
    data = torch.load(pt_file)
    source_rgbs = torch.nan_to_num(data["source_rgbs"].to(device, dtype=dtype), nan=0.0).clamp(-1.0, 1.0)
    pred_rgb = torch.nan_to_num(data["pred_rgb"].to(device, dtype=dtype), nan=0.0).clamp(-1.0, 1.0)
    H, W = source_rgbs.shape[-2:]

    source_depths = torch.nan_to_num(data["source_depths"].to(device, dtype=dtype), nan=0.0)
    if source_depths.shape[-2:] != (H, W):
        source_depths = F.interpolate(source_depths, size=(H, W), mode="bilinear", align_corners=False)

    blur_mask = data["blur_mask"].to(device, dtype=dtype)
    blur_mask, actual_mask_profile = build_mask_condition(data, blur_mask, (H, W), device, dtype, mask_profile)

    intrinsics = torch.nan_to_num(data["intrinsics"].to(device).float(), nan=0.0)
    extrinsics_key = "extrinsics_w2c" if "extrinsics_w2c" in data else "extrinsics"
    extrinsics = torch.nan_to_num(data[extrinsics_key].to(device).float(), nan=0.0)
    if data.get("extrinsics_format", "w2c") == "c2w":
        extrinsics = torch.inverse(extrinsics)

    target_depth_scale = 1.0
    if config.get("normalize_extrinsic", False):
        cam_centers = torch.inverse(extrinsics)[:, :3, 3]
        current_max = torch.max(cam_centers.abs().max(), source_depths.max()).item()
        target_limit = config.get("camera_longest_side", 5.0)
        if abs(current_max - target_limit) > (target_limit * 0.01) and current_max > 1e-5:
            scale = target_limit / current_max
            extrinsics[:, :3, 3] *= scale
            source_depths *= scale
            target_depth_scale = scale

    cond_num = int(data["cond_num"])
    nframe = cond_num + pred_rgb.shape[0]
    depth_condition, actual_depth_mode = select_depth_condition(
            data,
            source_depths,
            depth_mode,
            (H, W),
            device,
            dtype,
            target_depth_scale=target_depth_scale)
    total_images = torch.cat([source_rgbs, pred_rgb], dim=0)
    tar_idx_batch = np.arange(cond_num, nframe).tolist()
    return total_images, pred_rgb, blur_mask, depth_condition, intrinsics, extrinsics, cond_num, nframe, H, W, tar_idx_batch, actual_depth_mode, actual_mask_profile


def parse_args():
    parser = argparse.ArgumentParser(description="Run MVSGaussian scene-consistent diffusion refinement without frequency blending.")
    default_experiment_root = os.path.join(ROOT, "mvsgs_scene_consistent_results")
    parser.add_argument("--input_dir", default=os.path.join(ROOT, "mvsgs_predicted_source_inputs"))
    parser.add_argument("--input_root", default=None, help="Folder containing dataset subdirs such as dtu, llff, nerf_synthetic.")
    parser.add_argument("--dataset_input", action="append", default=[], help="Repeatable name=path dataset input, e.g. dtu=/path/to/mvsgs_predicted_source_inputs.")
    parser.add_argument("--datasets", default="dtu,llff,nerf_synthetic", help="Comma-separated dataset names used with --input_root.")
    parser.add_argument("--dataset_name", default=None, help="Name for a single --input_dir run.")
    parser.add_argument("--experiment_root", default=default_experiment_root)
    parser.add_argument("--output_dir", default=None, help="Legacy single-dataset output dir. Multi-dataset runs use --experiment_root/name.")
    parser.add_argument(
            "--depth_mode",
            default="source",
            choices=[
                "baseline", "source",
                "source_mvs", "src_mvs", "source_mvs_only",
                "source_mono", "src_mono", "source_mono_only",
                "source_completed", "src_completed", "source_mvs_mono", "source_mvs_mono_completed",
                "mvs", "mvs_only", "mono", "mono_only", "completed", "mvs_mono", "mvs_mono_completed",
            ],
            help="Depth condition: source_* modes keep geometry on source views; target modes are kept only for legacy checks.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N samples for a quick check.")
    parser.add_argument("--steps", type=int, default=50, help="DDIM inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Re-run diffusion even if refined PNGs already exist.")
    parser.add_argument("--mask_profile", default="orig", help="orig keeps the exported mask unchanged; ado_tuned enables ADO-style recomposition.")
    paste_group = parser.add_mutually_exclusive_group()
    paste_group.add_argument(
            "--enable_prior_paste",
            dest="prior_paste",
            action="store_true",
            help="Enable the center-paste shortcut in 3dpe+pixel prior.")
    paste_group.add_argument(
            "--disable_prior_paste",
            dest="prior_paste",
            action="store_false",
            help="Disable the center-paste shortcut in 3dpe+pixel prior.")
    parser.set_defaults(prior_paste=False)
    args = parser.parse_args()
    resolve_mask_profile(args.mask_profile)
    return args


def load_existing_result(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb[None]


def load_eval_tensors(pt_file, device, dtype, mask_profile="orig"):
    data = torch.load(pt_file)
    pred_rgb = torch.nan_to_num(data["pred_rgb"].to(device, dtype=dtype), nan=0.0).clamp(-1.0, 1.0)
    blur_mask = data["blur_mask"].to(device, dtype=dtype)
    blur_mask, actual_mask_profile = build_mask_condition(
            data,
            blur_mask,
            pred_rgb.shape[-2:],
            device,
            dtype,
            mask_profile)
    return pred_rgb, blur_mask, actual_mask_profile


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    dataset_runs = build_dataset_runs(args)
    multiple_datasets = len(dataset_runs) > 1
    os.makedirs(args.experiment_root, exist_ok=True)

    metric_models = load_metric_models(device)
    all_eval_results = []
    pipeline = config = None

    print("Fusion mode: smoothstep spatial mask blending")
    print(f"Depth mode: {args.depth_mode}")
    print(f"Mask profile: {resolve_mask_profile(args.mask_profile)[0]}")
    print(f"Pixel prior paste: {'enabled' if args.prior_paste else 'disabled'}")
    print(f"Experiment root: {os.path.abspath(args.experiment_root)}")

    for dataset_name, input_dir in dataset_runs:
        dirs = make_output_dirs(args, dataset_name, multiple_datasets)
        eval_results = []
        pt_files = list_diffusion_data(input_dir)
        if args.limit is not None:
            pt_files = pt_files[:args.limit]
        print(f"\n[{dataset_name}] Found {len(pt_files)} MVSGaussian diffusion tensors in {input_dir}")

        needs_pipeline = args.force
        if not needs_pipeline:
            for pt_file in pt_files:
                base_name = sample_name_from_path(pt_file)
                output_path = os.path.join(dirs["refined"], f"{base_name}_mvsgs_scene_refined_00.png")
                if not os.path.exists(output_path):
                    needs_pipeline = True
                    break
        if needs_pipeline and pipeline is None:
            pipeline, config = load_pipeline(device, dtype)
            config.model_cfg.enable_pixel_prior_paste = bool(args.prior_paste)
        elif not needs_pipeline:
            print(f"[{dataset_name}] All outputs already exist; reusing saved PNGs for evaluation.")

        for pt_file in pt_files:
            base_name = sample_name_from_path(pt_file)
            save_path = os.path.join(dirs["refined"], f"{base_name}_mvsgs_scene_refined_00.png")
            blended = None if args.force else load_existing_result(save_path)
            generated = False
            if blended is not None:
                pred_rgb, blur_mask, actual_mask_profile = load_eval_tensors(
                        pt_file,
                        device,
                        dtype,
                        mask_profile=args.mask_profile)
                print(f"[{dataset_name}] Reused scene-consistent refined MVSGaussian result for {base_name}")
            else:
                (
                    total_images,
                    pred_rgb,
                    blur_mask,
                    depth_condition,
                    intrinsics,
                    extrinsics,
                    cond_num,
                    nframe,
                    H,
                    W,
                    tar_idx_batch,
                    actual_depth_mode,
                    actual_mask_profile,
                ) = prepare_data(
                        pt_file,
                        device,
                        dtype,
                        config,
                        depth_mode=args.depth_mode,
                        mask_profile=args.mask_profile)
                if actual_depth_mode != args.depth_mode:
                    print(f"[{dataset_name}] Depth mode fallback for {base_name}: {actual_depth_mode}")
                if actual_mask_profile != resolve_mask_profile(args.mask_profile)[0]:
                    print(f"[{dataset_name}] Mask profile fallback for {base_name}: {actual_mask_profile}")

                generator = torch.Generator(device=device).manual_seed(args.seed)
                output = pipeline(
                        images=total_images,
                        pred_rgb=pred_rgb,
                        blur_mask=blur_mask,
                        cond_num=cond_num,
                        nframe=nframe,
                        height=H,
                        width=W,
                        depth=depth_condition,
                        intrinsics=intrinsics,
                        extrinsics=extrinsics,
                        preserve_target_depth=(depth_condition.shape[0] == nframe),
                        prompt="",
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        generator=generator,
                        output_type="np",
                        config=config,
                        tar_idx_batch=tar_idx_batch,
                        key_rescale=None,
                        class_label=0,
                        tag=["others"] * nframe)

                refined = output.images[cond_num:]
                blended = spatial_mask_blend(refined, pred_rgb, blur_mask)
                generated = True

            eval_result = evaluate_masked_region(
                    base_name,
                    input_dir,
                    blended,
                    pred_rgb,
                    blur_mask,
                    device,
                    metric_models)
            eval_result["dataset"] = dataset_name
            eval_result["mask_profile"] = actual_mask_profile
            eval_results.append(eval_result)

            save_sample_visuals(dirs, input_dir, base_name, pred_rgb, blended, blur_mask, eval_result, actual_mask_profile)

            if eval_result.get("skipped"):
                print(f"[{dataset_name}] Skipped metrics for {base_name}: {eval_result.get('reason')}")
            else:
                print(
                        f"[{dataset_name}] Metrics {base_name}: "
                        f"PSNR {eval_result['psnr_orig']:.2f}->{eval_result['psnr_diff']:.2f} | "
                        f"SSIM {eval_result['ssim_orig']:.3f}->{eval_result['ssim_diff']:.3f} | "
                        f"FSIM {eval_result['fsim_orig']:.3f}->{eval_result['fsim_diff']:.3f}")
            if generated:
                uint8 = (blended * 255).astype(np.uint8)
                for idx in range(uint8.shape[0]):
                    save_path = os.path.join(dirs["refined"], f"{base_name}_mvsgs_scene_refined_{idx:02d}.png")
                    cv2.imwrite(save_path, uint8[idx, :, :, ::-1])
                print(f"[{dataset_name}] Saved scene-consistent refined MVSGaussian result for {base_name}")

        report_path = write_masked_report(
                dirs["root"],
                eval_results,
                f"MVSGaussian predicted-source {dataset_name} local masked-region refinement report")
        csv_path = write_records_csv(dirs["root"], eval_results)
        print(f"[{dataset_name}] Masked report saved to: {report_path}")
        print(f"[{dataset_name}] Metrics CSV saved to: {csv_path}")

        for record in eval_results:
            combined = dict(record)
            combined["name"] = f"{dataset_name}/{record['name']}"
            all_eval_results.append(combined)

    if multiple_datasets and all_eval_results:
        combined_report = write_masked_report(
                args.experiment_root,
                all_eval_results,
                "MVSGaussian predicted-source multi-dataset local masked-region refinement report")
        combined_csv = write_records_csv(args.experiment_root, all_eval_results)
        print(f"\nCombined report saved to: {combined_report}")
        print(f"Combined metrics CSV saved to: {combined_csv}")


if __name__ == "__main__":
    main()
