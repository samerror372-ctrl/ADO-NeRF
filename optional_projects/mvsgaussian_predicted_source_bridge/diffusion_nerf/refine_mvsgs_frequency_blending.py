import glob
import os
import argparse
import sys

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


DIFFUSION_DATA_SUFFIX = "_diffusion_data.pt"


def list_diffusion_data(input_dir):
    return sorted(glob.glob(os.path.join(input_dir, f"*{DIFFUSION_DATA_SUFFIX}")))


def sample_name_from_path(path):
    name = os.path.basename(path)
    if not name.endswith(DIFFUSION_DATA_SUFFIX):
        raise ValueError(f"Expected MVSGaussian diffusion tensor ending with {DIFFUSION_DATA_SUFFIX}: {path}")
    return name[:-len(DIFFUSION_DATA_SUFFIX)]


def smoothstep(x):
    return x * x * (3.0 - 2.0 * x)


def low_frequency(img_batch, ksize=21, sigma=5.0):
    low = np.zeros_like(img_batch)
    for idx in range(img_batch.shape[0]):
        low[idx] = cv2.GaussianBlur(img_batch[idx].astype(np.float32), (ksize, ksize), sigma)
    return low


def frequency_blend(refined_preds, pred_rgb, blur_mask):
    original = (pred_rgb.detach().cpu().permute(0, 2, 3, 1).float().numpy() + 1.0) / 2.0
    original = np.clip(original, 0.0, 1.0)
    mask = blur_mask.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    if mask.shape[0] == 1 and original.shape[0] > 1:
        mask = np.repeat(mask, original.shape[0], axis=0)

    alpha = smoothstep(np.clip(mask / 0.35, 0.0, 1.0))
    low_orig = low_frequency(original)
    low_diff = low_frequency(refined_preds)
    high_diff = refined_preds - low_diff
    frequency_result = np.clip(low_orig + high_diff, 0.0, 1.0)
    return np.clip(frequency_result * alpha + original * (1.0 - alpha), 0.0, 1.0)


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


def prepare_data(pt_file, device, dtype, config, depth_mode="source"):
    data = torch.load(pt_file)
    source_rgbs = torch.nan_to_num(data["source_rgbs"].to(device, dtype=dtype), nan=0.0).clamp(-1.0, 1.0)
    pred_rgb = torch.nan_to_num(data["pred_rgb"].to(device, dtype=dtype), nan=0.0).clamp(-1.0, 1.0)
    H, W = source_rgbs.shape[-2:]

    source_depths = torch.nan_to_num(data["source_depths"].to(device, dtype=dtype), nan=0.0)
    if source_depths.shape[-2:] != (H, W):
        source_depths = F.interpolate(source_depths, size=(H, W), mode="bilinear", align_corners=False)

    blur_mask = data["blur_mask"].to(device, dtype=dtype)
    if blur_mask.shape[-2:] != (H, W):
        blur_mask = F.interpolate(blur_mask, size=(H, W), mode="bilinear", align_corners=False)
    blur_mask = blur_mask.clamp(0.0, 0.35)

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
    return total_images, pred_rgb, blur_mask, depth_condition, intrinsics, extrinsics, cond_num, nframe, H, W, tar_idx_batch, actual_depth_mode


def parse_args():
    parser = argparse.ArgumentParser(description="Run MVSGaussian scene-consistent diffusion refinement with frequency blending.")
    default_output_dir = os.path.join(ROOT, "mvsgs_frequency_blending_results")
    parser.add_argument("--input_dir", default=os.path.join(ROOT, "mvsgs_predicted_source_inputs"))
    parser.add_argument("--output_dir", default=default_output_dir)
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
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N samples.")
    parser.add_argument("--steps", type=int, default=50, help="DDIM inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Re-run diffusion even if refined PNGs already exist.")
    parser.add_argument(
            "--disable_prior_paste",
            action="store_true",
            help="Disable the center-paste shortcut in 3dpe+pixel prior; use only geometry-projected priors.")
    args = parser.parse_args()
    if args.depth_mode not in ("baseline", "source") and os.path.abspath(args.output_dir) == os.path.abspath(default_output_dir):
        safe_mode = args.depth_mode.replace("_only", "").replace("mvs_mono_completed", "completed")
        args.output_dir = "{}_depth_{}".format(default_output_dir, safe_mode)
    return args


def load_existing_result(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb[None]


def load_eval_tensors(pt_file, device, dtype):
    data = torch.load(pt_file)
    pred_rgb = torch.nan_to_num(data["pred_rgb"].to(device, dtype=dtype), nan=0.0).clamp(-1.0, 1.0)
    blur_mask = data["blur_mask"].to(device, dtype=dtype)
    if blur_mask.shape[-2:] != pred_rgb.shape[-2:]:
        blur_mask = F.interpolate(blur_mask, size=pred_rgb.shape[-2:], mode="bilinear", align_corners=False)
    blur_mask = blur_mask.clamp(0.0, 0.35)
    return pred_rgb, blur_mask


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    os.makedirs(args.output_dir, exist_ok=True)

    metric_models = load_metric_models(device)
    eval_results = []
    pt_files = list_diffusion_data(args.input_dir)
    if args.limit is not None:
        pt_files = pt_files[:args.limit]
    print(f"Found {len(pt_files)} MVSGaussian diffusion tensors in {args.input_dir}")
    print(f"Depth mode: {args.depth_mode}")

    needs_pipeline = args.force
    if not needs_pipeline:
        for pt_file in pt_files:
            base_name = sample_name_from_path(pt_file)
            output_path = os.path.join(args.output_dir, f"{base_name}_mvsgs_frequency_refined_00.png")
            if not os.path.exists(output_path):
                needs_pipeline = True
                break
    pipeline = config = None
    if needs_pipeline:
        pipeline, config = load_pipeline(device, dtype)
        if args.disable_prior_paste:
            config.model_cfg.enable_pixel_prior_paste = False
            print("Pixel prior paste: disabled")
    else:
        print("All outputs already exist; reusing saved PNGs for evaluation.")

    for pt_file in pt_files:
        base_name = sample_name_from_path(pt_file)
        save_path = os.path.join(args.output_dir, f"{base_name}_mvsgs_frequency_refined_00.png")
        blended = None if args.force else load_existing_result(save_path)
        generated = False
        if blended is not None:
            pred_rgb, blur_mask = load_eval_tensors(pt_file, device, dtype)
            print(f"Reused refined MVSGaussian result for {base_name}")
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
            ) = prepare_data(pt_file, device, dtype, config, depth_mode=args.depth_mode)
            if actual_depth_mode != args.depth_mode:
                print(f"Depth mode fallback for {base_name}: {actual_depth_mode}")

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
            blended = frequency_blend(refined, pred_rgb, blur_mask)
            generated = True
        eval_result = evaluate_masked_region(
                base_name,
                args.input_dir,
                blended,
                pred_rgb,
                blur_mask,
                device,
                metric_models)
        eval_results.append(eval_result)
        if eval_result.get("skipped"):
            print(f"Skipped metrics for {base_name}: {eval_result.get('reason')}")
        else:
            print(
                    f"Metrics {base_name}: "
                    f"PSNR {eval_result['psnr_orig']:.2f}->{eval_result['psnr_diff']:.2f} | "
                    f"SSIM {eval_result['ssim_orig']:.3f}->{eval_result['ssim_diff']:.3f} | "
                    f"FSIM {eval_result['fsim_orig']:.3f}->{eval_result['fsim_diff']:.3f}")
        if generated:
            uint8 = (blended * 255).astype(np.uint8)
            for idx in range(uint8.shape[0]):
                save_path = os.path.join(args.output_dir, f"{base_name}_mvsgs_frequency_refined_{idx:02d}.png")
                cv2.imwrite(save_path, uint8[idx, :, :, ::-1])
            print(f"Saved refined MVSGaussian result for {base_name}")

    report_path = write_masked_report(
            args.output_dir,
            eval_results,
            "MVSGaussian frequency-blending local masked-region refinement report")
    print(f"Masked report saved to: {report_path}")


if __name__ == "__main__":
    main()
