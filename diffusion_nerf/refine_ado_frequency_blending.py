import os
import glob
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from omegaconf import OmegaConf
from easydict import EasyDict

from diffusers import AutoencoderKL
from my_diffusers.models import UNet2DConditionModel
from src.modules.schedulers import get_diffusion_scheduler
from my_diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_multiview_inverse_for_nerfv1 import StableDiffusionMultiViewPipeline
from ado_refine_utils import (
    EvalAlignedMetricRunner,
    evaluate_refinement_image,
    format_metric_summary,
    load_gt_rgb,
    load_target_eval_mask,
    save_refinement_visuals,
    write_refinement_reports,
)

DIFFUSION_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DIFFUSION_ROOT)

try:
    import lpips
    import pyiqa
    HAS_METRICS = True
except ImportError:
    print("\n[WARNING] Metric packages are missing. Install lpips, pyiqa, piq, and scikit-image for evaluation.")
    HAS_METRICS = False

def print_stats(name, tensor):
    """Print a compact tensor sanity summary."""
    if tensor is None:
        print(f"  [STAT] {name}: None")
        return
    t = tensor.detach().float()
    print(f"  [STAT] {name:15s} | Shape: {str(list(t.shape)):25s} | Range: [{t.min():8.3f}, {t.max():8.3f}] | Mean: {t.mean():8.3f} | Std: {t.std():8.3f} | NaN: {torch.isnan(t).any()}")


def _env_int(name, default):
    value = os.environ.get(name, str(default))
    return int(value or default)


def _env_float(name, default):
    value = os.environ.get(name, str(default))
    return float(value or default)


def _collect_input_files(input_data_dir, max_files):
    pt_files = sorted(glob.glob(os.path.join(input_data_dir, "*_ado_data.pt")))
    if max_files > 0:
        pt_files = pt_files[:max_files]
    return pt_files


def _base_name_from_pt(path):
    return os.path.basename(path).replace("_ado_data.pt", "")


def _torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        return torch.load(path, map_location=map_location, weights_only=False)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float32 

    model_dir = os.path.join(DIFFUSION_ROOT, "check_points", "pretrained_model")
    sd_base_dir = os.path.join(DIFFUSION_ROOT, "check_points", "Stable-Diffusion-2-1-base")
    enable_paste_env = os.environ.get("ADO_ENABLE_PASTE", "0").strip().lower()
    enable_pixel_prior_paste = enable_paste_env not in ("0", "false", "no", "off")
    paste_suffix = "" if enable_pixel_prior_paste else "_nopaste"
    input_data_dir = os.environ.get("ADO_REFINE_INPUT_DIR", os.path.join(PROJECT_ROOT, "diffusion_input"))
    output_dir = os.environ.get("ADO_REFINE_OUTPUT_DIR", os.path.join(DIFFUSION_ROOT, f"ado_result_frequency{paste_suffix}"))
    debug_dir = os.environ.get("ADO_REFINE_DEBUG_DIR", os.path.join(DIFFUSION_ROOT, f"ado_debug_frequency{paste_suffix}"))
    max_files = _env_int("ADO_REFINE_MAX_FILES", 0)
    diffusion_steps = _env_int("ADO_DIFFUSION_STEPS", 50)
    guidance_scale = _env_float("ADO_GUIDANCE_SCALE", 1.5)
    repair_mask_threshold = _env_float("ADO_REPAIR_MASK_THRESHOLD", 0.01)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    print("Loading Config and Models...")
    config = EasyDict(OmegaConf.load(os.path.join(model_dir, "config.yaml")))
    config.pretrained_model_name_or_path = sd_base_dir  
    config.model_cfg.enable_pixel_prior_paste = enable_pixel_prior_paste
    print(f"Pixel prior paste: {'enabled' if enable_pixel_prior_paste else 'disabled'}")
    
    vae = AutoencoderKL.from_pretrained(sd_base_dir, subfolder="vae", local_files_only=True).to(device, dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(sd_base_dir, subfolder="unet", rank=0, model_cfg=config.model_cfg, low_cpu_mem_usage=False, ignore_mismatched_sizes=True, local_files_only=True).to(device, dtype=weight_dtype)
    
    print("\n  -> Loading UNet weights from ema_unet.pt...")
    weights = _torch_load(f"{model_dir}/ema_unet.pt", map_location="cpu")
    
    load_result = unet.load_state_dict(weights, strict=False)
    missing = load_result.missing_keys
    if len(missing) > 0:
        print(f"  [WARNING] Missing {len(missing)} UNet weights. First 5: {missing[:5]}")
        if any("conv_in" in k for k in missing):
            print("  [ERROR] conv_in weights are missing; refinement quality will be invalid.")
    else:
        print("  [OK] UNet weights loaded with no missing keys.")

    unet.eval()

    scheduler = get_diffusion_scheduler(config, name="DDIM")
    print(f"\n  -> Scheduler prediction type before override: {getattr(scheduler.config, 'prediction_type', 'epsilon')}")
    print("  -> Using v_prediction scheduler mode.")
    scheduler.config.prediction_type = "v_prediction"

    pipeline = StableDiffusionMultiViewPipeline.from_pretrained(sd_base_dir, vae=vae, unet=unet, scheduler=scheduler, safety_checker=None, torch_dtype=weight_dtype, local_files_only=True, zoom_scale=1.0).to(device)

    eval_results = []
    if HAS_METRICS:
        print("\n  -> Loading metric models: LPIPS, NIQE, and FSIM.")
        lpips_vgg = lpips.LPIPS(net='vgg').to(device)
        lpips_vgg.eval()
        niqe_metric = pyiqa.create_metric('niqe', device=device)
        metric_runner = EvalAlignedMetricRunner(device, lpips_vgg, niqe_metric)
    else:
        metric_runner = None

    pt_files = _collect_input_files(input_data_dir, max_files)
    print(f"\nFound {len(pt_files)} ADO diffusion tensor files.")
    print(f"[ADO Frequency] input_data_dir = {input_data_dir}")
    print(f"[ADO Frequency] output_dir = {output_dir}")
    print(f"[ADO Frequency] diffusion_steps = {diffusion_steps}, guidance_scale = {guidance_scale}")
    print(f"[ADO Frequency] repair_mask_threshold = {repair_mask_threshold}")
    if max_files > 0:
        print(f"[ADO Frequency] max_files = {max_files}")

    for file_idx, pt_file in enumerate(pt_files):
        base_name = _base_name_from_pt(pt_file)
        print(f"\n--- Processing {base_name} ---")

        data = _torch_load(pt_file)
        source_rgbs = data["source_rgbs"].to(device, dtype=weight_dtype)
        pred_rgb = data["pred_rgb"].to(device, dtype=weight_dtype)
        
        source_rgbs = torch.nan_to_num(source_rgbs, nan=0.0).clamp(-1.0, 1.0)
        pred_rgb = torch.nan_to_num(pred_rgb, nan=0.0).clamp(-1.0, 1.0)

        H, W = source_rgbs.shape[2], source_rgbs.shape[3]
        source_depths = data["source_depths"].to(device, dtype=weight_dtype)
        if source_depths.shape[-2:] != (H, W):
            source_depths = F.interpolate(source_depths, size=(H, W), mode='bilinear', align_corners=False)
        source_depths = torch.nan_to_num(source_depths, nan=0.0)

        blur_mask = data["blur_mask"].to(device, dtype=weight_dtype)
        if blur_mask.shape[-2:] != (H, W):
            blur_mask = F.interpolate(blur_mask, size=(H, W), mode='bilinear', align_corners=False)

        intrinsics = torch.nan_to_num(data["intrinsics"].to(device).float(), nan=0.0)
        extrinsics_key = "extrinsics_w2c" if "extrinsics_w2c" in data else "extrinsics"
        extrinsics = torch.nan_to_num(data[extrinsics_key].to(device).float(), nan=0.0)
        if data.get("extrinsics_format", "w2c") == "c2w" and extrinsics_key != "extrinsics_w2c":
            extrinsics = torch.inverse(extrinsics)

        if config.get("normalize_extrinsic", False):
            cam_centers = torch.inverse(extrinsics)[:, :3, 3]
            current_max = torch.max(cam_centers.abs().max(), source_depths.max()).item()
            target_limit = config.get("camera_longest_side", 5.0)
            if abs(current_max - target_limit) > (target_limit * 0.01) and current_max > 1e-5:
                scale = target_limit / current_max
                extrinsics[:, :3, 3] *= scale
                source_depths *= scale

        cond_num = data["cond_num"]
        nframe_total = cond_num + pred_rgb.shape[0]
        total_images = torch.cat([source_rgbs, pred_rgb], dim=0) 
        tar_idx_batch = np.arange(cond_num, nframe_total).tolist() 
        generator = torch.Generator(device=device).manual_seed(42)
        
        output = pipeline(
            images=total_images, pred_rgb=pred_rgb, blur_mask=blur_mask, cond_num=cond_num, 
            nframe=nframe_total, height=H, width=W, depth=source_depths, 
            intrinsics=intrinsics, extrinsics=extrinsics, prompt="", 
            num_inference_steps=diffusion_steps, guidance_scale=guidance_scale, generator=generator, 
            output_type="np", config=config, tar_idx_batch=tar_idx_batch, 
            key_rescale=None, class_label=0, tag=["others"] * nframe_total
        )
        
        refined_preds = output.images[cond_num:]  
        
        original_rgb_np = (pred_rgb.cpu().permute(0, 2, 3, 1).float().numpy() + 1.0) / 2.0
        original_rgb_np = np.clip(original_rgb_np, 0.0, 1.0)
        
        mask_np = blur_mask.cpu().permute(0, 2, 3, 1).float().numpy()
        if mask_np.shape[0] == 1 and original_rgb_np.shape[0] > 1:
            mask_np = np.repeat(mask_np, original_rgb_np.shape[0], axis=0)

        MAX_MASK_STRENGTH = 0.35 
        alpha_linear = np.clip(mask_np / MAX_MASK_STRENGTH, 0.0, 1.0)
        
        alpha_blend = alpha_linear * alpha_linear * (3.0 - 2.0 * alpha_linear)
        
        def get_low_freq(img_batch, ksize=21, sigma=5.0):
            """Extract low-frequency image structure with Gaussian blur."""
            low_freq = np.zeros_like(img_batch)
            for i in range(img_batch.shape[0]):
                low_freq[i] = cv2.GaussianBlur(img_batch[i].astype(np.float32), (ksize, ksize), sigma)
            return low_freq

        low_freq_orig = get_low_freq(original_rgb_np, ksize=21, sigma=5.0)
        
        low_freq_diff = get_low_freq(refined_preds, ksize=21, sigma=5.0)
        high_freq_diff = refined_preds - low_freq_diff
        
        frequency_blended_preds = np.clip(low_freq_orig + high_freq_diff, 0.0, 1.0)

        final_blended_preds = frequency_blended_preds * alpha_blend + original_rgb_np * (1.0 - alpha_blend)
        
        print_stats("BLENDED OUTPUT", torch.from_numpy(final_blended_preds))

        gt_rgb, gt_path = load_gt_rgb(input_data_dir, base_name, (H, W))
        eval_mask, eval_mask_source = load_target_eval_mask(data, input_data_dir, base_name, (H, W))
        metrics_row = None

        if HAS_METRICS:
            if gt_rgb is None:
                print(f"  [WARNING] GT image not found ({gt_path}); metrics skipped.")
            else:
                metrics_row = evaluate_refinement_image(
                    base_name=base_name,
                    gt_rgb=gt_rgb,
                    before_rgb=original_rgb_np[0],
                    final_rgb=final_blended_preds[0],
                    soft_mask=mask_np[0, :, :, 0],
                    eval_mask=eval_mask,
                    eval_mask_source=eval_mask_source,
                    metric_runner=metric_runner,
                    repair_mask_threshold=repair_mask_threshold,
                )
                eval_results.append(metrics_row)
                print(f"  [DTU eval] {format_metric_summary(metrics_row, 'dtu_eval')} | mask={eval_mask_source}")
                if metrics_row.get("repair_mask_valid") == 1:
                    print(
                        f"  [Repair mask] {format_metric_summary(metrics_row, 'repair_mask')} "
                        f"| pixels={metrics_row['repair_mask_pixels']}"
                    )
                else:
                    print("  [Repair mask] Region is too small; masked metrics skipped.")

        save_refinement_visuals(
            output_dir,
            base_name,
            gt_rgb,
            original_rgb_np,
            final_blended_preds,
            mask_np,
            eval_mask,
            metrics_row=metrics_row,
            eval_mask_source=eval_mask_source,
        )

    if HAS_METRICS and len(eval_results) > 0:
        report_path, csv_path = write_refinement_reports(eval_results, output_dir)
        print(f"\nEvaluation report saved to: {report_path}")
        print(f"Per-image metrics CSV saved to: {csv_path}")
    else:
        print(f"\nRun complete. Refined images were written to: {output_dir}")

if __name__ == "__main__":
    main()
