import csv
import math
import os

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

try:
    import piq

    HAS_PIQ = True
except ImportError:
    HAS_PIQ = False

MAX_MASK_STRENGTH = 0.35
MIN_MASK_PIXELS = 20
METRIC_NAMES = ("psnr", "ssim", "fsim", "lpips", "niqe")
LOWER_BETTER = {"lpips", "niqe"}


def save_rgb_png(path, rgb):
    rgb = np.clip(rgb, 0.0, 1.0)
    bgr = (rgb * 255.0).round().astype(np.uint8)[:, :, ::-1]
    cv2.imwrite(path, bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def _to_uint8(rgb):
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _to_tensor01(rgb, device):
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device).float().clamp(0, 1)


def _metric_gain(metric, orig, refined):
    if metric in LOWER_BETTER:
        return float(orig) - float(refined)
    return float(refined) - float(orig)


def _metric_direction(gain):
    if isinstance(gain, (float, int)) and math.isfinite(float(gain)) and float(gain) >= 0.0:
        return "better"
    return "worse"


def _safe_niqe(niqe_metric, pred_tensor):
    if niqe_metric is None:
        return float("nan")
    try:
        with torch.no_grad():
            return float(niqe_metric(pred_tensor).item())
    except Exception:
        return float("nan")


def load_gt_rgb(input_data_dir, base_name, size):
    gt_path = os.path.join(input_data_dir, f"{base_name}_0_gt_rgb.png")
    if not os.path.exists(gt_path):
        return None, gt_path

    gt_bgr = cv2.imread(gt_path, cv2.IMREAD_COLOR)
    if gt_bgr is None:
        return None, gt_path

    gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if gt_rgb.shape[:2] != size:
        gt_rgb = cv2.resize(gt_rgb, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return np.clip(gt_rgb, 0.0, 1.0), gt_path


def _mask_array(mask):
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().float().numpy()
    arr = np.asarray(mask, dtype=np.float32)
    while arr.ndim > 2:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = arr[0]
    return arr


def _resize_mask(mask, size):
    mask = _mask_array(mask)
    if mask.shape != size:
        mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def load_target_eval_mask(data, input_data_dir, base_name, size):
    for key in ("target_eval_mask", "dtu_eval_mask", "eval_mask", "target_mask"):
        if key in data:
            mask = _resize_mask(data[key], size) >= 0.5
            return mask, key

    for suffix in ("_9_dtu_eval_mask.png", "_9_eval_mask.png"):
        mask_path = os.path.join(input_data_dir, f"{base_name}{suffix}")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = _resize_mask(mask, size) >= 127
                return mask, os.path.basename(mask_path)

    return np.ones(size, dtype=bool), "full_image_fallback"


def repair_mask_from_soft_mask(mask_np, size, threshold):
    mask = _resize_mask(mask_np, size)
    return mask > threshold


def _bbox_from_mask(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return int(y0), int(y1), int(x0), int(x1)


def _crop_for_mask(rgb, mask):
    bbox = _bbox_from_mask(mask)
    if bbox is None:
        return rgb
    y0, y1, x0, x1 = bbox
    return rgb[y0:y1, x0:x1]


def _mask_to_heat(mask_np, size):
    mask = _resize_mask(mask_np, size)
    mask_disp = np.clip(mask / MAX_MASK_STRENGTH, 0.0, 1.0)
    mask_disp = mask_disp ** 2.0
    return cv2.applyColorMap((mask_disp * 255.0).astype(np.uint8), cv2.COLORMAP_JET), mask_disp


def make_mask_overlay_rgb(rgb, mask_np):
    heatmap_bgr, mask_disp = _mask_to_heat(mask_np, rgb.shape[:2])
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    alpha = np.clip(mask_disp * 0.70, 0.0, 0.70)[..., None]
    return np.clip(rgb * (1.0 - alpha) + heatmap_rgb * alpha, 0.0, 1.0)


def _label_panel(panel, label):
    panel_u8 = _to_uint8(panel)
    header = np.zeros((34, panel_u8.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([header, panel_u8], axis=0)


def _draw_text_block(lines, width):
    line_h = 24
    block = np.full((20 + line_h * len(lines), width, 3), 255, dtype=np.uint8)
    y = 28
    for line in lines:
        color = (30, 30, 30)
        if "better" in line:
            color = (20, 110, 35)
        elif "worse" in line:
            color = (170, 45, 45)
        cv2.putText(block, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += line_h
    return block


def _metric_line(row, prefix, metric):
    orig = row.get(f"{prefix}_{metric}_orig")
    refined = row.get(f"{prefix}_{metric}_refined")
    gain = row.get(f"{prefix}_{metric}_gain")
    if orig is None or refined is None or gain is None:
        return f"{metric.upper():<5} skipped"
    if not all(math.isfinite(float(x)) for x in (orig, refined, gain)):
        return f"{metric.upper():<5} {float(orig):>8.4f} -> {float(refined):>8.4f}  n/a"
    return (
        f"{metric.upper():<5} {float(orig):>8.4f} -> {float(refined):>8.4f}  "
        f"{float(gain):+.4f} {_metric_direction(gain)}"
    )


def _format_visual_lines(row):
    lines = []
    for title, prefix in (
        ("global", "full_image"),
        ("dtu_eval", "dtu_eval"),
        ("repair_mask", "repair_mask"),
    ):
        if row.get(f"{prefix}_valid") != 1:
            lines.append(f"[{title}] skipped")
            continue
        lines.append(f"[{title}]")
        for metric in METRIC_NAMES:
            lines.append(_metric_line(row, prefix, metric))
    return lines


def make_per_image_visual(base_name, gt_rgb, before_rgb, final_rgb, mask_overlay_rgb, metrics_row, eval_mask_source):
    panels = [
        _label_panel(gt_rgb, "GT"),
        _label_panel(before_rgb, "Original pred"),
        _label_panel(final_rgb, "Refined"),
        _label_panel(mask_overlay_rgb, "Repair mask overlay"),
    ]
    max_h = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] == max_h:
            padded.append(panel)
            continue
        canvas = np.full((max_h, panel.shape[1], 3), 245, dtype=np.uint8)
        canvas[: panel.shape[0], : panel.shape[1]] = panel
        padded.append(canvas)
    top = np.concatenate(padded, axis=1)

    title = np.full((46, top.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        title,
        f"{base_name} | dtu_mask={eval_mask_source}",
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )

    text = _draw_text_block(_format_visual_lines(metrics_row), top.shape[1])
    return np.concatenate([title, top, text], axis=0)


def save_montage_pages(visual_paths, out_dir, per_page=8):
    if not visual_paths:
        return
    per_page = max(1, int(per_page))
    os.makedirs(out_dir, exist_ok=True)
    for page_idx, start in enumerate(range(0, len(visual_paths), per_page)):
        imgs = []
        for path in visual_paths[start : start + per_page]:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                imgs.append(img)
        if not imgs:
            continue
        width = max(img.shape[1] for img in imgs)
        padded = []
        for img in imgs:
            if img.shape[1] == width:
                padded.append(img)
                continue
            canvas = np.full((img.shape[0], width, 3), 255, dtype=np.uint8)
            canvas[:, : img.shape[1]] = img
            padded.append(canvas)
        page = np.concatenate(padded, axis=0)
        cv2.imwrite(os.path.join(out_dir, f"sample_grid_{page_idx:02d}.png"), page, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def save_refinement_visuals(
    output_dir,
    base_name,
    gt_rgb,
    before_rgb,
    final_rgb,
    mask_np,
    eval_mask,
    metrics_row=None,
    eval_mask_source=None,
):
    os.makedirs(output_dir, exist_ok=True)

    if gt_rgb is not None:
        save_rgb_png(os.path.join(output_dir, f"{base_name}_gt_rgb.png"), gt_rgb)

    if eval_mask is not None:
        eval_mask_u8 = (eval_mask.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_dtu_eval_mask.png"), eval_mask_u8)

    num_targets = final_rgb.shape[0]
    for j in range(num_targets):
        suffix = f"{j:02d}"
        before = np.clip(before_rgb[j], 0.0, 1.0)
        final = np.clip(final_rgb[j], 0.0, 1.0)

        save_rgb_png(os.path.join(output_dir, f"{base_name}_v_pred_before_{suffix}.png"), before)
        save_rgb_png(os.path.join(output_dir, f"{base_name}_v_pred_blended_{suffix}.png"), final)

        mask_j = mask_np[j, :, :, 0] if mask_np.ndim == 4 else mask_np
        heatmap, mask_disp = _mask_to_heat(mask_j, before.shape[:2])
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask_heat_{suffix}.png"), heatmap)

        before_bgr = (before * 255.0).round().astype(np.uint8)[:, :, ::-1]
        alpha = np.clip(mask_disp * 0.70, 0.0, 0.70)[..., None]
        overlay = (before_bgr * (1.0 - alpha) + heatmap * alpha).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask_overlay_{suffix}.png"), overlay)

        if gt_rgb is not None and metrics_row is not None and j == 0:
            visual_dir = os.path.join(output_dir, "visual_compare")
            os.makedirs(visual_dir, exist_ok=True)
            visual = make_per_image_visual(
                base_name,
                gt_rgb,
                before,
                final,
                make_mask_overlay_rgb(before, mask_j),
                metrics_row,
                eval_mask_source or metrics_row.get("eval_mask_source", "unknown"),
            )
            visual_bgr = cv2.cvtColor(visual, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(visual_dir, f"{base_name}_gt_pred_refined_mask_metrics.png"),
                visual_bgr,
                [cv2.IMWRITE_PNG_COMPRESSION, 0],
            )


class EvalAlignedMetricRunner:
    def __init__(self, device, lpips_model, niqe_metric=None):
        self.device = device
        self.lpips_model = lpips_model
        self.niqe_metric = niqe_metric

    def compute(self, gt_rgb, pred_rgb, mask=None, masked_image_mode="zero"):
        if mask is None:
            mask = np.ones(gt_rgb.shape[:2], dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
        if gt_rgb.shape != pred_rgb.shape or gt_rgb.shape[:2] != mask.shape:
            raise ValueError(
                f"Metric shape mismatch: gt={gt_rgb.shape}, pred={pred_rgb.shape}, mask={mask.shape}"
            )
        if int(mask.sum()) <= MIN_MASK_PIXELS:
            return None

        psnr_value = compute_psnr(gt_rgb[mask], pred_rgb[mask], data_range=1.0)
        if masked_image_mode == "zero":
            gt_eval = gt_rgb.copy()
            pred_eval = pred_rgb.copy()
            gt_eval[~mask] = 0.0
            pred_eval[~mask] = 0.0
            ssim_value = compute_ssim(gt_eval, pred_eval, channel_axis=-1, data_range=1.0)
        elif masked_image_mode == "crop":
            gt_eval = _crop_for_mask(gt_rgb, mask)
            pred_eval = _crop_for_mask(pred_rgb, mask)
            _, ssim_map = compute_ssim(gt_rgb, pred_rgb, data_range=1.0, channel_axis=-1, full=True)
            ssim_value = float(np.mean(ssim_map[mask]))
        else:
            raise ValueError(f"Unknown masked_image_mode: {masked_image_mode}")

        gt_tensor = _to_tensor01(gt_eval, self.device)
        pred_tensor = _to_tensor01(pred_eval, self.device)
        with torch.no_grad():
            lpips_value = self.lpips_model(
                gt_tensor * 2.0 - 1.0,
                pred_tensor * 2.0 - 1.0,
            ).item()
            fsim_value = piq.fsim(pred_tensor, gt_tensor, data_range=1.0).item() if HAS_PIQ else float("nan")

        return {
            "psnr": float(psnr_value),
            "ssim": float(ssim_value),
            "fsim": float(fsim_value),
            "lpips": float(lpips_value),
            "niqe": _safe_niqe(self.niqe_metric, pred_tensor),
        }


def _put_metric_pair(row, prefix, orig_metrics, refined_metrics):
    if orig_metrics is None or refined_metrics is None:
        row[f"{prefix}_valid"] = 0
        return

    row[f"{prefix}_valid"] = 1
    for metric in METRIC_NAMES:
        row[f"{prefix}_{metric}_orig"] = orig_metrics[metric]
        row[f"{prefix}_{metric}_refined"] = refined_metrics[metric]
        row[f"{prefix}_{metric}_gain"] = _metric_gain(metric, orig_metrics[metric], refined_metrics[metric])


def evaluate_refinement_image(
    base_name,
    gt_rgb,
    before_rgb,
    final_rgb,
    soft_mask,
    eval_mask,
    eval_mask_source,
    metric_runner,
    repair_mask_threshold,
):
    size = gt_rgb.shape[:2]
    repair_mask = repair_mask_from_soft_mask(soft_mask, size, repair_mask_threshold)
    eval_mask = np.asarray(eval_mask, dtype=bool)
    repair_eval_mask = repair_mask & eval_mask

    row = {
        "name": base_name,
        "eval_mask_source": eval_mask_source,
        "eval_mask_pixels": int(eval_mask.sum()),
        "repair_mask_pixels": int(repair_mask.sum()),
        "repair_eval_mask_pixels": int(repair_eval_mask.sum()),
        "repair_mask_threshold": float(repair_mask_threshold),
    }

    for prefix, mask, mode in (
        ("dtu_eval", eval_mask, "zero"),
        ("repair_mask", repair_mask, "crop"),
        ("full_image", None, "zero"),
    ):
        orig_metrics = metric_runner.compute(gt_rgb, before_rgb, mask, masked_image_mode=mode)
        refined_metrics = metric_runner.compute(gt_rgb, final_rgb, mask, masked_image_mode=mode)
        _put_metric_pair(row, prefix, orig_metrics, refined_metrics)

    return row


def _finite_values(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (float, int)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _mean(rows, key):
    values = _finite_values(rows, key)
    if not values:
        return None
    return float(np.mean(values))


def _write_metric_section(f, rows, prefix, title):
    f.write(f"[{title}]\n")
    valid_rows = [row for row in rows if row.get(f"{prefix}_valid") == 1]
    f.write(f"valid_images: {len(valid_rows)} / {len(rows)}\n")
    for metric in METRIC_NAMES:
        orig = _mean(valid_rows, f"{prefix}_{metric}_orig")
        refined = _mean(valid_rows, f"{prefix}_{metric}_refined")
        gain = _mean(valid_rows, f"{prefix}_{metric}_gain")
        if orig is None or refined is None or gain is None:
            f.write(f"{metric.upper()}: n/a\n")
            continue
        better = "lower is better" if metric in LOWER_BETTER else "higher is better"
        f.write(
            f"{metric.upper():5s} ({better}): "
            f"{orig:.6f} -> {refined:.6f}  gain {gain:+.6f}\n"
        )
    f.write("\n")


def write_refinement_reports(rows, output_dir, report_name="masked_region_refinement_report.txt"):
    if not rows:
        return None, None

    report_path = os.path.join(output_dir, report_name)
    csv_path = os.path.join(output_dir, "per_image_refinement_metrics.csv")

    fieldnames = [
        "name",
        "eval_mask_source",
        "eval_mask_pixels",
        "repair_mask_pixels",
        "repair_eval_mask_pixels",
        "repair_mask_threshold",
    ]
    for prefix in ("dtu_eval", "repair_mask", "full_image"):
        fieldnames.append(f"{prefix}_valid")
        for metric in METRIC_NAMES:
            fieldnames.extend([
                f"{prefix}_{metric}_orig",
                f"{prefix}_{metric}_refined",
                f"{prefix}_{metric}_gain",
            ])

    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== ADO-NeRF refinement report: global / DTU eval / repair mask ===\n")
        f.write("DTU eval rule matches evaluators/ado_nerf.py: PSNR on DTU valid pixels; SSIM/FSIM/LPIPS/NIQE on images with invalid pixels zeroed.\n")
        f.write("Repair-mask rule matches ADO visual diagnostics: PSNR/SSIM use mask pixels; FSIM/LPIPS/NIQE use the repair-mask bounding box crop.\n")
        f.write("Full-image rule computes metrics on the whole target image.\n")
        f.write("Delta convention: positive is better. LPIPS/NIQE use original - refined.\n")
        f.write("DTU eval mask is loaded from target_eval_mask exported by run_ado_export.py; full_image_fallback means the export must be regenerated for exact DTU eval numbers.\n\n")

        mask_sources = sorted({str(row.get("eval_mask_source")) for row in rows})
        f.write(f"images: {len(rows)}\n")
        f.write(f"eval_mask_sources: {', '.join(mask_sources)}\n\n")

        _write_metric_section(f, rows, "dtu_eval", "DTU eval mask / original eval protocol")
        _write_metric_section(f, rows, "repair_mask", "Repair mask / ADO visual protocol")
        _write_metric_section(f, rows, "full_image", "Full image diagnostic")

        f.write("[Per-image metric changes]\n")
        for row in rows:
            f.write(f"{row['name']}\n")
            for prefix in ("dtu_eval", "repair_mask", "full_image"):
                f.write(f"  {prefix}\n")
                if row.get(f"{prefix}_valid") != 1:
                    f.write("    skipped\n")
                    continue
                for metric in METRIC_NAMES:
                    f.write(f"    {_metric_line(row, prefix, metric)}\n")

    visual_dir = os.path.join(output_dir, "visual_compare")
    if os.path.isdir(visual_dir):
        visual_paths = sorted(
            os.path.join(visual_dir, name)
            for name in os.listdir(visual_dir)
            if name.endswith("_gt_pred_refined_mask_metrics.png")
        )
        visual_max_samples = int(os.environ.get("ADO_VIS_MAX_SAMPLES", "0") or 0)
        if visual_max_samples > 0:
            visual_paths = visual_paths[:visual_max_samples]
        visual_samples_per_page = int(os.environ.get("ADO_VIS_SAMPLES_PER_PAGE", "8") or 8)
        save_montage_pages(visual_paths, visual_dir, visual_samples_per_page)

    return report_path, csv_path


def format_metric_summary(row, prefix):
    if row.get(f"{prefix}_valid") != 1:
        return "skipped"
    return " | ".join(_metric_line(row, prefix, metric) for metric in METRIC_NAMES)
