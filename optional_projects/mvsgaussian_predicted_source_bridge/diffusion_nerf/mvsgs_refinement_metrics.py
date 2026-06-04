import math
import os

import cv2
import numpy as np
import torch


try:
    import lpips
    import piq
    import pyiqa
    from skimage.metrics import structural_similarity as compute_ssim

    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False


def load_metric_models(device):
    if not HAS_METRICS:
        print("[WARNING] Missing one or more metric packages: lpips, piq, pyiqa, scikit-image.")
        return None

    models = {"lpips_vgg": lpips.LPIPS(net="vgg").to(device).eval(), "niqe": None}
    try:
        models["niqe"] = pyiqa.create_metric("niqe", device=device)
    except Exception as exc:
        print(f"[WARNING] NIQE metric unavailable, writing NIQE as nan: {exc}")
    return models


def _to_original_rgb(pred_rgb):
    original = (pred_rgb.detach().cpu().permute(0, 2, 3, 1).float().numpy() + 1.0) / 2.0
    return np.clip(original, 0.0, 1.0)


def _to_mask(blur_mask, target_count):
    mask = blur_mask.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    if mask.shape[0] == 1 and target_count > 1:
        mask = np.repeat(mask, target_count, axis=0)
    return mask


def _safe_float(value):
    try:
        value = float(value)
    except Exception:
        return float("nan")
    if math.isfinite(value):
        return value
    return float("nan")


def _read_pfm(path):
    with open(path, "rb") as f:
        header = f.readline().decode("ascii").rstrip()
        if header not in ("PF", "Pf"):
            raise ValueError(f"Not a PFM file: {path}")
        color = header == "PF"

        line = f.readline().decode("ascii").strip()
        while line.startswith("#"):
            line = f.readline().decode("ascii").strip()
        width, height = map(int, line.split())

        scale = float(f.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f")
        shape = (height, width, 3) if color else (height, width)
        return np.flipud(np.reshape(data, shape)).astype(np.float32)


def _find_project_root(input_dir):
    candidates = [
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        os.getcwd(),
    ]
    path = os.path.abspath(input_dir)
    while True:
        candidates.append(path)
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent

    seen = set()
    for root in candidates:
        if root in seen:
            continue
        seen.add(root)
        if os.path.exists(os.path.join(root, "mvs_training", "dtu", "Depths_raw")):
            return root
    return os.path.dirname(os.path.dirname(os.path.abspath(input_dir)))


def _dtu_valid_mask(base_name, input_dir, target_shape):
    parts = base_name.split("_")
    if len(parts) < 2:
        return None, "cannot parse DTU scene/view from base name"

    scene = parts[0]
    try:
        view = int(parts[1])
    except ValueError:
        return None, "cannot parse DTU target view from base name"

    project_root = _find_project_root(input_dir)
    dpt_path = os.path.join(project_root, "mvs_training", "dtu", "Depths_raw", scene, f"depth_map_{view:04d}.pfm")
    if not os.path.exists(dpt_path):
        return None, f"missing DTU depth: {dpt_path}"

    try:
        depth = _read_pfm(dpt_path)
    except Exception as exc:
        return None, f"failed to read DTU depth: {exc}"

    depth = cv2.resize(depth, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    depth = depth[44:556, 80:720]
    if depth.shape != target_shape[:2]:
        return None, f"DTU mask shape mismatch: {depth.shape} vs {target_shape[:2]}"
    mask = depth > 0.0
    if mask.sum() <= 20:
        return None, "DTU valid mask too small"
    return mask, ""


def _zero_outside_mask(image, mask):
    masked = image.copy()
    masked[~mask] = 0.0
    return masked


def evaluate_masked_region(base_name, input_dir, final_preds, pred_rgb, blur_mask, device, metric_models):
    record = {"name": base_name, "skipped": False, "reason": ""}
    if metric_models is None:
        record.update({"skipped": True, "reason": "metric packages unavailable"})
        return record

    gt_path = os.path.join(input_dir, f"{base_name}_0_gt_rgb.png")
    if not os.path.exists(gt_path):
        record.update({"skipped": True, "reason": f"missing GT: {gt_path}"})
        return record

    gt_bgr = cv2.imread(gt_path)
    if gt_bgr is None:
        record.update({"skipped": True, "reason": f"failed to read GT: {gt_path}"})
        return record

    gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    original = _to_original_rgb(pred_rgb)
    mask = _to_mask(blur_mask, original.shape[0])
    pred_diff = np.clip(final_preds[0], 0.0, 1.0)
    pred_orig = original[0]
    bool_mask = mask[0, :, :, 0] > 0.01

    if gt_rgb.shape != pred_diff.shape or bool_mask.sum() <= 20:
        record.update({"skipped": True, "reason": "GT shape mismatch or mask region too small"})
        return record

    full_mse_diff = ((gt_rgb - pred_diff) ** 2).mean()
    full_mse_orig = ((gt_rgb - pred_orig) ** 2).mean()
    global_psnr_diff = 10.0 * np.log10(1.0 / (full_mse_diff + 1e-8))
    global_psnr_orig = 10.0 * np.log10(1.0 / (full_mse_orig + 1e-8))
    global_ssim_orig = compute_ssim(gt_rgb, pred_orig, data_range=1.0, channel_axis=-1)
    global_ssim_diff = compute_ssim(gt_rgb, pred_diff, data_range=1.0, channel_axis=-1)

    dtu_mask, dtu_reason = _dtu_valid_mask(base_name, input_dir, gt_rgb.shape)
    if dtu_mask is not None:
        dtu_mse_diff = ((gt_rgb - pred_diff) ** 2)[dtu_mask].mean()
        dtu_mse_orig = ((gt_rgb - pred_orig) ** 2)[dtu_mask].mean()
        dtu_psnr_diff = 10.0 * np.log10(1.0 / (dtu_mse_diff + 1e-8))
        dtu_psnr_orig = 10.0 * np.log10(1.0 / (dtu_mse_orig + 1e-8))
        dtu_gt = _zero_outside_mask(gt_rgb, dtu_mask)
        dtu_diff = _zero_outside_mask(pred_diff, dtu_mask)
        dtu_orig = _zero_outside_mask(pred_orig, dtu_mask)
        # Match MVSGaussian's official evaluator: skimage 0.19 inferred data_range=2
        # for float images when the code omitted data_range.
        dtu_ssim_orig = compute_ssim(dtu_gt, dtu_orig, data_range=2.0, channel_axis=-1)
        dtu_ssim_diff = compute_ssim(dtu_gt, dtu_diff, data_range=2.0, channel_axis=-1)
    else:
        dtu_gt = dtu_diff = dtu_orig = None
        dtu_psnr_diff = dtu_psnr_orig = float("nan")
        dtu_ssim_orig = dtu_ssim_diff = float("nan")

    mse_diff = ((gt_rgb - pred_diff) ** 2)[bool_mask].mean()
    mse_orig = ((gt_rgb - pred_orig) ** 2)[bool_mask].mean()
    psnr_diff = 10.0 * np.log10(1.0 / (mse_diff + 1e-8))
    psnr_orig = 10.0 * np.log10(1.0 / (mse_orig + 1e-8))

    _, ssim_map_diff = compute_ssim(gt_rgb, pred_diff, data_range=1.0, channel_axis=-1, full=True)
    _, ssim_map_orig = compute_ssim(gt_rgb, pred_orig, data_range=1.0, channel_axis=-1, full=True)
    ssim_diff = ssim_map_diff[bool_mask].mean()
    ssim_orig = ssim_map_orig[bool_mask].mean()

    coords = np.argwhere(bool_mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1
    gt_crop = gt_rgb[y_min:y_max, x_min:x_max]
    diff_crop = pred_diff[y_min:y_max, x_min:x_max]
    orig_crop = pred_orig[y_min:y_max, x_min:x_max]

    gt_tensor = torch.from_numpy(gt_crop).permute(2, 0, 1).unsqueeze(0).to(device).clamp(0, 1)
    diff_tensor = torch.from_numpy(diff_crop).permute(2, 0, 1).unsqueeze(0).to(device).clamp(0, 1)
    orig_tensor = torch.from_numpy(orig_crop).permute(2, 0, 1).unsqueeze(0).to(device).clamp(0, 1)
    full_gt_tensor = torch.from_numpy(gt_rgb).permute(2, 0, 1).unsqueeze(0).to(device).clamp(0, 1)
    full_diff_tensor = torch.from_numpy(pred_diff).permute(2, 0, 1).unsqueeze(0).to(device).clamp(0, 1)
    full_orig_tensor = torch.from_numpy(pred_orig).permute(2, 0, 1).unsqueeze(0).to(device).clamp(0, 1)
    if dtu_mask is not None:
        dtu_gt_tensor = torch.from_numpy(dtu_gt).permute(2, 0, 1).unsqueeze(0).to(device).clamp(0, 1)
        dtu_diff_tensor = torch.from_numpy(dtu_diff).permute(2, 0, 1).unsqueeze(0).to(device).clamp(0, 1)
        dtu_orig_tensor = torch.from_numpy(dtu_orig).permute(2, 0, 1).unsqueeze(0).to(device).clamp(0, 1)
    else:
        dtu_gt_tensor = dtu_diff_tensor = dtu_orig_tensor = None

    with torch.no_grad():
        lpips_diff = metric_models["lpips_vgg"](diff_tensor * 2.0 - 1.0, gt_tensor * 2.0 - 1.0).item()
        lpips_orig = metric_models["lpips_vgg"](orig_tensor * 2.0 - 1.0, gt_tensor * 2.0 - 1.0).item()
        fsim_diff = piq.fsim(diff_tensor, gt_tensor, data_range=1.0).item()
        fsim_orig = piq.fsim(orig_tensor, gt_tensor, data_range=1.0).item()
        global_lpips_diff = metric_models["lpips_vgg"](full_diff_tensor * 2.0 - 1.0, full_gt_tensor * 2.0 - 1.0).item()
        global_lpips_orig = metric_models["lpips_vgg"](full_orig_tensor * 2.0 - 1.0, full_gt_tensor * 2.0 - 1.0).item()
        global_fsim_diff = piq.fsim(full_diff_tensor, full_gt_tensor, data_range=1.0).item()
        global_fsim_orig = piq.fsim(full_orig_tensor, full_gt_tensor, data_range=1.0).item()
        if dtu_mask is not None:
            dtu_lpips_diff = metric_models["lpips_vgg"](dtu_diff_tensor * 2.0 - 1.0, dtu_gt_tensor * 2.0 - 1.0).item()
            dtu_lpips_orig = metric_models["lpips_vgg"](dtu_orig_tensor * 2.0 - 1.0, dtu_gt_tensor * 2.0 - 1.0).item()
            dtu_fsim_diff = piq.fsim(dtu_diff_tensor, dtu_gt_tensor, data_range=1.0).item()
            dtu_fsim_orig = piq.fsim(dtu_orig_tensor, dtu_gt_tensor, data_range=1.0).item()
        else:
            dtu_lpips_diff = dtu_lpips_orig = float("nan")
            dtu_fsim_diff = dtu_fsim_orig = float("nan")
        if metric_models.get("niqe") is None:
            niqe_diff = float("nan")
            niqe_orig = float("nan")
            global_niqe_diff = float("nan")
            global_niqe_orig = float("nan")
            dtu_niqe_diff = float("nan")
            dtu_niqe_orig = float("nan")
        else:
            try:
                niqe_diff = metric_models["niqe"](diff_tensor).item()
                niqe_orig = metric_models["niqe"](orig_tensor).item()
            except Exception:
                niqe_diff = metric_models["niqe"](full_diff_tensor).item()
                niqe_orig = metric_models["niqe"](full_orig_tensor).item()
            global_niqe_diff = metric_models["niqe"](full_diff_tensor).item()
            global_niqe_orig = metric_models["niqe"](full_orig_tensor).item()
            if dtu_mask is not None:
                dtu_niqe_diff = metric_models["niqe"](dtu_diff_tensor).item()
                dtu_niqe_orig = metric_models["niqe"](dtu_orig_tensor).item()
            else:
                dtu_niqe_diff = float("nan")
                dtu_niqe_orig = float("nan")

    record.update({
        "psnr_orig": _safe_float(psnr_orig),
        "psnr_diff": _safe_float(psnr_diff),
        "ssim_orig": _safe_float(ssim_orig),
        "ssim_diff": _safe_float(ssim_diff),
        "lpips_orig": _safe_float(lpips_orig),
        "lpips_diff": _safe_float(lpips_diff),
        "fsim_orig": _safe_float(fsim_orig),
        "fsim_diff": _safe_float(fsim_diff),
        "niqe_orig": _safe_float(niqe_orig),
        "niqe_diff": _safe_float(niqe_diff),
        "global_psnr_orig": _safe_float(global_psnr_orig),
        "global_psnr_diff": _safe_float(global_psnr_diff),
        "global_ssim_orig": _safe_float(global_ssim_orig),
        "global_ssim_diff": _safe_float(global_ssim_diff),
        "global_lpips_orig": _safe_float(global_lpips_orig),
        "global_lpips_diff": _safe_float(global_lpips_diff),
        "global_fsim_orig": _safe_float(global_fsim_orig),
        "global_fsim_diff": _safe_float(global_fsim_diff),
        "global_niqe_orig": _safe_float(global_niqe_orig),
        "global_niqe_diff": _safe_float(global_niqe_diff),
        "dtu_valid": dtu_mask is not None,
        "dtu_reason": dtu_reason,
        "dtu_psnr_orig": _safe_float(dtu_psnr_orig),
        "dtu_psnr_diff": _safe_float(dtu_psnr_diff),
        "dtu_ssim_orig": _safe_float(dtu_ssim_orig),
        "dtu_ssim_diff": _safe_float(dtu_ssim_diff),
        "dtu_lpips_orig": _safe_float(dtu_lpips_orig),
        "dtu_lpips_diff": _safe_float(dtu_lpips_diff),
        "dtu_fsim_orig": _safe_float(dtu_fsim_orig),
        "dtu_fsim_diff": _safe_float(dtu_fsim_diff),
        "dtu_niqe_orig": _safe_float(dtu_niqe_orig),
        "dtu_niqe_diff": _safe_float(dtu_niqe_diff),
    })
    return record


def _mean(records, key):
    values = [r[key] for r in records if not r.get("skipped") and math.isfinite(r.get(key, float("nan")))]
    if not values:
        return float("nan")
    return float(np.mean(values))


def _fmt(value):
    if not math.isfinite(value):
        return "nan"
    return f"{value:.4f}"


def write_masked_report(output_dir, records, title):
    report_path = os.path.join(output_dir, "masked_region_refinement_report.txt")
    valid = [r for r in records if not r.get("skipped")]
    skipped = [r for r in records if r.get("skipped")]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"=== {title} ===\n")
        f.write("This report includes repair-mask, full-image, and DTU valid-depth-mask metrics.\n")
        f.write("The DTU valid-mask protocol follows the MVSGaussian evaluator: PSNR is computed inside the valid mask, while SSIM/LPIPS/FSIM/NIQE are computed after zeroing pixels outside the mask; SSIM uses data_range=2 for compatibility with the legacy skimage setup.\n")
        f.write(f"Total samples: {len(records)}\n")
        f.write(f"Evaluated samples: {len(valid)}\n")
        f.write(f"Skipped samples: {len(skipped)}\n\n")

        if valid:
            def write_summary(section_title, prefix):
                avg_p_orig = _mean(valid, f"{prefix}psnr_orig")
                avg_p_diff = _mean(valid, f"{prefix}psnr_diff")
                avg_s_orig = _mean(valid, f"{prefix}ssim_orig")
                avg_s_diff = _mean(valid, f"{prefix}ssim_diff")
                avg_l_orig = _mean(valid, f"{prefix}lpips_orig")
                avg_l_diff = _mean(valid, f"{prefix}lpips_diff")
                avg_f_orig = _mean(valid, f"{prefix}fsim_orig")
                avg_f_diff = _mean(valid, f"{prefix}fsim_diff")
                avg_n_orig = _mean(valid, f"{prefix}niqe_orig")
                avg_n_diff = _mean(valid, f"{prefix}niqe_diff")

                f.write(f"{section_title}\n")
                f.write(f"PSNR  (higher is better): {_fmt(avg_p_orig)}  ->  {_fmt(avg_p_diff)} (delta: {_fmt(avg_p_diff - avg_p_orig)})\n")
                f.write(f"SSIM  (higher is better): {_fmt(avg_s_orig)}  ->  {_fmt(avg_s_diff)} (delta: {_fmt(avg_s_diff - avg_s_orig)})\n")
                f.write(f"FSIM  (higher is better): {_fmt(avg_f_orig)}  ->  {_fmt(avg_f_diff)} (delta: {_fmt(avg_f_diff - avg_f_orig)})\n")
                f.write(f"LPIPS (lower is better): {_fmt(avg_l_orig)}  ->  {_fmt(avg_l_diff)} (delta: {_fmt(avg_l_orig - avg_l_diff)})\n")
                f.write(f"NIQE  (lower is better): {_fmt(avg_n_orig)}  ->  {_fmt(avg_n_diff)} (delta: {_fmt(avg_n_orig - avg_n_diff)})\n\n")

            write_summary("Mask-region average metrics (MVSGaussian original -> diffusion refined)", "")
            write_summary("Full-image average metrics (MVSGaussian original -> diffusion refined)", "global_")
            write_summary("DTU valid-mask average metrics (MVSGaussian original -> diffusion refined)", "dtu_")

            f.write("-" * 60 + "\n\n")
            f.write("Per-view metrics\n")
            for res in valid:
                f.write(f"View: {res['name']}\n")
                f.write("  Mask region:\n")
                f.write(f"    PSNR  : {_fmt(res['psnr_orig'])} -> {_fmt(res['psnr_diff'])}\n")
                f.write(f"    SSIM  : {_fmt(res['ssim_orig'])} -> {_fmt(res['ssim_diff'])}\n")
                f.write(f"    FSIM  : {_fmt(res['fsim_orig'])} -> {_fmt(res['fsim_diff'])}\n")
                f.write(f"    LPIPS : {_fmt(res['lpips_orig'])} -> {_fmt(res['lpips_diff'])}\n")
                f.write(f"    NIQE  : {_fmt(res['niqe_orig'])} -> {_fmt(res['niqe_diff'])}\n")
                f.write("  Full image:\n")
                f.write(f"    PSNR  : {_fmt(res['global_psnr_orig'])} -> {_fmt(res['global_psnr_diff'])}\n")
                f.write(f"    SSIM  : {_fmt(res['global_ssim_orig'])} -> {_fmt(res['global_ssim_diff'])}\n")
                f.write(f"    FSIM  : {_fmt(res['global_fsim_orig'])} -> {_fmt(res['global_fsim_diff'])}\n")
                f.write(f"    LPIPS : {_fmt(res['global_lpips_orig'])} -> {_fmt(res['global_lpips_diff'])}\n")
                f.write(f"    NIQE  : {_fmt(res['global_niqe_orig'])} -> {_fmt(res['global_niqe_diff'])}\n\n")
                f.write("  DTU valid mask:\n")
                if res.get("dtu_valid"):
                    f.write(f"    PSNR  : {_fmt(res['dtu_psnr_orig'])} -> {_fmt(res['dtu_psnr_diff'])}\n")
                    f.write(f"    SSIM  : {_fmt(res['dtu_ssim_orig'])} -> {_fmt(res['dtu_ssim_diff'])}\n")
                    f.write(f"    FSIM  : {_fmt(res['dtu_fsim_orig'])} -> {_fmt(res['dtu_fsim_diff'])}\n")
                    f.write(f"    LPIPS : {_fmt(res['dtu_lpips_orig'])} -> {_fmt(res['dtu_lpips_diff'])}\n")
                    f.write(f"    NIQE  : {_fmt(res['dtu_niqe_orig'])} -> {_fmt(res['dtu_niqe_diff'])}\n\n")
                else:
                    f.write(f"    skipped: {res.get('dtu_reason', '')}\n\n")
        else:
            f.write("No valid reference samples were available for evaluation.\n\n")

        if skipped:
            f.write("Skipped samples\n")
            for res in skipped:
                f.write(f"{res['name']}: {res.get('reason', '')}\n")

    return report_path
