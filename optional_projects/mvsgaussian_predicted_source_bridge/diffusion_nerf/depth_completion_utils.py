import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def to_nchw(x, device=None, dtype=None):
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if device is not None or dtype is not None:
        x = x.to(device=device, dtype=dtype)
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        if x.shape[0] in (1, 3):
            x = x.unsqueeze(0)
        else:
            x = x.unsqueeze(1)
    elif x.ndim != 4:
        raise ValueError("Expected a 2D, 3D, or 4D tensor, got {}".format(tuple(x.shape)))
    return x


def resize_nchw(x, size, mode="bilinear"):
    if x.shape[-2:] == tuple(size):
        return x
    align_corners = False if mode in ("bilinear", "bicubic") else None
    if align_corners is None:
        return F.interpolate(x, size=size, mode=mode)
    return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)


def gaussian_blur_nchw(x, k=31, sigma=9.0):
    if k % 2 == 0:
        k += 1
    if k <= 1:
        return x
    coord = torch.arange(k, dtype=x.dtype, device=x.device) - (k // 2)
    kernel_1d = torch.exp(-(coord ** 2) / (2.0 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-8)
    kernel = torch.mm(kernel_1d[:, None], kernel_1d[None, :])
    kernel = kernel[None, None].repeat(x.shape[1], 1, 1, 1)
    padded = F.pad(x, (k // 2, k // 2, k // 2, k // 2), mode="reflect")
    return F.conv2d(padded, kernel, groups=x.shape[1]).clamp(0.0, 1.0)


def confidence_from_std(mvs_depth, mvs_std, valid_mask):
    valid = valid_mask > 0.5
    if mvs_std is None:
        return valid.float()
    std = to_nchw(mvs_std, device=mvs_depth.device, dtype=mvs_depth.dtype)
    std = resize_nchw(std, mvs_depth.shape[-2:], mode="nearest")
    if valid.any():
        depth_scale = torch.median(mvs_depth[valid]).clamp_min(1e-4)
    else:
        depth_scale = mvs_depth[mvs_depth > 0].median().clamp_min(1e-4) if (mvs_depth > 0).any() else torch.tensor(1.0, device=mvs_depth.device, dtype=mvs_depth.dtype)
    conf = torch.exp(-std.abs() / (depth_scale * 0.05 + 1e-6))
    return (conf * valid.float()).clamp(0.0, 1.0)


def _weighted_inverse_fit(x, y, w, eps=1e-6):
    good = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(w) & (w > 0) & (x > eps) & (y > eps)
    if good.sum().item() < 32:
        return None
    x = x[good].float()
    y = y[good].float()
    w = w[good].float()
    w_sum = w.sum().clamp_min(eps)
    x_mean = (w * x).sum() / w_sum
    y_mean = (w * y).sum() / w_sum
    x_centered = x - x_mean
    denom = (w * x_centered * x_centered).sum().clamp_min(eps)
    scale = (w * x_centered * (y - y_mean)).sum() / denom
    shift = y_mean - scale * x_mean
    pred = scale * x + shift
    rmse = torch.sqrt((w * (pred - y) ** 2).sum() / w_sum)
    if not torch.isfinite(scale) or not torch.isfinite(shift) or not torch.isfinite(rmse):
        return None
    return scale, shift, rmse


def _median_scale_fallback(mvs_depth, mono_depth, valid_mask, eps=1e-6):
    good = (valid_mask > 0.5) & torch.isfinite(mvs_depth) & torch.isfinite(mono_depth) & (mvs_depth > eps) & (mono_depth > eps)
    if good.sum().item() < 32:
        return torch.nan_to_num(mvs_depth, nan=0.0).clamp_min(eps), {
            "align_scale": 1.0,
            "align_shift": 0.0,
            "align_mode": "fallback_mvs",
            "align_rmse": 0.0,
        }
    scale = torch.median(mvs_depth[good]) / torch.median(mono_depth[good]).clamp_min(eps)
    aligned = mono_depth * scale
    return aligned.clamp_min(eps), {
        "align_scale": float(scale.detach().cpu()),
        "align_shift": 0.0,
        "align_mode": "median_depth_scale",
        "align_rmse": 0.0,
    }


def align_mono_to_mvs_inverse_depth(mvs_depth, mono_depth, valid_mask, confidence=None, min_valid=100, eps=1e-6):
    mvs_depth = to_nchw(mvs_depth)
    mono_depth = resize_nchw(to_nchw(mono_depth, device=mvs_depth.device, dtype=mvs_depth.dtype), mvs_depth.shape[-2:])
    valid_mask = resize_nchw(to_nchw(valid_mask, device=mvs_depth.device, dtype=mvs_depth.dtype), mvs_depth.shape[-2:], mode="nearest")
    if confidence is None:
        confidence = valid_mask.float()
    else:
        confidence = resize_nchw(to_nchw(confidence, device=mvs_depth.device, dtype=mvs_depth.dtype), mvs_depth.shape[-2:], mode="nearest")

    valid = (
        (valid_mask > 0.5)
        & torch.isfinite(mvs_depth)
        & torch.isfinite(mono_depth)
        & (mvs_depth > eps)
        & (mono_depth > eps)
    )
    if valid.sum().item() < min_valid:
        return _median_scale_fallback(mvs_depth, mono_depth, valid_mask, eps=eps)

    inv_mvs = 1.0 / mvs_depth.clamp_min(eps)
    x_depth_like = 1.0 / mono_depth.clamp_min(eps)
    x_inverse_like = mono_depth.clamp_min(eps)
    w = (confidence * valid.float()).clamp_min(0.0)
    y = inv_mvs

    fit_depth = _weighted_inverse_fit(x_depth_like, y, w, eps=eps)
    fit_inverse = _weighted_inverse_fit(x_inverse_like, y, w, eps=eps)
    if fit_depth is None and fit_inverse is None:
        return _median_scale_fallback(mvs_depth, mono_depth, valid_mask, eps=eps)

    if fit_inverse is None or (fit_depth is not None and fit_depth[2] <= fit_inverse[2]):
        scale, shift, rmse = fit_depth
        x_all = x_depth_like
        mode = "depth_like"
    else:
        scale, shift, rmse = fit_inverse
        x_all = x_inverse_like
        mode = "inverse_like"

    aligned_inv = (scale * x_all + shift).clamp_min(eps)
    aligned_depth = (1.0 / aligned_inv).clamp_min(eps)
    aligned_depth = torch.nan_to_num(aligned_depth, nan=0.0, posinf=0.0, neginf=0.0)
    meta = {
        "align_scale": float(scale.detach().cpu()),
        "align_shift": float(shift.detach().cpu()),
        "align_mode": mode,
        "align_rmse": float(rmse.detach().cpu()),
    }
    return aligned_depth, meta


def fallback_mono_depth(mvs_depth, valid_mask, eps=1e-6):
    valid_depth = torch.where(valid_mask > 0.5, mvs_depth, torch.zeros_like(mvs_depth))
    inv_depth = torch.where(valid_depth > eps, 1.0 / valid_depth.clamp_min(eps), torch.zeros_like(valid_depth))
    valid_soft = gaussian_blur_nchw((valid_mask > 0.5).float(), k=51, sigma=15.0)
    inv_smooth = gaussian_blur_nchw(inv_depth, k=51, sigma=15.0)
    inv_smooth = inv_smooth / valid_soft.clamp_min(1e-4)
    fallback = torch.where(inv_smooth > eps, 1.0 / inv_smooth.clamp_min(eps), mvs_depth.clamp_min(eps))
    return torch.where(valid_mask > 0.5, mvs_depth, fallback).clamp_min(eps)


def complete_mvs_mono_depth(mvs_depth, valid_mask=None, mvs_std=None, mono_depth=None, blur_mask=None, max_mask_strength=0.35):
    mvs_depth = to_nchw(mvs_depth).float()
    device = mvs_depth.device
    dtype = mvs_depth.dtype
    if valid_mask is None:
        valid_mask = (torch.isfinite(mvs_depth) & (mvs_depth > 1e-6)).float()
    else:
        valid_mask = resize_nchw(to_nchw(valid_mask, device=device, dtype=dtype), mvs_depth.shape[-2:], mode="nearest")
        valid_mask = ((valid_mask > 0.5) & torch.isfinite(mvs_depth) & (mvs_depth > 1e-6)).float()

    mvs_confidence = confidence_from_std(mvs_depth, mvs_std, valid_mask)
    mono_source = "external"
    if mono_depth is None:
        mono_depth = fallback_mono_depth(mvs_depth, valid_mask)
        mono_source = "mvs_smoothed_fallback"
    else:
        mono_depth = resize_nchw(to_nchw(mono_depth, device=device, dtype=dtype), mvs_depth.shape[-2:])
        mono_depth = torch.nan_to_num(mono_depth, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)

    mono_aligned, align_meta = align_mono_to_mvs_inverse_depth(
        mvs_depth,
        mono_depth,
        valid_mask,
        confidence=mvs_confidence,
    )

    confidence = mvs_confidence
    if blur_mask is not None:
        blur = resize_nchw(to_nchw(blur_mask, device=device, dtype=dtype), mvs_depth.shape[-2:])
        blur = (blur / max_mask_strength).clamp(0.0, 1.0)
        confidence = confidence * (1.0 - 0.75 * blur)

    soft = gaussian_blur_nchw((valid_mask * confidence).clamp(0.0, 1.0), k=31, sigma=9.0)
    soft = torch.max(soft * 0.75, (valid_mask * confidence).clamp(0.0, 1.0))
    soft = soft.clamp(0.0, 1.0)
    complete_depth = (soft * mvs_depth + (1.0 - soft) * mono_aligned).clamp_min(1e-6)
    depth_confidence = (soft + (1.0 - soft) * 0.35).clamp(0.0, 1.0)

    meta = dict(align_meta)
    meta["mono_depth_source"] = mono_source
    return {
        "mvs_depth": mvs_depth,
        "mvs_valid_mask": valid_mask,
        "mvs_confidence": mvs_confidence,
        "mono_depth_raw": mono_depth,
        "mono_depth_aligned": mono_aligned,
        "complete_depth": complete_depth,
        "depth_confidence": depth_confidence,
        "fusion_soft_mask": soft,
        "meta": meta,
    }


def depth_to_uint8(depth, mask=None):
    if torch.is_tensor(depth):
        depth_np = depth.detach().cpu().float().numpy()
    else:
        depth_np = np.asarray(depth, dtype=np.float32)
    depth_np = np.squeeze(depth_np).astype(np.float32)
    finite = np.isfinite(depth_np) & (depth_np > 0)
    if mask is not None:
        if torch.is_tensor(mask):
            mask_np = mask.detach().cpu().float().numpy()
        else:
            mask_np = np.asarray(mask, dtype=np.float32)
        finite = finite & (np.squeeze(mask_np) > 0.5)
    if not finite.any():
        return np.zeros(depth_np.shape, dtype=np.uint8)
    lo, hi = np.percentile(depth_np[finite], [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((depth_np - lo) / (hi - lo), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def load_optional_mono_depth(base_name, search_dir, size, device=None, dtype=None):
    if not search_dir:
        return None, None
    candidates = [
        os.path.join(search_dir, base_name + "_mono_depth.npy"),
        os.path.join(search_dir, base_name + ".npy"),
        os.path.join(search_dir, base_name + "_mono_depth.png"),
        os.path.join(search_dir, base_name + ".png"),
    ]
    path = None
    for candidate in candidates:
        if os.path.exists(candidate):
            path = candidate
            break
    if path is None:
        return None, None

    if path.endswith(".npy"):
        arr = np.load(path).astype(np.float32)
    else:
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            return None, None
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        arr = arr.astype(np.float32)
    tensor = to_nchw(arr, device=device, dtype=dtype)
    tensor = resize_nchw(tensor, size, mode="bilinear")
    return tensor, path


def select_depth_condition(data, source_depths, mode, target_size, device, dtype, target_depth_scale=1.0):
    aliases = {
        "baseline": "source",
        "source": "source",
        "src": "source",
        "source_mvs": "source_mvs",
        "src_mvs": "source_mvs",
        "source_mvs_only": "source_mvs",
        "source_mono": "source_mono",
        "src_mono": "source_mono",
        "source_mono_only": "source_mono",
        "source_completed": "source_completed",
        "src_completed": "source_completed",
        "source_mvs_mono": "source_completed",
        "source_mvs_mono_completed": "source_completed",
        "mvs": "mvs",
        "mvs_only": "mvs",
        "mono": "mono",
        "mono_only": "mono",
        "completed": "completed",
        "mvs_mono": "completed",
        "mvs_mono_completed": "completed",
    }
    mode = aliases.get(mode, mode)
    if mode == "source":
        return source_depths, "source"

    source_field = {
        "source_mvs": "source_mvs_depths",
        "source_mono": "source_mono_depths_aligned",
        "source_completed": "source_complete_depths",
    }.get(mode)
    if source_field is not None:
        if source_field not in data:
            return source_depths, "source_missing_{}".format(source_field)
        src_depth = to_nchw(data[source_field], device=device, dtype=dtype)
        src_depth = resize_nchw(src_depth, target_size, mode="bilinear") * float(target_depth_scale)
        src_depth = torch.nan_to_num(src_depth, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        if src_depth.max().item() <= 0:
            return source_depths, "source_empty_{}".format(source_field)
        return src_depth, mode

    field = {
        "mvs": "target_mvs_depth",
        "mono": "mono_depth_aligned",
        "completed": "complete_depth",
    }.get(mode)
    if field is None or field not in data:
        return source_depths, "source_missing_{}".format(field or mode)

    target_depth = to_nchw(data[field], device=device, dtype=dtype)
    target_depth = resize_nchw(target_depth, target_size, mode="bilinear")[:1] * float(target_depth_scale)
    target_depth = torch.nan_to_num(target_depth, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if target_depth.max().item() <= 0:
        return source_depths, "source_empty_{}".format(field)
    return torch.cat([source_depths, target_depth], dim=0), mode
