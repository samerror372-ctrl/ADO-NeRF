import os
import random
import tqdm
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2

from configs import cfg, args
from datasets import make_data_loader
from networks import make_network
from utils.data_utils import to_cuda
from evaluators import make_evaluator
from utils import net_utils

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DIFFUSION_ROOT = os.path.join(PROJECT_ROOT, "diffusion_nerf")

if cfg.fix_random:
    random.seed(0)
    np.random.seed(0)

# =========================================================================
# 🛡️ [新增审计工具] 用于检测张量异常
# =========================================================================
def audit_tensor(name, t):
    if t is None:
        print(f"  [AUDIT] {name:15s} | !! MISSING !!")
        return
    t_f = t.detach().float()
    v_min, v_max = t_f.min().item(), t_f.max().item()
    v_mean, v_std = t_f.mean().item(), t_f.std().item()
    has_nan = torch.isnan(t_f).any().item()
    print(f"  [AUDIT] {name:15s} | Range: [{v_min:8.3f}, {v_max:8.3f}] | Mean: {v_mean:8.3f} | Std: {v_std:8.3f} | NaN: {has_nan}")


def _env_int(name, default=0):
    value = os.environ.get(name, "")
    return int(value) if value.strip() else default


MASK_STRENGTH_PROFILES = {
    "orig": {
        "setting": "orig",
        "kind": "orig",
        "scale": 1.0,
        "description": "Original V7.2 sine-dome mask without strength tuning.",
    },
    "orig_s0p50": {
        "setting": "orig_s0.50",
        "kind": "orig",
        "scale": 0.50,
        "description": "Pre-tuning mask used in the cross-dataset baseline.",
    },
    "grid_l1_0p10_l2_0p25_l3_0p10_s0p50": {
        "setting": "grid_l1.0.10_l2.0.25_l3.0.10_s0.50",
        "kind": "recompose",
        "w1": 0.10,
        "w2": 0.25,
        "w3": 0.10,
        "post_scale": 0.50,
        "description": "Tuned mask: recomposed Layer1/2/3 then globally scaled by 0.50.",
    },
}

MASK_PROFILE_ALIASES = {
    "pre_tuning": "orig_s0p50",
    "orig_s0.50": "orig_s0p50",
    "orig_s0p50": "orig_s0p50",
    "tuned": "grid_l1_0p10_l2_0p25_l3_0p10_s0p50",
    "grid": "grid_l1_0p10_l2_0p25_l3_0p10_s0p50",
    "grid_l1_0p10_l2_0p25_l3_0p10_s0p50": "grid_l1_0p10_l2_0p25_l3_0p10_s0p50",
    "orig": "orig",
}


def _select_mask_strength_profile():
    profile_name = os.environ.get("ADO_MASK_PROFILE", "").strip()
    if not profile_name:
        use_tuned = os.environ.get("ADO_USE_TUNED_MASK", "1").strip().lower()
        profile_name = "orig_s0p50" if use_tuned in {"0", "false", "no", "off"} else "grid_l1_0p10_l2_0p25_l3_0p10_s0p50"
    profile_name = MASK_PROFILE_ALIASES.get(profile_name, profile_name)
    if profile_name not in MASK_STRENGTH_PROFILES:
        valid = ", ".join(sorted(MASK_STRENGTH_PROFILES))
        raise ValueError(f"Unknown mask profile {profile_name!r}. Valid profiles: {valid}")
    return profile_name, dict(MASK_STRENGTH_PROFILES[profile_name])


def _gaussian_blur_4d(mask, kernel_size, sigma):
    coords = torch.arange(kernel_size, dtype=torch.float32, device=mask.device) - kernel_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = torch.mm(kernel_1d[:, None], kernel_1d[None, :])[None, None]
    pad = kernel_size // 2
    return F.conv2d(F.pad(mask, (pad, pad, pad, pad), mode="reflect"), kernel).clamp(0.0, 1.0)


def _recompose_mask_from_layers(layer1_raw_mask, layer2_soft_mask, layer3_raw_mask, profile):
    raw = (
        float(profile["w1"]) * layer1_raw_mask.float()
        + float(profile["w2"]) * layer2_soft_mask.float()
        + float(profile["w3"]) * layer3_raw_mask.float()
    ).clamp_min(0.0).unsqueeze(1)
    if raw.max() <= 1e-7:
        return torch.zeros_like(layer1_raw_mask.float())

    expanded = F.max_pool2d(raw, kernel_size=15, stride=1, padding=7)
    blur_level_1 = _gaussian_blur_4d(expanded, kernel_size=15, sigma=5.0)
    blur_level_2 = _gaussian_blur_4d(blur_level_1, kernel_size=51, sigma=15.0)
    blur_level_3 = _gaussian_blur_4d(blur_level_2, kernel_size=101, sigma=35.0)
    final_gradient = (blur_level_1 * 0.4) + (blur_level_2 * 0.4) + (blur_level_3 * 0.2)
    fg_max = final_gradient.max()
    if fg_max > 1e-7:
        angle = torch.tensor(torch.pi / 2.0, device=final_gradient.device)
        final_gradient = 0.35 * torch.sin(angle * (final_gradient / fg_max))
    final_gradient = final_gradient * float(profile.get("post_scale", 1.0))
    return final_gradient.squeeze(1).clamp(0.0, 0.35)


def _build_export_blur_mask(gradient_mask, layer1_raw_mask, layer2_soft_mask, layer3_raw_mask, profile):
    if profile["kind"] == "orig":
        return (gradient_mask.float() * float(profile.get("scale", 1.0))).clamp(0.0, 0.35)
    if profile["kind"] == "recompose":
        return _recompose_mask_from_layers(layer1_raw_mask, layer2_soft_mask, layer3_raw_mask, profile)
    raise ValueError(f"Unsupported mask profile kind: {profile['kind']}")


def _meta_scalar(meta, key, default):
    if not isinstance(meta, dict) or key not in meta:
        return default
    value = meta[key]
    if isinstance(value, (list, tuple)):
        value = value[0]
    if torch.is_tensor(value):
        value = value.detach().cpu().flatten()[0].item()
    return value


def _view_export_name(batch, batch_idx):
    meta = batch.get('meta', {})
    image_name = _meta_scalar(meta, 'image_name', None)
    if image_name is not None:
        return str(image_name)
    scene = str(_meta_scalar(meta, 'scene', 'scene'))
    tar_view = _meta_scalar(meta, 'tar_view', batch_idx)
    try:
        tar_view = int(tar_view)
        return f"{scene}_view_{tar_view:04d}"
    except (TypeError, ValueError):
        return f"{scene}_view_{batch_idx:04d}"


def _resize_source_maps(maps, size, mode='nearest'):
    if maps is None:
        return None
    if maps.dim() == 5 and maps.shape[2] == 1:
        maps = maps[:, :, 0]
    B, V = maps.shape[:2]
    if maps.shape[-2:] != size:
        maps = F.interpolate(
            maps.flatten(0, 1).unsqueeze(1).float(),
            size=size,
            mode=mode,
            align_corners=False if mode in ['bilinear', 'bicubic'] else None
        ).view(B, V, size[0], size[1])
    return torch.nan_to_num(maps.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _source_anchor_depths(batch, src_size):
    src_views = batch.get('src_views', {})
    if 'depth' not in src_views:
        return None
    return _resize_source_maps(src_views['depth'], src_size, mode='nearest').clamp_min(0.0)


def _source_anchor_masks(batch, src_size, src_depths=None):
    src_views = batch.get('src_views', {})
    if 'mask' in src_views:
        src_masks = _resize_source_maps(src_views['mask'], src_size, mode='nearest')
        src_masks = (src_masks > 0.5).float()
    elif src_depths is not None:
        src_masks = torch.ones_like(src_depths)
    else:
        return None
    if src_depths is not None:
        src_masks = ((src_masks > 0.5) & torch.isfinite(src_depths) & (src_depths > 1e-6)).float()
    return src_masks


def _resize_source_ranges(depth_ranges, size):
    if depth_ranges is None:
        return None
    if depth_ranges.dim() != 5:
        return None
    B, V = depth_ranges.shape[:2]
    if depth_ranges.shape[-2:] != size:
        depth_ranges = F.interpolate(
            depth_ranges.flatten(0, 1).float(),
            size=size,
            mode='nearest'
        ).view(B, V, 2, size[0], size[1])
    return torch.nan_to_num(depth_ranges.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)


def _confidence_from_ranges(depth_ranges, ref_depths, valid_masks=None):
    if ref_depths.dim() == 5 and ref_depths.shape[2] == 1:
        ref_depths = ref_depths[:, :, 0]
    if depth_ranges is None:
        conf = (torch.isfinite(ref_depths) & (ref_depths > 1e-6)).float()
    else:
        width = torch.abs(depth_ranges[:, :, 1] - depth_ranges[:, :, 0])
        rel_unc = width / ref_depths.abs().clamp_min(1e-6)
        conf = torch.sigmoid((0.05 - rel_unc) / 0.02)
        conf = conf * (torch.isfinite(ref_depths) & (ref_depths > 1e-6)).float()
    if valid_masks is not None:
        if valid_masks.dim() == 5 and valid_masks.shape[2] == 1:
            valid_masks = valid_masks[:, :, 0]
        conf = conf * (valid_masks > 0.5).float()
    return conf.clamp(0.0, 1.0)


def _target_depth(output, src_size):
    depth = output.get('mvs_depth', output.get('nerf_depth'))
    if depth is None:
        return None
    if depth.dim() == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]
    if depth.shape[-2:] != src_size:
        depth = F.interpolate(depth.unsqueeze(1), size=src_size, mode='nearest').squeeze(1)
    return torch.nan_to_num(depth.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)


def run_dataset():
    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    for batch in tqdm.tqdm(data_loader):
        pass


def run_network():
    network = make_network(cfg).cuda()
    net_utils.load_network(network, cfg.trained_model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    total_time = 0
    for batch in tqdm.tqdm(data_loader):
        batch = to_cuda(batch)
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            network(batch)
            torch.cuda.synchronize()
            total_time += time.time() - start
    print(total_time / len(data_loader))


# =========================================================================
# [核心算法 V7.2 - 正弦圆顶平滑特化版] 拒绝一刀切硬截断，构建柔和等高线
# =========================================================================
def compute_hybrid_soft_mask(batch, output):
    pred_rgb_all = output['rgb']
    _, _, H_rgb, W_rgb = pred_rgb_all.shape
    depth_map = output.get('mvs_depth', output['nerf_depth'])
    if depth_map.shape[-2:] != (H_rgb, W_rgb):
        depth_map = F.interpolate(
            depth_map.unsqueeze(1).float(),
            size=(H_rgb, W_rgb),
            mode='nearest').squeeze(1)
    depth_map = torch.nan_to_num(depth_map.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    B, H, W = depth_map.shape
    src_imgs = batch['src_views']['rgb']
    V = src_imgs.shape[1]
    H_src, W_src = src_imgs.shape[-2:]
    src_anchor_depths = _source_anchor_depths(batch, (H_src, W_src))
    src_mvs_depths = src_anchor_depths if src_anchor_depths is not None else output.get('src_mvs_depths', None)
    if src_mvs_depths is not None and src_mvs_depths.shape[-2:] != (H_src, W_src):
        src_mvs_depths = _resize_source_maps(src_mvs_depths, (H_src, W_src), mode='nearest')
    
    src_exts = batch['src_views']['extrinsics']
    src_ints = batch['src_views']['intrinsics']
    tar_exts = batch['tar_views']['extrinsics']
    tar_ints = batch['tar_views']['intrinsics']

    y, x = torch.meshgrid(torch.arange(H, device=depth_map.device),
                          torch.arange(W, device=depth_map.device), indexing='ij')
    grid = torch.stack([x.float() + 0.5, y.float() + 0.5, torch.ones_like(x)], dim=-1)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

    layer1_mask_out = torch.zeros((B, H, W), device=depth_map.device, dtype=torch.float32)
    final_soft_mask = torch.zeros((B, H, W), device=depth_map.device, dtype=torch.float32)
    layer3_mask_out = torch.zeros((B, H, W), device=depth_map.device, dtype=torch.float32)
    gradient_mask_out = torch.zeros((B, H, W), device=depth_map.device, dtype=torch.float32)

    def apply_gaussian_blur(m, k=15, s=5.0):
        x_coord = torch.arange(k, dtype=torch.float32, device=m.device)
        x_grid = x_coord - k // 2
        gauss = torch.exp(-(x_grid**2) / (2 * s**2))
        gauss = gauss / gauss.sum()
        gauss_2d = torch.mm(gauss.unsqueeze(1), gauss.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        m_pad = F.pad(m.unsqueeze(1), (k//2,)*4, mode='reflect')
        return F.conv2d(m_pad, gauss_2d).squeeze(1).clamp(0.0, 1.0)

    # 边缘空间衰减矩阵 (Edge Spatial Fade) 15%
    fade_h = int(H * 0.15) 
    fade_w = int(W * 0.15)
    
    ramp_h = torch.linspace(0.0, 1.0, fade_h, device=depth_map.device)
    ramp_w = torch.linspace(0.0, 1.0, fade_w, device=depth_map.device)
    
    y_fade = torch.ones(H, device=depth_map.device)
    y_fade[:fade_h] = ramp_h
    y_fade[-fade_h:] = ramp_h.flip(0)
    
    x_fade = torch.ones(W, device=depth_map.device)
    x_fade[:fade_w] = ramp_w
    x_fade[-fade_w:] = ramp_w.flip(0)
    
    spatial_fade = y_fade.unsqueeze(1) * x_fade.unsqueeze(0) 

    for b in range(B):
        tar_ints_inv = torch.inverse(tar_ints[b])
        cam_coords = torch.einsum('ij,hwj->hwi', tar_ints_inv, grid[b]) * depth_map[b].unsqueeze(-1)
        cam_coords_homo = torch.cat([cam_coords, torch.ones_like(cam_coords[..., :1])], dim=-1)
        world_coords = torch.einsum('ij,hwj->hwi', torch.inverse(tar_exts[b]), cam_coords_homo)
        wc = world_coords.unsqueeze(0).repeat(V, 1, 1, 1)
        sc = torch.einsum('vij,vhwj->vhwi', src_exts[b], wc)[..., :3]
        
        z_proj = sc[..., 2]
        valid_z = z_proj > 1e-3
        
        uv = torch.einsum('vij,vhwj->vhwi', src_ints[b], sc)
        u = uv[..., 0] / (uv[..., 2] + 1e-8)
        v = uv[..., 1] / (uv[..., 2] + 1e-8)
        u_norm = (u / W_src) * 2 - 1
        v_norm = (v / H_src) * 2 - 1
        uv_norm = torch.stack([u_norm, v_norm], dim=-1)
        
        valid_uv = (u_norm >= -1) & (u_norm <= 1) & (v_norm >= -1) & (v_norm <= 1)
        valid_proj = valid_z & valid_uv

        # =========================================================
        # 【Layer 1】: 几何区域标识
        # =========================================================
        out_of_bounds_count = V - valid_proj.float().sum(dim=0)
        mask_oob = out_of_bounds_count >= (V * 2.0 / 3.0)
        
        mask_conflict = torch.zeros_like(mask_oob)
        is_occluded = torch.zeros_like(valid_proj)
        if src_mvs_depths is not None:
            src_depth_b = src_mvs_depths[b].unsqueeze(1)
            sampled_src_depth = F.grid_sample(src_depth_b, uv_norm, align_corners=False).squeeze(1)
            tolerance = 0.02 * sampled_src_depth.clamp_min(1e-6)
            is_occluded = valid_proj & (z_proj > sampled_src_depth + tolerance)
            is_floater  = valid_proj & (z_proj < sampled_src_depth - tolerance)
            mask_conflict = (is_occluded | is_floater).any(dim=0)

        layer1_roi = mask_oob | mask_conflict
        layer1_mask_out[b] = layer1_roi.float()

        # =========================================================
        # 【Layer 2】: 光度残差 (👑 拔除二次方，舒缓基本坡度)
        # =========================================================
        in_bounds_mask = valid_proj & (~is_occluded) if src_mvs_depths is not None else valid_proj
        valid_count = in_bounds_mask.float().sum(dim=0)
        valid_mask_f = in_bounds_mask.float().unsqueeze(1)
        count_safe = valid_count.unsqueeze(0).clamp(min=1)
        
        pred_rgb_b = pred_rgb_all[b]
        if pred_rgb_b.min() < 0: 
            pred_rgb_b = (pred_rgb_b + 1.0) / 2.0
            
        sampled_rgb = F.grid_sample(src_imgs[b], uv_norm, align_corners=False)
        mean_sampled_rgb = (sampled_rgb * valid_mask_f).sum(dim=0) / count_safe
        
        sq_diff = ((sampled_rgb - mean_sampled_rgb.unsqueeze(0))**2) * valid_mask_f
        var_gray = (sq_diff.sum(dim=0) / count_safe).mean(dim=0)
        
        tar_src_diff = torch.abs(sampled_rgb - pred_rgb_b.unsqueeze(0)) * valid_mask_f
        error_gray = (tar_src_diff.sum(dim=0) / count_safe).mean(dim=0)
        
        l2_metric = var_gray + error_gray
        
        l2_threshold = 0.05  
        l2_max_err = 0.25    
        normalized_l2 = torch.clamp((l2_metric - l2_threshold) / (l2_max_err - l2_threshold + 1e-8), 0.0, 1.0)
        
        # 👑 【核心优化】：删除原版 normalized_l2 = normalized_l2 ** 2.0，让基础坡度更平缓

        mask_b = torch.zeros_like(var_gray)
        mask_b[layer1_roi] = normalized_l2[layer1_roi]
        final_soft_mask[b] = mask_b

        # =========================================================
        # 【Layer 3】: 纹理塌陷检测 (👑 拔除二次方)
        # =========================================================
        K = 7
        pad = K // 2
        
        def get_local_var(img):
            img_padded = img.unsqueeze(0) 
            mu = F.avg_pool2d(img_padded, kernel_size=K, stride=1, padding=pad)
            mu2 = F.avg_pool2d(img_padded**2, kernel_size=K, stride=1, padding=pad)
            return torch.clamp(mu2 - mu**2, min=0.0).mean(dim=1).squeeze(0)

        var_pred = get_local_var(pred_rgb_b)
        var_ref = get_local_var(mean_sampled_rgb)
        
        texture_diff = F.relu(var_ref - var_pred)
        
        tex_threshold = 0.01 
        tex_max_err = 0.04   
        normalized_tex_loss = torch.clamp((texture_diff - tex_threshold) / (tex_max_err - tex_threshold + 1e-8), 0.0, 1.0)
        
        # 👑 【核心优化】：删除原版 normalized_tex_loss = normalized_tex_loss ** 2.0

        layer3_roi = layer1_roi & (texture_diff > tex_threshold) 
        layer3_mask_out[b] = layer3_roi.float()

        # =========================================================
        # 【生成基础悬浮能量场】
        # =========================================================
        raw_gradient = normalized_l2 * 0.5 + normalized_tex_loss * 0.5
        
        masked_gradient = torch.zeros_like(raw_gradient)
        masked_gradient[layer1_roi] = raw_gradient[layer1_roi]

        # 智能盲区热力
        blind_spot = layer1_roi & (valid_count < 2)
        flatness_penalty = torch.clamp(var_pred / 0.008, 0.0, 1.0) 
        smart_blind_heat = 0.10 + 0.15 * flatness_penalty
        masked_gradient[blind_spot] = torch.max(masked_gradient[blind_spot], smart_blind_heat[blind_spot])

        # 边缘与背景空间衰减
        pred_rgb_gray = pred_rgb_b.mean(dim=0)
        is_bg = (pred_rgb_gray < 0.05) | (pred_rgb_gray > 0.95)
        
        bg_suppression = torch.ones_like(masked_gradient)
        bg_suppression[is_bg] = spatial_fade[is_bg]
        bg_suppression = apply_gaussian_blur(bg_suppression.unsqueeze(0), k=11, s=3.0)[0]
        
        gradient_mask_out[b] = masked_gradient * bg_suppression

    # ==================================================================================
    # 🎨 [等高线大手术] 形态学膨胀 + 广域级联高斯模糊 (拓展边界并拉长 T度坡度)
    # ==================================================================================
    def dilate_mask(m, k=15):
        m_in = m.unsqueeze(1)
        m_dilated = F.max_pool2d(m_in, kernel_size=k, stride=1, padding=k//2)
        return m_dilated.squeeze(1)

    # 1. 稍微向外扩大 7 像素，建立连贯的危险包裹带
    expanded_mask = dilate_mask(gradient_mask_out, k=15)

    # 2. 级联大范围高斯滤镜，将硬质边界彻底融化为极其漫长的山坡
    blur_level_1 = apply_gaussian_blur(expanded_mask, k=15, s=5.0)   # 核心区基础虚化
    blur_level_2 = apply_gaussian_blur(blur_level_1, k=51, s=15.0)  # 中段坡度延展
    blur_level_3 = apply_gaussian_blur(blur_level_2, k=101, s=35.0) # 边缘长拖尾

    # 混合多尺度权重，让坡度在全图极其均匀分布
    final_gradient = (blur_level_1 * 0.4) + (blur_level_2 * 0.4) + (blur_level_3 * 0.2)

    # ==================================================================================
    # 👑 [绝妙手术：正弦圆顶无损映射] 严格锁死最大值在 0.35 且绝无一刀切平台
    # ==================================================================================
    fg_max = final_gradient.max()
    if fg_max > 1e-6:
        # 将模糊后的信号线性规整到 [0, 1]
        final_gradient_norm = final_gradient / fg_max
        # 利用 sin(pi/2 * x) 映射。当 x=1 时响应刚好为 0.35，且由于导数为0，山顶极其圆滑
        final_gradient = 0.35 * torch.sin(torch.tensor(torch.pi / 2.0, device=final_gradient.device) * final_gradient_norm)
    # ==================================================================================

    def apply_small_blur(m):
        return apply_gaussian_blur(m, k=9, s=3.0)

    return (
        apply_small_blur(final_soft_mask),
        apply_small_blur(layer1_mask_out),
        apply_small_blur(layer3_mask_out),
        final_gradient,
        layer1_mask_out,
        final_soft_mask,
        layer3_mask_out,
    )


def run_evaluate():
    network = make_network(cfg).cuda()
    net_utils.load_network(network, cfg.trained_model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    net_time = []
    
    default_ado_dir = os.path.join(PROJECT_ROOT, "diffusion_input")
    ado_input_dir = os.environ.get("ADO_EXPORT_OUTPUT_DIR", default_ado_dir)
    os.makedirs(ado_input_dir, exist_ok=True)
    max_batches = _env_int("ADO_EXPORT_MAX_BATCHES", 0)
    mask_profile_name, mask_profile = _select_mask_strength_profile()

    print("\n🚀 Starting ADO-NeRF export with V7.2 mask profiles...\n")
    print(f"[ADO EXPORT] ado_input_dir = {ado_input_dir}")
    print(f"[ADO EXPORT] mask_profile = {mask_profile_name} | setting = {mask_profile['setting']}")
    print("[ADO EXPORT] set ADO_USE_TUNED_MASK=0 for orig_s0p50.")
    if max_batches > 0:
        print(f"[ADO EXPORT] max_batches = {max_batches}")

    for batch_idx, batch in enumerate(tqdm.tqdm(data_loader)):
        if max_batches > 0 and batch_idx >= max_batches:
            print(f"[ADO EXPORT] reached max batches: {max_batches}")
            break
        batch = to_cuda(batch)
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            output, mvs_depths, blend_rgbs = network(batch)
            torch.cuda.synchronize()
            end_time = time.time()
        net_time.append(end_time - start_time)
        evaluator.evaluate(output, batch)
        
        if 'rgb' in output and 'nerf_depth' in output:
            (
                soft_mask,
                hard_mask,
                blur_mask,
                gradient_mask,
                layer1_raw_mask,
                layer2_soft_mask,
                layer3_raw_mask,
            ) = compute_hybrid_soft_mask(batch, output)
            
            img_name = _view_export_name(batch, batch_idx)

            src_rgbs_raw = batch['src_views']['rgb'][0] 
            src_rgbs_ado = (src_rgbs_raw * 2.0 - 1.0).clamp(-1.0, 1.0).half()

            tar_pred_rgb_raw = output['rgb'][0]
            tar_rgb_ado = (tar_pred_rgb_raw * 2.0 - 1.0).clamp(-1.0, 1.0).unsqueeze(0).half()

            H_src, W_src = src_rgbs_raw.shape[-2:]
            src_anchor_depths = _source_anchor_depths(batch, (H_src, W_src))
            if src_anchor_depths is not None:
                source_depths_full = src_anchor_depths
            else:
                fallback_depth = F.interpolate(
                    output['nerf_depth'].unsqueeze(1),
                    size=(H_src, W_src),
                    mode='nearest').squeeze(1)
                source_depths_full = fallback_depth.unsqueeze(1).repeat(1, src_rgbs_ado.shape[0], 1, 1)
            source_valid_masks_full = (torch.isfinite(source_depths_full) & (source_depths_full > 1e-6)).float()
            src_depths_ado = source_depths_full[0].unsqueeze(1).float()
            src_valid_masks_ado = source_valid_masks_full[0].unsqueeze(1).float()

            export_blur_mask = _build_export_blur_mask(
                gradient_mask,
                layer1_raw_mask,
                layer2_soft_mask,
                layer3_raw_mask,
                mask_profile,
            )

            # 存储用
            blur_mask_ado = export_blur_mask[0].unsqueeze(0).unsqueeze(0).half()
            mask_layer1_raw_ado = layer1_raw_mask[0].unsqueeze(0).unsqueeze(0).half()
            mask_layer2_soft_ado = layer2_soft_mask[0].unsqueeze(0).unsqueeze(0).half()
            mask_layer3_raw_ado = layer3_raw_mask[0].unsqueeze(0).unsqueeze(0).half()
            target_eval_mask = (batch['tar_views']['mask'][0].float() >= 0.5).unsqueeze(0).unsqueeze(0).half()

            src_ints = batch['src_views']['intrinsics'][0]      
            tar_ints = batch['tar_views']['intrinsics'][0]      
            if tar_ints.dim() == 2: tar_ints = tar_ints.unsqueeze(0)
            all_intrinsics = torch.cat([src_ints, tar_ints], dim=0).float()
            
            src_exts = batch['src_views']['extrinsics'][0]      
            tar_exts = batch['tar_views']['extrinsics'][0]      
            if tar_exts.dim() == 2: tar_exts = tar_exts.unsqueeze(0)
            all_extrinsics_w2c = torch.cat([src_exts, tar_exts], dim=0).float()
            all_extrinsics_c2w = torch.inverse(all_extrinsics_w2c)

            cam_centers = all_extrinsics_c2w[:, :3, 3] 
            max_cam_val = cam_centers.abs().max()
            max_depth_val = src_depths_ado.max()
            scene_max_val = torch.max(max_cam_val, max_depth_val).item()
            scene_scale = 1.0
            
            if scene_max_val > 5.0:
                scene_scale = 5.0 / scene_max_val
                all_extrinsics_c2w[:, :3, 3] *= scene_scale
                src_depths_ado *= scene_scale
            all_extrinsics_w2c_scaled = torch.inverse(all_extrinsics_c2w)
                
            ado_data = {
                "source_rgbs": src_rgbs_ado.cpu(),
                "pred_rgb": tar_rgb_ado.cpu(),
                "source_depths": src_depths_ado.half().cpu(),
                "blur_mask": blur_mask_ado.cpu(),
                "mask_layer1_raw": mask_layer1_raw_ado.cpu(),
                "mask_layer2_soft": mask_layer2_soft_ado.cpu(),
                "mask_layer3_raw": mask_layer3_raw_ado.cpu(),
                "target_eval_mask": target_eval_mask.cpu(),
                "intrinsics": all_intrinsics.cpu(),
                "extrinsics": all_extrinsics_w2c_scaled.cpu(),
                "extrinsics_w2c": all_extrinsics_w2c_scaled.cpu(),
                "extrinsics_c2w": all_extrinsics_c2w.cpu(),
                "extrinsics_format": "w2c",
                "cond_num": src_rgbs_ado.shape[0],
                "source": "ADO-NeRF",
                "scene_scale": float(scene_scale),
                "mask_strength_profile": mask_profile_name,
                "mask_strength_setting": dict(mask_profile),
            }
            torch.save(ado_data, os.path.join(ado_input_dir, f"{img_name}_ado_data.pt"))

            if 'rgb' in batch['tar_views']:
                gt_rgb_raw = batch['tar_views']['rgb'][0]
                while gt_rgb_raw.dim() > 3:
                    gt_rgb_raw = gt_rgb_raw[0]
                if gt_rgb_raw.shape[0] in [1, 3, 4]:
                    gt_rgb_raw = gt_rgb_raw.permute(1, 2, 0)
                gt_rgb_np = gt_rgb_raw.cpu().float().numpy()
                if gt_rgb_np.min() < 0: gt_rgb_np = (gt_rgb_np + 1.0) / 2.0
                gt_rgb_np = np.clip(gt_rgb_np, 0, 1)
                gt_rgb_uint8 = (gt_rgb_np * 255).astype(np.uint8)
                if len(gt_rgb_uint8.shape) == 3 and gt_rgb_uint8.shape[-1] == 3:
                    gt_rgb_uint8 = gt_rgb_uint8[:, :, ::-1]
                cv2.imwrite(os.path.join(ado_input_dir, f"{img_name}_0_gt_rgb.png"), gt_rgb_uint8)

            pred_rgb_vis_np = (tar_rgb_ado[0].permute(1, 2, 0).float().cpu().numpy() + 1.0) / 2.0
            pred_rgb_vis_np = np.clip(pred_rgb_vis_np, 0, 1)
            pred_rgb_bgr = (pred_rgb_vis_np * 255).astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(ado_input_dir, f"{img_name}_1_pred_rgb.png"), pred_rgb_bgr)

            mask_float = blur_mask_ado[0, 0].cpu().float().numpy()
            mask_display_linear = np.clip(mask_float / 0.35, 0.0, 1.0)
            mask_gray_uint8 = (mask_display_linear * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(ado_input_dir, f"{img_name}_2_mask_gray.png"), mask_gray_uint8)

            layer1_float = mask_layer1_raw_ado[0, 0].cpu().float().numpy()
            layer2_float = mask_layer2_soft_ado[0, 0].cpu().float().numpy()
            layer3_float = mask_layer3_raw_ado[0, 0].cpu().float().numpy()
            cv2.imwrite(os.path.join(ado_input_dir, f"{img_name}_6_mask_layer1_raw.png"), (layer1_float * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(ado_input_dir, f"{img_name}_7_mask_layer2_soft.png"), (np.clip(layer2_float, 0.0, 1.0) * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(ado_input_dir, f"{img_name}_8_mask_layer3_raw.png"), (layer3_float * 255).astype(np.uint8))
            target_eval_mask_float = target_eval_mask[0, 0].cpu().float().numpy()
            cv2.imwrite(os.path.join(ado_input_dir, f"{img_name}_9_dtu_eval_mask.png"), (target_eval_mask_float * 255).astype(np.uint8))

            # =========================================================
            # 🎨 诊断可视化：自适应真实最大值映射 (绝不人为增强产生虚假硬块)
            # =========================================================
            # 始终按照理论最大值 0.35 进行无畸变归一化显示
            mask_disp = mask_display_linear
            
            # 使用平方曲线进一步收敛红色质心，彻底拉长外围绿、蓝拖尾
            mask_disp = mask_disp ** 2.0 
            heatmap = cv2.applyColorMap((mask_disp * 255).astype(np.uint8), cv2.COLORMAP_JET)

            alpha = np.clip(mask_disp * 0.70, 0, 0.70)[..., np.newaxis] 
            overlay_img = (pred_rgb_bgr * (1 - alpha) + heatmap * alpha).astype(np.uint8)
            cv2.imwrite(os.path.join(ado_input_dir, f"{img_name}_3_mask_overlay.png"), overlay_img)

            src0_depth = src_depths_ado[0:1].detach().cpu().float().numpy()
            src0_valid = src_valid_masks_ado[0, 0].detach().cpu().float().numpy()
            valid_depth = src0_depth[0, 0][src0_valid > 0.5]
            if valid_depth.size > 0:
                lo, hi = np.percentile(valid_depth, [2, 98])
                depth_vis = np.clip((src0_depth[0, 0] - lo) / (hi - lo + 1e-6), 0.0, 1.0)
            else:
                depth_vis = np.zeros_like(src0_depth[0, 0])
            cv2.imwrite(os.path.join(ado_input_dir, f"{img_name}_4_src0_anchor_depth.png"), (depth_vis * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(ado_input_dir, f"{img_name}_5_src0_valid_mask.png"), (src0_valid * 255).astype(np.uint8))

            if batch_idx % 20 == 0:
                print(f"  [INFO] Processed: {img_name} | Mask Profile: {mask_profile_name} | Mask Actual Max: {mask_float.max():.4f}")

    evaluator.summarize()
    if len(net_time) > 1: print('FPS: ', 1./np.mean(net_time[1:]))
    else: print('FPS: ', 1./np.mean(net_time))

if __name__ == '__main__':
    globals()['run_' + args.type]()
