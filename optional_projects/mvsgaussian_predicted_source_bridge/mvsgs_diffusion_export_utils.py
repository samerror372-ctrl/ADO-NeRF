import os

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F

from lib.config import cfg
from diffusion_nerf.depth_completion_utils import (
    complete_mvs_mono_depth,
    depth_to_uint8,
    load_optional_mono_depth,
)


def _ensure_4x4(extrinsics):
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"Unsupported extrinsics shape: {tuple(extrinsics.shape)}")
    pad_shape = list(extrinsics.shape[:-2]) + [1, 4]
    pad = torch.zeros(pad_shape, device=extrinsics.device, dtype=extrinsics.dtype)
    pad[..., 0, 3] = 1.0
    return torch.cat([extrinsics, pad], dim=-2)


def _scale_intrinsics(intrinsics, scale_y, scale_x):
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, :] *= scale_x
    intrinsics[..., 1, :] *= scale_y
    return intrinsics


def _meta_value(meta, key, index=0):
    value = meta[key]
    if isinstance(value, (list, tuple)):
        value = value[index]
    if torch.is_tensor(value):
        value = value[index] if value.ndim > 0 else value
        value = value.item()
    return value


def _final_render_level():
    render_if = list(cfg.mvsgs.cas_config.render_if)
    for level in range(len(render_if) - 1, -1, -1):
        if render_if[level]:
            return level
    raise ValueError("No enabled MVSGaussian render level found.")


def _mvsgs_rgb_chw(output, batch, level):
    B, _, _, H_img, W_img = batch['src_inps'].shape
    render_scale = cfg.mvsgs.cas_config.render_scale[level]
    H, W = int(H_img * render_scale), int(W_img * render_scale)
    rgb = output[f'rgb_level{level}'].reshape(B, H, W, 3)
    return rgb.permute(0, 3, 1, 2).contiguous().clamp(0.0, 1.0)


def _mvsgs_depth(output, batch, level):
    B, _, _, H_img, W_img = batch['src_inps'].shape
    render_scale = cfg.mvsgs.cas_config.render_scale[level]
    H, W = int(H_img * render_scale), int(W_img * render_scale)
    depth = output[f'depth_mvs_level{level}']
    if depth.shape[-2:] != (H, W):
        depth = F.interpolate(depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)
    return depth


def _mvsgs_std(output, batch, level):
    key = f'std_level{level}'
    if key not in output:
        return None
    B, _, _, H_img, W_img = batch['src_inps'].shape
    render_scale = cfg.mvsgs.cas_config.render_scale[level]
    H, W = int(H_img * render_scale), int(W_img * render_scale)
    std = output[key]
    if std.ndim == 4 and std.shape[1] == 1:
        std = std[:, 0]
    elif std.ndim == 2 and std.shape[1] == H * W:
        std = std.reshape(B, H, W)
    elif std.numel() == B * H * W:
        std = std.reshape(B, H, W)
    if std.shape[-2:] != (H, W):
        std = F.interpolate(std.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
    return std


def _mvsgs_valid_mask(batch, output, level, target_size):
    B = batch['src_inps'].shape[0]
    H, W = target_size
    depth = _mvsgs_depth(output, batch, level)
    mask_key = f'msk_{level}'
    if mask_key not in batch:
        return (torch.isfinite(depth) & (depth > 1e-6)).float()
    mask = batch[mask_key].float()
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    elif mask.ndim == 2 and mask.shape[1] == H * W:
        mask = mask.reshape(B, H, W)
    elif mask.numel() == B * H * W:
        mask = mask.reshape(B, H, W)
    if mask.shape[-2:] != (H, W):
        mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)
    return ((mask > 0.5) & torch.isfinite(depth) & (depth > 1e-6)).float()


def _source_depths(output, batch, level, src_size):
    key = f'src_mvs_depths_level{level}'
    src_depths = output.get(key, None)
    if src_depths is None:
        raise KeyError(
                f'Predicted-source MVSGaussian export requires output["{key}"]. '
                'Run export_mvsgs_predicted_source_depth.py so source depths are '
                'predicted by MVSGaussian instead of read from raw dataset depth.')
    if src_depths.shape[-2:] != src_size:
        B, V = src_depths.shape[:2]
        src_depths = F.interpolate(
                src_depths.flatten(0, 1).unsqueeze(1),
                size=src_size,
                mode='nearest').view(B, V, src_size[0], src_size[1])
    return src_depths


def _source_valid_masks(src_depths):
    return (torch.isfinite(src_depths) & (src_depths > 1e-6)).float()


def _meta_sequence(meta, key, index=0):
    if key not in meta:
        return []
    value = meta[key]
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return []
        if len(value) == 1:
            value = value[0]
        elif torch.is_tensor(value[0]):
            return [int(v.item()) for v in value]
    if torch.is_tensor(value):
        if value.ndim > 1:
            value = value[index]
        return [int(v) for v in value.detach().cpu().flatten().tolist()]
    if isinstance(value, np.ndarray):
        if value.ndim > 1:
            value = value[index]
        return [int(v) for v in value.reshape(-1).tolist()]
    try:
        return [int(v) for v in value]
    except TypeError:
        return [int(value)]


def _complete_source_depths(src_depths, src_valid_masks, base_name, H, W, device):
    source_mono_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'diffusion_nerf',
            'mono_depth_mvsgs_v7p_source')
    completions = []
    metas = []
    for src_idx in range(src_depths.shape[0]):
        mono_depth, mono_path = load_optional_mono_depth(
                f'{base_name}_src{src_idx}',
                source_mono_dir,
                (H, W),
                device=device,
                dtype=src_depths.dtype)
        completion = complete_mvs_mono_depth(
                src_depths[src_idx:src_idx + 1],
                valid_mask=src_valid_masks[src_idx:src_idx + 1],
                mvs_std=None,
                mono_depth=mono_depth,
                blur_mask=None)
        meta = dict(completion['meta'])
        meta['mono_depth_path'] = mono_path or ''
        meta['src_index'] = int(src_idx)
        completions.append(completion)
        metas.append(meta)

    def stack_field(name):
        return torch.cat([c[name] for c in completions], dim=0)

    return {
        'source_mvs_depths': stack_field('mvs_depth'),
        'source_mvs_valid_masks': stack_field('mvs_valid_mask'),
        'source_mvs_confidences': stack_field('mvs_confidence'),
        'source_mono_depths_raw': stack_field('mono_depth_raw'),
        'source_mono_depths_aligned': stack_field('mono_depth_aligned'),
        'source_complete_depths': stack_field('complete_depth'),
        'source_depth_confidences': stack_field('depth_confidence'),
        'source_fusion_soft_masks': stack_field('fusion_soft_mask'),
        'source_depth_completion_meta': metas,
    }


def _gaussian_blur(m, k=15, s=5.0):
    x_coord = torch.arange(k, dtype=torch.float32, device=m.device)
    x_grid = x_coord - k // 2
    gauss = torch.exp(-(x_grid ** 2) / (2 * s ** 2))
    gauss = gauss / gauss.sum()
    gauss_2d = torch.mm(gauss.unsqueeze(1), gauss.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    m_pad = F.pad(m.unsqueeze(1), (k // 2,) * 4, mode='reflect')
    return F.conv2d(m_pad, gauss_2d).squeeze(1).clamp(0.0, 1.0)


def compute_mvsgs_soft_mask(batch, output, level, return_layers=False):
    pred_rgb = _mvsgs_rgb_chw(output, batch, level)
    depth_map = _mvsgs_depth(output, batch, level)
    B, _, H, W = pred_rgb.shape

    render_scale = cfg.mvsgs.cas_config.render_scale[level]
    src_imgs = (batch['src_inps'] * 0.5 + 0.5).clamp(0.0, 1.0)
    V = src_imgs.shape[1]
    H_src, W_src = src_imgs.shape[-2:]

    src_exts = _ensure_4x4(batch['src_exts'])
    tar_exts = _ensure_4x4(batch['tar_ext'])
    src_ints = batch['src_ixts']
    tar_ints = _scale_intrinsics(batch['tar_ixt'], render_scale, render_scale)
    src_mvs_depths = _source_depths(output, batch, level, (H_src, W_src))

    y, x = torch.meshgrid(
            torch.arange(H, device=depth_map.device),
            torch.arange(W, device=depth_map.device),
            indexing='ij')
    grid = torch.stack([x.float() + 0.5, y.float() + 0.5, torch.ones_like(x)], dim=-1)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

    layer1_mask = torch.zeros((B, H, W), device=depth_map.device, dtype=torch.float32)
    final_soft_mask = torch.zeros((B, H, W), device=depth_map.device, dtype=torch.float32)
    layer3_mask = torch.zeros((B, H, W), device=depth_map.device, dtype=torch.float32)
    gradient_mask = torch.zeros((B, H, W), device=depth_map.device, dtype=torch.float32)

    fade_h = max(int(H * 0.15), 1)
    fade_w = max(int(W * 0.15), 1)
    y_fade = torch.ones(H, device=depth_map.device)
    x_fade = torch.ones(W, device=depth_map.device)
    ramp_h = torch.linspace(0.0, 1.0, fade_h, device=depth_map.device)
    ramp_w = torch.linspace(0.0, 1.0, fade_w, device=depth_map.device)
    y_fade[:fade_h] = ramp_h
    y_fade[-fade_h:] = ramp_h.flip(0)
    x_fade[:fade_w] = ramp_w
    x_fade[-fade_w:] = ramp_w.flip(0)
    spatial_fade = y_fade.unsqueeze(1) * x_fade.unsqueeze(0)

    for b in range(B):
        tar_int_inv = torch.inverse(tar_ints[b])
        cam_coords = torch.einsum('ij,hwj->hwi', tar_int_inv, grid[b]) * depth_map[b].unsqueeze(-1)
        cam_coords_homo = torch.cat([cam_coords, torch.ones_like(cam_coords[..., :1])], dim=-1)
        world_coords = torch.einsum('ij,hwj->hwi', torch.inverse(tar_exts[b]), cam_coords_homo)
        world_coords = world_coords.unsqueeze(0).repeat(V, 1, 1, 1)
        src_cam = torch.einsum('vij,vhwj->vhwi', src_exts[b], world_coords)[..., :3]

        z_proj = src_cam[..., 2]
        valid_z = z_proj > 1e-3
        uv = torch.einsum('vij,vhwj->vhwi', src_ints[b], src_cam)
        u = uv[..., 0] / (uv[..., 2] + 1e-8)
        v = uv[..., 1] / (uv[..., 2] + 1e-8)
        u_norm = (u / W_src) * 2 - 1
        v_norm = (v / H_src) * 2 - 1
        uv_norm = torch.stack([u_norm, v_norm], dim=-1)
        valid_uv = (u_norm >= -1) & (u_norm <= 1) & (v_norm >= -1) & (v_norm <= 1)
        valid_proj = valid_z & valid_uv

        out_of_bounds = V - valid_proj.float().sum(dim=0)
        mask_oob = out_of_bounds >= (V * 2.0 / 3.0)

        src_depth_b = src_mvs_depths[b].unsqueeze(1)
        sampled_src_depth = F.grid_sample(src_depth_b, uv_norm, align_corners=False).squeeze(1)
        tolerance = 0.02 * sampled_src_depth.clamp_min(1e-6)
        is_occluded = valid_proj & (z_proj > sampled_src_depth + tolerance)
        is_floater = valid_proj & (z_proj < sampled_src_depth - tolerance)
        mask_conflict = (is_occluded | is_floater).any(dim=0)
        layer1_roi = mask_oob | mask_conflict
        layer1_mask[b] = layer1_roi.float()

        in_bounds_mask = valid_proj & (~is_occluded)
        valid_count = in_bounds_mask.float().sum(dim=0)
        valid_mask_f = in_bounds_mask.float().unsqueeze(1)
        count_safe = valid_count.unsqueeze(0).clamp(min=1)

        pred_rgb_b = pred_rgb[b]
        sampled_rgb = F.grid_sample(src_imgs[b], uv_norm, align_corners=False)
        mean_sampled_rgb = (sampled_rgb * valid_mask_f).sum(dim=0) / count_safe

        sq_diff = ((sampled_rgb - mean_sampled_rgb.unsqueeze(0)) ** 2) * valid_mask_f
        var_gray = (sq_diff.sum(dim=0) / count_safe).mean(dim=0)
        tar_src_diff = torch.abs(sampled_rgb - pred_rgb_b.unsqueeze(0)) * valid_mask_f
        error_gray = (tar_src_diff.sum(dim=0) / count_safe).mean(dim=0)

        l2_metric = var_gray + error_gray
        normalized_l2 = torch.clamp((l2_metric - 0.05) / (0.25 - 0.05 + 1e-8), 0.0, 1.0)
        final_soft_mask[b][layer1_roi] = normalized_l2[layer1_roi]

        def local_var(img):
            img_padded = img.unsqueeze(0)
            mu = F.avg_pool2d(img_padded, kernel_size=7, stride=1, padding=3)
            mu2 = F.avg_pool2d(img_padded ** 2, kernel_size=7, stride=1, padding=3)
            return torch.clamp(mu2 - mu ** 2, min=0.0).mean(dim=1).squeeze(0)

        var_pred = local_var(pred_rgb_b)
        var_ref = local_var(mean_sampled_rgb)
        texture_diff = F.relu(var_ref - var_pred)
        normalized_tex = torch.clamp((texture_diff - 0.01) / (0.04 - 0.01 + 1e-8), 0.0, 1.0)
        layer3_roi = layer1_roi & (texture_diff > 0.01)
        layer3_mask[b] = layer3_roi.float()

        raw_gradient = normalized_l2 * 0.5 + normalized_tex * 0.5
        masked_gradient = torch.zeros_like(raw_gradient)
        masked_gradient[layer1_roi] = raw_gradient[layer1_roi]
        blind_spot = layer1_roi & (valid_count < 2)
        flatness_penalty = torch.clamp(var_pred / 0.008, 0.0, 1.0)
        smart_blind_heat = 0.10 + 0.15 * flatness_penalty
        masked_gradient[blind_spot] = torch.max(masked_gradient[blind_spot], smart_blind_heat[blind_spot])

        pred_gray = pred_rgb_b.mean(dim=0)
        is_bg = (pred_gray < 0.05) | (pred_gray > 0.95)
        bg_suppression = torch.ones_like(masked_gradient)
        bg_suppression[is_bg] = spatial_fade[is_bg]
        bg_suppression = _gaussian_blur(bg_suppression.unsqueeze(0), k=11, s=3.0)[0]
        gradient_mask[b] = masked_gradient * bg_suppression

    expanded_mask = F.max_pool2d(gradient_mask.unsqueeze(1), kernel_size=15, stride=1, padding=7).squeeze(1)
    blur_level_1 = _gaussian_blur(expanded_mask, k=15, s=5.0)
    blur_level_2 = _gaussian_blur(blur_level_1, k=51, s=15.0)
    blur_level_3 = _gaussian_blur(blur_level_2, k=101, s=35.0)
    final_gradient = (blur_level_1 * 0.4) + (blur_level_2 * 0.4) + (blur_level_3 * 0.2)

    fg_max = final_gradient.max()
    if fg_max > 1e-6:
        final_gradient_norm = final_gradient / fg_max
        final_gradient = 0.35 * torch.sin(torch.tensor(torch.pi / 2.0, device=final_gradient.device) * final_gradient_norm)

    ret = (
            _gaussian_blur(final_soft_mask, k=9, s=3.0),
            _gaussian_blur(layer1_mask, k=9, s=3.0),
            _gaussian_blur(layer3_mask, k=9, s=3.0),
            final_gradient)
    if return_layers:
        ret = ret + (layer1_mask, final_soft_mask, layer3_mask)
    return ret


def save_mvsgs_diffusion_data(batch, output, level, batch_idx, out_dir):
    pred_rgb = _mvsgs_rgb_chw(output, batch, level)
    B, _, H, W = pred_rgb.shape
    (
            soft_mask,
            hard_mask,
            blur_mask,
            gradient_mask,
            layer1_raw_mask,
            layer2_soft_mask,
            layer3_raw_mask,
    ) = compute_mvsgs_soft_mask(batch, output, level, return_layers=True)

    src_rgbs = batch['src_inps'][0].clamp(-1.0, 1.0)
    H_src, W_src = src_rgbs.shape[-2:]
    src_ints = batch['src_ixts'][0].clone()
    if (H_src, W_src) != (H, W):
        src_rgbs = F.interpolate(src_rgbs, size=(H, W), mode='bilinear', align_corners=False)
        src_ints = _scale_intrinsics(src_ints, H / H_src, W / W_src)

    src_depths_all = _source_depths(output, batch, level, (H, W))
    src_depths = src_depths_all[0].unsqueeze(1).float()
    src_valid_masks_all = _source_valid_masks(src_depths_all)
    src_valid_masks = src_valid_masks_all[0].unsqueeze(1).float()
    target_mvs_depth = _mvsgs_depth(output, batch, level)[0].unsqueeze(0).unsqueeze(0).float()
    target_mvs_std_map = _mvsgs_std(output, batch, level)
    target_mvs_std = None
    if target_mvs_std_map is not None:
        target_mvs_std = target_mvs_std_map[0].unsqueeze(0).unsqueeze(0).float()
    mvs_valid_mask = _mvsgs_valid_mask(batch, output, level, (H, W))[0].unsqueeze(0).unsqueeze(0).float()
    tar_rgb = (pred_rgb[0] * 2.0 - 1.0).clamp(-1.0, 1.0).unsqueeze(0)
    blur_mask_gamo = gradient_mask[0].unsqueeze(0).unsqueeze(0).clamp(0.0, 0.35)
    mask_layer1_raw_gamo = layer1_raw_mask[0].unsqueeze(0).unsqueeze(0).clamp(0.0, 1.0)
    mask_layer2_soft_gamo = layer2_soft_mask[0].unsqueeze(0).unsqueeze(0).clamp(0.0, 1.0)
    mask_layer3_raw_gamo = layer3_raw_mask[0].unsqueeze(0).unsqueeze(0).clamp(0.0, 1.0)

    render_scale = cfg.mvsgs.cas_config.render_scale[level]
    tar_int = _scale_intrinsics(batch['tar_ixt'][0], render_scale, render_scale).unsqueeze(0)
    all_intrinsics = torch.cat([src_ints, tar_int], dim=0).float()

    src_exts = _ensure_4x4(batch['src_exts'][0])
    tar_ext = _ensure_4x4(batch['tar_ext'][0]).unsqueeze(0)
    all_extrinsics_w2c = torch.cat([src_exts, tar_ext], dim=0).float()
    all_extrinsics_c2w = torch.inverse(all_extrinsics_w2c)

    cam_centers = all_extrinsics_c2w[:, :3, 3]
    scene_max = torch.max(cam_centers.abs().max(), src_depths.max()).item()
    scene_scale = 1.0
    if scene_max > 5.0:
        scene_scale = 5.0 / scene_max
        all_extrinsics_c2w[:, :3, 3] *= scene_scale
        src_depths *= scene_scale
        target_mvs_depth *= scene_scale
        if target_mvs_std is not None:
            target_mvs_std *= scene_scale
    all_extrinsics_w2c_scaled = torch.inverse(all_extrinsics_c2w)

    scene = _meta_value(batch['meta'], 'scene', 0)
    tar_view = _meta_value(batch['meta'], 'tar_view', 0)
    frame_id = _meta_value(batch['meta'], 'frame_id', 0)
    base_name = f"{scene}_{int(tar_view):03d}_{int(frame_id):03d}_{batch_idx:04d}"

    mono_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diffusion_nerf', 'mono_depth_mvsgs_v7p')
    mono_depth, mono_path = load_optional_mono_depth(
            base_name,
            mono_dir,
            (H, W),
            device=target_mvs_depth.device,
            dtype=target_mvs_depth.dtype)
    completion = complete_mvs_mono_depth(
            target_mvs_depth,
            valid_mask=mvs_valid_mask,
            mvs_std=target_mvs_std,
            mono_depth=mono_depth,
            blur_mask=blur_mask_gamo)
    completion_meta = completion['meta']
    completion_meta['mono_depth_path'] = mono_path or ''
    completion_meta['scene_scale'] = float(scene_scale)
    source_completion = _complete_source_depths(
            src_depths,
            src_valid_masks,
            base_name,
            H,
            W,
            device=src_depths.device)
    src_views = _meta_sequence(batch['meta'], 'src_views', 0)
    for idx, meta in enumerate(source_completion['source_depth_completion_meta']):
        if idx < len(src_views):
            meta['src_view'] = int(src_views[idx])
        meta['scene_scale'] = float(scene_scale)

    diffusion_data = {
            'source_rgbs': src_rgbs.half().cpu(),
            'pred_rgb': tar_rgb.half().cpu(),
            'source_depths': src_depths.half().cpu(),
            'source_valid_masks': src_valid_masks.half().cpu(),
            'source_mvs_depths': source_completion['source_mvs_depths'].half().cpu(),
            'source_mvs_valid_masks': source_completion['source_mvs_valid_masks'].half().cpu(),
            'source_mvs_confidences': source_completion['source_mvs_confidences'].half().cpu(),
            'source_mono_depths_raw': source_completion['source_mono_depths_raw'].half().cpu(),
            'source_mono_depths_aligned': source_completion['source_mono_depths_aligned'].half().cpu(),
            'source_complete_depths': source_completion['source_complete_depths'].half().cpu(),
            'source_depth_confidences': source_completion['source_depth_confidences'].half().cpu(),
            'source_fusion_soft_masks': source_completion['source_fusion_soft_masks'].half().cpu(),
            'source_depth_completion_meta': source_completion['source_depth_completion_meta'],
            'blur_mask': blur_mask_gamo.half().cpu(),
            'mask_layer1_raw': mask_layer1_raw_gamo.half().cpu(),
            'mask_layer2_soft': mask_layer2_soft_gamo.half().cpu(),
            'mask_layer3_raw': mask_layer3_raw_gamo.half().cpu(),
            'mvs_depth': target_mvs_depth.half().cpu(),
            'mvs_std': None if target_mvs_std is None else target_mvs_std.half().cpu(),
            'target_mvs_depth': target_mvs_depth.half().cpu(),
            'target_mvs_std': None if target_mvs_std is None else target_mvs_std.half().cpu(),
            'mvs_valid_mask': mvs_valid_mask.half().cpu(),
            'mvs_depth_confidence': completion['mvs_confidence'].half().cpu(),
            'mono_depth_raw': completion['mono_depth_raw'].half().cpu(),
            'mono_depth_aligned': completion['mono_depth_aligned'].half().cpu(),
            'complete_depth': completion['complete_depth'].half().cpu(),
            'depth_confidence': completion['depth_confidence'].half().cpu(),
            'fusion_soft_mask': completion['fusion_soft_mask'].half().cpu(),
            'depth_completion_meta': completion_meta,
            'intrinsics': all_intrinsics.cpu(),
            'extrinsics': all_extrinsics_w2c_scaled.cpu(),
            'extrinsics_w2c': all_extrinsics_w2c_scaled.cpu(),
            'extrinsics_c2w': all_extrinsics_c2w.cpu(),
            'extrinsics_format': 'w2c',
            'cond_num': src_rgbs.shape[0],
            'source': 'MVSGaussian',
            'render_level': level,
            'src_views': src_views,
    }
    torch.save(diffusion_data, os.path.join(out_dir, f'{base_name}_diffusion_data.pt'))

    if f'rgb_{level}' in batch:
        gt_rgb = batch[f'rgb_{level}'][0].reshape(H, W, 3).detach().cpu().numpy()
        imageio.imwrite(os.path.join(out_dir, f'{base_name}_0_gt_rgb.png'), (gt_rgb * 255).astype(np.uint8))

    pred_vis = pred_rgb[0].permute(1, 2, 0).detach().cpu().numpy()
    pred_bgr = (np.clip(pred_vis, 0.0, 1.0) * 255).astype(np.uint8)[:, :, ::-1]
    cv2.imwrite(os.path.join(out_dir, f'{base_name}_1_pred_rgb.png'), pred_bgr)

    mask_float = blur_mask_gamo[0, 0].detach().cpu().float().numpy()
    cv2.imwrite(os.path.join(out_dir, f'{base_name}_2_mask_gray.png'), (mask_float * 255).astype(np.uint8))
    mask_disp = np.clip(mask_float / 0.35, 0.0, 1.0) ** 2.0
    heatmap = cv2.applyColorMap((mask_disp * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = np.clip(mask_disp * 0.70, 0.0, 0.70)[..., None]
    overlay = (pred_bgr * (1.0 - alpha) + heatmap * alpha).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, f'{base_name}_3_mask_overlay.png'), overlay)

    cv2.imwrite(os.path.join(out_dir, f'{base_name}_4_target_mvs_depth.png'), depth_to_uint8(target_mvs_depth, mvs_valid_mask))
    cv2.imwrite(os.path.join(out_dir, f'{base_name}_5_mvs_valid_mask.png'), (mvs_valid_mask[0, 0].detach().cpu().numpy() * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(out_dir, f'{base_name}_6_mono_aligned_depth.png'), depth_to_uint8(completion['mono_depth_aligned'], mvs_valid_mask))
    cv2.imwrite(os.path.join(out_dir, f'{base_name}_7_complete_depth.png'), depth_to_uint8(completion['complete_depth']))
    cv2.imwrite(os.path.join(out_dir, f'{base_name}_8_depth_confidence.png'), (completion['depth_confidence'][0, 0].detach().cpu().numpy() * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(out_dir, f'{base_name}_9_src0_mvs_depth.png'), depth_to_uint8(source_completion['source_mvs_depths'][0:1], source_completion['source_mvs_valid_masks'][0:1]))
    cv2.imwrite(os.path.join(out_dir, f'{base_name}_10_src0_valid_mask.png'), (source_completion['source_mvs_valid_masks'][0, 0].detach().cpu().numpy() * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(out_dir, f'{base_name}_11_src0_mono_aligned_depth.png'), depth_to_uint8(source_completion['source_mono_depths_aligned'][0:1], source_completion['source_mvs_valid_masks'][0:1]))
    cv2.imwrite(os.path.join(out_dir, f'{base_name}_12_src0_complete_depth.png'), depth_to_uint8(source_completion['source_complete_depths'][0:1]))
    cv2.imwrite(os.path.join(out_dir, f'{base_name}_13_src0_depth_confidence.png'), (source_completion['source_depth_confidences'][0, 0].detach().cpu().numpy() * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(out_dir, f'{base_name}_14_mask_layer1_raw.png'), (mask_layer1_raw_gamo[0, 0].detach().cpu().numpy() * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(out_dir, f'{base_name}_15_mask_layer2_soft.png'), (mask_layer2_soft_gamo[0, 0].detach().cpu().numpy().clip(0.0, 1.0) * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(out_dir, f'{base_name}_16_mask_layer3_raw.png'), (mask_layer3_raw_gamo[0, 0].detach().cpu().numpy() * 255).astype(np.uint8))

    return base_name, float(mask_float.max()), float(mask_float.mean())
