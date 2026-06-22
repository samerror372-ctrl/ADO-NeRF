import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any
from types import SimpleNamespace
from .feature_net import FeatureNet
from .depth_net import DepthNet
from .nerf import NeRF
from .decoder_rdn import Decoder
from .bundle_sampler import BundleSampler
from . import utils


class Network(nn.Module):
    def __init__(self, 
                 config: SimpleNamespace):
        super(Network, self).__init__()

        # FPN
        base_channels = config.fpn.base_channels  # base channels of the FeatureNet
        feat_dims = config.fpn.feat_dims  # pyramid feature dimension at each level
        self.feature_net = FeatureNet(base_channels=base_channels, out_channels=feat_dims)

        # MVS
        self.voxel_dim = config.mvs.voxel_dim  # dimension of the voxel feature
        self.depth_net = DepthNet(config)

        # Sampler
        global_num_depth = config.nerf.global_num_depth  # number of global samples per ray
        max_mipmap_level = config.nerf.max_mipmap_level  # maximum mipmap level to use for interpolation
        self.max_num_samples = config.nerf.max_num_samples  # maximum number of samples per ray at each stage
        self.b_size = config.nerf.bundle_size  # ray bundle size (upsampling factor of the rendered feature map)
        if self.b_size <= 0 or (self.b_size & (self.b_size - 1)) != 0:
            raise ValueError('`Bundle size` must be a power of 2.')
        self.inv_depth = config.mvs.inv_depth[-1]  # whether to inverse depth for sampling
        self.is_adaptive = config.nerf.is_adaptive  # whether to use adaptive sampling
        self.sampler = BundleSampler(global_num_depth, max_mipmap_level)

        # NeRF
        self.feat_level = 0  # use the feature map at this level, which is closest to the bundle feature map
        while self.feat_level < len(config.fpn.feat_scales) and config.fpn.feat_scales[self.feat_level] < 1. / self.b_size:
            self.feat_level += 1
        feat_dim = feat_dims[self.feat_level]  # dimension of the used 2D feature
        self.nerf_hidden_dims = config.nerf.nerf_hidden_dims  # hidden dimensions of NeRF
        self.viewdir_agg = config.nerf.viewdir_agg  # whether to use view direction aggregation
        self.render_scale = 1.  # scale of the rendered image, default to 1
        self.nerf = NeRF(self.nerf_hidden_dims, feat_dim, self.voxel_dim, self.viewdir_agg)

        # Decoder
        self.dec_layers = config.nerf.dec_layers  # number of layers of the decoder at each stage
        self.upsampler = Decoder(feat_dim+3+self.voxel_dim, 3, num_feats=64, num_layers=self.dec_layers, upscale_factor=self.b_size)
        self.reweighting = config.nerf.reweighting  # whether to reweight
    
    def render_bundles(self, 
                       rgbs_feat_rgb_dir: torch.Tensor, 
                       vox_feat: torch.Tensor, 
                       z_vals: torch.Tensor, 
                       indices: torch.Tensor, 
                       samples_per_bundle: torch.Tensor,
                       vis_weights: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        """Render bundles."""
        sigma, feat = self.nerf(vox_feat, rgbs_feat_rgb_dir, vis_weights=vis_weights)  

        # Differentiable Volumetric Rendering
        num_bundles = samples_per_bundle.shape[0]
        weights, inverse_indices = utils.render_weight_from_density(sigma, indices, num_bundles)  
        
        if self.inv_depth:  # to disparity
            z_vals = 1. / z_vals

        feat, depth, opacity = utils.accumulate_value_along_rays(feat, z_vals, weights, indices, num_bundles, inverse_indices)  
        
        if self.inv_depth:  # to depth
            depth = 1. / depth

        return feat, depth, opacity

    def forward(self,
                batch: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        # Extract data
        src_views = batch['src_views']
        tar_views = batch['tar_views']
        near_far = batch['near_far']  # (B, 2)
        src_images = src_views['rgb']  # (B, num_views, 3, H_orig, W_orig)
        B, num_views, _, H_orig, W_orig = src_images.shape
        src_exts = src_views['extrinsics']  # (B, num_views, 4, 4)
        src_ints = src_views['intrinsics'].clone()  # (B, num_views, 3, 3)
        tar_exts = tar_views['extrinsics']  # (B, 4, 4)
        tar_ints = tar_views['intrinsics'].clone()  # (B, 3, 3)
        
        step_ratio = batch.get('step_ratio', 1.0)
        epoch_ratio = batch.get('epoch_ratio', step_ratio) 
        
        if isinstance(step_ratio, torch.Tensor):
            step_ratio = step_ratio.item()
        if isinstance(epoch_ratio, torch.Tensor):
            epoch_ratio = epoch_ratio.item()

        # Rescale if required
        if 'render_scale' in batch:
            self.render_scale = batch['render_scale'][0].item()
        if self.render_scale != 1.:
            src_images = F.interpolate(src_images.flatten(0, 1), scale_factor=self.render_scale, mode='bilinear', align_corners=False).unflatten(dim=0, sizes=(B, num_views))
            H_orig, W_orig = src_images.shape[-2:]
            src_ints[..., :2, :] *= self.render_scale
            tar_ints[:, :2, :] *= self.render_scale
        
        # Extract pyramid features
        ms_feats = self.feature_net(src_images.flatten(0, 1))  
        ms_feats = [f.unflatten(dim=0, sizes=(B, num_views)) for f in ms_feats]  

        # Estimate depth for TARGET view
        mvs_depths, depth_range_list, vol_range_list, feat_volume_list, blend_rgbs = \
            self.depth_net(src_images, ms_feats, src_exts, src_ints, tar_exts, tar_ints, near_far)
            
        depth_range = depth_range_list[-1]  
        vol_range = vol_range_list[-1]  
        feat_volume = feat_volume_list[-1]  
        mvs_depth = mvs_depths[-1]  

        # =========================================================================
        # 动态预测源视角 (Source Views) 原始深度
        # =========================================================================
        src_mvs_depths = None
        
        if not self.training or epoch_ratio >= 0.05:
            src_depths_list = []
            with torch.no_grad():
                for v in range(num_views):
                    ref_exts = src_exts[:, v]
                    ref_ints = src_ints[:, v]
                    
                    v_depths, _, _, _, _ = self.depth_net(
                        src_images, ms_feats, src_exts, src_ints, ref_exts, ref_ints, near_far
                    )
                    src_depths_list.append(v_depths[-1]) 
            
            src_mvs_depths = torch.stack(src_depths_list, dim=1) 
            
            H_bundle, W_bundle = H_orig // self.b_size, W_orig // self.b_size
            if src_mvs_depths.shape[-2:] != (H_bundle, W_bundle):
                src_mvs_depths = F.interpolate(
                    src_mvs_depths.flatten(0, 1).unsqueeze(1), 
                    size=(H_bundle, W_bundle), 
                    mode='nearest' 
                ).view(B, num_views, H_bundle, W_bundle)
        # =========================================================================

        # =========================================================================
        # 🛡️ [核心优化机制] 局部解耦的 Z-Buffer 遮挡校正 (不破坏原始 MVS 输出)
        # =========================================================================
        src_mvs_depths_for_encode = None
        
        if src_mvs_depths is not None:
            # 👑 核心：深度克隆一个副本，专供内部编码器的 Z-buffer 遮挡计算使用
            src_mvs_depths_for_encode = src_mvs_depths.clone()
            
            for b_idx in range(B):
                near_bound = near_far[b_idx, 0]  # 近平面物理边界值 (如 425.0)
                far_bound = near_far[b_idx, 1]   # 远平面物理边界值 (如 1000.0)
                
                # 计算近平面容差，判定哪些像素由于纹理匹配失败发生了“近平面塌陷”
                # 考虑到插值的影响，使用近平面向远端拉伸 1% 作为鲁棒的判定水位线
                is_collapsed_bg = src_mvs_depths_for_encode[b_idx] <= (near_bound + 0.01 * (far_bound - near_bound))
                
                # 【局部修正】将这些假的前置隐形墙，推到物理大后方（远平面）
                src_mvs_depths_for_encode[b_idx][is_collapsed_bg] = far_bound
        # =========================================================================

        # Build rays
        self.sampler.build_rays(tar_exts, tar_ints, (H_orig, W_orig), near_far[:, 0], near_far[:, 1])
        
        # Adaptive bundle sampling
        H, W = H_orig // self.b_size, W_orig // self.b_size  
        if depth_range.shape[2:] != (H, W):
            depth_range = F.interpolate(depth_range, size=(H, W), mode='bilinear', align_corners=False)
            vol_range = F.interpolate(vol_range, size=(H, W), mode='bilinear', align_corners=False)
            mvs_depth = F.interpolate(mvs_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)
            
        rays_xyz, uvd, z_vals, ball_radii, indices, samples_per_batch, samples_per_bundle = \
            self.sampler.sample(depth_range, vol_range, self.b_size, self.max_num_samples, self.inv_depth, self.is_adaptive)
        
        # Sphere-based encoding
        img_feat_rgb = ms_feats[self.feat_level]
        if img_feat_rgb.shape[-2:] != (H, W):
            img_feat_rgb = F.interpolate(img_feat_rgb.flatten(0, 1), size=(H, W), mode='bilinear', align_corners=False).unflatten(dim=0, sizes=(B, num_views)) 
        img_feat_rgb = torch.cat((img_feat_rgb, 
                                  F.interpolate(src_images.flatten(0, 1), size=(H, W), mode='bilinear', align_corners=False).unflatten(dim=0, sizes=(B, num_views))), 
                                 dim=2) 
        
        # 👑 传入优化后的临时克隆深度 `src_mvs_depths_for_encode`，实现正确的可见性过滤
        rgbs_feat_rgb_dir, vox_feat, vis_weights = \
            self.sampler.encode(src_images, img_feat_rgb, feat_volume, rays_xyz, uvd, ball_radii, 
                                src_exts, src_ints, tar_exts, samples_per_batch,
                                src_mvs_depths=src_mvs_depths_for_encode, step_ratio=step_ratio)
        
        # Render bundles
        bundle_feat, bundle_depth, bundle_opacity = self.render_bundles(rgbs_feat_rgb_dir, vox_feat, z_vals, indices, samples_per_bundle, vis_weights=vis_weights)
        
        nerf_feat = bundle_feat.view(B, H, W, -1).permute(0, 3, 1, 2)  
        nerf_depth = bundle_depth.view(B, H, W)
        nerf_opacity = bundle_opacity.view(B, H, W)
        num_rays_per_bundle = 3 * self.b_size ** 2
        rgb_c = self.upsampler(nerf_feat[:, num_rays_per_bundle:])  
        rgb_f = F.pixel_shuffle(nerf_feat[:, :num_rays_per_bundle], self.b_size)  
        nerf_depth = F.interpolate(nerf_depth.unsqueeze(1), scale_factor=self.b_size, mode='bilinear', align_corners=False).squeeze(1)  
        nerf_opacity = F.interpolate(nerf_opacity.unsqueeze(1), scale_factor=self.b_size, mode='bilinear', align_corners=False).squeeze(1)  
        
        img = rgb_c + rgb_f  
        if self.reweighting:
            img = 0.5 * (img + rgb_f)
            
        # =========================================================================
        # 👑 [返回校验] 确保外抛的 ret 字典装载的依然是【未修改的纯粹原始 MVS 深度】
        # =========================================================================
        ret = {'rgb': img, 
               'nerf_depth': nerf_depth, 
               'mvs_depth': mvs_depth,          # 完全未受影响的原始数据
               'opacity': nerf_opacity,
               'src_mvs_depths': src_mvs_depths} # 完全未受影响的原始数据

        return ret, mvs_depths, blend_rgbs
