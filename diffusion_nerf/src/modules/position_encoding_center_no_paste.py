import math
from typing import List

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


def slice_vae_encode(vae, image, sub_size):  # vae fails to encode large tensor directly, we need to slice it
    with torch.no_grad(), torch.autocast("cuda", enabled=True):
        if len(image.shape) == 5:  # [B,F,C,H,W]
            b, f, _, h, w = image.shape
            image = einops.rearrange(image, "b f c h w -> (b f) c h w")
        else:
            b, f, h, w = None, None, None, None

        if (image.shape[-1] > 256 and image.shape[0] > sub_size) or (image.shape[0] > 192):
            slice_num = image.shape[0] // sub_size
            if image.shape[0] % sub_size != 0:
                slice_num += 1
            latents = []
            for i in range(slice_num):
                latents_ = vae.encode(image[i * sub_size:(i + 1) * sub_size])
                latents_ = latents_.latent_dist.sample() if hasattr(latents_, "latent_dist") else latents_.sample()
                latents.append(latents_)
            latents = torch.cat(latents, dim=0)
        else:
            latents = vae.encode(image).latent_dist.sample()

        if f is not None:
            latents = einops.rearrange(latents, "(b f) c h w -> b f c h w", f=f)

        return latents


def freq_encoding(rays_3d, embed_dim=16, camera_longest_side=None):
    # rays_3d: [b, n, 3]
    rays_3d = rays_3d.to(torch.float32)
    if camera_longest_side is not None:
        rescale_value = 512 / camera_longest_side
        rays_3d = rays_3d * rescale_value
    else:  # camera translation is normalized into [-1~1]
        rays_3d = rays_3d * 512

    pe = []
    freq_bands = 2. ** torch.linspace(0., embed_dim // 2 - 1, steps=embed_dim // 2)
    for freq in freq_bands:
        for p_fn in [torch.sin, torch.cos]:
            pe.append(p_fn(rays_3d * freq))

    pe = torch.cat(pe, dim=-1)

    return pe


def depth_freq_encoding(depth, device, embed_dim=32):
    # depth: [b, 1, h, w]
    depth = depth.to(torch.float32).to(device)
    b, _, h, w = depth.shape

    pe = torch.zeros((b, embed_dim, h, w), dtype=torch.float32, device=device)
    freq_bands = 2. ** torch.linspace(0., embed_dim // 2 - 1, steps=embed_dim // 2)
    freq_bands = freq_bands.reshape(1, embed_dim // 2, 1, 1).to(device)
    pe[:, 0::2] = torch.sin(depth * freq_bands)
    pe[:, 1::2] = torch.cos(depth * freq_bands)

    return pe


def points_padding(points):
    padding = torch.ones_like(points)[..., 0:1]
    points = torch.cat([points, padding], dim=-1)
    return points


def get_3d_priors(config, depth, K, w2c, cond_num, nframe, device, contract=True,
                  colors=None, latents=None, vae=None, prior_type="3dpe", tar_idx_batch=None, zoom_scale=1):
    # prior_type: "3dpe", "latent", "3dpe+latent", "pixel"
    if prior_type == "3dpe":
        coords = global_position_encoding_3d(config, depth, K, w2c, cond_num, nframe, device,
                                             pe_scale=1 / 8, embed_dim=192, contract=contract, tar_idx_batch=tar_idx_batch)
        #breakpoint()
        
    elif prior_type == "latent":   ####color?
        #breakpoint()
        coords = global_position_encoding_3d(config, depth, K, w2c, cond_num, nframe, device,
                                             pe_scale=1.0, embed_dim=4, contract=True, colors=colors, vae=vae,  tar_idx_batch=tar_idx_batch)
        
    elif prior_type == "warp_latent":
        if len(latents.shape) == 5:
            latents = einops.rearrange(latents, 'b f c h w -> (b f) c h w')
        coords = global_position_encoding_3d(config, depth, K, w2c, cond_num, nframe, device,
                                             pe_scale=1 / 8, embed_dim=4, contract=True, latents=latents,
                                             vae=vae, return_pixel=True,  tar_idx_batch=tar_idx_batch)
    elif prior_type == "3dpe+latent":
        coords_3dpe = global_position_encoding_3d(config, depth, K, w2c, cond_num, nframe, device,
                                                  pe_scale=1 / 8, embed_dim=192, contract=contract)
        coords_latent = global_position_encoding_3d(config, depth, K, w2c, cond_num, nframe, device,
                                                    pe_scale=1.0, embed_dim=4, contract=True, colors=colors, vae=vae,  tar_idx_batch=tar_idx_batch)
        return [coords_3dpe, coords_latent]
    elif prior_type == "3dpe+warp_latent":
        coords_3dpe = global_position_encoding_3d(config, depth, K, w2c, cond_num, nframe, device,
                                                  pe_scale=1 / 8, embed_dim=192, contract=contract,  tar_idx_batch=tar_idx_batch)
        if len(latents.shape) == 5:
            latents = einops.rearrange(latents, 'b f c h w -> (b f) c h w')
        coords_latent = global_position_encoding_3d(config, depth, K, w2c, cond_num, nframe, device,
                                                    pe_scale=1 / 8, embed_dim=4, contract=True, latents=latents,
                                                    vae=vae, return_pixel=True, tar_idx_batch=tar_idx_batch)
        return [coords_3dpe, coords_latent]
    elif prior_type == "pixel":
        coords = global_position_encoding_3d(config, depth, K, w2c, cond_num, nframe, device,
                                             pe_scale=1.0, embed_dim=4, contract=True, colors=colors, vae=vae,
                                             return_pixel=True, tar_idx_batch=tar_idx_batch)
        print("pixel!")
        #breakpoint()
    elif prior_type == "3dpe+pixel":
        
        coords_3dpe = global_position_encoding_3d(config, depth, K, w2c, cond_num, nframe, device,
                                                  pe_scale=1 / 8, embed_dim=192, contract=contract, tar_idx_batch=tar_idx_batch, zoom_scale=zoom_scale)  # [27, 193, 33, 60]
        coords_pixel = global_position_encoding_3d(config, depth, K, w2c, cond_num, nframe, device,
                                                   pe_scale=1.0, embed_dim=4, contract=True, colors=colors, vae=vae,
                                                   return_pixel=True, tar_idx_batch=tar_idx_batch, zoom_scale=zoom_scale) # [27, 4, 264, 480]  #cond_num 20 #nframe 27
        
        
        # --- Save warped RGB(+mask) as RGBA PNGs ---
        import os
        from torchvision.utils import save_image

        save_dir = os.path.join(getattr(config, "save_path", "."), "coords_pixel_png")
        os.makedirs(save_dir, exist_ok=True)

        # 拆成 RGB 與 mask
        rgb  = coords_pixel[:, :3]                      # [-1, 1] → 需轉到 [0, 1]
        alpha = coords_pixel[:, 3:4]                    # 可見性遮罩，已是 0/1（或近似）

        # 轉成 RGBA（把 alpha 當成 PNG 透明度）
        rgba = torch.cat([ (rgb.clamp(-1, 1) + 1) / 2, alpha.clamp(0, 1) ], dim=1)

        for i in range(rgba.shape[0]):
            save_image(rgba[i], os.path.join(save_dir, f"frame_{i:03d}.png"))
                
        print("3dpe+pixel!")
        #breakpoint()
        
        return [coords_3dpe, coords_pixel]
    
    elif prior_type == "h3dpe+pixel":
        coords_3dpe = global_position_encoding_3d(config, depth, K, w2c, cond_num, nframe, device,
                                                  pe_scale=1.0, embed_dim=48, contract=contract)
        coords_pixel = global_position_encoding_3d(config, depth, K, w2c, cond_num, nframe, device,
                                                   pe_scale=1.0, embed_dim=4, contract=True, colors=colors, vae=vae,
                                                   return_pixel=True)
        
        return [coords_3dpe, coords_pixel]
    else:
        raise NotImplementedError

    return coords


def global_position_encoding_3d(config, depth, K, w2c, cond_num, nframe, device,
                                pe_scale=1 / 8, embed_dim=192, contract=True, colors=None, latents=None,
                                vae=None, return_pixel=False, tar_idx_batch=None, zoom_scale=None):
    # get 3d position in the world coordinate
    # colors are used for debug only [b*nframe,3,h,w]
    '''
    :param depth:[b*nframe,1,h,w]
    :param K: [b*nframe,3,3]
    :param w2c: [b*nframe,4,4]
    '''
    b = depth.shape[0] // nframe
    depth = depth.to(device).float()
    depth = einops.rearrange(depth, "(b f) c h w -> b f c h w", f=nframe)[:, :cond_num]
    # check zero depth
    if contract:
        mid_depth = torch.median(depth.reshape(b, cond_num, 1, -1), dim=-1)[0]
        mid_depth_ = mid_depth * config.model_cfg.depth_mid_times
        mid_depth_ = einops.repeat(mid_depth_, "b cond c -> b cond c h w", h=depth.shape[-2], w=depth.shape[-1])
        depth[depth > mid_depth_] = ((2 * mid_depth_[depth > mid_depth_]) - (mid_depth_[depth > mid_depth_] ** 2 / (depth[depth > mid_depth_] + 1e-6)))
    depth_mask = torch.mean(depth.abs(), dim=[3, 4], keepdim=True)
    depth_mask = (depth_mask != 0).float().reshape(b, cond_num)  # [b,cond]
    depth = einops.rearrange(depth, "b cond c h w -> (b cond) c h w")

    if colors is not None:
        pe_scale = 1.0
        colors = colors.to(device)
        #breakpoint()
        
    if pe_scale < 1:
        import torch.nn.functional as F
        depth = F.interpolate(depth, scale_factor=pe_scale, mode='nearest')
        K = K.clone()
        K[:, 0:2] *= pe_scale

    K_inv = K.cpu().inverse()
    c2w = w2c.cpu().inverse()

    K = K.to(device).float()
    K_inv = K_inv.to(device).float()
    c2w = c2w.to(device).float()
    w2c = w2c.to(device).float()

    bc, _, h, w = depth.shape
    points2d = torch.stack(torch.meshgrid(torch.arange(w, dtype=torch.float32),
                                          torch.arange(h, dtype=torch.float32), indexing="xy"), -1).to(device)  # [h,w,2]
    points3d = einops.repeat(points_padding(points2d), "h w c -> bc c (h w)", bc=bc)

    K_inv_ = einops.rearrange(K_inv, "(b f) c1 c2 -> b f c1 c2", f=nframe)[:, :cond_num].reshape(-1, 3, 3)
    c2w_ = einops.rearrange(c2w, "(b f) c1 c2 -> b f c1 c2", f=nframe)[:, :cond_num].reshape(-1, 4, 4)

    points3d = K_inv_ @ points3d * depth.reshape(bc, 1, h * w)  # [bc, 3, hw] points3d in camera coordinate
    known_points3d = c2w_ @ points_padding(points3d.permute(0, 2, 1)).permute(0, 2, 1)  # [bc, 4, hw]
    cond_points3d = einops.rearrange(known_points3d, "(b cond) c hw -> b cond c hw", cond=cond_num)[:, :, :3]
    known_points3d = einops.repeat(known_points3d, "(b cond) c hw -> (b tar) cond c hw", cond=cond_num, tar=nframe - cond_num)  # [b*tar,cond,4,hw]

    # warp known points3d to target views
    K_target = einops.rearrange(K, "(b f) c1 c2 -> b f c1 c2", f=nframe)[:, cond_num:].reshape(-1, 1, 3, 3)  # [b*tar,1,3,3]
    w2c_target = einops.rearrange(w2c, "(b f) c1 c2 -> b f c1 c2", f=nframe)[:, cond_num:].reshape(-1, 1, 4, 4)  # [b*tar,1,4,4]
    target_grid = K_target @ (w2c_target @ known_points3d)[:, :, :3]  # world to camera
    z = target_grid[:, :, 2:3]
    z = einops.rearrange(z, "(b tar) cond c hw -> b tar cond c hw", b=b)
    z = z * depth_mask.reshape(b, 1, cond_num, 1, 1) + (-1) * (1 - depth_mask.reshape(b, 1, cond_num, 1, 1))  # depth全=0区域z设为负数
    z = einops.rearrange(z, "b tar cond c hw -> (b tar) cond c hw")
    target_grid = target_grid[:, :, :2] / (z + 1e-6)  # [b*tar,cond,2,hw]

    ch = 3  # for pixel and xyz, ch=3; for latent, ch=4
    #### warp RGB
    if colors is not None: #warp rgb
        colors = einops.rearrange(colors, "(b f) c h w -> b f c h w", f=nframe)
        cond_colors = colors[:, :cond_num]
        known_points3d = einops.repeat(cond_colors, "b cond c h w -> (b tar) cond c (h w)", tar=nframe - cond_num)
        cond_points3d = einops.rearrange(cond_colors, "b cond c h w -> b cond c (h w)")
    elif latents is not None:
        latents = einops.rearrange(latents, "(b f) c h w -> b f c h w", f=nframe)
        cond_latents = latents[:, :cond_num]
        known_points3d = einops.repeat(cond_latents, "b cond c h w -> (b tar) cond c (h w)", tar=nframe - cond_num)
        cond_points3d = einops.rearrange(cond_latents, "b cond c h w -> b cond c (h w)")
        ch = 4

    proj_x, proj_y = target_grid[:, :, 0].round().long(), target_grid[:, :, 1].round().long()  # [b*tar,cond,hw]
    proj_index = proj_y * w + proj_x

    btar = proj_index.shape[0]
    target_points3d = torch.zeros((btar, ch, h * w), dtype=torch.float32, device=device)
    target_mask = torch.zeros((btar, 1, h * w), dtype=torch.float32, device=device)
    # 将b*tar和hw都压到第一维度, (N=b*tar*hw)，方便mask
    target_points3d = target_points3d.permute(0, 2, 1).reshape(-1, ch)  # [N,3]
    target_mask = target_mask.permute(0, 2, 1).reshape(-1, 1)
    z = z.permute(0, 3, 1, 2).reshape(-1, cond_num)  # [N,cond]
    z_save = torch.ones((target_points3d.shape[0],), dtype=torch.float32, device=device) * 1e4  # 保存z-score用来判断multiview覆盖关系
    known_points3d = known_points3d[:, :, :ch].permute(0, 3, 1, 2).reshape(-1, cond_num, ch)  # [b*tar,cond,4,hw]->[b*tar*hw,cond,3]

    batch_indices = torch.arange(btar, dtype=torch.long, device=device) * h * w  # [b*tar]
    proj_index = proj_index + batch_indices[:, None, None]
    proj_index = proj_index.permute(0, 2, 1).reshape(-1, cond_num)
    
    
    for i in range(cond_num): 
        
        #breakpoint()
        # get valid grid mask
        x_mask = ((proj_x[:, i] >= w) + (proj_x[:, i] < 0)).reshape(-1)
        y_mask = ((proj_y[:, i] >= h) + (proj_y[:, i] < 0)).reshape(-1)
        z_mask = z[:, i] <= 0.0  # 排除z负数的情况
        proj_mask = (1 - torch.clamp(x_mask + y_mask + z_mask, 0, 1)).float()  # valid=1, invalid=0 [b*tar,hw]
        proj_index_ = proj_index[:, i]  # 在target view下的目标坐标
        # 判断新的z是否比z_save中更近，如果更近，则更新
        proj_mask[proj_mask.bool()] *= (z[proj_mask.bool(), i] < z_save[proj_index_[proj_mask.bool()]]).float()
        target_points3d[proj_index_[proj_mask.bool()]] = known_points3d[proj_mask.bool(), i].float()
        z_save[proj_index_[proj_mask.bool()]] = z[proj_mask.bool(), i].float()
        target_mask[proj_index_[proj_mask.bool()]] = 1

    target_points3d = einops.rearrange(target_points3d, "(b tar hw) c -> b tar c hw",
                                       b=b, tar=nframe - cond_num, hw=h * w) # [1, 7, 3, 1980]
    points3d_prior = torch.cat([cond_points3d, target_points3d], dim=1)  # [b,f,c,hw] # [1, 27, 3, 1980]
    # cond_points3d [1, 20, 3, 1980]
    

    # process mask
    target_mask = einops.rearrange(target_mask, "(b tar hw) c -> b tar c hw",
                                   b=b, tar=nframe - cond_num, hw=h * w)
    cond_mask = torch.ones((b, cond_num, 1, h * w), dtype=torch.float32, device=device)  # [b,cond,1,hw]
    cond_mask = cond_mask * depth_mask.reshape(b, cond_num, 1, 1)
    coord_mask = torch.cat([cond_mask, target_mask], dim=1)  # [b,f,1,hw]
    

    
    # save image 
    if colors is not None or latents is not None: #[1, 27, 3, 264, 480]
        if vae is not None:
            imgs = einops.rearrange(points3d_prior, "b f c (h w) -> (b f) c h w", h=h, w=w)
            coord_mask = einops.rearrange(coord_mask, "b f c (h w) -> (b f) c h w", h=h, w=w)
            if return_pixel:
                
                
                #coords = torch.cat([imgs, coord_mask], dim=1) #[27, 4, 264, 480]
                tar_idx_to_cond_idx = [int(x) for x in tar_idx_batch] 
     
                    
                # 建立空白底：直接 zeros
                B = depth.shape[0] // nframe   # batch size
                H, W = h, w
                imgs_blank  = torch.zeros((B * nframe, 3, H, W), device=device, dtype=torch.float32)
                alpha_blank = torch.zeros((B * nframe, 1, H, W), device=device, dtype=torch.float32)
                
                # ✅ 1) 預設：直接回傳純 warp
                coords = torch.cat([imgs, coord_mask], dim=1)

                ENABLE_PASTE = False  # 或直接刪掉貼圖區塊

                # ✅ 2) 只有需要貼圖時，才計算對應表並覆寫 imgs/coord_mask
                if ENABLE_PASTE and (tar_idx_batch is not None):
                    tar_idx_to_cond_idx = [int(x) for x in tar_idx_batch]

                    B = imgs.shape[0] // nframe
                    import torch.nn.functional as F_torch

                    bf_imgs  = einops.rearrange(imgs,       "(b f) c h w -> b f c h w", b=B, f=nframe)
                    bf_alpha = einops.rearrange(coord_mask, "(b f) c h w -> b f c h w", b=B, f=nframe)

                    scale = zoom_scale
                    _, nF, C, H, W = bf_imgs.shape

                    for j, src in enumerate(tar_idx_to_cond_idx):
                        assert 0 <= src < cond_num
                        tgt = cond_num + j
                        ref = bf_imgs[:, src]

                        patch_h = max(1, int(round(H * scale)))
                        patch_w = max(1, int(round(W * scale)))
                        ref_patch = F_torch.interpolate(ref, size=(patch_h, patch_w),
                                                        mode='bilinear', align_corners=False)

                        y0 = (H - patch_h) // 2; x0 = (W - patch_w) // 2
                        y1 = y0 + patch_h;       x1 = x0 + patch_w

                        bf_imgs[:,  tgt, :, y0:y1, x0:x1] = ref_patch
                        bf_alpha[:, tgt, :, y0:y1, x0:x1] = 1.0

                    imgs       = einops.rearrange(bf_imgs,  "b f c h w -> (b f) c h w")
                    coord_mask = einops.rearrange(bf_alpha, "b f c h w -> (b f) c h w")
                    coords     = torch.cat([imgs, coord_mask], dim=1)

                return coords


                #breakpoint()
                #return coords
            else:
                latent = slice_vae_encode(vae, imgs.to(torch.float16), sub_size=24)
                latent = latent * 0.18215
                coord_mask = F.interpolate(coord_mask, size=(latent.shape[2], latent.shape[3]), mode='nearest')
                coords = torch.cat([latent, coord_mask], dim=1)  # [bf,c+1,h,w]
                return coords
        else:  # return numpy image for debug and showcase
            imgs = einops.rearrange(points3d_prior, "b f c (h w) -> b f c h w", h=h, w=w)
            imgs = (imgs + 1) / 2
            color_warps = []
            for i in range(imgs.shape[0]):
                res = []
                for j in range(imgs.shape[1]):
                    res.append(np.array(transforms.ToPILImage()(imgs[i, j])))
                res = np.concatenate(res, axis=1)  # [h,wf,3]
                color_warps.append(res)
            print("vae is none!")
            #breakpoint()
            return color_warps #### warped rgb
        
    else:
        # ===== 在做通道內正規化前，先留一份 raw CCM =====
        points3d_prior_raw = points3d_prior.clone()   ### <--- 新增：留 raw 版給可視化

        # （原本的通道內正規化，留給模型用）
        points3d_max = torch.max(points3d_prior, dim=2, keepdim=True)[0]
        points3d_min = torch.min(points3d_prior, dim=2, keepdim=True)[0]
        points3d_prior = (points3d_prior - points3d_min) / (points3d_max - points3d_min + 1e-6)

        # ====== 可視化：用 raw 版在空間維度做 min–max ======
        import os
        from torchvision.utils import save_image
        vis_dir = os.path.join(getattr(config, "save_path", "."), "ccm_vis")
        os.makedirs(vis_dir, exist_ok=True)

        # [b,f,3,hw] -> [B*F,3,H,W]
        xyz_raw = einops.rearrange(points3d_prior_raw, "b f c (h w) -> (b f) c h w", h=h, w=w)

        # 在空間維度做 per-image-per-channel 正規化（避免全白）
        mins = xyz_raw.amin(dim=(2,3), keepdim=True)  ### <---
        maxs = xyz_raw.amax(dim=(2,3), keepdim=True)  ### <---
        xyz_vis = (xyz_raw - mins) / (maxs - mins + 1e-6)

        # （可選）也把 mask 存一下
        mask_img = einops.rearrange(coord_mask, "b f c (h w) -> (b f) c h w", h=h, w=w)

        for i in range(xyz_vis.shape[0]):
            save_image(xyz_vis[i], os.path.join(vis_dir, f"ccm_xyz_{i:03d}.png"))  # x,y,z -> R,G,B
            save_image(xyz_vis[i,2:3], os.path.join(vis_dir, f"ccm_z_{i:03d}.png"))  # z 灰階
            save_image(mask_img[i],   os.path.join(vis_dir, f"ccm_mask_{i:03d}.png"))

        # convert to fourier position (CCM -> PE)
        div_term = torch.exp(torch.arange(0, embed_dim // 3, 2).float() * (-math.log(10000.0) / (embed_dim // 3))).to(device)
        div_term = div_term.reshape(1, 1, -1, 1)
        points3d_prior_pe = torch.zeros((b, nframe, embed_dim, h * w), dtype=torch.float32, device=device)  # [b,f,c,hw]
        points3d_prior_pe[:, :, 0::6] = torch.sin(points3d_prior[:, :, 0:1] * div_term)
        points3d_prior_pe[:, :, 1::6] = torch.cos(points3d_prior[:, :, 0:1] * div_term)
        points3d_prior_pe[:, :, 2::6] = torch.sin(points3d_prior[:, :, 1:2] * div_term)
        points3d_prior_pe[:, :, 3::6] = torch.cos(points3d_prior[:, :, 1:2] * div_term)
        points3d_prior_pe[:, :, 4::6] = torch.sin(points3d_prior[:, :, 2:3] * div_term)
        points3d_prior_pe[:, :, 5::6] = torch.cos(points3d_prior[:, :, 2:3] * div_term)

        
        """
        tar_idx_to_cond_idx = [int(x) for x in tar_idx_batch] 
        
        
        # === (新增) 依 tar_idx_to_cond_idx 對 target 的 CCM 做覆寫（在做全域遮罩前）===
        if tar_idx_batch is not None:
            tar_idx_to_cond_idx = [int(x) for x in tar_idx_batch]
            assert len(tar_idx_to_cond_idx) == (nframe - cond_num), \
                f"Expected {nframe - cond_num} mappings, got {len(tar_idx_to_cond_idx)}"

            bf_ccm  = einops.rearrange(points3d_prior_pe, "b f c (h w) -> b f c h w", h=h, w=w)
            bf_mask = einops.rearrange(coord_mask,        "b f c (h w) -> b f c h w", h=h, w=w)

            # ----- DEBUG FROM HERE -----
            DEBUG_WARP = True
            if DEBUG_WARP:
                dbg_before_ccm = bf_ccm.clone()

            for j, src in enumerate(tar_idx_to_cond_idx):
                assert 0 <= src < cond_num, f"src={src} 超出 ref 範圍 [0,{cond_num})"

                if DEBUG_WARP:
                    with torch.no_grad():
                        mse_before = F.mse_loss(dbg_before_ccm[:, cond_num + j], dbg_before_ccm[:, src]).item()
                        print(f"[CCM ] map tar f={cond_num + j} <- ref f={src} | MSE_before={mse_before:.6f}")
                        #breakpoint()
                # 覆寫 CCM（Fourier PE）
                bf_ccm[:,  cond_num + j] = bf_ccm[:, src]
                # 視需求也可覆寫 mask（或設 1.0），預設保留原 target 的可見性：
                # bf_mask[:, cond_num + j] = bf_mask[:, src]
                # bf_mask[:, cond_num + j] = 1.0

                if DEBUG_WARP:
                    with torch.no_grad():
                        mse_after = F.mse_loss(bf_ccm[:, cond_num + j], bf_ccm[:, src]).item()
                        print(f"[CCM ] map tar f={cond_num + j} <- ref f={src} | MSE_after ={mse_after:.6f}")

                    # 也存一張第一個 batch 的單 channel 熱力圖對照（只示範第 0 通道）
                    if bf_ccm.shape[2] > 0 and dbg_before_ccm.shape[0] > 0 and j == 0:
                        import torchvision.utils as vutils, os
                        os.makedirs("debug_warp_ccm", exist_ok=True)
                        # 取 channel 0 做可視化（因為 CCM 是多通道 Fourier PE）
                        ref_vis = bf_ccm[0, src, 0:1]                    # [1,H,W]
                        tar_vis = bf_ccm[0, cond_num + j, 0:1]           # [1,H,W]
                        diff    = (tar_vis - ref_vis).abs()
                        grid = torch.cat([ref_vis, tar_vis, diff], dim=0)  # 3x1xHxW
                        vutils.save_image(grid, f"debug_warp_ccm/map_{cond_num+j}_from_{src}.png", nrow=3, normalize=True)

            # ----- DEBUG END -----
            
            points3d_prior_pe = einops.rearrange(bf_ccm,  "b f c h w -> b f c (h w)")
            coord_mask        = einops.rearrange(bf_mask, "b f c h w -> b f c (h w)")
        """
        
        
        # （原邏輯）可見性全域遮罩
        coord_global_mask = coord_mask.mean(dim=3, keepdim=True)
        coord_global_mask = (coord_global_mask != 0).float()
        points3d_prior_pe = points3d_prior_pe * coord_global_mask

        coords = torch.cat([points3d_prior_pe, coord_mask], dim=2)
        coords = einops.rearrange(coords, "b f c (h w) -> (b f) c h w", h=h, w=w)
        
        #breakpoint()
        return coords



# YiYi to-do: refactor rope related functions/classes
def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)

    batch_size, seq_length = pos.shape
    out = torch.einsum("...n,d->...nd", pos, omega)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)

    stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.float()


# YiYi to-do: refactor rope related functions/classes
class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


class StableDiffusionRoPE(nn.Module):
    def __init__(self, theta: int, rope_layers):
        super().__init__()
        self.theta = theta
        self.axes_dims = dict()
        for down_scale_factor in rope_layers:
            dim = rope_layers[down_scale_factor]['ch'] // rope_layers[down_scale_factor]['nhead']
            self.axes_dims[down_scale_factor] = [dim // 2, dim // 2]

    def forward(self, height, width, device) -> dict:
        result = dict()
        for down_scale_factor in self.axes_dims:
            h = height // down_scale_factor
            w = width // down_scale_factor
            latent_image_ids = torch.zeros(h, w, 2)
            latent_image_ids[..., 0] = latent_image_ids[..., 0] + torch.arange(h)[:, None]
            latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(w)[None, :]
            latent_image_ids = latent_image_ids.reshape(1, -1, 2).to(device)

            n_axes = latent_image_ids.shape[-1]
            emb = torch.cat(
                [rope(latent_image_ids[..., i], self.axes_dims[down_scale_factor][i], self.theta) for i in range(n_axes)],
                dim=-3,
            )
            emb = emb.unsqueeze(1)  # [1,1,hw,c//2,2,2]
            result[emb.shape[2]] = emb

        return result
