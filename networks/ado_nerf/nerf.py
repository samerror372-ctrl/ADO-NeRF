import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRF(nn.Module):
    def __init__(self, 
                 hid_dim: int = 64, 
                 feat_dim: int = 16, 
                 voxel_dim: int = 8, 
                 viewdir_agg: bool = True) -> None:
        """Generalizable NeRF (generalizable NeRF) based on agregated 2D, 3D features and directions.
        """
        super(NeRF, self).__init__()
        self.feat_dim = feat_dim
        self.viewdir_agg = viewdir_agg
        
        # View direction aggregation
        if viewdir_agg:
            self.view_fc = nn.Sequential(
                nn.Linear(4, feat_dim+3), 
                nn.ReLU(inplace=True)
            )

        self.global_fc = nn.Sequential(
            nn.Linear((feat_dim+3)*3, 32), 
            nn.ReLU(inplace=True)
        )
        self.agg_w_fc = nn.Sequential(
            nn.Linear(32, 1), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 16), 
            nn.ReLU(inplace=True)
        )

        # Color aggregation
        self.lr0 = nn.Sequential(
            nn.Linear(voxel_dim+16, hid_dim), 
            nn.ReLU(inplace=True)
        )
        self.sigma = nn.Sequential(
            nn.Linear(hid_dim, 1), 
            nn.Softplus()
        )
        self.weight = nn.Sequential(
            nn.Linear(hid_dim+voxel_dim+16+feat_dim+3+4, hid_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(hid_dim, 1), 
            nn.ReLU(inplace=True)
        )
        self.feat_head = nn.Sequential(
            nn.Linear(hid_dim, voxel_dim), 
            nn.ReLU(inplace=True)
        )
    
    def agg_viewdir(self, 
                    feat_rgb_dir: torch.Tensor,
                    vis_weights: torch.Tensor = None) -> torch.Tensor: 
        """Aggregate view direction features with geometry-aware occlusion handling."""
        num_views = feat_rgb_dir.shape[0]
        img_feat_rgb = feat_rgb_dir[..., :-4]

        # 视角方向特征融合
        if self.viewdir_agg:
            view_feat = self.view_fc(feat_rgb_dir[..., -4:])  # (num_views, num_points, feat_dim+3)
            img_feat_rgb = img_feat_rgb + view_feat  # (num_views, num_points, feat_dim+3)

        # =====================================================================
        # [核心重构] 动态加权均值与方差 (Weighted Variance & Mean)
        # =====================================================================
        if vis_weights is not None:
            # 1. 权重归一化分母 (加上 epsilon 防止全遮挡时除以 0)
            # vis_weights shape: (num_views, num_points, 1)
            weight_sum = vis_weights.sum(dim=0, keepdim=True) + 1e-6
            
            # 2. 计算加权平均 (Weighted Mean)
            # 只让没有被遮挡的视角的特征参与平均
            avg_feat = (img_feat_rgb * vis_weights).sum(dim=0, keepdim=True) / weight_sum
            
            # 3. 计算加权方差 (Weighted Variance)
            # 被遮挡的视角(权重近乎0)不会对偏差平方和产生贡献，从而消除了虚假的高方差
            var_feat = (vis_weights * (img_feat_rgb - avg_feat) ** 2).sum(dim=0, keepdim=True) / weight_sum
            
            # 4. 对特征本身也进行软掩码削弱，防止残留噪声送入后续网络
            img_feat_rgb = img_feat_rgb * vis_weights
            
        else:
            # 前 5% epoch 或者未开启遮挡时，回退到普通的统计方法
            var_feat, avg_feat = torch.var_mean(img_feat_rgb, dim=0, keepdim=True)
        # =====================================================================

        # 扩展维度以匹配拼接要求
        var_feat = var_feat.expand(num_views, -1, -1)
        avg_feat = avg_feat.expand(num_views, -1, -1)

        feat = torch.cat([img_feat_rgb, var_feat, avg_feat], dim=-1)  # (num_views, num_points, (feat_dim+3)*3)
        global_feat = self.global_fc(feat)  # (num_views, num_points, 32)
        
        # 运用 Log-Space 掩码处理隐式注意力
        logits = self.agg_w_fc(global_feat) # (num_views, num_points, 1)
        
        if vis_weights is not None:
            # 在注意力层进行最后的“物理封杀”
            logits = logits + torch.log(vis_weights.clamp(min=1e-6))
            
        agg_w = F.softmax(logits, dim=0)  # (num_views, num_points, 1)
        im_feat = torch.sum(global_feat * agg_w, dim=0)  # (num_points, 32)
        
        return self.fc(im_feat)

    def forward(self, 
                vox_feat: torch.Tensor, 
                rgbs_feat_rgb_dir: torch.Tensor, 
                only_geo: bool = False,
                vis_weights: torch.Tensor = None) -> torch.Tensor: 
        """Predict color/feat and sigma for each ray sample."""
        num_views = rgbs_feat_rgb_dir.shape[0]
        feat_rgb_dir = rgbs_feat_rgb_dir[..., -(self.feat_dim+3+4):]  # (num_views, num_points, feat_dim+3+4)
        
        # 传递 vis_weights 给几何特征聚合层
        img_feat = self.agg_viewdir(feat_rgb_dir, vis_weights)  # (num_points, 16)
        
        vox_img_feat = torch.cat((vox_feat, img_feat), dim=-1)  # (num_points, voxel_dim+16)
        x = self.lr0(vox_img_feat)  # (num_points, hid_dim)
        sigma = self.sigma(x)  # (num_points, 1)

        rgbs_feat_rgb = None
        if not only_geo:
            w_feat = torch.cat((x, vox_img_feat), dim=-1)  # (num_points, hid_dim+voxel_dim+16)
            w_feat = w_feat.unsqueeze(0).expand(num_views, -1, -1)  # (num_views, num_points, hid_dim+voxel_dim+16)
            w_feat = torch.cat((w_feat, feat_rgb_dir), dim=-1)  # (num_views, num_points, hid_dim+voxel_dim+16+feat_dim+3+4)
            
            # 对 RGB 的混合也同样运用 Log-Space 掩码
            logits_rgb = self.weight(w_feat)
            
            if vis_weights is not None:
                logits_rgb = logits_rgb + torch.log(vis_weights.clamp(min=1e-6))
                
            weight = F.softmax(logits_rgb, dim=0)  # (num_views, num_points, 1)
            
            rgbs_feat_rgb = torch.sum((rgbs_feat_rgb_dir[..., :-4] * weight), dim=0)  # (num_points, 3*up_scale**2+feat_dim+3)
            feat = torch.cat((rgbs_feat_rgb, 
                              self.feat_head(x)), 
                              dim=-1)  # (num_points, 3*up_scale**2+feat_dim+3+voxel_dim)

        return sigma.squeeze(-1), feat