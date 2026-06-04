import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from .modules import conv_block2d

class FeatureNet(nn.Module):
    """ Feature pyramid network for feature extraction from 2D images
    集成了深度可分离自适应上采样原理 (Depthwise Separable Upsampling)
    """

    def __init__(self, 
                 base_channels: int = 8, 
                 out_channels: Tuple[int, int, int] = (32, 16, 8)) -> None:
        super(FeatureNet, self).__init__()

        # ================= 1. 自底向上的特征编码 =================
        self.conv0 = nn.Sequential(
            conv_block2d(3, base_channels, kernel_size=3, padding=1),
            conv_block2d(base_channels, base_channels, kernel_size=3, padding=1)
        )

        self.conv1 = nn.Sequential(
            conv_block2d(base_channels, base_channels * 2, kernel_size=5, stride=2, padding=2),
            conv_block2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1)
        )

        self.conv2 = nn.Sequential(
            conv_block2d(base_channels * 2, base_channels * 4, kernel_size=5, stride=2, padding=2),
            conv_block2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1)
        )

        final_chs = base_channels * 4

        # ================= 2. 内联的上采样原理模块 =================
        # 我们定义一个内部的辅助函数来生成可学习的上采样块
        # 原理: Depthwise Transpose Conv (空间放大) + 1x1 Conv (通道融合)
        def make_adaptive_up_block(channels):
            return nn.Sequential(
                # 1. 深度转置卷积：独立放大每个通道的空间分辨率，学习平滑插值
                nn.ConvTranspose2d(
                    in_channels=channels, out_channels=channels,
                    kernel_size=4, stride=2, padding=1,
                    groups=channels, bias=False
                ),
                nn.BatchNorm2d(channels),
                nn.LeakyReLU(0.2, inplace=True),
                
                # 2. 逐点卷积：跨通道的信息融合与特征投影
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.LeakyReLU(0.2, inplace=True),
                
                # 3. 输出平滑：进一步稳固特征
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.LeakyReLU(0.2, inplace=True)
            )

        # 实例化自顶向下的上采样层
        self.up_c2_to_c1 = make_adaptive_up_block(final_chs)
        self.up_c1_to_c0 = make_adaptive_up_block(final_chs)

        # 匹配通道数的 1x1 卷积 (用于横向连接)
        self.inner1 = nn.Conv2d(base_channels * 2, final_chs, kernel_size=1)
        self.inner2 = nn.Conv2d(base_channels, final_chs, kernel_size=1)

        # ================= 3. 多尺度输出头 =================
        self.out0 = nn.Conv2d(final_chs, out_channels[0], kernel_size=1)
        self.out1 = nn.Conv2d(final_chs, out_channels[1], kernel_size=3, padding=1, bias=False)
        self.out2 = nn.Conv2d(final_chs, out_channels[2], kernel_size=3, padding=1, bias=False)


    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """ Extract two-stage pyramid features from 2D images. """
        
        # --- 编码阶段 (Bottom-up) ---
        c0 = self.conv0(x)   # (B, base_channels, H, W)
        c1 = self.conv1(c0)  # (B, base_channels * 2, H/2, W/2)
        c2 = self.conv2(c1)  # (B, base_channels * 4, H/4, W/4)
        
        feats = []
        
        # --- Level 0 (最粗糙特征: 1/4) ---
        intra_feat = c2
        feats.append(self.out0(intra_feat))

        # --- Level 1 (中间特征: 1/2) ---
        # 1. 对 c2 运用深度可分离转置卷积上采样
        # 2. 与 c1 的降维特征进行特征相加 (横向连接)
        intra_feat = self.up_c2_to_c1(intra_feat) + self.inner1(c1)
        feats.append(self.out1(intra_feat))

        # --- Level 2 (最精细特征: 原图尺寸) ---
        # 再次进行可学习的上采样，并与 c0 融合
        intra_feat = self.up_c1_to_c0(intra_feat) + self.inner2(c0)
        feats.append(self.out2(intra_feat))

        return feats