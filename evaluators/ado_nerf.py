import os
import cv2
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import lpips

# === 补上丢失的这两个图像评价指标的导入 ===
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
# ==========================================
# 全局变量：用于存储整个训练过程的历史趋势
# ==========================================
EVAL_HISTORY = {
    'step': [],
    'psnr': [],
    'ssim': [],
    'lpips': []
}

class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.psnrs = []
        self.ssims = []
        self.lpips = []
        self.scene_psnrs = {}
        self.scene_ssims = {}
        self.scene_lpips = {}
        self.loss_fn_vgg = lpips.LPIPS(net='vgg')
        self.loss_fn_vgg.cuda()
        if self.cfg.test.eval_depth:
            # Following the setup of generalizable NeRF
            self.eval_depth_scenes = ['scan1', 'scan8', 'scan21', 'scan103', 'scan110']
            self.abs = []
            self.acc_2 = []
            self.acc_10 = []
            self.mvs_abs = []
            self.mvs_acc_2 = []
            self.mvs_acc_10 = []
        os.system('mkdir -p ' + self.cfg.result_dir)

    def evaluate(self, output, batch):
        B, _, _, H, W = batch['src_views']['rgb'].shape
        
        gt_rgb = batch['tar_views']['rgb'].detach().cpu().numpy()
        masks = (batch['tar_views']['mask'].cpu().numpy() >= 1).astype(np.uint8)
        pred_rgb = output['rgb'].permute(0, 2, 3, 1).detach().clamp(0., 1.).cpu().numpy()
        
        if self.cfg.test.eval_center:
            H_crop, W_crop = int(H*0.1), int(W*0.1)
            pred_rgb = pred_rgb[:, H_crop:-H_crop, W_crop:-W_crop]
            gt_rgb = gt_rgb[:, H_crop:-H_crop, W_crop:-W_crop]
            masks = masks[:, H_crop:-H_crop, W_crop:-W_crop]

        for b in range(B):
            if not batch['meta']['scene'][b] in self.scene_psnrs:
                self.scene_psnrs[batch['meta']['scene'][b]] = []
                self.scene_ssims[batch['meta']['scene'][b]] = []
                self.scene_lpips[batch['meta']['scene'][b]] = []
            if self.cfg.save_result:
                img_path = os.path.join(self.cfg.result_dir, '{}_{}_{}.png'.format(batch['meta']['scene'][b], batch['meta']['tar_view'][b].item(), batch['meta']['frame_id'][b].item()))
                img = (cv2.cvtColor(pred_rgb[b], cv2.COLOR_RGB2BGR) * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(img_path, img)

            mask = masks[b] == 1
            gt_rgb[b][mask==False] = 0.
            pred_rgb[b][mask==False] = 0.

            psnr_item = psnr(gt_rgb[b][mask], pred_rgb[b][mask], data_range=1.)
            self.psnrs.append(psnr_item)
            self.scene_psnrs[batch['meta']['scene'][b]].append(psnr_item)

            # 包含之前修复的 data_range=1.0
            ssim_item = ssim(gt_rgb[b], pred_rgb[b], channel_axis=-1, data_range=1.0)
            self.ssims.append(ssim_item)
            self.scene_ssims[batch['meta']['scene'][b]].append(ssim_item)

            if self.cfg.eval_lpips:
                gt, pred = torch.Tensor(gt_rgb[b])[None].permute(0, 3, 1, 2), torch.Tensor(pred_rgb[b])[None].permute(0, 3, 1, 2)
                gt, pred = (gt-0.5)*2., (pred-0.5)*2.
                lpips_item = self.loss_fn_vgg(gt.cuda(), pred.cuda()).item()
                self.lpips.append(lpips_item)
                self.scene_lpips[batch['meta']['scene'][b]].append(lpips_item)

            if self.cfg.test.eval_depth and batch['meta']['scene'][b] in self.eval_depth_scenes:
                nerf_depth = output['nerf_depth'].cpu().numpy()[b]
                nerf_gt_depth = batch['tar_views']['depth'].cpu().numpy()[b]
                mvs_depth = output['mvs_depth'].cpu().numpy()[b]
                mvs_gt_depth = batch['tar_gt_ms']['depth'][-1][b].cpu().numpy()
                nerf_depth = cv2.resize(nerf_depth, nerf_gt_depth.shape[-1:-3:-1] , interpolation=cv2.INTER_LINEAR)
                
                nerf_mask = nerf_gt_depth != 0.
                mvs_mask = mvs_gt_depth != 0.
                self.abs.append(np.abs(nerf_depth[nerf_mask] - nerf_gt_depth[nerf_mask]).mean())
                self.acc_2.append((np.abs(nerf_depth[nerf_mask] - nerf_gt_depth[nerf_mask]) < 2).mean())
                self.acc_10.append((np.abs(nerf_depth[nerf_mask] - nerf_gt_depth[nerf_mask]) < 10).mean())
                self.mvs_abs.append((np.abs(mvs_depth[mvs_mask] - mvs_gt_depth[mvs_mask])).mean())
                self.mvs_acc_2.append((np.abs(mvs_depth[mvs_mask] - mvs_gt_depth[mvs_mask]) < 2.).mean())
                self.mvs_acc_10.append((np.abs(mvs_depth[mvs_mask] - mvs_gt_depth[mvs_mask]) < 10.).mean())

    def summarize(self):
            ret = {}
            # ===== 修改这里：使用 float() 将 Numpy 数据转换为原生 Python 浮点数 =====
            cur_psnr = float(np.mean(self.psnrs))
            cur_ssim = float(np.mean(self.ssims))
            
            ret.update({'psnr': cur_psnr})
            ret.update({'ssim': cur_ssim})
            
            cur_lpips = 0.0
            if self.cfg.eval_lpips:
                # ===== 修改这里：同样加上 float() =====
                cur_lpips = float(np.mean(self.lpips))
                ret.update({'lpips': cur_lpips})

            # ==========================================
            # 记录当前 Eval 的数值到全局历史中
            # ==========================================
            eval_count = len(EVAL_HISTORY['step']) + 1
            EVAL_HISTORY['step'].append(eval_count)
            EVAL_HISTORY['psnr'].append(cur_psnr)
            EVAL_HISTORY['ssim'].append(cur_ssim)
            if self.cfg.eval_lpips:
                EVAL_HISTORY['lpips'].append(cur_lpips)

            # ==========================================
            # 同步执行：绘制趋势图 & 保存数据字典
            # ==========================================
            self.plot_and_save_trends()

            print('='*30)
            for scene in self.scene_psnrs:
                if self.cfg.eval_lpips:
                    print(scene.ljust(16), 'psnr: {:.2f} ssim: {:.3f} lpips:{:.3f}'.format(np.mean(self.scene_psnrs[scene]), np.mean(self.scene_ssims[scene]), np.mean(self.scene_lpips[scene])))
                else:
                    print(scene.ljust(16), 'psnr: {:.2f} ssim: {:.3f} '.format(np.mean(self.scene_psnrs[scene]), np.mean(self.scene_ssims[scene])))
            print('='*30)
            print(ret)
            
            if self.cfg.test.eval_depth:
                depth_ret = {}
                keys = ['abs', 'acc_2', 'acc_10']
                for key in keys:
                    depth_ret[key] = np.mean(getattr(self, key))
                    setattr(self, key, [])
                print(depth_ret)
                keys = ['mvs_abs', 'mvs_acc_2', 'mvs_acc_10']
                depth_ret = {}
                for key in keys:
                    depth_ret[key] = np.mean(getattr(self, key))
                    setattr(self, key, [])
                print(depth_ret)
                
            self.psnrs = []
            self.ssims = []
            self.lpips = []
            self.scene_psnrs = {}
            self.scene_ssims = {}
            self.scene_lpips = {}
            if self.cfg.save_result:
                print('Save visualization results to: {}'.format(self.cfg.result_dir))
            return ret

    # ==========================================
    # 核心方法：绘制折线图并以 JSON 格式保存历史记录
    # ==========================================
    def plot_and_save_trends(self):
        # 1. 保存图片
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(EVAL_HISTORY['step'], EVAL_HISTORY['psnr'], marker='o', color='r', label='PSNR')
        plt.title('Evaluation PSNR Trend')
        plt.xlabel('Eval Count')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(EVAL_HISTORY['step'], EVAL_HISTORY['ssim'], marker='s', color='b', label='SSIM')
        plt.title('Evaluation SSIM Trend')
        plt.xlabel('Eval Count')
        plt.ylabel('SSIM')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plot_path = os.path.join(self.cfg.result_dir, 'eval_trends.png')
        plt.savefig(plot_path)
        plt.close() # 必须关闭画布
        
        # 2. 将数据记录保存为 JSON 文件 (indent=4 保证文件具有良好的可读性)
        json_path = os.path.join(self.cfg.result_dir, 'eval_history.json')
        with open(json_path, 'w') as f:
            json.dump(EVAL_HISTORY, f, indent=4)