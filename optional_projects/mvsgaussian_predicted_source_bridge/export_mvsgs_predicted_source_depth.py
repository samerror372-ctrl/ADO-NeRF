import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import mvsgs_diffusion_export_utils as base
from lib.config import cfg, args
from lib.datasets import make_data_loader
from lib.networks import make_network
from lib.networks.mvsgs import utils as mvsgs_utils
from lib.utils import net_utils
from lib.utils.data_utils import to_cuda


def _depth_only_forward(network, batch):
    """Predict MVSGaussian cascade depths without running the Gaussian renderer."""
    feats = network.forward_feat(batch["src_inps"])
    ret = {}
    depth, std, near_far = None, None, None
    for level in range(cfg.mvsgs.cas_config.num):
        feature_volume, depth_values, near_far = mvsgs_utils.build_feature_volume(
            feats[f"level_{level}"],
            batch,
            D=cfg.mvsgs.cas_config.volume_planes[level],
            depth=depth,
            std=std,
            near_far=near_far,
            level=level,
        )
        _, depth_prob = getattr(network, f"cost_reg_{level}")(feature_volume)
        depth, std = mvsgs_utils.depth_regression(depth_prob, depth_values, level, batch)
        ret_depth = 1.0 / depth.clamp_min(1e-6) if cfg.mvsgs.cas_config.depth_inv[level] else depth
        ret[f"depth_mvs_level{level}"] = ret_depth
        ret[f"std_level{level}"] = std
    return ret


def _clone_batch_for_source_target(batch, src_idx):
    pred_batch = dict(batch)
    pred_batch["tar_ext"] = batch["src_exts"][:, src_idx].contiguous()
    pred_batch["tar_ixt"] = batch["src_ixts"][:, src_idx].contiguous()
    return pred_batch


def _predict_source_depths_no_raw(network, batch, level):
    """Predict source-view depths from MVSGaussian itself, without reading src_dpts."""
    src_depths = []
    src_stds = []
    num_src = batch["src_inps"].shape[1]
    for src_idx in range(num_src):
        pred_batch = _clone_batch_for_source_target(batch, src_idx)
        pred_output = _depth_only_forward(network, pred_batch)
        src_depths.append(pred_output[f"depth_mvs_level{level}"])
        if f"std_level{level}" in pred_output:
            src_stds.append(pred_output[f"std_level{level}"])
    ret = {
        f"src_mvs_depths_level{level}": torch.stack(src_depths, dim=1),
        "source_depth_origin": "mvsgs_predicted_source_view_no_raw_depth",
    }
    if src_stds:
        ret[f"src_mvs_stds_level{level}"] = torch.stack(src_stds, dim=1)
    return ret


def _strip_raw_source_geometry(batch):
    export_batch = dict(batch)
    export_batch.pop("src_dpts", None)
    export_batch.pop("src_msks", None)
    return export_batch


def run_evaluate():
    network = make_network(cfg).cuda()
    net_utils.load_network(network, cfg.trained_model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    level = base._final_render_level()
    out_dir = (
        os.environ.get("MVSGS_DIFFUSION_OUTPUT_DIR")
        or os.path.join(os.path.abspath(cfg.workspace), "diffusion_nerf", "mvsgs_predicted_source_inputs")
    )
    os.makedirs(out_dir, exist_ok=True)
    export_limit = int(
        os.environ.get("MVSGS_DIFFUSION_EXPORT_LIMIT")
        or "0"
    )

    net_time = []
    for batch_idx, batch in enumerate(tqdm.tqdm(data_loader)):
        if export_limit > 0 and batch_idx >= export_limit:
            break
        batch = to_cuda(batch)
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            output = network(batch)
            output.update(_predict_source_depths_no_raw(network, batch, level))
            torch.cuda.synchronize()
            net_time.append(time.time() - start_time)

        export_batch = _strip_raw_source_geometry(batch)
        base_name, mask_max, mask_mean = base.save_mvsgs_diffusion_data(
            export_batch, output, level, batch_idx, out_dir
        )
        saved_path = os.path.join(out_dir, f"{base_name}_diffusion_data.pt")
        diffusion_data = torch.load(saved_path, map_location="cpu")
        diffusion_data["source_depth_origin"] = output["source_depth_origin"]
        diffusion_data["uses_source_raw_depth"] = False
        torch.save(diffusion_data, saved_path)

        if batch_idx % 10 == 0:
            print(
                f"[MVSGS-PREDICTED-SOURCE] {base_name}: "
                f"mask max={mask_max:.4f}, mean={mask_mean:.4f}"
            )

    if net_time:
        usable = net_time[1:] if len(net_time) > 1 else net_time
        print("FPS including predicted source-depth passes: ", 1.0 / np.mean(usable))
    print(f"Saved predicted-source MVSGaussian diffusion inputs to: {out_dir}")


if __name__ == "__main__":
    globals()["run_" + args.type]()
