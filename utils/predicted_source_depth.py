from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F


SOURCE_GEOMETRY_TAG = "model_predicted_src_mvs_depth"


def _resize_depths(depths: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    batch_size, num_views = depths.shape[:2]
    if depths.shape[-2:] == size:
        return depths
    return F.interpolate(
        depths.reshape(batch_size * num_views, 1, *depths.shape[-2:]),
        size=size,
        mode="nearest",
    ).reshape(batch_size, num_views, *size)


def require_predicted_source_depths(
    output: Mapping[str, Any],
    size: Tuple[int, int],
    expected_views: Optional[int] = None,
) -> torch.Tensor:
    """Return per-source MVS predictions and reject legacy depth fallbacks."""
    depths = output.get("src_mvs_depths")
    if depths is None:
        raise RuntimeError(
            "Missing output['src_mvs_depths']. Source-view diffusion geometry must "
            "be predicted by the ADO/GBD MVS network; dataset depth and repeated "
            "target-depth fallbacks are forbidden."
        )
    if not isinstance(depths, torch.Tensor):
        raise TypeError(f"output['src_mvs_depths'] must be a tensor, got {type(depths)!r}")
    if depths.ndim == 5 and depths.shape[2] == 1:
        depths = depths[:, :, 0]
    if depths.ndim != 4:
        raise ValueError(
            "output['src_mvs_depths'] must have shape [B, V, H, W] or "
            f"[B, V, 1, H, W], got {tuple(depths.shape)}"
        )
    if expected_views is not None and depths.shape[1] != expected_views:
        raise ValueError(
            f"Predicted source-depth view count mismatch: expected {expected_views}, "
            f"got {depths.shape[1]}"
        )
    depths = _resize_depths(depths.float(), size)
    if not torch.isfinite(depths).all():
        raise ValueError("Predicted source MVS depth contains NaN or Inf values")
    if not torch.any(depths > 1e-6):
        raise ValueError("Predicted source MVS depth has no positive values")
    return depths.clamp_min(0.0)


def require_saved_predicted_source_depths(data: Mapping[str, Any]) -> torch.Tensor:
    """Load a saved condition while rejecting ambiguous legacy inputs."""
    geometry_source = data.get("source_geometry_source")
    if geometry_source != SOURCE_GEOMETRY_TAG:
        raise RuntimeError(
            f"Invalid source_geometry_source={geometry_source!r}; expected "
            f"{SOURCE_GEOMETRY_TAG!r}. Re-export this input with model-predicted "
            "source MVS depths."
        )
    depths = data.get("source_mvs_depths")
    if depths is None:
        raise RuntimeError(
            "Missing saved 'source_mvs_depths'. Legacy 'source_depths' alone is "
            "not accepted because its geometry source is ambiguous."
        )
    if not isinstance(depths, torch.Tensor):
        raise TypeError(f"source_mvs_depths must be a tensor, got {type(depths)!r}")
    if depths.ndim != 4 or depths.shape[1] != 1:
        raise ValueError(
            "Saved source_mvs_depths must have shape [V, 1, H, W], got "
            f"{tuple(depths.shape)}"
        )
    cond_num = int(data.get("cond_num", -1))
    if cond_num != depths.shape[0]:
        raise ValueError(
            f"cond_num/source_mvs_depths mismatch: cond_num={cond_num}, "
            f"depth views={depths.shape[0]}"
        )
    if not torch.isfinite(depths).all():
        raise ValueError("Saved source_mvs_depths contains NaN or Inf values")
    if not torch.any(depths > 1e-6):
        raise ValueError("Saved source_mvs_depths has no positive values")
    return depths
