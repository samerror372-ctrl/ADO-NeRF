import torch.nn.functional as F


def texture(input, uv, mip_level_bias=None, boundary_mode="clamp", max_mip_level=None):
    """Fallback implementation of nvdiffrast.torch.texture via grid_sample.

    ADO-NeRF only uses texture() to bilinearly sample feature maps. The real
    nvdiffrast op supports mip-level bias; this compatibility path ignores the
    bias but preserves shapes and clamp/border sampling semantics.
    """
    if input.ndim != 4:
        raise ValueError(f"Expected input [N,H,W,C], got {tuple(input.shape)}")
    if uv.ndim != 4 or uv.shape[-1] != 2:
        raise ValueError(f"Expected uv [N,Hout,Wout,2], got {tuple(uv.shape)}")

    grid = uv.to(input.dtype) * 2.0 - 1.0
    image = input.permute(0, 3, 1, 2).contiguous()
    padding_mode = "border" if boundary_mode == "clamp" else "zeros"
    sampled = F.grid_sample(image, grid, mode="bilinear", padding_mode=padding_mode, align_corners=False)
    return sampled.permute(0, 2, 3, 1).contiguous()
