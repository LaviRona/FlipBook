"""Optical flow utilities: RAFT, backward warping, forward-backward occlusion, point warping.

Conventions
  * images: float tensors in [0, 1], shape (B, 3, H, W)
  * flow:   (B, 2, H, W), channel 0 = dx, channel 1 = dy, in pixels
  * flow_fwd[t] maps pixels of frame t to frame t+1;  flow_bwd[t] maps frame t+1 to frame t
  * points: (N, 2) pixel coordinates (x, y), the same convention as pydiffvg / CLIPasso
"""
import torch
import torch.nn.functional as F


def load_raft(device="cuda"):
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    w = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=w, progress=False).to(device).eval()
    return model, w.transforms()


@torch.no_grad()
def raft_flow(model, transforms, img1, img2):
    """Flow from img1 to img2 (both (B,3,H,W) in [0,1], H and W divisible by 8)."""
    a, b = transforms(img1, img2)
    return model(a, b)[-1]  # last refinement iteration


def _grid(flow):
    """Sampling grid for grid_sample: location x + flow(x), normalized to [-1, 1]."""
    B, _, H, W = flow.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=flow.device, dtype=flow.dtype),
                            torch.arange(W, device=flow.device, dtype=flow.dtype), indexing="ij")
    x = xs[None] + flow[:, 0]
    y = ys[None] + flow[:, 1]
    return torch.stack([2 * x / (W - 1) - 1, 2 * y / (H - 1) - 1], dim=-1), x, y


def backward_warp(img, flow, padding_mode="border"):
    """out(x) = img(x + flow(x)), bilinear.

    To predict frame t+1 from frame t, pass img = frame t and flow = flow_bwd[t] (t+1 -> t).
    """
    grid, _, _ = _grid(flow)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)


def fb_valid(flow_a, flow_b, alpha=0.01, beta=0.5):
    """Forward-backward consistency in the coordinates of flow_a's source frame.

    flow_a: A -> B at pixels of A;  flow_b: B -> A at pixels of B.
    A pixel x of A is valid if x + flow_a(x) stays in the image and
        |flow_a(x) + flow_b(x + flow_a(x))|^2 <= alpha * (|flow_a|^2 + |flow_b(..)|^2) + beta
    (Sundaram et al. 2010, as used by Ruder et al. 2016). Returns bool (B, H, W).
    """
    B, _, H, W = flow_a.shape
    grid, x, y = _grid(flow_a)
    fb = F.grid_sample(flow_b, grid, mode="bilinear", padding_mode="border", align_corners=True)
    s = flow_a + fb
    lhs = (s ** 2).sum(1)
    rhs = alpha * ((flow_a ** 2).sum(1) + (fb ** 2).sum(1)) + beta
    inside = (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
    return (lhs <= rhs) & inside


def sample_at_points(field, points):
    """Bilinear sample a (C,H,W) field at (N,2) pixel points (x, y) -> (N, C)."""
    C, H, W = field.shape
    gx = 2 * points[:, 0] / (W - 1) - 1
    gy = 2 * points[:, 1] / (H - 1) - 1
    grid = torch.stack([gx, gy], -1).view(1, 1, -1, 2).to(field.dtype)
    out = F.grid_sample(field[None], grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out[0, :, 0].T


def warp_points(points, flow_fwd_t):
    """Move points of frame t to frame t+1: p + F_{t->t+1}(p). flow_fwd_t: (2,H,W)."""
    return points + sample_at_points(flow_fwd_t, points)
