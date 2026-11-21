"""Independent optical flow for EVALUATION only: GMFlow (UniMatch, scale-2, 6 refinement steps).

The methods move strokes with RAFT (src/flow.py). Measuring flicker with the same RAFT flow would reward
methods for following RAFT, so the main flicker metric warps with this different model instead.
No method imports this module.

Setup (once):
  git clone https://github.com/autonomousvision/unimatch.git third_party/unimatch   (commit 943a990)
  weights: third_party/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth
           from https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
UNIMATCH = REPO / "third_party" / "unimatch"
WEIGHTS = UNIMATCH / "pretrained" / "gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth"
NAME = "GMFlow (UniMatch) scale2-regrefine6, mixdata weights"

# Inference settings of the official demo for this checkpoint (scripts/gmflow_demo.sh).
_CFG = dict(attn_type="swin", attn_splits_list=[2, 8], corr_radius_list=[-1, 4], prop_radius_list=[-1, 1],
            num_reg_refine=6)


def load_gmflow(device="cuda"):
    sys.path.insert(0, str(UNIMATCH))
    from unimatch.unimatch import UniMatch
    model = UniMatch(feature_channels=128, num_scales=2, upsample_factor=4, num_head=1, ffn_dim_expansion=4,
                     num_transformer_layers=6, reg_refine=True, task="flow").to(device).eval()
    ck = torch.load(WEIGHTS, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    return model


@torch.no_grad()
def gmflow_flow(model, img1, img2, padding_factor=32):
    """Flow from img1 to img2. Images (B,3,H,W) in [0,1]; returns (B,2,H,W) in pixels of the input size."""
    H, W = img1.shape[-2:]
    h = -(-H // padding_factor) * padding_factor
    w = -(-W // padding_factor) * padding_factor
    a, b = img1 * 255.0, img2 * 255.0
    if (h, w) != (H, W):
        a = F.interpolate(a, size=(h, w), mode="bilinear", align_corners=True)
        b = F.interpolate(b, size=(h, w), mode="bilinear", align_corners=True)
    flow = model(a, b, task="flow", **_CFG)["flow_preds"][-1]
    if (h, w) != (H, W):
        flow = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=True)
        flow[:, 0] *= W / w
        flow[:, 1] *= H / h
    return flow
