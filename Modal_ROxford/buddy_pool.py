"""Create one Buddy (ROI) token per cue by 3×3 mean pooling
around each cue's nearest patch token."""
from __future__ import annotations

import torch
from torch import nn
import math
from einops import rearrange

# Configuration for ROI pooling
class CFG:
    roi_side = 3  # 3x3 ROI pooling like original

def _buddy_pool(cue, patches2d):
    """Original buddy pooling implementation matching aug_cls_repo"""
    B, H, W, d = patches2d.shape
    flat = rearrange(patches2d, "b h w d -> b (h w) d")
    sim  = torch.matmul(cue.unsqueeze(1), flat.transpose(1, 2)).squeeze(1)
    idx  = sim.argmax(dim=-1)
    h = idx // W
    w = idx %  W
    r = CFG.roi_side // 2
    roi = []
    for b in range(B):
        hs = slice(max(0, h[b]-r), min(H, h[b]+r+1))
        ws = slice(max(0, w[b]-r), min(W, w[b]+r+1))
        roi.append(patches2d[b, hs, ws, :].mean(dim=(0, 1)))
    return torch.stack(roi)


class BuddyPool(nn.Module):
    """Mean-pool 3×3 neighborhood of the nearest patch for each cue."""

    def forward(self, cue: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """
        cue:     (B, 5, D) - CLS + 4 register tokens
        patches: (B, H, W, D) - spatial patch tokens
        returns: (B, 5, D) – one Buddy per cue
        """
        b, k, d = cue.shape
        
        # Ensure patches are in spatial format (B, H, W, D)
        if len(patches.shape) == 3:
            # If flat (B, N, D), reshape to spatial
            _, n, _ = patches.shape
            # Calculate grid size (assuming square grid)
            h = w = int(math.sqrt(n))
            if h * w != n:
                raise ValueError(f"Cannot reshape {n} patches to square grid")
            patches = patches.view(b, h, w, d)
        
        # Apply buddy pooling for each cue token
        rois = torch.stack([_buddy_pool(cue[:, i], patches)
                           for i in range(k)], dim=1)
        
        return rois
