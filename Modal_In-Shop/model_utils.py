"""Load DINO-v2 register model and extract cue + patch tokens.

Cue tokens  = [CLS] + 4 register tokens  -> shape (B, 5, D)
Patch tokens = remaining tokens arranged into an (H, W) grid.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
from einops import rearrange

# -----------------------------------------------------------------------------
# Model & preprocessing
# -----------------------------------------------------------------------------
MODEL_NAME = "vit_base_patch14_reg4_dinov2.lvd142m"  # Using DINOv2-reg model
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration matching original
class CFG:
    model_name = MODEL_NAME
    embed_dim = 768
    num_registers = 4
    img_size = 518
    device = _device
    roi_side = 3

# Create the register model using timm like the original
_model = timm.create_model(CFG.model_name, pretrained=True, num_classes=0)
_model.train().to(_device)
_embed_dim = _model.embed_dim

# Default DINO-v2 preprocessing with random crop for training
_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(CFG.img_size),  # Use random crop like original
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -----------------------------------------------------------------------------

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


class MultiVectorEncoder(nn.Module):
    """Multi-vector encoder matching the original implementation."""
    
    def __init__(self):
        super().__init__()
        # The backbone is already created globally as _model
        
    @torch.no_grad()
    def forward(self, x):
        """Forward pass with no gradients like the original."""
        # Forward through backbone
        tokens = _model.forward_features(x.to(_device))
        
        # Extract tokens following original structure
        cls_tok = tokens[:, 0:1, :]  # CLS token: (B, 1, D)
        regs_tok = tokens[:, 1:1 + CFG.num_registers, :]  # Register tokens: (B, 4, D)
        patch_tok = tokens[:, 1 + CFG.num_registers:, :]  # Patch tokens: (B, N, D)
        
        # Reshape patch tokens to spatial grid
        g = int(CFG.img_size // 14)  # ViT-B/14 grid size
        patches2d = rearrange(patch_tok, "b (h w) d -> b h w d", h=g, w=g)
        
        # Combine CLS and register tokens to form cues
        cues = torch.cat([cls_tok, regs_tok], dim=1)  # (B, 5, D)
        
        # Apply buddy pooling to get ROIs
        rois = torch.stack([_buddy_pool(cues[:, i], patches2d)
                           for i in range(cues.size(1))], dim=1)
        
        # Combine cues and ROIs
        toks = torch.cat([cues, rois], dim=1)  # (B, 10, D)
        
        # Normalize like the original
        return torch.nn.functional.normalize(toks, dim=-1)


def _load_image(path: Path | str) -> torch.Tensor:
    """Load a PIL image → (1,3,518,518) tensor ready for the model."""
    img = Image.open(path).convert("RGB")
    return _preprocess(img)[None]


@torch.no_grad()
def forward_tokens(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return
        cue_tokens   – (B, 5, D) [CLS + 4 register tokens]
        patch_tokens – (B, H, W, D) spatial grid of patch tokens
    """
    # Forward pass through the model - no gradients like original
    tokens = _model.forward_features(img.to(_device))
    
    # Extract tokens following original structure
    cls_tok = tokens[:, 0:1, :]  # CLS token: (B, 1, D)
    regs_tok = tokens[:, 1:1 + CFG.num_registers, :]  # Register tokens: (B, 4, D)
    patch_tok = tokens[:, 1 + CFG.num_registers:, :]  # Patch tokens: (B, N, D)
    
    # Combine CLS and register tokens to form cue tokens
    cue_tokens = torch.cat([cls_tok, regs_tok], dim=1)  # (B, 5, D)
    
    # Reshape patch tokens to spatial grid
    g = int(CFG.img_size // 14)  # ViT-B/14 grid size
    patch_tokens = rearrange(patch_tok, "b (h w) d -> b h w d", h=g, w=g)
    
    return cue_tokens, patch_tokens


if __name__ == "__main__":
    import sys, pprint
    img = _load_image(sys.argv[1]) if len(sys.argv) > 1 else torch.randn(1,3,CFG.img_size,CFG.img_size)
    cues, patches = forward_tokens(img)
    pprint.pprint({"cue": cues.shape, "patch": patches.shape})
