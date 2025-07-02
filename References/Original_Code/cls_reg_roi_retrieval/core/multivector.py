import torch
from cls_reg_roi_retrieval.core.backbone import DINOv2RegBackbone
from cls_reg_roi_retrieval.config import CFG
from einops import rearrange

def _buddy_pool(cue, patches2d):
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

class MultiVectorEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DINOv2RegBackbone()

    @torch.no_grad()
    def forward(self, x):
        cls, regs, patches = self.backbone(x)
        patches2d = self.backbone.patch_grid(patches)
        cues = torch.cat([cls.unsqueeze(1), regs], dim=1)
        rois = torch.stack([_buddy_pool(cues[:, i], patches2d)
                            for i in range(cues.size(1))], dim=1)
        toks = torch.cat([cues, rois], dim=1)
        return torch.nn.functional.normalize(toks, dim=-1)


def colbert_score(q, d):
    """
    q: Tensor of shape (B, Q, D)
    d: Tensor of shape (B, K, D)
    returns: a single scalar equal to the sum over all batch-elements of
             sum_{i=1..Q} max_{j=1..K} (q[b,i] · d[b,j])
    """
    # Compute, for each batch element b, the Q×K similarity matrix:
    #   sim[b] = q[b] @ d[b].T
    sim = torch.bmm(q, d.transpose(1, 2))   # shape = (B, Q, K)

    # For each (b, i), take the maximum over j ∈ [1..K]:
    #   max_sim[b,i] = max_j sim[b,i,j]
    max_sim_per_token = sim.max(dim=2).values  # shape = (B, Q)

    # Sum over the Q dimension to get one score per batch element:
    scores_per_example = max_sim_per_token.sum(dim=1)  # shape = (B,)

    # Finally, sum over the batch if you want a single scalar:
    return scores_per_example.sum()

