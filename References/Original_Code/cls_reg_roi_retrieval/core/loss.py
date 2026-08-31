import torch
from cls_reg_roi_retrieval.core.multivector import colbert_score
from cls_reg_roi_retrieval.config import CFG

class TripletColbertLoss(torch.nn.Module):
    def __init__(self, margin=CFG.margin):
        super().__init__()
        self.margin = margin

    def forward(self, q, p, n):
        loss = torch.relu(self.margin + colbert_score(q, n) - colbert_score(q, p))
        return loss.mean()
