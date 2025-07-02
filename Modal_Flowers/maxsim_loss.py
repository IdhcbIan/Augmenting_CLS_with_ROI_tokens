"""ColBERT-style triplet loss with colbert scoring like the original implementation."""
from __future__ import annotations

import torch
from torch import nn


def colbert_score(q, d):
    """
    ColBERT scoring function from the original implementation.
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


class TripletColbertLoss(nn.Module):
    """Original triplet loss using ColBERT scoring."""
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, q, p, n):
        """
        q: query/anchor embeddings (B, T, D)
        p: positive embeddings (B, T, D)  
        n: negative embeddings (B, T, D)
        """
        pos_score = colbert_score(q, p)
        neg_score = colbert_score(q, n)
        loss = torch.relu(self.margin + neg_score - pos_score)
        return loss.mean()


# Alias for backward compatibility
MaxSimLoss = TripletColbertLoss
