import torch
from torch.utils.data import DataLoader
from cls_reg_roi_retrieval.data.dataset import TripletFolder
from cls_reg_roi_retrieval.config import CFG as DEFAULT_CFG
from cls_reg_roi_retrieval.core.multivector import MultiVectorEncoder
from cls_reg_roi_retrieval.core.loss import TripletColbertLoss

def main(cfg=DEFAULT_CFG, extra_epochs=1):
    # 1) Don’t put the encoder into eval() mode here if you want to train it.
    enc = MultiVectorEncoder().to(cfg.device)  # ← no .eval()
    enc.train()  # ← ensure it’s in training mode (this enables gradients)

    optim = torch.optim.AdamW(enc.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay)
    crit = TripletColbertLoss(cfg.margin).to(cfg.device)

    ds = TripletFolder(cfg.train_root)
    loader = DataLoader(ds, batch_size=cfg.batch_size, num_workers=8)
    max_steps = cfg.steps * extra_epochs

    for step, (q, p, n) in enumerate(loader):
        if step >= max_steps:
            break

        q, p, n = [t.to(cfg.device) for t in (q, p, n)]
        # Now enc(q), enc(p), enc(n) will require_grad=True
        loss = crit(enc(q), enc(p), enc(n))

        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % 10 == 0:
            print(f"[{step:06d}] loss={loss.item():.4f}")

