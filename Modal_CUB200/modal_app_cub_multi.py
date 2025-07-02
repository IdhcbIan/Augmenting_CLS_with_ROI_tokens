import modal
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import timm
from torchvision import transforms
from PIL import Image
from einops import rearrange
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Build Modal Image including local Python source code
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "torchvision", "tqdm", "timm==0.9.12", "einops==0.7.0", "pillow")
    .add_local_file("buddy_pool.py", "/root/buddy_pool.py")
    .add_local_file("maxsim_loss.py", "/root/maxsim_loss.py")
    .add_local_file("modal_app_cub_trainable.py", "/root/modal_app_cub_trainable.py")
)

# Define Modal App with dataset volume
app = modal.App(
    "CUB run on Multi-GPUs",
    image=image,
    volumes={"/mnt/data": modal.Volume.from_name("cub-data")}
)

class TrainableMultiVectorEncoder(nn.Module):
    """TRAINABLE Multi-vector encoder - clean version without checkpointing."""
    
    def __init__(self):
        super().__init__()
        MODEL_NAME = "vit_base_patch14_reg4_dinov2.lvd142m"
        
        # Configuration matching original
        self.embed_dim = 768
        self.num_registers = 4
        #self.img_size = 518
        self.img_size = 224
        self.roi_side = 3
        
        # Create the model - trainable with correct image size
        self.backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, img_size=self.img_size)
        
        # Add a small projection layer to make it clearly trainable
        self.projection = nn.Linear(self.embed_dim, self.embed_dim)
        
    def _buddy_pool(self, cue, patches2d):
        """Original buddy pooling implementation."""
        B, H, W, d = patches2d.shape
        flat = rearrange(patches2d, "b h w d -> b (h w) d")
        sim = torch.matmul(cue.unsqueeze(1), flat.transpose(1, 2)).squeeze(1)
        idx = sim.argmax(dim=-1)
        h = idx // W
        w = idx % W
        r = self.roi_side // 2
        roi = []
        for b in range(B):
            hs = slice(max(0, h[b]-r), min(H, h[b]+r+1))
            ws = slice(max(0, w[b]-r), min(W, w[b]+r+1))
            roi.append(patches2d[b, hs, ws, :].mean(dim=(0, 1)))
        return torch.stack(roi)
        
    def forward(self, x):
        """Clean forward pass - no checkpointing."""
        # Forward through backbone
        tokens = self.backbone.forward_features(x)
        
        # Apply projection (trainable layer)
        tokens = self.projection(tokens)
        
        # Extract tokens following original structure
        cls_tok = tokens[:, 0:1, :]  # CLS token: (B, 1, D)
        regs_tok = tokens[:, 1:1 + self.num_registers, :]  # Register tokens: (B, 4, D)
        patch_tok = tokens[:, 1 + self.num_registers:, :]  # Patch tokens: (B, N, D)
        
        # Reshape patch tokens to spatial grid
        g = int(self.img_size // 14)  # ViT-B/14 grid size
        patches2d = rearrange(patch_tok, "b (h w) d -> b h w d", h=g, w=g)
        
        # Combine CLS and register tokens to form cues
        cues = torch.cat([cls_tok, regs_tok], dim=1)  # (B, 5, D)
        
        # Apply buddy pooling to get ROIs
        rois = torch.stack([self._buddy_pool(cues[:, i], patches2d)
                           for i in range(cues.size(1))], dim=1)
        
        # Combine cues and ROIs
        toks = torch.cat([cues, rois], dim=1)  # (B, 10, D)
        
        # Normalize
        return F.normalize(toks, dim=-1)


def colbert_score(X, Y):
    """ColBERT scoring function from the original implementation."""
    return torch.einsum("bnd,bmd->bnm", X, Y).max(dim=-1).values.sum(dim=-1)


class TripletColbertLoss(nn.Module):
    """Triplet loss using ColBERT scoring from the original implementation."""
    
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        pos_score = colbert_score(anchor, positive)
        neg_score = colbert_score(anchor, negative)
        loss = F.relu(neg_score - pos_score + self.margin)
        return loss.mean()


def _load_image(path):
    """Load a PIL image and preprocess it to tensor."""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),  # Training augmentation - match img_size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(path).convert("RGB")
    return preprocess(img)  # Returns tensor


@app.function(
    #gpu="A100-80GB:4",  # 4 A100-80GB GPUs
    gpu="A100-80GB:2",  # 4 A100-80GB GPUs
    timeout=3600  # 1 hour timeout
)
def main(
    cub_root: str = "/mnt/data/CUB_200_2011",
    steps: int = 200,
    batch_size: int = 256,
    report_interval: int = 1,
    eval_batch_size: int = 100,  # Base eval batch size (will be multiplied by num_gpus)
    lr: float = 1e-5  # Lower learning rate for fine-tuning
    #lr: float = 5e-6  # Lower learning rate for fine-tuning
    #lr: float = 3e-4  # Lower learning rate for fine-tuning
    #lr: float = 3e-5  # Lower learning rate for fine-tuning
):
    """
    Train CUB triplet model on Modal with multiple A100 GPUs.
    Simple FP32 training matching original aug_cls_repo approach.
    """
    # Setup multi-GPU environment
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    cub_root_path = Path(cub_root)

    def parse_cub(root: Path):
        cls_map = {}
        for line in (root / "classes.txt").read_text().splitlines():
            cid, cname = line.split()
            cls_map[int(cid)] = cname

        img_to_cid = {}
        for line in (root / "image_class_labels.txt").read_text().splitlines():
            iid, cid = line.split()
            img_to_cid[int(iid)] = int(cid)

        img_map = {}
        for line in (root / "images.txt").read_text().splitlines():
            iid, rel = line.split()
            img_map[int(iid)] = "/mnt/data/CUB_200_2011/images/" + rel

        train_ids = set()
        for line in (root / "train_test_split.txt").read_text().splitlines():
            iid, flag = line.split()
            if int(flag):
                train_ids.add(int(iid))

        train_paths = {c: [] for c in cls_map.values()}
        test_paths = {c: [] for c in cls_map.values()}
        for iid, path in img_map.items():
            cname = cls_map[img_to_cid[iid]]
            (train_paths[cname] if iid in train_ids else test_paths[cname]).append(path)

        train_paths = {c: ps for c, ps in train_paths.items() if len(ps) >= 2}
        test_paths = {c: ps for c, ps in test_paths.items() if c in train_paths and len(ps) >= 1}
        return train_paths, test_paths

    def load_cub_triplet(class_to_paths: dict[str, list[Path]]):
        """CORRECT triplet construction using independent sampling."""
        cls_pos = random.choice(list(class_to_paths.keys()))
        # Use random.choice twice to allow anchor==positive (matching original)
        a = random.choice(class_to_paths[cls_pos])
        p = random.choice(class_to_paths[cls_pos])
        neg_cls = random.choice([c for c in class_to_paths if c != cls_pos])
        n = random.choice(class_to_paths[neg_cls])
        return a, p, n

    def get_batch(class_to_paths: dict[str, list[Path]], batch_size: int):
        anchors, positives, negatives = [], [], []
        for _ in range(batch_size):
            a, p, n = load_cub_triplet(class_to_paths)
            for lst, img_path in [(anchors, a), (positives, p), (negatives, n)]:
                img = _load_image(img_path)
                lst.append(img)
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

    def evaluate_retrieval_recalls(train_paths, test_paths, model, device, ks, eval_batch_size, num_gpus):
        model.eval()
        
        # Calculate effective evaluation batch size for multi-GPU
        effective_eval_batch_size = eval_batch_size * num_gpus
        print(f"🔍 Evaluation using effective batch size: {effective_eval_batch_size} (base: {eval_batch_size} × {num_gpus} GPUs)")
        
        classes = sorted(train_paths.keys())
        cls2idx = {c: i for i, c in enumerate(classes)}
        gallery_embs, gallery_labels = [], []
        
        # Build gallery from training set with batched processing
        with torch.no_grad():
            gallery_items = [(c, p) for c in classes for p in train_paths[c][:5]]  # Limit to 5 per class for speed
            
            for i in trange(0, len(gallery_items), effective_eval_batch_size, desc="Building gallery", unit="batch"):
                batch = gallery_items[i:i+effective_eval_batch_size]
                imgs = []
                batch_labels = []
                
                for c, p in batch:
                    img = _load_image(p)  # Already returns tensor
                    imgs.append(img)
                    batch_labels.append(cls2idx[c])
                
                if imgs:  # Only process if we have images
                    imgs = torch.stack(imgs).to(device)
                    embs = model(imgs)
                    
                    # Extract CLS tokens and add to gallery
                    for j, emb in enumerate(embs):
                        gallery_embs.append(emb[0, :].cpu())  # Use CLS token
                        gallery_labels.append(batch_labels[j])
        
        gallery = torch.stack(gallery_embs, dim=0)
        gallery_norm = F.normalize(gallery, dim=1)

        test_items = [(cls2idx[c], p) for c, paths in test_paths.items() for p in paths[:10]]  # Limit for speed
        total = len(test_items)
        hits = {k: 0 for k in ks}

        # Process test queries with effective batch size
        for i in trange(0, total, effective_eval_batch_size, desc="Evaluating queries", unit="batch"):
            batch = test_items[i:i+effective_eval_batch_size]
            labels = torch.tensor([lbl for lbl, _ in batch])
            imgs = []
            for _, p in batch:
                img = _load_image(p)  # Already returns tensor
                imgs.append(img)
            imgs = torch.stack(imgs).to(device)

            with torch.no_grad():
                embs = model(imgs)
                
                embs_cls = embs[:, 0, :].cpu()  # Use CLS token
                embs_norm = F.normalize(embs_cls, dim=1)

            sims = embs_norm @ gallery_norm.t()
            topk = sims.topk(max(ks), dim=1).indices.cpu().tolist()

            for k in ks:
                for qi, row in enumerate(topk):
                    if any(gallery_labels[idx] == labels[qi].item() for idx in row[:k]):
                        hits[k] += 1

        for k in ks:
            print(f"Recall@{k}: {hits[k] / total:.4f} ({hits[k]}/{total})")

        model.train()

    # Initialize everything
    train_paths, test_paths = parse_cub(cub_root_path)
    
    # Setup multi-GPU with DistributedDataParallel
    num_gpus = torch.cuda.device_count()
    print(f"🚀 Using {num_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
    
    # Create clean model
    model = TrainableMultiVectorEncoder()
    
    if num_gpus > 1:
        # Initialize distributed training
        # For Modal's multi-GPU setup, DataParallel is actually more appropriate
        # than DDP since Modal gives us multiple GPUs in a single container/process
        device = torch.device("cuda:0")
        model = model.to(device)
        
        # Use DataParallel but with stability improvements
        model = nn.DataParallel(model)
        print(f"✅ Model wrapped with DataParallel across {num_gpus} GPUs")
        effective_batch_size = batch_size * num_gpus
    else:
        device = torch.device("cuda:0")
        model = model.to(device)
        effective_batch_size = batch_size
    
    print(f"Primary device: {device}")
    
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    print(f"🎯 Effective batch size: {effective_batch_size} (per_gpu: {batch_size}, gpus: {num_gpus})")
    
    # Setup optimizer and loss with stability improvements
    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Use a stable scheduler instead of CosineAnnealingLR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=20, min_lr=1e-7
    )
    
    criterion = TripletColbertLoss(margin=0.2)
    
    hist = []

    
    for i in trange(steps, desc="train", unit="step"):
        # Get batch
        a, p, n = get_batch(train_paths, batch_size)
        a, p, n = a.to(device), p.to(device), n.to(device)

        optimizer.zero_grad()

        emb_a = model(a)
        emb_p = model(p)
        emb_n = model(n)
        loss = criterion(emb_a, emb_p, emb_n)
        
        loss.backward()
        print(f"Current Loss: {loss.item()}")
        print(f"Current LR: {scheduler.get_last_lr()[0]}")
        
        # Add gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update scheduler every few steps, not every step
        if i % 10 == 0:
            pass
            #scheduler.step(loss.item())

        hist.append(loss.item())

        if (i + 1) % report_interval == 0:
            avg_loss = sum(hist[-report_interval:]) / report_interval
            print(f"[step {i+1:4d}] avg loss: {avg_loss:.4f} (effective_bs: {effective_batch_size})")

            """
            # Quick evaluation every few steps  
            if (i + 1) % (report_interval * 2) == 0:
                print("📊 Quick evaluation:")
                evaluate_retrieval_recalls(
                    train_paths, test_paths, model, device,
                    ks=[1, 4], eval_batch_size=eval_batch_size, num_gpus=num_gpus
                )
            """

    # Final comprehensive evaluation
    print("--------------------------------")
    print("🎯 Final evaluation:")
    print(f"Eval batch size: {eval_batch_size * num_gpus}")
    print(f"Final loss: {hist[-1]:.8f}")
    evaluate_retrieval_recalls(
        train_paths, test_paths, model, device,
        ks=[1, 2, 4], eval_batch_size=eval_batch_size, num_gpus=num_gpus
    )
    
    print(f"✅ Multi-GPU ({num_gpus} GPUs) FP32 Training complete!")
    return {"final_loss": hist[-1] if hist else 0.0, "avg_final_loss": sum(hist[-10:]) / 10 if len(hist) >= 10 else 0.0}


if __name__ == "__main__":
    # For local testing
    with app.run():
        result = main.remote()
        print("Result:", result) 
