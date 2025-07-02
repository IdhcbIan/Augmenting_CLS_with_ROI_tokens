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

# Build Modal Image including local Python source code
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "torchvision", "tqdm", "timm==0.9.12", "einops==0.7.0", "pillow")
    .add_local_file("buddy_pool.py", "/root/buddy_pool.py")
    .add_local_file("maxsim_loss.py", "/root/maxsim_loss.py")
    .add_local_file("modal_app_roxford_duo.py", "/root/modal_app_roxford_duo.py")
)

# Define Modal App with dataset volume
app = modal.App(
    "ROxford run on Multi-GPUs",
    image=image,
    volumes={"/mnt/data": modal.Volume.from_name("cub-data")}
)

class TrainableMultiVectorEncoder(nn.Module):
    """TRAINABLE Multi-vector encoder - same architecture but allows gradients."""
    
    def __init__(self):
        super().__init__()
        MODEL_NAME = "vit_base_patch14_reg4_dinov2.lvd142m"
        
        # Configuration matching original
        self.embed_dim = 768
        self.num_registers = 4
        self.img_size = 518
        self.roi_side = 3
        
        # Create the model - but make it trainable by not using @torch.no_grad()
        self.backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
        
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
        """Forward pass WITH gradients - this is trainable!"""
        # Forward through backbone (with gradients)
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
    """Load a PIL image and preprocess it."""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(518),  # Training augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(path).convert("RGB")
    return preprocess(img)


@app.function(
    gpu="A100-80GB:4",  # 2 A100-80GB GPUs
    timeout=3600  # 1 hour timeout
)
def main(
    roxford_root: str = "/mnt/data/roxford5k_converted",
    steps: int = 500,
    batch_size: int = 16,  # Per GPU
    report_interval: int = 50,
    eval_batch_size: int = 20,  # Larger eval batch thanks to 2 GPUs
    lr: float = 1e-5,  # Keep same as working version
    # OR even lower: lr: float = 8e-6  # Scale by sqrt(batch_ratio) = sqrt(32/8) = 2
):
    """
    Train ROxford triplet model on Modal with 2 A100 GPUs.
    """
    roxford_root = Path(roxford_root)

    def parse_roxford(root: Path):
        """
        Parse ROxford5k converted metadata and return training/test image path dictionaries
        """
        cls_map = {}
        for line in (root / "classes.txt").read_text().splitlines():
            cid, cname = line.split()
            cls_map[int(cid)] = cname

        # Load building names for reference
        building_map = {}
        for line in (root / "building_names.txt").read_text().splitlines():
            cid, building_name = line.split()
            building_map[int(cid)] = building_name

        img_to_cid = {}
        for line in (root / "image_class_labels.txt").read_text().splitlines():
            iid, cid = line.split()
            img_to_cid[int(iid)] = int(cid)

        img_map = {}
        for line in (root / "images.txt").read_text().splitlines():
            iid, rel = line.split()
            img_map[int(iid)] = "/mnt/data/" + rel

        train_ids = set()
        for line in (root / "train_test_split.txt").read_text().splitlines():
            iid, flag = line.split()
            if int(flag):
                train_ids.add(int(iid))

        # Use building names as class keys for better readability
        train_paths = {building_map[int(c)]: [] for c in cls_map.keys()}
        test_paths  = {building_map[int(c)]: [] for c in cls_map.keys()}
        
        for iid, path in img_map.items():
            cid = img_to_cid[iid]
            building_name = building_map[cid]
            (train_paths[building_name] if iid in train_ids else test_paths[building_name]).append(path)

        train_paths = {c: ps for c, ps in train_paths.items() if len(ps) >= 2}
        test_paths  = {c: ps for c, ps in test_paths.items() if c in train_paths and len(ps) >= 1}
        
        return train_paths, test_paths

    def load_roxford_triplet(class_to_paths: dict[str, list[Path]]):
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
            a, p, n = load_roxford_triplet(class_to_paths)
            for lst, img_path in [(anchors, a), (positives, p), (negatives, n)]:
                img = _load_image(img_path)
                lst.append(img)
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

    def compute_map(ranks, gnd, ks):
        """
        Compute mAP and mP@k for given ranks and ground truth
        """
        import numpy as np
        
        map_score = 0.0
        recall_at_k = np.zeros(len(ks))
        
        for i in range(len(gnd)):
            # Ground truth for this query
            ok = gnd[i]['ok']
            junk = gnd[i]['junk']
            
            # Exclude junk images from ranking
            ranks_clean = []
            for rank in ranks[:, i]:
                if rank not in junk:
                    ranks_clean.append(rank)
            
            # Compute average precision
            if len(ok) > 0:
                old_recall = 0.0
                old_precision = 1.0
                ap = 0.0
                intersect_size = 0.0
                
                for j in range(len(ranks_clean)):
                    if ranks_clean[j] in ok:
                        intersect_size += 1.0
                    
                    recall = intersect_size / len(ok)
                    precision = intersect_size / (j + 1.0)
                    
                    ap += (recall - old_recall) * ((old_precision + precision) / 2.0)
                    old_recall = recall
                    old_precision = precision
                
                map_score += ap
                
                # Compute recall@k
                for k_idx, k in enumerate(ks):
                    if len(ranks_clean) >= k:
                        hits = sum(1 for r in ranks_clean[:k] if r in ok)
                        if len(ok) > 0:
                            recall_at_k[k_idx] += hits / len(ok)
        
        map_score /= len(gnd)
        recall_at_k /= len(gnd)
        
        return map_score, None, recall_at_k, None

    def evaluate_roxford_protocol(roxford_root, model, device, eval_batch_size):
        """
        Evaluate using proper ROxford protocol with Easy/Medium/Hard difficulties
        """
        import numpy as np
        
        model.eval()
        
        # Load ground truth data
        ground_truth = []
        with open(roxford_root / "ground_truth.txt", "r") as f:
            for line in f:
                # Split by spaces but handle multiple consecutive spaces 
                parts = [p for p in line.strip().split(' ') if p]
                query_id = int(parts[0])
                bbx = list(map(float, parts[1].split(',')))
                
                # Handle variable number of fields (some may be missing or empty)
                easy = list(map(int, parts[2].split(','))) if len(parts) > 2 and parts[2] else []
                hard = list(map(int, parts[3].split(','))) if len(parts) > 3 and parts[3] else []
                junk = list(map(int, parts[4].split(','))) if len(parts) > 4 and parts[4] else []
                
                ground_truth.append({
                    'bbx': bbx,
                    'easy': easy,
                    'hard': hard,
                    'junk': junk
                })
        
        # Load query images
        query_images = []
        with open(roxford_root / "query_images.txt", "r") as f:
            for line in f:
                parts = line.strip().split()
                img_id = int(parts[0])
                img_path = parts[1]
                query_images.append((img_id, img_path))
        
        # Load database images  
        database_images = []
        with open(roxford_root / "database_images.txt", "r") as f:
            for line in f:
                parts = line.strip().split()
                img_id = int(parts[0])
                img_path = parts[1]
                database_images.append((img_id, img_path))
        
        print(f"Loaded {len(query_images)} query images and {len(database_images)} database images")
        
        # Extract query features
        query_features = []
        print("Extracting query features...")
        for i in trange(len(query_images), desc="query"):
            img_id, path = query_images[i]
            img = _load_image(path).unsqueeze(0).to(device)
            
            with torch.no_grad():
                emb = model(img)
                # Use CLS token and normalize
                feat = F.normalize(emb[0, 0, :], dim=0).cpu()
                query_features.append(feat)
        
        query_features = torch.stack(query_features)
        
        # Extract database features
        database_features = []
        print("Extracting database features...")
        for i in trange(0, len(database_images), eval_batch_size, desc="database"):
            batch_imgs = []
            batch_end = min(i + eval_batch_size, len(database_images))
            
            for j in range(i, batch_end):
                img_id, path = database_images[j]
                img = _load_image(path)
                batch_imgs.append(img)
            
            batch_tensor = torch.stack(batch_imgs).to(device)
            
            with torch.no_grad():
                embs = model(batch_tensor)
                # Use CLS token and normalize
                feats = F.normalize(embs[:, 0, :], dim=1).cpu()
                for feat in feats:
                    database_features.append(feat)
        
        database_features = torch.stack(database_features)
        
        # Compute similarities and rankings
        print("Computing similarities and rankings...")
        similarities = torch.mm(database_features, query_features.t())  # [DB_SIZE, NUM_QUERIES]
        ranks = torch.argsort(-similarities, dim=0).numpy()  # Sort in descending order
        
        # Evaluation protocol
        ks = [1, 5, 10]
        
        # Easy evaluation (only easy positives)
        gnd_easy = []
        for i, gnd in enumerate(ground_truth):
            g = {
                'ok': np.array(gnd['easy']),
                'junk': np.concatenate([gnd['junk'], gnd['hard']])
            }
            gnd_easy.append(g)
        
        mapE, _, mprE, _ = compute_map(ranks, gnd_easy, ks)
        
        # Medium evaluation (easy + hard positives)
        gnd_medium = []
        for i, gnd in enumerate(ground_truth):
            g = {
                'ok': np.concatenate([gnd['easy'], gnd['hard']]),
                'junk': np.array(gnd['junk'])
            }
            gnd_medium.append(g)
        
        mapM, _, mprM, _ = compute_map(ranks, gnd_medium, ks)
        
        # Hard evaluation (only hard positives)
        gnd_hard = []
        for i, gnd in enumerate(ground_truth):
            g = {
                'ok': np.array(gnd['hard']),
                'junk': np.concatenate([gnd['junk'], gnd['easy']])
            }
            gnd_hard.append(g)
        
        mapH, _, mprH, _ = compute_map(ranks, gnd_hard, ks)
        
        # Print results
        print(f"mAP E: {mapE*100:.2f}, M: {mapM*100:.2f}, H: {mapH*100:.2f}")
        print(f"mP@k{ks} E: {mprE*100}, M: {mprM*100}, H: {mprH*100}")
        
        model.train()
        return mapE, mapM, mapH

    # Initialize everything
    train_paths, test_paths = parse_roxford(roxford_root)
    
    # Setup multi-GPU
    device = torch.device("cuda:0")  # Primary device
    num_gpus = torch.cuda.device_count()
    print(f"🚀 Using {num_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
    print(f"Primary device: {device}")
    print(f"Found {len(train_paths)} buildings for training: {list(train_paths.keys())}")
    print(f"Found {len(test_paths)} buildings for testing: {list(test_paths.keys())}")
    
    # Create TRAINABLE model
    model = TrainableMultiVectorEncoder()
    
    # Move to primary GPU first
    model = model.to(device)
    
    # Wrap with DataParallel for multi-GPU
    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"✅ Model wrapped with DataParallel across {num_gpus} GPUs")
        effective_batch_size = batch_size * num_gpus
    else:
        effective_batch_size = batch_size
    
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    print(f"🎯 Effective batch size: {effective_batch_size} (per_gpu: {batch_size}, gpus: {num_gpus})")
    
    # Setup optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = TripletColbertLoss(margin=0.2)
    
    hist = []

    print(f"🚀 Starting {num_gpus}-GPU TRAINABLE ROxford triplet training...")
    
    for i in trange(steps, desc="train", unit="step"):
        # Get batch
        a, p, n = get_batch(train_paths, batch_size)
        a, p, n = a.to(device), p.to(device), n.to(device)

        # Forward pass
        emb_a = model(a)
        emb_p = model(p)
        emb_n = model(n)

        # Compute loss
        loss = criterion(emb_a, emb_p, emb_n)
        hist.append(loss.item())

        print(f"Current Loss: {loss.item()}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % report_interval == 0:
            avg_loss = sum(hist[-report_interval:]) / report_interval
            print(f"[step {i+1:4d}] avg loss: {avg_loss:.4f} (effective_bs: {effective_batch_size})")
            
            # Quick evaluation every few steps
            if (i + 1) % (report_interval * 2) == 0:
                print("📊 Quick evaluation:")
                evaluate_roxford_protocol(
                    roxford_root, model, device, eval_batch_size
                )

    # Final comprehensive evaluation
    print("--------------------------------")
    print("🎯 Final evaluation:")
    print(f"Final loss: {hist[-1]:.8f}")
    mapE, mapM, mapH = evaluate_roxford_protocol(
        roxford_root, model, device, eval_batch_size
    )
    
    print(f"✅ {num_gpus}-GPU ROxford Training complete!")
    return {"final_loss": hist[-1] if hist else 0.0, "avg_final_loss": sum(hist[-10:]) / 10 if len(hist) >= 10 else 0.0}


if __name__ == "__main__":
    # For local testing
    with app.run():
        result = main.remote()
        print("Result:", result) 