import modal
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import trange

from model_utils import _load_image, MultiVectorEncoder
from maxsim_loss import TripletColbertLoss

# Build Modal Image including local Python source code
image = (
  modal.Image.debian_slim()
    .pip_install("torch","torchvision","tqdm","timm==0.9.12","einops==0.7.0","pillow")
    .add_local_file("model_utils.py",   "/root/model_utils.py")
    .add_local_file("buddy_pool.py",    "/root/buddy_pool.py")
    .add_local_file("maxsim_loss.py",   "/root/maxsim_loss.py")
    .add_local_file("modal_app.py",     "/root/modal_app.py")
)

# Define Modal App with dataset volume
app = modal.App(
    "cub_triplet_app",
    image=image,
    volumes={"/mnt/data": modal.Volume.from_name("cub-data")}
)

@app.function(
    #gpu="any"
    #gpu="T4"
    gpu="A100",
    #gpu="A100-80GB"
    #gpu="A100-80GB:4"
    timeout=3600  # 1 hour timeout instead of default 5 minutes
)
def main(
    cub_root: str = "/mnt/data/Flowers_converted",
    steps: int = 200,
    batch_size: int = 4,
    report_interval: int = 50,
    eval_batch_size: int = 64
):
    """
    Train CUB triplet model on Modal and evaluate recall@k.
    """
    print("Starting Modal function...")
    cub_root = Path(cub_root)

    def parse_cub(root: Path):
        print("Parsing CUB metadata...")
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
            img_map[int(iid)] = "/mnt/data/" + rel

        train_ids = set()
        for line in (root / "train_test_split.txt").read_text().splitlines():
            iid, flag = line.split()
            if int(flag):
                train_ids.add(int(iid))

        train_paths = {c: [] for c in cls_map.values()}
        test_paths  = {c: [] for c in cls_map.values()}
        for iid, path in img_map.items():
            cname = cls_map[img_to_cid[iid]]
            (train_paths[cname] if iid in train_ids else test_paths[cname]).append(path)

        train_paths = {c: ps for c, ps in train_paths.items() if len(ps) >= 2}
        test_paths  = {c: ps for c, ps in test_paths.items() if c in train_paths and len(ps) >= 1}
        
        print(f"Found {len(train_paths)} classes for training")
        print(f"Found {len(test_paths)} classes for testing")
        return train_paths, test_paths

    def load_cub_triplet(class_to_paths: dict[str, list[Path]]):
        cls_pos = random.choice(list(class_to_paths.keys()))
        # escolhe anchor e positivo independentemente da mesma classe (podem ser iguais ou diferentes)
        a = random.choice(class_to_paths[cls_pos])
        p = random.choice(class_to_paths[cls_pos])
        neg_cls = random.choice([c for c in class_to_paths if c != cls_pos])
        n = random.choice(class_to_paths[neg_cls])
        return a, p, n

    def get_batch(class_to_paths: dict[str, list[Path]], batch_size: int):
        anchors, positives, negatives = [], [], []
        for _ in range(batch_size):
            a, p, n = load_cub_triplet(class_to_paths)
            for lst, img in ((anchors, a), (positives, p), (negatives, n)):
                t = _load_image(img) if isinstance(img, (str, Path)) else img.float() / 255
                if t.ndim == 3:
                    t = t.unsqueeze(0)
                if t.ndim == 4 and t.shape[0] == 1:
                    t = t.squeeze(0)
                lst.append(t)
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

    def evaluate_retrieval_recalls(
        train_paths, test_paths, encoder, device, ks, eval_batch_size
    ):
        print("Setting encoder to eval mode...")
        encoder.eval()

        classes = sorted(train_paths.keys())
        cls2idx = {c: i for i, c in enumerate(classes)}
        gallery_embs, gallery_labels = [], []
        
        print("Building gallery embeddings...")
        total_gallery_images = sum(len(paths) for paths in train_paths.values())
        print(f"Processing {total_gallery_images} gallery images...")
        
        processed = 0
        with torch.no_grad():
            for c in classes:
                for p in train_paths[c]:
                    img = _load_image(p)
                    if img.ndim == 3:
                        img = img.unsqueeze(0)
                    img = img.to(device)
                    emb = encoder(img)  # (1, 10, D)
                    # usar o primeiro token (CLS) como representação
                    gallery_embs.append(emb[0, 0, :].cpu())  # (D,)
                    gallery_labels.append(cls2idx[c])
                    
                    processed += 1
                    if processed % 100 == 0:
                        print(f"Processed {processed}/{total_gallery_images} gallery images")
        
        gallery = torch.stack(gallery_embs, dim=0)
        gallery_norm = F.normalize(gallery, dim=1)
        print(f"Gallery built: {gallery.shape}")

        test_items = [(cls2idx[c], p) for c, paths in test_paths.items() for p in paths]
        total = len(test_items)
        hits = {k: 0 for k in ks}
        
        print(f"Evaluating {total} test images...")

        for i in range(0, total, eval_batch_size):
            batch = test_items[i:i+eval_batch_size]
            labels = torch.tensor([lbl for lbl, _ in batch])
            imgs = []
            for _, p in batch:
                img = _load_image(p)
                if img.ndim == 3:
                    img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0).to(device)

            with torch.no_grad():
                embs = encoder(imgs)  # [B, 10, D]
                # usar o primeiro token (CLS) como representação da query
                query_embs = embs[:, 0, :].cpu()  # [B, D]
                query_norm = F.normalize(query_embs, dim=1)  # [B, D]

            sims = query_norm @ gallery_norm.t()
            topk = sims.topk(max(ks), dim=1).indices.cpu().tolist()

            for k in ks:
                for qi, row in enumerate(topk):
                    if any(gallery_labels[idx] == labels[qi].item() for idx in row[:k]):
                        hits[k] += 1
            
            if (i + eval_batch_size) % (eval_batch_size * 10) == 0:
                print(f"Evaluated {min(i + eval_batch_size, total)}/{total} test images")

        for k in ks:
            print(f"Recall@{k}: {hits[k] / total:.4f} ({hits[k]}/{total})")

        encoder.train()

    # Running training
    print("Setting up training...")
    train_paths, test_paths = parse_cub(cub_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Use MultiVectorEncoder like the original (feature extraction only, no training)
    print("Loading MultiVectorEncoder (this may take a while for first download)...")
    encoder = MultiVectorEncoder().to(device)
    encoder.eval()  # Set to eval mode since we're doing feature extraction only
    print("Model loaded successfully!")

    # No optimizer needed since MultiVectorEncoder.forward() has @torch.no_grad()
    # This matches the original implementation which does feature extraction only
    
    # Use original loss function (for monitoring/evaluation purposes)
    criterion = TripletColbertLoss(margin=0.2)
    hist = []

    print(f"Starting training loop with {steps} steps...")
    for i in trange(steps, desc="train", unit="step"):
        a, p, n = get_batch(train_paths, batch_size)
        a, p, n = a.to(device), p.to(device), n.to(device)

        # Forward through encoder (no gradients due to @torch.no_grad())
        emb_a = encoder(a)  # (B, 10, D)
        emb_p = encoder(p)  # (B, 10, D)
        emb_n = encoder(n)  # (B, 10, D)

        # Compute ColBERT loss (for monitoring only, no backprop)
        with torch.no_grad():
            loss = criterion(emb_a, emb_p, emb_n)
            hist.append(loss.item())

        if (i + 1) % report_interval == 0:
            avg = sum(hist[-report_interval:]) / report_interval
            print(f"[step {i+1:4d}] avg loss: {avg:.4f}")

    print(f"Final loss: {hist[-1]:.4f}")
    print(f"Avg loss: {sum(hist) / len(hist):.4f}")

    # Evaluate model
    print("Starting evaluation...")
    evaluate_retrieval_recalls(
        train_paths, test_paths, encoder, device,
        ks=[1, 2, 4], eval_batch_size=eval_batch_size
    )
    print("Evaluation complete!")


