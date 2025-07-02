from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import trange, tqdm

from model_utils import _load_image, MultiVectorEncoder
from maxsim_loss import TripletColbertLoss


def parse_roxford(root: Path):
    """
    Ler metadados do ROxford5k convertido em root e retornar dicionarios de listas de imagens de treino e teste
    """
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
        img_map[int(iid)] = rel

    train_ids = set()
    for line in (root / "train_test_split.txt").read_text().splitlines():
        iid, flag = line.split()
        if int(flag):
            train_ids.add(int(iid))

    train_paths = {c: [] for c in cls_map.values()}
    test_paths  = {c: [] for c in cls_map.values()}
    for iid, path in img_map.items():
        cname = cls_map[img_to_cid[iid]]
        if iid in train_ids:
            train_paths[cname].append(path)
        else:
            test_paths[cname].append(path)

    # filtrar classes com pelo menos duas imagens em treino
    train_paths = {c: ps for c, ps in train_paths.items() if len(ps) >= 2}
    # filtrar em teste apenas classes que estao em treino e com ao menos uma imagem
    test_paths  = {c: ps for c, ps in test_paths.items()  if c in train_paths and len(ps) >= 1}
    return train_paths, test_paths


def load_roxford_triplet(class_to_paths: dict[str, list[Path]]):
    """
    Escolher aleatoriamente um triplet anchor positivo negativo do ROxford5k
    """
    cls_pos = random.choice(list(class_to_paths.keys()))
    # escolhe anchor e positivo independentemente da mesma classe (podem ser iguais ou diferentes)
    a = random.choice(class_to_paths[cls_pos])
    p = random.choice(class_to_paths[cls_pos])
    # para negativo, escolhe outra classe
    other = [c for c in class_to_paths if c != cls_pos]
    neg_cls = random.choice(other)
    n = random.choice(class_to_paths[neg_cls])
    return a, p, n


def get_batch(class_to_paths: dict[str, list[Path]], batch_size: int):
    """
    Construir um lote de triplets como tensores
    Retorna tensores anchors, positives, negatives com forma [batch_size, C, H, W]
    """
    anchors, positives, negatives = [], [], []
    for _ in range(batch_size):
        a, p, n = load_roxford_triplet(class_to_paths)
        for lst, img in ((anchors, a), (positives, p), (negatives, n)):
            t = _load_image(img) if isinstance(img, (str, Path)) else img.float() / 255
            # se vier sem dim de batch, adicionar
            if t.ndim == 3:
                t = t.unsqueeze(0)
            # caso venha com dim extra de batch unica
            if t.ndim == 4 and t.shape[0] == 1:
                t = t.squeeze(0)
            lst.append(t)
    # garante empilhar no formato correto
    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)


def evaluate_retrieval_recalls(
    train_paths: dict[str, list[Path]],
    test_paths:  dict[str, list[Path]],
    encoder:     MultiVectorEncoder,
    device:      torch.device,
    ks:          list[int],
    eval_batch_size: int
):
    """
    Calcular recall em varios k em lotes usando MultiVectorEncoder
    """
    encoder.eval()

    # montar galeria de embeddings usando o encoder
    classes = sorted(train_paths.keys())
    cls2idx = {c: i for i, c in enumerate(classes)}
    gallery_embs, gallery_labels = [], []
    
    with torch.no_grad():
        for c in classes:
            for p in train_paths[c]:
                img = _load_image(p)
                # garantir formato [1, C, H, W]
                if img.ndim == 3:
                    img = img.unsqueeze(0)
                img = img.to(device)
                emb = encoder(img)  # (1, 10, D)
                # usar o primeiro token (CLS) como representação
                gallery_embs.append(emb[0, 0, :].cpu())  # (D,)
                gallery_labels.append(cls2idx[c])
    
    # empilhar no formato [N_gallery, D]
    gallery = torch.stack(gallery_embs, dim=0)
    gallery_norm = F.normalize(gallery, dim=1)   # [N_gallery, D]

    # preparar lista de queries de teste
    test_items = [(cls2idx[c], p) for c, paths in test_paths.items() for p in paths]
    total = len(test_items)
    hits = {k: 0 for k in ks}

    # avaliar em lotes
    for i in range(0, total, eval_batch_size):
        batch = test_items[i:i+eval_batch_size]
        labels = torch.tensor([lbl for lbl, _ in batch])  # CPU tensor de labels
        imgs = []
        for _, p in batch:
            img = _load_image(p)
            # garantir formato [1, C, H, W] antes de concatenar
            if img.ndim == 3:
                img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0).to(device)  # [B, C, H, W]

        with torch.no_grad():
            embs = encoder(imgs)  # [B, 10, D]
            # usar o primeiro token (CLS) como representação da query
            query_embs = embs[:, 0, :].cpu()  # [B, D]
            query_norm = F.normalize(query_embs, dim=1)  # [B, D]

        # calcular similaridade via produto interno
        sims = query_norm @ gallery_norm.t()  # [B, N_gallery]
        topk = sims.topk(max(ks), dim=1).indices.cpu().tolist()

        # calcular hits
        for k in ks:
            for qi, row in enumerate(topk):
                # se em algum dos top k indices a label bater, contar como acerto
                if any(gallery_labels[idx] == labels[qi].item() for idx in row[:k]):
                    hits[k] += 1

    # exibir resultado
    for k in ks:
        print(f"Recall@{k}: {hits[k] / total:.4f} ({hits[k]}/{total})")

    encoder.train()


def main(
    roxford_root: Path,
    steps: int,
    batch_size: int,
    report_interval: int,
    eval_batch_size: int
):
    train_paths, test_paths = parse_roxford(roxford_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Use MultiVectorEncoder like the original (feature extraction only, no training)
    encoder = MultiVectorEncoder().to(device)
    encoder.eval()  # Set to eval mode since we're doing feature extraction only

    # No optimizer needed since MultiVectorEncoder.forward() has @torch.no_grad()
    # This matches the original implementation which does feature extraction only
    
    # Use original loss function (for monitoring/evaluation purposes)
    criterion = TripletColbertLoss(margin=0.2)
    hist = []

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

    evaluate_retrieval_recalls(
        train_paths, test_paths, encoder, device,
        ks=[1, 2, 4], eval_batch_size=eval_batch_size
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roxford_root", type=Path, default=Path("roxford5k_converted"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--report_interval", type=int, default=50)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    args = parser.parse_args()
    main(
        args.roxford_root, args.steps,
        args.batch_size, args.report_interval,
        args.eval_batch_size
    )

