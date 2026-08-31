from types import SimpleNamespace as NS

CFG = NS(
    model_name="vit_base_patch14_reg4_dinov2.lvd142m",
    embed_dim=768,
    num_registers=4,
    roi_side=3,
    rois_per_cue=1,
    batch_size=64,
    lr=3e-4,
    weight_decay=1e-2,
    margin=0.2,
    steps=10_000,
    device="cuda",
    img_size=518,
    train_root="data/train",
    test_root="data/test",
    recall_k=(1, 2, 4, 8),
)
