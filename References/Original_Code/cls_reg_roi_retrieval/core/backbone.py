import timm, torch
from einops import rearrange
from cls_reg_roi_retrieval.config import CFG

class DINOv2RegBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model(CFG.model_name, pretrained=True, num_classes=0)

    @torch.no_grad()
    def forward(self, x):
        #tokens = self.vit.forward_features(x, return_all_tokens=True)
        print("------- X -----------")
        print(x)
        print("------------------------")
        tokens = self.vit.forward_features(x)
        print("------- Tokens -----------")
        print(tokens)
        print("------------------------")
        cls_tok   = tokens[:, 0]
        regs_tok  = tokens[:, 1:1 + CFG.num_registers]
        patch_tok = tokens[:, 1 + CFG.num_registers:]
        print("------- Stack -----------")
        print(cls_tok, regs_tok, patch_tok)
        print("------------------------")
        return cls_tok, regs_tok, patch_tok

    def patch_grid(self, patch_tok):
        g = int(CFG.img_size // 14)  # ViT-B/14 grid
        return rearrange(patch_tok, "b (h w) d -> b h w d", h=g, w=g)
