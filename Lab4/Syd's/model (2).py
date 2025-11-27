import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet50ImageEncoder(nn.Module):
    """
    ResNet50 backbone (ImageNet-pretrained) + 2-layer projection head to 512-d space.
    """
    def __init__(self, embed_dim=512, proj_hidden_dim=1024):
        super().__init__()
        backbone = models.resnet50(pretrained=True)
        in_dim = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_hidden_dim),
            nn.GELU(),
            nn.Linear(proj_hidden_dim, embed_dim),
        )

    def forward(self, x):
        """
        x: [B, 3, 224, 224]
        returns: [B, embed_dim] (L2-normalized)
        """
        feats = self.backbone(x)        # [B, 2048]
        emb = self.proj(feats)          # [B, 512]
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return emb


class CLIPLoss(nn.Module):
    """
    Symmetric InfoNCE loss used in CLIP:
      - image->text
      - text->image
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_embs, text_embs):
        """
        image_embs: [B, D]
        text_embs:  [B, D]
        """
        image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)
        text_embs  = text_embs  / text_embs.norm(dim=-1, keepdim=True)

        logits = image_embs @ text_embs.t()    # [B, B]
        logits = logits / self.temperature

        targets = torch.arange(len(logits), device=logits.device)

        loss_i = F.cross_entropy(logits, targets)
        loss_t = F.cross_entropy(logits.t(), targets)

        return (loss_i + loss_t) / 2
