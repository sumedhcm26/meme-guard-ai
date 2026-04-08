"""
model_architecture.py
─────────────────────────────────────────────────────────────
Exact reconstruction of the HatefulMemeModel from the training
notebook (dl-v4.ipynb).

Architecture Overview:
  INPUT : img_feat (B, 768)  +  txt_feat (B, 768)
          — frozen CLIP ViT-L/14 outputs, L2-normalised

  1. AdapterBlock     : lightweight trainable projection per modality
  2. CrossModalFusion : bidirectional cross-attention
                        concat + hadamard product  → (B, 3×768)
  3. Classifier       : FC(3×768 → 512 → 256 → 1)

  OUTPUT : scalar logit  +  auxiliary unimodal logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# 1.  Adapter Block
# ──────────────────────────────────────────────────────────────
class AdapterBlock(nn.Module):
    """
    Lightweight trainable adapter on top of frozen CLIP features.

    Linear → LayerNorm → GELU → Linear → LayerNorm
    Kept shallow intentionally to limit overfitting on small datasets.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────
# 2.  Cross-Modal Fusion
# ──────────────────────────────────────────────────────────────
class CrossModalFusion(nn.Module):
    """
    Bidirectional cross-attention:
      • image queries attend to text  keys/values → image-in-text-context
      • text  queries attend to image keys/values → text-in-image-context

    Final output concatenates three signals:
      [img_ctx, txt_ctx, img_ctx ⊙ txt_ctx]

    The Hadamard product captures joint features that neither
    unimodal branch can represent on its own.
    """

    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.img_to_txt = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.txt_to_img = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm_img = nn.LayerNorm(dim)
        self.norm_txt = nn.LayerNorm(dim)

    def forward(
        self, img: torch.Tensor, txt: torch.Tensor
    ) -> torch.Tensor:
        # Both inputs: (B, D) — add sequence dimension for MHA API
        img_seq = img.unsqueeze(1)   # (B, 1, D)
        txt_seq = txt.unsqueeze(1)   # (B, 1, D)

        # Image attends to text
        img_ctx, _ = self.img_to_txt(img_seq, txt_seq, txt_seq)
        img_out    = self.norm_img(img + img_ctx.squeeze(1))  # residual

        # Text attends to image
        txt_ctx, _ = self.txt_to_img(txt_seq, img_seq, img_seq)
        txt_out    = self.norm_txt(txt + txt_ctx.squeeze(1))  # residual

        # Fusion: concat + interaction term  → (B, 3*D)
        fused = torch.cat([img_out, txt_out, img_out * txt_out], dim=-1)
        return fused


# ──────────────────────────────────────────────────────────────
# 3.  Full Model
# ──────────────────────────────────────────────────────────────
class HatefulMemeModel(nn.Module):
    """
    Full multimodal hateful-meme classifier.

    Forward signature:
        logit, img_logit, txt_logit = model(img_feat, txt_feat)

    During inference only `logit` is used.
    `img_logit` / `txt_logit` are auxiliary outputs used only
    by the training loss to prevent unimodal shortcuts.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        D = cfg["embed_dim"]   # 768  (ViT-L/14 dimension)

        # Per-modality adapters
        self.img_adapter = AdapterBlock(D, D)
        self.txt_adapter = AdapterBlock(D, D)

        # Bidirectional cross-attention fusion
        self.fusion = CrossModalFusion(
            dim=D, heads=cfg["fusion_heads"], dropout=0.1
        )

        # Classifier: 3*D → hidden_dim → hidden_dim/2 → 1
        self.classifier = nn.Sequential(
            nn.Linear(3 * D, cfg["hidden_dim"]),
            nn.GELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["hidden_dim"], cfg["hidden_dim"] // 2),
            nn.GELU(),
            nn.Dropout(cfg["dropout"] * 0.75),
            nn.Linear(cfg["hidden_dim"] // 2, 1),
        )

        # Auxiliary unimodal heads (used only during training)
        self.img_only_head = nn.Linear(D, 1)
        self.txt_only_head = nn.Linear(D, 1)

        self._init_weights()

    # ── Weight initialisation ──────────────────────────────────
    def _init_weights(self):
        """Kaiming (He) initialisation for all Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward pass ──────────────────────────────────────────
    def forward(
        self,
        img_feat: torch.Tensor,
        txt_feat: torch.Tensor,
    ):
        """
        Args:
            img_feat : (B, 768) — L2-normalised CLIP image embedding
            txt_feat : (B, 768) — L2-normalised CLIP text  embedding

        Returns:
            logit     : (B,)  main prediction logit
            img_logit : (B,)  image-only auxiliary logit
            txt_logit : (B,)  text-only  auxiliary logit
        """
        img = self.img_adapter(img_feat)
        txt = self.txt_adapter(txt_feat)

        fused     = self.fusion(img, txt)
        logit     = self.classifier(fused).squeeze(-1)       # (B,)

        img_logit = self.img_only_head(img).squeeze(-1)      # (B,)
        txt_logit = self.txt_only_head(txt).squeeze(-1)      # (B,)

        return logit, img_logit, txt_logit


# ──────────────────────────────────────────────────────────────
# Factory helper  (used by predict.py)
# ──────────────────────────────────────────────────────────────
def build_model(cfg: dict) -> HatefulMemeModel:
    """Instantiate model with the given config dict."""
    return HatefulMemeModel(cfg)
