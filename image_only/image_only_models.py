# image_only/image_only_models.py
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Common helpers
# =========================================================
class PatchProjector(nn.Module):
    def __init__(
        self,
        in_dim: int = 2048,
        out_dim: int = 512,
        dropout: float = 0.25,
        use_layernorm: bool = True,
        use_l2norm: bool = False,
    ):
        super().__init__()
        self.use_layernorm = use_layernorm
        self.use_l2norm = use_l2norm

        self.norm = nn.LayerNorm(in_dim, elementwise_affine=False) if use_layernorm else nn.Identity()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        if self.use_l2norm:
            x = F.normalize(x, p=2, dim=1)
        x = self.proj(x)
        return x


class GatedAttention(nn.Module):
    def __init__(self, L: int = 512, D: int = 128):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.attention_w = nn.Linear(D, 1)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        A = self.attention_w(self.attention_V(H) * self.attention_U(H))  # [N, 1]
        return A.squeeze(1)  # [N]


# =========================================================
# 1) Mean Pooling MIL
# =========================================================
class MeanPoolingMIL(nn.Module):
    def __init__(
        self,
        in_dim: int = 2048,
        L: int = 512,
        n_classes: int = 2,
        dropout: float = 0.25,
        use_layernorm: bool = True,
        use_l2norm: bool = False,
    ):
        super().__init__()
        self.projector = PatchProjector(
            in_dim=in_dim,
            out_dim=L,
            dropout=dropout,
            use_layernorm=use_layernorm,
            use_l2norm=use_l2norm,
        )
        self.classifier = nn.Sequential(
            nn.Linear(L, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, img_feats: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        if img_feats.ndim == 3:
            x = img_feats.squeeze(0)
        else:
            x = img_feats

        H = self.projector(x)           # [N, L]
        M = H.mean(dim=0, keepdim=True) # [1, L]
        bag_logits = self.classifier(M)

        return bag_logits, None, {
            "H": H,
            "bag_embedding": M,
        }


# =========================================================
# 2) ABMIL
# =========================================================
class ABMIL(nn.Module):
    def __init__(
        self,
        in_dim: int = 2048,
        L: int = 512,
        D: int = 128,
        n_classes: int = 2,
        dropout: float = 0.25,
        use_layernorm: bool = True,
        use_l2norm: bool = False,
    ):
        super().__init__()
        self.projector = PatchProjector(
            in_dim=in_dim,
            out_dim=L,
            dropout=dropout,
            use_layernorm=use_layernorm,
            use_l2norm=use_l2norm,
        )
        self.attn = GatedAttention(L=L, D=D)
        self.classifier = nn.Sequential(
            nn.Linear(L, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, img_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        if img_feats.ndim == 3:
            x = img_feats.squeeze(0)
        else:
            x = img_feats

        H = self.projector(x)          # [N, L]
        A_raw = self.attn(H)           # [N]
        A = torch.softmax(A_raw, dim=0)  # [N]
        M = torch.matmul(A.unsqueeze(0), H)  # [1, L]
        bag_logits = self.classifier(M)

        return bag_logits, A.unsqueeze(0), {
            "H": H,
            "A_raw": A_raw,
            "bag_embedding": M,
        }


# =========================================================
# 3) CLAM (simplified single-branch version)
# =========================================================
class CLAM_SB(nn.Module):
    def __init__(
        self,
        in_dim: int = 2048,
        L: int = 512,
        D: int = 128,
        n_classes: int = 2,
        dropout: float = 0.25,
        k_sample: int = 8,
        use_layernorm: bool = True,
        use_l2norm: bool = False,
    ):
        super().__init__()
        self.k_sample = k_sample
        self.n_classes = n_classes

        self.projector = PatchProjector(
            in_dim=in_dim,
            out_dim=L,
            dropout=dropout,
            use_layernorm=use_layernorm,
            use_l2norm=use_l2norm,
        )
        self.attn = GatedAttention(L=L, D=D)

        self.bag_classifier = nn.Sequential(
            nn.Linear(L, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

        # instance classifier for auxiliary instance loss if needed later
        self.instance_classifier = nn.Linear(L, n_classes)

    def get_instance_logits(
        self,
        H: torch.Tensor,
        A: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        N = H.size(0)
        if N == 0:
            return {}

        k = min(self.k_sample, N)

        top_idx = torch.topk(A, k=k, largest=True).indices
        bot_idx = torch.topk(A, k=k, largest=False).indices

        top_feats = H[top_idx]  # [k, L]
        bot_feats = H[bot_idx]  # [k, L]

        top_logits = self.instance_classifier(top_feats)
        bot_logits = self.instance_classifier(bot_feats)

        return {
            "top_idx": top_idx,
            "bot_idx": bot_idx,
            "top_logits": top_logits,
            "bot_logits": bot_logits,
        }

    def forward(self, img_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        if img_feats.ndim == 3:
            x = img_feats.squeeze(0)
        else:
            x = img_feats

        H = self.projector(x)             # [N, L]
        A_raw = self.attn(H)              # [N]
        A = torch.softmax(A_raw, dim=0)   # [N]

        M = torch.matmul(A.unsqueeze(0), H)  # [1, L]
        bag_logits = self.bag_classifier(M)

        inst_dict = self.get_instance_logits(H, A)

        extra = {
            "H": H,
            "A_raw": A_raw,
            "bag_embedding": M,
            **inst_dict,
        }
        return bag_logits, A.unsqueeze(0), extra


# =========================================================
# 4) TransMIL (lightweight practical version)
# =========================================================
class NystromLikePositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_tokens: int = 4096):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(1)
        return x + self.pos_embed[:, :N, :]


class TransMIL(nn.Module):
    def __init__(
        self,
        in_dim: int = 2048,
        L: int = 512,
        n_classes: int = 2,
        dropout: float = 0.25,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 1024,
        max_tokens: int = 4096,
        use_layernorm: bool = True,
        use_l2norm: bool = False,
    ):
        super().__init__()
        self.projector = PatchProjector(
            in_dim=in_dim,
            out_dim=L,
            dropout=dropout,
            use_layernorm=use_layernorm,
            use_l2norm=use_l2norm,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, L))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc = NystromLikePositionalEncoding(dim=L, max_tokens=max_tokens + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=L,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(L)
        self.classifier = nn.Sequential(
            nn.Linear(L, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, img_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        if img_feats.ndim == 3:
            x = img_feats.squeeze(0)
        else:
            x = img_feats

        H = self.projector(x)             # [N, L]
        H = H.unsqueeze(0)                # [1, N, L]

        cls_tok = self.cls_token.expand(H.size(0), -1, -1)  # [1, 1, L]
        tokens = torch.cat([cls_tok, H], dim=1)             # [1, 1+N, L]
        tokens = self.pos_enc(tokens)
        tokens = self.transformer(tokens)                   # [1, 1+N, L]
        tokens = self.norm(tokens)

        cls_out = tokens[:, 0, :]          # [1, L]
        patch_tokens = tokens[:, 1:, :]    # [1, N, L]

        bag_logits = self.classifier(cls_out)

        # proxy attention: cls-patch cosine similarity
        cls_norm = F.normalize(cls_out, dim=1)          # [1, L]
        patch_norm = F.normalize(patch_tokens, dim=2)   # [1, N, L]
        A_proxy = torch.matmul(cls_norm.unsqueeze(1), patch_norm.transpose(1, 2)).squeeze(1)  # [1, N]
        A_proxy = torch.softmax(A_proxy, dim=1)

        return bag_logits, A_proxy, {
            "H": H.squeeze(0),
            "tokens": tokens,
            "bag_embedding": cls_out,
        }


# =========================================================
# Factory
# =========================================================
def build_image_only_model(
    model_name: str,
    in_dim: int = 2048,
    n_classes: int = 2,
    **kwargs,
) -> nn.Module:
    name = str(model_name).lower()

    if name in ["meanpoolingmil", "mean_pooling_mil", "mean", "meanpooling"]:
        return MeanPoolingMIL(in_dim=in_dim, n_classes=n_classes, **kwargs)

    if name in ["abmil", "attentionmil", "attention_mil"]:
        return ABMIL(in_dim=in_dim, n_classes=n_classes, **kwargs)

    if name in ["clam", "clam_sb", "clamsb"]:
        return CLAM_SB(in_dim=in_dim, n_classes=n_classes, **kwargs)

    if name in ["transmil", "trans_mil"]:
        return TransMIL(in_dim=in_dim, n_classes=n_classes, **kwargs)

    raise ValueError(f"Unknown model_name: {model_name}")