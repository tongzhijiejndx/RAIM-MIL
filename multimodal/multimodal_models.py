import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalMILModel(nn.Module):
    def __init__(
        self,
        clin_dim: int,
        n_classes: int = 2,
        pooling_mode: str = "attention",
        L: int = 512,
        D: int = 128,
        K: int = 1,
        img_dropout: float = 0.25,
        clin_dropout: float = 0.20,
        cls_dropout: float = 0.25,
        use_img_layernorm: bool = True,
        use_img_l2norm: bool = False,
    ):
        super().__init__()

        self.clin_dim = int(clin_dim)
        self.n_classes = int(n_classes)
        self.pooling_mode = pooling_mode
        self.L = int(L)
        self.D = int(D)
        self.K = int(K)
        self.use_img_layernorm = bool(use_img_layernorm)
        self.use_img_l2norm = bool(use_img_l2norm)

        # -----------------------------
        # Image branch
        # -----------------------------
        self.img_feat_norm = nn.LayerNorm(2048, elementwise_affine=False)

        self.feature_extractor_part1 = nn.Sequential(
            nn.Linear(2048, self.L),
            nn.ReLU(),
            nn.Dropout(img_dropout),
        )

        if self.pooling_mode == "attention":
            self.attention_V = nn.Sequential(
                nn.Linear(self.L, self.D),
                nn.Tanh(),
            )
            self.attention_U = nn.Sequential(
                nn.Linear(self.L, self.D),
                nn.Sigmoid(),
            )
            self.attention_weights = nn.Linear(self.D, self.K)

        # -----------------------------
        # Clinical branch
        # -----------------------------
        self.clin_net = nn.Sequential(
            nn.Linear(self.clin_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(clin_dropout),
        )

        # -----------------------------
        # Fusion + classifier
        # -----------------------------
        fusion_dim = self.L + 128
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(cls_dropout),
            nn.Linear(64, self.n_classes),
        )

    def encode_image(self, img_feats: torch.Tensor):
        if img_feats is None:
            raise ValueError("img_feats is None")

        x = img_feats.squeeze(0)  # [N, 2048]

        if self.use_img_layernorm:
            x = self.img_feat_norm(x)
        if self.use_img_l2norm:
            x = F.normalize(x, p=2, dim=1)

        H = self.feature_extractor_part1(x)  # [N, L]

        if self.pooling_mode == "attention":
            A_V = self.attention_V(H)
            A_U = self.attention_U(H)
            A = self.attention_weights(A_V * A_U)  # [N, K]
            A = torch.transpose(A, 1, 0)           # [K, N]
            A = F.softmax(A, dim=1)

            if A.size(0) == 1:
                M = torch.mm(A, H)                 # [1, L]
            else:
                M = torch.mean(
                    torch.stack([torch.mm(A[k:k+1], H) for k in range(A.size(0))], dim=0),
                    dim=0
                )
            return M, A, H

        if self.pooling_mode == "mean":
            M = torch.mean(H, dim=0, keepdim=True)
            return M, None, H

        if self.pooling_mode == "max":
            M, _ = torch.max(H, dim=0, keepdim=True)
            return M, None, H

        raise ValueError(f"Unknown pooling_mode: {self.pooling_mode}")

    def encode_clinical(self, clin_feats: torch.Tensor):
        if clin_feats is None:
            raise ValueError("clin_feats is None")

        if clin_feats.dim() == 1:
            clin_feats = clin_feats.unsqueeze(0)

        C = self.clin_net(clin_feats)
        return C

    def forward(self, img_feats: torch.Tensor, clin_feats: torch.Tensor):
        M, A, H = self.encode_image(img_feats)
        C = self.encode_clinical(clin_feats)

        combined = torch.cat([M, C], dim=1)
        bag_logits = self.classifier(combined)

        extras = {
            "image_bag_embedding": M,
            "clinical_embedding": C,
            "patch_embeddings": H,
        }
        return bag_logits, A, extras


class ClinicalMLP(nn.Module):
    def __init__(self, clin_dim: int, n_classes: int = 2, dropout: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clin_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, clin_feats: torch.Tensor):
        if clin_feats.dim() == 1:
            clin_feats = clin_feats.unsqueeze(0)
        return self.net(clin_feats)