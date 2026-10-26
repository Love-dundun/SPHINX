"""
SPHINX — Spatial Proteomic inference from Histology with Integrated Neighborhood conteXt

Architecture overview:
  - Dual image encoder backbones (detail 96px + context 192px) fused via learned alpha
    Supported backbones:
      vit          — ViT-Base/16, ImageNet pretrained (primary model, recommended)
      ctranspath   — Swin-Tiny + ConvStem, TCGA H&E pretrained (external dataset adaptation)
  - Protein prediction head: fused image features → 18 protein expressions
  - ProteinGAT: biology-prior-guided graph attention to refine protein predictions
  - CellGAT: neighborhood graph attention over K-nearest spatial neighbors
  - Multi-Gate + Router: per-cell-type adaptive fusion of image and protein+graph features
  - Path B (ProteinMLPClassifier): direct protein→cell-type regularization branch
  - UncertaintyWeightedLoss: automatic multi-task loss balancing (Kendall et al. 2018)
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Dict, List
from timm.models.swin_transformer import SwinTransformerBlock, PatchMerging

from .protein_prior import (
    NEGATIVE_PAIRS,
    POSITIVE_PAIRS,
    PROTEIN_ORDER,
    build_bias_matrix,
)

log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
#  CTransPath backbone — faithful re-implementation
#  (Wang et al., Medical Image Analysis 2022, DOI 10.1016/j.media.2022.102559)
#
#  Architecture:
#    patch_embed : ConvStem  3→96,  224→56 (stride-4)
#    layers.0    : 2 SwinBlocks  dim=96   + PatchMerging → 192
#    layers.1    : 2 SwinBlocks  dim=192  + PatchMerging → 384
#    layers.2    : 6 SwinBlocks  dim=384  + PatchMerging → 768
#    layers.3    : 2 SwinBlocks  dim=768  (no downsample)
#    norm        : LayerNorm(768)
#    output      : global-average-pool → [B, 768]
# ────────────────────────────────────────────────────────────

class _CTPatchEmbed(nn.Module):
    """ConvStem patch embedding matching CTransPath checkpoint key layout:
      patch_embed.proj.{0,1,3,4,6}  patch_embed.norm
    """

    def __init__(self, in_chans: int = 3, embed_dim: int = 96):
        super().__init__()
        stem = embed_dim // 8   # 12
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, stem,       3, stride=2, padding=1, bias=False),  # idx 0
            nn.BatchNorm2d(stem),                                                   # idx 1
            nn.GELU(),                                                              # idx 2
            nn.Conv2d(stem, stem * 2,       3, stride=2, padding=1, bias=False),  # idx 3
            nn.BatchNorm2d(stem * 2),                                               # idx 4
            nn.GELU(),                                                              # idx 5
            nn.Conv2d(stem * 2, embed_dim,  1),                                    # idx 6
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                     # [B, C, H/4, W/4]
        x = x.permute(0, 2, 3, 1)           # [B, H/4, W/4, C]  (NHWC for swin blocks)
        x = self.norm(x)
        return x


class _CTStage(nn.Module):
    """One CTransPath stage: N SwinBlocks followed by optional PatchMerging.

    The downsample is at the END of each stage, matching the original ckpt.
    """

    def __init__(self, dim: int, depth: int, num_heads: int,
                 input_resolution: tuple, window_size: int = 7,
                 mlp_ratio: float = 4.0,
                 drop_path_rates: Optional[List[float]] = None,
                 downsample: bool = True):
        super().__init__()
        drop_path_rates = drop_path_rates or [0.0] * depth
        self.blocks = nn.Sequential(*[
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path_rates[i],
                dynamic_mask=True,
            )
            for i in range(depth)
        ])
        self.downsample: Optional[nn.Module] = (
            PatchMerging(dim=dim) if downsample else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class CTransPathBackbone(nn.Module):
    """Full CTransPath backbone.

    Exactly mirrors the checkpoint weight layout so SPHINX checkpoints trained
    with the CTransPath backbone load correctly (strict=True compatible).
    Output: [B, 768] global-average-pooled feature vector.
    """

    def __init__(self, drop_path_rate: float = 0.0):
        super().__init__()
        embed_dim = 96
        depths    = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
        img_size  = 56   # after ConvStem stride-4 on 224px input
        resolutions = [
            (img_size,      img_size),
            (img_size // 2, img_size // 2),
            (img_size // 4, img_size // 4),
            (img_size // 8, img_size // 8),
        ]
        dims = [embed_dim * (2 ** i) for i in range(4)]   # 96, 192, 384, 768

        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.patch_embed = _CTPatchEmbed(in_chans=3, embed_dim=embed_dim)

        cursor = 0
        self.layers = nn.ModuleList()
        for i in range(4):
            stage_dpr = dpr[cursor: cursor + depths[i]]
            cursor += depths[i]
            self.layers.append(_CTStage(
                dim=dims[i],
                depth=depths[i],
                num_heads=num_heads[i],
                input_resolution=resolutions[i],
                drop_path_rates=stage_dpr,
                downsample=(i < 3),   # layers 0,1,2 have downsample; layer 3 does not
            ))

        self.norm = nn.LayerNorm(dims[-1])   # 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)    # [B, 56, 56, 96]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x.mean(dim=(1, 2))    # global average pool → [B, 768]
        return x


def _build_ctranspath_backbone(
    pretrained: bool,
    ctranspath_weights: Optional[str],
    drop_path_rate: float,
) -> nn.Module:
    """Build a CTransPath backbone and optionally load TCGA pretrained weights.

    CTransPath is a pathology-specific model with a ConvStem patch embedding
    pretrained on large-scale TCGA H&E tiles (Wang et al. 2022).
    See: https://github.com/Xiyue-Wang/TransPath

    Pass *pretrained=False* when loading a full SPHINX checkpoint (the backbone
    weights are already embedded in the checkpoint).
    Pass *pretrained=True, ctranspath_weights=<path>* when fine-tuning SPHINX
    from scratch with the public CTransPath pretrained weights.
    """
    backbone = CTransPathBackbone(drop_path_rate=drop_path_rate)

    if pretrained and ctranspath_weights is not None:
        ckpt = torch.load(ctranspath_weights, map_location='cpu')
        state = ckpt.get('model', ckpt)
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        real_missing = [k for k in missing
                        if 'relative_position_index' not in k and 'attn_mask' not in k]
        real_unexpected = [k for k in unexpected
                           if 'relative_position_index' not in k and 'attn_mask' not in k]
        if real_missing:
            raise RuntimeError(f'[CTransPath] missing keys: {real_missing}')
        if real_unexpected:
            log.warning('[CTransPath] %d unexpected keys ignored: %s ...',
                        len(real_unexpected), real_unexpected[:2])
        log.info('Loaded CTransPath weights from: %s', ctranspath_weights)
    elif pretrained and ctranspath_weights is None:
        log.warning(
            "backbone='ctranspath' with pretrained=True but no "
            "--ctranspath_weights path provided; using random init."
        )

    return backbone


# ─────────────────────────────────────────────────────────────
#  ProteinGAT
# ─────────────────────────────────────────────────────────────

class ProteinGAT(nn.Module):
    """Biology-prior-guided graph attention over 18 proteins.

    Positive pairs (co-expression) and negative pairs (lineage exclusion)
    are encoded as attention biases, learned residually on top of the
    initial protein prediction from the image backbone.

    The pair lists are imported from `sphinx/protein_prior.py`, which is the
    single source of truth shared with `bio_prior_loss`, so the graph prior and
    the prior regularisation term always use the same relationships.
    """

    _PROTEIN_NAMES = PROTEIN_ORDER
    _POSITIVE_PAIRS = POSITIVE_PAIRS
    _NEGATIVE_PAIRS = NEGATIVE_PAIRS

    def __init__(self, num_proteins: int = 18, feat_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.num_proteins = num_proteins
        self.feat_dim = feat_dim
        self.num_heads = num_heads

        adj_bias = torch.tensor(build_bias_matrix(num_proteins))
        self.register_buffer('adj_bias', adj_bias)

        self.proj = nn.Linear(1, feat_dim)
        self.heads = nn.ModuleList([nn.Linear(feat_dim, feat_dim) for _ in range(num_heads)])
        self.attn_w = nn.ModuleList([nn.Linear(feat_dim * 2, 1) for _ in range(num_heads)])
        self.out_proj = nn.Linear(feat_dim * num_heads, 1)
        self.norm = nn.LayerNorm(num_proteins)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, protein_pred: torch.Tensor) -> torch.Tensor:
        B, P = protein_pred.shape
        x = self.proj(protein_pred.unsqueeze(-1))
        head_outs = []
        for h_idx in range(self.num_heads):
            hx = self.heads[h_idx](x)
            hx_i = hx.unsqueeze(2).expand(-1, -1, P, -1)
            hx_j = hx.unsqueeze(1).expand(-1, P, -1, -1)
            e = self.attn_w[h_idx](torch.cat([hx_i, hx_j], dim=-1)).squeeze(-1)
            e = e + self.adj_bias.unsqueeze(0)
            attn = F.softmax(e, dim=-1)
            head_outs.append(torch.matmul(attn, hx))
        agg = torch.cat(head_outs, dim=-1)
        delta = self.out_proj(agg).squeeze(-1)
        return protein_pred + self.residual_weight * self.norm(delta)


# ─────────────────────────────────────────────────────────────
#  CellGATBlock
# ─────────────────────────────────────────────────────────────

class CellGATBlock(nn.Module):
    """Pre-norm Transformer-style graph attention block.

    Operates on ALL nodes (center + K neighbors) simultaneously,
    enabling true multi-layer message passing when stacked.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.norm1 = nn.LayerNorm(dim)
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.Dropout(dropout),
        )

    def forward(self, nodes: torch.Tensor, adj_bias: torch.Tensor) -> torch.Tensor:
        """
        Args:
            nodes:    [B, N, dim]    — all node features (center + K neighbors)
            adj_bias: [B, 1, N, N]  — log-adjacency attention bias
        Returns:
            [B, N, dim] — updated node features
        """
        B, N, D = nodes.shape
        h = self.norm1(nodes)
        Q = self.W_q(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K_mat = self.W_k(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K_mat.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores + adj_bias
        attn = self.attn_drop(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, N, D)
        nodes = nodes + out
        nodes = nodes + self.ffn(self.norm2(nodes))
        return nodes


# ─────────────────────────────────────────────────────────────
#  Auxiliary modules
# ─────────────────────────────────────────────────────────────

class ProteinEncoder(nn.Module):
    """18-dim protein expression → 256-dim feature vector."""

    def __init__(self, num_proteins: int = 18, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(num_proteins),
            nn.Linear(num_proteins, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, out_dim), nn.GELU(), nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProteinMLPClassifier(nn.Module):
    """Path B: direct protein → cell-type classification (regularization branch)."""

    def __init__(self, protein_feat_dim: int = 256, num_celltypes: int = 12,
                 hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(protein_feat_dim),
            nn.Linear(protein_feat_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_celltypes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UncertaintyWeightedLoss(nn.Module):
    """Automatic multi-task loss balancing (Kendall et al., NeurIPS 2018).

    loss_total = Σ_i [ L_i / (2 σ_i²) + log σ_i ]

    init_log_sigma: initial log(σ) per task. Negative values give higher
    initial weight (e.g. -0.5 → effective weight ≈ 1.36 vs default 0.5).
    """

    def __init__(self, num_tasks: int = 4, init_log_sigma: Optional[List[float]] = None):
        super().__init__()
        init = torch.zeros(num_tasks)
        if init_log_sigma is not None:
            for i, v in enumerate(init_log_sigma):
                init[i] = v
        self.log_sigma = nn.Parameter(init)

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        total = torch.tensor(0.0, device=self.log_sigma.device, dtype=self.log_sigma.dtype)
        for i, loss in enumerate(losses):
            sigma2 = torch.exp(2.0 * self.log_sigma[i])
            total = total + loss / (2.0 * sigma2) + self.log_sigma[i]
        return total

    def get_weights(self) -> List[float]:
        with torch.no_grad():
            return [float(1.0 / (2.0 * torch.exp(2.0 * s).item())) for s in self.log_sigma]


# ─────────────────────────────────────────────────────────────
#  Main model: ViTProteinModel (SPHINX)
# ─────────────────────────────────────────────────────────────

class ViTProteinModel(nn.Module):
    """SPHINX: dual-ViT model for simultaneous protein expression prediction
    and cell type classification from H&E image patches.

    Forward pass outputs:
        protein_pred      [B, 18]   — predicted protein expression values
        protein_aux       [B, 18]   — deep supervision auxiliary prediction (training only)
        celltype_logits_A [B, 12]   — primary cell type logits (image + graph fusion)
        celltype_logits_B [B, 12]   — path B cell type logits (protein direct)
        router_logits     [B, 12]   — router auxiliary logits
        modality_gate     [B, 1]    — per-cell fusion gate (→1 image, →0 protein+graph)
        gate_per_type     [B, 12]   — per-cell-type gate weights (diagnostic)

    Input:
        detail_x    [B, 3, 224, 224] — 96px patch centered on cell, resized to 224
        context_x   [B, 3, 224, 224] — 192px context patch, resized to 224
        protein_gt  [B, 18]          — ground-truth proteins (training only, scheduled mixing)
        nb_protein  [B, K, 18]       — neighbor protein values for CellGAT
        adj         [B, 1+K, 1+K]    — spatial adjacency matrix (Gaussian distance weights)
    """

    def __init__(
        self,
        num_proteins: int = 18,
        num_celltypes: int = 12,
        drop_path_rate: float = 0.1,
        pretrained: bool = True,
        backbone: str = 'vit',
        ctranspath_weights: Optional[str] = None,
        use_protein_gat: bool = True,
        protein_enc_dim: int = 256,
        use_protein_mlp_cls: bool = True,
        use_deep_supervision: bool = True,
        use_uncertainty_loss: bool = True,
        gt_mix_ratio: float = 1.0,
        cell_gat_dim: int = 128,
        cell_gat_heads: int = 4,
        cell_gat_out: int = 256,
        num_gat_layers: int = 2,
        nb_protein_dropout: float = 0.3,
    ):
        super().__init__()
        self.num_proteins = num_proteins
        self.num_celltypes = num_celltypes
        self.use_protein_gat = use_protein_gat
        self.protein_enc_dim = protein_enc_dim
        self.use_protein_mlp_cls = use_protein_mlp_cls
        self.use_deep_supervision = use_deep_supervision
        self.use_uncertainty_loss = use_uncertainty_loss
        self.gt_mix_ratio = gt_mix_ratio
        self.nb_protein_dropout = nb_protein_dropout

        assert cell_gat_out == protein_enc_dim, (
            f"cell_gat_out({cell_gat_out}) must equal protein_enc_dim({protein_enc_dim})"
        )

        if backbone not in ('vit', 'ctranspath'):
            raise ValueError(
                f"Unknown backbone '{backbone}'. Choose 'vit' or 'ctranspath'."
            )
        self.backbone_type = backbone

        # ── Dual image backbones ──
        # Both branches encode the same cell but at different spatial scales:
        #   detail  branch: 96 px patch  → fine cellular morphology
        #   context branch: 192 px patch → wider tissue architecture
        if backbone == 'ctranspath':
            # CTransPath Swin-Tiny outputs 768-d features (same as ViT-Base)
            img_dim = 768
            self.detail_backbone = _build_ctranspath_backbone(
                pretrained, ctranspath_weights, drop_path_rate
            )
            self.context_backbone = _build_ctranspath_backbone(
                pretrained, ctranspath_weights, drop_path_rate
            )
        else:
            img_dim = 768  # ViT-Base/16 output dimension
            self.detail_backbone = timm.create_model(
                'vit_base_patch16_224', pretrained=pretrained, num_classes=0,
                drop_path_rate=drop_path_rate,
            )
            self.context_backbone = timm.create_model(
                'vit_base_patch16_224', pretrained=pretrained, num_classes=0,
                drop_path_rate=drop_path_rate,
            )

        # Learned fusion: sigmoid(0)=0.5, neutral initialisation
        self.fusion_alpha = nn.Parameter(torch.tensor(0.0))
        self.fusion_refine = nn.Sequential(
            nn.LayerNorm(img_dim),
            nn.Linear(img_dim, img_dim), nn.GELU(), nn.Dropout(0.1),
        )

        # ── Protein prediction head (image only, no GT leakage) ──
        self.protein_head = nn.Sequential(
            nn.LayerNorm(img_dim),
            nn.Linear(img_dim, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
        )
        self.protein_output = nn.Linear(256, num_proteins)

        if use_deep_supervision:
            self.protein_aux_head = nn.Sequential(
                nn.LayerNorm(img_dim),
                nn.Linear(img_dim, 128), nn.GELU(),
                nn.Linear(128, num_proteins),
            )

        if use_protein_gat:
            self.protein_gat = ProteinGAT(num_proteins=num_proteins, feat_dim=64, num_heads=4)

        # ── Cell typing path ──
        self.protein_encoder = ProteinEncoder(
            num_proteins=num_proteins, out_dim=protein_enc_dim, dropout=0.1,
        )

        # CellGAT: projection layers + stackable attention blocks
        self.cell_gat_center_proj = nn.Sequential(
            nn.LayerNorm(protein_enc_dim),
            nn.Linear(protein_enc_dim, cell_gat_dim),
        )
        self.cell_gat_nb_proj = nn.Sequential(
            nn.LayerNorm(num_proteins),
            nn.Linear(num_proteins, 64), nn.GELU(),
            nn.Linear(64, cell_gat_dim),
        )
        self.cell_gat_layers = nn.ModuleList([
            CellGATBlock(cell_gat_dim, cell_gat_heads, dropout=0.1)
            for _ in range(num_gat_layers)
        ])
        self.cell_gat_out_proj = nn.Sequential(
            nn.Linear(cell_gat_dim, cell_gat_out), nn.GELU(), nn.Dropout(0.1),
        )
        self.cell_gat_out_norm = nn.LayerNorm(cell_gat_out)

        # Multi-Gate: per-cell-type independent gate (→1 image, →0 protein+graph)
        self.multi_gate = nn.Sequential(
            nn.Linear(img_dim + cell_gat_out, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, num_celltypes),
        )

        # Router: coarse cell-type prediction from image for soft gate selection
        self.ct_router = nn.Sequential(
            nn.LayerNorm(img_dim),
            nn.Linear(img_dim, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, num_celltypes),
        )

        # Project GAT output to image feature dimension for gate fusion
        self.gat_to_img_proj = nn.Sequential(
            nn.LayerNorm(cell_gat_out),
            nn.Linear(cell_gat_out, img_dim),
        )

        # Layer norms to equalise scale before gate fusion
        self.gate_img_norm  = nn.LayerNorm(img_dim)
        self.gate_prot_norm = nn.LayerNorm(img_dim)

        self.celltype_head = nn.Sequential(
            nn.LayerNorm(img_dim),
            nn.Linear(img_dim, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, num_celltypes),
        )

        if use_protein_mlp_cls:
            self.protein_mlp_cls = ProteinMLPClassifier(
                protein_feat_dim=protein_enc_dim, num_celltypes=num_celltypes,
                hidden_dim=256, dropout=0.1,
            )

        # UncertaintyWeightedLoss: tasks = [prot, ctA, router, ctB(optional)]
        if use_uncertainty_loss:
            n_tasks = 4 if use_protein_mlp_cls else 3
            init_ls = [-0.5] + [0.0] * (n_tasks - 1)  # protein task gets higher initial weight
            self.uncertainty_loss = UncertaintyWeightedLoss(
                num_tasks=n_tasks, init_log_sigma=init_ls)

        # Gate statistics buffers (diagnostic, not used in loss)
        self.register_buffer('_alpha_sum',    torch.tensor(0.0))
        self.register_buffer('_alpha_count',  torch.tensor(0))
        self.register_buffer('_gate_sum',     torch.tensor(0.0))
        self.register_buffer('_gate_count',   torch.tensor(0))

    def forward(
        self,
        detail_x: torch.Tensor,
        context_x: torch.Tensor,
        protein_gt: Optional[torch.Tensor] = None,
        protein_mask: Optional[torch.Tensor] = None,
        nb_protein: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        # ── Dual-branch ViT encoding ──
        detail_feat  = self.detail_backbone(detail_x)    # [B, 768]
        context_feat = self.context_backbone(context_x)  # [B, 768]

        alpha = torch.sigmoid(self.fusion_alpha)
        fused = alpha * detail_feat + (1.0 - alpha) * context_feat
        fused = self.fusion_refine(fused) + fused        # [B, 768]

        if self.training:
            self._alpha_sum   += alpha.detach().float()
            self._alpha_count += 1

        # ── Protein prediction (image only) ──
        protein_hidden = self.protein_head(fused)
        protein_pred   = self.protein_output(protein_hidden)  # [B, 18]

        out: Dict[str, torch.Tensor] = {'protein_pred': protein_pred}

        if self.use_deep_supervision:
            out['protein_aux'] = self.protein_aux_head(fused)

        if self.use_protein_gat:
            protein_pred = self.protein_gat(protein_pred)
            out['protein_pred'] = protein_pred

        # ── Cell typing path ──
        # Scheduled mixing: blend GT and predicted proteins during training.
        # protein_mask [B, 18]: 1 = protein present, 0 = missing panel.
        # For missing proteins we always use protein_pred to avoid the -5.0
        # MISSING_PROTEIN_SENTINEL leaking into ProteinEncoder input.
        if protein_gt is not None and self.training:
            r = self.gt_mix_ratio
            mixed = r * protein_gt.detach() + (1.0 - r) * protein_pred.detach()
            if protein_mask is not None:
                # Replace missing-protein positions with pure prediction
                center_protein = torch.where(protein_mask.bool(), mixed,
                                             protein_pred.detach())
            else:
                center_protein = mixed
        else:
            center_protein = protein_pred.detach()

        protein_enc_feat = self.protein_encoder(center_protein)  # [B, 256]

        # CellGAT: multi-layer message passing over spatial neighborhood
        gat_feat = protein_enc_feat
        if nb_protein is not None and adj is not None:
            nb_prot = nb_protein
            if self.training and self.nb_protein_dropout > 0:
                nb_prot = F.dropout(nb_prot, p=self.nb_protein_dropout, training=True)

            center_h = self.cell_gat_center_proj(protein_enc_feat).unsqueeze(1)  # [B, 1, D]
            nb_h     = self.cell_gat_nb_proj(nb_prot)                            # [B, K, D]
            nodes    = torch.cat([center_h, nb_h], dim=1)                        # [B, N, D]
            adj_bias = torch.log(adj.unsqueeze(1).clamp(min=1e-6))               # [B, 1, N, N]

            for layer in self.cell_gat_layers:
                nodes = layer(nodes, adj_bias)

            center_out = self.cell_gat_out_proj(nodes[:, 0, :])
            gat_feat   = self.cell_gat_out_norm(center_out + protein_enc_feat)   # residual

        # Multi-Gate + Router: soft routing over per-type gates
        gat_proj      = self.gat_to_img_proj(gat_feat)
        gate_per_type = torch.sigmoid(self.multi_gate(torch.cat([fused, gat_feat], dim=1)))
        router_logits = self.ct_router(fused.detach())
        ct_probs      = F.softmax(router_logits, dim=1)
        gate          = (ct_probs.detach() * gate_per_type).sum(dim=1, keepdim=True)  # [B, 1]

        fused_normed    = self.gate_img_norm(fused)
        gat_proj_normed = self.gate_prot_norm(gat_proj)
        celltype_input  = gate * fused_normed + (1.0 - gate) * gat_proj_normed

        out['celltype_logits_A'] = self.celltype_head(celltype_input)
        out['router_logits']     = router_logits
        out['modality_gate']     = gate.detach()
        out['gate_per_type']     = gate_per_type.detach()

        if self.training:
            self._gate_sum   += gate.detach().mean().float()
            self._gate_count += 1

        if self.use_protein_mlp_cls:
            out['celltype_logits_B'] = self.protein_mlp_cls(protein_enc_feat)

        return out

    @torch.no_grad()
    def predict_proteins_only(
        self, detail_x: torch.Tensor, context_x: torch.Tensor
    ) -> torch.Tensor:
        """First pass of two-pass inference: predict proteins for all cells."""
        detail_feat  = self.detail_backbone(detail_x)
        context_feat = self.context_backbone(context_x)
        alpha = torch.sigmoid(self.fusion_alpha)
        fused = alpha * detail_feat + (1.0 - alpha) * context_feat
        fused = self.fusion_refine(fused) + fused
        protein_pred = self.protein_output(self.protein_head(fused))
        if self.use_protein_gat:
            protein_pred = self.protein_gat(protein_pred)
        return protein_pred

    def get_gate_stats(self) -> Dict[str, float]:
        avg_alpha = float(self._alpha_sum / self._alpha_count) \
            if self._alpha_count > 0 else 0.5
        avg_gate  = float(self._gate_sum  / self._gate_count)  \
            if self._gate_count  > 0 else 0.5
        return {'avg_alpha': avg_alpha, 'avg_modgate': avg_gate}

    def reset_gate_stats(self):
        self._alpha_sum.zero_();  self._alpha_count.zero_()
        self._gate_sum.zero_();   self._gate_count.zero_()
