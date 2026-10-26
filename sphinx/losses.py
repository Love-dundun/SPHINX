"""
SPHINX loss functions.

This module defines all training objectives used by the SPHINX model:

  masked_mse_loss     — protein regression loss, ignoring missing channels
  pearson_loss        — per-cell Pearson correlation penalty for protein outputs
  focal_loss          — class-imbalance-robust cell-type classification loss with
                        optional absent-class masking and label smoothing
  bio_prior_loss      — biology-informed regularisation based on known protein
                        co-expression and mutual-exclusion relationships
  compute_class_weights — inverse-frequency class weights for focal_loss

The protein co-expression (POS_PAIRS) and mutual-exclusion (NEG_PAIRS) index
tensors are derived from the single canonical prior defined in
``sphinx/protein_prior.py`` — the same source that builds ProteinGAT's
attention bias — so the graph prior and the regularisation term cannot diverge.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .dataset import CELLTYPE_COLS, CELLTYPE_NAMES
from .protein_prior import LOSS_NEGATIVE_INDEX_PAIRS, LOSS_POSITIVE_INDEX_PAIRS


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error loss that ignores channels with missing ground-truth.

    Proteins that were not measured for a given sample have mask=0 and are
    excluded from both numerator and denominator, so the loss remains
    unbiased regardless of which protein channels are present.

    Args:
        pred:   [B, P] — predicted protein expression values.
        target: [B, P] — ground-truth protein expression values.
        mask:   [B, P] — binary validity mask (1 = measured, 0 = missing).

    Returns:
        Scalar MSE loss averaged over all valid (cell, protein) pairs.
    """
    diff = (pred - target) ** 2
    valid = mask.sum().clamp(min=1)
    return (diff * mask).sum() / valid


def pearson_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-cell Pearson correlation loss over the measured protein channels.

    Minimising (1 - r) encourages the model to preserve the relative ranking
    of protein expression values within each cell, complementing the MSE loss
    which focuses on absolute expression magnitudes.

    Cells with fewer than 2 valid protein channels are skipped.  Returns 0.0
    if no cells in the batch have enough valid channels.

    Args:
        pred:   [B, P] — predicted protein expression values.
        target: [B, P] — ground-truth protein expression values.
        mask:   [B, P] — binary validity mask (1 = measured, 0 = missing).

    Returns:
        Scalar mean (1 - Pearson r) over valid cells.
    """
    losses = []
    for i in range(pred.size(0)):
        m = mask[i].bool()
        if m.sum() < 2:
            continue
        p, t = pred[i][m], target[i][m]
        vp = p - p.mean()
        vt = t - t.mean()
        denom = (vp.norm() * vt.norm()).clamp(min=1e-8)
        losses.append(1.0 - (vp * vt).sum() / denom)
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=pred.device)


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.05,
    ct_valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if ct_valid_mask is not None:
        absent = ct_valid_mask < 0.5
        shift = logits.detach().min(dim=1, keepdim=True).values - 100.0
        logits = logits + absent.float() * (shift - logits.detach())

        if label_smoothing > 0:
            n_valid = ct_valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
            log_probs = F.log_softmax(logits, dim=1)
            one_hot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1.0)
            smooth_tgt = (1.0 - label_smoothing) * one_hot + label_smoothing / n_valid * ct_valid_mask
            ce = -(smooth_tgt * log_probs).sum(dim=1)
            if weight is not None:
                ce = ce * weight[labels]
            pt = torch.exp(-ce.clamp(max=80))
            return ((1.0 - pt) ** gamma * ce).mean()

        ce = F.cross_entropy(logits, labels, weight=weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** gamma * ce).mean()

    ce = F.cross_entropy(logits, labels, weight=weight, label_smoothing=label_smoothing, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt) ** gamma * ce).mean()


def bio_prior_loss(
    protein_pred: torch.Tensor,
    pos_pairs: torch.Tensor,
    neg_pairs: torch.Tensor,
) -> torch.Tensor:
    """Biology-informed protein co-expression regularisation loss.

    Penalises predictions that violate the documented protein co-expression
    (positive) and mutual-exclusion (negative) relationships.  The pairs come
    from `sphinx/protein_prior.py`, the same source that builds the attention
    bias of ProteinGAT, so the loss and the graph prior always agree.

    Args:
        protein_pred: [B, P] — predicted protein expression values.
        pos_pairs:    [N, 2] — index pairs that should co-express.
        neg_pairs:    [M, 2] — index pairs that should be mutually exclusive.

    Returns:
        Scalar mean of the positive and negative cosine similarity penalties.
    """
    pos_pairs = pos_pairs.to(protein_pred.device)
    neg_pairs = neg_pairs.to(protein_pred.device)
    p = protein_pred
    pos_loss = -F.cosine_similarity(p[:, pos_pairs[:, 0]], p[:, pos_pairs[:, 1]], dim=0).mean().clamp(min=-1)
    neg_loss = F.cosine_similarity(p[:, neg_pairs[:, 0]], p[:, neg_pairs[:, 1]], dim=0).mean().clamp(min=-1)
    return (pos_loss + neg_loss) * 0.5


# ─────────────────────────────────────────────────────────────
# Protein pair index tensors for bio_prior_loss.
#
# These are NOT maintained here any more: the documented co-expression and
# mutual-exclusion pairs live in `sphinx/protein_prior.py`, which is the single
# source of truth shared with ProteinGAT (sphinx/model.py).  Both consumers used
    # Protein pairs are imported from protein_prior, so the loss and the
    # graph attention always use the same relationships.

POS_PAIRS = torch.tensor(LOSS_POSITIVE_INDEX_PAIRS, dtype=torch.long)

# Mutual-exclusion pairs: proteins from distinct cell lineages rarely
# co-expressed in the same cell (e.g. epithelial PanCK vs. T-cell CD3).
NEG_PAIRS = torch.tensor(LOSS_NEGATIVE_INDEX_PAIRS, dtype=torch.long)


def compute_class_weights(sample_dirs: list[str], device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights for focal_loss from training data.

    Reads the ground-truth one-hot cell-type columns from each sample's
    cell table and accumulates per-class counts.  Weights are (1/freq)^0.15,
    normalised to mean=1.0.  The mild exponent softens the correction so rare
    classes receive a boost without overwhelming the gradient from common ones.

    Args:
        sample_dirs: List of sample directory paths, each containing
                     ``sphinx_cell_table.csv``.
        device:      Target device for the returned weight tensor.

    Returns:
        FloatTensor of shape [num_celltypes] on *device*.
    """
    counts = np.zeros(len(CELLTYPE_COLS), dtype=np.float64)
    for sample_dir in sample_dirs:
        csv_path = os.path.join(sample_dir, "sphinx_cell_table.csv")
        df = pd.read_csv(csv_path, usecols=CELLTYPE_COLS)
        labels = np.argmax(df[CELLTYPE_COLS].values, axis=1)
        for class_idx in range(len(CELLTYPE_COLS)):
            counts[class_idx] += (labels == class_idx).sum()

    freq = np.clip(counts / counts.sum(), 1e-6, None)
    weights = (1.0 / freq) ** 0.15
    weights /= weights.mean()

    print("Class weights (freq^-0.15, mean=1.0):")
    for i, name in enumerate(CELLTYPE_NAMES):
        print(f"  {name:>22s}: freq={freq[i] * 100:>5.1f}%  w={weights[i]:.3f}")
    return torch.tensor(weights, dtype=torch.float32, device=device)
