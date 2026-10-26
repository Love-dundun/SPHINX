"""
Dataset for SPHINX: single-sample cell patch loading with adaptive protein normalisation.

Data layout (per sample directory):
    sphinx_cell_table.csv  — cell table with coordinates, proteins, cell types
    {sample}_HE.ome.tiff             — H&E whole-slide image (OME-TIFF, zarr-accessible)

Normalisation strategy (per protein column):
    Rank-norm proteins (dynamic range too compressed for log1p):
        ATPase, CD11c, CD56, FOXP3, CTLA4, KI67, PD1
        → percentile rank mapped to [-2, +2] using training-set edges
    Remaining 11 proteins:
        → log1p then z-score using training-set mean/std

Patch extraction:
    detail  patch: 96×96 px centred on cell → resized to 224×224 (ViT input)
    context patch: 192×192 px centred on cell → resized to 224×224 (wider spatial context)
    Lazy zarr reads per worker (fork-safe, zero pre-caching for validation set).

Neighbourhood graph (CellGAT):
    K=6 nearest spatial neighbours per cell (cKDTree).
    Adjacency weights: Gaussian(distance, σ=30 px), pre-computed at dataset init.
"""

import gc
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import tifffile
import zarr
from typing import Optional, Tuple, Dict
from scipy.spatial import cKDTree
import cv2


# ─────────────────────────────────────────────────────────────
#  Column name constants
# ─────────────────────────────────────────────────────────────

PROTEIN_COLS = [
    'ATPase_intensity_mean', 'CD11c_intensity_mean', 'CD20_intensity_mean',
    'CD3_intensity_mean',    'CD31_intensity_mean',  'CD4_intensity_mean',
    'CD45RA_intensity_mean', 'CD56_intensity_mean',  'CD68_intensity_mean',
    'CD8_intensity_mean',    'CTLA4_intensity_mean', 'FOXP3_intensity_mean',
    'HLA-DR_intensity_mean', 'KI67_intensity_mean',  'PD1_intensity_mean',
    'PDL1_intensity_mean',   'PanCK_intensity_mean', 'aSMA_intensity_mean',
]  # 18 proteins, alphabetical order

CELLTYPE_COLS = [
    'celltype_B_Cells',           'celltype_Cancer_Cells',      'celltype_Dendritic_Cells',
    'celltype_Endothelial_Cells', 'celltype_Fibroblasts',       'celltype_Macrophages',
    'celltype_Monocytes',         'celltype_NK_Cells',          'celltype_Neutrophils',
    'celltype_Smooth_Muscle_Cells','celltype_T_Cells',          'celltype_Epithelial_Cells',
]  # 12 cell types, one-hot columns

CELLTYPE_NAMES = [
    'B_Cells', 'Cancer_Cells', 'Dendritic_Cells', 'Endothelial_Cells',
    'Fibroblasts', 'Macrophages', 'Monocytes', 'NK_Cells', 'Neutrophils',
    'Smooth_Muscle_Cells', 'T_Cells', 'Epithelial_Cells',
]  # index matches CELLTYPE_COLS

MORPH_COLS = [
    'morph_area', 'morph_perimeter', 'morph_eccentricity', 'morph_solidity',
    'morph_extent', 'morph_major_axis', 'morph_minor_axis', 'morph_diameter',
    'morph_euler_number', 'morph_feret_diameter', 'morph_aspect_ratio',
    'morph_circularity', 'morph_perimeter_complexity',
]  # 13 morphological features

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
#  Adaptive protein normalisation
# ─────────────────────────────────────────────────────────────

# Proteins with compressed dynamic range that benefit from rank normalisation.
# Determined from 80-sample population statistics (see paper §Methods).
_RANK_NORM_PROTEINS = {
    'ATPase_intensity_mean',   # zero%=30%, log1p(p99)=4.27
    'CD11c_intensity_mean',    # log1p(p99)=5.31, signal too weak
    'CD56_intensity_mean',     # log1p(p99)=3.03, extremely weak
    'FOXP3_intensity_mean',    # log1p(p99)=4.12, extremely weak
    'CTLA4_intensity_mean',    # zero%=18%, log1p(p99)=4.75
    'KI67_intensity_mean',     # p99/p50=110×, highly skewed
    'PD1_intensity_mean',      # p99/p50=157×, highly skewed
}


def compute_norm_params(protein_raw: np.ndarray, col_names: list) -> dict:
    """Compute per-column normalisation parameters from training-set protein matrix.

    Args:
        protein_raw: [N, 18] float32, may contain NaN for missing panel proteins
        col_names:   list of 18 column names (must match PROTEIN_COLS order)

    Returns dict with keys:
        use_rank   : np.ndarray[bool, 18]
        prot_mean  : np.ndarray[float32, 18]  (log1p mean, used for log1p columns)
        prot_std   : np.ndarray[float32, 18]  (log1p std,  used for log1p columns)
        rank_edges : list[np.ndarray|None]     (101 percentile edges, used for rank columns)
    """
    P = protein_raw.shape[1]
    use_rank   = np.zeros(P, dtype=bool)
    prot_mean  = np.zeros(P, dtype=np.float32)
    prot_std   = np.ones(P,  dtype=np.float32)
    rank_edges = [None] * P

    for i, col in enumerate(col_names):
        vals = protein_raw[:, i]
        valid = vals[~np.isnan(vals)]
        if len(valid) < 10:
            continue
        if col in _RANK_NORM_PROTEINS:
            use_rank[i] = True
            rank_edges[i] = np.percentile(valid, np.arange(0, 101)).astype(np.float32)
        else:
            log_vals = np.log1p(np.clip(valid, 0, None))
            prot_mean[i] = float(log_vals.mean())
            prot_std[i]  = float(log_vals.std()) + 1e-8

    return {
        'use_rank':   use_rank,
        'prot_mean':  prot_mean,
        'prot_std':   prot_std,
        'rank_edges': rank_edges,
    }


MISSING_PROTEIN_SENTINEL = -5.0
"""Sentinel value for proteins absent from a sample's panel.

Value -5.0 is far outside the normal normalised range ([-2, +2] for rank-norm,
roughly [-3, +3] for log1p z-score). This lets ProteinEncoder / CellGAT learn
to distinguish "missing panel" from "true zero / average expression", preventing
gradient interference when batches mix 15-protein and 18-protein samples.
The loss functions use protein_mask=0 to exclude missing proteins from training
signal regardless of this sentinel value.
"""


def apply_norm(protein_raw: np.ndarray, norm_params: dict) -> np.ndarray:
    """Apply pre-computed normalisation to a protein matrix.

    Args:
        protein_raw: [N, 18] float32, may contain NaN
        norm_params: dict from compute_norm_params()

    Returns:
        [N, 18] float32, normalised; missing columns filled with
        MISSING_PROTEIN_SENTINEL (-5.0) to distinguish from real values.
    """
    N, P = protein_raw.shape
    out        = np.full((N, P), MISSING_PROTEIN_SENTINEL, dtype=np.float32)
    use_rank   = norm_params['use_rank']
    prot_mean  = norm_params['prot_mean']
    prot_std   = norm_params['prot_std']
    rank_edges = norm_params['rank_edges']

    for i in range(P):
        col  = protein_raw[:, i].copy()
        mask = ~np.isnan(col)
        if mask.sum() == 0:
            continue
        if use_rank[i]:
            edges = rank_edges[i]                               # [101]
            pct   = np.searchsorted(edges, col[mask], side='left').clip(0, 100)
            out[mask, i] = (pct / 25.0 - 2.0).astype(np.float32)  # → [-2, +2]
        else:
            log_vals = np.log1p(np.clip(col[mask], 0, None))
            out[mask, i] = ((log_vals - prot_mean[i]) / prot_std[i]).astype(np.float32)

    return out


# ─────────────────────────────────────────────────────────────
#  Patch cropping helpers
# ─────────────────────────────────────────────────────────────

def _crop_patch(src, cy: float, cx: float, size: int, h: int, w: int) -> np.ndarray:
    """Crop a square patch from src (zarr array or ndarray stripe).

    Handles boundary clipping with zero-padding.
    src may be a zarr array (index-based) or a pre-read numpy stripe.
    """
    half = size // 2
    y0 = int(round(cy)) - half;  x0 = int(round(cx)) - half
    y1 = y0 + size;               x1 = x0 + size
    vy0, vx0 = max(0, y0), max(0, x0)
    vy1, vx1 = min(h, y1),  min(w, x1)
    if vy0 >= vy1 or vx0 >= vx1:
        return np.zeros((size, size, 3), dtype=np.uint8)
    patch_valid = np.array(src[vy0:vy1, vx0:vx1, :])
    if vy0 != y0 or vx0 != x0 or vy1 != y1 or vx1 != x1:
        patch = np.zeros((size, size, 3), dtype=np.uint8)
        patch[vy0 - y0:vy0 - y0 + (vy1 - vy0),
              vx0 - x0:vx0 - x0 + (vx1 - vx0), :] = patch_valid
        return patch
    return patch_valid.copy()


# ─────────────────────────────────────────────────────────────
#  SingleSampleDataset
# ─────────────────────────────────────────────────────────────

class SingleSampleDataset(Dataset):
    """Cell-patch dataset for one WSI sample.

    Args:
        sample_dir:   path to sample directory
        indices:      cell row indices to use (train or val subset)
        norm_params:  protein normalisation dict from compute_norm_params()
        morph_mean:   [13] morphology z-score mean
        morph_std:    [13] morphology z-score std
        detail_size:  source crop size for detail patch (default 96 px)
        context_size: source crop size for context patch (default 192 px)
        target_size:  resize target for both patches (default 224 px, ViT input)
        augment:      apply random flips/rotations and colour jitter (training only)
        num_neighbors: K for KNN neighbourhood graph (default 6)
    """

    def __init__(
        self,
        sample_dir: str,
        indices: np.ndarray,
        norm_params: dict,
        morph_mean: np.ndarray,
        morph_std: np.ndarray,
        detail_size: int = 96,
        context_size: int = 192,
        target_size: int = 224,
        augment: bool = False,
        num_neighbors: int = 6,
    ):
        self.sample_dir   = sample_dir
        self.indices      = indices
        self.norm_params  = norm_params
        self.detail_size  = detail_size
        self.context_size = context_size
        self.target_size  = target_size
        self.augment      = augment
        self.num_neighbors = num_neighbors

        csv_path = os.path.join(sample_dir, 'sphinx_cell_table.csv')
        # Load FULL CSV — we keep full arrays so KNN global indices are valid.
        df_full = pd.read_csv(csv_path)
        N_full  = len(df_full)
        df      = df_full.iloc[indices]   # subset view for labels/proteins/morphs

        # Full-sample coordinates (needed for adj computation via global KNN indices)
        self.centroids_y_full = df_full['centroid_y'].values.astype(np.float32)  # [N_full]
        self.centroids_x_full = df_full['centroid_x'].values.astype(np.float32)  # [N_full]
        # Subset coordinates (used for image crop positions in __getitem__)
        self.centroids_y = self.centroids_y_full[indices]
        self.centroids_x = self.centroids_x_full[indices]

        # Protein normalisation (NaN = panel missing → mask=0 → excluded from loss)
        protein_raw = np.full((len(indices), len(PROTEIN_COLS)), np.nan, dtype=np.float32)
        for i, col in enumerate(PROTEIN_COLS):
            if col in df.columns:
                protein_raw[:, i] = df[col].values.astype(np.float32)
        self.protein_mask = (~np.isnan(protein_raw)).astype(np.float32)
        self.proteins     = apply_norm(protein_raw, norm_params)

        # Full-sample normalised proteins (needed when neighbours are outside the subset)
        protein_raw_full = np.full((N_full, len(PROTEIN_COLS)), np.nan, dtype=np.float32)
        for i, col in enumerate(PROTEIN_COLS):
            if col in df_full.columns:
                protein_raw_full[:, i] = df_full[col].values.astype(np.float32)
        self.proteins_full = apply_norm(protein_raw_full, norm_params)  # [N_full, 18]

        # Sample-level protein panel
        self.protein_panel = (self.protein_mask.sum(axis=0) > 0).astype(np.float32)  # [18]

        # Cell type labels
        ct_onehot = np.zeros((len(indices), len(CELLTYPE_COLS)), dtype=np.float32)
        for i, col in enumerate(CELLTYPE_COLS):
            if col in df.columns:
                ct_onehot[:, i] = df[col].values.astype(np.float32)
        self.celltype_labels = np.argmax(ct_onehot, axis=1).astype(np.int64)

        self.celltype_valid_mask = np.zeros(len(CELLTYPE_COLS), dtype=np.float32)  # [12]
        for cls_idx in np.unique(self.celltype_labels):
            self.celltype_valid_mask[cls_idx] = 1.0
        self._ct_valid_tensor = torch.from_numpy(self.celltype_valid_mask)  # [12]

        # Morphology (z-score)
        morph_raw    = df[MORPH_COLS].values.astype(np.float32)
        self.morphs  = np.nan_to_num(
            (morph_raw - morph_mean) / (morph_std + 1e-8), nan=0.0
        ).astype(np.float32)

        # KNN neighbourhood — built on FULL sample, cached to disk.
        # Indices are GLOBAL (into full CSV), so neighbours are always valid.
        knn_cache = os.path.join(sample_dir, f'knn_k{num_neighbors}.npz')
        if os.path.exists(knn_cache):
            _c = np.load(knn_cache)
            full_knn_idx  = _c['knn_idx']   # [N_full, K]
            full_knn_dist = _c['knn_dist']  # [N_full, K]
        else:
            _coords_full = np.stack([self.centroids_y_full,
                                     self.centroids_x_full], axis=1)
            _tree  = cKDTree(_coords_full)
            _dists, _idx = _tree.query(_coords_full, k=num_neighbors + 1, workers=-1)
            full_knn_idx  = _idx[:, 1:].astype(np.int64)
            full_knn_dist = _dists[:, 1:].astype(np.float32)
            np.savez_compressed(knn_cache,
                                knn_idx=full_knn_idx, knn_dist=full_knn_dist)
        # Take rows for the subset — indices stay GLOBAL (no remap needed)
        self.neighbor_indices = full_knn_idx[indices]    # [N_subset, K]  global indices
        self.neighbor_dists   = full_knn_dist[indices]   # [N_subset, K]
        del df_full, protein_raw_full  # free; full arrays retained via self.* refs above
        gc.collect()

        # Adjacency matrix weights: stored as compact per-cell arrays, built on-demand
        # in __getitem__ to avoid the O(N*K*K) memory cost (~197 GB for 1M cells).
        # We pre-compute only center→neighbour weights [N, K] which are already stored
        # in neighbor_dists.  Neighbour-neighbour weights are computed per cell in __getitem__.
        self._adj_sigma = 30.0  # Gaussian sigma (pixels)

        # TIFF path and image dimensions (lazy zarr per worker)
        tiff_files = [f for f in os.listdir(sample_dir) if f.endswith('_HE.ome.tiff')]
        assert len(tiff_files) == 1, f"Expected 1 TIFF, found {len(tiff_files)} in {sample_dir}"
        self._tiff_path = os.path.join(sample_dir, tiff_files[0])
        _t = tifffile.TiffFile(self._tiff_path)
        _z = zarr.open(_t.pages[0].aszarr(), mode='r')
        self._img_h, self._img_w = _z.shape[:2]
        _t.close()
        self._tiff = None   # opened lazily per DataLoader worker
        self._zarr = None

    # ── Lazy zarr init (fork-safe) ──────────────────────────

    def _ensure_zarr(self):
        if self._zarr is None:
            self._tiff = tifffile.TiffFile(self._tiff_path)
            self._zarr = zarr.open(self._tiff.pages[0].aszarr(), mode='r')

    def close(self):
        if self._tiff is not None:
            self._tiff.close()

    # ── Patch preprocessing ─────────────────────────────────

    def _preprocess(self, patch: np.ndarray, aug: Optional[dict]) -> torch.Tensor:
        img = cv2.resize(patch, (self.target_size, self.target_size),
                         interpolation=cv2.INTER_LINEAR)
        if self.augment and aug is not None:
            if aug['hflip']:
                img = img[:, ::-1, :].copy()
            if aug['vflip']:
                img = img[::-1, :, :].copy()
            k = aug['rot90']
            if k:
                img = np.rot90(img, k).copy()
            img_f  = img.astype(np.float32)
            img_f *= (1.0 + (np.random.random() - 0.5) * 0.2)       # brightness
            mean_v = img_f.mean()
            img_f  = (img_f - mean_v) * (1.0 + (np.random.random() - 0.5) * 0.2) + mean_v
            img    = np.clip(img_f, 0, 255).astype(np.uint8)
        arr = img.astype(np.float32) * (1.0 / 255.0)
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(arr.transpose(2, 0, 1))  # [3, 224, 224]

    def _build_adj(self, idx: int) -> torch.Tensor:
        """Build [1+K, 1+K] adjacency matrix on-the-fly for a single cell.
        Memory cost: O(K^2) per call instead of O(N*K^2) pre-allocation.
        """
        K      = self.num_neighbors
        Nn     = 1 + K
        sigma  = self._adj_sigma
        nb_idx = self.neighbor_indices[idx]          # [K]
        d_c2n  = self.neighbor_dists[idx]            # [K]  center→neighbour distances

        adj = np.eye(Nn, dtype=np.float32)           # identity diagonal

        # Center ↔ neighbour weights
        center_w = np.exp(-(d_c2n ** 2) / (sigma ** 2)).astype(np.float32)
        adj[0, 1:] = center_w
        adj[1:, 0] = center_w

        # Neighbour ↔ neighbour weights  (nb_idx are GLOBAL indices)
        nb_y = self.centroids_y_full[nb_idx]         # [K]
        nb_x = self.centroids_x_full[nb_idx]         # [K]
        nb_coords = np.stack([nb_y, nb_x], axis=-1)  # [K, 2]
        diff  = nb_coords[:, np.newaxis, :] - nb_coords[np.newaxis, :, :]  # [K, K, 2]
        nb_w  = np.exp(-(diff ** 2).sum(axis=-1) / (sigma ** 2)).astype(np.float32)
        nb_w[nb_w < 0.01] = 0.0
        adj[1:, 1:] = nb_w

        return torch.from_numpy(adj)                 # [1+K, 1+K]

    # ── Dataset interface ────────────────────────────────────

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        self._ensure_zarr()
        cy, cx = float(self.centroids_y[idx]), float(self.centroids_x[idx])
        detail_patch  = _crop_patch(self._zarr, cy, cx, self.detail_size,
                                    self._img_h, self._img_w)
        context_patch = _crop_patch(self._zarr, cy, cx, self.context_size,
                                    self._img_h, self._img_w)

        aug = None
        if self.augment:
            aug = {
                'hflip': np.random.random() > 0.5,
                'vflip': np.random.random() > 0.5,
                'rot90': int(np.random.randint(0, 4)),
            }

        detail_t  = self._preprocess(detail_patch,  aug)
        context_t = self._preprocess(context_patch, aug)

        nb_idx = self.neighbor_indices[idx]   # [K]

        return {
            'detail':              detail_t,                                      # [3, 224, 224]
            'context':             context_t,                                     # [3, 224, 224]
            'protein':             torch.from_numpy(self.proteins[idx]),          # [18]
            'protein_mask':        torch.from_numpy(self.protein_mask[idx]),      # [18]
            'celltype':            torch.tensor(self.celltype_labels[idx], dtype=torch.long),
            'celltype_valid_mask': self._ct_valid_tensor,                         # [12] sample-level
            'morph':               torch.from_numpy(self.morphs[idx]),            # [13]
            'nb_protein':          torch.from_numpy(self.proteins_full[nb_idx]),  # [K, 18]
            'nb_dist':             torch.from_numpy(self.neighbor_dists[idx]),    # [K]
            'adj':                 self._build_adj(idx),                          # [1+K, 1+K]
        }


# ─────────────────────────────────────────────────────────────
#  Convenience: prepare train/val split for one sample
# ─────────────────────────────────────────────────────────────

def prepare_sample(
    sample_dir: str,
    train_ratio: float = 0.7,
    seed: int = 42,
    max_cells: Optional[int] = None,
) -> Tuple['SingleSampleDataset', 'SingleSampleDataset', dict]:
    """Stratified train/val split for a single sample.

    Returns (train_ds, val_ds, stats_dict).
    stats_dict contains normalisation parameters needed to create more datasets
    from other samples using the same statistics.
    """
    csv_path = os.path.join(sample_dir, 'sphinx_cell_table.csv')
    df = pd.read_csv(csv_path)
    n_total = len(df)

    rng = np.random.RandomState(seed)
    if max_cells is not None and max_cells < n_total:
        selected = np.sort(rng.choice(n_total, max_cells, replace=False))
    else:
        selected = np.arange(n_total)

    # Stratified split to ensure rare cell types appear in both splits
    ct_labels = np.argmax(df.iloc[selected][CELLTYPE_COLS].values, axis=1)
    train_list, val_list = [], []
    for cls in np.unique(ct_labels):
        cls_idx = selected[ct_labels == cls]
        n_tr    = max(1, int(len(cls_idx) * train_ratio))
        perm    = rng.permutation(len(cls_idx))
        train_list.append(cls_idx[perm[:n_tr]])
        val_list.append(  cls_idx[perm[n_tr:]])

    train_idx = np.sort(np.concatenate(train_list))
    val_idx   = np.sort(np.concatenate(val_list))

    # Normalisation statistics from training set only (no val leakage)
    protein_raw_train = np.full((len(train_idx), len(PROTEIN_COLS)), np.nan, dtype=np.float32)
    df_train = df.iloc[train_idx]
    for i, col in enumerate(PROTEIN_COLS):
        if col in df_train.columns:
            protein_raw_train[:, i] = df_train[col].values.astype(np.float32)
    norm_params = compute_norm_params(protein_raw_train, PROTEIN_COLS)

    morph_raw_train = df_train[MORPH_COLS].values.astype(np.float32)
    morph_mean = np.nanmean(morph_raw_train, axis=0)
    morph_std  = np.nanstd(morph_raw_train,  axis=0)

    # Panel info: detect which proteins and cell types exist in this sample
    # Based on the FULL sample (not just train split), so val evaluation is consistent
    protein_panel_full   = np.zeros(len(PROTEIN_COLS),  dtype=bool)
    celltype_panel_full  = np.zeros(len(CELLTYPE_COLS), dtype=bool)
    for i, col in enumerate(PROTEIN_COLS):
        if col in df.columns and df[col].notna().any():
            protein_panel_full[i] = True
    ct_labels_full = np.argmax(df[CELLTYPE_COLS].values.astype(np.float32), axis=1)
    for cls_idx in np.unique(ct_labels_full):
        celltype_panel_full[cls_idx] = True

    panel_info = {
        'protein_panel':       protein_panel_full,           # [18] bool
        'celltype_panel':      celltype_panel_full,          # [12] bool
        'n_proteins':          int(protein_panel_full.sum()),
        'n_celltypes':         int(celltype_panel_full.sum()),
        'has_epithelial':      bool(celltype_panel_full[11]),  # index 11 = Epithelial_Cells
        'present_celltype_idx': np.where(celltype_panel_full)[0].tolist(),
        'present_protein_idx':  np.where(protein_panel_full)[0].tolist(),
    }

    stats = {
        'n_total':     n_total,
        'n_train':     len(train_idx),
        'n_val':       len(val_idx),
        'norm_params': norm_params,
        'morph_mean':  morph_mean,
        'morph_std':   morph_std,
        'panel_info':  panel_info,
    }

    shared_kw = dict(norm_params=norm_params, morph_mean=morph_mean, morph_std=morph_std)
    train_ds = SingleSampleDataset(sample_dir, train_idx, augment=True,  **shared_kw)
    val_ds   = SingleSampleDataset(sample_dir, val_idx,   augment=False, **shared_kw)

    return train_ds, val_ds, stats
