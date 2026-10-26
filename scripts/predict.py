import argparse
import os
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import tifffile
import zarr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sphinx.dataset import (
    CELLTYPE_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MORPH_COLS,
    PROTEIN_COLS,
    _crop_patch,
    apply_norm,
    compute_norm_params,
)
from sphinx.model import ViTProteinModel


def amp_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def denormalize_proteins(pred_norm: np.ndarray, norm_params: dict) -> np.ndarray:
    out = pred_norm.copy().astype(np.float32)
    for i in range(pred_norm.shape[1]):
        col = pred_norm[:, i]
        if norm_params["use_rank"][i]:
            edges = norm_params["rank_edges"][i]
            pct = np.clip((col + 2.0) * 25.0, 0.0, 100.0)
            idx_lo = np.floor(pct).astype(int).clip(0, 99)
            frac = pct - idx_lo
            out[:, i] = edges[idx_lo] * (1.0 - frac) + edges[(idx_lo + 1).clip(0, 100)] * frac
        else:
            out[:, i] = np.expm1(col * norm_params["prot_std"][i] + norm_params["prot_mean"][i]).clip(0, None)
    return out


class InferenceDataset(Dataset):
    def __init__(
        self,
        sample_dir: str,
        norm_params: dict,
        morph_mean: np.ndarray,
        morph_std: np.ndarray,
        detail_size: int = 96,
        context_size: int = 192,
        target_size: int = 224,
        num_neighbors: int = 6,
    ):
        self.sample_dir = sample_dir
        self.detail_size = detail_size
        self.context_size = context_size
        self.target_size = target_size
        self.num_neighbors = num_neighbors

        csv_path = os.path.join(sample_dir, "sphinx_cell_table.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Expected cell table: {csv_path}")
        df = pd.read_csv(csv_path)
        if "cell_id" in df.columns:
            df = df.sort_values("cell_id").reset_index(drop=True)
        self.df = df
        self.n_cells = len(df)

        self.centroids_y = df["centroid_y"].values.astype(np.float32)
        self.centroids_x = df["centroid_x"].values.astype(np.float32)

        protein_raw = np.full((self.n_cells, len(PROTEIN_COLS)), np.nan, dtype=np.float32)
        for i, col in enumerate(PROTEIN_COLS):
            if col in df.columns:
                protein_raw[:, i] = df[col].values.astype(np.float32)
        self.proteins = apply_norm(protein_raw, norm_params)

        morph_raw = df[MORPH_COLS].values.astype(np.float32) if all(c in df.columns for c in MORPH_COLS) else np.zeros((self.n_cells, len(MORPH_COLS)), np.float32)
        self.morphs = np.nan_to_num((morph_raw - morph_mean) / (morph_std + 1e-8), nan=0.0).astype(np.float32)

        knn_path = os.path.join(sample_dir, f"knn_k{num_neighbors}.npz")
        if os.path.exists(knn_path):
            cache = np.load(knn_path)
            self.neighbor_indices = cache["knn_idx"].astype(np.int64)
            self.neighbor_dists = cache["knn_dist"].astype(np.float32)
        else:
            from scipy.spatial import cKDTree

            coords = np.stack([self.centroids_y, self.centroids_x], axis=1)
            tree = cKDTree(coords)
            dists, knn_idx = tree.query(coords, k=num_neighbors + 1, workers=-1)
            self.neighbor_indices = knn_idx[:, 1:].astype(np.int64)
            self.neighbor_dists = dists[:, 1:].astype(np.float32)
            np.savez_compressed(knn_path, knn_idx=self.neighbor_indices, knn_dist=self.neighbor_dists)

        self._adj_sigma = 30.0

        tiff_files = [f for f in os.listdir(sample_dir) if f.endswith("_HE.ome.tiff")]
        if len(tiff_files) != 1:
            raise FileNotFoundError(f"Expected exactly one *_HE.ome.tiff in {sample_dir}, found {len(tiff_files)}")
        self._tiff_path = os.path.join(sample_dir, tiff_files[0])
        tiff = tifffile.TiffFile(self._tiff_path)
        zarr_arr = zarr.open(tiff.pages[0].aszarr(), mode="r")
        self._img_h, self._img_w = zarr_arr.shape[:2]
        tiff.close()
        self._tiff = None
        self._zarr = None

    def _ensure_zarr(self):
        if self._zarr is None:
            self._tiff = tifffile.TiffFile(self._tiff_path)
            self._zarr = zarr.open(self._tiff.pages[0].aszarr(), mode="r")

    def _preprocess(self, patch: np.ndarray) -> torch.Tensor:
        img = cv2.resize(patch, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        arr = img.astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(arr.transpose(2, 0, 1))

    def _build_adj(self, idx: int) -> torch.Tensor:
        k = self.num_neighbors
        n_nodes = 1 + k
        sigma = self._adj_sigma
        nb_idx = self.neighbor_indices[idx]
        center_dist = self.neighbor_dists[idx]

        adj = np.eye(n_nodes, dtype=np.float32)
        center_w = np.exp(-(center_dist ** 2) / (sigma ** 2)).astype(np.float32)
        adj[0, 1:] = center_w
        adj[1:, 0] = center_w

        nb_y = self.centroids_y[nb_idx]
        nb_x = self.centroids_x[nb_idx]
        nb_coords = np.stack([nb_y, nb_x], axis=-1)
        diff = nb_coords[:, None, :] - nb_coords[None, :, :]
        nb_w = np.exp(-(diff ** 2).sum(axis=-1) / (sigma ** 2)).astype(np.float32)
        nb_w[nb_w < 0.01] = 0.0
        adj[1:, 1:] = nb_w
        return torch.from_numpy(adj)

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx: int) -> dict:
        self._ensure_zarr()
        cy = float(self.centroids_y[idx])
        cx = float(self.centroids_x[idx])
        detail_patch = _crop_patch(self._zarr, cy, cx, self.detail_size, self._img_h, self._img_w)
        context_patch = _crop_patch(self._zarr, cy, cx, self.context_size, self._img_h, self._img_w)
        return {
            "idx": torch.tensor(idx, dtype=torch.long),
            "detail": self._preprocess(detail_patch),
            "context": self._preprocess(context_patch),
            "adj": self._build_adj(idx),
        }


@torch.no_grad()
def run_inference(model: ViTProteinModel, dataset: InferenceDataset, batch_size: int, num_workers: int, device: torch.device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    print("Pass 1: predicting proteins")
    all_proteins = []
    for batch in loader:
        detail = batch["detail"].to(device)
        context = batch["context"].to(device)
        with amp_context(device):
            protein_pred = model.predict_proteins_only(detail, context)
        all_proteins.append(protein_pred.float().cpu())
    all_proteins = torch.cat(all_proteins, 0)

    print("Pass 2: predicting cell types")
    all_logits = []
    for batch in loader:
        idx = batch["idx"]
        detail = batch["detail"].to(device)
        context = batch["context"].to(device)
        adj = batch["adj"].to(device)
        nb_idx = dataset.neighbor_indices[idx.numpy()]
        nb_protein = all_proteins[nb_idx.reshape(-1)].view(idx.size(0), dataset.num_neighbors, len(PROTEIN_COLS)).to(device)

        with amp_context(device):
            out = model(detail, context, nb_protein=nb_protein, adj=adj)
        all_logits.append(out["celltype_logits_A"].float().cpu())

    logits = torch.cat(all_logits, 0)
    probs = torch.softmax(logits, dim=1)
    labels = probs.argmax(dim=1)
    return all_proteins.numpy(), probs.numpy(), labels.numpy(), logits.numpy()


def main():
    parser = argparse.ArgumentParser(description="Run two-pass SPHINX inference on one sample.")
    parser.add_argument("--sample_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--backbone", default="vit", choices=["vit", "ctranspath"],
                        help="Image backbone: 'vit' (default) or 'ctranspath'. "
                             "When loading a SPHINX checkpoint the backbone type "
                             "must match the one used during training.")
    parser.add_argument("--ctranspath_weights", default=None,
                        help="Path to ctranspath.pth (only needed when training "
                             "from scratch; not required when loading a SPHINX checkpoint).")
    parser.add_argument("--mask_epithelial", action="store_true",
                        help="Mask Epithelial_Cells if this class is absent in the sample/panel.")
    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = os.path.join(args.sample_dir, "sphinx_predictions.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Reconstruct the exact architecture that was used during training.
    # Falls back to all-enabled defaults for checkpoints saved before model_config was added.
    model_config = checkpoint.get("model_config", dict(
        num_proteins=18, num_celltypes=12,
        backbone=args.backbone,
        use_protein_gat=True, use_protein_mlp_cls=True,
        use_deep_supervision=True, use_uncertainty_loss=True,
    ))
    model = ViTProteinModel(pretrained=False, **model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    csv_path = os.path.join(args.sample_dir, "sphinx_cell_table.csv")
    df_all = pd.read_csv(csv_path)
    if "cell_id" in df_all.columns:
        df_all = df_all.sort_values("cell_id").reset_index(drop=True)

    protein_raw = np.full((len(df_all), len(PROTEIN_COLS)), np.nan, dtype=np.float32)
    for i, col in enumerate(PROTEIN_COLS):
        if col in df_all.columns:
            protein_raw[:, i] = df_all[col].values.astype(np.float32)
    norm_params = compute_norm_params(protein_raw, PROTEIN_COLS)

    if all(c in df_all.columns for c in MORPH_COLS):
        morph_raw = df_all[MORPH_COLS].values.astype(np.float32)
    else:
        morph_raw = np.zeros((len(df_all), len(MORPH_COLS)), dtype=np.float32)
    morph_mean = np.nanmean(morph_raw, axis=0)
    morph_std = np.nanstd(morph_raw, axis=0)

    dataset = InferenceDataset(args.sample_dir, norm_params=norm_params, morph_mean=morph_mean, morph_std=morph_std)
    protein_pred, ct_probs, ct_labels, ct_logits = run_inference(model, dataset, args.batch_size, args.num_workers, device)

    if args.mask_epithelial:
        ct_logits[:, 11] = -1e9
        ct_probs = torch.softmax(torch.from_numpy(ct_logits), dim=1).numpy()
        ct_labels = ct_probs.argmax(axis=1)

    protein_pred_denorm = denormalize_proteins(protein_pred, norm_params)

    out_df = df_all.copy()
    for i, name in enumerate([c.replace("_intensity_mean", "") for c in PROTEIN_COLS]):
        out_df[f"pred_{name}"] = protein_pred_denorm[:, i]
    for i, name in enumerate(CELLTYPE_NAMES):
        out_df[f"prob_{name}"] = ct_probs[:, i]
    out_df["pred_celltype_idx"] = ct_labels
    out_df["pred_celltype_name"] = [CELLTYPE_NAMES[int(label)] for label in ct_labels]

    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions: {args.output_csv}")


if __name__ == "__main__":
    main()
