from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import tifffile
from matplotlib import cm
from matplotlib.colors import to_rgb
from skimage.segmentation import expand_labels


PROTEIN_ORDER: list[str] = [
    "ATPase", "CD11c", "CD20", "CD3", "CD31", "CD4", "CD45RA", "CD56",
    "CD68", "CD8", "CTLA4", "FOXP3", "HLA-DR", "KI67", "PD1", "PDL1",
    "PanCK", "aSMA",
]

PROTEIN_COLORS: dict[str, str] = {
    "ATPase": "#17becf",
    "CD11c": "#40a9a8",
    "CD20": "#7f7f7f",
    "CD31": "#d62728",
    "CD3": "#1f77b4",
    "CD4": "#aec7e8",
    "CD45RA": "#e1e190",
    "CD56": "#bcbd22",
    "CD68": "#ffbb78",
    "CD8": "#f7b6d2",
    "CTLA4": "#e377c2",
    "FOXP3": "#c49c94",
    "HLA-DR": "#9467bd",
    "KI67": "#c5b0d5",
    "PD1": "#8c564b",
    "PDL1": "#6b9e59",
    "PanCK": "#ff7f0e",
    "aSMA": "#2ca02c",
}

STRUCTURAL_PROTEINS: list[str] = ["PanCK", "aSMA", "CD31"]
IMMUNE_PROTEINS: list[str] = [
    "CD3", "CD4", "CD8", "FOXP3", "PD1", "PDL1", "CTLA4", "CD20",
    "CD56", "CD68", "CD11c", "HLA-DR", "KI67", "CD45RA",
]
DEFAULT_SINGLE_PROTEINS: list[str] = ["PanCK", "CD3"]

CELLTYPE_COLORS: dict[str, str] = {
    "B_Cells": "#e15759",
    "Cancer_Cells": "#f1ce63",
    "Dendritic_Cells": "#4e79a7",
    "Endothelial_Cells": "#9c755f",
    "Fibroblasts": "#b07aa1",
    "Macrophages": "#808080",
    "Monocytes": "#f28e2b",
    "NK_Cells": "#499894",
    "Neutrophils": "#b6992d",
    "Smooth_Muscle_Cells": "#86bcb6",
    "T_Cells": "#ff9d9a",
    "Epithelial_Cells": "#59a14f",
    "Acinar_Cells": "#005f87",
    "Alpha_Cells": "#e8a838",
    "Beta_Cells": "#6db6d4",
    "Ductal_Cells": "#af7aa1",
    "PanIN": "#ff9da7",
    "Stellate_Cells": "#8cd17d",
    "Neuron": "#3a5a78",
    "Pericytes": "#c0a96a",
    "Plasma_Cells": "#e7969c",
}

BOUNDARY_RGB = np.array([60, 60, 60], dtype=np.uint8)
UNASSIGNED_RGB = np.array([176, 176, 176], dtype=np.uint8)
MISSING_PROTEIN_RGB = np.array([176, 176, 176], dtype=np.uint8)


def hex_to_rgb8(hex_color: str) -> np.ndarray:
    return (np.array(to_rgb(hex_color), dtype=np.float32) * 255).round().astype(np.uint8)


def load_mask(mask_npz: Path, mask_key: str, expand_px: int = -1) -> np.ndarray:
    """Load an instance mask from a .npz file and optionally expand each label.

    Args:
        mask_npz:  Path to the .npz file.
        mask_key:  Key to read (e.g. 'nuclei_exp' or 'mask').
        expand_px: Expand each cell region outward by this many pixels using a
                   Voronoi-style nearest-label fill (skimage.segmentation.expand_labels).
                   Use -1 for automatic behavior: raw nucleus masks (mask_key='mask')
                   are expanded by 25 px, while pre-expanded masks such as
                   'nuclei_exp' are left unchanged.  Set to 0 to skip expansion.
    """
    with np.load(mask_npz) as data:
        if mask_key not in data:
            keys = ", ".join(data.files)
            raise KeyError(f"Mask key '{mask_key}' not found in {mask_npz}. Available keys: {keys}")
        mask = np.asarray(data[mask_key])
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}")
    if not np.issubdtype(mask.dtype, np.integer):
        mask = mask.astype(np.int64)
    if expand_px < 0:
        expand_px = 25 if mask_key == "mask" else 0
    if expand_px > 0:
        mask = expand_labels(mask, distance=expand_px).astype(mask.dtype)
    return mask


def make_boundary(mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[:, :-1] |= (mask[:, :-1] != mask[:, 1:]) & (mask[:, :-1] > 0)
    boundary[:, 1:] |= (mask[:, :-1] != mask[:, 1:]) & (mask[:, 1:] > 0)
    boundary[:-1, :] |= (mask[:-1, :] != mask[1:, :]) & (mask[:-1, :] > 0)
    boundary[1:, :] |= (mask[:-1, :] != mask[1:, :]) & (mask[1:, :] > 0)
    return boundary


def rank_normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    ranks = np.zeros(values.shape, dtype=np.float32)
    for col_idx in range(values.shape[1]):
        col = values[:, col_idx]
        finite = np.isfinite(col)
        if finite.sum() == 0:
            continue
        finite_idx = np.flatnonzero(finite)
        order = finite_idx[np.argsort(col[finite], kind="mergesort")]
        if len(order) == 1:
            ranks[order, col_idx] = 1.0
        else:
            ranks[order, col_idx] = np.arange(len(order), dtype=np.float32) / float(len(order) - 1)
    return ranks


def protein_column(df: pd.DataFrame, protein: str) -> str:
    candidates = [f"pred_{protein}", protein, f"{protein}_intensity_mean"]
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Missing protein column for {protein}. Tried: {candidates}")


def protein_matrix(df: pd.DataFrame, proteins: Sequence[str]) -> np.ndarray:
    cols = [protein_column(df, protein) for protein in proteins]
    return df.loc[:, cols].to_numpy(dtype=np.float32)


def available_proteins(df: pd.DataFrame, proteins: Sequence[str]) -> list[str]:
    available = []
    for protein in proteins:
        try:
            protein_column(df, protein)
        except KeyError:
            continue
        available.append(protein)
    if not available:
        raise KeyError(f"None of the requested protein columns were found: {list(proteins)}")
    return available


def save_ome_rgb(image: np.ndarray, path: Path, image_name: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        print(f"SKIP existing: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(path),
        image,
        photometric="rgb",
        metadata={"axes": "YXS", "Name": image_name},
        compression="deflate",
        compressionargs={"level": 1},
        tile=(512, 512),
    )
    print(f"Saved: {path}")


def assign_lut_colors(
    cell_ids: np.ndarray,
    colors: np.ndarray,
    id_max: int,
    default_rgb: np.ndarray,
) -> np.ndarray:
    lut = np.tile(default_rgb.reshape(1, 3), (id_max, 1)).astype(np.uint8)
    valid_ids = (cell_ids >= 0) & (cell_ids < id_max)
    if valid_ids.sum() < len(cell_ids):
        print(f"WARNING: ignored {len(cell_ids) - int(valid_ids.sum())} cell_ids outside mask range")
    lut[cell_ids[valid_ids]] = colors[valid_ids]
    return lut


def render_dominant_protein(
    df: pd.DataFrame,
    proteins: Sequence[str],
    mask: np.ndarray,
    valid: np.ndarray,
    id_max: int,
) -> np.ndarray:
    cell_ids = df["cell_id"].to_numpy(dtype=np.int64)
    values = protein_matrix(df, proteins)
    ranks = rank_normalize(values)
    winner = np.argmax(ranks, axis=1)
    top_rank = ranks[np.arange(len(df)), winner]
    rgb = np.stack([hex_to_rgb8(PROTEIN_COLORS[p]) for p in proteins], axis=0).astype(np.float32)
    brightness = 0.65 + 0.35 * top_rank.astype(np.float32)
    cell_colors = np.clip(rgb[winner] * brightness[:, None], 0, 255).astype(np.uint8)
    lut = assign_lut_colors(cell_ids, cell_colors, id_max, MISSING_PROTEIN_RGB)
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[valid] = lut[mask[valid]]
    return canvas


def render_single_protein(
    df: pd.DataFrame,
    protein: str,
    mask: np.ndarray,
    valid: np.ndarray,
    id_max: int,
) -> np.ndarray:
    cell_ids = df["cell_id"].to_numpy(dtype=np.int64)
    values = protein_matrix(df, [protein]).ravel()
    ranks = rank_normalize(values).ravel()
    cell_colors = (cm.plasma(ranks)[:, :3] * 255).round().astype(np.uint8)
    lut = assign_lut_colors(cell_ids, cell_colors, id_max, np.array([0, 0, 0], dtype=np.uint8))
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[valid] = lut[mask[valid]]
    return canvas


def render_celltypes(
    df: pd.DataFrame,
    mask: np.ndarray,
    valid: np.ndarray,
    boundary: np.ndarray,
    id_max: int,
) -> np.ndarray:
    if "pred_celltype_name" not in df.columns:
        raise KeyError("Missing required column: pred_celltype_name")
    cell_ids = df["cell_id"].to_numpy(dtype=np.int64)
    labels = df["pred_celltype_name"].astype(str).to_numpy()
    cell_colors = np.stack(
        [hex_to_rgb8(CELLTYPE_COLORS.get(label, "#c8c8c8")) for label in labels],
        axis=0,
    )
    lut = assign_lut_colors(cell_ids, cell_colors, id_max, np.array([255, 255, 255], dtype=np.uint8))
    canvas = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    canvas[valid] = lut[mask[valid]]
    canvas[boundary & (mask > 0)] = BOUNDARY_RGB
    known_labels = np.isin(labels, list(CELLTYPE_COLORS))
    if known_labels.sum() < len(labels):
        unknown_ids = cell_ids[~known_labels]
        valid_unknown = (unknown_ids >= 0) & (unknown_ids < id_max)
        lut[unknown_ids[valid_unknown]] = UNASSIGNED_RGB
        canvas[valid] = lut[mask[valid]]
        canvas[boundary & (mask > 0)] = BOUNDARY_RGB
    return canvas


def read_predictions(pred_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(pred_csv)
    if "cell_id" not in df.columns:
        raise KeyError("Missing required column: cell_id")
    return df.sort_values("cell_id").reset_index(drop=True)


def render_protein_outputs(
    df: pd.DataFrame,
    mask: np.ndarray,
    valid: np.ndarray,
    id_max: int,
    output_dir: Path,
    sample: str,
    single_proteins: Sequence[str],
    legacy_extra_maps: bool,
    overwrite: bool,
) -> None:
    protein_dir = output_dir / "protein"
    outputs = [(available_proteins(df, PROTEIN_ORDER), protein_dir / f"{sample}_allprot.ome.tiff", "all proteins")]
    if legacy_extra_maps:
        outputs.extend(
            [
                (available_proteins(df, STRUCTURAL_PROTEINS), protein_dir / f"{sample}_structural.ome.tiff", "structural proteins"),
                (available_proteins(df, IMMUNE_PROTEINS), protein_dir / f"{sample}_immune.ome.tiff", "immune proteins"),
            ]
        )
    for proteins, path, label in outputs:
        canvas = render_dominant_protein(df, proteins, mask, valid, id_max)
        save_ome_rgb(canvas, path, f"{sample} {label}", overwrite)
    for protein in single_proteins:
        canvas = render_single_protein(df, protein, mask, valid, id_max)
        save_ome_rgb(canvas, protein_dir / f"{sample}_{protein}.ome.tiff", f"{sample} {protein}", overwrite)


def render_celltype_output(
    df: pd.DataFrame,
    mask: np.ndarray,
    valid: np.ndarray,
    boundary: np.ndarray,
    id_max: int,
    output_dir: Path,
    sample: str,
    overwrite: bool,
) -> None:
    canvas = render_celltypes(df, mask, valid, boundary, id_max)
    save_ome_rgb(canvas, output_dir / "celltype" / f"{sample}_celltype.ome.tiff", f"{sample} celltype", overwrite)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render SPHINX protein and cell-type staining OME-TIFF maps.")
    parser.add_argument("--pred_csv", required=True, type=Path)
    parser.add_argument("--mask_npz", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--sample", default=None)
    parser.add_argument(
        "--expand_px", type=int, default=-1,
        help=(
            "Expand each nucleus label outward by this many pixels (Voronoi fill) before "
            "rendering.  Default -1 uses auto mode: 25 px for --mask_key mask and 0 px "
            "for pre-expanded masks such as nuclei_exp.  Set explicitly to override."
        ),
    )
    parser.add_argument(
        "--mask_key", default="nuclei_exp",
        help=(
            "Key to read from --mask_npz.  "
            "'nuclei_exp' (default): expanded-nucleus mask where each cell ID covers "
            "the full cell body — produced by the nucleus-expansion step after segmentation. "
            "'mask': raw nucleus-only mask as written by scripts/segment.py.  "
            "Use 'nuclei_exp' for whole-cell staining maps; use 'mask' only when "
            "nucleus-level coloring is intended."
        ),
    )
    parser.add_argument("--mode", default="all", choices=["all", "protein", "celltype"])
    parser.add_argument("--single_proteins", nargs="*", default=DEFAULT_SINGLE_PROTEINS)
    parser.add_argument("--legacy_extra_maps", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = args.sample or args.pred_csv.stem
    df = read_predictions(args.pred_csv)
    mask = load_mask(args.mask_npz, args.mask_key, expand_px=args.expand_px)
    id_max = int(mask.max()) + 1
    boundary = make_boundary(mask)
    valid = (mask > 0) & ~boundary

    if args.mode in {"all", "protein"}:
        render_protein_outputs(
            df=df,
            mask=mask,
            valid=valid,
            id_max=id_max,
            output_dir=args.output_dir,
            sample=sample,
            single_proteins=args.single_proteins,
            legacy_extra_maps=args.legacy_extra_maps,
            overwrite=args.overwrite,
        )
    if args.mode in {"all", "celltype"}:
        render_celltype_output(
            df=df,
            mask=mask,
            valid=valid,
            boundary=boundary,
            id_max=id_max,
            output_dir=args.output_dir,
            sample=sample,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
