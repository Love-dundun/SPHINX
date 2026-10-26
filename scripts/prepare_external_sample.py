"""
prepare_external_sample.py — Convert any external cell table into a
SPHINX-ready sample directory.

Supported input formats
-----------------------
Cell segmentation tables from three common tools are auto-detected:

  qupath   Columns exported by QuPath Measure Cell Detections:
             "Centroid X µm" / "Centroid Y µm" (or pixel variants),
             "Cell: <marker> mean" or "Nucleus: <marker> mean"

  cellpose  Plain CSV with columns such as "x", "y" (pixel coordinates)

  stardist  StarDist/deepcell polygon CSV with "centroid-1"/"centroid-0"
             (row/column in pixel coordinates)

  auto      Inspect the header and pick the closest match (default)

Output
------
A directory that SPHINX scripts accept directly::

    <output_dir>/
    ├── sphinx_cell_table.csv   ← SPHINX cell table
    ├── <sample_name>_HE.ome.tiff         ← symlink (or copy) of the H&E image
    └── knn_k6.npz                        ← optional, generated if not present

Usage
-----
Minimum (inference-only, no protein ground truth):

    python scripts/prepare_external_sample.py \\
        --input_csv  /data/my_cells.csv \\
        --he_image   /data/my_slide.tiff \\
        --output_dir /data/sphinx_ready/sample_01 \\
        --pixel_size 0.25

With protein ground truth columns (for training or evaluation):

    python scripts/prepare_external_sample.py \\
        --input_csv      /data/my_cells.csv \\
        --he_image       /data/my_slide.tiff \\
        --output_dir     /data/sphinx_ready/sample_01 \\
        --pixel_size     0.25 \\
        --protein_prefix "Cell: " \\
        --celltype_col   "Cell type"

Notes on pixel_size
-------------------
All centroid coordinates written to the output CSV are in **native pixel
units** of the H&E image (i.e. the same coordinate system as the zarr
index).  SPHINX's patch extraction reads the zarr array by pixel index,
so native-pixel coordinates are required.

The ``--pixel_size`` value is recorded in the CSV for reference and is
passed to ``downstream/niche_pipeline.py`` and ``downstream/hotcold.py``,
where it is used only for physical-radius calculations (neighbourhood
kernel size = 60 µm / pixel_size).  It does **not** rescale the stored
coordinates.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# SPHINX reference pixel size
# ---------------------------------------------------------------------------
SPHINX_PX_UM: float = 0.3125  # µm per pixel used during SPHINX training


# ---------------------------------------------------------------------------
# Expected SPHINX column names
# ---------------------------------------------------------------------------
PROTEIN_COLS = [
    "ATPase_intensity_mean",
    "CD11c_intensity_mean",
    "CD20_intensity_mean",
    "CD3_intensity_mean",
    "CD31_intensity_mean",
    "CD4_intensity_mean",
    "CD45RA_intensity_mean",
    "CD56_intensity_mean",
    "CD68_intensity_mean",
    "CD8_intensity_mean",
    "CTLA4_intensity_mean",
    "FOXP3_intensity_mean",
    "HLA-DR_intensity_mean",
    "KI67_intensity_mean",
    "PD1_intensity_mean",
    "PDL1_intensity_mean",
    "PanCK_intensity_mean",
    "aSMA_intensity_mean",
]

PROTEIN_SHORT = [c.replace("_intensity_mean", "") for c in PROTEIN_COLS]

CELLTYPE_COLS = [
    "celltype_B_Cells",
    "celltype_Cancer_Cells",
    "celltype_Dendritic_Cells",
    "celltype_Endothelial_Cells",
    "celltype_Fibroblasts",
    "celltype_Macrophages",
    "celltype_Monocytes",
    "celltype_NK_Cells",
    "celltype_Neutrophils",
    "celltype_Smooth_Muscle_Cells",
    "celltype_T_Cells",
    "celltype_Epithelial_Cells",
]

MORPH_COLS = [
    "morph_area",
    "morph_perimeter",
    "morph_eccentricity",
    "morph_solidity",
    "morph_extent",
    "morph_major_axis",
    "morph_minor_axis",
    "morph_diameter",
    "morph_euler_number",
    "morph_feret_diameter",
    "morph_aspect_ratio",
    "morph_circularity",
    "morph_perimeter_complexity",
]


# ---------------------------------------------------------------------------
# Format-specific column name mappings
# ---------------------------------------------------------------------------

# QuPath: "Centroid X µm" style (physical units, must be converted to pixels)
_QUPATH_X_UM = ["Centroid X µm", "Centroid X um", "centroid_x_um"]
_QUPATH_Y_UM = ["Centroid Y µm", "Centroid Y um", "centroid_y_um"]
# QuPath pixel centroid (when exported without physical calibration)
_QUPATH_X_PX = ["Centroid X", "centroid_x_px"]
_QUPATH_Y_PX = ["Centroid Y", "centroid_y_px"]

# Cellpose
_CELLPOSE_X = ["x", "X", "centroid_x"]
_CELLPOSE_Y = ["y", "Y", "centroid_y"]

# StarDist / deepcell (row = Y, column = X in image)
_STARDIST_ROW = ["centroid-0", "row", "axis-0"]
_STARDIST_COL = ["centroid-1", "col", "column", "axis-1"]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first candidate column name that exists in *df*."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_format(df: pd.DataFrame) -> str:
    """Heuristically identify the input tool format."""
    cols_lower = {c.lower() for c in df.columns}
    if any("centroid x" in c.lower() for c in df.columns):
        return "qupath"
    if "centroid-0" in df.columns or "centroid-1" in df.columns:
        return "stardist"
    if "x" in cols_lower or "centroid_x" in cols_lower:
        return "cellpose"
    return "unknown"


def extract_coordinates(
    df: pd.DataFrame, fmt: str, pixel_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (x_px, y_px) arrays in **native pixel** coordinates.

    If the source has physical-unit centroids (e.g. QuPath µm export)
    they are divided by *pixel_size* to produce native pixel coordinates.
    Pixel-based formats (Cellpose, StarDist pixel export, QuPath pixel export)
    are returned unchanged — they are already in native pixels.

    The output coordinates are in **native pixel** units, matching the
    zarr image index used by SPHINX patch extraction.
    """
    if fmt == "qupath":
        x_um_col = _find_col(df, _QUPATH_X_UM)
        y_um_col = _find_col(df, _QUPATH_Y_UM)
        if x_um_col and y_um_col:
            # Physical unit export: convert µm → native pixels
            x = df[x_um_col].values.astype(np.float32) / pixel_size
            y = df[y_um_col].values.astype(np.float32) / pixel_size
            return x, y
        x_px_col = _find_col(df, _QUPATH_X_PX)
        y_px_col = _find_col(df, _QUPATH_Y_PX)
        if x_px_col and y_px_col:
            # Already in native pixels — no transformation needed
            return df[x_px_col].values.astype(np.float32), \
                   df[y_px_col].values.astype(np.float32)
        raise ValueError(
            "QuPath format detected but no centroid columns found.  "
            "Expected 'Centroid X µm'/'Centroid Y µm' or 'Centroid X'/'Centroid Y'."
        )

    if fmt == "stardist":
        row_col = _find_col(df, _STARDIST_ROW)
        col_col = _find_col(df, _STARDIST_COL)
        if row_col is None or col_col is None:
            raise ValueError(
                "StarDist format detected but 'centroid-0'/'centroid-1' columns not found."
            )
        # StarDist: row = Y, col = X — already in native pixels
        return df[col_col].values.astype(np.float32), \
               df[row_col].values.astype(np.float32)

    # cellpose / generic — already in native pixels
    x_col = _find_col(df, _CELLPOSE_X)
    y_col = _find_col(df, _CELLPOSE_Y)
    if x_col is None or y_col is None:
        raise ValueError(
            f"Cannot locate centroid columns in the input CSV.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Provide columns named 'centroid_x'/'centroid_y' or use --x_col/--y_col."
        )
    return df[x_col].values.astype(np.float32), \
           df[y_col].values.astype(np.float32)


def map_protein_columns(
    df: pd.DataFrame, protein_prefix: str
) -> Dict[str, Optional[str]]:
    """
    Attempt to find a source column for each SPHINX protein.

    Tries (in order):
      1. Exact SPHINX name (e.g. ``CD3_intensity_mean``)
      2. ``{protein_prefix}{short_name} mean`` (QuPath style)
      3. Short name case-insensitive match
    """
    mapping: Dict[str, Optional[str]] = {}
    for short, col in zip(PROTEIN_SHORT, PROTEIN_COLS):
        if col in df.columns:
            mapping[col] = col
            continue
        qupath_name = f"{protein_prefix}{short} mean"
        if qupath_name in df.columns:
            mapping[col] = qupath_name
            continue
        # Case-insensitive fallback
        lower_map = {c.lower(): c for c in df.columns}
        candidate = lower_map.get(short.lower())
        if candidate:
            mapping[col] = candidate
            continue
        mapping[col] = None  # Not found — will be filled with NaN
    return mapping


def map_celltype_column(
    df: pd.DataFrame, celltype_col: Optional[str]
) -> Optional[str]:
    """Return the source column containing cell-type label strings, or None."""
    if celltype_col and celltype_col in df.columns:
        return celltype_col
    for candidate in ["celltype", "cell_type", "Cell type", "label", "class"]:
        if candidate in df.columns:
            return candidate
    return None


_CELLTYPE_ALIASES: Dict[str, str] = {
    # Normalise common label variations → SPHINX canonical name
    "b_cells": "B_Cells",
    "b cell": "B_Cells",
    "bcell": "B_Cells",
    "cancer": "Cancer_Cells",
    "cancer_cells": "Cancer_Cells",
    "tumor": "Cancer_Cells",
    "tumour": "Cancer_Cells",
    "dendritic": "Dendritic_Cells",
    "dendritic_cells": "Dendritic_Cells",
    "endothelial": "Endothelial_Cells",
    "endothelial_cells": "Endothelial_Cells",
    "fibroblast": "Fibroblasts",
    "fibroblasts": "Fibroblasts",
    "macrophage": "Macrophages",
    "macrophages": "Macrophages",
    "monocyte": "Monocytes",
    "monocytes": "Monocytes",
    "nk": "NK_Cells",
    "nk_cells": "NK_Cells",
    "neutrophil": "Neutrophils",
    "neutrophils": "Neutrophils",
    "smooth_muscle": "Smooth_Muscle_Cells",
    "smooth muscle": "Smooth_Muscle_Cells",
    "smooth_muscle_cells": "Smooth_Muscle_Cells",
    "t_cells": "T_Cells",
    "t cell": "T_Cells",
    "tcell": "T_Cells",
    "epithelial": "Epithelial_Cells",
    "epithelial_cells": "Epithelial_Cells",
}

SPHINX_CELLTYPES = [c.replace("celltype_", "") for c in CELLTYPE_COLS]


def normalise_celltype_label(raw: str) -> str:
    """Map a raw cell-type string to the nearest SPHINX canonical name."""
    key = raw.strip().lower().replace(" ", "_")
    if key in _CELLTYPE_ALIASES:
        return _CELLTYPE_ALIASES[key]
    # Direct match after removing underscores
    for ct in SPHINX_CELLTYPES:
        if ct.lower().replace("_", "") == key.replace("_", ""):
            return ct
    return raw  # Return as-is; will produce all-zero one-hot


def build_celltype_onehot(labels: np.ndarray) -> pd.DataFrame:
    """
    Convert an array of cell-type label strings to SPHINX one-hot columns.

    Unknown labels produce all-zero rows (treated as unlabelled by the
    training loss via the ``celltype_valid_mask`` mechanism).
    """
    rows = {col: np.zeros(len(labels), dtype=np.float32) for col in CELLTYPE_COLS}
    for i, raw in enumerate(labels):
        canonical = normalise_celltype_label(str(raw))
        col = f"celltype_{canonical}"
        if col in rows:
            rows[col][i] = 1.0
    return pd.DataFrame(rows)


def precompute_knn(
    x: np.ndarray, y: np.ndarray, k: int, out_path: Path
) -> None:
    """Build a K-nearest-neighbour cache and save it to *out_path*."""
    coords = np.stack([y, x], axis=1)
    tree = cKDTree(coords)
    dists, idx = tree.query(coords, k=k + 1, workers=-1)
    knn_idx = idx[:, 1:].astype(np.int64)
    knn_dist = dists[:, 1:].astype(np.float32)
    np.savez_compressed(str(out_path), knn_idx=knn_idx, knn_dist=knn_dist)
    print(f"  KNN cache written: {out_path}  ({len(coords)} cells, k={k})")


# ---------------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------------

def prepare_sample(
    input_csv: str,
    he_image: str,
    output_dir: str,
    pixel_size: float,
    fmt: str = "auto",
    protein_prefix: str = "Cell: ",
    celltype_col: Optional[str] = None,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    knn_k: int = 6,
    copy_image: bool = False,
) -> Path:
    """
    Convert an external cell table to a SPHINX sample directory.

    Parameters
    ----------
    input_csv:
        Path to the input cell table CSV.
    he_image:
        Path to the H&E whole-slide image (any TIFF format).
    output_dir:
        Target directory; will be created if it does not exist.
    pixel_size:
        Pixel size of the H&E image in µm/pixel.
    fmt:
        Input format: ``'auto'``, ``'qupath'``, ``'stardist'``,
        ``'cellpose'``, or ``'generic'``.
    protein_prefix:
        Prefix used by the source tool before the protein/marker name
        (e.g. ``'Cell: '`` for QuPath ``'Cell: CD3 mean'`` columns).
    celltype_col:
        Column containing cell-type label strings.  If ``None`` the
        script will search for common names; if none found, cell-type
        columns are left as all-zero (inference-only mode).
    x_col, y_col:
        Override automatic centroid column detection.
    knn_k:
        Number of nearest neighbours to precompute (default 6).
    copy_image:
        If ``True`` the H&E file is copied into the output directory.
        If ``False`` (default) a symbolic link is created instead.

    Returns
    -------
    Path to the output directory.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load input CSV
    # ------------------------------------------------------------------
    print(f"Reading input CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    n_cells = len(df)
    print(f"  {n_cells} cells found")

    # ------------------------------------------------------------------
    # 2. Detect format and extract coordinates
    # ------------------------------------------------------------------
    if fmt == "auto":
        fmt = detect_format(df)
        print(f"  Detected input format: {fmt}")

    if x_col and y_col:
        # User-specified column names — values are assumed to be native pixels
        cx = df[x_col].values.astype(np.float32)
        cy = df[y_col].values.astype(np.float32)
    else:
        cx, cy = extract_coordinates(df, fmt, pixel_size)

    print(
        f"  Coordinate range  x: [{cx.min():.1f}, {cx.max():.1f}]  "
        f"y: [{cy.min():.1f}, {cy.max():.1f}] (native px)"
    )

    # ------------------------------------------------------------------
    # 3. Build output DataFrame — start with coordinates
    # ------------------------------------------------------------------
    out = pd.DataFrame({"centroid_x": cx, "centroid_y": cy})

    # ------------------------------------------------------------------
    # 4. Protein columns
    # ------------------------------------------------------------------
    protein_map = map_protein_columns(df, protein_prefix)
    found_proteins = sum(1 for v in protein_map.values() if v is not None)
    print(f"  Protein channels found: {found_proteins} / {len(PROTEIN_COLS)}")
    for sphinx_col, src_col in protein_map.items():
        if src_col is not None:
            out[sphinx_col] = df[src_col].values.astype(np.float32)
        else:
            out[sphinx_col] = np.nan

    # ------------------------------------------------------------------
    # 5. Cell-type columns
    # ------------------------------------------------------------------
    ct_src = map_celltype_column(df, celltype_col)
    if ct_src:
        print(f"  Cell-type labels from column: '{ct_src}'")
        ct_df = build_celltype_onehot(df[ct_src].values)
        for col in CELLTYPE_COLS:
            out[col] = ct_df[col].values
    else:
        print(
            "  No cell-type label column found — "
            "inference-only mode (all celltype columns set to 0)"
        )
        for col in CELLTYPE_COLS:
            out[col] = 0.0

    # ------------------------------------------------------------------
    # 6. Morphology columns (passthrough if present, zeros otherwise)
    # ------------------------------------------------------------------
    morph_found = 0
    for col in MORPH_COLS:
        if col in df.columns:
            out[col] = df[col].values.astype(np.float32)
            morph_found += 1
        else:
            out[col] = 0.0
    print(f"  Morphology columns found: {morph_found} / {len(MORPH_COLS)}")

    # ------------------------------------------------------------------
    # 7. Write SPHINX cell table
    # ------------------------------------------------------------------
    csv_out = out_path / "sphinx_cell_table.csv"
    out.to_csv(csv_out, index=False)
    print(f"  Cell table written: {csv_out}")

    # ------------------------------------------------------------------
    # 8. Link or copy H&E image
    # ------------------------------------------------------------------
    he_src = Path(he_image).resolve()
    sample_name = out_path.name
    # Ensure the file name follows the SPHINX convention
    he_dst = out_path / f"{sample_name}_HE.ome.tiff"

    if not he_src.exists():
        print(f"  WARNING: H&E image not found at {he_src} — skipping link/copy")
    elif copy_image:
        shutil.copy2(str(he_src), str(he_dst))
        print(f"  H&E image copied: {he_dst}")
    else:
        if he_dst.exists() or he_dst.is_symlink():
            he_dst.unlink()
        he_dst.symlink_to(he_src)
        print(f"  H&E image symlinked: {he_dst} -> {he_src}")

    # ------------------------------------------------------------------
    # 9. Precompute KNN cache
    # ------------------------------------------------------------------
    knn_path = out_path / f"knn_k{knn_k}.npz"
    precompute_knn(cx, cy, knn_k, knn_path)

    print(f"\nSample directory ready: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input_csv", required=True,
                   help="Input cell table CSV (QuPath, Cellpose, StarDist, or generic).")
    p.add_argument("--he_image", required=True,
                   help="H&E whole-slide image path (TIFF / OME-TIFF).")
    p.add_argument("--output_dir", required=True,
                   help="Output sample directory (will be created).")
    p.add_argument("--pixel_size", type=float, required=True,
                   help="Pixel size of the H&E image in µm/pixel (e.g. 0.25).")
    p.add_argument("--format", dest="fmt", default="auto",
                   choices=["auto", "qupath", "stardist", "cellpose", "generic"],
                   help="Input table format (default: auto-detect).")
    p.add_argument("--protein_prefix", default="Cell: ",
                   help="Column prefix used by source tool before marker name "
                        "(default: 'Cell: ' for QuPath).")
    p.add_argument("--celltype_col", default=None,
                   help="Column name containing cell-type label strings.")
    p.add_argument("--x_col", default=None,
                   help="Override: column name for X (horizontal) centroid in pixels.")
    p.add_argument("--y_col", default=None,
                   help="Override: column name for Y (vertical) centroid in pixels.")
    p.add_argument("--knn_k", type=int, default=6,
                   help="Number of KNN neighbours to precompute (default: 6).")
    p.add_argument("--copy_image", action="store_true",
                   help="Copy H&E image into output directory instead of symlinking.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    prepare_sample(
        input_csv=args.input_csv,
        he_image=args.he_image,
        output_dir=args.output_dir,
        pixel_size=args.pixel_size,
        fmt=args.fmt,
        protein_prefix=args.protein_prefix,
        celltype_col=args.celltype_col,
        x_col=args.x_col,
        y_col=args.y_col,
        knn_k=args.knn_k,
        copy_image=args.copy_image,
    )


if __name__ == "__main__":
    main()
