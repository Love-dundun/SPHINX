"""
hotcold.py — Immune Hot/Cold tumour phenotyping from SPHINX predictions.

Generates a dynamic binary tumour-immune interaction map
that classifies every pixel inside the tumour area as either:

  Hot  (red)  — locally lymphocyte-rich microregion
  Cold (blue) — locally cancer-cell-dominant microregion

Method summary
--------------
1. Rasterise predicted cancer-cell and lymphocyte coordinates to a
   downsampled grid.
2. Build an automatic tumour mask from cancer-cell density:
   Gaussian blur → high-percentile threshold → morphological closing
   → flood-fill interior holes → keep the N largest connected components.
3. Within the tumour mask, compute a local hotness score at each pixel:

       local_hotness = lymph_density / (lymph_density + cancer_density + ε)

   using a Gaussian kernel with σ = MICRO_RADIUS_UM / (pixel_size × DS).

4. Compute a sample-level global immune score:

       global_score = lymph_in_tumour / (lymph_in_tumour + cancer_in_tumour)

5. Apply a dynamic threshold that adapts to the overall immune state:

       threshold = clip(0.12 - 0.35 × global_score,  0.04,  0.12)

   This ensures that highly immune-hot samples show more hot microregions
   and cold samples do not artificially inflate hot area.

6. Binary classification:

       Hot  = tumour_mask & density_gate & (local_hotness >= threshold)
       Cold = tumour_mask & ~Hot

7. Render the final figure:
   - H&E background outside the tumour mask (untouched).
   - Opaque red/blue colouring inside the tumour mask.
   - Thick black contour line around the tumour boundary.

Colour semantics (fixed, do not alter):
   Hot  → RGB (220, 35, 35)   — red
   Cold → RGB (35, 100, 215)  — blue

Input
-----
SPHINX predictions CSV produced by ``scripts/predict.py``.
Required columns: ``centroid_x``, ``centroid_y``, ``pred_celltype_name``

Output
------
``<output_dir>/<sample_name>_hotcold.png``   — main figure
``<output_dir>/<sample_name>_hotcold.json``  — hot/cold metrics

Usage
-----
Single sample::

    python downstream/hotcold.py \\
        --pred_csv   /path/to/sphinx_predictions.csv \\
        --he_image   /path/to/sample_HE.ome.tiff \\
        --output_dir /path/to/output \\
        --sample     sample_name \\
        --pixel_size 0.3125

Multiple samples via a list file::

    python downstream/hotcold.py \\
        --sample_list samples.txt \\
        --output_dir  /path/to/output

where ``samples.txt`` contains one line per sample::

    /path/to/A/sphinx_predictions.csv /path/to/A/HE.ome.tiff sample_A 0.3125
    /path/to/B/sphinx_predictions.csv /path/to/B/HE.ome.tiff sample_B 0.25
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import tifffile


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPHINX_PX_UM: float = 0.3125  # µm/px at which SPHINX coordinates are expressed

# Downsampling factor for raster grid and density maps
DS: int = 8

# Gaussian smoothing radius for local density estimation (µm)
MICRO_RADIUS_UM: float = 60.0

# Tumour mask generation: morphological margin (µm)
_TUMOR_MARGIN_UM: float = 100.0

# Final figure: display downsampling (on top of DS)
_DISPLAY_DS: int = 4

# Cell-type groups
_CANCER_TYPES = frozenset({"Cancer_Cells", "Epithelial_Cells"})
_LYMPH_TYPES  = frozenset({"T_Cells", "B_Cells", "NK_Cells"})

# Visualisation colours (uint8 RGB tuples used by OpenCV/NumPy)
_HOT_RGB     = (220, 35,  35)
_COLD_RGB    = (35,  100, 215)
_CONTOUR_RGB = (10,  10,  10)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_predictions(pred_csv: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load centroid coordinates and boolean cancer/lymph masks from CSV.

    Returns
    -------
    coords : float32 [N, 2]  (x, y)
    cancer : bool [N]
    lymph  : bool [N]
    """
    xs: List[float] = []
    ys: List[float] = []
    labels: List[str] = []

    with open(pred_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["centroid_x"]))
            ys.append(float(row["centroid_y"]))
            labels.append(row["pred_celltype_name"])

    coords = np.stack(
        [np.array(xs, np.float32), np.array(ys, np.float32)], axis=1
    )
    lbl_arr = np.array(labels)
    return coords, np.isin(lbl_arr, list(_CANCER_TYPES)), np.isin(lbl_arr, list(_LYMPH_TYPES))


def load_he_image(he_path: str | Path) -> np.ndarray:
    """Load the first page of an OME-TIFF as a uint8 RGB array."""
    with tifffile.TiffFile(str(he_path)) as tif:
        img = tif.pages[0].asarray()
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img_max = img.max()
        img = (img / max(img_max, 1) * 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Rasterisation helpers
# ---------------------------------------------------------------------------

def _rasterize(
    coords: np.ndarray, mask: np.ndarray,
    h_ds: int, w_ds: int,
) -> np.ndarray:
    """Rasterise selected cells into a downsampled count grid.

    Coordinates are expected to be in **native pixels** (same coordinate
    system as the H&E zarr index).  They are divided by DS to obtain
    grid indices.
    """
    x = np.clip((coords[mask, 0] / DS).astype(np.int32), 0, w_ds - 1)
    y = np.clip((coords[mask, 1] / DS).astype(np.int32), 0, h_ds - 1)
    out = np.zeros((h_ds, w_ds), np.float32)
    np.add.at(out, (y, x), 1.0)
    return out


# ---------------------------------------------------------------------------
# Tumour mask
# ---------------------------------------------------------------------------

def make_tumor_mask(
    cancer_count: np.ndarray,
    pixel_size: float,
    keep_components: int = 2,
) -> np.ndarray:
    """Generate a tumour mask automatically from cancer-cell density.

    Steps:
    1. Gaussian blur of the cancer-cell count map.
    2. Threshold at the 70th percentile of non-zero values.
    3. Morphological closing to bridge gaps (margin ~ 100 µm).
    4. Flood-fill to close interior holes.
    5. Keep the *keep_components* largest connected components.

    Parameters
    ----------
    cancer_count:    Cancer-cell count map at downsampled resolution.
    pixel_size:      µm/px of the original H&E image.
    keep_components: How many connected components to retain.

    Returns
    -------
    Boolean tumour mask at downsampled resolution.
    """
    h_ds, w_ds = cancer_count.shape
    dens = cv2.GaussianBlur(cancer_count, (0, 0), sigmaX=12)
    nz = dens[dens > 0]
    if len(nz) == 0:
        return np.zeros((h_ds, w_ds), dtype=bool)

    thr = float(np.percentile(nz, 70))
    binary = (dens > thr).astype(np.uint8)

    margin_ds = max(1, int(round(_TUMOR_MARGIN_UM / (pixel_size * DS) / 2)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (margin_ds * 2 + 1, margin_ds * 2 + 1)
    )
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Flood-fill from the corner to identify the background, then invert
    flood = closed.copy()
    fill_mask = np.zeros((h_ds + 2, w_ds + 2), np.uint8)
    cv2.floodFill(flood, fill_mask, (0, 0), 1)
    filled = (closed | (1 - flood)).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    out = np.zeros_like(filled)
    if n > 1:
        areas = sorted(
            [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)], reverse=True
        )
        for _, idx in areas[:keep_components]:
            out[labels == idx] = 1

    return out.astype(bool)


# ---------------------------------------------------------------------------
# hot/cold score computation
# ---------------------------------------------------------------------------

def compute_ati(
    coords: np.ndarray,
    cancer: np.ndarray,
    lymph: np.ndarray,
    tumor_mask: np.ndarray,
    pixel_size: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute the hot/cold state map and associated metrics.

    Parameters
    ----------
    coords:     [N, 2] float32 cell centroids (x, y) in SPHINX pixels.
    cancer:     Boolean mask — cancer/epithelial cells.
    lymph:      Boolean mask — T/B/NK cells.
    tumor_mask: Boolean tumour mask at downsampled resolution.
    pixel_size: µm/px of the H&E image.

    Returns
    -------
    state : uint8 array at downsampled resolution:
            0 = outside tumour, 1 = Cold, 2 = Hot
    metrics : dict with hot/cold scores and parameters
    """
    h_ds, w_ds = tumor_mask.shape

    # Coordinates are in native pixels; divide by DS to get grid indices.
    x_ds = np.clip((coords[:, 0] / DS).astype(np.int32), 0, w_ds - 1)
    y_ds = np.clip((coords[:, 1] / DS).astype(np.int32), 0, h_ds - 1)

    inside = tumor_mask[y_ds, x_ds]
    cancer_in = cancer & inside
    lymph_in  = lymph  & inside

    # Accumulate cells into density grids
    cancer_local = np.zeros((h_ds, w_ds), np.float32)
    lymph_local  = np.zeros((h_ds, w_ds), np.float32)
    np.add.at(cancer_local, (y_ds[cancer_in], x_ds[cancer_in]), 1.0)
    np.add.at(lymph_local,  (y_ds[lymph_in],  x_ds[lymph_in]),  1.0)

    sigma_ds = MICRO_RADIUS_UM / (pixel_size * DS)
    cancer_density = cv2.GaussianBlur(cancer_local, (0, 0), sigmaX=sigma_ds)
    lymph_density  = cv2.GaussianBlur(lymph_local,  (0, 0), sigmaX=sigma_ds)

    eps = 1e-6
    local_hotness = lymph_density / (lymph_density + cancer_density + eps)
    local_total   = lymph_density + cancer_density

    # Density gate: only classify pixels with meaningful local cell support
    has_cells = tumor_mask & (local_total > 0)
    if has_cells.any():
        density_floor = float(np.percentile(local_total[has_cells], 10))
    else:
        density_floor = 0.0
    density_gate = local_total >= density_floor

    # Dynamic threshold
    lymph_count  = int(lymph_in.sum())
    cancer_count = int(cancer_in.sum())
    global_score = float(lymph_count / max(lymph_count + cancer_count, 1))
    dynamic_threshold = float(np.clip(0.12 - 0.35 * global_score, 0.04, 0.12))

    hot_micro  = tumor_mask & density_gate & (local_hotness >= dynamic_threshold)
    cold_micro = tumor_mask & ~hot_micro

    state = np.zeros((h_ds, w_ds), dtype=np.uint8)
    state[cold_micro] = 1
    state[hot_micro]  = 2

    metrics = {
        "micro_radius_um":    MICRO_RADIUS_UM,
        "global_score":       global_score,
        "dynamic_threshold":  dynamic_threshold,
        "hot_area_ratio":     float(hot_micro.sum()  / max(tumor_mask.sum(), 1)),
        "cold_area_ratio":    float(cold_micro.sum() / max(tumor_mask.sum(), 1)),
        "mean_hotness":       float(np.mean(local_hotness[tumor_mask])) if tumor_mask.any() else 0.0,
        "lymph_count_in_tumor":  lymph_count,
        "cancer_count_in_tumor": cancer_count,
    }
    return state, metrics


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------

def render_hotcold_map(
    he_img: np.ndarray,
    tumor_mask: np.ndarray,
    state: np.ndarray,
    out_path: str | Path,
    sample_name: str,
    metrics: Dict[str, float],
    contour_thickness: int = 70,
) -> None:
    """Render and save the dynamic binary hot/cold figure.

    Visual conventions (fixed):
    - Hot microregions: opaque red   RGB(220, 35, 35)
    - Cold microregions: opaque blue RGB(35, 100, 215)
    - Tumour boundary: thick black contour
    - Non-tumour area: original H&E (untouched)

    Parameters
    ----------
    he_img:             Full-resolution H&E image (uint8 RGB).
    tumor_mask:         Boolean mask at downsampled resolution.
    state:              hot/cold state map (0=outside, 1=cold, 2=hot) at DS resolution.
    out_path:           Output PNG path.
    sample_name:        Sample name shown in the figure title.
    metrics:            Metrics dict from ``compute_ati``.
    contour_thickness:  Thickness of the tumour boundary line in pixels
                        (at display resolution; default 70).
    """
    h, w = he_img.shape[:2]
    h_disp = h // _DISPLAY_DS
    w_disp = w // _DISPLAY_DS

    he_disp = cv2.resize(he_img[..., :3], (w_disp, h_disp), interpolation=cv2.INTER_AREA)

    # Upscale state and mask to display resolution
    state_disp = cv2.resize(state, (w_disp, h_disp), interpolation=cv2.INTER_NEAREST)

    # Smooth the tumour mask for clean contour extraction
    tumor_disp = cv2.resize(
        tumor_mask.astype(np.uint8) * 255, (w_disp, h_disp), interpolation=cv2.INTER_NEAREST
    )
    tumor_smooth = cv2.GaussianBlur(tumor_disp, (0, 0), sigmaX=4)
    tumor_bin = (tumor_smooth > 127).astype(np.uint8)
    contours, _ = cv2.findContours(tumor_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Colour the display image
    img = he_disp.copy()
    img[state_disp == 1] = _COLD_RGB
    img[state_disp == 2] = _HOT_RGB
    cv2.drawContours(img, contours, -1, _CONTOUR_RGB, contour_thickness)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("white")
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(
        f"{sample_name}\n"
        f"thr={metrics['dynamic_threshold']:.3f}   "
        f"Hot {metrics['hot_area_ratio']:.1%}   "
        f"Cold {metrics['cold_area_ratio']:.1%}",
        fontsize=12,
        color="black",
    )
    handles = [
        mpatches.Patch(color=tuple(v / 255 for v in _HOT_RGB),  label="Hot"),
        mpatches.Patch(color=tuple(v / 255 for v in _COLD_RGB), label="Cold"),
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.85, facecolor="white")
    fig.tight_layout(pad=0.2)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-sample pipeline
# ---------------------------------------------------------------------------

def run_sample(
    pred_csv: str | Path,
    he_image: str | Path,
    output_dir: str | Path,
    sample_name: str,
    pixel_size: float = SPHINX_PX_UM,
    keep_tumor_components: int = 2,
) -> Dict[str, float]:
    """Run the full hot/cold phenotyping pipeline for one sample.

    Parameters
    ----------
    pred_csv:                SPHINX predictions CSV.
    he_image:                H&E OME-TIFF path.
    output_dir:              Directory for output files.
    sample_name:             Sample identifier.
    pixel_size:              µm/px of the H&E image (default: SPHINX training resolution).
    keep_tumor_components:   Number of connected tumour regions to retain (default 2).

    Returns
    -------
    Metrics dict with hot/cold scores and parameters.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[{sample_name}] Loading predictions …")
    coords, cancer, lymph = load_predictions(pred_csv)
    print(f"[{sample_name}]   {len(coords):,} cells   "
          f"cancer={int(cancer.sum()):,}  lymph={int(lymph.sum()):,}")

    print(f"[{sample_name}] Loading H&E image …")
    he_img = load_he_image(he_image)
    h, w = he_img.shape[:2]
    # Grid dimensions: native H&E pixels downsampled by DS.
    # Coordinates (native pixels) divide by DS to map into the grid.
    h_ds = max(1, int(round(h / DS)))
    w_ds = max(1, int(round(w / DS)))

    cancer_count = _rasterize(coords, cancer, h_ds, w_ds)
    print(f"[{sample_name}] Building tumour mask …")
    tumor_mask = make_tumor_mask(cancer_count, pixel_size, keep_tumor_components)
    tumor_frac = float(tumor_mask.sum() / tumor_mask.size)
    print(f"[{sample_name}]   tumour area fraction: {tumor_frac:.1%}")

    print(f"[{sample_name}] Computing hot/cold map …")
    state, metrics = compute_ati(coords, cancer, lymph, tumor_mask, pixel_size)
    print(
        f"[{sample_name}]   global_score={metrics['global_score']:.3f}  "
        f"threshold={metrics['dynamic_threshold']:.3f}  "
        f"Hot={metrics['hot_area_ratio']:.1%}  "
        f"Cold={metrics['cold_area_ratio']:.1%}"
    )

    png_path  = out / f"{sample_name}_hotcold.png"
    json_path = out / f"{sample_name}_hotcold.json"

    render_hotcold_map(he_img, tumor_mask, state, png_path, sample_name, metrics)
    with open(json_path, "w") as f:
        json.dump({"sample": sample_name, "pixel_size": pixel_size, **metrics}, f, indent=2)

    print(f"[{sample_name}] Saved: {png_path.name}  {json_path.name}")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    single = p.add_argument_group("single sample")
    single.add_argument("--pred_csv",   default=None, help="SPHINX predictions CSV.")
    single.add_argument("--he_image",   default=None, help="H&E OME-TIFF path.")
    single.add_argument("--sample",     default=None, help="Sample name / identifier.")
    single.add_argument("--pixel_size", type=float, default=SPHINX_PX_UM,
                        help=f"µm/px of the H&E image (default: {SPHINX_PX_UM}).")

    batch = p.add_argument_group("batch (sample list file)")
    batch.add_argument(
        "--sample_list", default=None,
        help="Text file with one sample per line: "
             "<pred_csv> <he_image> <sample_name> [<pixel_size>]",
    )

    p.add_argument("--output_dir", required=True, help="Directory for output files.")
    p.add_argument("--keep_tumor_components", type=int, default=2,
                   help="Number of connected tumour components to retain (default: 2).")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.sample_list:
        with open(args.sample_list) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 3:
                    parts.append(str(SPHINX_PX_UM))
                pred_csv, he_img, name, px = parts
                try:
                    run_sample(pred_csv, he_img, args.output_dir, name,
                               float(px), args.keep_tumor_components)
                except Exception as exc:
                    warnings.warn(f"Sample {name} failed: {exc}")
    elif args.pred_csv and args.he_image and args.sample:
        run_sample(
            args.pred_csv, args.he_image, args.output_dir,
            args.sample, args.pixel_size, args.keep_tumor_components,
        )
    else:
        _build_parser().print_help()
        raise SystemExit(
            "\nProvide either --pred_csv / --he_image / --sample "
            "or --sample_list."
        )


if __name__ == "__main__":
    main()
