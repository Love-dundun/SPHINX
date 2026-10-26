"""
niche_pipeline.py — Adaptive spatial niche detection from SPHINX predictions.

Computes three tumour microenvironment niches at single-cell spatial resolution:

  TIL  Tumour-Infiltrating Lymphocytes
         Score = cancer_ratio × lymph_ratio
         High score: cancer cells and lymphocytes locally co-exist.

  TLS  Tertiary Lymphoid Structures
         Score = (T+B cell ratio) × T/B balance weight
         High score: T and B cells are both locally abundant.

  CAF  Cancer-Associated Fibroblasts
         Score = stroma_ratio  (fibroblasts + smooth-muscle cells)
         High score: fibroblast/stromal dominance in the local neighbourhood.

Algorithm
---------
1. Rasterise cell coordinates to a downsampled grid.
2. Apply a circular neighbourhood kernel (radius = 60 µm) to compute
   local cell-type ratios at every grid position.
3. Compute niche scores from the ratios.
4. Search a parameter grid (percentile threshold, morphology operations,
   minimum component area) and score each candidate by a composite
   heuristic that penalises biologically implausible outcomes.
5. Save the best overlay PNG and a JSON summary for each niche.

Input
-----
SPHINX predictions CSV produced by ``scripts/predict.py``.
Required columns: ``centroid_x``, ``centroid_y``, ``pred_celltype_name``

Output (per sample)
-------------------
``<output_dir>/
    <sample>_TIL.png
    <sample>_TLS.png
    <sample>_CAF.png
    <sample>_niche_summary.json``

Usage
-----
Single sample::

    python downstream/niche_pipeline.py \\
        --pred_csv   /path/to/sphinx_predictions.csv \\
        --he_image   /path/to/sample_HE.ome.tiff \\
        --output_dir /path/to/output \\
        --sample     sample_name \\
        --pixel_size 0.3125

Multiple samples via a list file::

    python downstream/niche_pipeline.py \\
        --sample_list samples.txt \\
        --output_dir  /path/to/output \\
        --workers     8

where ``samples.txt`` contains one line per sample::

    /path/to/sample_A/sphinx_predictions.csv /path/to/sample_A/HE.ome.tiff sample_A 0.3125
    /path/to/sample_B/sphinx_predictions.csv /path/to/sample_B/HE.ome.tiff sample_B 0.25
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

cv2.setNumThreads(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPHINX_PX_UM: float = 0.3125  # µm/px at which SPHINX coordinates are expressed

# Downsampling factor for the raster grid
DS: int = 8

# Neighbourhood analysis radius in micrometres
RADIUS_UM: float = 60.0

# Colour for each niche (RGB float 0–1)
NICHE_COLORS: Dict[str, np.ndarray] = {
    "TIL": np.array([0.00, 0.82, 0.27], np.float32),
    "TLS": np.array([0.20, 0.60, 1.00], np.float32),
    "CAF": np.array([1.00, 0.86, 0.00], np.float32),
}

# Biological plausibility targets used for candidate scoring.
# These are soft priors, not hard thresholds.
_NICHE_PRIORS: Dict[str, Dict] = {
    "TIL": dict(
        area_center=0.10, area_low=0.03, area_high=0.25,
        comp_target=90,   largest_target=0.42,
        score_low=0.005,  score_high=0.20,
    ),
    "TLS": dict(
        area_center=0.08, area_low=0.02, area_high=0.22,
        comp_target=120,  largest_target=0.35,
        score_low=0.05,   score_high=0.85,
    ),
    "CAF": dict(
        area_center=0.12, area_low=0.03, area_high=0.35,
        comp_target=180,  largest_target=0.28,
        score_low=0.03,   score_high=0.60,
    ),
}

# Parameter search grids
_QUALITY_GRID: Dict = dict(
    min_cells=[50, 80],
    p_high=[70, 75, 80, 85],
    close_r=[4, 6, 8],
    open_r=[1],
    min_comp=[200, 500, 1000],
)

# SPHINX cell-type name groups
_CANCER_TYPES = {"Cancer_Cells", "Epithelial_Cells"}
_LYMPH_TYPES = {"T_Cells", "B_Cells", "NK_Cells"}
_T_TYPES = {"T_Cells"}
_B_TYPES = {"B_Cells"}
_STROMA_TYPES = {"Fibroblasts", "Smooth_Muscle_Cells"}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_predictions(pred_csv: str | Path) -> Dict[str, np.ndarray]:
    """Load centroid coordinates and predicted cell-type labels from a SPHINX CSV.

    Returns a dict with keys:
        ``coords``   float32 [N, 2] (x, y)
        ``cancer``   bool [N]
        ``lymph``    bool [N]
        ``T``        bool [N]
        ``B``        bool [N]
        ``stroma``   bool [N]
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

    return {
        "coords": coords,
        "cancer": np.isin(lbl_arr, list(_CANCER_TYPES)),
        "lymph":  np.isin(lbl_arr, list(_LYMPH_TYPES)),
        "T":      np.isin(lbl_arr, list(_T_TYPES)),
        "B":      np.isin(lbl_arr, list(_B_TYPES)),
        "stroma": np.isin(lbl_arr, list(_STROMA_TYPES)),
    }


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
# Rasterisation and ratio maps
# ---------------------------------------------------------------------------

def _rasterize_channel(
    x_ds: np.ndarray, y_ds: np.ndarray, mask: np.ndarray,
    h_ds: int, w_ds: int,
) -> np.ndarray:
    """Count selected cells into a downsampled grid."""
    out = np.zeros((h_ds, w_ds), np.float32)
    xc = np.clip(x_ds[mask], 0, w_ds - 1)
    yc = np.clip(y_ds[mask], 0, h_ds - 1)
    np.add.at(out, (yc, xc), 1.0)
    return out


def _disk_kernel(radius: int) -> np.ndarray:
    r = max(1, int(round(radius)))
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    return ((xx * xx + yy * yy) <= r * r).astype(np.float32)


def build_ratio_maps(
    data: Dict[str, np.ndarray],
    h_ds: int,
    w_ds: int,
    pixel_size: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
    """Build local cell-type ratio maps and niche scores.

    Parameters
    ----------
    data:       Output of ``load_predictions``.
    h_ds, w_ds: Downsampled grid dimensions.
    pixel_size: µm/px of the H&E image.  Used only to compute the physical
                neighbourhood kernel radius (RADIUS_UM / pixel_size / DS).

    Returns
    -------
    ratio_maps:  Dict of ratio arrays, one per cell-type group.
    scores:      Dict of niche score arrays (TIL, TLS, CAF).
    cell_total:  Total cell density map (downsampled grid).
    """
    coords = data["coords"]
    # Coordinates are in native pixels (same as the H&E zarr index).
    # Divide by DS to obtain grid indices.
    x_ds = (coords[:, 0] / DS).astype(np.int32)
    y_ds = (coords[:, 1] / DS).astype(np.int32)

    # Count maps per cell group
    total_map  = _rasterize_channel(x_ds, y_ds, np.ones(len(x_ds), bool), h_ds, w_ds)
    cancer_map = _rasterize_channel(x_ds, y_ds, data["cancer"], h_ds, w_ds)
    lymph_map  = _rasterize_channel(x_ds, y_ds, data["lymph"],  h_ds, w_ds)
    t_map      = _rasterize_channel(x_ds, y_ds, data["T"],      h_ds, w_ds)
    b_map      = _rasterize_channel(x_ds, y_ds, data["B"],      h_ds, w_ds)
    stroma_map = _rasterize_channel(x_ds, y_ds, data["stroma"], h_ds, w_ds)

    # Circular neighbourhood kernel
    r_ds = RADIUS_UM / (pixel_size * DS)
    kernel = _disk_kernel(int(round(r_ds)))

    def smooth(m: np.ndarray) -> np.ndarray:
        return cv2.filter2D(m, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    ct_smooth     = smooth(total_map)
    cell_total    = ct_smooth
    denom         = np.maximum(ct_smooth, 1.0)

    ratio_maps = {
        "cancer": smooth(cancer_map) / denom,
        "lymph":  smooth(lymph_map)  / denom,
        "T":      smooth(t_map)      / denom,
        "B":      smooth(b_map)      / denom,
        "stroma": smooth(stroma_map) / denom,
    }

    # TLS: T+B combined with a T/B balance weight
    tb_ratio = ratio_maps["T"] + ratio_maps["B"]
    tb_sum = np.maximum(tb_ratio, 1e-6)
    balance = np.sqrt(
        np.clip(2.0 * np.minimum(ratio_maps["T"], ratio_maps["B"]) / tb_sum, 0.0, 1.0)
    )
    scores = {
        "TIL": ratio_maps["cancer"] * ratio_maps["lymph"],
        "TLS": tb_ratio * (0.65 + 0.35 * balance),
        "CAF": ratio_maps["stroma"],
    }

    return ratio_maps, scores, cell_total


# ---------------------------------------------------------------------------
# Candidate search and scoring
# ---------------------------------------------------------------------------

def _stable_ratio_score(value: float, target: float) -> float:
    if value <= 0 or target <= 0:
        return 0.0
    return float(np.exp(-abs(np.log(value / target))))


def _band_score(value: float, low: float, high: float) -> float:
    if value <= 0:
        return 0.0
    if low <= value <= high:
        return 1.0
    if value < low:
        return _stable_ratio_score(value, low)
    return _stable_ratio_score(value, high)


def _score_candidate(
    mask: np.ndarray,
    comp_areas: List[int],
    threshold: float,
    niche: str,
) -> float:
    """Compute a composite plausibility score for one candidate mask."""
    prior = _NICHE_PRIORS[niche]
    total_px = mask.size
    area = int(mask.sum())
    area_frac = area / total_px
    n_comp = len(comp_areas)

    if area == 0 or n_comp == 0:
        return 0.0

    arr = np.array(comp_areas, dtype=float)
    largest_frac = float(arr.max() / area)
    small_frac = float(arr[arr < 500].sum() / area) if len(arr) else 0.0

    area_score = (
        0.65 * _band_score(area_frac, prior["area_low"], prior["area_high"])
        + 0.35 * _stable_ratio_score(area_frac, prior["area_center"])
    )
    comp_score    = _stable_ratio_score(max(n_comp, 1), prior["comp_target"])
    largest_score = _stable_ratio_score(max(largest_frac, 1e-6), prior["largest_target"])
    thr_score     = _band_score(threshold, prior["score_low"], prior["score_high"])

    tiny_penalty = float(np.clip(1.0 - small_frac * 0.7, 0.3, 1.0))
    total = (
        0.32 * area_score
        + 0.28 * comp_score
        + 0.22 * largest_score
        + 0.18 * thr_score
    ) * tiny_penalty

    if area_frac < prior["area_low"]:
        total *= max(0.20, area_frac / max(prior["area_low"], 1e-6))

    # Niche-specific soft penalties
    if niche == "TIL" and n_comp > 280:
        total *= 0.85
    elif niche == "TLS" and largest_frac > 0.75:
        total *= 0.75
    elif niche == "CAF" and n_comp > 300:
        total *= 0.90

    return float(total)


def _apply_morphology(
    raw_mask: np.ndarray, close_r: int, open_r: int, min_comp_area: int
) -> Tuple[np.ndarray, List[int]]:
    """Close → open → remove small components."""
    m = raw_mask.astype(np.uint8)
    if close_r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_r + 1,) * 2)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if open_r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_r + 1,) * 2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m, dtype=bool)
    comp_areas: List[int] = []
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= min_comp_area:
            out[labels == lab] = True
            comp_areas.append(area)
    return out, comp_areas


def _search_one_combo(
    score_map: np.ndarray,
    cell_total: np.ndarray,
    niche: str,
    min_cells: int,
    p_high: float,
    close_r: int,
    open_r: int,
    min_comp_list: List[int],
) -> Optional[Dict]:
    """Evaluate one parameter combination, return the best min_comp result."""
    valid = cell_total >= min_cells
    vals = score_map[valid]
    if len(vals) == 0:
        return None

    threshold = float(np.percentile(vals, p_high))
    raw = valid & (score_map > threshold)

    best: Optional[Dict] = None
    for min_comp in min_comp_list:
        mask, comp_areas = _apply_morphology(raw, close_r, open_r, min_comp)
        s = _score_candidate(mask, comp_areas, threshold, niche)
        if best is None or s > best["score"]:
            best = dict(
                score=s, mask=mask, threshold=threshold,
                p_high=p_high, min_cells=min_cells,
                close_r=close_r, open_r=open_r, min_comp=min_comp,
                area_frac=float(mask.sum() / mask.size),
                n_comp=len(comp_areas),
            )
    return best


def find_best_niche_mask(
    score_map: np.ndarray,
    cell_total: np.ndarray,
    niche: str,
    workers: int = 8,
) -> Dict:
    """Grid-search over threshold/morphology parameters; return best candidate."""
    grid = _QUALITY_GRID
    p_list = [75, 80, 85] if niche == "CAF" else grid["p_high"]

    tasks = [
        (score_map, cell_total, niche, mc, ph, cr, or_, grid["min_comp"])
        for mc in grid["min_cells"]
        for ph in p_list
        for cr in grid["close_r"]
        for or_ in grid["open_r"]
    ]

    best: Optional[Dict] = None
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_search_one_combo, *t) for t in tasks]
        for fut in as_completed(futures):
            result = fut.result()
            if result is None:
                continue
            if best is None or result["score"] > best["score"]:
                best = result

    if best is None:
        # Fallback: empty mask
        best = dict(
            score=0.0, mask=np.zeros_like(score_map, dtype=bool),
            threshold=float("nan"), p_high=80, min_cells=50,
            close_r=4, open_r=1, min_comp=500,
            area_frac=0.0, n_comp=0,
        )
    return best


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_niche_overlay(
    he_img: np.ndarray,
    mask: np.ndarray,
    niche: str,
    out_path: str | Path,
    sample_name: str,
    summary: Dict,
    alpha: float = 0.65,
) -> None:
    """Overlay niche mask on H&E background and save as PNG.

    Parameters
    ----------
    he_img:      Full-resolution H&E image (uint8 RGB).
    mask:        Boolean niche mask at downsampled resolution.
    niche:       Niche name ('TIL', 'TLS', or 'CAF').
    out_path:    Output PNG path.
    sample_name: Sample identifier shown in the figure title.
    summary:     Parameter summary dict (shown in subtitle).
    alpha:       Overlay opacity (0 = transparent, 1 = opaque).
    """
    color = NICHE_COLORS[niche]
    hd, wd = mask.shape
    he_ds = cv2.resize(he_img[..., :3], (wd, hd), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    overlay = he_ds.copy()
    overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("white")
    ax.imshow(overlay)
    ax.axis("off")
    ax.set_title(
        f"{sample_name}  {niche}\n"
        f"area={summary['area_frac']:.1%}  "
        f"n_comp={summary['n_comp']}  "
        f"thr={summary['threshold']:.4f}",
        fontsize=11,
    )
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
    workers: int = 8,
) -> Dict:
    """Run the full niche pipeline for one sample.

    Parameters
    ----------
    pred_csv:    Path to SPHINX predictions CSV (must contain ``pred_celltype_name``).
    he_image:    Path to H&E OME-TIFF.
    output_dir:  Directory for output PNGs and JSON.
    sample_name: Sample identifier used in file names and figure titles.
    pixel_size:  µm/px of the H&E image (default: SPHINX training resolution).
    workers:     CPU threads for parallel candidate search.

    Returns
    -------
    Summary dict with per-niche selected parameters and scores.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[{sample_name}] Loading predictions …")
    data = load_predictions(pred_csv)
    n_cells = len(data["coords"])
    print(f"[{sample_name}]   {n_cells:,} cells")

    print(f"[{sample_name}] Loading H&E image …")
    he_img = load_he_image(he_image)
    h, w = he_img.shape[:2]
    # Grid dimensions match the H&E image dimensions downsampled by DS.
    # Coordinates (native pixels) divide by DS to map into the grid.
    h_ds = max(1, int(round(h / DS)))
    w_ds = max(1, int(round(w / DS)))
    print(f"[{sample_name}]   H&E {h}×{w}  grid {h_ds}×{w_ds}")

    print(f"[{sample_name}] Building ratio maps …")
    _, scores, cell_total = build_ratio_maps(data, h_ds, w_ds, pixel_size)

    summary: Dict = {"sample": sample_name, "niches": {}}

    for niche in ("TIL", "TLS", "CAF"):
        print(f"[{sample_name}] Searching best {niche} mask …")
        best = find_best_niche_mask(scores[niche], cell_total, niche, workers=workers)
        png_path = out_path / f"{sample_name}_{niche}.png"
        save_niche_overlay(he_img, best["mask"], niche, png_path, sample_name, best)
        print(
            f"[{sample_name}]   {niche}: area={best['area_frac']:.1%}  "
            f"n_comp={best['n_comp']}  score={best['score']:.4f}  → {png_path.name}"
        )
        summary["niches"][niche] = {k: v for k, v in best.items() if k != "mask"}

    json_path = out_path / f"{sample_name}_niche_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{sample_name}] Summary written: {json_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    single = p.add_argument_group("single sample")
    single.add_argument("--pred_csv",    default=None, help="SPHINX predictions CSV.")
    single.add_argument("--he_image",    default=None, help="H&E OME-TIFF path.")
    single.add_argument("--sample",      default=None, help="Sample name / identifier.")
    single.add_argument("--pixel_size",  type=float, default=SPHINX_PX_UM,
                        help=f"µm/px of the H&E image (default: {SPHINX_PX_UM}).")

    batch = p.add_argument_group("batch (sample list file)")
    batch.add_argument(
        "--sample_list", default=None,
        help="Text file with one sample per line: "
             "<pred_csv> <he_image> <sample_name> [<pixel_size>]",
    )

    p.add_argument("--output_dir", required=True, help="Directory for output files.")
    p.add_argument("--workers", type=int, default=8,
                   help="CPU threads for parallel candidate search (default: 8).")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.sample_list:
        entries = []
        with open(args.sample_list) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 3:
                    parts.append(str(SPHINX_PX_UM))
                pred_csv, he_img, name, px = parts
                entries.append((pred_csv, he_img, name, float(px)))
        for pred_csv, he_img, name, px in entries:
            try:
                run_sample(pred_csv, he_img, args.output_dir, name, px, args.workers)
            except Exception as exc:
                warnings.warn(f"Sample {name} failed: {exc}")
    elif args.pred_csv and args.he_image and args.sample:
        run_sample(
            args.pred_csv, args.he_image, args.output_dir,
            args.sample, args.pixel_size, args.workers,
        )
    else:
        _build_parser().print_help()
        raise SystemExit(
            "\nProvide either --pred_csv / --he_image / --sample "
            "or --sample_list."
        )


if __name__ == "__main__":
    main()
