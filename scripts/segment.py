"""
SPHINX-Seg: Nucleus Segmentation Inference Script

Runs SPHINX-Seg (a fine-tuned Cellpose model) on an H&E OME-TIFF whole-slide
image to produce per-nucleus instance masks and a cell coordinate table
compatible with the SPHINX inference pipeline.

Outputs (written to --output_dir):
  nuclei_mask.npz        -- compressed 2-D uint32 instance mask
                            (pixel value = cell index, 0 = background)
  cell_table_raw.csv     -- one row per detected nucleus with columns:
                            cell_id, centroid_x, centroid_y,
                            morph_area, morph_bbox_width, morph_bbox_height

Usage example:
  python scripts/segment.py \
      --he_image   /data/sample.ome.tiff \
      --model_path /path/to/SPHINX-Seg \
      --output_dir /data/sphinx_ready/sample

The output CSV is the starting point for scripts/prepare_external_sample.py.
"""

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.measure import regionprops

log = logging.getLogger(__name__)


# ── Default inference parameters derived from SPHINX-Seg training config ─────
DEFAULT_CELLPROB_THRESHOLD = -1.5   # Lower = more permissive nucleus detection
DEFAULT_FLOW_THRESHOLD = 0.4        # Cellpose flow error threshold
DEFAULT_MIN_SIZE = 40               # Minimum nucleus area in pixels
DEFAULT_TILE_SIZE = 1024            # Tile width/height for tiled WSI inference
DEFAULT_TILE_OVERLAP = 128          # Overlap between adjacent tiles (pixels)
DEFAULT_CHANNELS = [0, 0]           # [cyto_channel, nucleus_channel]; 0=grayscale


def load_he_image(he_image_path: str) -> np.ndarray:
    """Load an H&E OME-TIFF or standard TIFF as a uint8 RGB array.

    For multi-page or pyramidal OME-TIFFs the highest-resolution level (page 0)
    is loaded.  Very large images should be processed in tiles via run_tiled().

    Args:
        he_image_path: Absolute path to the H&E image file.

    Returns:
        numpy uint8 array of shape [H, W, 3] (RGB).
    """
    try:
        import tifffile
        with tifffile.TiffFile(he_image_path) as tif:
            img = tif.pages[0].asarray()
    except ImportError:
        from PIL import Image
        img = np.array(Image.open(he_image_path))

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]

    if img.dtype != np.uint8:
        img_max = img.max()
        img = (img / max(img_max, 1) * 255).astype(np.uint8)

    return img


def run_tiled(
    model,
    image: np.ndarray,
    tile_size: int,
    overlap: int,
    cellprob_threshold: float,
    flow_threshold: float,
    min_size: int,
) -> np.ndarray:
    """Run Cellpose inference on a large image using a sliding tile window.

    Tiles are stitched with simple maximum-overlap merging at boundaries.
    Cell indices are globally unique across all tiles.

    Args:
        model:               Loaded CellposeModel instance.
        image:               Full-resolution H&E image [H, W, 3] uint8.
        tile_size:           Width and height of each tile in pixels.
        overlap:             Pixel overlap between adjacent tiles.
        cellprob_threshold:  Cellpose cell probability threshold.
        flow_threshold:      Cellpose flow error threshold.
        min_size:            Minimum nucleus size to keep (pixels).

    Returns:
        Full-resolution instance mask [H, W] uint32; 0 = background.
    """
    H, W = image.shape[:2]
    full_mask = np.zeros((H, W), dtype=np.uint32)
    global_id = 1
    step = tile_size - overlap

    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)
            tile = image[y0:y1, x0:x1]

            masks, _, _ = model.eval(
                tile,
                diameter=None,
                channels=DEFAULT_CHANNELS,
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold,
                min_size=min_size,
            )
            # Remap local tile indices to global IDs
            local_ids = np.unique(masks[masks > 0])
            for lid in local_ids:
                region = (masks == lid)
                # Only place cell where full_mask is still background
                write_region = region & (full_mask[y0:y1, x0:x1] == 0)
                full_mask[y0:y1, x0:x1][write_region] = global_id
                global_id += 1

    log.info("Tiled segmentation complete: %d nuclei detected", global_id - 1)
    return full_mask


def mask_to_cell_table(mask: np.ndarray) -> pd.DataFrame:
    """Convert an instance mask to a cell coordinate table.

    Computes per-nucleus centroids and bounding-box morphology attributes.

    Args:
        mask: Instance mask [H, W] uint32 where each non-zero value is one cell.

    Returns:
        DataFrame with columns: cell_id, centroid_x, centroid_y,
        morph_area, morph_bbox_width, morph_bbox_height.
    """
    props = regionprops(mask.astype(np.int32))
    rows = []
    for p in props:
        cy, cx = p.centroid
        minr, minc, maxr, maxc = p.bbox
        rows.append({
            "cell_id":           int(p.label),
            "centroid_x":        float(cx),
            "centroid_y":        float(cy),
            "morph_area":        int(p.area),
            "morph_bbox_width":  int(maxc - minc),
            "morph_bbox_height": int(maxr - minr),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run SPHINX-Seg nucleus segmentation on an H&E WSI."
    )
    parser.add_argument(
        "--he_image", required=True,
        help="Path to the H&E image (OME-TIFF or standard TIFF)."
    )
    parser.add_argument(
        "--model_path", required=True,
        help="Path to the SPHINX-Seg Cellpose model directory."
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory where nuclei_mask.npz and cell_table_raw.csv will be written."
    )
    parser.add_argument(
        "--cellprob_threshold", type=float, default=DEFAULT_CELLPROB_THRESHOLD,
        help=f"Cellpose cell probability threshold (default {DEFAULT_CELLPROB_THRESHOLD}). "
             "Lower values detect more (but potentially noisier) nuclei."
    )
    parser.add_argument(
        "--flow_threshold", type=float, default=DEFAULT_FLOW_THRESHOLD,
        help=f"Cellpose flow error threshold (default {DEFAULT_FLOW_THRESHOLD}). "
             "Higher values are more permissive."
    )
    parser.add_argument(
        "--min_size", type=int, default=DEFAULT_MIN_SIZE,
        help=f"Minimum nucleus area in pixels to keep (default {DEFAULT_MIN_SIZE})."
    )
    parser.add_argument(
        "--tile_size", type=int, default=DEFAULT_TILE_SIZE,
        help=f"Tile size for tiled WSI inference (default {DEFAULT_TILE_SIZE} px)."
    )
    parser.add_argument(
        "--tile_overlap", type=int, default=DEFAULT_TILE_OVERLAP,
        help=f"Overlap between adjacent tiles in pixels (default {DEFAULT_TILE_OVERLAP})."
    )
    parser.add_argument(
        "--gpu", action="store_true", default=None,
        help="Use GPU for Cellpose inference. Defaults to True if CUDA is available."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    log.info("Loading SPHINX-Seg model from %s", args.model_path)
    import torch
    from cellpose.models import CellposeModel
    use_gpu = args.gpu if args.gpu is not None else torch.cuda.is_available()
    model = CellposeModel(pretrained_model=args.model_path, gpu=use_gpu)

    # ── Load image ────────────────────────────────────────────────────────────
    log.info("Loading H&E image from %s", args.he_image)
    t0 = time.time()
    image = load_he_image(args.he_image)
    log.info("Image loaded: %s  (%.1f s)", image.shape, time.time() - t0)

    # ── Segmentation ──────────────────────────────────────────────────────────
    log.info("Running tiled segmentation  tile=%d  overlap=%d",
             args.tile_size, args.tile_overlap)
    t1 = time.time()
    mask = run_tiled(
        model, image,
        tile_size=args.tile_size,
        overlap=args.tile_overlap,
        cellprob_threshold=args.cellprob_threshold,
        flow_threshold=args.flow_threshold,
        min_size=args.min_size,
    )
    log.info("Segmentation done in %.1f s", time.time() - t1)

    # ── Save outputs ──────────────────────────────────────────────────────────
    # Saved under key="mask" (raw nucleus pixels only).
    # To render whole-cell staining maps with render_staining_maps.py, run a
    # nucleus-expansion step first to produce the "nuclei_exp" key (cell-body
    # approximation).  Pass --mask_key mask to render_staining_maps.py if you
    # intentionally want nucleus-level coloring without the expansion step.
    mask_path = output_dir / "nuclei_mask.npz"
    np.savez_compressed(str(mask_path), mask=mask)
    log.info("Saved instance mask -> %s", mask_path)

    cell_table = mask_to_cell_table(mask)
    csv_path = output_dir / "cell_table_raw.csv"
    cell_table.to_csv(str(csv_path), index=False)
    log.info("Saved cell table (%d cells) -> %s", len(cell_table), csv_path)


if __name__ == "__main__":
    main()
