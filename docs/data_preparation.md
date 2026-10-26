# Data Preparation for External Datasets

This guide explains how to prepare an external cell segmentation table and
H&E whole-slide image (WSI) for use with SPHINX.

---

## Quick start

```bash
python scripts/prepare_external_sample.py \
    --input_csv   /path/to/my_cells.csv \
    --he_image    /path/to/my_slide.tiff \
    --output_dir  /path/to/sphinx_ready/sample_01 \
    --pixel_size  0.25
```

This creates a SPHINX-ready sample directory:

```
sample_01/
├── sphinx_cell_table.csv   ← SPHINX cell table
├── sample_01_HE.ome.tiff             ← symlink to H&E image
└── knn_k6.npz                        ← KNN neighbour cache
```

---

## Required columns

SPHINX needs a per-cell CSV table with the following columns.

### Always required (inference)

| Column | Description |
|---|---|
| `centroid_x` | Cell centroid X in pixels (horizontal axis) |
| `centroid_y` | Cell centroid Y in pixels (vertical axis) |

Coordinates must be in the **same pixel coordinate system** as the
accompanying H&E OME-TIFF (i.e. raw pixel coordinates at the scan
resolution, not rescaled or physical units).

### Required for training (optional for inference-only)

| Column group | Example columns |
|---|---|
| Protein expression | `CD3_intensity_mean`, `CD20_intensity_mean`, … |
| Cell-type labels | `celltype_T_Cells`, `celltype_Cancer_Cells`, … (one-hot) |

If protein columns are absent or contain `NaN`, the corresponding
protein channels are excluded from the regression loss via a binary
protein mask — the model can still be trained on the available channels.

### Optional (improves training)

| Column group | Example columns |
|---|---|
| Cell morphology | `morph_area`, `morph_eccentricity`, `morph_solidity`, … |

---

## Supported input formats

`prepare_external_sample.py` auto-detects the input format.  You can
also specify it explicitly with `--format`.

### QuPath

Columns exported by **Measure Cell Detections** or **Cell Detection**:

| QuPath column | SPHINX mapping |
|---|---|
| `Centroid X µm` / `Centroid Y µm` | converted to pixels using `--pixel_size` |
| `Centroid X` / `Centroid Y` (pixel export) | used directly |
| `Cell: CD3 mean` | `CD3_intensity_mean` |
| `Cell: PanCK mean` | `PanCK_intensity_mean` |

```bash
python scripts/prepare_external_sample.py \
    --input_csv   qupath_cells.csv \
    --he_image    slide.tiff \
    --output_dir  ./sample_01 \
    --pixel_size  0.25 \
    --format      qupath \
    --protein_prefix "Cell: "
```

### StarDist / DeepCell

Polygon centroid CSVs with `centroid-0` (row = Y) and `centroid-1`
(col = X) columns:

```bash
python scripts/prepare_external_sample.py \
    --input_csv  stardist_cells.csv \
    --he_image   slide.tiff \
    --output_dir ./sample_01 \
    --pixel_size 0.25 \
    --format     stardist
```

### Cellpose

Plain CSV with `x` / `y` (or `centroid_x` / `centroid_y`) pixel columns:

```bash
python scripts/prepare_external_sample.py \
    --input_csv  cellpose_cells.csv \
    --he_image   slide.tiff \
    --output_dir ./sample_01 \
    --pixel_size 0.25 \
    --format     cellpose
```

### Generic / custom

If your CSV uses non-standard column names, specify them directly:

```bash
python scripts/prepare_external_sample.py \
    --input_csv  my_cells.csv \
    --he_image   slide.tiff \
    --output_dir ./sample_01 \
    --pixel_size 0.25 \
    --x_col      "Cell_X" \
    --y_col      "Cell_Y"
```

---

## Coordinate rescaling

SPHINX was trained with a pixel size of **0.3125 µm/px**.
If your slide has a different resolution, pass `--pixel_size <value>`.
The script rescales the stored centroid coordinates so that all downstream
modules operate at the correct spatial scale.

The H&E image itself is **not** resampled — only the coordinates in the CSV
are adjusted.

Common scanner pixel sizes:

| Scanner / objective | µm/px |
|---|---:|
| Leica GT450, 40× | 0.2630 |
| Aperio AT2, 40× | 0.2528 |
| Hamamatsu NanoZoomer, 40× | 0.2274 |
| 20× equivalent (various) | ~0.50 |
| **SPHINX training (default)** | **0.3125** |

To read the pixel size from an OME-TIFF:

```python
import tifffile
with tifffile.TiffFile("slide.ome.tiff") as tif:
    # Physical pixel size is stored in OME-XML metadata
    ome = tif.ome_metadata
    print(ome)  # search for PhysicalSizeX
```

---

## Inference-only mode (no protein ground truth)

If your dataset has no multiplexed protein measurements (e.g. pure H&E
without IMC/CODEX), you can still run SPHINX inference to obtain
predicted protein expression and cell-type labels.

In this case, omit protein columns; the script will fill them with `NaN`
and SPHINX will skip the protein regression loss.

```bash
python scripts/predict.py \
    --sample_dir  /path/to/sphinx_ready/sample_01 \
    --checkpoint  /path/to/best.pth \
    --output_csv  sphinx_predictions.csv
```

---

## H&E image requirements

| Property | Requirement |
|---|---|
| Format | OME-TIFF (`.ome.tiff`) recommended; regular TIFF also works |
| Colour | RGB (3-channel uint8) |
| File name | Must end with `_HE.ome.tiff` in the sample directory |
| Tiling | Not required; zarr-based lazy reading handles large files |

To convert a TIFF to OME-TIFF format:

```bash
# Using tifffile (Python)
python -c "
import tifffile, numpy as np
img = tifffile.imread('my_slide.tiff')
tifffile.imwrite('my_slide_HE.ome.tiff', img, photometric='rgb')
"

# Using bfconvert (Bio-Formats command-line tool)
bfconvert my_slide.svs my_slide_HE.ome.tiff
```

---

## Cell-type label mapping

`prepare_external_sample.py` automatically maps common label strings to
SPHINX canonical cell-type names.  For example:

| Input label | SPHINX name |
|---|---|
| `"T cell"`, `"T_Cells"`, `"tcell"` | `T_Cells` |
| `"cancer"`, `"tumor"`, `"Cancer_Cells"` | `Cancer_Cells` |
| `"fibroblast"`, `"Fibroblasts"` | `Fibroblasts` |

If your labels do not match any known alias, the corresponding cells will
have all-zero one-hot columns and will be treated as unlabelled during
training (excluded from the cell-type loss via the validity mask).

Use `--celltype_col <column_name>` to specify which column holds your
cell-type labels.

---

## Batch preparation

Prepare multiple samples in sequence:

```bash
for sample in /path/to/cohort/*/; do
    name=$(basename "$sample")
    python scripts/prepare_external_sample.py \
        --input_csv  "$sample/cells.csv" \
        --he_image   "$sample/slide.tiff" \
        --output_dir "/path/to/sphinx_ready/$name" \
        --pixel_size 0.25
done
```
