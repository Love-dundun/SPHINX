# SPHINX downstream analysis

This directory contains three modules that operate on the per-cell prediction
table (`sphinx_predictions.csv`) produced by `scripts/predict.py` together with
the nucleus mask (`nuclei_mask.npz`) written by `scripts/segment.py`.

| Module | Purpose | Figure in paper |
|---|---|---|
| `render_staining_maps.py` | Pseudo-multiplex protein and cell-type maps | Figs. 2, 3 |
| `niche_pipeline.py` | TIL, TLS and CAF-enriched spatial niches | Fig. 5 |
| `hotcold.py` | Immune hot/cold tumour phenotyping | Fig. 6 |

All three write their results into `--output_dir` and accept a single sample
(`--pred_csv` + `--he_image`) or a batch (`--sample_list`).

---

## render_staining_maps.py — pseudo-multiplex maps

Assigns every predicted value to its nucleus mask region and renders the result
as a calibrated OME-TIFF, so that the predictions can be compared directly with
reference multiplex images.

```bash
python downstream/render_staining_maps.py \
    --pred_csv    /path/to/sphinx_predictions.csv \
    --mask_npz    /path/to/nuclei_mask.npz \
    --output_dir  /path/to/rendered \
    --sample      my_sample \
    --mask_key    mask \
    --mode        all \
    --single_proteins CD3 CD8 PanCK PDL1
```

| Parameter | Default | Description |
|---|---|---|
| `--pred_csv` | required | Per-cell prediction table from `predict.py` |
| `--mask_npz` | required | Nucleus mask archive from `segment.py` |
| `--output_dir` | required | Directory for the rendered maps |
| `--sample` | stem of `--pred_csv` | Prefix for output file names |
| `--mode` | `all` | `all`, `protein` or `celltype` |
| `--single_proteins` | all 18 | Protein channels to render individually |
| `--expand_px` | `-1` | Pixels added to each nucleus before rendering; `-1` uses 25 px for the `mask` key and 0 otherwise |
| `--mask_key` | `nuclei_exp` | Array inside `nuclei_mask.npz`. **`scripts/segment.py` writes the key `mask`**, so pass `--mask_key mask` for its output; the 25-pixel whole-cell expansion is then applied automatically |
| `--legacy_extra_maps` | off | Also write the additional grouped protein map set (structural / immune channel groups) |
| `--overwrite` | off | Replace existing outputs |

Outputs are written to two subdirectories of `--output_dir` (OME-TIFF, pixel size
inherited from the source image metadata):

- `protein/` — `<sample>_allprot.ome.tiff`, `<sample>_structural.ome.tiff`, `<sample>_immune.ome.tiff`, and one
  `<sample>_<protein>.ome.tiff` per protein listed in `--single_proteins`
- `celltype/` — `<sample>_celltype.ome.tiff` (RGB cell-type map)

---

## niche_pipeline.py — TIL / TLS / CAF niches

Builds cell-type ratio maps from the predicted table, smooths them with a
**circular neighbourhood kernel of 60 µm radius (192 px at 0.3125 µm/px)**, and
thresholds the resulting scores into niche masks:

- **TIL** = tumour fraction × lymphocyte fraction
- **TLS** = T/B co-occurrence fraction × (0.65 + 0.35 × T–B balance)
- **CAF** = stromal fraction

The percentile threshold, morphology radii and minimum component area are
selected per specimen from a discrete grid (70th–85th percentile; closing radius
4–8 px; opening radius 1 px; minimum area 200–1,000 px²) and scored against soft
biological priors for the expected area fraction, component number and
largest-component fraction, rather than against a single fixed cut-off.

```bash
python downstream/niche_pipeline.py \
    --pred_csv    /path/to/sphinx_predictions.csv \
    --he_image    /path/to/slide.ome.tiff \
    --output_dir  /path/to/niches \
    --sample      my_sample \
    --pixel_size  0.3125
```

| Parameter | Default | Description |
|---|---|---|
| `--pred_csv` | — | Per-cell prediction table (single-sample mode) |
| `--he_image` | — | H&E OME-TIFF, used to draw the overlays |
| `--sample` | stem of `--pred_csv` | Sample identifier |
| `--sample_list` | — | Text file, one sample per line: `pred_csv he_image sample pixel_size` (space-separated) |
| `--pixel_size` | `0.3125` | µm per pixel of the input image |
| `--output_dir` | required | Directory for outputs |
| `--workers` | `8` | Parallel worker processes in batch mode |

Outputs: `<sample>_TIL.png`, `<sample>_TLS.png`, `<sample>_CAF.png` and
`<sample>_niche_summary.json` (mask areas, thresholds and quality metrics).

---

## hotcold.py — immune hot/cold phenotyping

Rasterises predicted cancer cells and lymphocytes, derives a tumour mask from
cancer-cell density, and classifies every position inside the tumour as immune
**hot** or **cold** from a local hotness score:

```
local_hotness = lymphocytes / (lymphocytes + malignant cells)      # Gaussian kernel
global_score  = lymphocytes_in_tumour / (lymphocytes_in_tumour + malignant_in_tumour)
threshold     = clip(0.12 − 0.35 × global_score, 0.04, 0.12)
```

The slide-level threshold adapts to the overall immune state of the specimen, so
that immune-rich sections are not artificially inflated by a fixed cut-off.

```bash
python downstream/hotcold.py \
    --pred_csv    /path/to/sphinx_predictions.csv \
    --he_image    /path/to/slide.ome.tiff \
    --output_dir  /path/to/hotcold \
    --sample      my_sample \
    --pixel_size  0.3125
```

| Parameter | Default | Description |
|---|---|---|
| `--pred_csv` | — | Per-cell prediction table (single-sample mode) |
| `--he_image` | — | H&E OME-TIFF |
| `--sample` | stem of `--pred_csv` | Sample identifier |
| `--sample_list` | — | Text file, one sample per line: `pred_csv he_image sample pixel_size` (space-separated) |
| `--pixel_size` | `0.3125` | µm per pixel of the input image |
| `--output_dir` | required | Directory for outputs |
| `--keep_tumor_components` | `2` | Number of largest connected tumour regions retained |

Outputs: `<sample>_hotcold.png` (overlay) and `<sample>_hotcold.json` (global
immune score, adaptive threshold, hot and cold tumour area fractions).

---

## Notes on pixel size

All three modules work in the pixel space of the input image; `--pixel_size`
converts the fixed pixel radii into micrometres. For the G4X platform used to
train SPHINX the native pixel size is **0.3125 µm/px**; for other scanners read
the value from the OME-TIFF/SVS metadata (or QuPath) and pass it explicitly. No
resampling to a common pixel size is performed.
