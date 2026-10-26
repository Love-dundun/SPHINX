# SPHINX

**Spatial Proteomic inference from Histology with Integrated Neighborhood conteXt**

SPHINX predicts multiplex protein expression, cell identity and spatial tissue
organisation directly from routine haematoxylin and eosin (H&E) whole-slide
images at single-cell resolution. The model combines a dual-scale vision
transformer encoder, a biology-prior graph attention module, a spatial
neighbourhood graph attention module and a router–multi-gate fusion, and it is
trained on co-registered H&E images and spatially resolved protein measurements.

This repository contains the code used to produce the results reported in the
accompanying manuscript.

---

## What SPHINX outputs

Given the centroid of a segmented nucleus and the surrounding H&E patches:

- **18 protein channels** — ATPase, CD11c, CD20, CD3, CD31, CD4, CD45RA, CD56,
  CD68, CD8, CTLA4, FOXP3, HLA-DR, KI67, PD1, PDL1, PanCK, aSMA
- **12 cell types** — B cells, cancer cells, dendritic cells, endothelial cells,
  epithelial cells, fibroblasts, macrophages, monocytes, NK cells, neutrophils,
  smooth muscle cells, T cells

Downstream modules convert these per-cell predictions into pseudo-multiplex
images, spatial niches (TIL / TLS / CAF) and immune hot/cold phenotypes.

## Repository contents

| Path | Purpose | Figure in paper |
|---|---|---|
| `scripts/segment.py` | SPHINX-Seg nucleus segmentation on a WSI | Fig. 1c |
| `scripts/prepare_external_sample.py` | Convert an external cell table into the SPHINX sample layout | — |
| `scripts/precompute_knn.py` | Build and cache the K-nearest-neighbour spatial graphs | — |
| `scripts/train.py` | Train SPHINX | — |
| `scripts/predict.py` | Two-pass single-cell inference | Figs. 2–4 |
| `sphinx/model.py` | Dual-branch encoder, ProteinGAT, CellGAT, Router–Multi-Gate | Fig. 1c |
| `sphinx/losses.py` | Masked MSE, Pearson correlation, biology-prior and uncertainty weighting | — |
| `sphinx/protein_prior.py` | Protein co-expression / mutual-exclusion prior matrix | — |
| `downstream/render_staining_maps.py` | Render pseudo-multiplex protein and cell-type maps | Figs. 2, 3 |
| `downstream/niche_pipeline.py` | TIL, TLS and CAF-enriched niche detection | Fig. 5 |
| `downstream/hotcold.py` | Immune hot/cold tumour phenotyping | Fig. 6 |

Documentation: [`docs/data_preparation.md`](docs/data_preparation.md) for the
sample directory layout and the required cell-table columns,
[`docs/backbone_guide.md`](docs/backbone_guide.md) for the backbone options, and
[`downstream/README.md`](downstream/README.md) for the downstream modules.

## Requirements

Python ≥ 3.10 and the packages listed in `requirements.txt`:

```
torch>=2.1.0
torchvision>=0.16.0
timm>=0.9.12,<2.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
scikit-learn>=1.3.0
tifffile>=2023.7.10
zarr>=2.16.0
opencv-python>=4.8.0
matplotlib>=3.7.0
scikit-image>=0.21.0
cellpose>=3.0.0
```

A CUDA-capable GPU is strongly recommended; inference runs on CPU but is slow.

```bash
cd SPHINX
pip install -r requirements.txt
```

## Quick start (inference only)

The minimum input is **one H&E whole-slide image**. The three steps below take a
slide from pixels to a per-cell prediction table.

```bash
# 1. Nuclear segmentation (SPHINX-Seg)
python scripts/segment.py \
    --he_image    /path/to/slide.ome.tiff \
    --model_path  /path/to/sphinx_seg \
    --output_dir  /path/to/sample_dir \
    --gpu

# 2. (External data only) convert an existing cell table to the SPHINX layout
python scripts/prepare_external_sample.py \
    --input_csv   /path/to/cells.csv \
    --he_image    /path/to/slide.ome.tiff \
    --output_dir  /path/to/sample_dir \
    --pixel_size  0.3125

# 3. Single-cell inference (two-pass, ProteinGAT + CellGAT)
python scripts/predict.py \
    --sample_dir  /path/to/sample_dir \
    --checkpoint  /path/to/sphinx_hnsc_vit.pth \
    --output_csv  /path/to/sample_dir/predictions.csv \
    --batch_size  256
```

K-nearest-neighbour graphs are built on the fly by `predict.py`; to build them
once for many samples, use `scripts/precompute_knn.py --list_file samples.txt`.

`predict.py` reads the prepared cell table from
`<sample_dir>/sphinx_cell_table.csv` (written by `prepare_external_sample.py`);
the per-cell predictions are written to the path given by `--output_csv`.

## Training

Training uses one directory per specimen (see
[`docs/data_preparation.md`](docs/data_preparation.md)) and reproduces the
settings reported in the paper:

```bash
python scripts/train.py \
    --sample_dirs ./data/sample_01 ./data/sample_02 \
    --epochs 50 \
    --batch_size 256 \
    --lr 2e-4 \
    --backbone_lr_scale 0.1 \
    --seed 42 \
    --focal_gamma 2.0 \
    --label_smoothing 0.05 \
    --pearson_weight 0.4 \
    --bio_prior_weight 0.05 \
    --deep_sup_weight 0.3 \
    --save_dir ./runs
```

The optimisation uses AdamW with a weight decay of 1 × 10⁻⁴, a learning rate of
2 × 10⁻⁴ for newly initialised modules and 2 × 10⁻⁵ (10%) for the pretrained ViT
backbones. Losses are combined by learned per-task uncertainty weighting, and a
fixed seed (42) is used for initialisation, augmentation and mini-batch shuffling.

Within each training specimen, cells are additionally split 70:30 into
optimisation and internal-validation subsets for checkpoint selection; this
cell-level split never moves cells across the patient-level partitions.

## Downstream analyses

```bash
# Pseudo-multiplex protein and cell-type maps
python downstream/render_staining_maps.py \
    --pred_csv   /path/to/sample_dir/predictions.csv \
    --mask_npz   /path/to/sample_dir/nuclei_mask.npz \
    --mask_key   mask \
    --output_dir /path/to/rendered

# TIL / TLS / CAF-enriched niches (60 um neighbourhood kernel, Dice vs reference)
python downstream/niche_pipeline.py \
    --pred_csv   /path/to/sample_dir/predictions.csv \
    --he_image   /path/to/slide.ome.tiff \
    --output_dir /path/to/niches \
    --pixel_size 0.3125

# Immune hot/cold phenotyping
python downstream/hotcold.py \
    --pred_csv   /path/to/sample_dir/predictions.csv \
    --he_image   /path/to/slide.ome.tiff \
    --output_dir /path/to/hotcold \
    --pixel_size 0.3125
```

Both niche and hot/cold scripts accept `--sample_list` to process a batch of
samples and write one summary JSON per specimen.

## Image scale

SPHINX works at the native resolution of the input image, and all spatial
parameters are expressed in pixels. For the data used in the paper, the
fluorescence H&E images from the G4X platform have a pixel size of
**0.3125 µm**, at which the 96- and 192-pixel views span 30.0 and 60.0 µm. TCGA
diagnostic slides are mostly 40× (~0.2525 µm per pixel), so the same views cover
~24.2 and ~48.5 µm there. Pass the correct `--pixel_size` for your own data;
no resampling to a common pixel size is performed.

## Model weights

Two sets of weights are distributed with the **v1.0.0 release** of this
repository:

| File | Used by | Content |
|---|---|---|
| `sphinx_hnsc_vit.pth` | `scripts/predict.py --checkpoint` | SPHINX model trained on the HNSC cohort |
| `sphinx_seg.zip` (unzip to `sphinx_seg/`) | `scripts/segment.py --model_path` | SPHINX-Seg nucleus segmentation model (Cellpose checkpoint directory) |

The SPHINX weights were trained on G4X-profiled HNSC specimens. **For other
scanners, staining panels or cancer types the model should be retrained** on
labelled data from the platform of interest. Training the SPHINX model requires
only ImageNet-initialised ViT-Base backbones (loaded automatically through
`timm`), so the training code is self-contained; segmentation additionally
requires the SPHINX-Seg weights listed above.

## Reproducing the paper

| Analysis | Entry point |
|---|---|
| Segmentation and cell correspondence (Fig. 1) | `scripts/segment.py` |
| Protein prediction and tissue reconstruction (Figs. 2–3) | `scripts/predict.py`, `downstream/render_staining_maps.py` |
| Cell typing and cross-cancer evaluation (Fig. 4) | `scripts/predict.py` |
| Spatial niches (Fig. 5) | `downstream/niche_pipeline.py` |
| Immune phenotypes and survival (Fig. 6) | `downstream/hotcold.py` |

Metric definitions (pooled cell-weighted and sample-equal-weight aggregations,
bootstrap confidence intervals, paired tests) are described in the Methods of
the paper.

## Citation

If you use this code, please cite:

> Li, Y., Yuan, H., Lin, Q.N., Li, D., Ren, Y., Hu, E., An, J., Liu, Q.,
> Prystowsky, M.B., Xiang, W., Deng, W., Wu, D. & Yin, S.
> *SPHINX: Spatial Proteomic inference from Histology with Integrated
> Neighborhood conteXt.* (2026).

Citation metadata is also provided in [`CITATION.cff`](CITATION.cff).

## License

Released under the MIT License; see [`LICENSE`](LICENSE).

## Contact

Please open an issue for questions about running the code, or contact the
corresponding authors listed in the paper.
