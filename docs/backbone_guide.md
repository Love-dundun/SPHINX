# SPHINX Backbone Guide

## Overview

SPHINX uses **ViT-Base/16** (Vision Transformer, base model with 16×16 patch
size) as its image encoder backbone.  Two ViT-Base instances run in parallel —
one on a 96 px detail patch centred on the cell, and one on a 192 px context
patch — and their 768-d CLS tokens are fused via a learned scalar weight.

| Component | Value |
|---|---|
| Architecture | ViT-Base/16 (Dosovitskiy et al., ICLR 2021) |
| Pretraining | ImageNet-21k → ImageNet-1k fine-tune (timm default) |
| Input size | 224 × 224 px (both branches) |
| Output feature | 768-d CLS token |
| Parameters per branch | ~86 M |
| Total (both branches + heads) | ~176 M |

---

## Inference with the SPHINX-HNSC checkpoint

The provided example checkpoint (`sphinx_hnsc_vit.pth`) is self-contained: it
stores the full ViT-Base backbone weights internally.  No separate backbone
download is needed.

```bash
python scripts/predict.py \
    --sample_dir   /path/to/sample \
    --checkpoint   /path/to/sphinx_hnsc_vit.pth \
    --output_csv   sphinx_predictions.csv
```

In Python:

```python
import torch
from sphinx.model import ViTProteinModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt   = torch.load("sphinx_hnsc_vit.pth", map_location=device)
model  = ViTProteinModel(num_proteins=18, num_celltypes=12, pretrained=False)
model.load_state_dict(ckpt["model"])
model.eval()
```

---

## Training a new SPHINX model from scratch

When training on a new cohort, the ViT-Base backbone is initialised with
ImageNet pretrained weights automatically (via `timm`) when `pretrained=True`
(the default in `scripts/train.py`).  No additional weight files are needed.

```bash
python scripts/train.py \
    --sample_dirs  /path/to/sample_001 /path/to/sample_002 \
    --epochs       50 \
    --save_dir     ./runs
```

The backbone learning rate is automatically scaled by `--backbone_lr_scale`
(default `0.1`) relative to the head learning rate (`--lr`), which preserves
the pretrained representations during early training.

---

## Drop-path regularisation

The ViT-Base backbone uses stochastic depth (drop-path) regularisation
fixed at `0.1` in the `ViTProteinModel` constructor.  This value was selected
to balance regularisation and training stability on the HNSC cohort.

---

## Why ViT-Base?

ViT-Base/16 was selected as the primary encoder because:

- **Spatial adaptability**: The attention mechanism attends globally over the
  224 px input tile, capturing both cellular texture and local tissue structure
  without inductive biases from convolutions.
- **Reproducibility**: The ImageNet-pretrained weights are publicly available
  via `timm` with a single function call, making SPHINX fully reproducible
  without proprietary pretrained models.
- **Performance**: ViT-Base achieves state-of-the-art protein and cell-type
  prediction performance on the HNSC cohort used to train SPHINX-HNSC.
