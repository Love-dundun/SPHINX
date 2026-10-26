import argparse
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import ConcatDataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sphinx.dataset import CELLTYPE_NAMES, PROTEIN_COLS, prepare_sample
from sphinx.losses import (
    NEG_PAIRS,
    POS_PAIRS,
    bio_prior_loss,
    compute_class_weights,
    focal_loss,
    masked_mse_loss,
    pearson_loss,
)
from sphinx.model import ViTProteinModel


def amp_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def train_epoch(model, loader, optimizer, scaler, scheduler, class_weight, args, device, epoch, total_epochs):
    model.train()
    model.reset_gate_stats()
    t0 = time.time()
    total_protein = 0.0
    total_celltype = 0.0
    total_n = 0

    gt_mix = max(0.0, 1.0 - epoch / max(total_epochs - 1, 1))
    if hasattr(model, "gt_mix_ratio"):
        model.gt_mix_ratio = gt_mix

    has_uncertainty = hasattr(model, "uncertainty_loss")

    for batch in loader:
        detail = batch["detail"].to(device, non_blocking=True)
        context = batch["context"].to(device, non_blocking=True)
        protein_gt = batch["protein"].to(device, non_blocking=True)
        protein_mask = batch["protein_mask"].to(device, non_blocking=True)
        celltype_gt = batch["celltype"].to(device, non_blocking=True)
        ct_valid = batch["celltype_valid_mask"].to(device, non_blocking=True)
        nb_protein = batch["nb_protein"].to(device, non_blocking=True)
        adj = batch["adj"].to(device, non_blocking=True)
        batch_size = detail.size(0)

        optimizer.zero_grad(set_to_none=True)
        with amp_context(device):
            out = model(
                detail,
                context,
                protein_gt=protein_gt,
                protein_mask=protein_mask,
                nb_protein=nb_protein,
                adj=adj,
            )

            protein_pred = out["protein_pred"]
            logits_a = out["celltype_logits_A"]
            logits_b = out.get("celltype_logits_B")
            router_logits = out.get("router_logits")

            protein_loss = masked_mse_loss(protein_pred, protein_gt, protein_mask)
            protein_loss = protein_loss + args.pearson_weight * pearson_loss(protein_pred, protein_gt, protein_mask)
            if "protein_aux" in out:
                protein_loss = protein_loss + args.deep_sup_weight * masked_mse_loss(
                    out["protein_aux"], protein_gt, protein_mask
                )

            ct_a_loss = focal_loss(
                logits_a,
                celltype_gt,
                weight=class_weight,
                gamma=args.focal_gamma,
                label_smoothing=args.label_smoothing,
                ct_valid_mask=ct_valid,
            )
            if logits_b is not None:
                ct_b_loss = focal_loss(
                    logits_b,
                    celltype_gt,
                    weight=class_weight,
                    gamma=args.focal_gamma,
                    label_smoothing=args.label_smoothing,
                    ct_valid_mask=ct_valid,
                )
            else:
                ct_b_loss = torch.zeros(1, device=device)

            if router_logits is not None:
                # Only train the router on cells whose GT class exists in this sample;
                # cells with an absent GT class (e.g. argmax of all-zero one-hot = 0
                # when class 0 is not in the panel) would inject incorrect supervision.
                valid_cells = ct_valid[torch.arange(batch_size), celltype_gt] > 0.5
                router_logits = router_logits.masked_fill(ct_valid < 0.5, -1e4)
                if valid_cells.any():
                    router_loss = F.cross_entropy(
                        router_logits[valid_cells], celltype_gt[valid_cells],
                        weight=class_weight,
                    )
                else:
                    router_loss = torch.zeros(1, device=device)
            else:
                router_loss = torch.zeros(1, device=device)

            prior_loss = bio_prior_loss(protein_pred, POS_PAIRS, NEG_PAIRS) if args.bio_prior_weight > 0 else torch.zeros(1, device=device)

            if has_uncertainty:
                tasks = [protein_loss, ct_a_loss, router_loss]
                if logits_b is not None:
                    tasks.append(ct_b_loss)
                loss = model.uncertainty_loss(tasks)
            else:
                loss = protein_loss + 0.3 * ct_a_loss + 0.1 * router_loss
                if logits_b is not None:
                    loss = loss + 0.3 * ct_b_loss
            loss = loss + args.bio_prior_weight * prior_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_protein += float(protein_loss.item()) * batch_size
        total_celltype += float(ct_a_loss.item()) * batch_size
        total_n += batch_size

    gate = model.get_gate_stats()
    elapsed = time.time() - t0
    print(
        f"Epoch {epoch:03d}/{total_epochs}: "
        f"protein={total_protein / total_n:.4f} "
        f"celltype={total_celltype / total_n:.4f} "
        f"alpha={gate['avg_alpha']:.3f} "
        f"gate={gate['avg_modgate']:.3f} "
        f"gt_mix={gt_mix:.2f} "
        f"time={elapsed:.0f}s"
    )


@torch.no_grad()
def validate(model, val_ds, args, device):
    model.eval()
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    all_protein_pred = []
    for batch in loader:
        detail = batch["detail"].to(device)
        context = batch["context"].to(device)
        with amp_context(device):
            protein_pred = model.predict_proteins_only(detail, context)
        all_protein_pred.append(protein_pred.float().cpu())
    all_protein_pred = torch.cat(all_protein_pred, 0)

    protein_pred_full = torch.from_numpy(val_ds.proteins_full.copy())
    val_global_idx = torch.from_numpy(val_ds.indices.copy())
    protein_pred_full[val_global_idx] = all_protein_pred

    nb_indices = torch.from_numpy(val_ds.neighbor_indices)
    valid_ct_indices = [i for i, v in enumerate(val_ds.celltype_valid_mask) if v > 0]

    all_logits = []
    all_gt = []
    all_gt_protein = []
    all_pred_protein = []
    all_mask = []

    cursor = 0
    for batch in loader:
        detail = batch["detail"].to(device)
        context = batch["context"].to(device)
        adj = batch["adj"].to(device)
        protein_gt = batch["protein"]
        protein_mask = batch["protein_mask"]
        celltype_gt = batch["celltype"]
        batch_size = detail.size(0)

        idx_range = torch.arange(cursor, cursor + batch_size)
        nb_idx = nb_indices[idx_range]
        nb_protein = protein_pred_full[nb_idx.reshape(-1)].view(batch_size, val_ds.num_neighbors, len(PROTEIN_COLS)).to(device)
        cursor += batch_size

        with amp_context(device):
            out = model(detail, context, nb_protein=nb_protein, adj=adj)

        logits = out["celltype_logits_A"].float()
        invalid_ct_indices = [i for i in range(12) if i not in valid_ct_indices]
        if invalid_ct_indices:
            logits[:, invalid_ct_indices] = -1e4

        all_logits.append(logits.cpu())
        all_gt.append(celltype_gt)
        all_gt_protein.append(protein_gt)
        all_pred_protein.append(out["protein_pred"].float().cpu())
        all_mask.append(protein_mask)

    all_logits = torch.cat(all_logits, 0)
    all_gt = torch.cat(all_gt, 0)
    all_gt_protein = torch.cat(all_gt_protein, 0)
    all_pred_protein = torch.cat(all_pred_protein, 0)
    all_mask = torch.cat(all_mask, 0)

    preds = all_logits.argmax(dim=1)
    acc = float((preds == all_gt).float().mean())
    f1_macro = f1_score(all_gt.numpy(), preds.numpy(), labels=valid_ct_indices, average="macro", zero_division=0)
    report = classification_report(
        all_gt.numpy(),
        preds.numpy(),
        labels=valid_ct_indices,
        target_names=[CELLTYPE_NAMES[i] for i in valid_ct_indices],
        zero_division=0,
    )

    per_protein_pearson = []
    for i in range(18):
        valid = all_mask[:, i].bool()
        if valid.sum() < 10:
            per_protein_pearson.append(float("nan"))
            continue
        pred_i = all_pred_protein[valid, i]
        true_i = all_gt_protein[valid, i]
        pred_i = pred_i - pred_i.mean()
        true_i = true_i - true_i.mean()
        r = float((pred_i * true_i).sum() / (pred_i.norm() * true_i.norm()).clamp(min=1e-8))
        per_protein_pearson.append(r)

    mean_pearson = float(np.nanmean(per_protein_pearson)) if any(not np.isnan(v) for v in per_protein_pearson) else 0.0
    n_valid_proteins = int((all_mask.float().mean(dim=0) > 0.01).sum().item())

    print("\nValidation")
    print(f"  Pearson={mean_pearson:.4f}  F1={f1_macro:.4f}  Acc={acc:.4f}")
    print(f"  Valid classes={len(valid_ct_indices)}/12  Valid proteins={n_valid_proteins}/18")
    for name, r in zip([c.replace("_intensity_mean", "") for c in PROTEIN_COLS], per_protein_pearson):
        value = f"{r:.4f}" if not np.isnan(r) else "N/A"
        print(f"  {name:>10s}: {value}")
    print(f"\n{report}\n")

    return mean_pearson, f1_macro, acc


def main():
    parser = argparse.ArgumentParser(description="Train SPHINX on one or more samples.")
    parser.add_argument("--sample_dir", default=None, help="Single sample directory.")
    parser.add_argument("--sample_dirs", nargs="+", default=None, help="Multiple sample directories.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone_lr_scale", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--max_cells_per_sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", default="./runs")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--bio_prior_weight", type=float, default=0.05)
    parser.add_argument("--pearson_weight", type=float, default=0.4)
    parser.add_argument("--deep_sup_weight", type=float, default=0.3)
    parser.add_argument("--no_protein_gat", action="store_true")
    parser.add_argument("--no_protein_mlp_cls", action="store_true")
    parser.add_argument("--no_deep_supervision", action="store_true")
    parser.add_argument("--no_uncertainty_loss", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--backbone", default="vit", choices=["vit", "ctranspath"],
                        help="Image backbone: 'vit' (ImageNet ViT-Base, default) or "
                             "'ctranspath' (TCGA-pretrained Swin-Tiny).")
    parser.add_argument("--ctranspath_weights", default=None,
                        help="Path to ctranspath.pth weights file (required when "
                             "--backbone ctranspath and --pretrained).")
    args = parser.parse_args()

    if args.sample_dirs:
        sample_dirs = args.sample_dirs
    elif args.sample_dir:
        sample_dirs = [args.sample_dir]
    else:
        raise ValueError("Provide --sample_dir or --sample_dirs.")

    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_config = dict(
        num_proteins=18,
        num_celltypes=12,
        backbone=args.backbone,
        use_protein_gat=not args.no_protein_gat,
        use_protein_mlp_cls=not args.no_protein_mlp_cls,
        use_deep_supervision=not args.no_deep_supervision,
        use_uncertainty_loss=not args.no_uncertainty_loss,
    )
    model = ViTProteinModel(
        pretrained=True,
        ctranspath_weights=args.ctranspath_weights,
        **model_config,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    backbone_params = list(model.detail_backbone.parameters()) + list(model.context_backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr * args.backbone_lr_scale},
            {"params": other_params, "lr": args.lr},
        ],
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch - 1}")

    train_ds_list = []
    val_ds = None
    for sample_dir in sample_dirs:
        train_ds, sample_val_ds, stats = prepare_sample(
            sample_dir,
            train_ratio=args.train_ratio,
            seed=args.seed,
            max_cells=args.max_cells_per_sample,
        )
        train_ds_list.append(train_ds)
        if val_ds is None:
            val_ds = sample_val_ds
        print(f"{Path(sample_dir).name}: train={stats['n_train']:,} val={stats['n_val']:,}")

    train_dataset = train_ds_list[0] if len(train_ds_list) == 1 else ConcatDataset(train_ds_list)
    class_weight = compute_class_weights(sample_dirs, device)

    total_steps = args.epochs * (len(train_dataset) // args.batch_size + 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr * args.backbone_lr_scale, args.lr],
        total_steps=total_steps,
        pct_start=0.1,
    )

    best_pearson = -float("inf")
    for epoch in range(start_epoch, args.epochs + 1):
        loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=False,
        )
        train_epoch(model, loader, optimizer, scaler, scheduler, class_weight, args, device, epoch, args.epochs)
        pearson, f1, acc = validate(model, val_ds, args, device)

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "model_config": model_config,
            "optimizer": optimizer.state_dict(),
            "pearson": pearson,
            "f1": f1,
            "acc": acc,
        }
        epoch_path = os.path.join(args.save_dir, f"epoch_{epoch:03d}.pth")
        latest_path = os.path.join(args.save_dir, "latest.pth")
        torch.save(checkpoint, epoch_path)
        torch.save(checkpoint, latest_path)
        print(f"Saved: {epoch_path}")

        if pearson > best_pearson:
            best_pearson = pearson
            best_path = os.path.join(args.save_dir, "best.pth")
            torch.save(checkpoint, best_path)
            print(f"New best Pearson={pearson:.4f}: {best_path}")

    print(f"Training complete. Best Pearson={best_pearson:.4f}")


if __name__ == "__main__":
    main()
