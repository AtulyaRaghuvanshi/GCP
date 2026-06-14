"""Training script for the GCP multi-task model.

Example:
    python train.py \
        --train_dir /path/to/train_dataset \
        --output_dir ./checkpoints \
        --backbone resnet34 \
        --epochs 40 \
        --batch_size 32
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from dataset import GCPTrainDataset, build_samples, get_train_transform, get_val_transform
from model import GCPMultiTaskModel
from utils import (
    SHAPE_CLASSES,
    WingLoss,
    keypoint_pixel_error,
    load_labels,
    resolve_labels_path,
    set_seed,
)


def get_group_key(rel_path):
    """Group by project/survey so the same physical GCP / survey does not
    appear in both train and validation splits (avoids data leakage from
    near-duplicate frames of the same marker)."""
    parts = rel_path.split("/")
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return rel_path


def compute_class_weights(samples, num_classes):
    counts = np.zeros(num_classes, dtype=np.float64)
    for s in samples:
        counts[s["shape_idx"]] += 1
    counts = np.clip(counts, 1, None)  # avoid div by zero
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, optimizer, kp_loss_fn, cls_loss_fn, device, train=True,
               kp_weight=1.0, cls_weight=1.0):
    model.train() if train else model.eval()

    total_loss, total_kp_loss, total_cls_loss = 0.0, 0.0, 0.0
    all_px_errors = []
    correct, total = 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            images = batch["image"].to(device)
            kp_target = batch["keypoint"].to(device)
            cls_target = batch["shape"].to(device)

            kp_pred, cls_logits = model(images)

            loss_kp = kp_loss_fn(kp_pred, kp_target)
            loss_cls = cls_loss_fn(cls_logits, cls_target)
            loss = kp_weight * loss_kp + cls_weight * loss_cls

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_kp_loss += loss_kp.item() * images.size(0)
            total_cls_loss += loss_cls.item() * images.size(0)

            # Pixel error using the network input resolution as the
            # reference frame for relative comparison across epochs.
            # For a true PCK against original resolution, see infer.py.
            from utils import INPUT_W, INPUT_H
            px_err = keypoint_pixel_error(kp_pred.detach(), kp_target.detach(), INPUT_W, INPUT_H)
            all_px_errors.append(px_err.cpu().numpy())

            preds = cls_logits.argmax(dim=1)
            correct += (preds == cls_target).sum().item()
            total += cls_target.size(0)

    n = len(loader.dataset)
    all_px_errors = np.concatenate(all_px_errors)
    metrics = {
        "loss": total_loss / n,
        "kp_loss": total_kp_loss / n,
        "cls_loss": total_cls_loss / n,
        "mean_px_error": float(all_px_errors.mean()),
        "cls_acc": correct / total,
    }
    return metrics


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    labels_path = resolve_labels_path(args.train_dir)
    labels = load_labels(labels_path)
    samples = build_samples(labels)
    print(f"Loaded {len(samples)} valid labeled samples.")

    groups = [get_group_key(s["rel_path"]) for s in samples]
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(samples, groups=groups))

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")

    train_ds = GCPTrainDataset(train_samples, args.train_dir, transform=get_train_transform())
    val_ds = GCPTrainDataset(val_samples, args.train_dir, transform=get_val_transform())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = GCPMultiTaskModel(backbone_name=args.backbone, num_classes=len(SHAPE_CLASSES)).to(device)

    class_weights = compute_class_weights(train_samples, len(SHAPE_CLASSES)).to(device)
    print(f"Class weights ({SHAPE_CLASSES}): {class_weights.tolist()}")

    kp_loss_fn = WingLoss()
    cls_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.output_dir, exist_ok=True)
    latest_ckpt_path = os.path.join(args.output_dir, "last_checkpoint.pt")
    best_val_loss = float("inf")
    history = []
    start_epoch = 1

    resume_path = args.resume
    if resume_path == "auto" and os.path.isfile(latest_ckpt_path):
        resume_path = latest_ckpt_path

    if resume_path and resume_path != "auto":
        print(f"Resuming training from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        history = checkpoint.get("history", [])
        print(f"Continuing at epoch {start_epoch}/{args.epochs}; best_val_loss={best_val_loss:.4f}")

    if start_epoch > args.epochs:
        print(f"Checkpoint already reached epoch {start_epoch - 1}; nothing to train.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, kp_loss_fn, cls_loss_fn,
                                   device, train=True,
                                   kp_weight=args.kp_weight, cls_weight=args.cls_weight)
        val_metrics = run_epoch(model, val_loader, optimizer, kp_loss_fn, cls_loss_fn,
                                 device, train=False,
                                 kp_weight=args.kp_weight, cls_weight=args.cls_weight)
        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_metrics['loss']:.4f} "
              f"(kp={train_metrics['kp_loss']:.4f}, cls={train_metrics['cls_loss']:.4f}, "
              f"px_err={train_metrics['mean_px_error']:.2f}, acc={train_metrics['cls_acc']:.3f}) | "
              f"val_loss={val_metrics['loss']:.4f} "
              f"(kp={val_metrics['kp_loss']:.4f}, cls={val_metrics['cls_loss']:.4f}, "
              f"px_err={val_metrics['mean_px_error']:.2f}, acc={val_metrics['cls_acc']:.3f})")

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            ckpt_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "backbone": args.backbone,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "history": history,
                "val_metrics": val_metrics,
            }, ckpt_path)
            print(f"  -> Saved new best model to {ckpt_path}")

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "backbone": args.backbone,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "history": history,
            "val_metrics": val_metrics,
        }, latest_ckpt_path)
        print(f"  -> Saved latest checkpoint to {latest_ckpt_path}")

    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default=None,
                        help="Path to train_dataset directory. If omitted, auto-downloaded from Google Drive.")
    parser.add_argument("--data_dir", default=None,
                        help="Override the local download directory (default: ./data)")
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--backbone", default="resnet34", choices=["resnet18", "resnet34", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--kp_weight", type=float, default=1.0)
    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None,
                        help="Path to a checkpoint to resume, or 'auto' to use output_dir/last_checkpoint.pt if present.")
    args = parser.parse_args()

    if args.train_dir is None:
        from download_dataset import ensure_dataset, DEFAULT_DATA_DIR
        data_dir = args.data_dir or DEFAULT_DATA_DIR
        args.train_dir, _ = ensure_dataset(data_dir)

    main(args)
