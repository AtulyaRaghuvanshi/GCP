"""Inference script: runs the trained model on test_dataset and writes
predictions.json in the same format as curated_gcp_marks.json.

Example:
    python infer.py \
        --test_dir /path/to/test_dataset \
        --checkpoint ./checkpoints/best_model.pt \
        --output predictions.json
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import GCPTestDataset, get_test_transform
from model import GCPMultiTaskModel
from utils import IDX_TO_SHAPE, find_images


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    backbone = checkpoint.get("backbone", "resnet34")
    print(f"Loaded checkpoint trained with backbone={backbone}, epoch={checkpoint.get('epoch')}")

    model = GCPMultiTaskModel(backbone_name=backbone, num_classes=3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    rel_paths = find_images(args.test_dir)
    print(f"Found {len(rel_paths)} test images.")

    test_ds = GCPTestDataset(rel_paths, args.test_dir, transform=get_test_transform())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    predictions = {}

    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch["image"].to(device)
            kp_pred, cls_logits = model(images)

            kp_pred = kp_pred.cpu().numpy()
            cls_pred = cls_logits.argmax(dim=1).cpu().numpy()

            orig_w = batch["orig_w"]
            orig_h = batch["orig_h"]
            rel_paths_batch = batch["rel_path"]

            for i in range(len(rel_paths_batch)):
                w = float(orig_w[i])
                h = float(orig_h[i])
                x_px = float(kp_pred[i, 0] * w)
                y_px = float(kp_pred[i, 1] * h)
                shape_idx = int(cls_pred[i])

                predictions[rel_paths_batch[i]] = {
                    "mark": {"x": x_px, "y": y_px},
                    "verified_shape": IDX_TO_SHAPE[shape_idx],
                }

    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Wrote predictions for {len(predictions)} images to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", default=None,
                        help="Path to test_dataset directory. If omitted, auto-downloaded from Google Drive.")
    parser.add_argument("--data_dir", default=None,
                        help="Override the local download directory (default: ./data)")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--output", default="predictions.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    if args.test_dir is None:
        from download_dataset import ensure_dataset, DEFAULT_DATA_DIR
        data_dir = args.data_dir or DEFAULT_DATA_DIR
        _, args.test_dir = ensure_dataset(data_dir)

    main(args)
