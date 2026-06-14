"""Shared utilities for the GCP keypoint + shape classification pipeline."""

import os
import random
import json

import numpy as np
import torch
import torch.nn as nn


SHAPE_CLASSES = ["Cross", "Square", "L-Shaped"]
SHAPE_TO_IDX = {name: i for i, name in enumerate(SHAPE_CLASSES)}
IDX_TO_SHAPE = {i: name for name, i in SHAPE_TO_IDX.items()}
SHAPE_ALIASES = {
    "L-Shape": "L-Shaped",
}

IMG_EXTENSIONS = (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG")
LABEL_FILENAMES = ("curated_gcp_marks.json", "gcp_marks.json")

# Network input size (keeps the 2048x1365 aspect ratio of 1.5)
INPUT_W, INPUT_H = 384, 256


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_images(root_dir):
    """Recursively find all image files under root_dir.

    Returns a list of paths relative to root_dir, using forward slashes,
    matching the format used in curated_gcp_marks.json.
    """
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(IMG_EXTENSIONS):
                full = os.path.join(dirpath, fname)
                rel = os.path.relpath(full, root_dir)
                rel = rel.replace(os.sep, "/")
                paths.append(rel)
    return sorted(paths)


def load_labels(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def normalize_shape(shape):
    return SHAPE_ALIASES.get(shape, shape)


def resolve_labels_path(train_dir):
    """Return the first supported label JSON path found under train_dir."""
    for filename in LABEL_FILENAMES:
        candidate = os.path.join(train_dir, filename)
        if os.path.isfile(candidate):
            return candidate
    expected = ", ".join(LABEL_FILENAMES)
    raise FileNotFoundError(f"No label JSON found in {train_dir}. Expected one of: {expected}")


class WingLoss(nn.Module):
    """Wing loss for robust keypoint regression.

    Behaves like a smoothed L1 for large errors and like a log-based
    loss (steeper gradient) for small errors, which empirically helps
    keypoint localization converge to sub-pixel precision.

    Reference: "Wing Loss for Robust Facial Landmark Localisation with
    Convolutional Neural Networks" (Feng et al., 2018).
    """

    def __init__(self, omega=10.0, epsilon=2.0):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = omega - omega * np.log(1 + omega / epsilon)

    def forward(self, pred, target):
        diff = (pred - target).abs()
        flag = (diff < self.omega).float()
        loss = flag * self.omega * torch.log(1 + diff / self.epsilon) + \
            (1 - flag) * (diff - self.C)
        return loss.mean()


def keypoint_pixel_error(pred_norm, target_norm, img_w, img_h):
    """Convert normalized [0,1] coordinates back to pixels and compute
    Euclidean distance error. pred_norm/target_norm: (N, 2) tensors.
    img_w, img_h: scalars or (N,) tensors with the original image size.
    """
    pred_px = pred_norm.clone()
    target_px = target_norm.clone()
    pred_px[:, 0] = pred_px[:, 0] * img_w
    pred_px[:, 1] = pred_px[:, 1] * img_h
    target_px[:, 0] = target_px[:, 0] * img_w
    target_px[:, 1] = target_px[:, 1] * img_h
    return torch.norm(pred_px - target_px, dim=1)
