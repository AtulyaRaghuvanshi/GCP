"""Dataset classes for the GCP keypoint + shape classification task."""

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image as PILImage
from PIL import ImageOps

from utils import SHAPE_TO_IDX, INPUT_W, INPUT_H, normalize_shape


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_image_rgb(img_path):
    """Read an image as RGB and apply EXIF orientation if present."""
    try:
        with PILImage.open(img_path) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            return np.array(pil_img.convert("RGB"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not read image: {img_path}")


def get_train_transform():
    """Augmentations applied during training. All geometric transforms
    are keypoint-aware so the (x, y) label is transformed consistently
    with the image. Crops are intentionally avoided so the marker is
    never cut out of frame.
    """
    return A.Compose(
        [
            A.Resize(INPUT_H, INPUT_W),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.4),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=5),
                    A.GaussNoise(var_limit=(0.02, 0.11)),
                ],
                p=0.3,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def get_val_transform():
    return A.Compose(
        [
            A.Resize(INPUT_H, INPUT_W),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def get_test_transform():
    return A.Compose(
        [
            A.Resize(INPUT_H, INPUT_W),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class GCPTrainDataset(Dataset):
    """Dataset built from curated_gcp_marks.json.

    samples: list of dicts, each containing:
        - "rel_path": path relative to image_root
        - "x", "y": pixel coordinates in the ORIGINAL image
        - "shape_idx": integer class index
    """

    def __init__(self, samples, image_root, transform=None):
        self.samples = samples
        self.image_root = image_root
        self.transform = transform or get_train_transform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.image_root, sample["rel_path"])
        image = read_image_rgb(img_path)
        h, w = image.shape[:2]

        kp = [(float(sample["x"]), float(sample["y"]))]

        transformed = self.transform(image=image, keypoints=kp)
        img_t = transformed["image"]
        kx, ky = transformed["keypoints"][0]

        # Normalize to [0, 1] relative to the (resized) network input.
        kx_norm = np.clip(kx / INPUT_W, 0.0, 1.0)
        ky_norm = np.clip(ky / INPUT_H, 0.0, 1.0)

        target_kp = torch.tensor([kx_norm, ky_norm], dtype=torch.float32)
        target_cls = torch.tensor(sample["shape_idx"], dtype=torch.long)

        return {
            "image": img_t,
            "keypoint": target_kp,
            "shape": target_cls,
            "orig_w": w,
            "orig_h": h,
            "rel_path": sample["rel_path"],
        }


class GCPTestDataset(Dataset):
    """Dataset for the unlabeled test set: yields image tensors plus the
    original image size and relative path, needed to rescale predictions
    back to pixel coordinates and to build predictions.json.
    """

    def __init__(self, rel_paths, image_root, transform=None):
        self.rel_paths = rel_paths
        self.image_root = image_root
        self.transform = transform or get_test_transform()

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, idx):
        rel_path = self.rel_paths[idx]
        img_path = os.path.join(self.image_root, rel_path)
        image = read_image_rgb(img_path)
        h, w = image.shape[:2]

        transformed = self.transform(image=image)
        img_t = transformed["image"]

        return {
            "image": img_t,
            "orig_w": w,
            "orig_h": h,
            "rel_path": rel_path,
        }


def build_samples(labels_dict):
    """Convert the raw curated_gcp_marks.json dict into a flat list of
    sample dicts, skipping any malformed entries (with a warning).
    """
    samples = []
    skipped = []
    for rel_path, ann in labels_dict.items():
        try:
            mark = ann["mark"]
            x, y = float(mark["x"]), float(mark["y"])
            shape = normalize_shape(ann["verified_shape"])
            shape_idx = SHAPE_TO_IDX[shape]
        except (KeyError, TypeError, ValueError):
            skipped.append(rel_path)
            continue
        samples.append({"rel_path": rel_path, "x": x, "y": y, "shape_idx": shape_idx})

    if skipped:
        print(f"[build_samples] Skipped {len(skipped)} malformed label entries.")
    return samples
