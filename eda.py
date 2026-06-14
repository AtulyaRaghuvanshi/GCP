"""Exploratory Data Analysis for the GCP dataset.

Run:
    python eda.py --train_dir /path/to/train_dataset --test_dir /path/to/test_dataset

This script checks for the kinds of issues commonly found in
"production" datasets and prints a report. It does NOT modify any
files; it is purely diagnostic and should be run before training.
"""

import argparse
import json
import os
from collections import Counter

import cv2

from utils import SHAPE_CLASSES, find_images, load_labels, normalize_shape, resolve_labels_path


def main(args):
    labels_path = resolve_labels_path(args.train_dir)
    labels = load_labels(labels_path)
    print(f"Total labeled entries in JSON: {len(labels)}")

    # 1. Class distribution
    shape_counter = Counter()
    for ann in labels.values():
        shape = normalize_shape(ann.get("verified_shape"))
        shape_counter[shape] += 1
    print("\n--- Shape class distribution ---")
    for shape in SHAPE_CLASSES:
        print(f"  {shape}: {shape_counter.get(shape, 0)}")
    unknown = set(shape_counter) - set(SHAPE_CLASSES)
    if unknown:
        print(f"  WARNING: unexpected class labels found: {unknown}")

    # 2. Discover images on disk vs. labels
    all_images = find_images(args.train_dir)
    # filter out the json itself if it were ever matched (it won't be, but be safe)
    print(f"\nTotal images found on disk: {len(all_images)}")

    image_set = set(all_images)
    label_set = set(labels.keys())

    missing_images = label_set - image_set
    unlabeled_images = image_set - label_set
    print(f"Labels referencing missing image files: {len(missing_images)}")
    if missing_images:
        for p in list(missing_images)[:5]:
            print(f"    missing: {p}")
    print(f"Images on disk with no label entry: {len(unlabeled_images)}")
    if unlabeled_images:
        for p in list(unlabeled_images)[:5]:
            print(f"    unlabeled: {p}")

    # 3. Image size consistency + corrupt file check
    sizes = Counter()
    corrupt = []
    sample_paths = all_images
    for rel_path in sample_paths:
        full_path = os.path.join(args.train_dir, rel_path)
        img = cv2.imread(full_path)
        if img is None:
            corrupt.append(rel_path)
            continue
        h, w = img.shape[:2]
        sizes[(w, h)] += 1

    print("\n--- Image size distribution ---")
    for size, count in sizes.most_common(10):
        print(f"  {size}: {count}")
    print(f"Corrupt / unreadable images: {len(corrupt)}")
    for p in corrupt[:5]:
        print(f"    corrupt: {p}")

    # 4. Keypoint coordinate sanity checks (out-of-bounds / suspicious values)
    out_of_bounds = []
    for rel_path, ann in labels.items():
        full_path = os.path.join(args.train_dir, rel_path)
        img = cv2.imread(full_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        x, y = ann["mark"]["x"], ann["mark"]["y"]
        if not (0 <= x <= w and 0 <= y <= h):
            out_of_bounds.append((rel_path, x, y, w, h))

    print(f"\nKeypoints outside image bounds: {len(out_of_bounds)}")
    for item in out_of_bounds[:5]:
        print(f"    {item}")

    # 5. Duplicate detection by file size (cheap proxy; for exact dup check
    #    use a hash, omitted here for speed on large datasets)
    size_counter = Counter()
    for rel_path in all_images:
        full_path = os.path.join(args.train_dir, rel_path)
        size_counter[os.path.getsize(full_path)] += 1
    dup_sizes = {k: v for k, v in size_counter.items() if v > 1}
    print(f"\nFile-size collisions (possible duplicates): {len(dup_sizes)} groups")

    # 6. Survey / project structure summary
    projects = Counter()
    surveys = Counter()
    for rel_path in all_images:
        parts = rel_path.split("/")
        if len(parts) >= 2:
            projects[parts[0]] += 1
        if len(parts) >= 3:
            surveys[(parts[0], parts[1])] += 1
    print(f"\nNumber of projects: {len(projects)}")
    print(f"Number of surveys: {len(surveys)}")

    # 7. Test set summary
    if args.test_dir and os.path.isdir(args.test_dir):
        test_images = find_images(args.test_dir)
        print(f"\nTest images found: {len(test_images)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default=None,
                        help="Path to train_dataset directory. If omitted, auto-downloaded from Google Drive.")
    parser.add_argument("--test_dir", default=None,
                        help="Path to test_dataset directory. If omitted, auto-downloaded from Google Drive.")
    parser.add_argument("--data_dir", default=None,
                        help="Override the local download directory (default: ./data)")
    args = parser.parse_args()

    if args.train_dir is None or args.test_dir is None:
        from download_dataset import ensure_dataset, DEFAULT_DATA_DIR
        data_dir = args.data_dir or DEFAULT_DATA_DIR
        train_dir, test_dir = ensure_dataset(data_dir)
        args.train_dir = args.train_dir or train_dir
        args.test_dir  = args.test_dir  or test_dir

    main(args)
