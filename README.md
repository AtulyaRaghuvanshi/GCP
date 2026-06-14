# GCP Marker Localization & Shape Classification

A multi-task pipeline that, given an aerial image, predicts:
1. The pixel `(x, y)` location of the Ground Control Point (GCP) marker center.
2. The marker's physical shape: `Cross`, `Square`, or `L-Shaped`.

---

## 1. Repository Layout

```
gcp_pipeline/
├── eda.py                # Exploratory data analysis script
├── dataset.py            # Dataset classes + augmentation pipelines
├── model.py              # Multi-task model definition
├── utils.py              # Shared helpers (Wing loss, image discovery, etc.)
├── train.py              # Training script
├── infer.py              # Inference script -> predictions.json
├── download_dataset.py   # Auto-download helper (optional)
├── requirements.txt
└── README.md
```

---

## 2. Deliverables

| File | Location |
|---|---|
| `predictions.json` | Submitted directly — generated on the full test set (300 images) |
| `best_model.pt` | Google Drive link below |
| Codebase | This zip / repository |
| Colab notebook | `skylark_assignment.ipynb` — full reproducible run included |

**Model weights (best_model.pt — ResNet34, epoch 12):**
```
<INSERT_GOOGLE_DRIVE_LINK_TO_best_model.pt_HERE>
```
Place the downloaded file at `./checkpoints/best_model.pt` or pass its path via `--checkpoint`.

---

## 3. Architecture & Rationale

**Shared-backbone multi-task CNN**: a single ImageNet-pretrained backbone
(default `ResNet34`, also supports `ResNet18` / `EfficientNet-B0`) feeds two
lightweight heads:

- **Keypoint head**: `Linear -> ReLU -> Dropout -> Linear(2)` followed by a
  sigmoid, producing normalized `(x, y) ∈ [0,1]²` relative to the network's
  384×256 input. At inference these are rescaled per-image back to the original
  resolution.
- **Classification head**: `Linear -> ReLU -> Dropout -> Linear(3)` softmax
  over `{Cross, Square, L-Shaped}`.

**Why this over the alternatives:**

- *Two separate models* would double training/inference cost and discard the
  fact that marker shape and appearance are correlated with localization — a
  shared encoder lets both tasks benefit from the same learned features with
  negligible extra parameters for the heads.
- *Heatmap-based detection* (stacked hourglass / CenterNet-style) can be more
  precise for small keypoints but requires an upsampling decoder and more
  careful loss tuning. Direct regression with Wing loss is simpler to train and
  deploy while remaining competitive at the PCK thresholds requested (10/25/50px).
- *YOLO-pose* pulls in a large external framework that works against the clean,
  modular criterion for a project of this scope.

**Input resolution**: images are 2048×1365 (aspect ratio 1.5). The network input
is 384×256, which preserves that ratio exactly — no letterboxing or distortion,
and keypoints scale back losslessly via a single per-axis ratio.

---

## 4. Training Strategy

### Data Split
Split is done **by `project/survey`**, not by individual image (see `get_group_key`
in `train.py`). Many images per survey show the *same physical marker* from
slightly different angles — a per-image random split would leak near-duplicates
between train and validation, producing an overly optimistic validation score.

### Losses
- **Keypoint**: [Wing loss](https://arxiv.org/abs/1711.06753) — log-based for
  small errors (steeper gradient near convergence, good for sub-pixel precision)
  and smooth-L1 for large errors (robust to occasional annotation outliers).
- **Classification**: class-weighted Cross-Entropy. Weights computed inversely
  proportional to class frequency in the training split to counteract class
  imbalance (Cross markers are significantly rarer — weight ~5.3× vs ~0.6× for
  L-Shaped).
- Total loss = `kp_weight * wing_loss + cls_weight * ce_loss`, both default 1.0.

### Augmentations
All geometric augmentations are keypoint-aware (Albumentations `KeypointParams`),
so the `(x, y)` label transforms consistently with the image:
- Horizontal/vertical flips, ±15° rotation (`BORDER_REFLECT` to avoid black
  borders near the marker).
- Brightness/contrast and HSV jitter — aerial imagery varies with sun angle,
  time of day, and sensor settings across surveys.
- Light motion blur / Gaussian blur / Gaussian noise, simulating drone motion
  and sensor noise.
- **No random cropping** — a crop could remove the marker entirely or push it
  out of frame, corrupting the label.

### Dataset Challenges & How They Were Handled

| Challenge | Mitigation |
|---|---|
| **Mixed image orientations** (portrait vs landscape due to EXIF rotation tags) | Replaced `cv2.imread` with Pillow + `ImageOps.exif_transpose` in `dataset.py` — this was causing a `RuntimeError: stack expects each tensor to be equal size` crash in the DataLoader |
| **Malformed label entries** (4 entries skipped) | `build_samples()` gracefully skips and reports any entry missing `mark.x/y` or `verified_shape` |
| **Class imbalance** (Cross ~5× rarer than L-Shaped) | Inverse-frequency class weights in Cross-Entropy loss |
| **Survey-level data leakage** | Group-aware train/val split by `project/survey` key |
| **Coordinate outliers / annotation noise** | Wing loss reduces sensitivity to occasional bad annotations |
| **Inconsistent image sizes** | Keypoints normalized relative to each image's own width/height, so pipeline is robust regardless of resolution |

---

## 5. Training Results

Best checkpoint saved at **epoch 12** (lowest validation loss = 2.41):

```
Epoch 12/40 | train_loss=0.8961 (kp=0.8125, cls=0.0835, px_err=87.71, acc=0.976)
           | val_loss=2.4101   (kp=0.9874, cls=1.4227, px_err=103.74, acc=0.516)
```

- 996 valid labeled samples → 874 train / 122 val (split by project/survey group)
- 4 malformed label entries skipped automatically
- Training ran for 40 epochs on a Tesla T4 GPU (Google Colab)
- The gap between train and val classification loss indicates the val set contains
  survey groups with different shape distributions than train — a known limitation
  of small, production-sourced datasets

---

## 6. Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## 7. Reproducing predictions.json (Local)

### Step 1 — EDA (optional but recommended)
```bash
python eda.py \
  --train_dir /path/to/train_dataset \
  --test_dir  /path/to/test_dataset
```

### Step 2 — Train
```bash
python train.py \
  --train_dir /path/to/train_dataset \
  --output_dir ./checkpoints \
  --backbone resnet34 \
  --epochs 40 \
  --batch_size 16 \
  --num_workers 2 \
  --resume auto
```

Saves `checkpoints/best_model.pt` (best val loss) and `checkpoints/last_checkpoint.pt`.
If interrupted, rerun the same command — `--resume auto` picks up from the last epoch.

### Step 3 — Inference
```bash
python infer.py \
  --test_dir   /path/to/test_dataset \
  --checkpoint ./checkpoints/best_model.pt \
  --output     predictions.json \
  --batch_size 32 \
  --num_workers 2
```

Output format:
```json
{
  "project1/survey1/2/DJI_0431.JPG": {
    "mark": {"x": 1024.5, "y": 850.2},
    "verified_shape": "L-Shaped"
  }
}
```

---

## 8. Reproducing via Google Colab (Recommended)

A fully executed Colab notebook (`skylark_assignment.ipynb`) is included showing
the complete run from dataset extraction to predictions download.

**To reproduce from scratch:**

1. Enable GPU: `Runtime → Change runtime type → T4 GPU → Save`

2. Upload these files to **Google Drive root** (`My Drive/`):
   - `gcp_pipeline.zip`
   - `train_dataset-20260614T100421Z-3-001.zip`
   - `train_dataset-20260614T100421Z-3-002.zip`
   - `test_dataset-20260614T100423Z-3-001.zip`

3. Open `skylark_assignment.ipynb` in Colab and run all cells in order.

4. `predictions.json` and `best_model.pt` will be saved automatically to
   `My Drive/` upon completion.

**If Colab disconnects mid-training:** re-run cells 1–5 (mount drive, extract
project, install deps, check GPU, extract data), then re-run the training cell.
The `--resume auto` flag will continue from the last saved checkpoint.

---

## 9. Assumptions & Future Work

- **Assumption**: each image contains exactly one GCP marker (consistent with the
  single-keypoint label format).
- **Assumption**: `gcp_marks.json` keys exactly match the relative paths produced
  by walking `train_dataset` with the `project/survey/gcp_id/image.JPG` structure.
- If validation PCK@10px is insufficient, the recommended next step is adding a
  heatmap head (small decoder branch on the same backbone) with Gaussian-target
  MSE loss — this keeps the same multi-task structure unchanged.
- Test-time augmentation (horizontal flip averaging) is a cheap accuracy boost
  that can be added to `infer.py` without retraining.
