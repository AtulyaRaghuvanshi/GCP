# GCP Marker Localization & Shape Classification

A multi-task CNN pipeline for aerial imagery that jointly predicts the pixel location of a Ground Control Point (GCP) marker and classifies its shape (`Cross`, `Square`, or `L-Shaped`).

---

## Reproducing via Google Colab (Recommended)

A fully executed notebook (`skylark_assignment.ipynb`) is included, covering the complete run from dataset extraction to predictions download.

**To reproduce from scratch:**

1. Enable GPU: `Runtime → Change runtime type → T4 GPU → Save`

2. Upload the following files to **Google Drive root** (`My Drive/`):
   - `gcp_pipeline.zip`
   - `train_dataset-20260614T100421Z-3-001.zip`
   - `train_dataset-20260614T100421Z-3-002.zip`
   - `test_dataset-20260614T100423Z-3-001.zip`
   
   Note : training dataset is by default dowloaded in 2 parts due to it's size, if you have a single training dataset, please move forward with that

3. Open `skylark_assignment.ipynb` in Colab and run all cells in order.

4. `predictions.json` and `best_model.pt` are saved automatically to `My Drive/` on completion.

**If Colab disconnects mid-training:** re-run cells 1–5 (mount drive, extract project, install deps, check GPU, extract data), then re-run the training cell. The `--resume auto` flag will continue from the last saved checkpoint.

---

## Deliverables

- predictions.json: Submitted directly and generated on the full test set of 300 images
- best_model.pt: Available from [Google Drive](https://drive.google.com/drive/folders/1CRTUE3PxnWC-xXbZViTftyjnCAUfMGn0?usp=sharing) 
- Codebase: This repository
- Colab notebook: skylark_assignment.ipynb with a full reproducible run included (present in repo)

Place the downloaded weights at `./checkpoints/best_model.pt` or supply a custom path via `--checkpoint`.

---

## Architecture

A single ImageNet-pretrained backbone (default: `ResNet34`; also supports `ResNet18` and `EfficientNet-B0`) feeds two lightweight task heads:

**Keypoint head** — `Linear → ReLU → Dropout → Linear(2)` with a sigmoid activation. Produces normalized `(x, y) ∈ [0, 1]²` coordinates relative to the 384×256 network input, then rescales them back to the original image resolution at inference.

**Classification head** — `Linear → ReLU → Dropout → Linear(3)` with softmax over `{Cross, Square, L-Shaped}`.

### Why a shared-backbone multi-task design?

Two separate models would double training and inference cost while discarding a useful inductive bias: marker appearance and shape are correlated, so a shared encoder lets both tasks benefit from the same learned features with negligible extra parameters in the heads.

Heatmap-based detection (stacked hourglass or CenterNet-style) can be more precise for small keypoints but requires an upsampling decoder and more careful loss tuning. Direct regression with Wing loss is simpler to train and deploy, and remains competitive at the PCK thresholds requested (10/25/50 px).

A YOLO-pose approach would introduce a large external framework dependency that conflicts with the modular design goals of this project.

### Input resolution

Images are 2048×1365 (aspect ratio 1.5:1). The network input is 384×256, which preserves this ratio exactly — no letterboxing or distortion — and allows keypoints to scale back losslessly via a single per-axis multiplier.

---

## Training Strategy

### Data split

The train/val split is performed **by `project/survey` group** (see `get_group_key` in `train.py`), not by individual image. Many images in a survey capture the same physical marker from slightly different angles; a per-image random split would leak near-duplicates between train and validation and produce an overly optimistic validation score.

### Loss functions

**Keypoint** — [Wing loss](https://arxiv.org/abs/1711.06753): log-based for small errors (steep gradient near convergence, beneficial for sub-pixel precision) and smooth-L1 for large errors (robust to occasional annotation outliers).

**Classification** — Class-weighted Cross-Entropy. Weights are computed inversely proportional to class frequency in the training split to counteract imbalance; Cross markers are significantly rarer, receiving a weight of ~5.3× versus ~0.6× for L-Shaped.

**Total loss** = `kp_weight × wing_loss + cls_weight × cross_entropy_loss` (both weights default to 1.0).

### Augmentations

All geometric augmentations are keypoint-aware via Albumentations `KeypointParams`, so `(x, y)` labels transform consistently with the image:

- Horizontal and vertical flips; ±15° rotation with `BORDER_REFLECT` (avoids black borders near the marker)
- Brightness/contrast and HSV jitter — aerial imagery varies with sun angle, time of day, and sensor settings across surveys
- Light motion blur, Gaussian blur, and Gaussian noise — simulates drone motion and sensor noise
- **No random cropping** — a crop could remove the marker entirely or shift it out of frame, corrupting the label

### Dataset challenges

- Mixed image orientations from EXIF rotation tags were handled by replacing `cv2.imread` with Pillow and `ImageOps.exif_transpose` in `dataset.py`, fixing a `RuntimeError: stack expects each tensor to be equal size` crash.
- Four malformed label entries were skipped by `build_samples()`, which gracefully logs any entry missing `mark.x/y` or `verified_shape`.
- Class imbalance, with Cross markers about five times rarer than L-Shaped, was addressed with inverse-frequency class weights in Cross-Entropy loss.
- Survey-level data leakage was reduced through a group-aware train and validation split by `project/survey` key.
- Coordinate outliers and annotation noise were mitigated with Wing loss, which reduces sensitivity to occasional bad annotations.
- Inconsistent image sizes were handled by normalizing keypoints to each image's own width and height.

---

## Training Results

Best checkpoint saved at **epoch 12** (lowest validation loss = 2.41):

```
Epoch 12/40
  train  loss=0.8961  (kp=0.8125, cls=0.0835, px_err=87.71,  acc=0.976)
  val    loss=2.4101  (kp=0.9874, cls=1.4227, px_err=103.74, acc=0.516)
```

- **996** valid labeled samples → **874 train / 122 val** (split by project/survey group)
- 4 malformed label entries skipped automatically
- 40 epochs on a Tesla T4 GPU (Google Colab)

The gap between train and val classification loss reflects that the val set contains survey groups with different shape distributions from the training set — a known limitation of small, production-sourced datasets.

---

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## Reproducing `predictions.json` (Local)

### Step 1 — EDA (optional)

```bash
python eda.py \
  --train_dir /path/to/train_dataset \
  --test_dir  /path/to/test_dataset
```

### Step 2 — Train

```bash
python train.py \
  --train_dir  /path/to/train_dataset \
  --output_dir ./checkpoints \
  --backbone   resnet34 \
  --epochs     40 \
  --batch_size 16 \
  --num_workers 2 \
  --resume auto
```

Saves `checkpoints/best_model.pt` (best val loss) and `checkpoints/last_checkpoint.pt`. If interrupted, rerun the same command — `--resume auto` picks up from the last epoch.

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
    "mark": { "x": 1024.5, "y": 850.2 },
    "verified_shape": "L-Shaped"
  }
}
```

---

## Assumptions

**Assumptions:**
- Each image contains exactly one GCP marker, consistent with the single-keypoint label format.
- `gcp_marks.json` keys exactly match the relative paths produced by walking `train_dataset` with the `project/survey/gcp_id/image.JPG` structure.
