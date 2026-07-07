# Data Sources & Contract

This document is the single source of truth for how data is sourced, labeled, split, and
preprocessed. Both the segmentation model and the classifier must follow it exactly —
any change here needs to be reflected in both training pipelines.

## Datasets

**Classification — HAM10000** (`kmader/skin-cancer-mnist-ham10000` on Kaggle)
- 10,015 images, 7,470 unique lesions (`lesion_id`), all images 600x450 px, uniform.
- Metadata columns: `lesion_id`, `image_id`, `dx`, `dx_type`, `age`, `sex`, `localization`.
- No skin-tone / Fitzpatrick field — see **Fairness metadata limitation** below.
- Ignore `hmnist_8_8_RGB.csv` / `hmnist_8_8_L.csv` in this dataset — those are downsampled
  8x8 pixel-array exports, not used anywhere in this project.

**Segmentation — ISIC2018 Task 1** (`tschandl/isic2018-challenge-task1-data-segmentation`)
- Pre-split by the organizers into `Training_Input` / `Validation_Input` / `Test_Input`,
  each paired with `*_GroundTruth` masks matched by filename
  (`ISIC_XXXXXXX.jpg` ↔ `ISIC_XXXXXXX_segmentation.png`).
- Image sizes vary widely (576x543 up to 6708x4459 in a 100-image sample, mean ~3136x2125,
  no dominant resolution) — resize is mandatory, see Preprocessing.
- `ISIC2018_Task1-2_Training_Input/` also contains a non-image `ATTRIBUTION.txt` — filter
  file listings to `*.jpg` before loading.
- Mask-alignment spot check (4 samples) confirmed masks tightly trace lesion boundaries,
  including under hair occlusion and dermoscope vignette borders.

## Label scheme

Binary malignant-vs-benign mapping from the 7-class `dx` field:

| Group | Classes |
|---|---|
| Malignant | `mel` (melanoma), `bcc` (basal cell carcinoma), `akiec` (actinic keratoses / Bowen's — treated as malignant here since it is potentially pre-cancerous; document as a modeling choice, not a clinical fact) |
| Benign | `nv` (melanocytic nevi), `bkl` (benign keratosis), `vasc` (vascular lesions), `df` (dermatofibroma) |

- Raw 7-class imbalance ratio: **58.3:1** (`nv` 6,705 vs `df` 115).
- Binary imbalance ratio: **4.13:1** (benign 8,061 vs malignant 1,954).
- The classifier is trained and evaluated on the binary label. Handle the 4.13:1
  imbalance with a weighted loss or weighted sampler (Day 6) — do not ignore it.

## Split strategy

- **Split by `lesion_id`, not `image_id`.** 1,956 of 7,470 lesions (~26%) have 2+ images
  in HAM10000. A random image-level split would leak the same lesion's photos across
  train/val/test, silently inflating every downstream metric.
- Use a grouped split (e.g. `sklearn.model_selection.GroupShuffleSplit` or
  `StratifiedGroupKFold` if available) keyed on `lesion_id`, stratified on `binary_label`
  where possible.
- Target split: 70% train / 15% val / 15% test (adjust once actual split is run; record
  final counts here).
- **Update (Day 5):** ISIC2018's organizer-provided `Validation_Input`/`Test_Input`
  folders have no released ground-truth masks (held back for the competition
  leaderboard), so they cannot be used for supervised evaluation. The segmentation
  model instead uses its own 70/15/15 split (train 1,815 / val 389 / test 390,
  `random_state=42`) carved from the 2,594 labeled `Training_Input` images, verified
  disjoint. See `evaluation/segmentation_report.md` for the full rationale, including a
  model-selection leakage issue this caught and fixed.

## Preprocessing

**Classifier (HAM10000)**
- Source images uniform at 600x450 — resize to a fixed input size for the model
  (EfficientNet-B0 via `timm`; standard input 224x224, confirm against chosen `timm` config).
- Normalize using ImageNet mean/std (since using a pretrained encoder):
  `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.

**Segmentation (ISIC2018)**
- Source images/masks have highly variable resolution and aspect ratio — resize both to a
  fixed training size. **Update (Day 5): trained/evaluated at 256x256**, not 512x512 as
  originally planned, to keep iteration and GPU time manageable — revisit if small-lesion
  quality becomes a bottleneck downstream.
- **Image resize: bilinear interpolation. Mask resize: nearest-neighbor interpolation** —
  bilinear on the mask would blur the binary 0/255 labels into gray values and corrupt them.
- Aspect ratio is not preserved (plain resize, no padding) — accepted tradeoff, documented
  here rather than solved with letterboxing.
- Augmentation via `albumentations`, applied identically to image and mask pair.

## Fairness metadata limitation

HAM10000 metadata has no Fitzpatrick skin-type or skin-tone field. This is a known,
documented limitation, not an oversight. For the Day 8 fairness audit, use an established
proxy method (e.g. individual typology angle estimated from image pixels) and clearly
state in `docs/model_card.md` that this is an estimated proxy, not ground-truth skin type,
with the corresponding caveat on how much weight to put on the resulting fairness numbers.
