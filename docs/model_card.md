# Model Card — Risk Classifier

Model: EfficientNet-B0 (ImageNet-pretrained, via `timm`), binary malignant-vs-benign
classifier trained on HAM10000. This card covers training setup, held-out test
performance, calibration, and the fairness audit — the numbers below are all measured,
not estimated or projected.

## Intended use and limitations

This is an **awareness tool component, not a diagnostic device**. It estimates a risk
score for a single lesion photo; it does not diagnose, and a "low risk" output does not
rule out malignancy. See `docs/responsible_ai.md` (Week 2) for full usage constraints.
The numbers in this card, especially the fairness table, should inform how the tool is
framed to end users — they are not a formality.

## Data and split

- 7,470 unique lesions (`lesion_id`) in HAM10000, split 70/15/15 by lesion (not by
  image) to prevent the same lesion's multiple photos leaking across splits, stratified
  on the binary label. Verified zero lesion-ID overlap between splits.
- Train: 7,010 images / 5,229 lesions. Val: 1,500 images / 1,120 lesions. Test: 1,505
  images / 1,121 lesions.
- Binary label: `mel`/`bcc`/`akiec` = malignant, `nv`/`bkl`/`vasc`/`df` = benign.
  Imbalance ratio ~4.1:1 (benign:malignant), consistent across all three splits.
- No lesion cropping — trained on full images. The Day-5 segmentation masks were not
  used to crop inputs here (a documented scope decision, not an oversight): HAM10000 and
  ISIC2018 only partially overlap in image IDs, and cropping would have added a
  non-trivial data-pipeline dependency for an explicitly optional step.

## Training

- EfficientNet-B0, `timm`, ImageNet-pretrained, single-logit output.
- Loss: `BCEWithLogitsLoss` with `pos_weight` = 4.12 (train-set imbalance ratio) to
  upweight the minority malignant class.
- Optimizer: Adam, LR 1e-4 (lower than the segmentation model's 1e-3 — pretrained
  classification features are more easily disrupted by an aggressive LR), `ReduceLROnPlateau`
  (mode=max on val AUC, factor 0.5, patience 3), early stopping patience 7.
- Batch size 32, dual Tesla T4 (`nn.DataParallel`), 224x224 input, ImageNet
  normalization, albumentations augmentation (flips, rotation, affine, brightness/contrast).
- Training ran 16 epochs before early stopping; best checkpoint at epoch 8. Train loss
  kept falling after epoch 8 (0.40 → 0.26) while val AUC plateaued/drifted down slightly —
  early stopping correctly kept the epoch-8 checkpoint rather than a later, more-overfit one.

## Test set performance (held-out, first and only evaluation)

| Metric | Value |
|---|---|
| Validation AUC (best epoch) | 0.9276 |
| **Test AUC** | **0.9049** |
| Sensitivity @ threshold 0.5 | 0.7674 |
| Specificity @ threshold 0.5 | 0.8611 |
| **Sensitivity @ ~95% specificity** | **0.5069** (actual specificity 0.9507) |

Confusion matrix @ threshold 0.5 (rows = true, cols = predicted; [benign, malignant]):

```
[[1048  169]
 [  67  221]]
```

Confusion matrix @ 95%-specificity threshold (0.9035 on raw probabilities):

```
[[1157   60]
 [ 142  146]]
```

**Read this plainly:** at the operating point that keeps false alarms to ~5% of benign
cases, the model still misses about **half of actual malignant lesions**. This is the
single most important number in this card for understanding what the tool can and
cannot be relied on for. An AUC of 0.90 alone would not convey this.

## Calibration

Reliability curve showed clear overconfidence, worst at the high-confidence end (model
said ~0.95, only ~70% of those were actually malignant).

| | ECE |
|---|---|
| Before temperature scaling | 0.1011 |
| After temperature scaling (T=1.674, fit on val, applied to test) | 0.0842 |

Temperature scaling improved calibration (~17% relative reduction in ECE) but did not
fully fix it — a residual gap remains at the high-confidence end even after correction
(observed ~85% actual malignancy at ~0.95 predicted, post-scaling). AUC is unchanged by
this transform (0.9049 before and after — confirmed numerically), since it's a monotonic
rescaling of the same ranking. **Any displayed risk score should be presented as an
approximate, imperfectly-calibrated estimate, not a precise probability.**

## Fairness audit

### Methodology and why it changed mid-audit

HAM10000 has no real skin-tone field (see `docs/data_sources.md`). The original plan was
to estimate skin tone via the standard Individual Typology Angle (ITA) formula,
`arctan((L*-50)/b*)`, sampled from an annulus of peripheral image pixels (avoiding both
the lesion center and, for dermoscope images, the outer vignette).

This was abandoned after direct investigation: manually inspecting the most extreme ITA
outliers showed clearly light-skinned people being classified as "dark" skin. The root
cause was a blue-violet color cast common in polarized dermoscopy imaging (confirmed by
inspecting raw RGB means — Blue channel was as high as Red on a visually light-skinned
sample), which corrupts the `b*` channel ITA depends on. A gray-world white-balance
correction was attempted to rescue the standard formula, but this is mathematically
self-defeating: gray-world correction forces the image's average color toward neutral
gray by definition, which drives `b*` toward zero for nearly every image regardless of
true content — destroying the exact signal being measured, and producing unstable
values at the arctan asymptote.

**Final method used: L\* (lightness) only**, dropping the `b*`-dependent color
component entirely. This is a simplified proxy, not the textbook ITA formula, and is
labeled as such throughout. Skin-tone groups are **quartiles of this dataset's own L\*
distribution** (not fixed universal thresholds, since absolute L* values don't reliably
transfer across different cameras/lighting) — `darkest_quartile`, `dark_medium_quartile`,
`light_medium_quartile`, `lightest_quartile`.

Quartile edges (L*): 58.99 / 65.25 / 70.88. Group sizes on the full dataset are
near-equal by construction (~2503-2505 each), with malignant counts of 431-551 per
group — sufficient for a stable per-group sensitivity estimate.

### Results (test set, calibrated probabilities, ~95%-specificity operating point)

| Group | n | n malignant | Sensitivity | Specificity |
|---|---|---|---|---|
| lightest_quartile | 398 | 81 | **0.5556** | 0.9558 |
| light_medium_quartile | 371 | 81 | 0.5185 | 0.9517 |
| dark_medium_quartile | 388 | 64 | 0.5156 | 0.9475 |
| darkest_quartile | 348 | 62 | **0.4194** | 0.9476 |

### Honest interpretation

Sensitivity degrades consistently from the lightest to the darkest quartile (55.6% →
41.9%), a real and consistent trend across all four groups, not noise concentrated in
one bucket. Specificity is essentially flat across all groups (~0.948–0.956) — the model
is not producing more false alarms on darker skin, it is specifically **missing more
actual malignant cases** on darker skin. In relative terms, the darkest quartile misses
malignant lesions about 30% more often than the lightest quartile (58.1% missed vs.
44.4% missed).

**Critical limitation on interpreting this table:** HAM10000 skews heavily toward light
skin overall — an earlier attempt to bucket by proper color-based skin tone found almost
no samples in genuinely dark categories at all. "Darkest quartile" here means the
darkest *relative to this mostly-light dataset*, not necessarily genuinely dark skin
(e.g., Fitzpatrick V-VI) in absolute terms. **This finding should be read as "sensitivity
degrades even within a population skewed toward lighter skin tones," not as "we have
validated performance on dark skin."** The latter claim would not be supported by this
data. A genuine assessment of performance on darker skin tones requires testing against
a supplementary dataset with real (not proxy) skin-tone labels and meaningful
representation of darker Fitzpatrick types — this is a documented gap, not a solved
problem.

## ONNX export

Exported via `torch.onnx.export(..., opset_version=17, dynamic_axes=..., dynamo=False)`.
Verified against PyTorch on 5 real test-set images: max abs diff 2.575e-5, mean abs diff
8.85e-6 — well within tolerance (`rtol=1e-3, atol=1e-5`).

**Note for the Day-9 inference pipeline:** the exported ONNX model outputs raw logits
from the *uncalibrated* classifier. Temperature scaling (T=1.674) is a single learned
scalar applied post-hoc to logits before the sigmoid — it is not baked into the ONNX
graph, and must be applied explicitly (`sigmoid(logit / 1.674)`) wherever this model is
served, or the output will be the overconfident, pre-calibration probability.

## Artifacts persisted

- `best_classifier.pt` — best checkpoint (epoch 8, val AUC 0.9276), downloaded locally.
- `classifier_model.onnx` — ~16MB, verified equivalent to the PyTorch checkpoint above,
  downloaded locally.

## Known limitations (summary)

- Sensitivity @ 95% specificity is ~51% overall — roughly half of malignant cases are
  missed at that operating point. This is the most important limitation in this card.
- Calibration is improved but not fully fixed by temperature scaling; residual
  overconfidence remains at high-confidence outputs.
- The fairness audit uses a simplified L*-only lightness proxy, not true skin-tone
  labels, and cannot make claims about performance on genuinely dark skin due to
  HAM10000's light-skin skew. The measured sensitivity gap (lightest vs. darkest
  quartile) is real within this data but should not be generalized beyond it.
- No lesion cropping was used; unclear whether cropping to the Day-5 segmentation mask
  would change these numbers.
