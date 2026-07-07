# Segmentation Model Evaluation Report

Model: DeepLabV3+ (ResNet-34 encoder, ImageNet-pretrained) trained on ISIC2018 Task 1
lesion segmentation masks. This report covers the pipeline sanity check, training,
held-out test evaluation, qualitative failure analysis, and ONNX export verification.

## Deviations from `docs/data_sources.md`

Two things were decided differently in practice than originally documented there, and
the contract doc should be updated to match:

- **Test split source.** The contract said to inherit ISIC2018's organizer-provided
  Training/Validation/Test folders as-is. In practice, the `Validation_Input` and
  `Test_Input` folders have no released ground-truth masks (they were held back for the
  competition leaderboard), so they cannot be used for supervised evaluation. Instead,
  all 2,594 labeled images in `Training_Input`/`Training_GroundTruth` were split
  ourselves into train/val/test.
- **Training resolution.** The contract specified ~512x512. Training and evaluation
  were run at **256x256** to keep iteration and GPU time manageable during initial
  development. This should be revisited if a higher-resolution pass is done later — a
  note to that effect belongs in the model card.

## Pipeline sanity check (overfit test)

Before any real training, the full pipeline (Dataset → albumentations augmentation →
DeepLabV3+ → Dice+BCE loss → optimizer step) was validated by overfitting a fixed
10-image subset for 100 epochs with no augmentation:

| Epoch | Loss | Pixel IoU |
|---|---|---|
| 0 | 1.35 | 0.28 |
| 10 | 0.30 | 0.92 |
| 50 | 0.05 | 0.98 |
| 99 | 0.03 | 0.98 |

Loss collapsed and IoU converged to ~0.98, and a visual check confirmed the predicted
mask matched the ground truth mask on the sample image (no inversion, no misalignment).
This confirmed the pipeline itself was correct before committing GPU time to full
training.

## A methodology issue caught and corrected

The first full training run used a single train/val split (2,204 / 390 images). That
val set drove three model-selection decisions every epoch: the `ReduceLROnPlateau`
learning-rate schedule, checkpoint selection (best val IoU), and early stopping. Reusing
that same val set as a "test set" afterward would have reported an inflated, circular
metric — the checkpoint would necessarily look best on the exact data used to pick it.

This was caught before any test number was reported. The fix: a proper three-way split
(train 1,815 / val 389 / test 390, `random_state=42`, verified disjoint by set
intersection) was built, and the model was retrained from scratch on the new train/val
so that the test set was never touched by any training decision. Both the original
(`best_segmentation_model.pt`) and corrected (`best_segmentation_model1.pt`) checkpoints
are retained for the record; **all metrics below are from the corrected model**, since it
is the only one with an honest, leak-free test evaluation.

## Training

- Architecture: DeepLabV3+, ResNet-34 encoder, ImageNet-pretrained weights, via
  `segmentation-models-pytorch` 0.5.0.
- Loss: custom Dice + BCE (`BCEWithLogitsLoss` + soft Dice, summed).
- Optimizer: Adam, initial LR 1e-3, `ReduceLROnPlateau` (mode=max on val IoU, factor
  0.5, patience 3).
- Early stopping: patience 7 epochs on val IoU.
- Batch size 32, dual Tesla T4 (`nn.DataParallel`), 256x256 input, albumentations
  augmentation (flips, 90° rotation, affine shift/scale/rotate, brightness/contrast).
- Training ran 35 epochs before early stopping (epoch 34), best checkpoint at epoch 27.

| Split | IoU | Dice |
|---|---|---|
| Validation (best epoch) | 0.8096 | 0.8835 |
| **Test (held-out, first and only evaluation)** | **0.8127** | **0.8820** |

Test IoU came in slightly *above* validation IoU, which is a reassuring sign — the
model wasn't overfit to whatever quirks were in the val set that drove its selection.

A second independent retrain (after a Kaggle session restart lost the first corrected
checkpoint before it could be downloaded) reproduced a near-identical validation result
(0.8115 vs. 0.8096), which is a useful secondary confirmation that these numbers are a
stable property of the setup rather than a lucky training run.

## Qualitative results

Best and worst cases were identified by per-image IoU on the 390-image test set.

**Best cases:** IoU 0.976–0.979. Visually, these are clean, well-contained, single
lesions with clear contrast against surrounding skin — unremarkable in the way you'd
want the easy cases to be.

**Failure cases:** IoU 0.017–0.073. Two distinct failure patterns emerged, and they are
not the same problem:

1. In three of the four worst cases (`ISIC_0014457`, `ISIC_0012837`, `ISIC_0014489`),
   the ground-truth mask covers a large fraction of the frame, while the model predicted
   a small, tightly-bounded blob around the one visually obvious pigmented spot. This
   looks less like a model failure and more like a possible ground-truth annotation
   question — a mask covering ~80% of the frame for what is visually a small, contained
   lesion is unusual for ISIC2018 Task 1. **This needs a manual look at those three raw
   masks** to confirm whether the annotation is intentional (e.g., a genuinely diffuse
   pigmented patch) or a data quality issue, before drawing conclusions about model
   performance from these specific cases.
2. The fourth case (`ISIC_0015251`) is a genuine model limitation: the ground truth is a
   fine, fragmented, speckled multifocal pigment pattern, and the model predicted one
   solid connected blob covering the whole cluster. The model has no mechanism for
   producing disconnected/speckled masks — it defaults to single blobs. Worth noting in
   the model card as a known limitation class.

## ONNX export and verification

Exported with `torch.onnx.export(..., opset_version=17, dynamic_axes=..., dynamo=False)`
— the `dynamo=False` flag was required to force the legacy TorchScript-based exporter,
since the newer dynamo-based default in this PyTorch version needs an additional
`onnxscript` dependency not installed. Dynamic batch axis is supported (useful for Day 9
inference batching).

Verified numerically against the PyTorch model on 5 real test-set images (not synthetic
noise, to catch any preprocessing mismatch):

- Max absolute difference: 8.85e-6
- Mean absolute difference: 8.0e-7
- `np.testing.assert_allclose(rtol=1e-3, atol=1e-5)` — **passed**

## Artifacts persisted

- `best_segmentation_model.pt` — original (leaky-val-selection) checkpoint, retained
  for the audit trail only, not used for any reported metric.
- `best_segmentation_model1.pt` — corrected checkpoint, ~90MB, val IoU 0.8096, used for
  all metrics and the ONNX export in this report.
- `segmentation_model.onnx` — ~90MB, verified equivalent to the PyTorch checkpoint above.

## Known limitations

- No k-fold or repeated-split variance estimate — all numbers are from a single
  70/15/15 split at `random_state=42`. The metric could shift somewhat under a
  different split; it hasn't been quantified.
- No encoder or augmentation ablation was run (Day 4 allowed for this but time was
  spent instead on fixing the split-leakage issue). ResNet-34 and the augmentation set
  above are the only configuration tried.
- The GT-boundary question flagged in three failure cases above is unresolved and
  should be checked before citing the failure-case IoUs as pure model error.
- Training/eval resolution (256x256) is lower than originally planned (512x512);
  revisit if segmentation quality on small lesions becomes a bottleneck downstream.
