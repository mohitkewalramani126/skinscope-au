"""
Day 13 monitoring: Evidently drift report.

Honest scope note (read before trusting this as "production drift
monitoring"): the real training set (HAM10000, 10,015 images) lives on
Kaggle, not in this repo -- only two local sample photos are kept here
(test_images/), and no incoming user photos are persisted (the disclaimer
banner says "your photo is analysed, not stored", and that's true -- the
FastAPI layer never writes uploaded bytes to disk). So there is no real
"training distribution vs. live traffic" data pair available locally to
compare.

What this script actually does: computes simple, defensible image-level
feature stats (brightness, contrast, per-channel colour means, resolution,
aspect ratio) for two directories of images and runs Evidently's
DataDriftPreset between them. Out of the box, both --reference-dir and
--current-dir default to test_images/, which will correctly report "no
drift" (comparing a set to itself) -- that's a smoke test proving the
pipeline works, not a claim that real drift has been checked. To use this
for real: point --current-dir at a folder of newly-collected photos (e.g.
saved manually from real usage with consent) and --reference-dir at a
representative sample of the training distribution.

Run: python evaluation/drift_report.py --reference-dir test_images --current-dir test_images
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _image_features(path: Path) -> dict:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float64)
    w, h = img.size
    return {
        "filename": path.name,
        "width": w,
        "height": h,
        "aspect_ratio": w / h,
        "megapixels": (w * h) / 1_000_000,
        "brightness_mean": arr.mean(),
        "brightness_std": arr.std(),  # a simple contrast proxy
        "red_mean": arr[:, :, 0].mean(),
        "green_mean": arr[:, :, 1].mean(),
        "blue_mean": arr[:, :, 2].mean(),
    }


def _features_for_dir(dir_path: Path) -> pd.DataFrame:
    paths = sorted(p for p in dir_path.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
    if not paths:
        raise ValueError(f"No images found in {dir_path} (looked for {IMG_EXTENSIONS})")
    return pd.DataFrame([_image_features(p) for p in paths])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-dir", default="test_images", help="Folder standing in for the training distribution")
    parser.add_argument("--current-dir", default="test_images", help="Folder of newer/incoming images to compare")
    parser.add_argument("--output", default="evaluation/drift_report.html", help="Output HTML report path")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    ref_dir = (root / args.reference_dir) if not Path(args.reference_dir).is_absolute() else Path(args.reference_dir)
    cur_dir = (root / args.current_dir) if not Path(args.current_dir).is_absolute() else Path(args.current_dir)

    print(f"Reference: {ref_dir} | Current: {cur_dir}")
    reference_df = _features_for_dir(ref_dir).drop(columns=["filename"])
    current_df = _features_for_dir(cur_dir).drop(columns=["filename"])

    if ref_dir == cur_dir:
        print(
            "NOTE: reference and current are the same folder -- this will report "
            "no drift by construction. This is a smoke test of the pipeline, not a "
            "real drift check. Pass --current-dir with a different folder of images "
            "for a meaningful comparison."
        )

    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report([DataDriftPreset()])
    my_eval = report.run(current_df, reference_df)

    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    my_eval.save_html(str(output_path))
    print(f"Drift report written to {output_path}")


if __name__ == "__main__":
    main()
