"""Clean dataset_raw/data and split it 70/15/15 into dataset/.

Raw files are never modified. Run from the project root:
    python training\prepare_dataset.py
Add --force to rebuild an existing split.
"""
import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset_raw" / "data"
OUT = ROOT / "dataset"
REPORT = ROOT / "results" / "dataset_report.json"

CLASSES = ["with_mask", "without_mask"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  # test gets the remainder (0.15)
MIN_SIDE = 32  # images smaller than this are reported, not removed


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_class(cls):
    """Return valid files (path, md5) plus problem lists for one class."""
    folder = RAW / cls
    valid, corrupt, non_image, small = [], [], [], []
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in EXTS:
            non_image.append(p.name)
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                im.load()
                w, h = im.size
        except Exception:
            corrupt.append(p.name)
            continue
        if min(w, h) < MIN_SIDE:
            small.append(p.name)
        valid.append((p, file_md5(p)))
    return valid, corrupt, non_image, small


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild existing split")
    args = ap.parse_args()

    for cls in CLASSES:
        if not (RAW / cls).is_dir():
            sys.exit(f"Missing folder: {RAW / cls}")

    split_dirs = [OUT / s for s in ("train", "validation", "test")]
    if any(d.exists() for d in split_dirs):
        if not args.force:
            sys.exit("dataset/ already has a split. Re-run with --force to rebuild it.")
        for d in split_dirs:
            shutil.rmtree(d, ignore_errors=True)

    report = {"seed": SEED, "ratios": [0.70, 0.15, 0.15], "classes": {}}
    scanned = {}
    for cls in CLASSES:
        scanned[cls] = scan_class(cls)

    # Hashes that appear in both classes = conflicting labels, drop everywhere
    hashes_by_class = {c: {h for _, h in scanned[c][0]} for c in CLASSES}
    cross = hashes_by_class[CLASSES[0]] & hashes_by_class[CLASSES[1]]

    final = {}
    for cls in CLASSES:
        valid, corrupt, non_image, small = scanned[cls]
        seen, unique, dup_count, cross_count = set(), [], 0, 0
        for p, h in valid:
            if h in cross:
                cross_count += 1
                continue
            if h in seen:
                dup_count += 1
                continue
            seen.add(h)
            unique.append(p)
        final[cls] = unique
        report["classes"][cls] = {
            "files_found": len(valid) + len(corrupt) + len(non_image),
            "corrupt_removed": len(corrupt),
            "non_image_removed": len(non_image),
            "exact_duplicates_removed": dup_count,
            "cross_class_conflicts_removed": cross_count,
            "small_images_flagged_kept": len(small),
            "kept_after_cleaning": len(unique),
        }

    rng = random.Random(SEED)
    for cls in CLASSES:
        files = list(final[cls])
        rng.shuffle(files)
        n = len(files)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)
        parts = {
            "train": files[:n_train],
            "validation": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for split, items in parts.items():
            dest = OUT / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for p in items:
                shutil.copy2(p, dest / p.name)
            report["classes"][cls][f"{split}_count"] = len(items)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Dataset preparation report ===")
    for cls in CLASSES:
        print(f"\n[{cls}]")
        for k, v in report["classes"][cls].items():
            print(f"  {k}: {v}")
    print(f"\nSaved: {REPORT}")


if __name__ == "__main__":
    main()
