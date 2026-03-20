"""Export current pipeline detection results as GT annotation.

Usage:
  python tools/eval/export_detection_as_gt.py --image "detector cantitati/raw/lot/NUME.png"

This runs the pipeline on the image and saves the detection as a GT JSON file
in the matching gt/ directory. Use this for images with color markup where
the detection is already correct.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.pipeline import AnalysisPipeline
from core.worldfile_scale import read_worldfile_px_per_m


def main():
    parser = argparse.ArgumentParser(description="Export pipeline detection as GT.")
    parser.add_argument("--image", type=str, required=True, help="Path to raw image")
    parser.add_argument("--force", action="store_true", help="Overwrite existing GT")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.is_absolute():
        img_path = ROOT / img_path
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return 1

    # Determine GT output path
    # raw/subdir/file.png -> gt/subdir/file.json
    rel = img_path.relative_to(ROOT / "detector cantitati" / "raw")
    gt_path = ROOT / "detector cantitati" / "gt" / rel.with_suffix(".json")
    gt_path.parent.mkdir(parents=True, exist_ok=True)

    if gt_path.exists() and not args.force:
        with open(gt_path, encoding="utf-8") as f:
            existing = json.load(f)
        has_data = any(
            len(existing.get(k, [])) > 0
            for k in ["facades", "windows", "doors", "socles"]
        )
        if has_data:
            print(f"GT already exists and has data: {gt_path}")
            print("Use --force to overwrite.")
            return 1

    # Load image
    data = np.fromfile(str(img_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Cannot load image: {img_path}")
        return 1

    img_h, img_w = image.shape[:2]

    # Get scale from worldfile
    world = read_worldfile_px_per_m(str(img_path))
    px_per_m = world["px_per_m"] if world else None

    # Run pipeline
    pipeline = AnalysisPipeline()
    pipeline.ocr_engine.extract_all_text = lambda _image: []
    pipeline.ocr_engine.enrich_regions = lambda _image, _regions: None
    pipeline.run(image, manual_linear_scale_px_per_m=px_per_m)

    det = pipeline.detection_results or {}

    # Extract bounding boxes
    facades = [list(r.bbox) for r in det.get("facades", []) if r.bbox]
    windows = [list(r.bbox) for r in det.get("windows", []) if r.bbox]
    doors = [list(r.bbox) for r in det.get("doors", []) if r.bbox]
    socles = [list(r.bbox) for r in det.get("socles", []) if r.bbox]

    # Build sample_id from filename
    sample_id = img_path.stem.lower().replace(" ", "_").replace("-", "_")

    gt_data = {
        "sample_id": sample_id,
        "image_size": {"width": img_w, "height": img_h},
        "source": "auto_export_from_pipeline",
        "pgw_px_per_m": round(float(px_per_m), 3) if px_per_m else None,
        "facades": [{"bbox": f, "type": "facade"} for f in facades],
        "windows": [{"bbox": w, "type": "window"} for w in windows],
        "doors": [{"bbox": d, "type": "door"} for d in doors],
        "socles": [{"bbox": s, "type": "socle"} for s in socles],
    }

    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, indent=2, ensure_ascii=False)

    print(f"Exported GT: {gt_path}")
    print(f"  Image: {img_w}x{img_h}")
    print(f"  Facades: {len(facades)}")
    print(f"  Windows: {len(windows)}")
    print(f"  Doors: {len(doors)}")
    print(f"  Socles: {len(socles)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
