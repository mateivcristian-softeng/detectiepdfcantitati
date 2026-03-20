"""Populate empty GT JSON files from processed_eval_report.json.

The eval report contains GT bounding boxes (derived from processed/annotated images)
for all 21 samples. The GT files in detector cantitati/gt/ are currently empty.
This script populates them.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    eval_path = ROOT / "detector cantitati" / "processed_eval_report.json"
    with open(eval_path, encoding="utf-8") as f:
        report = json.load(f)

    gt_dir = ROOT / "detector cantitati" / "gt"
    populated = 0
    skipped = 0

    for s in report["samples"]:
        raw_rel = s.get("raw", "")
        if not raw_rel:
            continue

        # Build GT file path: raw/subdir/file.png -> gt/subdir_raw/file.json
        # The raw paths use 'raw\subdir\file.png' format
        # The gt paths use 'gt\subdir_raw\file.json' format
        raw_name = os.path.basename(raw_rel).replace(".png", ".json").replace(".jpg", ".json")

        # Find matching GT file by filename
        gt_path = None
        for dirpath, dirnames, filenames in os.walk(str(gt_dir)):
            if raw_name in filenames:
                gt_path = Path(dirpath) / raw_name
                break

        if gt_path is None:
            print(f"  SKIP (no GT file for): {raw_name}")
            skipped += 1
            continue

        # Read existing
        with open(gt_path, encoding="utf-8") as f:
            existing = json.load(f)

        has_data = any(
            len(existing.get(k, [])) > 0
            for k in ["facades", "windows", "doors", "socles"]
        )
        if has_data:
            print(f"  ALREADY: {gt_path.name}")
            continue

        # Extract GT from eval report
        gt = s.get("gt", {})
        facades = gt.get("facades", [])
        windows = gt.get("windows", [])
        doors = gt.get("doors", [])
        socles = gt.get("socles", [])
        img_size = gt.get("image_size", [0, 0])

        gt_data = {
            "sample_id": existing.get("sample_id", ""),
            "image_size": {
                "width": img_size[0] if img_size else 0,
                "height": img_size[1] if img_size else 0,
            },
            "source": "auto-populated from processed_eval_report.json",
            "facades": [{"bbox": f_box, "type": "facade"} for f_box in facades],
            "windows": [{"bbox": w_box, "type": "window"} for w_box in windows],
            "doors": [{"bbox": d_box, "type": "door"} for d_box in doors],
            "socles": [{"bbox": s_box, "type": "socle"} for s_box in socles],
        }

        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt_data, f, indent=2, ensure_ascii=False)

        n = len(facades) + len(windows) + len(doors) + len(socles)
        populated += 1
        print(
            f"  POPULATED: {gt_path.name} "
            f"(F={len(facades)} W={len(windows)} D={len(doors)} S={len(socles)})"
        )

    print(f"\nTotal: {populated} populated, {skipped} skipped")


if __name__ == "__main__":
    main()
