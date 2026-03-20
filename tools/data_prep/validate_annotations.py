"""
DrawQuantPDF - Validate annotation consistency
Checks JSON annotation files against docs/DATASET_SCHEMA.md rules.

Usage:
  python -m tools.data_prep.validate_annotations [--dir PATH] [--strict]
"""

import argparse
import json
import os
import sys
from pathlib import Path


# Bounds from OCR engine / pipeline (aligned with core)
WINDOW_AREA_MIN, WINDOW_AREA_MAX = 0.15, 15.0
WINDOW_WIDTH_MIN, WINDOW_WIDTH_MAX = 0.2, 8.0
WINDOW_HEIGHT_MIN, WINDOW_HEIGHT_MAX = 0.2, 5.0

DOOR_AREA_MIN, DOOR_AREA_MAX = 0.5, 6.5
DOOR_WIDTH_MIN, DOOR_WIDTH_MAX = 0.5, 5.0
DOOR_HEIGHT_MIN, DOOR_HEIGHT_MAX = 1.4, 5.0

FACADE_AREA_MIN, FACADE_AREA_MAX = 1.0, 5000.0

REQUIRED_ROOT = {"version", "image_file", "image_width", "image_height",
                 "scale_refs", "facades", "windows", "doors"}


def _bbox_in_image(bbox: list, w: int, h: int) -> bool:
    """Check bbox [x,y,w,h] is fully inside image."""
    if len(bbox) != 4:
        return False
    x, y, bw, bh = bbox
    return (0 <= x < w and 0 <= y < h and
            bw > 0 and bh > 0 and
            x + bw <= w and y + bh <= h)


def _bbox_center(bbox: list) -> tuple:
    x, y, w, h = bbox
    return (x + w / 2, y + h / 2)


def _bbox_contains(outer: list, inner: list, tol: int = 20) -> bool:
    """Check if inner bbox is inside outer (with optional tolerance)."""
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return (ox - tol <= ix and oy - tol <= iy and
            ox + ow + tol >= ix + iw and oy + oh + tol >= iy + ih)


def validate_annotation(filepath: Path, strict: bool = False) -> list:
    """
    Validate a single JSON annotation file.
    Returns list of error/warning strings (empty if OK).
    """
    errors = []

    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [f"JSON invalid: {e}"]
    except Exception as e:
        return [f"Read error: {e}"]

    # Root structure
    for key in REQUIRED_ROOT:
        if key not in data:
            errors.append(f"Missing required key: {key}")

    if errors:
        return errors

    version = data.get("version", "")
    img_file = data.get("image_file", "")
    img_w = data.get("image_width", 0)
    img_h = data.get("image_height", 0)
    scale_refs = data.get("scale_refs", [])
    facades = data.get("facades", [])
    windows = data.get("windows", [])
    doors = data.get("doors", [])

    if not isinstance(scale_refs, list):
        errors.append("scale_refs must be an array")
    if not isinstance(facades, list):
        errors.append("facades must be an array")
    if not isinstance(windows, list):
        errors.append("windows must be an array")
    if not isinstance(doors, list):
        errors.append("doors must be an array")

    if img_w <= 0 or img_h <= 0:
        errors.append("image_width and image_height must be positive")

    facade_labels = {f.get("label") for f in facades if f.get("label")}

    # Scale refs
    has_linear = False
    for i, sr in enumerate(scale_refs):
        if not isinstance(sr, dict):
            errors.append(f"scale_refs[{i}]: must be object")
            continue
        t = sr.get("type")
        if t == "linear":
            if sr.get("ref_m", 0) <= 0 or sr.get("ref_px", 0) <= 0:
                errors.append(f"scale_refs[{i}]: linear ref_m and ref_px must be positive")
            has_linear = True
        elif t == "linear_from_ocr":
            if sr.get("value_m", 0) <= 0:
                errors.append(f"scale_refs[{i}]: linear_from_ocr value_m must be positive")
            has_linear = True
        if "bbox" in sr and not _bbox_in_image(sr["bbox"], img_w, img_h):
            errors.append(f"scale_refs[{i}]: bbox out of image bounds")

    if not has_linear and (facades or windows or doors) and strict:
        errors.append("At least one linear scale ref recommended when annotations exist")

    # Facades
    for i, f in enumerate(facades):
        if not isinstance(f, dict):
            errors.append(f"facades[{i}]: must be object")
            continue
        label = f.get("label")
        if not label or not isinstance(label, str):
            errors.append(f"facades[{i}]: label required")
        bbox = f.get("bbox")
        if not bbox or len(bbox) != 4:
            errors.append(f"facades[{i}]: bbox [x,y,w,h] required")
        elif not _bbox_in_image(bbox, img_w, img_h):
            errors.append(f"facades[{i}]: bbox out of image bounds")
        total = f.get("total_area_m2")
        if total is not None and not (FACADE_AREA_MIN <= float(total) <= FACADE_AREA_MAX):
            errors.append(f"facades[{i}]: total_area_m2 {total} outside plausible range")

    # Windows
    for i, w in enumerate(windows):
        if not isinstance(w, dict):
            errors.append(f"windows[{i}]: must be object")
            continue
        label = w.get("label")
        if not label or not isinstance(label, str):
            errors.append(f"windows[{i}]: label required")
        bbox = w.get("bbox")
        if not bbox or len(bbox) != 4:
            errors.append(f"windows[{i}]: bbox [x,y,w,h] required")
        elif not _bbox_in_image(bbox, img_w, img_h):
            errors.append(f"windows[{i}]: bbox out of image bounds")
        parent = w.get("parent_facade")
        if parent and parent not in facade_labels and facades:
            errors.append(f"windows[{i}]: parent_facade '{parent}' not found in facades")
        area = w.get("area_m2")
        if area is not None:
            a = float(area)
            if not (WINDOW_AREA_MIN <= a <= WINDOW_AREA_MAX):
                errors.append(f"windows[{i}]: area_m2 {area} outside 0.15-15 m²")
        wm, hm = w.get("width_m"), w.get("height_m")
        if wm is not None:
            v = float(wm)
            if not (WINDOW_WIDTH_MIN <= v <= WINDOW_WIDTH_MAX):
                errors.append(f"windows[{i}]: width_m {wm} outside 0.2-8 m")
        if hm is not None:
            v = float(hm)
            if not (WINDOW_HEIGHT_MIN <= v <= WINDOW_HEIGHT_MAX):
                errors.append(f"windows[{i}]: height_m {hm} outside 0.2-5 m")

    # Doors
    for i, d in enumerate(doors):
        if not isinstance(d, dict):
            errors.append(f"doors[{i}]: must be object")
            continue
        label = d.get("label")
        if not label or not isinstance(label, str):
            errors.append(f"doors[{i}]: label required")
        bbox = d.get("bbox")
        if not bbox or len(bbox) != 4:
            errors.append(f"doors[{i}]: bbox [x,y,w,h] required")
        elif not _bbox_in_image(bbox, img_w, img_h):
            errors.append(f"doors[{i}]: bbox out of image bounds")
        parent = d.get("parent_facade")
        if parent and parent not in facade_labels and facades:
            errors.append(f"doors[{i}]: parent_facade '{parent}' not found in facades")
        area = d.get("area_m2")
        if area is not None:
            a = float(area)
            if not (DOOR_AREA_MIN <= a <= DOOR_AREA_MAX):
                errors.append(f"doors[{i}]: area_m2 {area} outside 0.5-6.5 m²")
        wm, hm = d.get("width_m"), d.get("height_m")
        if wm is not None:
            v = float(wm)
            if not (DOOR_WIDTH_MIN <= v <= DOOR_WIDTH_MAX):
                errors.append(f"doors[{i}]: width_m {wm} outside 0.5-5 m")
        if hm is not None:
            v = float(hm)
            if not (DOOR_HEIGHT_MIN <= v <= DOOR_HEIGHT_MAX):
                errors.append(f"doors[{i}]: height_m {hm} outside 1.4-5 m")

    # Spatial: windows/doors inside facade
    for f in facades:
        fb = f.get("bbox")
        if not fb or len(fb) != 4:
            continue
        for i, w in enumerate(windows):
            if w.get("parent_facade") == f.get("label") and w.get("bbox"):
                if not _bbox_contains(fb, w["bbox"]):
                    errors.append(f"windows[{i}] '{w.get('label')}' not fully inside facade '{f.get('label')}'")
        for i, d in enumerate(doors):
            if d.get("parent_facade") == f.get("label") and d.get("bbox"):
                if not _bbox_contains(fb, d["bbox"]):
                    errors.append(f"doors[{i}] '{d.get('label')}' not fully inside facade '{f.get('label')}'")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate DrawQuantPDF annotations")
    parser.add_argument("--dir", "-d", default=None,
                        help="Directory to validate (default: data/train data/val data/test)")
    parser.add_argument("--strict", "-s", action="store_true",
                        help="Strict mode: require scale ref when annotations exist")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print OK files")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[2]
    if args.dir:
        dirs = [Path(args.dir)]
    else:
        dirs = [
            base / "data" / "train",
            base / "data" / "val",
            base / "data" / "test",
        ]

    total = 0
    failed = 0

    for d in dirs:
        if not d.exists():
            print(f"Skip (missing): {d}")
            continue
        json_files = sorted(d.glob("*.json"))
        for jf in json_files:
            total += 1
            errs = validate_annotation(jf, strict=args.strict)
            if errs:
                failed += 1
                print(f"FAIL {jf.relative_to(base)}:")
                for e in errs:
                    print(f"  - {e}")
            elif args.verbose:
                print(f"OK   {jf.relative_to(base)}")

    print(f"\nValidated: {total} | Failed: {failed} | Passed: {total - failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
