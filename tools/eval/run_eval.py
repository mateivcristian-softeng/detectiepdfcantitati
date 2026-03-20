#!/usr/bin/env python3
"""
DrawQuantPDF - Metrics Evaluator CLI
Run evaluation on fixtures and output JSON/CSV.
"""

import argparse
import csv
import json
import os
import sys

# Allow import from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
from dataclasses import dataclass

# Import pipeline - we use it, don't modify it
from core.pipeline import AnalysisPipeline

from tools.eval.metrics import evaluate_sample, EvalResult


@dataclass
class GTRegion:
    """Ground truth region from JSON."""
    bbox: tuple  # (x, y, w, h)
    area_m2: float = 0.0
    label: str = ""


def load_ground_truth(path: str) -> tuple[list, list]:
    """Load GT from JSON. Returns (facades, windows)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    facades = []
    for fa in data.get("facades", []):
        b = fa.get("bbox", [0, 0, 1, 1])
        facades.append(GTRegion(
            bbox=(b[0], b[1], b[2], b[3]),
            area_m2=float(fa.get("area_m2", 0) or fa.get("total_area_m2", 0)),
            label=fa.get("name", ""),
        ))
    windows = []
    for w in data.get("windows", []):
        b = w.get("bbox", [0, 0, 1, 1])
        windows.append(GTRegion(
            bbox=(b[0], b[1], b[2], b[3]),
            area_m2=float(w.get("area_m2", 0) or w.get("area", 0)),
            label=w.get("label", ""),
        ))
    return (facades, windows)


def pipeline_to_eval_input(pipeline: AnalysisPipeline) -> tuple[list, list]:
    """Extract pred facades and windows from pipeline for metrics."""
    pred_facades = []
    for dr in pipeline.detection_results.get("facades", []):
        pred_facades.append(_DetectWrapper(dr))

    pred_windows = []
    for dr in pipeline.detection_results.get("windows", []):
        pred_windows.append(_DetectWrapper(dr))

    return (pred_facades, pred_windows)


class _DetectWrapper:
    """Wrap DetectedRegion/parsed for metrics (bbox, area)."""
    def __init__(self, r):
        self.bbox = getattr(r, "bbox", None) or getattr(r, "region_bbox", None)
        self.position = getattr(r, "center", None) or (
            (r.bbox[0] + r.bbox[2] // 2, r.bbox[1] + r.bbox[3] // 2) if getattr(r, "bbox", None) else (0, 0)
        )
        self.area_m2 = getattr(r, "area_m2", None) or getattr(r, "area", 0) or 0.0


def run_eval(image_path: str, gt_path: str, sample_id: str = "") -> EvalResult:
    """Run pipeline on image, compare with GT, return EvalResult."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    pipeline = AnalysisPipeline()
    pipeline.run(img)

    pred_facades, pred_windows = pipeline_to_eval_input(pipeline)
    gt_facades, gt_windows = load_ground_truth(gt_path)

    return evaluate_sample(
        pred_facades=pred_facades,
        pred_windows=pred_windows,
        gt_facades=gt_facades,
        gt_windows=gt_windows,
        sample_id=sample_id or os.path.basename(image_path),
    )


def run_batch(fixtures_dir: str) -> list[EvalResult]:
    """Run eval on all fixtures in dir. Expects images/ and gt/ subdirs."""
    images_dir = os.path.join(fixtures_dir, "images")
    gt_dir = os.path.join(fixtures_dir, "gt")
    if not os.path.isdir(images_dir) or not os.path.isdir(gt_dir):
        return []
    results = []
    for name in os.listdir(images_dir):
        base, ext = os.path.splitext(name)
        if ext.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        gt_path = os.path.join(gt_dir, base + ".json")
        img_path = os.path.join(images_dir, name)
        if not os.path.exists(gt_path):
            continue
        try:
            r = run_eval(img_path, gt_path, base)
            results.append(r)
        except Exception as e:
            results.append(EvalResult(sample_id=base))
    return results


def main():
    parser = argparse.ArgumentParser(description="DrawQuantPDF Metrics Evaluator")
    parser.add_argument("image", nargs="?", help="Path to input image")
    parser.add_argument("gt", nargs="?", help="Path to ground truth JSON")
    parser.add_argument("--batch", "-b", metavar="FIXTURES_DIR",
                        help="Run on fixtures dir (images/ + gt/)")
    parser.add_argument("--id", dest="sample_id", default="", help="Sample ID")
    parser.add_argument("--output", "-o", choices=["json", "csv"], default="json")
    parser.add_argument("--out-file", help="Output file path")
    args = parser.parse_args()

    if args.batch:
        results = run_batch(args.batch)
        data = {"samples": [r.to_dict() for r in results], "count": len(results)}
        out = args.out_file or sys.stdout
        if isinstance(out, str):
            out = open(out, "w", encoding="utf-8")
        try:
            if args.output == "json":
                json.dump(data, out, indent=2, ensure_ascii=False)
            else:
                if results:
                    w = csv.DictWriter(out, fieldnames=results[0].to_dict().keys())
                    w.writeheader()
                    for r in results:
                        w.writerow(r.to_dict())
        finally:
            if out is not sys.stdout:
                out.close()
        return

    if not args.image or not args.gt:
        parser.error("image and gt required when not using --batch")
    result = run_eval(args.image, args.gt, args.sample_id)

    out = args.out_file or sys.stdout
    if isinstance(out, str):
        out = open(out, "w", encoding="utf-8")
    try:
        if args.output == "json":
            json.dump(result.to_dict(), out, indent=2, ensure_ascii=False)
        else:
            w = csv.DictWriter(out, fieldnames=result.to_dict().keys())
            w.writeheader()
            w.writerow(result.to_dict())
    finally:
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    main()
