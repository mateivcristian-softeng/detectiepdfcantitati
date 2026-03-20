#!/usr/bin/env python3
"""
Batch evaluator for "detector cantitati" dataset.

Features:
- scans raw images recursively
- runs eval where GT json exists
- groups metrics by scenario folder
- can initialize missing GT templates
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.eval.run_eval import run_eval


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    ascii_text = ascii_text.replace("\\", " ").replace("/", " ")
    out = []
    for ch in ascii_text.lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _list_images(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            files.append(path)
    files.sort()
    return files


def _make_gt_template(image_path: Path, gt_path: Path, sample_name: str) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        return
    h, w = img.shape[:2]
    template = {
        "sample_id": sample_name,
        "image_size": {"width": int(w), "height": int(h)},
        "facades": [],
        "windows": [],
    }
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.write_text(json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8")


def _result_dict(result_obj: Any) -> dict[str, Any]:
    if hasattr(result_obj, "to_dict"):
        return result_obj.to_dict()
    try:
        return asdict(result_obj)
    except Exception:
        return {"error": "unable_to_serialize_result"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch eval for detector cantitati dataset."
    )
    parser.add_argument(
        "--dataset-root",
        default="detector cantitati",
        help="Dataset root that contains raw/ and optional gt/",
    )
    parser.add_argument(
        "--raw-subdir",
        default="raw",
        help="Relative folder under dataset-root with input images",
    )
    parser.add_argument(
        "--gt-subdir",
        default="gt",
        help="Relative folder under dataset-root with GT json files",
    )
    parser.add_argument(
        "--output-json",
        default="detector_cantitati_eval_report.json",
        help="Output JSON report path",
    )
    parser.add_argument(
        "--init-missing-gt",
        action="store_true",
        help="Create missing GT templates under gt-subdir",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    raw_root = dataset_root / args.raw_subdir
    gt_root = dataset_root / args.gt_subdir

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw folder not found: {raw_root}")

    images = _list_images(raw_root)
    per_sample: list[dict[str, Any]] = []
    scenario_buckets: dict[str, dict[str, Any]] = {}

    for image_path in images:
        rel = image_path.relative_to(raw_root)
        scenario = rel.parts[0] if len(rel.parts) > 1 else "default"
        scenario_key = _slug(scenario) or "default"
        sample_name = _slug(str(rel.with_suffix("")))
        gt_path = gt_root / rel.with_suffix(".json")

        bucket = scenario_buckets.setdefault(
            scenario_key,
            {
                "scenario_label": scenario,
                "num_images": 0,
                "num_evaluated": 0,
                "num_missing_gt": 0,
                "facade_iou_mean_avg": 0.0,
                "window_precision_avg": 0.0,
                "window_recall_avg": 0.0,
                "window_f1_avg": 0.0,
                "area_mae_m2_avg": 0.0,
            },
        )
        bucket["num_images"] += 1

        item: dict[str, Any] = {
            "image": str(image_path),
            "scenario": scenario,
            "sample_id": sample_name,
            "gt": str(gt_path),
            "status": "",
        }

        if not gt_path.exists():
            item["status"] = "missing_gt"
            bucket["num_missing_gt"] += 1
            if args.init_missing_gt:
                _make_gt_template(image_path, gt_path, sample_name)
                item["status"] = "missing_gt_template_created"
            per_sample.append(item)
            continue

        try:
            result = run_eval(str(image_path), str(gt_path), sample_id=sample_name)
            rd = _result_dict(result)
            item["status"] = "evaluated"
            item["metrics"] = rd

            bucket["num_evaluated"] += 1
            bucket["facade_iou_mean_avg"] += float(rd.get("facade_iou_mean", 0.0))
            bucket["window_precision_avg"] += float(rd.get("window_precision", 0.0))
            bucket["window_recall_avg"] += float(rd.get("window_recall", 0.0))
            bucket["window_f1_avg"] += float(rd.get("window_f1", 0.0))
            bucket["area_mae_m2_avg"] += float(rd.get("area_mae_m2", 0.0))
        except Exception as exc:
            item["status"] = "error"
            item["error"] = str(exc)

        per_sample.append(item)

    for bucket in scenario_buckets.values():
        n = int(bucket["num_evaluated"])
        if n > 0:
            bucket["facade_iou_mean_avg"] = round(bucket["facade_iou_mean_avg"] / n, 4)
            bucket["window_precision_avg"] = round(bucket["window_precision_avg"] / n, 4)
            bucket["window_recall_avg"] = round(bucket["window_recall_avg"] / n, 4)
            bucket["window_f1_avg"] = round(bucket["window_f1_avg"] / n, 4)
            bucket["area_mae_m2_avg"] = round(bucket["area_mae_m2_avg"] / n, 4)

    summary = {
        "dataset_root": str(dataset_root),
        "raw_root": str(raw_root),
        "gt_root": str(gt_root),
        "num_images_total": len(images),
        "num_evaluated_total": sum(
            1 for item in per_sample if item.get("status") == "evaluated"
        ),
        "num_missing_gt_total": sum(
            1 for item in per_sample if item.get("status", "").startswith("missing_gt")
        ),
        "num_errors_total": sum(
            1 for item in per_sample if item.get("status") == "error"
        ),
    }
    report = {
        "summary": summary,
        "scenarios": sorted(scenario_buckets.values(), key=lambda x: x["scenario_label"]),
        "samples": per_sample,
    }

    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = Path(os.getcwd()) / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] Report saved to: {output_path}")
    print(
        "[SUMMARY] "
        f"images={summary['num_images_total']} "
        f"evaluated={summary['num_evaluated_total']} "
        f"missing_gt={summary['num_missing_gt_total']} "
        f"errors={summary['num_errors_total']}"
    )


if __name__ == "__main__":
    main()
