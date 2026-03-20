from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.pipeline import AnalysisPipeline
from core.worldfile_scale import read_worldfile_px_per_m


BBox = Tuple[int, int, int, int]


@dataclass
class MatchStats:
    gt: int = 0
    pred: int = 0
    matched: int = 0

    @property
    def precision(self) -> float:
        return self.matched / self.pred if self.pred else 0.0

    @property
    def recall(self) -> float:
        return self.matched / self.gt if self.gt else 0.0


def load_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot load image: {path}")
    return img


def palette_mask(image: np.ndarray, palette: List[Tuple[int, int, int]], tolerance: int = 4) -> np.ndarray:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for color in palette:
        ref = np.array(color, dtype=np.int16).reshape(1, 1, 3)
        diff = np.abs(image.astype(np.int16) - ref)
        hit = np.all(diff <= tolerance, axis=2)
        mask[hit] = 255
    return mask


def extract_components(mask: np.ndarray, min_area: int, min_w: int, min_h: int) -> List[BBox]:
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes: List[BBox] = []
    for idx in range(1, n_labels):
        x, y, w, h, area = stats[idx]
        if area < min_area or w < min_w or h < min_h:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def extract_processed_markup(processed_path: Path) -> Dict[str, List[BBox]]:
    image = load_image(processed_path)

    facade_mask = palette_mask(image, palette=[(255, 0, 0)])
    window_mask = palette_mask(image, palette=[(180, 30, 130)])
    door_mask = palette_mask(image, palette=[(50, 100, 200)])
    socle_mask = palette_mask(image, palette=[(255, 0, 255)])

    facades = extract_components(facade_mask, min_area=500, min_w=120, min_h=80)
    windows = extract_components(window_mask, min_area=80, min_w=24, min_h=28)
    doors = extract_components(door_mask, min_area=80, min_w=24, min_h=40)
    socles = extract_components(socle_mask, min_area=120, min_w=100, min_h=10)

    if facades:
        facades = [max(facades, key=lambda b: b[2] * b[3])]

    return {
        "facades": facades,
        "windows": windows,
        "doors": doors,
        "socles": socles,
        "image_size": [int(image.shape[1]), int(image.shape[0])],
    }


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def scale_boxes(boxes: List[BBox], src_size: List[int], dst_size: List[int]) -> List[BBox]:
    if not src_size or not dst_size or len(src_size) != 2 or len(dst_size) != 2:
        return list(boxes)
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        return list(boxes)
    sx = dst_w / src_w
    sy = dst_h / src_h
    scaled = []
    for x, y, w, h in boxes:
        nx = int(round(x * sx))
        ny = int(round(y * sy))
        nw = max(1, int(round(w * sx)))
        nh = max(1, int(round(h * sy)))
        scaled.append((nx, ny, nw, nh))
    return scaled


def greedy_match(gt_boxes: List[BBox], pred_boxes: List[BBox], iou_thr: float) -> Dict[str, object]:
    candidates = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            iou = bbox_iou(gt, pred)
            if iou >= iou_thr:
                candidates.append((iou, gi, pi))
    candidates.sort(reverse=True)

    used_gt = set()
    used_pred = set()
    matches = []
    for iou, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append({"gt_index": gi, "pred_index": pi, "iou": round(float(iou), 3)})

    return {
        "gt": len(gt_boxes),
        "pred": len(pred_boxes),
        "matched": len(matches),
        "matches": matches,
        "missed_gt": [gt_boxes[i] for i in range(len(gt_boxes)) if i not in used_gt],
        "extra_pred": [pred_boxes[i] for i in range(len(pred_boxes)) if i not in used_pred],
    }


def find_raw_match(raw_root: Path, processed_name: str) -> Path | None:
    normalized = processed_name.replace("_procesat", "")
    normalized = normalized.replace("_raw", "")
    raw_by_name = {p.name.lower(): p for p in raw_root.rglob("*.png")}
    return raw_by_name.get(normalized.lower())


def predict_raw(raw_path: Path) -> Dict[str, object]:
    image = load_image(raw_path)
    wf = read_worldfile_px_per_m(raw_path)
    px_per_m = wf["px_per_m"] if wf else None
    pipeline = AnalysisPipeline()
    pipeline.ocr_engine.extract_all_text = lambda _image: []
    pipeline.ocr_engine.enrich_regions = lambda _image, _regions: None
    pipeline.run(image, manual_linear_scale_px_per_m=px_per_m)
    det = pipeline.detection_results or {}
    return {
        "facades": [tuple(map(int, r.bbox)) for r in det.get("facades", []) if r.bbox],
        "windows": [tuple(map(int, r.bbox)) for r in det.get("windows", []) if r.bbox],
        "doors": [tuple(map(int, r.bbox)) for r in det.get("doors", []) if r.bbox],
        "socles": [tuple(map(int, r.bbox)) for r in det.get("socles", []) if r.bbox],
        "pgw_px_per_m": px_per_m,
        "image_size": [int(image.shape[1]), int(image.shape[0])],
    }


def evaluate(dataset_root: Path, iou_thr: float) -> Dict[str, object]:
    processed_root = dataset_root / "procesate"
    raw_root = dataset_root / "raw"
    aggregate = {name: MatchStats() for name in ("facades", "windows", "doors")}
    samples = []

    for processed_path in sorted(processed_root.rglob("*.png")):
        raw_path = find_raw_match(raw_root, processed_path.name)
        if raw_path is None:
            continue

        gt = extract_processed_markup(processed_path)
        pred = predict_raw(raw_path)
        gt_scaled = {
            **gt,
            "facades": scale_boxes(gt["facades"], gt["image_size"], pred["image_size"]),
            "windows": scale_boxes(gt["windows"], gt["image_size"], pred["image_size"]),
            "doors": scale_boxes(gt["doors"], gt["image_size"], pred["image_size"]),
            "socles": scale_boxes(gt["socles"], gt["image_size"], pred["image_size"]),
        }
        sample = {
            "processed": str(processed_path.relative_to(dataset_root)),
            "raw": str(raw_path.relative_to(dataset_root)),
            "pgw_px_per_m": pred["pgw_px_per_m"],
            "gt": gt_scaled,
            "gt_original": gt,
            "pred": pred,
            "metrics": {},
        }
        for key in ("facades", "windows", "doors"):
            stats = greedy_match(gt_scaled[key], pred[key], iou_thr=iou_thr)
            sample["metrics"][key] = stats
            agg = aggregate[key]
            agg.gt += stats["gt"]
            agg.pred += stats["pred"]
            agg.matched += stats["matched"]
        samples.append(sample)

    return {
        "dataset_root": str(dataset_root),
        "iou_threshold": iou_thr,
        "aggregate": {
            key: {
                **asdict(value),
                "precision": round(value.precision, 3),
                "recall": round(value.recall, 3),
            }
            for key, value in aggregate.items()
        },
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate raw detections against processed markup examples.")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "detector cantitati")
    parser.add_argument("--iou-threshold", type=float, default=0.35)
    parser.add_argument("--output-json", type=Path, default=ROOT / "detector cantitati" / "processed_eval_report.json")
    args = parser.parse_args()

    report = evaluate(args.dataset_root, args.iou_threshold)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    agg = report["aggregate"]
    print("Processed-markup evaluation")
    print(" windows:", agg["windows"])
    print(" doors:  ", agg["doors"])
    print(" facades:", agg["facades"])
    print(" report:", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

