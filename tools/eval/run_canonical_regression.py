from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.pipeline import AnalysisPipeline
from core.worldfile_scale import read_worldfile_px_per_m
from tools.eval.reference_validation import (
    expand_norm_bounds,
    polygon_iou_norm as shared_polygon_iou_norm,
    polygon_norm_bounds,
    validate_reference_dict,
)


def load_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot load image: {path}")
    return image


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _bbox_to_dict(bbox: tuple[int, int, int, int] | None) -> dict[str, int] | None:
    if not bbox:
        return None
    x, y, w, h = map(int, bbox)
    return {"x": x, "y": y, "w": w, "h": h}


def _contour_to_points(contour: Any, closed: bool = True) -> list[list[int]]:
    if not isinstance(contour, np.ndarray) or contour.size < 4:
        return []
    pts = contour.reshape(-1, 2).astype(np.int32)
    if pts.shape[0] < 2:
        return []
    if closed and pts.shape[0] >= 3:
        contour_i = pts.reshape((-1, 1, 2))
        perim = float(cv2.arcLength(contour_i, True))
        eps = max(1.0, perim * 0.004)
        pts = cv2.approxPolyDP(contour_i, eps, True).reshape(-1, 2).astype(np.int32)
    elif not closed and pts.shape[0] > 2:
        order = np.argsort(pts[:, 0], kind="mergesort")
        polyline = pts[order].reshape((-1, 1, 2))
        perim = float(cv2.arcLength(polyline, False))
        eps = max(1.0, perim * 0.004)
        approx = cv2.approxPolyDP(polyline, eps, False)
        pts = approx.reshape(-1, 2).astype(np.int32) if approx is not None and approx.size >= 4 else pts[order]
    return [[int(x), int(y)] for x, y in pts.tolist()]


def _count_check(actual: int, expected: dict[str, Any], key: str) -> list[str]:
    warnings: list[str] = []
    if key in expected and actual != int(expected[key]):
        warnings.append(f"{key}: expected {expected[key]}, got {actual}")
    min_key = f"min_{key}"
    if min_key in expected and actual < int(expected[min_key]):
        warnings.append(f"{key}: expected >= {expected[min_key]}, got {actual}")
    max_key = f"max_{key}"
    if max_key in expected and actual > int(expected[max_key]):
        warnings.append(f"{key}: expected <= {expected[max_key]}, got {actual}")
    return warnings


def _line_y_at_x(points: list[list[int]], x: float) -> float | None:
    if len(points) < 2:
        return None
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    if float(x) <= float(pts[0][0]):
        return float(pts[0][1])
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if x1 == x0:
            if abs(float(x) - float(x0)) <= 1e-6:
                return float(min(y0, y1))
            continue
        if float(x0) <= float(x) <= float(x1):
            alpha = (float(x) - float(x0)) / float(x1 - x0)
            alpha = max(0.0, min(1.0, alpha))
            return float(y0 + (y1 - y0) * alpha)
    return float(pts[-1][1])


def _geometry_metrics(
    facade_bbox: dict[str, int] | None,
    socle_bbox: dict[str, int] | None,
    socle_profile: list[list[int]],
    door_bboxes: list[dict[str, int]],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if facade_bbox and socle_bbox:
        metrics["facade_left_minus_socle_left_px"] = int(facade_bbox["x"] - socle_bbox["x"])
        metrics["socle_right_minus_facade_right_px"] = int(
            (socle_bbox["x"] + socle_bbox["w"]) - (facade_bbox["x"] + facade_bbox["w"])
        )
    if facade_bbox and socle_profile:
        x_center = facade_bbox["x"] + facade_bbox["w"] * 0.5
        y_line = _line_y_at_x(socle_profile, x_center)
        if y_line is not None:
            metrics["facade_bottom_minus_socle_profile_px"] = int(
                round((facade_bbox["y"] + facade_bbox["h"]) - y_line)
            )
    door_gaps = []
    if socle_profile and door_bboxes:
        for door in door_bboxes:
            x_center = door["x"] + door["w"] * 0.5
            y_line = _line_y_at_x(socle_profile, x_center)
            if y_line is None:
                continue
            gap = int(round((door["y"] + door["h"]) - y_line))
            door_gaps.append({"x_center": int(round(x_center)), "bottom_minus_socle_profile_px": gap})
    metrics["door_bottom_gaps_px"] = door_gaps
    return metrics


def _normalize_points_to_bbox(points: list[list[int]], bbox: dict[str, int] | None) -> list[list[float]]:
    if not points or not bbox:
        return []
    x = float(bbox["x"])
    y = float(bbox["y"])
    w = max(1.0, float(bbox["w"]))
    h = max(1.0, float(bbox["h"]))
    out = []
    for px, py in points:
        out.append([(float(px) - x) / w, (float(py) - y) / h])
    return out


def _normalize_boxes_to_bbox(boxes: list[dict[str, int]], bbox: dict[str, int] | None) -> list[dict[str, float]]:
    if not boxes or not bbox:
        return []
    x = float(bbox["x"])
    y = float(bbox["y"])
    w = max(1.0, float(bbox["w"]))
    h = max(1.0, float(bbox["h"]))
    out: list[dict[str, float]] = []
    for box in boxes:
        out.append({
            "x": (float(box["x"]) - x) / w,
            "y": (float(box["y"]) - y) / h,
            "w": float(box["w"]) / w,
            "h": float(box["h"]) / h,
        })
    return out


def _sort_norm_boxes(boxes: list[dict[str, float]]) -> list[dict[str, float]]:
    return sorted(boxes, key=lambda b: (b["x"] + b["w"] * 0.5, b["y"], b["w"], b["h"]))


def _box_iou(a: dict[str, float], b: dict[str, float]) -> float:
    ax1, ay1 = float(a["x"]), float(a["y"])
    ax2, ay2 = ax1 + float(a["w"]), ay1 + float(a["h"])
    bx1, by1 = float(b["x"]), float(b["y"])
    bx2, by2 = bx1 + float(b["w"]), by1 + float(b["h"])
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(1e-9, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return float(inter) / float(union)


def _mean_box_iou(cur_boxes: list[dict[str, float]], ref_boxes: list[dict[str, float]]) -> float | None:
    if not ref_boxes:
        return None
    if not cur_boxes:
        return 0.0
    cur_sorted = _sort_norm_boxes(cur_boxes)
    ref_sorted = _sort_norm_boxes(ref_boxes)
    if len(cur_sorted) != len(ref_sorted):
        return 0.0
    ious = [_box_iou(c, r) for c, r in zip(cur_sorted, ref_sorted)]
    if not ious:
        return None
    return float(sum(ious)) / float(len(ious))


def _polygon_iou_norm(a: list[list[float]],
                      b: list[list[float]],
                      size: int = 512,
                      bounds: dict[str, float] | None = None,
                      band_like: bool = False) -> float | None:
    return shared_polygon_iou_norm(a, b, size=size, bounds=bounds, band_like=band_like)


def _load_reference_target(case: dict[str, Any]) -> dict[str, Any] | None:
    ref_path = case.get("reference_json_path")
    if not ref_path:
        return None
    path = ROOT / ref_path
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _reference_validation(ref: dict[str, Any], raw_image: np.ndarray) -> dict[str, Any]:
    report = validate_reference_dict(ref, raw_image, "")
    size_info = report.get("image_size_check") or {}
    return {
        "issues": list(report.get("issues", [])),
        "geometry_valid": bool(report.get("geometry_valid", False)),
        "openings_valid": bool(report.get("openings_valid", False)),
        "image_size_match": bool(size_info.get("match", True)),
    }


def _reference_shape_metrics(result: dict[str, Any], case: dict[str, Any], raw_image: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    ref = _load_reference_target(case)
    if not ref:
        return {}, {"issues": [], "geometry_valid": False, "openings_valid": False, "image_size_match": True}
    validation = _reference_validation(ref, raw_image)
    targets = ref.get("targets", {})
    metrics: dict[str, Any] = {
        "reference_id": ref.get("reference_id", ""),
        "reference_has_geometry": False,
        "reference_has_openings": False,
        "reference_geometry_valid": validation.get("geometry_valid", False),
        "reference_openings_valid": validation.get("openings_valid", False),
    }

    facade_bbox = result.get("facades", [None])[0] if result.get("facades") else None
    facade_shape = result.get("shape_signatures", {}).get("facade_contour", [])
    socle_shape = result.get("shape_signatures", {}).get("socle_contour", [])

    cur_facade_norm = _normalize_points_to_bbox(facade_shape, facade_bbox)
    ref_facade_norm = targets.get("facade_contour_norm_in_bbox", [])
    facade_bounds = polygon_norm_bounds("facade_contour_norm_in_bbox", targets)
    facade_bounds = expand_norm_bounds(facade_bounds, cur_facade_norm)
    facade_bounds = expand_norm_bounds(facade_bounds, ref_facade_norm)
    facade_iou = _polygon_iou_norm(cur_facade_norm, ref_facade_norm, bounds=facade_bounds)
    metrics["facade_shape_iou"] = round(float(facade_iou), 4) if facade_iou is not None else None

    cur_socle_norm = _normalize_points_to_bbox(socle_shape, facade_bbox) if facade_bbox else []
    ref_socle_norm = targets.get("socle_contour_norm_in_facade_bbox", [])
    socle_bounds = polygon_norm_bounds("socle_contour_norm_in_facade_bbox", targets)
    socle_bounds = expand_norm_bounds(socle_bounds, cur_socle_norm)
    socle_bounds = expand_norm_bounds(socle_bounds, ref_socle_norm)
    socle_iou = _polygon_iou_norm(cur_socle_norm, ref_socle_norm, bounds=socle_bounds, band_like=True)
    metrics["socle_shape_iou_in_facade_bbox"] = round(float(socle_iou), 4) if socle_iou is not None else None
    if ref_facade_norm or ref_socle_norm:
        metrics["reference_has_geometry"] = bool(validation.get("geometry_valid", False))

    ref_window_boxes = targets.get("window_boxes_norm_in_facade_bbox", [])
    ref_door_boxes = targets.get("door_boxes_norm_in_facade_bbox", [])
    cur_window_boxes = _normalize_boxes_to_bbox(result.get("windows", []), facade_bbox)
    cur_door_boxes = _normalize_boxes_to_bbox(result.get("doors", []), facade_bbox)
    win_iou = _mean_box_iou(cur_window_boxes, ref_window_boxes)
    door_iou = _mean_box_iou(cur_door_boxes, ref_door_boxes)
    metrics["window_box_mean_iou_in_facade_bbox"] = round(float(win_iou), 4) if win_iou is not None else None
    metrics["door_box_mean_iou_in_facade_bbox"] = round(float(door_iou), 4) if door_iou is not None else None
    metrics["reference_window_count"] = len(ref_window_boxes)
    metrics["reference_door_count"] = len(ref_door_boxes)
    if ref_window_boxes or ref_door_boxes:
        metrics["reference_has_openings"] = bool(validation.get("openings_valid", False))
    return metrics, validation


def _geometry_warnings(result: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    image_w, _ = result["image_size"]
    facades = result["facades"]
    socles = result["socles"]
    geom = result.get("geometry", {})
    ref_metrics = result.get("reference_metrics", {})
    ref_validation = result.get("reference_validation", {})
    facade_shape = result.get("shape_signatures", {}).get("facade_contour", [])
    if facades:
        facade = facades[0]
        fx1 = facade["x"]
        fy1 = facade["y"]
        fx2 = fx1 + facade["w"]
        border_tol = max(4, int(round(image_w * 0.005)))
        if fx1 <= border_tol or fx2 >= image_w - border_tol:
            warnings.append("facade touches image border; likely following noise")
        if fy1 <= border_tol:
            warnings.append("facade top touches image border; likely overextended")
        if socles:
            if geom.get("facade_left_minus_socle_left_px", 0) > 5 or geom.get("socle_right_minus_facade_right_px", 0) > 5:
                warnings.append("socle extends outside facade width")
            if geom.get("facade_bottom_minus_socle_profile_px", 0) > 18:
                warnings.append("facade continues below socle top")
        if result.get("scene_class") == "gable_facade" and facade["w"] < int(image_w * 0.45):
            warnings.append("gable facade likely truncated by bright/white gap")
    facade_iou = ref_metrics.get("facade_shape_iou")
    if ref_metrics.get("reference_geometry_valid") and facade_iou is not None and facade_iou < 0.72:
        warnings.append(f"facade shape IoU below target: {facade_iou:.3f}")
    socle_iou = ref_metrics.get("socle_shape_iou_in_facade_bbox")
    if ref_metrics.get("reference_geometry_valid") and socle_iou is not None and socle_iou < 0.45:
        warnings.append(f"socle shape IoU below target: {socle_iou:.3f}")
    win_iou = ref_metrics.get("window_box_mean_iou_in_facade_bbox")
    if ref_metrics.get("reference_openings_valid") and win_iou is not None and win_iou < 0.40:
        warnings.append(f"window box mean IoU below target: {win_iou:.3f}")
    door_iou = ref_metrics.get("door_box_mean_iou_in_facade_bbox")
    if ref_metrics.get("reference_openings_valid") and door_iou is not None and door_iou < 0.45:
        warnings.append(f"door box mean IoU below target: {door_iou:.3f}")
    if ref_validation.get("issues"):
        if not ref_metrics.get("reference_geometry_valid"):
            warnings.append("reference geometry target invalid for this raw image; manual review required")
        if not ref_metrics.get("reference_openings_valid"):
            warnings.append("reference openings target invalid for this raw image; manual review required")
    if result.get("id") == "ROMCEA":
        if not ref_metrics.get("reference_has_geometry"):
            warnings.append("ROMCEA reference geometry target missing; UI review remains mandatory")
        if len(facade_shape) <= 4:
            warnings.append("ROMCEA facade contour still lacks right-eave taper geometry")
    return warnings


def _sort_boxes(boxes: list[dict[str, int]]) -> list[dict[str, int]]:
    return sorted(boxes, key=lambda b: (b["x"], b["y"], b["w"], b["h"]))


def _bbox_delta(a: dict[str, int] | None, b: dict[str, int] | None) -> int:
    if a is None and b is None:
        return 0
    if a is None or b is None:
        return 10**6
    return max(abs(int(a[k]) - int(b[k])) for k in ("x", "y", "w", "h"))


def _points_delta(a: list[list[int]], b: list[list[int]]) -> int:
    if not a and not b:
        return 0
    if len(a) != len(b):
        return 10**6
    return max(abs(int(ax) - int(bx)) + abs(int(ay) - int(by)) for (ax, ay), (bx, by) in zip(a, b))


def _compare_to_baseline(result: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    if baseline is None:
        return {"changed": None, "messages": []}
    messages: list[str] = []
    changed = False

    for key in ("facades", "windows", "doors", "socles"):
        current_count = len(result.get(key, []))
        baseline_count = len(baseline.get(key, []))
        if current_count != baseline_count:
            changed = True
            messages.append(f"{key} count changed: {baseline_count} -> {current_count}")

    for key in ("facades", "socles"):
        cur_box = result.get(key, [None])[0] if result.get(key) else None
        base_box = baseline.get(key, [None])[0] if baseline.get(key) else None
        delta = _bbox_delta(cur_box, base_box)
        if delta > 2:
            changed = True
            messages.append(f"{key} bbox delta={delta}px")

    for key in ("facade_contour", "socle_contour", "socle_profile_line"):
        delta = _points_delta(
            result.get("shape_signatures", {}).get(key, []),
            baseline.get("shape_signatures", {}).get(key, []),
        )
        if delta > 4:
            changed = True
            messages.append(f"{key} changed (delta={delta})")

    cur_windows = _sort_boxes(result.get("windows", []))
    base_windows = _sort_boxes(baseline.get("windows", []))
    if len(cur_windows) == len(base_windows):
        win_delta = max((_bbox_delta(c, b) for c, b in zip(cur_windows, base_windows)), default=0)
        if win_delta > 2:
            changed = True
            messages.append(f"window geometry delta={win_delta}px")

    cur_doors = _sort_boxes(result.get("doors", []))
    base_doors = _sort_boxes(baseline.get("doors", []))
    if len(cur_doors) == len(base_doors):
        door_delta = max((_bbox_delta(c, b) for c, b in zip(cur_doors, base_doors)), default=0)
        if door_delta > 2:
            changed = True
            messages.append(f"door geometry delta={door_delta}px")

    if not changed:
        messages.append("geometry unchanged vs baseline")
    return {"changed": changed, "messages": messages}


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    raw_path = ROOT / case["raw_path"]
    image = load_image(raw_path)

    world = read_worldfile_px_per_m(str(raw_path))
    px_per_m = world["px_per_m"] if world else None

    pipeline = AnalysisPipeline()
    pipeline.ocr_engine.extract_all_text = lambda _image: []
    pipeline.ocr_engine.enrich_regions = lambda _image, _regions: None
    pipeline.run(image, manual_linear_scale_px_per_m=px_per_m)

    det = pipeline.detection_results or {}
    facade_regions = [r for r in det.get("facades", []) if r.bbox]
    window_regions = [r for r in det.get("windows", []) if r.bbox]
    door_regions = [r for r in det.get("doors", []) if r.bbox]
    socle_regions = [r for r in det.get("socles", []) if r.bbox]
    socle_profile_regions = [r for r in det.get("socle_profiles", []) if r.bbox]

    facades = [_bbox_to_dict(r.bbox) for r in facade_regions]
    windows = [_bbox_to_dict(r.bbox) for r in window_regions]
    doors = [_bbox_to_dict(r.bbox) for r in door_regions]
    socles = [_bbox_to_dict(r.bbox) for r in socle_regions]
    socle_profiles = [_bbox_to_dict(r.bbox) for r in socle_profile_regions]
    scene_types = sorted({f.scene_type for f in pipeline.parsed_facades if getattr(f, "scene_type", "")})

    facade_contour = _contour_to_points(facade_regions[0].contour, closed=True) if facade_regions else []
    socle_contour = _contour_to_points(socle_regions[0].contour, closed=True) if socle_regions else []
    socle_profile_line = _contour_to_points(socle_profile_regions[0].contour, closed=False) if socle_profile_regions else []

    result = {
        "id": case["id"],
        "owner": case["owner"],
        "scene_class": case["scene_class"],
        "raw_path": case["raw_path"],
        "pgw_px_per_m": round(float(px_per_m), 3) if px_per_m else None,
        "linear_scale_source": pipeline.linear_scale_source or "none",
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "scene_types": scene_types,
        "facades": facades,
        "windows": windows,
        "doors": doors,
        "socles": socles,
        "socle_profiles": socle_profiles,
        "shape_signatures": {
            "facade_contour": facade_contour,
            "socle_contour": socle_contour,
            "socle_profile_line": socle_profile_line,
        },
        "geometry": _geometry_metrics(
            facades[0] if facades else None,
            socles[0] if socles else None,
            socle_profile_line,
            doors,
        ),
        "reference_metrics": {},
        "reference_validation": {},
        "warnings": list(pipeline.warnings),
        "focus": case.get("focus", []),
        "checks": [],
    }

    expected = case.get("expected", {})
    actual_counts = {
        "facades": len(facades),
        "windows": len(windows),
        "doors": len(doors),
        "socles": len(socles),
    }
    for key, value in actual_counts.items():
        result["checks"].extend(_count_check(value, expected, key))

    if "scene_type" in expected and expected["scene_type"] not in scene_types:
        result["checks"].append(
            f"scene_type: expected {expected['scene_type']}, got {scene_types or ['unknown']}"
        )

    result["reference_metrics"], result["reference_validation"] = _reference_shape_metrics(result, case, image)
    result["checks"].extend(_geometry_warnings(result))
    return result


def _load_baseline(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {case["id"]: case for case in payload.get("cases", [])}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run canonical facade regression cases.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs" / "reference_cases" / "canonical_scenarios.json")
    parser.add_argument("--output-json", type=Path, default=ROOT / "docs" / "reference_cases" / "canonical_regression_report.json")
    parser.add_argument("--case-id", type=str, default="")
    parser.add_argument("--stdout-json", action="store_true")
    parser.add_argument("--baseline-json", type=Path, default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    cases = manifest.get("scenarios", [])
    if args.baseline_json is not None:
        args.baseline_json = args.baseline_json.resolve()
    baseline_map = _load_baseline(args.baseline_json)

    if args.case_id:
        case = next((item for item in cases if item.get("id") == args.case_id), None)
        if case is None:
            raise SystemExit(f"Unknown case id: {args.case_id}")
        result = run_case(case)
        result["delta_vs_baseline"] = _compare_to_baseline(result, baseline_map.get(result["id"]))
        if args.stdout_json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            print(f"{result['id']}: scenes={result['scene_types']} facades={len(result['facades'])} windows={len(result['windows'])} doors={len(result['doors'])} socles={len(result['socles'])}")
            for check in result["checks"]:
                print(f"  ! {check}")
            for msg in result["delta_vs_baseline"]["messages"]:
                print(f"  ~ {msg}")
        return 0

    results = []
    script_path = Path(__file__).resolve()
    for case in cases:
        cmd = [sys.executable, str(script_path), "--manifest", str(args.manifest), "--case-id", str(case.get("id", "")), "--stdout-json"]
        if args.baseline_json is not None:
            cmd.extend(["--baseline-json", str(args.baseline_json)])
        completed = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True)
        payload = completed.stdout.strip()
        if not payload:
            raise RuntimeError(f"No JSON output for case {case.get('id')}")
        results.append(json.loads(payload))

    report = {
        "generated_at": manifest.get("generated_at"),
        "manifest": str(args.manifest.relative_to(ROOT)),
        "baseline": str(args.baseline_json.relative_to(ROOT)) if args.baseline_json else None,
        "cases": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Canonical regression")
    for item in results:
        print(f"- {item['id']}: scenes={item['scene_types']} facades={len(item['facades'])} windows={len(item['windows'])} doors={len(item['doors'])} socles={len(item['socles'])}")
        for check in item["checks"]:
            print(f"  ! {check}")
        for msg in item.get("delta_vs_baseline", {}).get("messages", []):
            print(f"  ~ {msg}")
    print(f"report: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
