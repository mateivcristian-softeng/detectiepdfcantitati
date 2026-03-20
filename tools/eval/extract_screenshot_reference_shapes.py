from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

CASES = [
    {
        "id": "ARION",
        "scene": "composite_stepped_facade",
        "source_image_path": Path(r"C:\Users\Admin\Pictures\Screenshots\Screenshot 2026-03-11 185304.png"),
        "copy_reference_path": ROOT / "assets" / "reference" / "arion_expected_reference_chat_2026-03-11.png",
        "output_json_path": ROOT / "docs" / "reference_cases" / "arion_expected.json",
    },
]


def load_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot load image: {path}")
    return img


def normalize_points(points: list[list[int]], ref_bbox: list[int]) -> list[list[float]]:
    x, y, w, h = ref_bbox
    w = max(1, int(w))
    h = max(1, int(h))
    out = []
    for px, py in points:
        out.append([
            round((float(px) - float(x)) / float(w), 6),
            round((float(py) - float(y)) / float(h), 6),
        ])
    return out


def normalize_boxes(boxes: list[list[int]], ref_bbox: list[int]) -> list[dict[str, float]]:
    x, y, w, h = ref_bbox
    w = max(1.0, float(w))
    h = max(1.0, float(h))
    out = []
    for bx, by, bw, bh in boxes:
        out.append({
            "x": round((float(bx) - float(x)) / w, 6),
            "y": round((float(by) - float(y)) / h, 6),
            "w": round(float(bw) / w, 6),
            "h": round(float(bh) / h, 6),
        })
    return out


def largest_blue_contour(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = (((hsv[:, :, 0] >= 100) & (hsv[:, :, 0] <= 130) & (hsv[:, :, 1] >= 50) & (hsv[:, :, 2] >= 100)).astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("No blue contour found")
    c = max(cnts, key=cv2.contourArea)
    eps = max(2.0, 0.003 * cv2.arcLength(c, True))
    return cv2.approxPolyDP(c, eps, True)


def select_socle_contour(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = (((hsv[:, :, 0] >= 135) & (hsv[:, :, 0] <= 175) & (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] >= 100)).astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("No magenta contour found")
    h, w = mask.shape[:2]
    best = None
    best_score = None
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area <= 0:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        bottom_ratio = float(y + bh) / float(h)
        width_ratio = float(bw) / float(w)
        if bottom_ratio < 0.75 or width_ratio < 0.25:
            continue
        score = (bottom_ratio * 3.0) + (width_ratio * 2.0) + min(area / max(1.0, w * h), 0.1)
        if best is None or score > best_score:
            best = c
            best_score = score
    if best is None:
        best = max(cnts, key=cv2.contourArea)
    eps = max(2.0, 0.003 * cv2.arcLength(best, True))
    return cv2.approxPolyDP(best, eps, True)


def _extract_components(mask: np.ndarray) -> list[list[int]]:
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes: list[list[int]] = []
    for i in range(1, num):
        x, y, w, h, area = map(int, stats[i])
        if area <= 0:
            continue
        boxes.append([x, y, w, h])
    return boxes


def _cluster_boxes_by_x(boxes: list[list[int]], max_gap: int = 24) -> list[list[int]]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
    clusters: list[list[list[int]]] = []
    for box in boxes:
        x, y, w, h = box
        x1 = x
        x2 = x + w
        matched = False
        for cluster in clusters:
            cx1 = min(item[0] for item in cluster)
            cx2 = max(item[0] + item[2] for item in cluster)
            if x1 <= cx2 + max_gap and x2 >= cx1 - max_gap:
                cluster.append(box)
                matched = True
                break
        if not matched:
            clusters.append([box])
    out: list[list[int]] = []
    for cluster in clusters:
        x1 = min(item[0] for item in cluster)
        y1 = min(item[1] for item in cluster)
        x2 = max(item[0] + item[2] for item in cluster)
        y2 = max(item[1] + item[3] for item in cluster)
        out.append([x1, y1, x2 - x1, y2 - y1])
    return sorted(out, key=lambda b: (b[0], b[1]))


def extract_window_boxes(img: np.ndarray) -> list[list[int]]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = (((hsv[:, :, 0] >= 130) & (hsv[:, :, 0] <= 155) & (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 100)).astype(np.uint8) * 255)
    h, w = mask.shape[:2]
    boxes = []
    for x, y, bw, bh in _extract_components(mask):
        if bw >= int(w * 0.4) or bh >= int(h * 0.4):
            continue
        if bw < 35 or bh < 90:
            continue
        if y < int(h * 0.35) or y > int(h * 0.82):
            continue
        boxes.append([x, y, bw, bh])
    return sorted(boxes, key=lambda b: (b[0], b[1]))


def extract_door_boxes(img: np.ndarray) -> list[list[int]]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = (((hsv[:, :, 0] >= 5) & (hsv[:, :, 0] <= 25) & (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] >= 80)).astype(np.uint8) * 255)
    h, w = mask.shape[:2]
    parts = []
    for x, y, bw, bh in _extract_components(mask):
        if x > int(w * 0.92) or y > int(h * 0.92):
            continue
        if y < int(h * 0.45):
            continue
        if bw < 2 and bh < 10:
            continue
        parts.append([x, y, bw, bh])
    merged = _cluster_boxes_by_x(parts, max_gap=28)
    boxes = []
    for x, y, bw, bh in merged:
        if bw < 80 or bh < 150:
            continue
        boxes.append([x, y, bw, bh])
    return sorted(boxes, key=lambda b: (b[0], b[1]))


def main() -> int:
    for case in CASES:
        copy_path = case["copy_reference_path"]
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(case["source_image_path"], copy_path)

        img = load_image(copy_path)
        facade = largest_blue_contour(img)
        socle = select_socle_contour(img)
        windows = extract_window_boxes(img)
        doors = extract_door_boxes(img)

        facade_pts = facade.reshape(-1, 2).astype(int).tolist()
        socle_pts = socle.reshape(-1, 2).astype(int).tolist()
        facade_bbox = list(map(int, cv2.boundingRect(facade)))
        socle_bbox = list(map(int, cv2.boundingRect(socle)))

        payload = {
            "reference_id": case["id"].lower() + "_expected_reference_chat_2026-03-11",
            "scene": case["scene"],
            "reference_image_path": str(copy_path.relative_to(ROOT)),
            "image_size": [int(img.shape[1]), int(img.shape[0])],
            "targets": {
                "facade_bbox": facade_bbox,
                "socle_bbox": socle_bbox,
                "facade_contour": facade_pts,
                "socle_contour": socle_pts,
                "facade_contour_norm_in_bbox": normalize_points(facade_pts, facade_bbox),
                "socle_contour_norm_in_facade_bbox": normalize_points(socle_pts, facade_bbox),
                "window_boxes": windows,
                "door_boxes": doors,
                "window_boxes_norm_in_facade_bbox": normalize_boxes(windows, facade_bbox),
                "door_boxes_norm_in_facade_bbox": normalize_boxes(doors, facade_bbox),
            },
        }
        case["output_json_path"].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(case["output_json_path"])
        print("windows=", windows)
        print("doors=", doors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
