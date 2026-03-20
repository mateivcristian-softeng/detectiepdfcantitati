from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

CASES = [
    {
        "id": "MARGINEAN_DREAPTA",
        "processed_reference_path": ROOT / "detector cantitati" / "procesate" / "2 imagini pe fundal alb 2 imagini pe fundal negru" / "MARGINEAN ANA_colorized_fatada_dreapta_procesat.png",
        "output_json_path": ROOT / "docs" / "reference_cases" / "marginean_dreapta_expected.json",
        "scene": "gable_facade",
    },
    {
        "id": "MUNTEAN_STANGA",
        "processed_reference_path": ROOT / "detector cantitati" / "procesate" / "2 imagini pe fundal alb 2 imagini pe fundal negru" / "MUNTEAN LUCRETIA_colorized_fatada_stanga_procesat.png",
        "output_json_path": ROOT / "docs" / "reference_cases" / "muntean_stanga_expected.json",
        "scene": "sparse_openings_flat",
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


def largest_blue_contour(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 0] >= 100) & (hsv[:, :, 0] <= 130) & (hsv[:, :, 1] >= 120) & (hsv[:, :, 2] >= 100)).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("No blue contour found")
    c = max(cnts, key=cv2.contourArea)
    eps = max(2.0, 0.003 * cv2.arcLength(c, True))
    return cv2.approxPolyDP(c, eps, True)


def select_socle_contour(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 0] >= 135) & (hsv[:, :, 0] <= 175) & (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 80)).astype(np.uint8) * 255
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
        if bottom_ratio < 0.82:
            continue
        score = (bottom_ratio * 3.0) + (width_ratio * 2.0) + min(area / max(1.0, w * h), 0.1)
        if best is None or score > best_score:
            best = c
            best_score = score
    if best is None:
        best = max(cnts, key=cv2.contourArea)
    eps = max(2.0, 0.003 * cv2.arcLength(best, True))
    return cv2.approxPolyDP(best, eps, True)


def main() -> int:
    for case in CASES:
        img = load_image(case["processed_reference_path"])
        facade = largest_blue_contour(img)
        socle = select_socle_contour(img)
        facade_pts = facade.reshape(-1, 2).astype(int).tolist()
        socle_pts = socle.reshape(-1, 2).astype(int).tolist()
        facade_bbox = list(map(int, cv2.boundingRect(facade)))
        socle_bbox = list(map(int, cv2.boundingRect(socle)))
        payload = {
            "reference_id": case["id"].lower() + "_processed_reference",
            "scene": case["scene"],
            "processed_reference_path": str(case["processed_reference_path"].relative_to(ROOT)),
            "image_size": [int(img.shape[1]), int(img.shape[0])],
            "targets": {
                "facade_bbox": facade_bbox,
                "socle_bbox": socle_bbox,
                "facade_contour": facade_pts,
                "socle_contour": socle_pts,
                "facade_contour_norm_in_bbox": normalize_points(facade_pts, facade_bbox),
                "socle_contour_norm_in_facade_bbox": normalize_points(socle_pts, facade_bbox),
            },
        }
        case["output_json_path"].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(case["output_json_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
