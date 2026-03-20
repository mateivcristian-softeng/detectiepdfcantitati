from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def polygon_norm_bounds(label: str, targets: dict[str, Any] | None = None) -> dict[str, float]:
    bounds = {
        "x_min": 0.0,
        "x_max": 1.0,
        "y_min": 0.0,
        "y_max": 1.0,
    }
    if label != "socle_contour_norm_in_facade_bbox" or not targets:
        return bounds

    facade_bbox = targets.get("facade_bbox")
    socle_bbox = targets.get("socle_bbox")
    if not facade_bbox or not socle_bbox or len(facade_bbox) != 4 or len(socle_bbox) != 4:
        return bounds

    fx, fy, fw, fh = [float(v) for v in facade_bbox]
    sx, sy, sw, sh = [float(v) for v in socle_bbox]
    if fw <= 0.0 or fh <= 0.0:
        return bounds

    pad = 0.02
    bounds["x_min"] = min(0.0, (sx - fx) / fw - pad)
    bounds["x_max"] = max(1.0, (sx + sw - fx) / fw + pad)
    bounds["y_max"] = max(1.0, (sy + sh - fy) / fh + pad)
    return bounds


def expand_norm_bounds(bounds: dict[str, float] | None,
                       points: Sequence[Sequence[float]] | None,
                       pad: float = 0.0) -> dict[str, float]:
    merged = dict(bounds or {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0})
    if not points:
        return merged
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    merged["x_min"] = min(float(merged["x_min"]), min(xs) - pad)
    merged["x_max"] = max(float(merged["x_max"]), max(xs) + pad)
    merged["y_min"] = min(float(merged["y_min"]), min(ys) - pad)
    merged["y_max"] = max(float(merged["y_max"]), max(ys) + pad)
    if merged["x_max"] <= merged["x_min"]:
        merged["x_max"] = merged["x_min"] + 1.0
    if merged["y_max"] <= merged["y_min"]:
        merged["y_max"] = merged["y_min"] + 1.0
    return merged


def _map_norm_point(px: float, py: float, bounds: dict[str, float], size: int) -> tuple[float, float]:
    span_x = max(1e-6, float(bounds["x_max"]) - float(bounds["x_min"]))
    span_y = max(1e-6, float(bounds["y_max"]) - float(bounds["y_min"]))
    nx = (float(px) - float(bounds["x_min"])) / span_x
    ny = (float(py) - float(bounds["y_min"])) / span_y
    return nx * float(size - 1), ny * float(size - 1)


def _prepare_band_polygon(points: Sequence[Sequence[float]], x_tol: float = 0.015) -> list[list[float]]:
    if len(points) < 4:
        return [[float(px), float(py)] for px, py in points]

    pts = sorted([[float(px), float(py)] for px, py in points], key=lambda p: (p[0], p[1]))
    clusters: list[list[list[float]]] = []
    for px, py in pts:
        if clusters and abs(px - clusters[-1][-1][0]) <= x_tol:
            clusters[-1].append([px, py])
        else:
            clusters.append([[px, py]])
    if len(clusters) < 2:
        return [[float(px), float(py)] for px, py in points]

    upper: list[list[float]] = []
    lower: list[list[float]] = []
    for cluster in clusters:
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        x_mid = float(np.mean(np.array(xs, dtype=np.float32)))
        upper.append([x_mid, float(min(ys))])
        lower.append([x_mid, float(max(ys))])

    polygon = upper + list(reversed(lower))
    cleaned: list[list[float]] = []
    for px, py in polygon:
        if cleaned and abs(cleaned[-1][0] - px) <= 1e-6 and abs(cleaned[-1][1] - py) <= 1e-6:
            continue
        cleaned.append([px, py])
    if len(cleaned) >= 2 and abs(cleaned[0][0] - cleaned[-1][0]) <= 1e-6 and abs(cleaned[0][1] - cleaned[-1][1]) <= 1e-6:
        cleaned.pop()
    return cleaned if len(cleaned) >= 3 else [[float(px), float(py)] for px, py in points]


def load_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot load image: {path}")
    return image


def rasterize_norm_polygon(points: list[list[float]],
                           size: int = 512,
                           bounds: dict[str, float] | None = None,
                           band_like: bool = False) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    if len(points) < 3:
        return mask
    prepared = _prepare_band_polygon(points) if band_like else [[float(px), float(py)] for px, py in points]
    active_bounds = expand_norm_bounds(bounds, prepared, pad=0.0)
    arr = []
    for px, py in prepared:
        x_f, y_f = _map_norm_point(float(px), float(py), active_bounds, size)
        x = int(round(max(0.0, min(float(size - 1), x_f))))
        y = int(round(max(0.0, min(float(size - 1), y_f))))
        arr.append([x, y])
    contour = np.array(arr, dtype=np.int32).reshape((-1, 1, 2))
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask


def check_norm_bounds(points: list[list[float]],
                      label: str,
                      bounds: dict[str, float] | None = None,
                      size: int = 512) -> list[dict[str, Any]]:
    issues = []
    active_bounds = dict(bounds or {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0})
    tol = 1e-6
    for i, (px, py) in enumerate(points):
        oob = []
        if px < float(active_bounds["x_min"]) - tol or px > float(active_bounds["x_max"]) + tol:
            oob.append(f"x={px:.6f}")
        if py < float(active_bounds["y_min"]) - tol or py > float(active_bounds["y_max"]) + tol:
            oob.append(f"y={py:.6f}")
        if oob:
            x_f, y_f = _map_norm_point(float(px), float(py), active_bounds, size)
            x_px = int(round(x_f))
            y_px = int(round(y_f))
            issues.append({
                "polygon": label,
                "point_idx": i,
                "norm_x": round(px, 6),
                "norm_y": round(py, 6),
                "pixel_x": x_px,
                "pixel_y": y_px,
                "outside_canvas": x_px < 0 or x_px > (size - 1) or y_px < 0 or y_px > (size - 1),
                "oob_coords": ", ".join(oob),
            })
    return issues


def check_raster_area(points: list[list[float]],
                      label: str,
                      size: int = 512,
                      bounds: dict[str, float] | None = None) -> dict[str, Any]:
    mask = rasterize_norm_polygon(
        points,
        size,
        bounds=bounds,
        band_like=(label == "socle_contour_norm_in_facade_bbox"),
    )
    area = int(np.count_nonzero(mask))
    max_area = size * size
    return {
        "polygon": label,
        "num_points": len(points),
        "raster_area_px": area,
        "canvas_size": size,
        "area_fraction": round(area / max_area, 6) if max_area > 0 else 0,
        "degenerate": area < 1000,
    }


def polygon_iou_norm(a: list[list[float]],
                     b: list[list[float]],
                     size: int = 512,
                     bounds: dict[str, float] | None = None,
                     band_like: bool = False) -> float | None:
    if len(a) < 3 or len(b) < 3:
        return None
    a_prepared = _prepare_band_polygon(a) if band_like else [[float(px), float(py)] for px, py in a]
    b_prepared = _prepare_band_polygon(b) if band_like else [[float(px), float(py)] for px, py in b]
    active_bounds = expand_norm_bounds(bounds, a_prepared, pad=0.0)
    active_bounds = expand_norm_bounds(active_bounds, b_prepared, pad=0.0)
    ma = rasterize_norm_polygon(a_prepared, size=size, bounds=active_bounds, band_like=False)
    mb = rasterize_norm_polygon(b_prepared, size=size, bounds=active_bounds, band_like=False)
    inter = np.logical_and(ma > 0, mb > 0).sum()
    union = np.logical_or(ma > 0, mb > 0).sum()
    if union <= 0:
        return None
    return float(inter) / float(union)


def image_size_check(ref_json: dict[str, Any], raw_image: np.ndarray | None) -> dict[str, Any] | None:
    if raw_image is None:
        return None
    ref_size = ref_json.get("image_size")
    if not ref_size:
        return None
    raw_h, raw_w = raw_image.shape[:2]
    ref_w, ref_h = ref_size
    match = (raw_w == ref_w and raw_h == ref_h)
    return {
        "reference_image_size": ref_size,
        "raw_image_size": [raw_w, raw_h],
        "match": match,
        "width_ratio": round(raw_w / max(1, ref_w), 4),
        "height_ratio": round(raw_h / max(1, ref_h), 4),
    }


def scale_ref_box_to_raw(box: Sequence[int], ref_size: Sequence[int], raw_size: Sequence[int]) -> list[int]:
    ref_w = max(1.0, float(ref_size[0]))
    ref_h = max(1.0, float(ref_size[1]))
    raw_w = max(1.0, float(raw_size[0]))
    raw_h = max(1.0, float(raw_size[1]))
    x, y, w, h = [float(v) for v in box]
    sx = int(round(x * raw_w / ref_w))
    sy = int(round(y * raw_h / ref_h))
    sw = max(1, int(round(w * raw_w / ref_w)))
    sh = max(1, int(round(h * raw_h / ref_h)))
    return [sx, sy, sw, sh]


def analyze_opening_box(gray: np.ndarray, bbox: Sequence[int]) -> dict[str, Any]:
    raw_h, raw_w = gray.shape[:2]
    x, y, w, h = [int(v) for v in bbox]
    x0 = max(0, min(raw_w - 1, x))
    y0 = max(0, min(raw_h - 1, y))
    x1 = max(x0 + 1, min(raw_w, x + max(1, w)))
    y1 = max(y0 + 1, min(raw_h, y + max(1, h)))
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return {
            "scaled_box": [x0, y0, max(1, x1 - x0), max(1, y1 - y0)],
            "mean_gray": 0.0,
            "dark_frac": 0.0,
            "std_gray": 0.0,
            "edge_density": 0.0,
            "jamb_score": 0.0,
            "context_contrast": 0.0,
            "support_score": 0.0,
            "likely_opening": False,
            "bright_wall": False,
        }

    mean_gray = float(np.mean(crop))
    dark_frac = float(np.count_nonzero(crop < 120)) / float(crop.size)
    std_gray = float(np.std(crop))

    grad_x = np.abs(cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3))
    grad_y = np.abs(cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3))
    edge = grad_x + grad_y * 0.45

    edge_mean = float(np.mean(edge)) if edge.size else 0.0
    edge_std = float(np.std(edge)) if edge.size else 0.0
    strong_edge_thr = max(18.0, edge_mean + edge_std * 1.20)
    edge_density = float(np.count_nonzero(edge >= strong_edge_thr)) / float(edge.size) if edge.size else 0.0

    vertical_mean = float(np.mean(grad_x)) if grad_x.size else 0.0
    vertical_std = float(np.std(grad_x)) if grad_x.size else 0.0
    strong_vertical_thr = max(14.0, vertical_mean + vertical_std * 1.15)
    band = max(1, min(crop.shape[1] // 4, int(round(crop.shape[1] * 0.18))))
    left_band = grad_x[:, :band] >= strong_vertical_thr
    right_band = grad_x[:, -band:] >= strong_vertical_thr
    jamb_score = float(left_band.mean() + right_band.mean()) * 0.5 if band > 0 else 0.0

    pad_x = max(4, int(round((x1 - x0) * 0.25)))
    pad_y = max(4, int(round((y1 - y0) * 0.25)))
    rx0 = max(0, x0 - pad_x)
    ry0 = max(0, y0 - pad_y)
    rx1 = min(raw_w, x1 + pad_x)
    ry1 = min(raw_h, y1 + pad_y)
    ring = gray[ry0:ry1, rx0:rx1]
    ring_mask = np.ones(ring.shape, dtype=bool)
    ring_mask[(y0 - ry0):(y1 - ry0), (x0 - rx0):(x1 - rx0)] = False
    ring_pixels = ring[ring_mask]
    context_mean = float(ring_pixels.mean()) if ring_pixels.size else mean_gray
    context_contrast = float(abs(context_mean - mean_gray))

    tone_opening = dark_frac > 0.15 or mean_gray < 140.0
    structured_opening = std_gray >= 18.0 and edge_density >= 0.10 and jamb_score >= 0.05
    contrast_opening = context_contrast >= 10.0 and std_gray >= 14.0 and (edge_density >= 0.08 or jamb_score >= 0.08)
    support_score = (
        min(1.0, dark_frac / 0.18) * 0.45 +
        min(1.0, max(0.0, 145.0 - mean_gray) / 45.0) * 0.15 +
        min(1.0, std_gray / 26.0) * 0.12 +
        min(1.0, edge_density / 0.14) * 0.12 +
        min(1.0, jamb_score / 0.10) * 0.10 +
        min(1.0, context_contrast / 18.0) * 0.06
    )
    likely_opening = tone_opening or structured_opening or contrast_opening or support_score >= 0.52
    bright_wall = (
        mean_gray > 200.0
        and dark_frac < 0.02
        and std_gray < 14.0
        and edge_density < 0.08
        and jamb_score < 0.05
        and context_contrast < 10.0
        and support_score < 0.45
    )

    return {
        "scaled_box": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
        "mean_gray": round(mean_gray, 1),
        "dark_frac": round(dark_frac, 4),
        "std_gray": round(std_gray, 1),
        "edge_density": round(edge_density, 4),
        "jamb_score": round(jamb_score, 4),
        "context_contrast": round(context_contrast, 1),
        "support_score": round(float(support_score), 4),
        "likely_opening": bool(likely_opening),
        "bright_wall": bool(bright_wall and not likely_opening),
    }


def validate_reference_dict(ref: dict[str, Any], raw_image: np.ndarray, path_label: str = "") -> dict[str, Any]:
    targets = ref.get("targets", {})
    ref_id = ref.get("reference_id", "reference")

    report: dict[str, Any] = {
        "reference_id": ref_id,
        "path": path_label,
        "issues": [],
        "bounds_violations": [],
        "raster_checks": [],
        "image_size_check": image_size_check(ref, raw_image),
        "box_content_checks": [],
    }

    for key in ("facade_contour_norm_in_bbox", "socle_contour_norm_in_facade_bbox"):
        pts = targets.get(key, [])
        if not pts:
            continue
        bounds = polygon_norm_bounds(key, targets)
        report["bounds_violations"].extend(check_norm_bounds(pts, key, bounds=bounds))
        report["raster_checks"].append(check_raster_area(pts, key, bounds=bounds))

    ref_size = ref.get("image_size")
    gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    raw_h, raw_w = gray.shape[:2]
    if ref_size:
        for box_key, box_type in (("window_boxes", "window"), ("door_boxes", "door")):
            for idx, box in enumerate(targets.get(box_key, [])):
                if isinstance(box, list):
                    ref_box = [int(v) for v in box]
                else:
                    ref_box = [int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])]
                scaled = scale_ref_box_to_raw(ref_box, ref_size, (raw_w, raw_h))
                metrics = analyze_opening_box(gray, scaled)
                metrics.update({
                    "box_type": box_type,
                    "box_idx": idx,
                    "ref_box": ref_box,
                })
                report["box_content_checks"].append(metrics)

    n_oob = len(report["bounds_violations"])
    n_outside_canvas = sum(1 for v in report["bounds_violations"] if v["outside_canvas"])
    n_degenerate = sum(1 for r in report["raster_checks"] if r["degenerate"])
    n_bright_wall = sum(1 for c in report["box_content_checks"] if c.get("bright_wall"))
    size_mismatch = report["image_size_check"] and not report["image_size_check"].get("match", True)

    if n_oob > 0:
        report["issues"].append(f"{n_oob} normalized point(s) outside expected envelope")
    if n_outside_canvas > 0:
        report["issues"].append(f"{n_outside_canvas} point(s) outside 512x512 canvas after rasterization")
    if n_degenerate > 0:
        report["issues"].append(f"{n_degenerate} polygon(s) with degenerate raster area (<1000px on 512x512)")
    if n_bright_wall > 0:
        report["issues"].append(
            f"{n_bright_wall} opening box(es) fall on bright wall in raw image (low structure support)"
        )
    if size_mismatch:
        report["issues"].append("reference image_size does not match raw PNG dimensions")

    report["geometry_valid"] = (n_degenerate == 0)
    report["openings_valid"] = (n_bright_wall == 0)
    report["verdict"] = "INVALID" if (n_degenerate > 0 or n_bright_wall > 0) else (
        "WARNING" if (n_oob > 0 or size_mismatch) else "VALID"
    )
    return report


def validate_reference_file(ref_path: Path, raw_path: Path | None) -> dict[str, Any]:
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    raw_image = load_image(raw_path) if raw_path and raw_path.exists() else None
    if raw_image is None:
        raise FileNotFoundError(f"Cannot load raw image for reference validation: {raw_path}")
    try:
        rel_label = str(ref_path.relative_to(ROOT))
    except ValueError:
        rel_label = str(ref_path)
    return validate_reference_dict(ref, raw_image, rel_label)
