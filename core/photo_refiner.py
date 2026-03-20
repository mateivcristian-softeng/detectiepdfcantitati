"""
Optional ROI refinement for photo window candidates.
Classic methods: GrabCut or heuristics. Returns refined region or None to keep original.
"""

import cv2
import numpy as np
from typing import Optional

from core.color_detector import DetectedRegion


def _is_photo_window_candidate(region: DetectedRegion) -> bool:
    src = (region.color_detected or "").lower()
    return "photo" in src and "window" in src


def refine(
    image: np.ndarray,
    region: DetectedRegion,
    method: str = "grabcut",
) -> Optional[DetectedRegion]:
    """
    Optionally refine window ROI. Returns new DetectedRegion or None to keep original.
    method: "grabcut" or "heuristic"
    """
    if not _is_photo_window_candidate(region):
        return None

    x, y, w, h = region.bbox
    h_img, w_img = image.shape[:2]

    # Pad to avoid border issues
    pad = max(4, min(w, h) // 8)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w_img, x + w + pad)
    y1 = min(h_img, y + h + pad)
    roi = image[y0:y1, x0:x1].copy()

    if roi.size < 200:
        return None

    refined_bbox = None
    if method == "grabcut":
        refined_bbox = _refine_grabcut(roi, pad, pad + w, pad, pad + h)
    elif method == "heuristic":
        refined_bbox = _refine_heuristic(roi)

    if refined_bbox is None:
        return None

    rx, ry, rw, rh = refined_bbox
    if rw < 10 or rh < 10:
        return None

    # Map back to image coords
    new_x = x0 + rx
    new_y = y0 + ry
    new_area = float(rw * rh)
    if new_area < 50:
        return None

    return DetectedRegion(
        label=region.label,
        region_type=region.region_type,
        bbox=(new_x, new_y, rw, rh),
        contour=region.contour,
        area_px=new_area,
        area_m2=region.area_m2,
        width_m=region.width_m,
        height_m=region.height_m,
        ocr_text=region.ocr_text,
        parent_facade=region.parent_facade,
        color_detected=region.color_detected,
    )


def _refine_grabcut(roi: np.ndarray, x: int, x2: int, y: int, y2: int) -> Optional[tuple]:
    """GrabCut with rect init. Returns (x,y,w,h) in roi coords or None."""
    rh, rw = roi.shape[:2]
    rect = (max(0, x), max(0, y), max(2, min(rw - x, x2 - x)), max(2, min(rh - y, y2 - y)))
    if rect[2] < 5 or rect[3] < 5:
        return None

    bgd = np.zeros((1, 65), dtype=np.float64)
    fgd = np.zeros((1, 65), dtype=np.float64)
    try:
        mask, _, _ = cv2.grabCut(roi, np.zeros(roi.shape[:2], dtype=np.uint8), rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None

    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 30:
        return None

    bx, by, bw, bh = cv2.boundingRect(c)
    return (bx, by, bw, bh)


def _refine_heuristic(roi: np.ndarray) -> Optional[tuple]:
    """Tighten bbox to dark foreground. Returns (x,y,w,h) in roi coords."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = np.percentile(gray, 40)
    binary = (gray <= thresh).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    coords = np.argwhere(binary > 0)
    if coords.size < 50:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    w = int(x_max - x_min + 1)
    h = int(y_max - y_min + 1)
    if w < 10 or h < 10:
        return None
    return (int(x_min), int(y_min), w, h)
