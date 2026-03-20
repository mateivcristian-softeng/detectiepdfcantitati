"""
Feature-based validator for photo window candidates.
No training; uses geometric and intensity heuristics to reject roof/oversized FPs.
"""

import cv2
import numpy as np
from typing import List, Optional

from core.color_detector import DetectedRegion


def _is_photo_window_candidate(region: DetectedRegion) -> bool:
    """True if region comes from photo detection path."""
    src = (region.color_detected or "").lower()
    return "photo" in src and "window" in src


def validate(
    regions: List[DetectedRegion],
    image: np.ndarray,
    facades: Optional[List[DetectedRegion]] = None,
) -> List[DetectedRegion]:
    """
    Filter photo window candidates using feature-based heuristics.
    Non-photo candidates pass through unchanged.
    Returns filtered list.
    """
    if not regions:
        return []

    h_img, w_img = image.shape[:2]
    img_area = float(h_img * w_img)

    facade_boxes = []
    if facades:
        for f in facades:
            if f.region_type == "facade" and f.bbox:
                facade_boxes.append(f.bbox)

    validated = []
    for r in regions:
        if not _is_photo_window_candidate(r):
            validated.append(r)
            continue

        x, y, w, h = r.bbox
        if w <= 0 or h <= 0:
            validated.append(r)
            continue

        area_px = float(w * h)
        ratio = w / h if h else 0

        # Roof/strip: very wide and thin
        if ratio > 4.5 or ratio < 0.22:
            continue

        # Oversized vs image
        if area_px > img_area * 0.25:
            continue

        # Oversized vs parent facade
        parent = None
        for fb in facade_boxes:
            fx, fy, fw, fh = fb
            cx, cy = r.center
            if fx <= cx <= fx + fw and fy <= cy <= fy + fh:
                parent = fb
                break

        if parent:
            fx, fy, fw, fh = parent
            facade_area = float(fw * fh)
            if facade_area > 0 and area_px > facade_area * 0.35:
                continue
            # Roof band: top 15% of facade, full width
            if h < fh * 0.15 and w > fw * 0.6 and y < fy + int(fh * 0.20):
                continue
            # Bottom band: candidates touching facade base are usually socle/ground noise.
            local_bottom_ratio = (y - fy + h) / max(1.0, float(fh))
            if local_bottom_ratio > 0.92:
                continue

        # Edge consistency: windows often have darker interior
        roi = image[
            max(0, y) : min(h_img, y + h),
            max(0, x) : min(w_img, x + w),
        ]
        if roi.size < 100:
            validated.append(r)
            continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(gray))
        # Very bright uniform = likely sky/reflection, not window
        if mean_val > 200 and np.std(gray) < 35:
            continue

        validated.append(r)

    return validated
