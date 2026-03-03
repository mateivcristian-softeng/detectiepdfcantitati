"""
DrawQuantPDF - Color-based Region Detector
Specialized for detecting yellow window rectangles and orange/red annotations
in architectural facade drawings.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DetectedRegion:
    label: str
    region_type: str  # "facade", "window", "door"
    bbox: tuple  # (x, y, w, h) in pixels
    contour: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    area_px: float = 0.0
    area_m2: Optional[float] = None
    width_m: Optional[float] = None
    height_m: Optional[float] = None
    ocr_text: str = ""
    parent_facade: Optional[str] = None
    color_detected: str = ""

    @property
    def center(self):
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)


class ColorDetector:

    WINDOW_HSV = {
        "yellow": {"lower": (15, 50, 100), "upper": (35, 255, 255)},
    }

    DOOR_HSV = {
        "magenta": {"lower": (130, 40, 80), "upper": (170, 255, 255)},
        "purple": {"lower": (110, 30, 80), "upper": (135, 255, 255)},
    }

    def __init__(self):
        self.scale_factor = None

    def detect_windows(self, image: np.ndarray,
                       min_area: int = 1200, max_area: int = 50000) -> list:
        """Detect yellow window rectangles. Tuned for clear yellow annotation boxes."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for bounds in self.WINDOW_HSV.values():
            m = cv2.inRange(hsv, np.array(bounds["lower"]),
                            np.array(bounds["upper"]))
            mask = cv2.bitwise_or(mask, m)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            ratio = w / h if h > 0 else 0
            if ratio < 0.3 or ratio > 4.0:
                continue

            # Rectangularity check: contour area vs bounding rect area
            rect_area = w * h
            rectangularity = area / rect_area if rect_area > 0 else 0
            if rectangularity < 0.4:
                continue

            regions.append(DetectedRegion(
                label=f"F_{len(regions)+1}",
                region_type="window",
                bbox=(x, y, w, h),
                contour=contour,
                area_px=area,
                color_detected="yellow",
            ))

        regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return regions

    def detect_doors_by_color(self, image: np.ndarray,
                              min_area: int = 1500, max_area: int = 80000) -> list:
        """Detect door annotations by magenta/purple outlines."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for bounds in self.DOOR_HSV.values():
            m = cv2.inRange(hsv, np.array(bounds["lower"]),
                            np.array(bounds["upper"]))
            mask = cv2.bitwise_or(mask, m)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            ratio = w / h if h > 0 else 0
            # Doors: not extremely wide-thin lines
            if ratio > 6 and h < 20:
                continue
            if ratio < 0.1:
                continue

            regions.append(DetectedRegion(
                label=f"U_{len(regions)+1}",
                region_type="door",
                bbox=(x, y, w, h),
                contour=contour,
                area_px=area,
                color_detected="magenta/purple",
            ))

        regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return regions

    def draw_detections(self, image: np.ndarray, results: dict) -> np.ndarray:
        """Draw detection results as colored overlays for visualization."""
        viz = image.copy()

        style_map = {
            "facades": ((255, 180, 0), 3, 0.12),
            "windows": ((0, 255, 255), 2, 0.20),
            "doors":   ((0, 100, 255), 2, 0.20),
        }

        for key, (color, thickness, alpha) in style_map.items():
            for region in results.get(key, []):
                x, y, w, h = region.bbox

                overlay = viz.copy()
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
                cv2.addWeighted(overlay, alpha, viz, 1 - alpha, 0, viz)
                cv2.rectangle(viz, (x, y), (x + w, y + h), color, thickness)

                label = region.label
                if region.area_m2 is not None:
                    label += f" {region.area_m2:.2f}m²"

                fs = 0.45
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1
                )
                ty = max(y - 4, th + 5)
                cv2.rectangle(viz, (x, ty - th - 3), (x + tw + 4, ty + 3),
                              color, -1)
                cv2.putText(viz, label, (x + 2, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1)

        return viz
