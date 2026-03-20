"""
DrawQuantPDF - OCR Engine
Extracts text, measurements, and labels from architectural drawings using EasyOCR.
"""

import re
import cv2
import numpy as np
from typing import Optional
from dataclasses import dataclass

import config


@dataclass
class OCRResult:
    text: str
    bbox: tuple  # (x_min, y_min, x_max, y_max)
    confidence: float
    parsed_value: Optional[float] = None
    unit: str = ""


class OCREngine:
    def __init__(self, languages=None, gpu=None):
        self.languages = languages or config.OCR_LANGUAGES
        self.gpu = gpu if gpu is not None else config.OCR_GPU
        self._reader = None

    @property
    def reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader

    def extract_all_text(self, image: np.ndarray, upscale: int = 2) -> list:
        """Extract all text from an image, with optional upscaling for small text."""
        if upscale > 1:
            h, w = image.shape[:2]
            image_proc = cv2.resize(image, (w * upscale, h * upscale),
                                    interpolation=cv2.INTER_CUBIC)
        else:
            image_proc = image
            upscale = 1

        # Enhance contrast for better OCR
        lab = cv2.cvtColor(image_proc, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(l_channel)
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        results = self.reader.readtext(enhanced)
        ocr_results = []
        for bbox_pts, text, conf in results:
            pts = np.array(bbox_pts)
            x_min, y_min = pts.min(axis=0).astype(int)
            x_max, y_max = pts.max(axis=0).astype(int)

            # Scale coordinates back to original size
            x_min = int(x_min / upscale)
            y_min = int(y_min / upscale)
            x_max = int(x_max / upscale)
            y_max = int(y_max / upscale)

            parsed = self._parse_measurement(text)
            ocr_results.append(OCRResult(
                text=text,
                bbox=(x_min, y_min, x_max, y_max),
                confidence=conf,
                parsed_value=parsed[0] if parsed else None,
                unit=parsed[1] if parsed else "",
            ))
        return ocr_results

    def extract_text_in_region(self, image: np.ndarray, bbox: tuple) -> list:
        """Extract text only within a specific bounding box region."""
        x, y, w, h = bbox
        pad = 10
        y1 = max(0, y - pad)
        y2 = min(image.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(image.shape[1], x + w + pad)

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return []

        results = self.reader.readtext(roi)
        ocr_results = []
        for bbox_pts, text, conf in results:
            pts = np.array(bbox_pts)
            rx_min, ry_min = pts.min(axis=0).astype(int)
            rx_max, ry_max = pts.max(axis=0).astype(int)

            parsed = self._parse_measurement(text)
            ocr_results.append(OCRResult(
                text=text,
                bbox=(rx_min + x1, ry_min + y1, rx_max + x1, ry_max + y1),
                confidence=conf,
                parsed_value=parsed[0] if parsed else None,
                unit=parsed[1] if parsed else "",
            ))
        return ocr_results

    def find_area_values(self, ocr_results: list) -> list:
        """Filter OCR results to find area measurements (m²)."""
        area_results = []
        for r in ocr_results:
            if r.parsed_value is not None and r.unit in ("m²", "m2", "mp"):
                area_results.append(r)
        return area_results

    def find_dimension_values(self, ocr_results: list) -> list:
        """Filter OCR results to find linear measurements (m)."""
        dim_results = []
        for r in ocr_results:
            if r.parsed_value is not None and r.unit in ("m", ""):
                if 0.1 < r.parsed_value < 50:
                    dim_results.append(r)
        return dim_results

    def find_labels(self, ocr_results: list) -> list:
        """Find facade/window/door labels like FATADA, F2, U2, etc."""
        label_results = []
        label_patterns = [
            r"FATADA",
            r"F\d+",
            r"U\d+",
            r"PRINCIPALA",
            r"POSTERIOARA",
            r"LATERALA",
            r"STANGA",
            r"DREAPTA",
        ]
        combined = "|".join(label_patterns)
        for r in ocr_results:
            if re.search(combined, r.text.upper()):
                label_results.append(r)
        return label_results

    def _parse_measurement(self, text: str) -> Optional[tuple]:
        """Parse a measurement string into (value, unit).
        Handles formats like: '24.29 m²', '6.080 m', '24.29m2', '6,080',
        and OCR artifacts like '226m' which should be '2.26 m'.
        """
        text = text.strip().replace(",", ".")

        # Standard area: "24.29 m²", "58.86m2", "31.53 mp"
        area_match = re.search(
            r"(\d+\.?\d*)\s*(?:m[²2]|mp)", text, re.IGNORECASE
        )
        if area_match:
            val = float(area_match.group(1))
            # Correct OCR artifact: "5886m" → 58.86, "226m" → 2.26
            if val > 100 and "." not in area_match.group(1):
                val = self._fix_missing_decimal(val)
            return (val, "m²")

        # Standard dimension: "6.080 m", "5.860m"
        dim_match = re.search(r"(\d+\.?\d*)\s*m(?![\d²2p])", text, re.IGNORECASE)
        if dim_match:
            val = float(dim_match.group(1))
            if val > 100 and "." not in dim_match.group(1):
                val = self._fix_missing_decimal(val)
            return (val, "m")

        # Number ending with 'm' where OCR dropped the decimal
        # e.g. "226m" → 2.26, "6220m" → 6.220
        m_match = re.search(r"(\d{3,5})\s*m", text, re.IGNORECASE)
        if m_match:
            raw = m_match.group(1)
            val = self._fix_missing_decimal(float(raw))
            if 0.1 < val < 200:
                return (val, "m" if val < 30 else "m²")

        # Bare number with decimal: "24.29", "6.080", "54.279"
        num_match = re.search(r"(\d+\.\d{2,3})\b", text)
        if num_match:
            val = float(num_match.group(1))
            if 0.1 < val < 200:
                return (val, "m" if val < 30 else "m²")

        return None

    @staticmethod
    def _fix_missing_decimal(val: float) -> float:
        """Fix OCR artifacts where the decimal point was dropped.
        226 → 2.26, 5886 → 58.86, 6220 → 6.220, 6320 → 6.320
        """
        s = str(int(val))
        if len(s) == 3:
            return float(s[0] + "." + s[1:])       # 226 → 2.26
        elif len(s) == 4:
            v1 = float(s[:2] + "." + s[2:])        # 5886 → 58.86
            v2 = float(s[0] + "." + s[1:])         # 6220 → 6.220
            # Prefer dimension-range (0.5-15m) over large area
            if 0.5 < v2 < 15:
                return v2
            if 1 < v1 < 200:
                return v1
            return v2
        elif len(s) == 5:
            v1 = float(s[:2] + "." + s[2:])        # 54279 → 54.279 (won't work)
            v2 = float(s[:3] + "." + s[2:])        # not useful
            # Most likely pattern: 54279 → 54.279
            return float(s[:2] + "." + s[2:])
        return val

    @staticmethod
    def _area_bounds_for_region(region_type: str) -> tuple:
        if region_type == "window":
            return (0.1, 15.0)
        if region_type == "door":
            return (0.5, 6.5)
        if region_type == "facade":
            return (1.0, 5000.0)
        return (0.05, 5000.0)

    @staticmethod
    def _dimension_bounds_for_region(region_type: str) -> tuple:
        if region_type == "window":
            return (0.2, 4.5)
        if region_type == "door":
            return (0.6, 3.5)
        if region_type == "facade":
            return (0.5, 100.0)
        return (0.2, 50.0)

    def _pick_dimension_pair(self, dims: list, area_min: float,
                             area_max: float) -> Optional[tuple]:
        vals = sorted(set(float(v) for v in dims))
        if len(vals) < 2:
            return None

        best = None
        best_score = float("inf")
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                a = vals[i]
                b = vals[j]
                area = a * b
                if not (area_min <= area <= area_max):
                    continue

                # Prefer compact pairs over very different dimensions.
                score = abs(a - b)
                if score < best_score:
                    best_score = score
                    best = (max(a, b), min(a, b))

        if best:
            return best

        # Fallback: take two smallest values if no pair is plausible.
        return (vals[1], vals[0])

    def enrich_regions(self, image: np.ndarray, regions: list) -> list:
        """Run OCR on each detected region and populate its fields."""
        for region in regions:
            ocr_results = self.extract_text_in_region(image, region.bbox)
            texts = [r.text for r in ocr_results]
            region.ocr_text = " | ".join(texts)

            region_type = getattr(region, "region_type", "")
            area_min, area_max = self._area_bounds_for_region(region_type)
            dim_min, dim_max = self._dimension_bounds_for_region(region_type)

            area_vals = [
                a.parsed_value for a in self.find_area_values(ocr_results)
                if a.parsed_value is not None and area_min <= a.parsed_value <= area_max
            ]
            if area_vals:
                region.area_m2 = float(np.median(area_vals))

            dim_vals = [
                d.parsed_value for d in self.find_dimension_values(ocr_results)
                if d.parsed_value is not None and dim_min <= d.parsed_value <= dim_max
            ]
            if len(dim_vals) >= 2:
                pair = self._pick_dimension_pair(dim_vals, area_min, area_max)
                if pair:
                    region.width_m, region.height_m = pair
                    if not region.area_m2:
                        inferred_area = round(region.width_m * region.height_m, 3)
                        if area_min <= inferred_area <= area_max:
                            region.area_m2 = inferred_area
            elif len(dim_vals) == 1 and region.area_m2:
                region.width_m = float(dim_vals[0])
                if region.width_m > 0:
                    region.height_m = round(region.area_m2 / region.width_m, 3)

            labels = self.find_labels(ocr_results)
            if labels:
                region.label = labels[0].text.strip()

        return regions
