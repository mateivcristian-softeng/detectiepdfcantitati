"""
DrawQuantPDF - Color-based Region Detector
Specialized for detecting yellow window rectangles and orange/red annotations
in architectural facade drawings.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import config


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
    length_m: Optional[float] = None
    ocr_text: str = ""
    parent_facade: Optional[str] = None
    color_detected: str = ""
    is_open_path: bool = False

    @property
    def center(self):
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)


class ColorDetector:
    def __init__(self):
        self.scale_factor = None
        self.facade_hsv = config.COLOR_RANGES.get("facade", {})
        self.window_hsv = config.COLOR_RANGES.get("window", {})
        self.door_hsv = config.COLOR_RANGES.get("door", {})

    @staticmethod
    def _build_mask(hsv: np.ndarray, ranges: dict) -> np.ndarray:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for bounds in ranges.values():
            lower = np.array(bounds["lower"], dtype=np.uint8)
            upper = np.array(bounds["upper"], dtype=np.uint8)
            segment = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_or(mask, segment)
        return mask

    @staticmethod
    def _cleanup_mask(mask: np.ndarray, close_kernel=(5, 5),
                      open_kernel=(3, 3), dilate_iterations: int = 0) -> np.ndarray:
        close_k = cv2.getStructuringElement(cv2.MORPH_RECT, close_kernel)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)

        if open_kernel:
            open_k = cv2.getStructuringElement(cv2.MORPH_RECT, open_kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_k)

        if dilate_iterations > 0:
            cleaned = cv2.dilate(cleaned, close_k, iterations=dilate_iterations)

        return cleaned

    @staticmethod
    def _keep_seed_component(mask: np.ndarray, seed: tuple) -> np.ndarray:
        """Keep connected component containing a seed point."""
        if mask.size == 0:
            return mask

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if num_labels <= 1:
            return mask

        sx = max(0, min(mask.shape[1] - 1, int(seed[0])))
        sy = max(0, min(mask.shape[0] - 1, int(seed[1])))
        target = int(labels[sy, sx])
        if target == 0:
            # Fallback: largest non-background component.
            areas = stats[1:, cv2.CC_STAT_AREA]
            if len(areas) == 0:
                return np.zeros_like(mask)
            target = int(np.argmax(areas)) + 1

        component = np.zeros_like(mask)
        component[labels == target] = 255
        return component

    @staticmethod
    def _fill_holes(mask: np.ndarray) -> np.ndarray:
        """Fill internal holes in a binary mask."""
        filled = mask.copy()
        h, w = filled.shape[:2]
        flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(filled, flood, (0, 0), 255)
        inv = cv2.bitwise_not(filled)
        return cv2.bitwise_or(mask, inv)

    @staticmethod
    def _shift_contour(contour: np.ndarray, dx: int, dy: int) -> np.ndarray:
        return contour + np.array([[[dx, dy]]], dtype=contour.dtype)

    @staticmethod
    def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
        """Drop tiny connected components likely caused by texture noise."""
        if mask.size == 0 or min_area <= 1:
            return mask
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n_labels <= 1:
            return mask
        out = np.zeros_like(mask)
        for idx in range(1, n_labels):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area >= min_area:
                out[labels == idx] = 255
        return out

    @staticmethod
    def _extract_main_contour(mask: np.ndarray, simplify_ratio: float = 0.005):
        """Return largest contour and its area from a binary mask."""
        if mask.size == 0:
            return None, 0.0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0
        contour = max(contours, key=cv2.contourArea)
        if simplify_ratio > 0:
            perim = float(cv2.arcLength(contour, True))
            eps = max(1.0, perim * simplify_ratio)
            contour = cv2.approxPolyDP(contour, eps, True)
        area = float(cv2.contourArea(contour))
        return contour, area

    @staticmethod
    def _merge_region_masks(shape: tuple, regions: list) -> tuple:
        """Union multiple region contours/bboxes into one simplified contour."""
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for region in regions:
            if region is None or not region.bbox:
                continue
            contour = getattr(region, "contour", None)
            if isinstance(contour, np.ndarray) and contour.size >= 6:
                cv2.drawContours(mask, [contour.astype(np.int32)], -1, 255, -1)
            else:
                x, y, bw, bh = region.bbox
                cv2.rectangle(mask, (x, y), (x + bw - 1, y + bh - 1), 255, -1)
        contour, area = ColorDetector._extract_main_contour(mask, simplify_ratio=0.0035)
        if contour is None or area <= 0:
            return np.array([]), 0.0
        return contour, float(area)

    @staticmethod
    def _region_to_mask(shape: tuple, region: Optional[DetectedRegion]) -> np.ndarray:
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if region is None or not region.bbox:
            return mask
        contour = getattr(region, "contour", None)
        if isinstance(contour, np.ndarray) and contour.size >= 6:
            cv2.drawContours(mask, [contour.astype(np.int32)], -1, 255, -1)
        else:
            x, y, bw, bh = region.bbox
            cv2.rectangle(mask, (x, y), (x + bw - 1, y + bh - 1), 255, -1)
        return mask

    @staticmethod
    def _horizontal_overlap(a: tuple, b: tuple) -> int:
        ax, _, aw, _ = a
        bx, _, bw, _ = b
        return max(0, min(ax + aw, bx + bw) - max(ax, bx))

    def _detect_photo_facade_cap_line(self, image: np.ndarray,
                                     region: Optional[DetectedRegion]):
        """Find the dominant long roof-wall line used to cap the facade body."""
        if image is None or image.size == 0 or region is None or not region.bbox:
            return None

        x, y, w, h = region.bbox
        if w < 120 or h < 80:
            return None
        roi = image[y:y + h, x:x + w]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        edges = cv2.Canny(gray, 60, 150)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=max(40, int(w * 0.05)),
            minLineLength=max(60, int(w * 0.45)),
            maxLineGap=max(20, int(w * 0.04)),
        )
        if lines is None:
            return None

        candidates = []
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, line)
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < max(24, int(w * 0.10)):
                continue
            slope = abs(dy / max(1.0, float(dx)))
            if slope > 0.12:
                continue
            length = float((dx * dx + dy * dy) ** 0.5)
            mean_y = (y1 + y2) * 0.5
            if mean_y < h * 0.04 or mean_y > h * 0.52:
                continue
            if min(y1, y2) > h * 0.40:
                continue
            candidates.append((mean_y, length, (x1, y1, x2, y2)))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        x1, y1, x2, y2 = candidates[0][2]
        if x2 == x1:
            return None
        a = (y2 - y1) / float(x2 - x1)
        b = y1 - a * x1
        return float(a), float(b)

    def _cap_photo_region_with_line(self, image_shape: tuple,
                                    region: Optional[DetectedRegion],
                                    line_params) -> Optional[DetectedRegion]:
        """Trim a facade region above a detected roof-wall cap line."""
        if region is None or line_params is None or not region.bbox:
            return region

        a, b = line_params
        x, y, w, h = region.bbox
        full_mask = self._region_to_mask(image_shape, region)
        local_mask = full_mask[y:y + h, x:x + w].copy()
        if local_mask.size == 0:
            return region

        for cx in range(w):
            cy = int(round(a * cx + b))
            cy = max(0, min(h - 1, cy))
            local_mask[:cy, cx] = 0

        contour, area = self._extract_main_contour(local_mask, simplify_ratio=0.0035)
        if contour is None or area <= 0:
            return region
        contour = self._shift_contour(contour, x, y)
        bx, by, bw, bh = cv2.boundingRect(contour)
        return DetectedRegion(
            label=region.label,
            region_type=region.region_type,
            bbox=(int(bx), int(by), int(bw), int(bh)),
            contour=contour,
            area_px=float(area),
            color_detected=region.color_detected or "photo-capped-facade",
        )

    def _merge_attached_photo_facade(self, image_shape: tuple,
                                     base_region: Optional[DetectedRegion],
                                     attachment_region: Optional[DetectedRegion]) -> Optional[DetectedRegion]:
        """Merge a lower facade body with a plausible upper attached volume."""
        if base_region is None or attachment_region is None:
            return None

        bx, by, bw, bh = base_region.bbox
        ax, ay, aw, ah = attachment_region.bbox
        if bw <= 0 or bh <= 0 or aw <= 0 or ah <= 0:
            return None

        min_w_ratio = float(getattr(config, "PHOTO_FACADE_ATTACHMENT_MIN_WIDTH_RATIO", 0.10))
        max_w_ratio = float(getattr(config, "PHOTO_FACADE_ATTACHMENT_MAX_WIDTH_RATIO", 0.78))
        min_h_ratio = float(getattr(config, "PHOTO_FACADE_ATTACHMENT_MIN_HEIGHT_RATIO", 0.16))
        min_overlap_ratio = float(getattr(config, "PHOTO_FACADE_ATTACHMENT_MIN_OVERLAP_RATIO", 0.45))
        touch_ratio = float(getattr(config, "PHOTO_FACADE_ATTACHMENT_TOUCH_RATIO", 0.14))

        width_ratio = aw / max(1.0, float(bw))
        height_ratio = ah / max(1.0, float(bh))
        overlap_x = self._horizontal_overlap(base_region.bbox, attachment_region.bbox)
        if width_ratio < min_w_ratio or width_ratio > max_w_ratio:
            return None
        if height_ratio < min_h_ratio or height_ratio > 0.98:
            return None
        if overlap_x < min(aw, bw) * min_overlap_ratio:
            return None
        if ay >= by - max(16, int(round(bh * 0.04))):
            return None
        if ay + ah < by - max(20, int(round(bh * touch_ratio))):
            return None

        merged_contour, merged_area = self._merge_region_masks(
            image_shape, [base_region, attachment_region]
        )
        if merged_area <= 0 or merged_contour.size < 6:
            return None

        mx, my, mw, mh = cv2.boundingRect(merged_contour)
        return DetectedRegion(
            label=base_region.label or attachment_region.label or "FATADA_PHOTO_1",
            region_type="facade",
            bbox=(int(mx), int(my), int(mw), int(mh)),
            contour=merged_contour,
            area_px=float(merged_area),
            color_detected="photo-segmentation-merged",
        )

    @staticmethod
    def _photo_window_width_ok(fw: int, fh: int, w_box: int, h_box: int) -> bool:
        min_w_ratio = float(getattr(config, "PHOTO_WINDOW_MIN_WIDTH_RATIO", 0.020))
        strict_min_w_ratio = float(getattr(config, "PHOTO_WINDOW_STRICT_MIN_WIDTH_RATIO", 0.035))
        narrow_min_h_ratio = float(getattr(config, "PHOTO_WINDOW_NARROW_MIN_HEIGHT_RATIO", 0.16))
        min_w_px = max(14, int(round(fw * min_w_ratio)))
        strict_min_w_px = max(min_w_px, int(round(fw * strict_min_w_ratio)))
        if w_box < min_w_px:
            return False
        if w_box < strict_min_w_px and h_box < int(round(fh * narrow_min_h_ratio)):
            return False
        return True

    @staticmethod
    def _has_dark_background(image: np.ndarray) -> bool:
        if image is None or image.size == 0:
            return False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        border = max(8, min(h, w) // 35)
        border_pixels = np.concatenate([
            gray[:border, :].ravel(),
            gray[-border:, :].ravel(),
            gray[:, :border].ravel(),
            gray[:, -border:].ravel(),
        ])
        return float(np.percentile(border_pixels, 80)) <= 25.0

    def _refine_facade_candidate_mask(self, mask: np.ndarray, seed: tuple) -> np.ndarray:
        """Denoise facade candidate and keep only the dominant wall component."""
        if mask.size == 0:
            return mask
        refined = self._cleanup_mask(
            mask,
            close_kernel=(11, 11),
            open_kernel=(5, 5),
            dilate_iterations=0,
        )
        min_comp = max(120, int(refined.shape[0] * refined.shape[1] * 0.003))
        refined = self._remove_small_components(refined, min_comp)
        refined = self._keep_seed_component(refined, seed)
        refined = self._fill_holes(refined)
        smooth_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, smooth_k)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, smooth_k)
        refined = self._fill_holes(refined)
        return refined

    def _trim_side_spikes(self, mask: np.ndarray) -> np.ndarray:
        """Trim narrow lateral protrusions based on column occupancy stability."""
        if mask.size == 0:
            return mask

        h, w = mask.shape[:2]
        col_occ = (mask > 0).sum(axis=0).astype(np.float32)
        nz = col_occ[col_occ > 0]
        if len(nz) < 8:
            return mask

        occ_ratio = float(getattr(config, "PHOTO_FACADE_SIDE_OCC_PERCENT_RATIO", 0.58))
        occ_thr = max(3.0, float(np.percentile(nz, 70)) * occ_ratio)
        valid = np.where(col_occ >= occ_thr)[0]
        if len(valid) == 0:
            return mask

        runs = []
        start = int(valid[0])
        prev = int(valid[0])
        for c in valid[1:]:
            c = int(c)
            if c == prev + 1:
                prev = c
                continue
            runs.append((start, prev))
            start = prev = c
        runs.append((start, prev))

        min_run = max(8, int(w * float(getattr(config, "PHOTO_FACADE_SIDE_MIN_RUN_RATIO", 0.18))))
        long_runs = [r for r in runs if (r[1] - r[0] + 1) >= min_run]
        if long_runs:
            runs = long_runs

        center = w // 2
        best = None
        best_score = -1.0
        for a, b in runs:
            occ_sum = float(col_occ[a:b + 1].sum())
            run_center = (a + b) * 0.5
            center_pen = abs(run_center - center) / max(1.0, float(w))
            score = occ_sum * (1.0 - 0.35 * center_pen)
            if a <= center <= b:
                score *= 1.15
            if score > best_score:
                best = (a, b)
                best_score = score

        if best is None:
            return mask

        margin = int(w * float(getattr(config, "PHOTO_FACADE_SIDE_MARGIN_RATIO", 0.010)))
        left = max(0, best[0] - margin)
        right = min(w - 1, best[1] + margin)

        trimmed = np.zeros_like(mask)
        trimmed[:, left:right + 1] = mask[:, left:right + 1]

        old_area = float(np.count_nonzero(mask))
        new_area = float(np.count_nonzero(trimmed))
        min_keep = float(getattr(config, "PHOTO_FACADE_SIDE_MIN_KEEP_RATIO", 0.72))
        if old_area <= 0 or (new_area / old_area) < min_keep:
            return mask
        return trimmed

    @staticmethod
    def _nms_regions(regions: list, iou_threshold: float = 0.30) -> list:
        """Simple bbox NMS on DetectedRegion list."""
        if not regions:
            return []
        regions = sorted(regions, key=lambda r: r.area_px, reverse=True)
        kept = []
        for region in regions:
            drop = False
            for prev in kept:
                # Use existing IoU utility for consistency.
                ax, ay, aw, ah = region.bbox
                bx, by, bw, bh = prev.bbox
                inter_x1 = max(ax, bx)
                inter_y1 = max(ay, by)
                inter_x2 = min(ax + aw, bx + bw)
                inter_y2 = min(ay + ah, by + bh)
                if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                    continue
                inter = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
                union = float(aw * ah + bw * bh - inter)
                iou = inter / union if union > 0 else 0.0
                contain = inter / max(1.0, min(float(aw * ah), float(bw * bh)))
                if iou > iou_threshold or contain >= 0.72:
                    drop = True
                    break
            if not drop:
                kept.append(region)
        kept.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return kept

    def _detect_photo_facade_region_by_foreground(
        self, image: np.ndarray, min_area_ratio: float
    ) -> list:
        """Deterministic facade detector from foreground/background separation."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        border = max(8, min(h, w) // 35)
        border_pixels = np.concatenate([
            gray[:border, :].ravel(),
            gray[-border:, :].ravel(),
            gray[:, :border].ravel(),
            gray[:, -border:].ravel(),
        ])
        bg_pct = getattr(config, "PHOTO_FACADE_BG_BORDER_PERCENTILE", 65.0)
        delta = getattr(config, "PHOTO_FACADE_BG_THRESHOLD_DELTA", 10.0)
        bg_level = float(np.percentile(border_pixels, bg_pct))
        bg_dark_level = float(np.percentile(border_pixels, 80))
        dark_background = bg_dark_level <= 25.0
        if dark_background:
            thr = int(max(8, min(180, bg_level + max(8.0, delta))))
            mask = (gray > thr).astype(np.uint8) * 255
        else:
            thr = int(max(80, min(245, bg_level - delta)))
            mask = (gray < thr).astype(np.uint8) * 255
        mask = self._cleanup_mask(
            mask,
            close_kernel=(7, 7),
            open_kernel=(3, 3),
            dilate_iterations=0,
        )

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n_labels <= 1:
            return []
        idx = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
        comp_area = float(stats[idx, cv2.CC_STAT_AREA])
        if comp_area < (h * w * max(0.005, min_area_ratio * 0.4)):
            return []

        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        component = np.zeros_like(mask)
        component[labels == idx] = 255
        roi = component[y:y + bh, x:x + bw]
        if roi.size == 0:
            return []

        row_occ = (roi > 0).sum(axis=1).astype(np.float32)
        max_occ = float(max(1.0, np.max(row_occ)))
        occ_min = getattr(config, "PHOTO_FACADE_WALL_ROW_OCC_MIN", 0.78)
        occ_mean = getattr(config, "PHOTO_FACADE_WALL_ROW_OCC_MEAN", 0.74)
        scan_max = getattr(config, "PHOTO_FACADE_WALL_ROW_SCAN_MAX", 0.72)
        scan_end = max(8, int(len(row_occ) * scan_max))
        wall_top_local = 0
        for r in range(max(1, scan_end - 8)):
            if row_occ[r] >= max_occ * occ_min and row_occ[r:r + 8].mean() >= max_occ * occ_mean:
                wall_top_local = r
                break

        wall_top = y + wall_top_local
        wall_h = max(1, (y + bh) - wall_top)
        min_wall_h = int(h * getattr(config, "PHOTO_FACADE_WALL_MIN_HEIGHT_RATIO", 0.20))
        if wall_h < min_wall_h:
            return []

        wall_roi = roi[wall_top_local:, :]
        col_occ = (wall_roi > 0).sum(axis=0).astype(np.float32)
        col_min = max(8.0, wall_h * getattr(config, "PHOTO_FACADE_WALL_COL_OCC_MIN", 0.08))
        valid_cols = np.where(col_occ >= col_min)[0]
        if len(valid_cols) == 0:
            return []
        left = x + int(valid_cols[0])
        right = x + int(valid_cols[-1])
        wall_w = max(1, right - left + 1)

        # Optional bottom cut to remove noisy ground band from facade wall.
        wall_gray = gray[wall_top:wall_top + wall_h, left:left + wall_w]
        if wall_gray.size > 0:
            blur = cv2.GaussianBlur(wall_gray, (5, 5), 0)
            grad_y = np.abs(cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3))
            row_score = np.mean(grad_y, axis=1)
            b_start = int(wall_h * getattr(config, "PHOTO_FACADE_BOTTOM_CUT_Y_START", 0.55))
            b_end = int(wall_h * getattr(config, "PHOTO_FACADE_BOTTOM_CUT_Y_END", 0.97))
            if b_end > b_start + 6:
                zone = row_score[b_start:b_end]
                cut_local = int(np.argmax(zone) + b_start)
                peak = float(row_score[cut_local])
                base = float(np.median(zone)) + 1e-6
                band_ratio = (wall_h - cut_local) / max(1, wall_h)
                min_band = float(getattr(config, "PHOTO_FACADE_BOTTOM_CUT_MIN_BAND", 0.04))
                max_band = float(getattr(config, "PHOTO_FACADE_BOTTOM_CUT_MAX_BAND", 0.30))
                peak_ratio = float(getattr(config, "PHOTO_FACADE_BOTTOM_CUT_PEAK_RATIO", 1.25))
                if min_band <= band_ratio <= max_band and peak >= base * peak_ratio:
                    wall_h = max(1, cut_local)

        wall_zone = component[wall_top:wall_top + wall_h, left:left + wall_w]
        raw_area_px = float(np.count_nonzero(wall_zone))
        wall_zone_trimmed = self._trim_side_spikes(wall_zone)
        area_px = raw_area_px
        if area_px < (h * w * min_area_ratio):
            return []

        contours, _ = cv2.findContours(
            wall_zone_trimmed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour = None
        if contours:
            contour = max(contours, key=cv2.contourArea)
            contour = self._shift_contour(contour, left, wall_top)

        region = DetectedRegion(
            label="FATADA_PHOTO_1",
            region_type="facade",
            bbox=(left, wall_top, wall_w, wall_h),
            contour=contour if contour is not None else np.array([]),
            area_px=area_px,
            color_detected="photo-foreground-wall",
        )
        return [region]

    def detect_photo_facade_region(self, image: np.ndarray,
                                   min_area_ratio: Optional[float] = None) -> list:
        """Detect primary facade wall area in raw facade photos."""
        h, w = image.shape[:2]
        roi_y_start = getattr(config, "PHOTO_FACADE_ROI_Y_START", 0.15)
        roi_y_end = getattr(config, "PHOTO_FACADE_ROI_Y_END", 0.95)
        roof_start = getattr(config, "PHOTO_FACADE_ROOF_CUT_Y_START", 0.48)
        roof_end = getattr(config, "PHOTO_FACADE_ROOF_CUT_Y_END", 0.90)
        roof_min = getattr(config, "PHOTO_FACADE_ROOF_CUT_MIN", 0.50)
        roof_max = getattr(config, "PHOTO_FACADE_ROOF_CUT_MAX", 0.88)
        top_cut_start = getattr(config, "PHOTO_FACADE_TOP_CUT_Y_START", 0.05)
        top_cut_end = getattr(config, "PHOTO_FACADE_TOP_CUT_Y_END", 0.45)
        top_cut_min = getattr(config, "PHOTO_FACADE_TOP_CUT_MIN", 0.08)
        top_cut_max = getattr(config, "PHOTO_FACADE_TOP_CUT_MAX", 0.60)
        top_peak_ratio = getattr(config, "PHOTO_FACADE_TOP_CUT_PEAK_RATIO", 1.25)
        max_h_ratio = getattr(config, "PHOTO_FACADE_MAX_HEIGHT_RATIO", 0.78)
        eff_min_area = min_area_ratio if min_area_ratio is not None else getattr(
            config, "PHOTO_FACADE_MIN_AREA_RATIO", 0.08
        )

        x0 = int(w * 0.05)
        x1 = int(w * 0.95)
        y0 = int(h * roi_y_start)
        y1 = int(h * roi_y_end)
        roi = image[y0:y1, x0:x1]
        if roi.size == 0:
            return []
        fallback_regions = self._detect_photo_facade_region_by_foreground(
            image, eff_min_area
        )

        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        data = lab.reshape((-1, 3)).astype(np.float32)
        k = 4
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            20,
            1.0,
        )
        _, labels, _ = cv2.kmeans(
            data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
        labels = labels.reshape(lab.shape[:2])

        seed = (int(roi.shape[1] * 0.5), int(roi.shape[0] * 0.65))
        seed_label = int(labels[seed[1], seed[0]])
        mask = np.zeros(labels.shape, dtype=np.uint8)
        mask[labels == seed_label] = 255
        mask = self._cleanup_mask(mask, close_kernel=(15, 15),
                                  open_kernel=(5, 5), dilate_iterations=1)
        mask = self._keep_seed_component(mask, seed)
        mask = self._fill_holes(mask)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        grad_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        support = (mask > 0).astype(np.float32)
        row_score = (grad_y * support).sum(axis=1) / (support.sum(axis=1) + 1e-6)
        start = int(roi.shape[0] * roof_start)
        end = int(roi.shape[0] * roof_end)
        if end > start:
            cut = int(np.argmax(row_score[start:end]) + start)
            cut_min = int(roi.shape[0] * roof_min)
            cut_max = int(roi.shape[0] * roof_max)
            if cut_min <= cut <= cut_max:
                mask[cut + 2:, :] = 0

        # Trim roof/sky spill at the top when a strong horizontal transition exists.
        top_start = int(roi.shape[0] * top_cut_start)
        top_end = int(roi.shape[0] * top_cut_end)
        if top_end > top_start + 5:
            upper = row_score[top_start:top_end]
            if upper.size > 0:
                top_cut = int(np.argmax(upper) + top_start)
                top_min = int(roi.shape[0] * top_cut_min)
                top_max = int(roi.shape[0] * top_cut_max)
                if top_min <= top_cut <= top_max:
                    peak = float(row_score[top_cut])
                    base = float(np.median(upper)) + 1e-6
                    if peak >= base * top_peak_ratio:
                        mask[:max(0, top_cut - 2), :] = 0

        # Stabilize facade top/bottom using mask occupancy to suppress roof noise.
        support = (mask > 0).astype(np.uint8)
        row_occ = support.sum(axis=1).astype(np.float32)
        if row_occ.size > 0:
            max_occ = float(max(1.0, np.max(row_occ)))
            occ_ratio = row_occ / max_occ
            valid = np.where(occ_ratio >= 0.22)[0]
            if len(valid) > 0:
                top_keep = int(valid[0])
                bottom_keep = int(valid[-1])

                scan_end = max(top_keep + 4, int(mask.shape[0] * 0.46))
                stable_top = top_keep
                for r in range(max(1, scan_end - 6)):
                    if np.mean(occ_ratio[r:r + 6]) >= 0.68:
                        stable_top = r
                        break
                top_keep = max(top_keep, stable_top)

                mask[:max(0, top_keep - 1), :] = 0
                if bottom_keep + 1 < mask.shape[0]:
                    mask[bottom_keep + 1:, :] = 0

        contours_raw, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
        if not contours_raw:
            return fallback_regions
        contour_raw = max(contours_raw, key=cv2.contourArea)
        raw_area = float(cv2.contourArea(contour_raw))
        if raw_area < (h * w * eff_min_area):
            return fallback_regions

        mask_trimmed = self._trim_side_spikes(mask)
        contours_trim, _ = cv2.findContours(
            mask_trimmed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour = max(contours_trim, key=cv2.contourArea) if contours_trim else contour_raw
        area = float(cv2.contourArea(contour))
        if area < (h * w * eff_min_area):
            return fallback_regions

        bx, by, bw, bh = cv2.boundingRect(contour)
        max_fh = int(h * max_h_ratio)
        if bh > max_fh:
            by = by + bh - max_fh
            bh = max_fh

        contour_global = self._shift_contour(contour, x0, y0)
        region = DetectedRegion(
            label="FATADA_PHOTO_1",
            region_type="facade",
            bbox=(x0 + bx, y0 + by, bw, bh),
            contour=contour_global,
            area_px=area,
            color_detected="photo-segmentation",
        )

        fallback_primary = fallback_regions[0] if fallback_regions else None
        merged_facade = self._merge_attached_photo_facade(
            image.shape, fallback_primary, region
        )

        best = fallback_primary if fallback_primary is not None else region
        if merged_facade is not None:
            if best is None or merged_facade.area_px > float(best.area_px) * 1.05:
                best = merged_facade

        return [best] if best is not None else []

    def detect_photo_windows(self, image: np.ndarray, facades: list) -> list:
        """Detect dark window-like regions inside a detected facade."""
        if not facades:
            return []

        regions = []
        idx = 1
        for facade in facades:
            fx, fy, fw, fh = facade.bbox
            roi = image[fy:fy + fh, fx:fx + fw]
            if roi.size == 0:
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            thresh = np.percentile(gray, 36)
            dark = (gray <= thresh).astype(np.uint8) * 255
            edges = cv2.Canny(gray, 60, 150)
            edges = cv2.dilate(edges, cv2.getStructuringElement(
                cv2.MORPH_RECT, (3, 3)
            ), iterations=1)

            # Black-hat highlights openings that are darker than the wall plane.
            bh_kw = max(9, int(round(fw * 0.09)))
            bh_kh = max(9, int(round(fh * 0.14)))
            if bh_kw % 2 == 0:
                bh_kw += 1
            if bh_kh % 2 == 0:
                bh_kh += 1
            bh_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (bh_kw, bh_kh)
            )
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, bh_kernel)
            bh_thr = float(np.percentile(blackhat, 84.0))
            bh_mask = (blackhat >= bh_thr).astype(np.uint8) * 255
            bh_mask = self._cleanup_mask(
                bh_mask,
                close_kernel=(7, 7),
                open_kernel=(3, 3),
                dilate_iterations=0,
            )

            mask = cv2.bitwise_and(dark, edges)
            dark_soft = self._cleanup_mask(
                dark, close_kernel=(9, 9), open_kernel=(3, 3), dilate_iterations=0
            )
            mask = cv2.bitwise_or(mask, dark_soft)
            mask = cv2.bitwise_or(mask, bh_mask)
            mask = self._cleanup_mask(mask, close_kernel=(7, 7),
                                      open_kernel=(3, 3), dilate_iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            facade_area = max(1, fw * fh)
            min_a = max(180, int(facade_area * 0.0015))
            max_a = int(facade_area * 0.20)

            for c in contours:
                area = cv2.contourArea(c)
                if area < min_a or area > max_a:
                    continue

                x, y, w_box, h_box = cv2.boundingRect(c)
                if not self._photo_window_width_ok(fw, fh, w_box, h_box):
                    continue
                if h_box < int(fh * 0.05):
                    continue
                if w_box > int(fw * 0.65) or h_box > int(fh * 0.85):
                    continue
                ratio = w_box / h_box if h_box else 0
                if ratio < 0.3 or ratio > 3.5:
                    continue

                rect_area = w_box * h_box
                extent = area / rect_area if rect_area else 0
                if extent < 0.18:
                    continue

                # Avoid border artefacts.
                if x < int(fw * 0.02) or y < int(fh * 0.02):
                    continue
                if (x + w_box) > int(fw * 0.98) or (y + h_box) > int(fh * 0.98):
                    continue

                # Exclude roof zone (false positive windows on roof).
                cy_local = y + h_box // 2
                y_ratio = cy_local / fh if fh else 0
                min_y = getattr(config, "PHOTO_WINDOW_MIN_Y_RATIO", 0.12)
                max_y = getattr(config, "PHOTO_WINDOW_MAX_Y_RATIO", 0.92)
                if y_ratio < min_y or y_ratio > max_y:
                    continue
                bottom_ratio = (y + h_box) / max(1, fh)
                # Tall openings near the floor are more likely doors.
                if bottom_ratio >= 0.80 and h_box >= int(fh * 0.16) and 0.28 <= ratio <= 1.80:
                    continue

                # Reject very flat/bright textures; typical roof/sky false positives.
                win_gray = gray[y:y + h_box, x:x + w_box]
                if win_gray.size > 0:
                    lap_var = float(cv2.Laplacian(win_gray, cv2.CV_32F).var())
                    std_val = float(np.std(win_gray))
                    mean_val = float(np.mean(win_gray))
                    min_lap = float(getattr(config, "PHOTO_WINDOW_MIN_LAPLACIAN_VAR", 22.0))
                    strict_lap = float(getattr(
                        config, "PHOTO_WINDOW_LAPLACIAN_STRICT_MIN", 12000.0
                    ))
                    std_min = max(16.0, float(getattr(config, "PHOTO_WINDOW_TEXTURE_STD_MIN", 24.0)) * 0.80)
                    max_bright = float(getattr(config, "PHOTO_WINDOW_MAX_BRIGHT_MEAN", 185.0))

                    # Keep strong roof filtering, but do not reject valid low-texture
                    # windows (closed shutters / diffuse reflection).
                    if lap_var < min_lap and std_val < std_min * 0.75:
                        continue
                    if mean_val > max_bright and lap_var < min_lap * 1.8:
                        continue
                    if mean_val > (max_bright + 10.0) and lap_var < min(strict_lap, 2500.0):
                        continue

                contour_global = self._shift_contour(c, fx, fy)
                regions.append(DetectedRegion(
                    label=f"F_PHOTO_{idx}",
                    region_type="window",
                    bbox=(fx + x, fy + y, w_box, h_box),
                    contour=contour_global,
                    area_px=float(area),
                    color_detected="photo-dark-window",
                ))
                idx += 1

        filtered = self._nms_regions(regions, iou_threshold=0.30)
        appearance = self._detect_photo_windows_by_appearance(
            image, facades, start_idx=len(filtered) + 1
        )
        if appearance:
            filtered = self._nms_regions(filtered + appearance, iou_threshold=0.28)

        local_contrast = self._detect_photo_windows_by_local_contrast(
            image, facades, start_idx=len(filtered) + 1
        )
        if local_contrast:
            filtered = self._nms_regions(filtered + local_contrast, iou_threshold=0.26)

        filtered.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        for i, region in enumerate(filtered, 1):
            region.label = f"F_PHOTO_{i}"
        return filtered

    def _detect_photo_windows_by_appearance(
        self, image: np.ndarray, facades: list, start_idx: int = 1
    ) -> list:
        """Fallback window detector based on gradient + bright appearance cues."""
        regions = []
        idx = start_idx

        grad_pct = float(getattr(config, "PHOTO_WINDOW_APPEARANCE_GRADIENT_PERCENTILE", 91.0))
        bright_pcts = getattr(config, "PHOTO_WINDOW_APPEARANCE_BRIGHT_PERCENTILES", (78.0, 82.0))
        if not isinstance(bright_pcts, (list, tuple)):
            bright_pcts = (78.0, 82.0)

        std_min = max(16.0, float(getattr(config, "PHOTO_WINDOW_TEXTURE_STD_MIN", 24.0)) * 0.80)
        lap_min = min(2500.0, float(getattr(config, "PHOTO_WINDOW_LAPLACIAN_STRICT_MIN", 12000.0)))

        img_h, _ = image.shape[:2]
        for facade in facades:
            fx, fy, fw, fh = facade.bbox
            search_top = max(0, fy - int(round(fh * 0.60)))
            search_bottom = min(img_h, fy + fh + int(round(fh * 0.05)))
            roi = image[search_top:search_bottom, fx:fx + fw]
            if roi.size == 0:
                continue

            gray_raw = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray_raw)
            facade_area = max(1.0, float(fw * fh))

            # 1) Gradient-driven parts for textured/window-frame candidates.
            blur = cv2.bilateralFilter(gray, 9, 60, 60)
            grad_x = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
            grad_y = np.abs(cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3))
            energy = grad_x * 1.2 + grad_y * 0.8
            g_thr = float(np.percentile(energy, grad_pct))
            g_mask = (energy >= g_thr).astype(np.uint8) * 255
            g_mask = self._cleanup_mask(
                g_mask, close_kernel=(5, 5), open_kernel=(3, 3), dilate_iterations=0
            )

            contours_g, _ = cv2.findContours(g_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            parts = []
            for c in contours_g:
                area = cv2.contourArea(c)
                x, y, w_box, h_box = cv2.boundingRect(c)
                global_y = search_top + y
                y_ratio = ((global_y + h_box * 0.5) - fy) / max(1, fh)
                if not self._photo_window_width_ok(fw, fh, w_box, h_box):
                    continue
                if h_box < fh * 0.10:
                    continue
                if w_box > fw * 0.40 or h_box > fh * 0.55:
                    continue
                if y_ratio < -0.60 or y_ratio > 0.90:
                    continue
                if area < facade_area * 0.003:
                    continue
                if x < fw * 0.03 or (x + w_box) > fw * 0.97:
                    continue
                parts.append([x, y, w_box, h_box])

            parts.sort(key=lambda b: (b[1], b[0]))
            merged = []
            for x0, y0, w0, h0 in parts:
                attached = False
                for box in merged:
                    x1, y1, w1, h1 = box
                    overlap = max(0, min(y0 + h0, y1 + h1) - max(y0, y1))
                    gap = max(0, max(x0 - (x1 + w1), x1 - (x0 + w0)))
                    min_h = max(1, min(h0, h1))
                    if (overlap / min_h) >= 0.45 and gap <= fw * 0.08:
                        nx = min(x0, x1)
                        ny = min(y0, y1)
                        nx2 = max(x0 + w0, x1 + w1)
                        ny2 = max(y0 + h0, y1 + h1)
                        box[0], box[1], box[2], box[3] = nx, ny, nx2 - nx, ny2 - ny
                        attached = True
                        break
                if not attached:
                    merged.append([x0, y0, w0, h0])

            for x, y, w_box, h_box in merged:
                ratio = w_box / h_box if h_box else 0
                global_y = search_top + y
                y_ratio = ((global_y + h_box * 0.5) - fy) / max(1, fh)
                if not self._photo_window_width_ok(fw, fh, w_box, h_box):
                    continue
                if h_box < fh * 0.12:
                    continue
                if w_box > fw * 0.62 or h_box > fh * 0.60:
                    continue
                if ratio < 0.35 or ratio > 2.8:
                    continue
                if y_ratio < -0.20 or y_ratio > 0.86:
                    continue
                bottom_ratio = ((global_y + h_box) - fy) / max(1, fh)
                if bottom_ratio < 0.04:
                    continue
                if y_ratio < 0.12 and h_box < int(fh * 0.22):
                    continue
                if bottom_ratio >= 0.80 and h_box >= int(fh * 0.16) and ratio <= 1.80:
                    continue

                win_gray = gray[y:y + h_box, x:x + w_box]
                if win_gray.size == 0:
                    continue
                lap_var = float(cv2.Laplacian(win_gray, cv2.CV_32F).var())
                std_val = float(np.std(win_gray))
                if lap_var < lap_min or std_val < std_min:
                    continue

                regions.append(DetectedRegion(
                    label=f"F_PHOTO_{idx}",
                    region_type="window",
                    bbox=(fx + x, int(global_y), int(w_box), int(h_box)),
                    contour=np.array([]),
                    area_px=float(w_box * h_box),
                    color_detected="photo-gradient-window",
                ))
                idx += 1

            # 2) Bright-region fallback (captures light windows with dark frame).
            for pct in bright_pcts:
                b_thr = float(np.percentile(gray, float(pct)))
                b_mask = (gray >= b_thr).astype(np.uint8) * 255
                open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                b_mask = cv2.morphologyEx(b_mask, cv2.MORPH_OPEN, open_k)
                b_mask = cv2.morphologyEx(b_mask, cv2.MORPH_CLOSE, close_k)
                contours_b, _ = cv2.findContours(
                    b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for c in contours_b:
                    x, y, w_box, h_box = cv2.boundingRect(c)
                    area_ratio = (w_box * h_box) / facade_area
                    ratio = w_box / h_box if h_box else 0
                    global_y = search_top + y
                    y_ratio = ((global_y + h_box * 0.5) - fy) / max(1, fh)
                    if not self._photo_window_width_ok(fw, fh, w_box, h_box):
                        continue
                    if h_box < fh * 0.10:
                        continue
                    if w_box > fw * 0.55 or h_box > fh * 0.58:
                        continue
                    if ratio < 0.35 or ratio > 2.8:
                        continue
                    if y_ratio < -0.20 or y_ratio > 0.88:
                        continue
                    bottom_ratio = ((global_y + h_box) - fy) / max(1, fh)
                    if bottom_ratio < 0.04:
                        continue
                    if y_ratio < 0.12 and h_box < int(fh * 0.22):
                        continue
                    if bottom_ratio >= 0.80 and h_box >= int(fh * 0.16) and ratio <= 1.80:
                        continue
                    if x < fw * 0.03 or (x + w_box) > fw * 0.97:
                        continue
                    if area_ratio > 0.14:
                        continue

                    win_gray = gray[y:y + h_box, x:x + w_box]
                    if win_gray.size == 0:
                        continue
                    lap_var = float(cv2.Laplacian(win_gray, cv2.CV_32F).var())
                    std_val = float(np.std(win_gray))
                    if lap_var < lap_min or std_val < std_min:
                        continue

                    regions.append(DetectedRegion(
                        label=f"F_PHOTO_{idx}",
                        region_type="window",
                        bbox=(fx + x, int(global_y), int(w_box), int(h_box)),
                        contour=np.array([]),
                        area_px=float(w_box * h_box),
                        color_detected="photo-bright-window",
                    ))
                    idx += 1

        regions = self._nms_regions(regions, iou_threshold=0.30)
        return regions

    def _detect_photo_windows_by_local_contrast(
        self, image: np.ndarray, facades: list, start_idx: int = 1
    ) -> list:
        """Fallback detector for dark openings that stand out only against the local wall tone."""
        regions = []
        idx = start_idx

        contrast_pcts = getattr(
            config, "PHOTO_WINDOW_LOCAL_CONTRAST_PERCENTILES", (82.0, 86.0)
        )
        if not isinstance(contrast_pcts, (list, tuple)):
            contrast_pcts = (82.0, 86.0)

        sigma_x_ratio = float(getattr(
            config, "PHOTO_WINDOW_LOCAL_CONTRAST_SIGMA_X_RATIO", 0.022
        ))
        sigma_y_ratio = float(getattr(
            config, "PHOTO_WINDOW_LOCAL_CONTRAST_SIGMA_Y_RATIO", 0.030
        ))
        std_min = float(getattr(
            config, "PHOTO_WINDOW_LOCAL_CONTRAST_STD_MIN", 14.0
        ))
        max_bright = float(getattr(config, "PHOTO_WINDOW_MAX_BRIGHT_MEAN", 185.0))

        img_h, _ = image.shape[:2]
        for facade in facades:
            fx, fy, fw, fh = facade.bbox
            search_top = max(0, fy - int(round(fh * 0.60)))
            search_bottom = min(img_h, fy + fh + int(round(fh * 0.05)))
            roi = image[search_top:search_bottom, fx:fx + fw]
            if roi.size == 0:
                continue

            gray_raw = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray_raw)
            facade_area = max(1.0, float(fw * fh))

            sigma_x = max(3.0, float(fw) * sigma_x_ratio)
            sigma_y = max(3.0, float(fh) * sigma_y_ratio)
            blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
            local_contrast = cv2.subtract(blur, gray)

            for pct in contrast_pcts:
                c_thr = float(np.percentile(local_contrast, float(pct)))
                c_mask = (local_contrast >= c_thr).astype(np.uint8) * 255
                c_mask = self._cleanup_mask(
                    c_mask, close_kernel=(7, 7), open_kernel=(3, 3), dilate_iterations=0
                )
                contours_c, _ = cv2.findContours(
                    c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for c in contours_c:
                    area = cv2.contourArea(c)
                    x, y, w_box, h_box = cv2.boundingRect(c)
                    rect_area = max(1.0, float(w_box * h_box))
                    area_ratio = rect_area / facade_area
                    ratio = w_box / h_box if h_box else 0.0
                    global_y = search_top + y
                    y_ratio = ((global_y + h_box * 0.5) - fy) / max(1, fh)
                    bottom_ratio = ((global_y + h_box) - fy) / max(1, fh)

                    if area < max(180.0, facade_area * 0.0022):
                        continue
                    if not self._photo_window_width_ok(fw, fh, w_box, h_box):
                        continue
                    if h_box < fh * 0.10:
                        continue
                    if w_box > fw * 0.34 or h_box > fh * 0.58:
                        continue
                    if ratio < 0.35 or ratio > 2.6:
                        continue
                    if y_ratio < -0.20 or y_ratio > 0.88:
                        continue
                    if bottom_ratio < 0.04:
                        continue
                    if y_ratio < 0.12 and h_box < int(fh * 0.22):
                        continue
                    if x < fw * 0.03 or (x + w_box) > fw * 0.97:
                        continue
                    if area_ratio > 0.09:
                        continue
                    if bottom_ratio >= 0.80 and h_box >= int(fh * 0.16) and ratio <= 1.80:
                        continue

                    win_gray = gray[y:y + h_box, x:x + w_box]
                    if win_gray.size == 0:
                        continue
                    std_val = float(np.std(win_gray))
                    mean_val = float(np.mean(win_gray))
                    if std_val < std_min:
                        continue
                    if mean_val > (max_bright + 12.0) and std_val < (std_min * 1.25):
                        continue

                    regions.append(DetectedRegion(
                        label=f"F_PHOTO_{idx}",
                        region_type="window",
                        bbox=(fx + x, int(global_y), int(w_box), int(h_box)),
                        contour=np.array([]),
                        area_px=float(rect_area),
                        color_detected="photo-local-contrast-window",
                    ))
                    idx += 1

        return self._nms_regions(regions, iou_threshold=0.30)

    def detect_photo_doors(self, image: np.ndarray, facades: list, windows: Optional[list] = None) -> list:
        """Detect door-like openings in raw facade photos."""
        if not facades:
            return []

        windows = windows or []
        regions = []
        idx = 1
        bright_pcts = getattr(config, "PHOTO_DOOR_BRIGHT_PERCENTILES", (72.0, 78.0))
        if not isinstance(bright_pcts, (list, tuple)):
            bright_pcts = (72.0, 78.0)

        for facade in facades:
            fx, fy, fw, fh = facade.bbox
            roi = image[fy:fy + fh, fx:fx + fw]
            if roi.size == 0:
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            facade_area = max(1.0, float(fw * fh))

            masks = []
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            grad = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
            g_thr = float(np.percentile(grad, 86.0))
            g_mask = (grad >= g_thr).astype(np.uint8) * 255
            g_mask = self._cleanup_mask(g_mask, close_kernel=(5, 9), open_kernel=(3, 3), dilate_iterations=0)
            masks.append(("gradient", g_mask))

            for pct in bright_pcts:
                b_thr = float(np.percentile(gray, float(pct)))
                b_mask = (gray >= b_thr).astype(np.uint8) * 255
                b_mask = self._cleanup_mask(b_mask, close_kernel=(5, 9), open_kernel=(3, 3), dilate_iterations=0)
                masks.append((f"bright-{pct}", b_mask))

            sigma_x = max(3.0, float(fw) * 0.020)
            sigma_y = max(3.0, float(fh) * 0.030)
            local_blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
            local_contrast = cv2.subtract(local_blur, gray)
            for pct in (88.0, 90.0):
                c_thr = float(np.percentile(local_contrast, pct))
                c_mask = (local_contrast >= c_thr).astype(np.uint8) * 255
                c_mask = self._cleanup_mask(c_mask, close_kernel=(5, 9), open_kernel=(3, 3), dilate_iterations=0)
                masks.append((f"local-contrast-{pct}", c_mask))

            for source, mask in masks:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    x, y, w_box, h_box = cv2.boundingRect(c)
                    is_local = "local-contrast" in source
                    min_w_ratio = getattr(config, "PHOTO_DOOR_MIN_WIDTH_RATIO", 0.05)
                    min_h_ratio = getattr(config, "PHOTO_DOOR_MIN_HEIGHT_RATIO", 0.20)
                    min_aspect = getattr(config, "PHOTO_DOOR_MIN_ASPECT", 0.28)
                    min_bottom_ratio = getattr(config, "PHOTO_DOOR_MIN_BOTTOM_RATIO", 0.72)
                    min_y_ratio = getattr(config, "PHOTO_DOOR_MIN_Y_RATIO", 0.42)
                    min_lap = float(getattr(config, "PHOTO_DOOR_MIN_LAPLACIAN", 2500.0))
                    min_std = float(getattr(config, "PHOTO_DOOR_MIN_STD", 18.0))
                    if is_local:
                        min_w_ratio = min(min_w_ratio, 0.03)
                        min_h_ratio = min(min_h_ratio, 0.24)
                        min_aspect = min(min_aspect, 0.18)
                        min_bottom_ratio = min(min_bottom_ratio, 0.58)
                        min_y_ratio = min(min_y_ratio, 0.28)
                        min_lap *= 0.55
                        min_std *= 0.85
                        min_local_top = int(fh * 0.22)
                        if y < min_local_top:
                            trim = min_local_top - y
                            y = min_local_top
                            h_box -= trim
                            if h_box <= 0:
                                continue
                    ratio = w_box / h_box if h_box else 0
                    cy_ratio = (y + h_box * 0.5) / max(1, fh)
                    bottom_ratio = (y + h_box) / max(1, fh)
                    area_ratio = (w_box * h_box) / facade_area
                    if w_box < fw * min_w_ratio:
                        continue
                    if w_box > fw * getattr(config, "PHOTO_DOOR_MAX_WIDTH_RATIO", 0.28):
                        continue
                    if h_box < fh * min_h_ratio:
                        continue
                    if h_box > fh * getattr(config, "PHOTO_DOOR_MAX_HEIGHT_RATIO", 0.82):
                        continue
                    if ratio < min_aspect:
                        continue
                    if ratio > getattr(config, "PHOTO_DOOR_MAX_ASPECT", 1.25):
                        continue
                    if cy_ratio < min_y_ratio:
                        continue
                    if cy_ratio > getattr(config, "PHOTO_DOOR_MAX_Y_RATIO", 0.94):
                        continue
                    if bottom_ratio < min_bottom_ratio:
                        continue
                    if area_ratio < 0.008 or area_ratio > 0.18:
                        continue
                    if x < fw * 0.03 or (x + w_box) > fw * 0.97:
                        continue

                    roi_gray = gray[y:y + h_box, x:x + w_box]
                    if roi_gray.size == 0:
                        continue
                    lap = float(cv2.Laplacian(roi_gray, cv2.CV_32F).var())
                    std_val = float(np.std(roi_gray))
                    if lap < min_lap:
                        continue
                    if std_val < min_std:
                        continue

                    global_bbox = (fx + x, fy + y, int(w_box), int(h_box))
                    # Avoid re-detecting obvious windows.
                    if any(self._bbox_iou(global_bbox, w.bbox) > 0.28 for w in windows):
                        continue

                    score = self._score_promoted_door_candidate(
                        facade.bbox, global_bbox, float(w_box * h_box), parts_count=1
                    )
                    min_score = 0.28 if is_local else 0.34
                    if score < min_score:
                        continue

                    regions.append(DetectedRegion(
                        label=f"U_PHOTO_{idx}",
                        region_type="door",
                        bbox=global_bbox,
                        contour=np.array([]),
                        area_px=float(w_box * h_box),
                        color_detected=("photo-door-local-contrast" if is_local else "photo-door"),
                    ))
                    idx += 1

        regions = self._nms_regions(regions, iou_threshold=0.25)
        regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        for i, region in enumerate(regions, 1):
            region.label = f"U_PHOTO_{i}"
        return regions

    def _refine_photo_openings(self, image: np.ndarray, facades: list, windows: list, doors: list) -> tuple:
        """Merge fragmented door parts and suppress obvious low false windows."""
        if not facades:
            return windows, doors

        refined_windows = list(windows)
        refined_doors = list(doors)

        # Add supplemental bright-opening candidates only on very dark-background inputs.
        if self._has_dark_background(image):
            extra_windows = []
            for facade in facades:
                fx, fy, fw, fh = facade.bbox
                roi = image[fy:fy + fh, fx:fx + fw]
                if roi.size == 0:
                    continue
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
                facade_area = max(1.0, float(fw * fh))
                candidate_boxes = []
                for pct in (68.0, 72.0):
                    thr = float(np.percentile(gray, pct))
                    mask = (gray >= thr).astype(np.uint8) * 255
                    mask = cv2.morphologyEx(
                        mask, cv2.MORPH_OPEN,
                        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    )
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    for c in contours:
                        x, y, w_box, h_box = cv2.boundingRect(c)
                        ratio = w_box / h_box if h_box else 0.0
                        cy_ratio = (y + h_box * 0.5) / max(1, fh)
                        bottom_ratio = (y + h_box) / max(1, fh)
                        area_ratio = (w_box * h_box) / facade_area
                        if w_box < fw * 0.04 or w_box > fw * 0.18:
                            continue
                        if h_box < fh * 0.18 or h_box > fh * 0.78:
                            continue
                        if ratio < 0.45 or ratio > 1.75:
                            continue
                        if cy_ratio < 0.22 or cy_ratio > 0.82:
                            continue
                        if bottom_ratio > 0.90:
                            continue
                        if area_ratio < 0.010 or area_ratio > 0.16:
                            continue
                        if x < fw * 0.02 or (x + w_box) > fw * 0.98:
                            continue
                        candidate_boxes.append([x, y, w_box, h_box])

                candidate_boxes.sort(key=lambda b: (b[1], b[0]))
                merged_boxes = []
                for x0, y0, w0, h0 in candidate_boxes:
                    attached = False
                    for box in merged_boxes:
                        x1, y1, w1, h1 = box
                        overlap_y = max(0, min(y0 + h0, y1 + h1) - max(y0, y1))
                        gap_x = max(0, max(x0 - (x1 + w1), x1 - (x0 + w0)))
                        min_h = max(1, min(h0, h1))
                        if (overlap_y / min_h) >= 0.45 and gap_x <= max(18, int(fw * 0.05)):
                            nx = min(x0, x1)
                            ny = min(y0, y1)
                            nx2 = max(x0 + w0, x1 + w1)
                            ny2 = max(y0 + h0, y1 + h1)
                            if (nx2 - nx) <= int(fw * 0.26):
                                box[0], box[1], box[2], box[3] = nx, ny, nx2 - nx, ny2 - ny
                                attached = True
                                break
                    if not attached:
                        merged_boxes.append([x0, y0, w0, h0])

                for x, y, w_box, h_box in merged_boxes:
                    global_bbox = (fx + x, fy + y, int(w_box), int(h_box))
                    if any(self._bbox_iou(global_bbox, d.bbox) > 0.18 for d in refined_doors):
                        continue
                    extra_windows.append(DetectedRegion(
                        label="F_PHOTO_EXTRA",
                        region_type="window",
                        bbox=global_bbox,
                        contour=np.array([]),
                        area_px=float(w_box * h_box),
                        color_detected="photo-dark-bright-window",
                    ))

            if extra_windows:
                refined_windows = self._nms_regions(refined_windows + extra_windows, iou_threshold=0.25)

        # Merge split fragments before promoting a door; on real facades one glazed
        # opening can be split by texture/noise into 2 close boxes.
        refined_windows = self._merge_photo_window_fragments(facades, refined_windows)
        refined_windows = self._refine_right_photo_windows(
            image, facades, refined_windows
        )

        scored_doors = []
        for door in refined_doors:
            score = -1.0
            for facade in facades:
                fx, fy, fw, fh = facade.bbox
                cx, cy = door.center
                if fx <= cx <= fx + fw and fy <= cy <= fy + fh:
                    score = self._score_promoted_door_candidate(
                        facade.bbox, door.bbox, float(door.area_px), parts_count=1
                    )
                    break
            if score >= 0.30:
                scored_doors.append((score, door))
        refined_doors = [door for _, door in scored_doors]
        best_existing_door_score = max((score for score, _ in scored_doors), default=-1.0)

        # 0) Promote the most plausible door-like opening when photo door detection
        #    missed it or only produced a weak candidate.
        if len(refined_windows) >= 2:
            promoted = self._select_promoted_photo_door(facades, refined_windows)
            if promoted and (not refined_doors or promoted["score"] > best_existing_door_score + 0.04):
                absorbed = promoted["windows"]
                refined_windows = [w for w in refined_windows if w not in absorbed]
                if refined_doors and best_existing_door_score < promoted["score"]:
                    refined_doors = []
                refined_doors.append(DetectedRegion(
                    label="U_PHOTO_1",
                    region_type="door",
                    bbox=promoted["bbox"],
                    contour=np.array([]),
                    area_px=promoted["area_px"],
                    color_detected="photo-door-promoted",
                ))
        refined_doors = self._refine_photo_door_geometry(
            image, facades, refined_doors, refined_windows
        )
        refined_doors = self._recover_composite_gap_doors(
            image, facades, refined_windows, refined_doors
        )

        # 1) Absorb nearby upper fragments into doors.
        for door in refined_doors:
            dx, dy, dw, dh = door.bbox
            parent = None
            for facade in facades:
                fx, fy, fw, fh = facade.bbox
                cx, cy = door.center
                if fx <= cx <= fx + fw and fy <= cy <= fy + fh:
                    parent = facade
                    break
            if parent is None:
                continue
            fx, fy, fw, fh = parent.bbox
            if "promoted" in (door.color_detected or ""):
                continue
            absorb = []
            for win in refined_windows:
                wx, wy, ww, wh = win.bbox
                gap_x = max(0, max(wx - (dx + dw), dx - (wx + ww)))
                overlap_x = max(0, min(wx + ww, dx + dw) - max(wx, dx))
                max_gap = max(20, min(40, int(max(dw, ww) * 0.30)))
                if gap_x > max_gap and overlap_x <= 0:
                    continue
                if (wy + wh) > dy + int(dh * 0.65):
                    continue
                if wy < fy + int(fh * 0.05):
                    continue
                if ww > int(fw * 0.20) or wh > int(fh * 0.45):
                    continue
                absorb.append(win)

            if not absorb:
                continue

            xs = [dx, dx + dw]
            ys = [dy, dy + dh]
            total_area = float(door.area_px)
            for win in absorb:
                wx, wy, ww, wh = win.bbox
                xs.extend([wx, wx + ww])
                ys.extend([wy, wy + wh])
                total_area += float(win.area_px)
            nx1, nx2 = min(xs), max(xs)
            ny1, ny2 = min(ys), max(ys)
            if (nx2 - nx1) <= int(fw * 0.34) and (ny2 - ny1) <= int(fh * 0.86):
                door.bbox = (nx1, ny1, nx2 - nx1, ny2 - ny1)
                door.area_px = total_area
                refined_windows = [w for w in refined_windows if w not in absorb]

        refined_windows = self._recover_photo_door_sidelights(
            image, facades, refined_windows, refined_doors
        )
        refined_windows, refined_doors = self._refine_composite_opening_clusters(
            facades, refined_windows, refined_doors
        )
        refined_windows, refined_doors = self._align_composite_photo_door_clusters(
            image, facades, refined_windows, refined_doors
        )
        refined_windows, refined_doors = self._regularize_composite_opening_geometry(
            image, facades, refined_windows, refined_doors
        )
        refined_windows, refined_doors = self._align_flat_photo_central_clusters(
            image, facades, refined_windows, refined_doors
        )

        # 2) Remove low/small window fragments that are not plausible final openings.
        filtered_windows = []
        for win in refined_windows:
            parent = None
            for facade in facades:
                fx, fy, fw, fh = facade.bbox
                cx, cy = win.center
                if fx <= cx <= fx + fw and fy <= cy <= fy + fh:
                    parent = facade
                    break
            if parent is None:
                filtered_windows.append(win)
                continue
            fx, fy, fw, fh = parent.bbox
            x, y, w_box, h_box = win.bbox
            bottom_ratio = (y + h_box - fy) / max(1.0, float(fh))
            ratio = w_box / h_box if h_box else 0.0
            if bottom_ratio > 0.78 and h_box < int(fh * 0.22):
                continue
            if w_box < int(fw * 0.045) and h_box < int(fh * 0.22):
                continue
            if (x + w_box) > (fx + fw - int(fw * 0.05)) and w_box < int(fw * 0.07):
                continue
            if bottom_ratio > 0.64 and h_box < int(fh * 0.24) and w_box < int(fw * 0.07) and ratio < 1.45:
                continue
            if ratio > 1.9 and bottom_ratio > 0.65:
                continue
            filtered_windows.append(win)

        filtered_windows = self._nms_regions(filtered_windows, iou_threshold=0.25)
        refined_doors = self._nms_regions(refined_doors, iou_threshold=0.25)
        for i, region in enumerate(filtered_windows, 1):
            region.label = f"F_PHOTO_{i}"
        for i, region in enumerate(refined_doors, 1):
            region.label = f"U_PHOTO_{i}"
        return filtered_windows, refined_doors

    def _recover_composite_gap_doors(self, image: np.ndarray,
                                     facades: list, windows: list, doors: list) -> list:
        """Recover extra lower doors from large gaps inside merged/composite facades."""
        if image is None or image.size == 0 or not facades or len(windows) < 2:
            return doors

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        refined_doors = list(doors)
        existing = list(windows) + list(refined_doors)
        added_idx = 0

        for facade in facades:
            if 'merged' not in (facade.color_detected or '').lower():
                continue
            fx, fy, fw, fh = facade.bbox
            lower_openings = []
            for reg in existing:
                cx, cy = reg.center
                if not (fx <= cx <= fx + fw and fy <= cy <= fy + fh):
                    continue
                cy_ratio = (cy - fy) / max(1.0, float(fh))
                if cy_ratio < 0.34 or cy_ratio > 0.88:
                    continue
                lower_openings.append(reg)
            if len(lower_openings) < 2:
                continue

            lower_openings.sort(key=lambda r: r.bbox[0])
            for left_reg, right_reg in zip(lower_openings, lower_openings[1:]):
                lx, ly, lw, lh = left_reg.bbox
                rx, ry, rw, rh = right_reg.bbox
                gap_x0 = lx + lw
                gap_x1 = rx
                gap_w = gap_x1 - gap_x0
                if gap_w < max(90, int(fw * 0.05)):
                    continue
                if gap_w > int(fw * 0.24):
                    continue

                sx = max(fx, gap_x0 - int(gap_w * 0.10))
                ex = min(fx + fw, gap_x1 + int(gap_w * 0.10))
                sy = max(fy + int(fh * 0.38), min(ly, ry) - int(fh * 0.04))
                ey = min(fy + fh, max(ly + lh, ry + rh) + int(fh * 0.08))
                if ex - sx < 42 or ey - sy < 90:
                    continue

                roi = gray[sy:ey, sx:ex]
                blur = cv2.GaussianBlur(
                    roi, (0, 0),
                    sigmaX=max(5.0, float(ex - sx) * 0.10),
                    sigmaY=max(5.0, float(ey - sy) * 0.12),
                )
                dark = cv2.subtract(blur, roi)
                thr = float(np.percentile(dark, 86.0))
                mask = (dark >= thr).astype(np.uint8) * 255
                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (5, 11)),
                )
                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_OPEN,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)),
                )

                n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
                best_bbox = None
                best_score = -1.0
                gap_center = (gap_x0 + gap_x1) * 0.5
                tall_parts = []
                for idx in range(1, n_labels):
                    x, y, w_box, h_box, area = stats[idx]
                    if area < 500:
                        continue
                    gx, gy = sx + int(x), sy + int(y)
                    bbox = (gx, gy, int(w_box), int(h_box))
                    if any(self._bbox_iou(bbox, reg.bbox) > 0.10 for reg in existing):
                        continue
                    width_ratio = w_box / max(1.0, float(fw))
                    height_ratio = h_box / max(1.0, float(fh))
                    aspect = w_box / max(1.0, float(h_box))
                    bottom_ratio = (gy + h_box - fy) / max(1.0, float(fh))
                    if (
                        0.012 <= width_ratio <= 0.030 and
                        0.16 <= height_ratio <= 0.40 and
                        0.20 <= aspect <= 0.55 and
                        bottom_ratio >= 0.62
                    ):
                        tall_parts.append((bbox, float(area)))
                    if not (0.03 <= width_ratio <= 0.11):
                        continue
                    if not (0.18 <= height_ratio <= 0.58):
                        continue
                    if not (0.22 <= aspect <= 0.92):
                        continue
                    if bottom_ratio < 0.70:
                        continue
                    score = self._score_promoted_door_candidate(
                        facade.bbox, bbox, float(area), parts_count=1
                    )
                    if score < 0.32:
                        continue
                    gap_bonus = 1.0 - min(1.0, abs((gx + w_box * 0.5) - gap_center) / max(24.0, gap_w * 0.5))
                    score += gap_bonus * 0.12
                    if score > best_score:
                        best_score = score
                        best_bbox = bbox

                if best_bbox is None and len(tall_parts) >= 3:
                    tall_parts.sort(key=lambda item: item[0][0])
                    best_triplet = None
                    best_triplet_score = -1.0
                    for i in range(len(tall_parts) - 2):
                        triplet = tall_parts[i:i + 3]
                        xs = [b[0][0] + b[0][2] * 0.5 for b in triplet]
                        ys = [b[0][1] for b in triplet]
                        hs = [b[0][3] for b in triplet]
                        spread = max(xs) - min(xs)
                        if spread > gap_w * 0.90:
                            continue
                        if max(hs) > min(hs) * 1.70:
                            continue
                        triplet_center = sum(xs) / 3.0
                        score = 1.0 - min(1.0, abs(triplet_center - gap_center) / max(24.0, gap_w * 0.5))
                        if score > best_triplet_score:
                            best_triplet_score = score
                            best_triplet = triplet
                    if best_triplet is not None:
                        mid_bbox = best_triplet[1][0]
                        mx, my, mw, mh = mid_bbox
                        top_y = min(item[0][1] for item in best_triplet)
                        bottom_y = max(item[0][1] + item[0][3] for item in best_triplet)
                        target_w = max(int(round(mw * 1.55)), int(round(fw * 0.045)))
                        target_h = max(bottom_y - top_y, int(round(fh * 0.24)))
                        nx = int(round((mx + mw * 0.5) - target_w * 0.5))
                        nx = max(fx, min(nx, fx + fw - target_w))
                        best_bbox = (nx, top_y, target_w, target_h)

                if best_bbox is None:
                    continue

                best_bbox = self._complete_photo_door_hole_bbox(gray, facade.bbox, best_bbox)
                best_bbox = self._extend_door_by_jamb_continuity(gray, facade.bbox, best_bbox)
                if any(self._bbox_iou(best_bbox, reg.bbox) > 0.14 for reg in existing):
                    continue

                added_idx += 1
                door = DetectedRegion(
                    label=f'U_PHOTO_{len(refined_doors) + 1}',
                    region_type='door',
                    bbox=best_bbox,
                    contour=np.array([]),
                    area_px=float(best_bbox[2] * best_bbox[3]),
                    color_detected='photo-door-composite-gap',
                )
                refined_doors.append(door)
                existing.append(door)

        return self._nms_regions(refined_doors, iou_threshold=0.22)

    def _recover_photo_door_sidelights(self, image: np.ndarray,
                                      facades: list, windows: list, doors: list) -> list:
        """Recover narrow side windows adjacent to a detected photo door."""
        if image is None or image.size == 0 or not facades or not doors:
            return windows

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        working_windows = list(windows)
        extras = []

        for door in doors:
            dx, dy, dw, dh = door.bbox
            parent = None
            for facade in facades:
                fx, fy, fw, fh = facade.bbox
                cx, cy = door.center
                if fx <= cx <= fx + fw and fy <= cy <= fy + fh:
                    parent = facade
                    break
            if parent is None:
                continue

            fx, fy, fw, fh = parent.bbox
            side_hits = {}
            for side in ("left", "right"):
                if side == "left":
                    sx = max(fx, dx - int(dw * 2.2))
                    ex = max(sx + 28, dx - int(dw * 0.10))
                else:
                    sx = min(dx + dw + int(dw * 0.10), fx + fw - 28)
                    ex = min(fx + fw, dx + dw + int(dw * 1.9))
                sy = max(fy, dy - int(dh * 0.30))
                ey = min(fy + fh, dy + int(dh * 0.38))
                if ex - sx < 24 or ey - sy < 48:
                    continue

                roi = gray[sy:ey, sx:ex]
                grad = (
                    np.abs(cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)) * 1.1 +
                    np.abs(cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)) * 0.6
                )
                best = None
                best_score = 0.0
                best_replace = None
                search_h = max(1, ey - sy)
                top_guard = int(round(search_h * 0.18))

                for pct in (82.0, 84.0):
                    thr = float(np.percentile(grad, pct))
                    mask = (grad >= thr).astype(np.uint8) * 255
                    mask = cv2.morphologyEx(
                        mask, cv2.MORPH_CLOSE,
                        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 7))
                    )
                    mask = cv2.morphologyEx(
                        mask, cv2.MORPH_OPEN,
                        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    )
                    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
                    for idx in range(1, n_labels):
                        x, y, w_box, h_box, area = stats[idx]
                        ratio = w_box / max(1.0, float(h_box))
                        if area < 220:
                            continue
                        if y < top_guard:
                            continue
                        if w_box < max(26, int(dw * 0.18)) or w_box > int(dw * 1.25):
                            continue
                        if h_box < int(dh * 0.30) or h_box > int(dh * 0.72):
                            continue
                        if ratio < 0.14 or ratio > 1.20:
                            continue

                        bbox = (sx + int(x), sy + int(y), int(w_box), int(h_box))
                        if self._bbox_iou(bbox, door.bbox) > 0.12:
                            continue

                        replace_win = None
                        blocked = False
                        contain_thr = float(
                            getattr(config, "PHOTO_DOOR_SIDELIGHT_SUPPRESS_CONTAIN_RATIO", 0.48)
                        )
                        top_tol = max(8, int(round(h_box * 0.14)))
                        cand_area = float(w_box * h_box)
                        for win in working_windows:
                            iou = self._bbox_iou(bbox, win.bbox)
                            if iou <= 0.20:
                                continue
                            wx, wy, ww, wh = win.bbox
                            inter_x1 = max(bbox[0], wx)
                            inter_y1 = max(bbox[1], wy)
                            inter_x2 = min(bbox[0] + bbox[2], wx + ww)
                            inter_y2 = min(bbox[1] + bbox[3], wy + wh)
                            inter = float(max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1))
                            contain = inter / max(1.0, min(cand_area, float(ww * wh)))
                            if contain >= contain_thr and abs(bbox[1] - wy) <= top_tol and ww >= int(round(w_box * 1.20)):
                                replace_win = win
                                continue
                            blocked = True
                            break
                        if blocked:
                            continue

                        gap_x = max(0, max(bbox[0] - (dx + dw), dx - (bbox[0] + bbox[2])))
                        if gap_x > max(36, int(dw * 0.95)):
                            continue
                        cy_ratio = ((bbox[1] + bbox[3] * 0.5) - fy) / max(1.0, float(fh))
                        if cy_ratio < 0.18 or cy_ratio > 0.76:
                            continue

                        score = float(area) / max(1.0, abs(gap_x) + 18.0)
                        if side == "left" and (bbox[0] + bbox[2]) <= dx:
                            score *= 1.10
                        if side == "right" and bbox[0] >= (dx + dw):
                            score *= 1.10
                        if replace_win is not None:
                            score *= 1.08
                        if score > best_score:
                            best = bbox
                            best_score = score
                            best_replace = replace_win

                if best is None:
                    continue
                if best_replace is not None and best_replace in working_windows:
                    working_windows.remove(best_replace)
                bx, by, bw_box, bh_box = best
                side_hits[side] = (bx, by, bw_box, bh_box)
                extras.append(DetectedRegion(
                    label="F_PHOTO_SIDE",
                    region_type="window",
                    bbox=(bx, by, bw_box, bh_box),
                    contour=np.array([]),
                    area_px=float(bw_box * bh_box),
                    color_detected="photo-door-sidelight",
                ))

            if len(side_hits) == 1:
                hit_side, ref_bbox = next(iter(side_hits.items()))
                bx, by, bw_box, bh_box = ref_bbox
                if hit_side == "right":
                    gap = max(0, bx - (dx + dw))
                    mx = dx - gap - bw_box
                else:
                    gap = max(0, dx - (bx + bw_box))
                    mx = dx + dw + gap
                my = by
                mx = max(fx, min(int(mx), fx + fw - bw_box))
                mirror_bbox = (int(mx), int(my), int(bw_box), int(bh_box))

                overlap_blocked = False
                for win in working_windows + extras:
                    if self._bbox_iou(mirror_bbox, win.bbox) > 0.20:
                        overlap_blocked = True
                        break
                if not overlap_blocked and self._bbox_iou(mirror_bbox, door.bbox) <= 0.12:
                    rx, ry, rw, rh = mirror_bbox
                    if rw >= max(26, int(dw * 0.18)) and rh >= int(dh * 0.30):
                        roi = gray[ry:ry + rh, rx:rx + rw]
                        if roi.size > 0:
                            std_val = float(np.std(roi))
                            lap = float(cv2.Laplacian(roi, cv2.CV_32F).var())
                            blur = cv2.GaussianBlur(
                                roi, (0, 0),
                                sigmaX=max(3.0, float(rw) * 0.18),
                                sigmaY=max(3.0, float(rh) * 0.12),
                            )
                            dark = cv2.subtract(blur, roi)
                            dark_p85 = float(np.percentile(dark, 85))
                            mean_val = float(np.mean(roi))
                            if std_val >= 42.0 and lap >= 18000.0 and dark_p85 >= 48.0 and mean_val <= 185.0:
                                extras.append(DetectedRegion(
                                    label="F_PHOTO_SIDE",
                                    region_type="window",
                                    bbox=mirror_bbox,
                                    contour=np.array([]),
                                    area_px=float(rw * rh),
                                    color_detected="photo-door-sidelight-mirror",
                                ))

        if not extras and len(working_windows) == len(windows):
            return windows
        return self._nms_regions(working_windows + extras, iou_threshold=0.25)


    def _refine_composite_opening_clusters(self, facades: list,
                                           windows: list, doors: list) -> tuple:
        """Refine merged/composite opening clusters without touching flat-facade logic."""
        if not facades or not windows or not doors:
            return windows, doors

        refined_windows = list(windows)
        refined_doors = list(doors)

        def _parent_facade(region):
            cx, cy = region.center
            for facade in facades:
                fx, fy, fw, fh = facade.bbox
                if fx <= cx <= fx + fw and fy <= cy <= fy + fh:
                    return facade
            return None

        survivors = []
        for win in refined_windows:
            parent = _parent_facade(win)
            if parent is None or 'merged' not in (parent.color_detected or '').lower():
                survivors.append(win)
                continue

            fx, fy, fw, fh = parent.bbox
            wx, wy, ww, wh = win.bbox
            drop = False
            if 'sidelight' not in (win.color_detected or '').lower():
                for door in refined_doors:
                    if _parent_facade(door) is not parent:
                        continue
                    dx, dy, dw, dh = door.bbox
                    overlap_x = max(0, min(wx + ww, dx + dw) - max(wx, dx))
                    top_gap = dy - (wy + wh)
                    win_cy_ratio = ((wy + wh * 0.5) - fy) / max(1.0, float(fh))
                    if overlap_x < min(ww, dw) * 0.32:
                        continue
                    if not (-int(dh * 0.40) <= top_gap <= int(fh * 0.10)):
                        continue
                    if ww < max(int(dw * 1.20), int(fw * 0.06)):
                        continue
                    if win_cy_ratio < 0.22:
                        continue
                    drop = True
                    break
            if not drop:
                survivors.append(win)
        refined_windows = survivors

        extras = []
        for facade in facades:
            if 'merged' not in (facade.color_detected or '').lower():
                continue
            fx, fy, fw, fh = facade.bbox
            facade_doors = []
            for door in refined_doors:
                cx, cy = door.center
                if fx <= cx <= fx + fw and fy <= cy <= fy + fh:
                    facade_doors.append(door)
            if not facade_doors:
                continue

            local_windows = []
            for win in refined_windows + extras:
                cx, cy = win.center
                if fx <= cx <= fx + fw and fy <= cy <= fy + fh:
                    local_windows.append(win)

            for door in sorted(facade_doors, key=lambda r: r.bbox[0]):
                dx, dy, dw, dh = door.bbox
                bottom_ratio = (dy + dh - fy) / max(1.0, float(fh))
                if bottom_ratio < 0.64:
                    continue

                side_hits = {}
                for win in local_windows:
                    wx, wy, ww, wh = win.bbox
                    if ww > max(60, int(dw * 0.65)):
                        continue
                    overlap_y = max(0, min(dy + dh, wy + wh) - max(dy, wy))
                    gap_x = max(0, max(wx - (dx + dw), dx - (wx + ww)))
                    if overlap_y < min(wh, dh) * 0.25:
                        continue
                    if gap_x > max(42, int(dw * 0.75)):
                        continue
                    if wx + ww <= dx:
                        prev = side_hits.get('left')
                        if prev is None or gap_x < prev[0]:
                            side_hits['left'] = (gap_x, win.bbox)
                    elif wx >= dx + dw:
                        prev = side_hits.get('right')
                        if prev is None or gap_x < prev[0]:
                            side_hits['right'] = (gap_x, win.bbox)

                if len(side_hits) != 1:
                    continue

                hit_side, payload = next(iter(side_hits.items()))
                gap_ref, ref_bbox = payload
                bx, by, bw_box, bh_box = ref_bbox
                if hit_side == 'right':
                    mx = dx - gap_ref - bw_box
                else:
                    mx = dx + dw + gap_ref
                my = by
                mirror_bbox = (int(mx), int(my), int(bw_box), int(bh_box))
                if mirror_bbox[0] < fx or (mirror_bbox[0] + mirror_bbox[2]) > (fx + fw):
                    continue

                blocked = False
                for reg in local_windows + facade_doors + extras:
                    if self._bbox_iou(mirror_bbox, reg.bbox) > 0.12:
                        blocked = True
                        break
                if blocked or self._bbox_iou(mirror_bbox, door.bbox) > 0.12:
                    continue

                extras.append(DetectedRegion(
                    label='F_PHOTO_SIDE',
                    region_type='window',
                    bbox=mirror_bbox,
                    contour=np.array([]),
                    area_px=float(mirror_bbox[2] * mirror_bbox[3]),
                    color_detected='photo-door-sidelight-composite-mirror',
                ))
                local_windows.append(extras[-1])

        if extras:
            refined_windows = self._nms_regions(refined_windows + extras, iou_threshold=0.22)
        return refined_windows, refined_doors

    def _align_composite_photo_door_clusters(self, image: np.ndarray,
                                             facades: list, windows: list, doors: list) -> tuple:
        """Split merged-facade glazed door clusters into sidelight + door + sidelight."""
        if image is None or image.size == 0 or not facades or not windows or not doors:
            return windows, doors

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        refined_windows = list(windows)
        refined_doors = list(doors)

        for d_idx, door in enumerate(list(refined_doors)):
            parent = None
            for facade in facades:
                fx, fy, fw, fh = facade.bbox
                cx, cy = door.center
                if fx <= cx <= fx + fw and fy <= cy <= fy + fh:
                    parent = facade
                    break
            if parent is None or 'merged' not in (parent.color_detected or '').lower():
                continue

            fx, fy, fw, fh = parent.bbox
            dx, dy, dw, dh = door.bbox
            bottom_ratio = (dy + dh - fy) / max(1.0, float(fh))
            if bottom_ratio < 0.62:
                continue

            near = []
            for win in refined_windows:
                wx, wy, ww, wh = win.bbox
                overlap_y = max(0, min(dy + dh, wy + wh) - max(dy, wy))
                gap_x = max(0, max(wx - (dx + dw), dx - (wx + ww)))
                if overlap_y < min(dh, wh) * 0.12 and abs((wy + wh * 0.5) - (dy + dh * 0.5)) > int(fh * 0.16):
                    continue
                if gap_x > max(150, int(dw * 2.3)):
                    continue
                if ww <= max(120, int(dw * 1.15)) or 'sidelight' in (win.color_detected or ''):
                    near.append(win)
            left_near = [win for win in near if (win.bbox[0] + win.bbox[2]) <= dx + max(10, int(dw * 0.10))]
            right_near = [win for win in near if win.bbox[0] >= (dx + dw - max(10, int(dw * 0.10)))]
            if not near or not left_near or not right_near:
                continue
            cluster_span = (
                max(dx + dw, max(win.bbox[0] + win.bbox[2] for win in near)) -
                min(dx, min(win.bbox[0] for win in near))
            )
            if cluster_span > max(260, int(dw * 2.8)):
                continue

            sx = max(fx, min([dx] + [w.bbox[0] for w in near]) - int(dw * 1.7))
            ex = min(fx + fw, max([dx + dw] + [w.bbox[0] + w.bbox[2] for w in near]) + int(dw * 1.7))
            sy = max(fy, min([dy] + [w.bbox[1] for w in near]) - int(dh * 0.18))
            ey = min(fy + fh, max(dy + int(dh * 0.82), max([w.bbox[1] + w.bbox[3] for w in near])) + int(dh * 0.10))
            if ex - sx < max(165, int(dw * 2.8)) or ey - sy < max(120, int(dh * 0.55)):
                continue

            roi = gray[sy:ey, sx:ex]
            blur = cv2.GaussianBlur(roi, (0, 0), sigmaX=12, sigmaY=12)
            dark = cv2.subtract(blur, roi)
            mask = (dark >= float(np.percentile(dark, 77.0))).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 11)))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            best_bbox = None
            best_mask = None
            best_score = -1.0
            for idx in range(1, n_labels):
                rx, ry, rw, rh, area = stats[idx]
                if area < 1200:
                    continue
                gx, gy = sx + int(rx), sy + int(ry)
                overlap_x = max(0, min(gx + rw, dx + dw) - max(gx, dx))
                if overlap_x < dw * 0.22:
                    continue
                if rw < dw * 1.7 or rw > dw * 5.6:
                    continue
                if rh < dh * 0.42 or rh > dh * 1.05:
                    continue
                score = float(area) + float(overlap_x * 24) - float(abs((gx + rw * 0.5) - (dx + dw * 0.5)) * 8)
                if score > best_score:
                    best_score = score
                    best_bbox = (int(gx), int(gy), int(rw), int(rh))
                    best_mask = (labels[ry:ry + rh, rx:rx + rw] == idx).astype(np.uint8) * 255

            if best_bbox is None or best_mask is None:
                continue

            bx, by, bw, bh = best_bbox
            proj = (best_mask > 0).sum(axis=0).astype(np.float32)
            proj = cv2.GaussianBlur(proj.reshape(1, -1), (1, 17), 0).ravel()
            left_start, left_end = int(bw * 0.12), int(bw * 0.44)
            right_start, right_end = int(bw * 0.56), int(bw * 0.88)
            if left_end <= left_start + 4 or right_end <= right_start + 4:
                continue
            left_sep = left_start + int(np.argmin(proj[left_start:left_end]))
            right_sep = right_start + int(np.argmin(proj[right_start:right_end]))
            if left_sep < max(22, int(dw * 0.16)) or right_sep <= left_sep + max(48, int(dw * 0.42)):
                continue

            segments = [('left', 0, left_sep), ('door', left_sep, right_sep), ('right', right_sep, bw)]
            seg_boxes = {}
            for name, s0, s1 in segments:
                if s1 <= s0 + 8:
                    continue
                sub = best_mask[:, s0:s1]
                cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    continue
                c = max(cnts, key=cv2.contourArea)
                sx0, sy0, sw0, sh0 = cv2.boundingRect(c)
                seg_boxes[name] = (bx + s0 + sx0, by + sy0, sw0, sh0)

            if 'left' not in seg_boxes or 'door' not in seg_boxes or 'right' not in seg_boxes:
                continue

            lbox = seg_boxes['left']
            mbox = seg_boxes['door']
            rbox = seg_boxes['right']
            if lbox[2] < max(22, int(dw * 0.18)) or rbox[2] < max(22, int(dw * 0.18)):
                continue

            survivors = []
            cluster_x0 = min(lbox[0], dx) - 12
            cluster_x1 = max(rbox[0] + rbox[2], dx + dw) + 12
            for win in refined_windows:
                wx, wy, ww, wh = win.bbox
                if wx + ww >= cluster_x0 and wx <= cluster_x1 and (wy + wh) >= by and wy <= (by + bh + 30):
                    continue
                survivors.append(win)
            refined_windows = survivors

            refined_windows.append(DetectedRegion(
                label='F_PHOTO_SIDE_L',
                region_type='window',
                bbox=lbox,
                contour=np.array([]),
                area_px=float(lbox[2] * lbox[3]),
                color_detected='photo-door-sidelight-composite-split',
            ))
            refined_windows.append(DetectedRegion(
                label='F_PHOTO_SIDE_R',
                region_type='window',
                bbox=rbox,
                contour=np.array([]),
                area_px=float(rbox[2] * rbox[3]),
                color_detected='photo-door-sidelight-composite-split',
            ))

            mx, my, mw, mh = mbox
            target_w = min(
                max(int(round(mw * 0.96)), dw + 10),
                max(dw + 24, int(round(mw * 1.04)))
            )
            target_w = min(target_w, max(1, ex - sx))
            target_h = max(
                int(round(mh * 0.96)),
                min(dh, int(round(dh * 0.90)))
            )
            target_h = min(target_h, max(1, fy + fh - my))
            target_cx = mx + mw * 0.5
            nx = int(round(target_cx - target_w * 0.5))
            nx = max(fx, min(nx, fx + fw - target_w))
            ny = int(round(my))
            max_top_shift = max(14, int(round(dh * 0.22)))
            ny = max(dy - max_top_shift, min(ny, dy + max_top_shift))
            refined_doors[d_idx] = self._clone_region_with_bbox(
                door,
                (int(nx), int(ny), int(target_w), int(target_h)),
                color_detected='photo-door-composite-cluster',
            )

        refined_windows = self._nms_regions(refined_windows, iou_threshold=0.24)
        refined_doors = self._nms_regions(refined_doors, iou_threshold=0.22)
        refined_doors.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return refined_windows, refined_doors

    def _regularize_composite_opening_geometry(self, image: np.ndarray,
                                               facades: list, windows: list, doors: list) -> tuple:
        """Regularize merged-facade openings by tightening window frames and fitting doors into real gaps."""
        if image is None or image.size == 0 or not facades:
            return windows, doors

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        refined_windows = list(windows)
        refined_doors = list(doors)

        def _parent_facade(region):
            cx, cy = region.center
            for facade in facades:
                fx, fy, fw, fh = facade.bbox
                if fx <= cx <= fx + fw and fy <= cy <= fy + fh:
                    return facade
            return None

        for idx, win in enumerate(list(refined_windows)):
            parent = _parent_facade(win)
            if parent is None or 'merged' not in (parent.color_detected or '').lower():
                continue
            if 'photo' not in (win.color_detected or ''):
                continue
            _fx, fy, fw, fh = parent.bbox
            _wx, _wy, ww, wh = win.bbox
            cy_ratio = (win.center[1] - fy) / max(1.0, float(fh))
            width_ratio = ww / max(1.0, float(fw))
            height_ratio = wh / max(1.0, float(fh))
            if cy_ratio < 0.50 and width_ratio > 0.08 and height_ratio < 0.27:
                continue
            new_bbox = self._refine_composite_window_frame(gray, parent.bbox, win.bbox)
            if new_bbox and new_bbox != win.bbox:
                refined_windows[idx] = self._clone_region_with_bbox(win, new_bbox)

        for facade in facades:
            if 'merged' not in (facade.color_detected or '').lower():
                continue
            fx, fy, fw, fh = facade.bbox
            local_windows = [
                (idx, win) for idx, win in enumerate(refined_windows)
                if fx <= win.center[0] <= fx + fw and fy <= win.center[1] <= fy + fh
            ]
            if len(local_windows) >= 2:
                rows = self._group_regions_by_row([pair[1] for pair in local_windows], facade.bbox)
                for row in rows:
                    if len(row) < 3:
                        continue
                    tops = np.array([reg.bbox[1] for reg in row], dtype=np.float32)
                    bottoms = np.array([reg.bbox[1] + reg.bbox[3] for reg in row], dtype=np.float32)
                    row_top = int(round(float(np.median(tops))))
                    row_bottom = int(round(float(np.median(bottoms))))
                    row_h = row_bottom - row_top
                    if row_h <= 0:
                        continue
                    top_iqr = float(np.percentile(tops, 75) - np.percentile(tops, 25))
                    bottom_iqr = float(np.percentile(bottoms, 75) - np.percentile(bottoms, 25))
                    if top_iqr > max(28.0, row_h * 0.38) or bottom_iqr > max(30.0, row_h * 0.42):
                        continue
                    row_ids = {id(reg) for reg in row}
                    for idx, win in local_windows:
                        if id(win) not in row_ids:
                            continue
                        x, y, w_box, h_box = win.bbox
                        if abs(y - row_top) > max(28, int(h_box * 0.45)) and abs((y + h_box) - row_bottom) > max(28, int(h_box * 0.45)):
                            continue
                        refined_windows[idx] = self._clone_region_with_bbox(
                            win, (x, row_top, w_box, row_h)
                        )

            local_windows = [
                win for win in refined_windows
                if fx <= win.center[0] <= fx + fw and fy <= win.center[1] <= fy + fh
            ]
            for idx, door in enumerate(list(refined_doors)):
                if not (fx <= door.center[0] <= fx + fw and fy <= door.center[1] <= fy + fh):
                    continue
                candidates = []
                dcx, dcy = door.center
                for win in local_windows:
                    wcx, wcy = win.center
                    if abs(wcy - dcy) > max(120, int(fh * 0.26)):
                        continue
                    if win.bbox[0] + win.bbox[2] <= dcx:
                        gap = dcx - (win.bbox[0] + win.bbox[2])
                        candidates.append(("left", gap, win.bbox))
                    elif win.bbox[0] >= dcx:
                        gap = win.bbox[0] - dcx
                        candidates.append(("right", gap, win.bbox))
                left = None
                right = None
                for side, gap, bbox in sorted(candidates, key=lambda item: item[1]):
                    if side == "left" and left is None:
                        left = bbox
                    elif side == "right" and right is None:
                        right = bbox
                    if left is not None and right is not None:
                        break
                if left is None or right is None:
                    continue
                fitted = self._fit_door_to_window_gap(facade.bbox, door.bbox, left, right)
                if fitted and fitted != door.bbox:
                    refined_doors[idx] = self._clone_region_with_bbox(
                        door, fitted, color_detected='photo-door-gap-fit'
                    )

        refined_windows = self._nms_regions(refined_windows, iou_threshold=0.22)
        refined_doors = self._nms_regions(refined_doors, iou_threshold=0.22)
        refined_windows.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        refined_doors.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return refined_windows, refined_doors

    def _align_flat_photo_central_clusters(self, image: np.ndarray,
                                           facades: list, windows: list, doors: list) -> tuple:
        """Split flat-facade central dark clusters into sidelight + door + sidelight."""
        if image is None or image.size == 0 or not facades or not windows or not doors:
            return windows, doors

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        refined_windows = list(windows)
        refined_doors = list(doors)

        for d_idx, door in enumerate(list(refined_doors)):
            parent = None
            for facade in facades:
                fx, fy, fw, fh = facade.bbox
                cx, cy = door.center
                if fx <= cx <= fx + fw and fy <= cy <= fy + fh:
                    parent = facade
                    break
            if parent is None:
                continue
            if 'merged' in (parent.color_detected or '').lower():
                continue

            dx, dy, dw, dh = door.bbox
            near = []
            for win in refined_windows:
                wx, wy, ww, wh = win.bbox
                overlap_y = max(0, min(dy + dh, wy + wh) - max(dy, wy))
                gap_x = max(0, max(wx - (dx + dw), dx - (wx + ww)))
                if overlap_y < min(dh, wh) * 0.20:
                    continue
                if gap_x > max(90, int(dw * 1.5)):
                    continue
                if 'sidelight' in (win.color_detected or '') or ww <= max(90, int(dw * 1.1)):
                    near.append(win)
            if not near:
                continue

            fx, fy, fw, fh = parent.bbox
            sx = max(fx, min([dx] + [w.bbox[0] for w in near]) - int(dw * 0.9))
            ex = min(fx + fw, max([dx + dw] + [w.bbox[0] + w.bbox[2] for w in near]) + int(dw * 0.9))
            sy = max(fy, min([dy] + [w.bbox[1] for w in near]) - int(dh * 0.10))
            ey = min(fy + fh, max(dy + int(dh * 0.72), max([w.bbox[1] + w.bbox[3] for w in near])) + int(dh * 0.08))
            if ex - sx < max(140, int(dw * 2.2)) or ey - sy < max(120, int(dh * 0.55)):
                continue

            roi = gray[sy:ey, sx:ex]
            blur = cv2.GaussianBlur(roi, (0, 0), sigmaX=10, sigmaY=12)
            dark = cv2.subtract(blur, roi)
            mask = (dark >= float(np.percentile(dark, 78.0))).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9)))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            best_bbox = None
            best_mask = None
            best_score = -1.0
            for idx in range(1, n_labels):
                rx, ry, rw, rh, area = stats[idx]
                if area < 1200:
                    continue
                gx, gy = sx + int(rx), sy + int(ry)
                overlap_x = max(0, min(gx + rw, dx + dw) - max(gx, dx))
                if overlap_x < dw * 0.35:
                    continue
                if rw < dw * 1.8 or rw > dw * 3.6:
                    continue
                if rh < dh * 0.45 or rh > dh * 0.95:
                    continue
                score = float(area) + float(overlap_x * 20) - float(abs(gx - dx) * 10)
                if score > best_score:
                    best_score = score
                    best_bbox = (int(gx), int(gy), int(rw), int(rh))
                    best_mask = (labels[ry:ry + rh, rx:rx + rw] == idx).astype(np.uint8) * 255

            if best_bbox is None or best_mask is None:
                continue

            bx, by, bw, bh = best_bbox
            proj = (best_mask > 0).sum(axis=0).astype(np.float32)
            proj = cv2.GaussianBlur(proj.reshape(1, -1), (1, 15), 0).ravel()
            left_start, left_end = int(bw * 0.16), int(bw * 0.45)
            right_start, right_end = int(bw * 0.55), int(bw * 0.84)
            if left_end <= left_start + 4 or right_end <= right_start + 4:
                continue
            left_sep = left_start + int(np.argmin(proj[left_start:left_end]))
            right_sep = right_start + int(np.argmin(proj[right_start:right_end]))
            if left_sep < max(28, int(dw * 0.22)) or right_sep <= left_sep + max(70, int(dw * 0.68)):
                continue

            segments = [('left', 0, left_sep), ('door', left_sep, right_sep), ('right', right_sep, bw)]
            seg_boxes = {}
            for name, s0, s1 in segments:
                if s1 <= s0 + 8:
                    continue
                sub = best_mask[:, s0:s1]
                cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    continue
                c = max(cnts, key=cv2.contourArea)
                sx0, sy0, sw0, sh0 = cv2.boundingRect(c)
                seg_boxes[name] = (bx + s0 + sx0, by + sy0, sw0, sh0)

            if 'left' not in seg_boxes or 'door' not in seg_boxes or 'right' not in seg_boxes:
                continue

            lbox = seg_boxes['left']
            mbox = seg_boxes['door']
            rbox = seg_boxes['right']
            if lbox[2] < max(50, int(dw * 0.45)) or rbox[2] < max(50, int(dw * 0.45)):
                continue

            # Replace nearby small central windows with the split sidelights.
            survivors = []
            cluster_x0 = min(lbox[0], dx) - 10
            cluster_x1 = max(rbox[0] + rbox[2], dx + dw) + 10
            for win in refined_windows:
                wx, wy, ww, wh = win.bbox
                if wx + ww >= cluster_x0 and wx <= cluster_x1 and (wy + wh) >= by and wy <= (by + bh + 25):
                    continue
                survivors.append(win)
            refined_windows = survivors

            refined_windows.append(DetectedRegion(
                label='F_PHOTO_SIDE_L',
                region_type='window',
                bbox=lbox,
                contour=np.array([]),
                area_px=float(lbox[2] * lbox[3]),
                color_detected='photo-door-sidelight-split',
            ))
            refined_windows.append(DetectedRegion(
                label='F_PHOTO_SIDE_R',
                region_type='window',
                bbox=rbox,
                contour=np.array([]),
                area_px=float(rbox[2] * rbox[3]),
                color_detected='photo-door-sidelight-split',
            ))

            mx, my, mw, mh = mbox
            target_w = min(
                max(int(round(mw * 0.94)), dw + 12),
                int(round(mw * 0.98))
            )
            target_h = min(
                dh,
                max(int(round(mh * 0.92)), int(round(dh * 0.90)))
            )
            target_cx = mx + mw * 0.5
            nx = int(round(target_cx - target_w * 0.5))
            nx = max(fx, min(nx, fx + fw - target_w))
            refined_doors[d_idx] = DetectedRegion(
                label=door.label,
                region_type=door.region_type,
                bbox=(int(nx), int(dy), int(target_w), int(target_h)),
                contour=np.array([]),
                area_px=float(target_w * target_h),
                area_m2=door.area_m2,
                width_m=door.width_m,
                height_m=door.height_m,
                length_m=door.length_m,
                ocr_text=door.ocr_text,
                parent_facade=door.parent_facade,
                color_detected=door.color_detected,
                is_open_path=door.is_open_path,
            )

        refined_windows = self._nms_regions(refined_windows, iou_threshold=0.25)
        refined_doors.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return refined_windows, refined_doors

    @staticmethod
    def _clone_region_with_bbox(region: DetectedRegion, bbox: tuple,
                                color_detected: Optional[str] = None) -> DetectedRegion:
        return DetectedRegion(
            label=region.label,
            region_type=region.region_type,
            bbox=bbox,
            contour=np.array([]),
            area_px=float(bbox[2] * bbox[3]),
            area_m2=region.area_m2,
            width_m=region.width_m,
            height_m=region.height_m,
            length_m=region.length_m,
            ocr_text=region.ocr_text,
            parent_facade=region.parent_facade,
            color_detected=(color_detected if color_detected is not None else region.color_detected),
            is_open_path=region.is_open_path,
        )

    @staticmethod
    def _group_regions_by_row(regions: list, facade_bbox: tuple) -> list[list]:
        if len(regions) < 2:
            return [list(regions)] if regions else []

        _fx, _fy, _fw, fh = facade_bbox
        gap_thr = max(42, int(fh * 0.12))
        ordered = sorted(regions, key=lambda r: (r.center[1], r.bbox[0]))
        rows: list[list] = [[ordered[0]]]
        last_cy = ordered[0].center[1]
        for region in ordered[1:]:
            cy = region.center[1]
            if cy - last_cy > gap_thr:
                rows.append([region])
            else:
                rows[-1].append(region)
            last_cy = cy
        return rows

    @staticmethod
    def _fit_door_to_window_gap(facade_bbox: tuple,
                                door_bbox: tuple,
                                left_bbox: tuple,
                                right_bbox: tuple) -> tuple | None:
        fx, _fy, fw, _fh = facade_bbox
        dx, dy, dw, dh = door_bbox
        gap_x0 = left_bbox[0] + left_bbox[2]
        gap_x1 = right_bbox[0]
        gap_w = gap_x1 - gap_x0
        if gap_w < max(28, int(fw * 0.035)):
            return None
        wide_gap = gap_w > max(int(fw * 0.12), int(dw * 2.4))
        if wide_gap:
            target_w = min(
                max(dw, int(round(dw * 1.10))),
                max(int(round(fw * 0.085)), dw + 18),
            )
            gap_center = (gap_x0 + gap_x1) * 0.5
            door_center = dx + dw * 0.5
            target_center = gap_center * 0.55 + door_center * 0.45
        else:
            target_w = min(
                gap_w,
                max(dw, int(round(gap_w * 0.98))),
            )
            target_center = (gap_x0 + gap_x1) * 0.5
        target_w = max(dw, min(target_w, int(round(fw * 0.12))))
        nx = int(round(target_center - target_w * 0.5))
        nx = max(fx, min(nx, fx + fw - target_w))
        return (int(nx), int(dy), int(target_w), int(dh))

    def _refine_composite_window_frame(self, gray: np.ndarray,
                                       facade_bbox: tuple,
                                       window_bbox: tuple) -> tuple | None:
        fx, fy, fw, fh = facade_bbox
        x, y, w_box, h_box = window_bbox
        if w_box < 12 or h_box < 24:
            return None

        narrow = w_box <= max(34, int(fw * 0.026))
        pad_left = max(20, int(w_box * (2.1 if narrow else 1.2)))
        pad_right = max(20, int(w_box * (1.7 if narrow else 1.0)))
        pad_top = max(14, int(h_box * 0.55))
        pad_bottom = max(14, int(h_box * 0.45))
        sx = max(fx, x - pad_left)
        ex = min(fx + fw, x + w_box + pad_right)
        sy = max(fy, y - pad_top)
        ey = min(fy + fh, y + h_box + pad_bottom)
        if ex - sx < 24 or ey - sy < 24:
            return None

        roi = gray[sy:ey, sx:ex]
        grad = (
            np.abs(cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)) +
            np.abs(cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)) * 0.55
        )
        seed_area = max(1.0, float(w_box * h_box))
        best_bbox = None
        best_score = -1.0

        for pct in (76.0, 79.0, 82.0):
            thr = float(np.percentile(grad, pct))
            mask = (grad >= thr).astype(np.uint8) * 255
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 7)),
            )
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            )

            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            for idx in range(1, n_labels):
                rx, ry, rw, rh, _area = stats[idx]
                gx, gy = sx + int(rx), sy + int(ry)
                inter_w = max(0, min(x + w_box, gx + rw) - max(x, gx))
                inter_h = max(0, min(y + h_box, gy + rh) - max(y, gy))
                inter = float(inter_w * inter_h)
                if inter <= 0:
                    continue
                width_growth = rw / max(1.0, float(w_box))
                height_growth = rh / max(1.0, float(h_box))
                area_ratio = (rw * rh) / max(1.0, float(fw * fh))
                aspect = rw / max(1.0, float(rh))
                center_shift = abs((gx + rw * 0.5) - (x + w_box * 0.5))
                if narrow:
                    if not (0.75 <= width_growth <= 3.80):
                        continue
                    if not (0.80 <= height_growth <= 1.85):
                        continue
                    if not (0.12 <= aspect <= 0.85):
                        continue
                    if center_shift > max(140, int(w_box * 6.0)):
                        continue
                else:
                    if not (0.58 <= width_growth <= 2.10):
                        continue
                    if not (0.62 <= height_growth <= 1.55):
                        continue
                    if not (0.40 <= aspect <= 2.50):
                        continue
                    if center_shift > max(170, int(w_box * 1.45)):
                        continue
                if not (0.002 <= area_ratio <= 0.14):
                    continue
                expansion = (
                    max(0, x - gx) +
                    max(0, gx + rw - (x + w_box)) +
                    max(0, y - gy) +
                    max(0, gy + rh - (y + h_box))
                )
                score = (
                    inter / seed_area * 1.20 +
                    min(1.0, expansion / max(18.0, float(w_box + h_box))) * 0.35 -
                    center_shift / max(120.0, float(w_box * (5.0 if narrow else 1.5))) * 0.10
                )
                if score > best_score:
                    best_score = score
                    best_bbox = (gx, gy, int(rw), int(rh))

        if best_bbox is None or best_score < 0.30:
            return None
        return best_bbox

    @staticmethod
    def _assign_parent_facades(facades: list, regions: list):
        if not facades:
            return

        for region in regions:
            cx, cy = region.center
            best = None
            best_dist = float("inf")

            for facade in facades:
                fx, fy, fw, fh = facade.bbox
                if fx <= cx <= fx + fw and fy <= cy <= fy + fh:
                    region.parent_facade = facade.label
                    best = None
                    break

                fcx = fx + fw // 2
                fcy = fy + fh // 2
                dist = ((cx - fcx) ** 2 + (cy - fcy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best = facade

            if region.parent_facade is None and best is not None:
                region.parent_facade = best.label

    @staticmethod
    def _bbox_iou(a: tuple, b: tuple) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
        union = float(aw * ah + bw * bh - inter)
        if union <= 0:
            return 0.0
        return inter / union

    @staticmethod
    def _bbox_contains(a: tuple, b: tuple) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return (ax <= bx and ay <= by and
                ax + aw >= bx + bw and ay + ah >= by + bh)

    @staticmethod
    def _score_promoted_door_candidate(facade_bbox: tuple,
                                       bbox: tuple,
                                       area_px: float,
                                       parts_count: int = 1) -> float:
        """Score how plausible a photo opening is as a door."""
        fx, fy, fw, fh = facade_bbox
        x, y, w_box, h_box = bbox
        if fw <= 0 or fh <= 0 or w_box <= 0 or h_box <= 0:
            return -1.0

        facade_area = max(1.0, float(fw * fh))
        width_ratio = w_box / max(1.0, float(fw))
        height_ratio = h_box / max(1.0, float(fh))
        bottom_ratio = (y + h_box - fy) / max(1.0, float(fh))
        cx_ratio = (x + w_box * 0.5 - fx) / max(1.0, float(fw))
        aspect_ratio = w_box / max(1.0, float(h_box))
        area_ratio = area_px / facade_area

        # Broad plausibility gates; narrower ranking comes from the score.
        if not (0.030 <= width_ratio <= 0.32):
            return -1.0
        if not (0.30 <= height_ratio <= 0.92):
            return -1.0
        if not (0.18 <= aspect_ratio <= 0.72):
            return -1.0
        if not (0.18 <= cx_ratio <= 0.82):
            return -1.0
        if not (0.58 <= bottom_ratio <= 1.02):
            return -1.0
        if not (0.008 <= area_ratio <= 0.26):
            return -1.0

        center_score = max(0.0, 1.0 - abs(cx_ratio - 0.5) / 0.32)
        bottom_score = min(1.0, max(0.0, (bottom_ratio - 0.58) / 0.25))
        height_score = min(1.0, max(0.0, (height_ratio - 0.22) / 0.40))
        width_score = max(0.0, 1.0 - abs(width_ratio - 0.13) / 0.14)
        aspect_score = max(0.0, 1.0 - abs(aspect_ratio - 0.42) / 0.38)
        parts_bonus = 0.08 if parts_count > 1 else 0.0

        return (
            center_score * 0.34 +
            bottom_score * 0.28 +
            height_score * 0.18 +
            width_score * 0.12 +
            aspect_score * 0.08 +
            parts_bonus
        )

    def _merge_photo_window_fragments(self, facades: list, windows: list) -> list:
        """Merge adjacent fragments that belong to the same photo window opening."""
        if not facades or len(windows) <= 1:
            return windows

        merged_windows = list(windows)
        changed = True

        while changed:
            changed = False
            for facade in facades:
                fx, fy, fw, fh = facade.bbox
                facade_windows = [
                    w for w in merged_windows
                    if fx <= w.center[0] <= fx + fw and fy <= w.center[1] <= fy + fh
                ]
                facade_windows.sort(key=lambda w: (w.bbox[0], w.bbox[1]))

                merged_pair = None
                for i, w1 in enumerate(facade_windows):
                    x1, y1, ww1, wh1 = w1.bbox
                    for w2 in facade_windows[i + 1:]:
                        x2, y2, ww2, wh2 = w2.bbox
                        gap_x = max(0, max(x2 - (x1 + ww1), x1 - (x2 + ww2)))
                        if gap_x > max(16, int(fw * 0.02)):
                            continue

                        overlap_y = max(0, min(y1 + wh1, y2 + wh2) - max(y1, y2))
                        if overlap_y < min(wh1, wh2) * 0.45:
                            continue

                        nx = min(x1, x2)
                        ny = min(y1, y2)
                        nx2 = max(x1 + ww1, x2 + ww2)
                        ny2 = max(y1 + wh1, y2 + wh2)
                        nw = nx2 - nx
                        nh = ny2 - ny
                        width_ratio = nw / max(1.0, float(fw))
                        height_ratio = nh / max(1.0, float(fh))
                        cy_ratio = (ny + nh * 0.5 - fy) / max(1.0, float(fh))
                        aspect_ratio = nw / max(1.0, float(nh))

                        # Window-like union: moderate width, not too close to socle, not ultra-tall.
                        if not (0.06 <= width_ratio <= 0.20):
                            continue
                        if not (0.16 <= height_ratio <= 0.55):
                            continue
                        if not (0.22 <= cy_ratio <= 0.70):
                            continue
                        if not (0.55 <= aspect_ratio <= 2.20):
                            continue

                        merged_pair = (w1, w2, (nx, ny, nw, nh))
                        break
                    if merged_pair:
                        break

                if not merged_pair:
                    continue

                w1, w2, bbox = merged_pair
                merged_windows = [
                    w for w in merged_windows
                    if w not in (w1, w2)
                ]
                merged_windows.append(DetectedRegion(
                    label=w1.label or w2.label or "F_PHOTO_MERGED",
                    region_type="window",
                    bbox=bbox,
                    contour=np.array([]),
                    area_px=float(w1.area_px + w2.area_px),
                    color_detected="photo-window-merged",
                ))
                changed = True
                break

        merged_windows.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return merged_windows

    @staticmethod
    def _estimate_photo_window_row_band(windows: list):
        """Estimate common top/bottom band for facade windows when they form one stable row."""
        if len(windows) < 3:
            return None

        tops = np.array([w.bbox[1] for w in windows], dtype=np.float32)
        bottoms = np.array([w.bbox[1] + w.bbox[3] for w in windows], dtype=np.float32)
        heights = np.array([w.bbox[3] for w in windows], dtype=np.float32)
        row_top = int(round(float(np.median(tops))))
        row_bottom = int(round(float(np.median(bottoms))))
        row_height = int(round(float(np.median(heights))))
        if row_bottom <= row_top or row_height <= 0:
            return None

        top_iqr = float(np.percentile(tops, 75) - np.percentile(tops, 25))
        bottom_iqr = float(np.percentile(bottoms, 75) - np.percentile(bottoms, 25))
        max_spread = max(18.0, row_height * 0.35)
        if top_iqr > max_spread or bottom_iqr > (max_spread * 1.10):
            return None

        return row_top, row_bottom, row_height

    def _refine_right_photo_windows(self, image: np.ndarray,
                                    facades: list, windows: list) -> list:
        """Expand truncated right-side photo windows using local edge components."""
        if not facades or not windows:
            return windows

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        row_band = self._estimate_photo_window_row_band(windows)
        refined = []

        for win in windows:
            updated = win
            if "photo" not in (win.color_detected or ""):
                refined.append(updated)
                continue

            parent = None
            for facade in facades:
                fx, fy, fw, fh = facade.bbox
                if fx <= win.center[0] <= fx + fw and fy <= win.center[1] <= fy + fh:
                    parent = facade
                    break
            if parent is None:
                refined.append(updated)
                continue

            fx, fy, fw, fh = parent.bbox
            x, y, w_box, h_box = win.bbox
            width_ratio = w_box / max(1.0, float(fw))
            height_ratio = h_box / max(1.0, float(fh))
            cx_ratio = (x + w_box * 0.5 - fx) / max(1.0, float(fw))

            # Refine obviously truncated/narrow windows, including central groups.
            if width_ratio >= 0.08 and height_ratio >= 0.25:
                refined.append(updated)
                continue

            if cx_ratio < 0.38:
                left_pad = 0.45
            elif cx_ratio < 0.62:
                left_pad = 0.85
            else:
                left_pad = 0.55

            if cx_ratio > 0.62:
                right_pad = 0.45
            elif cx_ratio > 0.38:
                right_pad = 0.85
            else:
                right_pad = 0.55

            sx = max(fx, x - int(w_box * left_pad))
            ex = min(fx + fw, x + w_box + int(w_box * right_pad))
            sy = max(fy, y - int(h_box * 0.35))
            ey = min(fy + fh, y + h_box + int(h_box * 0.35))
            roi = gray[sy:ey, sx:ex]
            if roi.size == 0:
                refined.append(updated)
                continue

            orig_area = max(1.0, float(w_box * h_box))
            best_bbox = None
            best_score = 0.0

            grad = (
                np.abs(cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)) +
                np.abs(cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)) * 0.4
            )
            for pct in (78.0, 80.0, 82.0, 84.0):
                thr = float(np.percentile(grad, pct))
                mask = (grad >= thr).astype(np.uint8) * 255
                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7)),
                )
                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_OPEN,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3)),
                )

                n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
                for idx in range(1, n_labels):
                    rx, ry, rw, rh, _ = stats[idx]
                    gx, gy = sx + int(rx), sy + int(ry)
                    inter_w = max(0, min(x + w_box, gx + rw) - max(x, gx))
                    inter_h = max(0, min(y + h_box, gy + rh) - max(y, gy))
                    inter = float(inter_w * inter_h)
                    if inter <= 0:
                        continue

                    width_growth = rw / max(1.0, float(w_box))
                    height_growth = rh / max(1.0, float(h_box))
                    shift = abs((gx + rw * 0.5) - (x + w_box * 0.5))
                    aspect = rw / max(1.0, float(rh))
                    if not (0.85 <= width_growth <= 1.45):
                        continue
                    if not (1.0 <= height_growth <= 1.35):
                        continue
                    if shift > max(42, int(w_box * 0.45)):
                        continue
                    if not (0.55 <= aspect <= 2.30):
                        continue

                    score = inter / orig_area + (height_growth - 1.0) * 0.25
                    if score > best_score:
                        best_score = score
                        best_bbox = (gx, gy, int(rw), int(rh))

            if best_bbox and best_score >= 0.48:
                bx, by, bw, bh = best_bbox
                if row_band:
                    row_top, row_bottom, row_height = row_band
                    if bh > int(row_height * 1.12) or abs(by - row_top) > 18:
                        by = row_top
                        bh = max(1, row_bottom - row_top)
                    best_bbox = (bx, by, bw, bh)
                updated = DetectedRegion(
                    label=win.label,
                    region_type=win.region_type,
                    bbox=best_bbox,
                    contour=np.array([]),
                    area_px=float(best_bbox[2] * best_bbox[3]),
                    area_m2=win.area_m2,
                    width_m=win.width_m,
                    height_m=win.height_m,
                    length_m=win.length_m,
                    ocr_text=win.ocr_text,
                    parent_facade=win.parent_facade,
                    color_detected=win.color_detected,
                    is_open_path=win.is_open_path,
                )

            refined.append(updated)

        refined.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return refined

    def _refine_photo_door_geometry(self, image: np.ndarray,
                                    facades: list, doors: list, windows: list | None = None) -> list:
        """Expand narrow promoted/detected photo doors to the usable opening."""
        if not facades or not doors:
            return doors

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        refined = []
        windows = windows or []

        for door in doors:
            updated = door
            if "photo" not in (door.color_detected or ""):
                refined.append(updated)
                continue

            parent = None
            for facade in facades:
                fx, fy, fw, fh = facade.bbox
                if fx <= door.center[0] <= fx + fw and fy <= door.center[1] <= fy + fh:
                    parent = facade
                    break
            if parent is None:
                refined.append(updated)
                continue

            fx, fy, fw, fh = parent.bbox
            x, y, w_box, h_box = door.bbox
            width_ratio = w_box / max(1.0, float(fw))
            bottom_ratio = (y + h_box - fy) / max(1.0, float(fh))
            cx_ratio = (x + w_box * 0.5 - fx) / max(1.0, float(fw))
            is_merged = "merged" in (parent.color_detected or "").lower()
            max_width_ratio = 0.12 if is_merged else 0.10
            if width_ratio > max_width_ratio or bottom_ratio > 1.06 or (not is_merged and not (0.32 <= cx_ratio <= 0.68)):
                refined.append(updated)
                continue

            sx = max(fx, x - int(w_box * (1.35 if is_merged else 0.8)))
            ex = min(fx + fw, x + w_box + int(w_box * (1.45 if is_merged else 1.1)))
            sy = max(fy, y - int(h_box * (0.24 if is_merged else 0.15)))
            ey = min(fy + fh, y + h_box + int(h_box * (0.32 if is_merged else 0.40)))
            roi = gray[sy:ey, sx:ex]
            if roi.size == 0:
                refined.append(updated)
                continue

            grad_x = np.abs(cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3))
            grad_y = np.abs(cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3))
            col = grad_x.sum(axis=0)
            col = cv2.GaussianBlur(col.reshape(1, -1), (1, 9), 0).ravel()
            row = grad_y.sum(axis=1)
            row = cv2.GaussianBlur(row.reshape(-1, 1), (1, 9), 0).ravel()
            rel_left = x - sx
            rel_right = rel_left + w_box

            left_start = max(0, rel_left - int(w_box * (0.34 if is_merged else 0.20)))
            left_end = min(col.shape[0], rel_left + int(w_box * (0.28 if is_merged else 0.18)))
            if is_merged:
                right_start = max(0, rel_right - int(w_box * 0.08))
                right_end = min(col.shape[0], rel_right + max(54, int(fw * 0.06)))
            else:
                right_start = min(col.shape[0], max(0, rel_right + max(4, int(w_box * 0.06))))
                right_end = min(col.shape[0], rel_right + max(36, int(fw * 0.05)))
            if left_end <= left_start or right_end <= right_start:
                refined.append(updated)
                continue

            left_idxs = np.arange(left_start, left_end)
            right_idxs = np.arange(right_start, right_end)
            left_scores = col[left_idxs] / (1.0 + np.abs(left_idxs - rel_left) * 0.10)
            right_scores = col[right_idxs]
            new_left = int(left_idxs[int(np.argmax(left_scores))])
            new_right = int(right_idxs[int(np.argmax(right_scores))])

            if new_right <= new_left + max(30, int(w_box * 0.9)):
                refined.append(updated)
                continue

            rel_top = y - sy
            rel_bottom = rel_top + h_box
            top_start = max(0, rel_top - int(h_box * (0.24 if is_merged else 0.12)))
            top_end = min(row.shape[0], rel_top + int(h_box * (0.18 if is_merged else 0.10)))
            bot_start = max(0, rel_bottom - int(h_box * (0.16 if is_merged else 0.10)))
            bot_end = min(row.shape[0], rel_bottom + int(h_box * (0.24 if is_merged else 0.35)))
            if top_end <= top_start or bot_end <= bot_start:
                refined.append(updated)
                continue

            top_idxs = np.arange(top_start, top_end)
            bot_idxs = np.arange(bot_start, bot_end)
            top_scores = row[top_idxs]
            bot_scores = row[bot_idxs]
            top = sy + int(top_idxs[int(np.argmax(top_scores))])
            bottom = sy + int(bot_idxs[int(np.argmax(bot_scores))])
            if bottom <= top + max(40, int(h_box * 0.75)):
                top = y
                bottom = y + h_box
            new_bbox = (sx + new_left, top, new_right - new_left, bottom - top)
            new_bbox = self._complete_photo_door_hole_bbox(
                gray, parent.bbox, new_bbox
            )
            new_bbox = self._extend_door_by_jamb_continuity(
                gray, parent.bbox, new_bbox
            )
            if "promoted" not in (door.color_detected or "") and "local-contrast" not in (door.color_detected or ""):
                new_bbox = self._expand_door_width_by_jambs(
                    gray, parent.bbox, new_bbox
                )
            if "local-contrast" in (door.color_detected or ""):
                nx, ny, nw, nh = new_bbox
                max_w = max(w_box + 8, int(round(w_box * 1.45)))
                max_h = max(h_box + 8, int(round(h_box * 1.15)))
                nw = min(nw, max_w)
                nh = min(nh, max_h)
                cx0 = x + w_box * 0.5
                cx1 = nx + nw * 0.5
                blend = float(getattr(config, "PHOTO_DOOR_LOCAL_CENTER_BLEND", 0.65))
                max_shift = max(10.0, float(w_box) * float(getattr(config, "PHOTO_DOOR_LOCAL_MAX_SHIFT_RATIO", 0.45)))
                target_cx = cx0 * (1.0 - blend) + cx1 * blend
                target_cx = max(cx0 - max_shift, min(target_cx, cx0 + max_shift))
                nx = int(round(target_cx - nw * 0.5))
                nx = max(fx, min(nx, fx + fw - nw))
                ny = max(y, min(ny, fy + fh - nh))
                new_bbox = (nx, ny, nw, nh)
            elif is_merged:
                nx, ny, nw, nh = new_bbox
                max_h = max(h_box + 14, int(round(h_box * 1.18)))
                nh = min(nh, max_h)
                max_top_shift = max(14, int(round(h_box * 0.18)))
                ny = max(y - max_top_shift, min(ny, y + max_top_shift))
                new_bbox = (int(nx), int(ny), int(nw), int(nh))

            if windows and "promoted" in (door.color_detected or ""):
                nx, ny, nw, nh = new_bbox
                row_tol = max(18, int(round(nh * 0.14)))
                gap_limit = max(48, int(round(nw * 0.90)))
                right_gaps = []
                left_gaps = []
                for win in windows:
                    wx, wy, ww, wh = win.bbox
                    wcx = wx + ww * 0.5
                    wcy = wy + wh * 0.5
                    if not (fx <= wcx <= fx + fw and fy <= wcy <= fy + fh):
                        continue
                    overlap_y = max(0, min(ny + nh, wy + wh) - max(ny, wy))
                    if overlap_y < min(nh, wh) * 0.35 and abs(wy - ny) > row_tol:
                        continue
                    gap_right = wx - (nx + nw)
                    gap_left = nx - (wx + ww)
                    if 0 <= gap_right <= gap_limit:
                        right_gaps.append(float(gap_right))
                    if 0 <= gap_left <= gap_limit:
                        left_gaps.append(float(gap_left))
                if right_gaps and (not left_gaps or min(right_gaps) + 6.0 < min(left_gaps)):
                    shift = min(
                        int(round(min(right_gaps) * 0.55)),
                        max(8, int(round(nw * 0.35)))
                    )
                    if shift > 0:
                        nx = min(fx + fw - nw, nx + shift)
                        new_bbox = (int(nx), int(ny), int(nw), int(nh))
            updated = DetectedRegion(
                label=door.label,
                region_type=door.region_type,
                bbox=new_bbox,
                contour=np.array([]),
                area_px=float(new_bbox[2] * new_bbox[3]),
                area_m2=door.area_m2,
                width_m=door.width_m,
                height_m=door.height_m,
                length_m=door.length_m,
                ocr_text=door.ocr_text,
                parent_facade=door.parent_facade,
                color_detected=door.color_detected,
                is_open_path=door.is_open_path,
            )
            refined.append(updated)

        refined.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return refined

    @staticmethod
    def _complete_photo_door_hole_bbox(gray: np.ndarray,
                                       facade_bbox: tuple,
                                       door_bbox: tuple) -> tuple:
        """Extend a door seed downward when the lower opaque panel still differs from wall."""
        fx, fy, fw, fh = facade_bbox
        x, y, w_box, h_box = door_bbox
        if w_box < 8 or h_box < 16:
            return door_bbox

        img_h, img_w = gray.shape[:2]
        x = max(0, min(img_w - 1, x))
        y = max(0, min(img_h - 1, y))
        door_right = min(img_w, x + w_box)
        facade_right = min(img_w, fx + fw)
        facade_bottom = min(img_h - 1, fy + fh - 1)
        if door_right - x < 6 or facade_bottom <= y + h_box:
            return door_bbox

        wall_band = max(12, int(w_box * 0.65))
        min_wall = max(6, int(w_box * 0.18))

        def row_signal(abs_y: int) -> float:
            if abs_y < 0 or abs_y >= img_h:
                return 0.0
            inner = gray[abs_y, x:door_right]
            left = gray[abs_y, max(fx, x - wall_band):x]
            right = gray[abs_y, door_right:min(facade_right, door_right + wall_band)]
            if inner.size < max(6, int(w_box * 0.35)):
                return 0.0
            parts = []
            if left.size >= min_wall:
                parts.append(left)
            if right.size >= min_wall:
                parts.append(right)
            if not parts:
                return 0.0
            wall = np.concatenate(parts)
            diff = abs(float(np.mean(inner)) - float(np.mean(wall)))
            texture = float(np.std(inner))
            return diff + texture * 0.20

        seed_start = y + max(0, int(h_box * 0.35))
        seed_end = min(facade_bottom, y + h_box - 1)
        seed_signals = [
            row_signal(abs_y)
            for abs_y in range(seed_start, seed_end + 1)
        ]
        seed_signals = [s for s in seed_signals if s > 0]
        if not seed_signals:
            return door_bbox

        active_threshold = max(12.0, float(np.median(seed_signals)) * 0.38)
        quiet_threshold = active_threshold * 0.72
        max_extra = max(24, int(fh * 0.28))
        scan_end = min(facade_bottom, y + h_box + max_extra)

        last_active = y + h_box - 1
        quiet_run = 0
        for abs_y in range(y + h_box, scan_end + 1):
            signal = row_signal(abs_y)
            if signal >= active_threshold or (quiet_run < 3 and signal >= quiet_threshold):
                last_active = abs_y
                quiet_run = 0
            else:
                quiet_run += 1
                if quiet_run >= 6:
                    break

        new_bottom = max(y + h_box, last_active + 1)
        new_bottom = min(new_bottom, facade_bottom + 1)
        return (x, y, w_box, max(1, new_bottom - y))

    @staticmethod
    def _extend_door_by_jamb_continuity(gray: np.ndarray,
                                        facade_bbox: tuple,
                                        door_bbox: tuple) -> tuple:
        """Extend door downward while lateral jamb edges remain visible."""
        fx, fy, fw, fh = facade_bbox
        x, y, w_box, h_box = door_bbox
        if w_box < 10 or h_box < 20:
            return door_bbox

        img_h, img_w = gray.shape[:2]
        facade_bottom = min(img_h - 1, fy + fh - 1)
        if y + h_box >= facade_bottom:
            return door_bbox

        grad_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        left_x = max(2, min(img_w - 3, x))
        right_x = max(2, min(img_w - 3, x + w_box - 1))
        band = max(2, min(4, int(round(w_box * 0.06))))

        def jamb_strength(abs_y: int, edge_x: int) -> float:
            if abs_y < 0 or abs_y >= img_h:
                return 0.0
            x0 = max(0, edge_x - band)
            x1 = min(img_w, edge_x + band + 1)
            row = grad_x[abs_y, x0:x1]
            if row.size == 0:
                return 0.0
            return float(np.max(row))

        seed_start = y + max(0, int(h_box * 0.15))
        seed_end = min(facade_bottom, y + h_box - 1)
        left_seed = [jamb_strength(abs_y, left_x) for abs_y in range(seed_start, seed_end + 1)]
        right_seed = [jamb_strength(abs_y, right_x) for abs_y in range(seed_start, seed_end + 1)]
        left_seed = [v for v in left_seed if v > 0]
        right_seed = [v for v in right_seed if v > 0]
        if not left_seed and not right_seed:
            return door_bbox

        left_thr = max(18.0, float(np.median(left_seed)) * 0.28) if left_seed else 9999.0
        right_thr = max(18.0, float(np.median(right_seed)) * 0.28) if right_seed else 9999.0
        max_extra = max(28, int(fh * 0.34))
        scan_end = min(facade_bottom, y + h_box + max_extra)

        last_active = y + h_box - 1
        quiet_run = 0
        for abs_y in range(y + h_box, scan_end + 1):
            left_ok = jamb_strength(abs_y, left_x) >= left_thr
            right_ok = jamb_strength(abs_y, right_x) >= right_thr

            # For mixed doors, one jamb can weaken temporarily; continue while at
            # least one side clearly persists, but stop after a sustained quiet zone.
            if left_ok or right_ok:
                last_active = abs_y
                quiet_run = 0
            else:
                quiet_run += 1
                if quiet_run >= 8:
                    break

        new_bottom = max(y + h_box, last_active + 1)
        new_bottom = min(new_bottom, facade_bottom + 1)
        return (x, y, w_box, max(1, new_bottom - y))

    @staticmethod
    def _expand_door_width_by_jambs(gray: np.ndarray,
                                    facade_bbox: tuple,
                                    door_bbox: tuple) -> tuple:
        """Expand door left/right using vertically consistent jamb edges."""
        fx, fy, fw, fh = facade_bbox
        x, y, w_box, h_box = door_bbox
        img_h, img_w = gray.shape[:2]
        if w_box < 10 or h_box < 24:
            return door_bbox

        grad_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        band = max(2, min(4, int(round(w_box * 0.06))))
        y0 = max(0, y + int(h_box * 0.10))
        y1 = min(img_h, y + int(h_box * 0.88))
        if y1 <= y0 + 8:
            return door_bbox

        def edge_score(edge_x: int) -> float:
            edge_x = max(band, min(img_w - band - 1, edge_x))
            stripe = grad_x[y0:y1, edge_x - band:edge_x + band + 1]
            if stripe.size == 0:
                return 0.0
            row_strength = np.max(stripe, axis=1)
            active = row_strength > max(20.0, float(np.median(row_strength)) * 0.55)
            continuity = float(np.mean(active)) if active.size else 0.0
            return float(np.mean(row_strength)) * (0.45 + continuity)

        max_expand = max(16, int(min(fw * 0.05, w_box * 0.55)))
        left_candidates = range(max(fx + 2, x - max_expand), min(x + 1, x + int(w_box * 0.12)))
        right_candidates = range(max(x + w_box - int(w_box * 0.12), x + 1),
                                 min(fx + fw - 2, x + w_box + max_expand))

        base_left = edge_score(x)
        base_right = edge_score(x + w_box - 1)
        best_left = x
        best_right = x + w_box - 1
        best_left_score = base_left
        best_right_score = base_right
        min_shift = max(3, int(w_box * 0.04))
        left_valid = [x]
        right_valid = [x + w_box - 1]

        for cand in left_candidates:
            score = edge_score(cand)
            # Prefer a slightly more exterior jamb only if continuity is comparable.
            if (
                score >= best_left_score * 0.82 and
                cand <= best_left - min_shift
            ):
                left_valid.append(cand)
            if score > best_left_score:
                best_left_score = score

        for cand in right_candidates:
            score = edge_score(cand)
            if (
                score >= best_right_score * 0.82 and
                cand >= best_right + min_shift
            ):
                right_valid.append(cand)
            if score > best_right_score:
                best_right_score = score

        best_left = min(left_valid)
        best_right = max(right_valid)

        new_left = max(fx, best_left)
        new_right = min(fx + fw - 1, best_right)
        new_width = new_right - new_left + 1
        max_width = max(int(round(w_box * 2.0)), int(fw * 0.14))
        if new_width < w_box or new_width > max_width:
            return door_bbox
        return (new_left, y, new_width, h_box)

    def _select_promoted_photo_door(self, facades: list, windows: list):
        """Pick the most plausible door candidate from photo-window openings."""
        candidates = []

        for facade in facades:
            fx, fy, fw, fh = facade.bbox

            facade_windows = []
            for win in windows:
                wx, wy, ww, wh = win.bbox
                cx = wx + ww * 0.5
                cy = wy + wh * 0.5
                if fx <= cx <= fx + fw and fy <= cy <= fy + fh:
                    facade_windows.append(win)

            if not facade_windows:
                continue

            for win in facade_windows:
                score = self._score_promoted_door_candidate(
                    facade.bbox, win.bbox, float(win.area_px), parts_count=1
                )
                if score < 0:
                    continue
                candidates.append({
                    "score": score,
                    "bbox": win.bbox,
                    "area_px": float(win.area_px),
                    "windows": [win],
                })

            for i, w1 in enumerate(facade_windows):
                x1, y1, ww1, wh1 = w1.bbox
                for w2 in facade_windows[i + 1:]:
                    x2, y2, ww2, wh2 = w2.bbox
                    cx1 = x1 + ww1 * 0.5
                    cx2 = x2 + ww2 * 0.5
                    if abs(cx1 - cx2) > max(42, int(fw * 0.12)):
                        continue

                    overlap_x = max(0, min(x1 + ww1, x2 + ww2) - max(x1, x2))
                    if overlap_x < min(ww1, ww2) * 0.30:
                        continue

                    gap_y = max(0, max(y1, y2) - min(y1 + wh1, y2 + wh2))
                    if gap_y > max(44, int(fh * 0.22)):
                        continue

                    nx = min(x1, x2)
                    ny = min(y1, y2)
                    nx2 = max(x1 + ww1, x2 + ww2)
                    ny2 = max(y1 + wh1, y2 + wh2)
                    bbox = (nx, ny, nx2 - nx, ny2 - ny)
                    area_px = float(w1.area_px + w2.area_px)
                    score = self._score_promoted_door_candidate(
                        facade.bbox, bbox, area_px, parts_count=2
                    )
                    if score < 0:
                        continue
                    candidates.append({
                        "score": score,
                        "bbox": bbox,
                        "area_px": area_px,
                        "windows": [w1, w2],
                    })

        if not candidates:
            return None

        candidates.sort(key=lambda item: item["score"], reverse=True)
        best = candidates[0]
        runner_up = candidates[1] if len(candidates) > 1 else None

        # For sparse opening sets (2-3 windows), keep promotion strict but
        # allow selection without large margin over runner-up.
        min_score = 0.55
        min_margin = 0.06
        if len(candidates) == 1:
            min_score = 0.42
            min_margin = 0.0
        elif len(candidates) <= 2:
            min_score = 0.50
            min_margin = 0.0

        if best["score"] < min_score:
            return None
        if runner_up and best["score"] < runner_up["score"] + min_margin:
            return None

        return best

    def _merge_overlapping_facades(self, facades: list) -> list:
        if not facades:
            return []

        merged = []
        for region in facades:
            matched = None
            for keep in merged:
                iou = self._bbox_iou(keep.bbox, region.bbox)
                if iou > 0.35 or self._bbox_contains(keep.bbox, region.bbox) or \
                   self._bbox_contains(region.bbox, keep.bbox):
                    matched = keep
                    break

            if matched is None:
                merged.append(region)
                continue

            kx, ky, kw, kh = matched.bbox
            rx, ry, rw, rh = region.bbox
            x1 = min(kx, rx)
            y1 = min(ky, ry)
            x2 = max(kx + kw, rx + rw)
            y2 = max(ky + kh, ry + rh)
            matched.bbox = (x1, y1, x2 - x1, y2 - y1)
            matched.area_px = float((x2 - x1) * (y2 - y1))

        merged.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        for idx, region in enumerate(merged, 1):
            region.label = f"FATADA_AUTO_{idx}"
        return merged

    def detect_facades(self, image: np.ndarray,
                       min_area: int = config.MIN_FACADE_AREA_PX,
                       max_area: Optional[int] = None) -> list:
        """Detect facade regions from cyan/blue contour annotations."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = self._build_mask(hsv, self.facade_hsv)

        # Facade contours are often thin lines; dilate to close the shape.
        mask = self._cleanup_mask(mask, close_kernel=(9, 9),
                                  open_kernel=(5, 5), dilate_iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        image_area = image.shape[0] * image.shape[1]
        if max_area is None:
            max_area = int(image_area * 0.90)
        effective_min_area = max(min_area, int(image_area * 0.015))

        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < effective_min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < 150 or h < 70:
                continue

            ratio = w / h if h > 0 else 0
            if ratio < 0.35 or ratio > 8.0:
                continue

            rect_area = w * h
            extent = area / rect_area if rect_area else 0
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = area / hull_area if hull_area else 0
            if extent < 0.08 and solidity < 0.20:
                continue

            regions.append(DetectedRegion(
                label=f"FATADA_{len(regions) + 1}",
                region_type="facade",
                bbox=(x, y, w, h),
                contour=contour,
                area_px=area,
                color_detected="cyan/blue",
            ))

        regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return regions

    def detect_windows(self, image: np.ndarray,
                       min_area: int = config.MIN_WINDOW_AREA_PX,
                       max_area: Optional[int] = None) -> list:
        """Detect window regions from yellow/green annotation boxes."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = self._build_mask(hsv, self.window_hsv)
        mask = self._cleanup_mask(mask, close_kernel=(5, 5),
                                  open_kernel=(3, 3), dilate_iterations=1)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        image_area = image.shape[0] * image.shape[1]
        if max_area is None:
            max_area = int(image_area * 0.15)

        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < 15 or h < 15:
                continue

            ratio = w / h if h > 0 else 0
            if ratio < 0.25 or ratio > 5.5:
                continue

            # Rectangularity check: contour area vs bounding rect area
            rect_area = w * h
            rectangularity = area / rect_area if rect_area > 0 else 0
            if rectangularity < 0.25:
                continue

            regions.append(DetectedRegion(
                label=f"F_{len(regions) + 1}",
                region_type="window",
                bbox=(x, y, w, h),
                contour=contour,
                area_px=area,
                color_detected="yellow/green",
            ))

        regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return regions

    def detect_doors_by_color(self, image: np.ndarray,
                              min_area: int = config.MIN_DOOR_AREA_PX,
                              max_area: Optional[int] = None) -> list:
        """Detect door annotations by magenta/orange outlines."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Red text labels generate many false positives, so prioritize
        # annotation colors used for doors (magenta/orange).
        filtered_ranges = {
            name: bounds for name, bounds in self.door_hsv.items()
            if not name.lower().startswith("red")
        }
        if not filtered_ranges:
            filtered_ranges = self.door_hsv

        mask = self._build_mask(hsv, filtered_ranges)
        mask = self._cleanup_mask(mask, close_kernel=(7, 7),
                                  open_kernel=(3, 3), dilate_iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        image_area = image.shape[0] * image.shape[1]
        if max_area is None:
            max_area = int(image_area * 0.20)

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
            if ratio < 0.08:
                continue
            if w < 12 or h < 18:
                continue
            rect_area = w * h
            extent = area / rect_area if rect_area else 0
            if extent < 0.15:
                continue

            regions.append(DetectedRegion(
                label=f"U_{len(regions) + 1}",
                region_type="door",
                bbox=(x, y, w, h),
                contour=contour,
                area_px=area,
                color_detected="magenta/orange",
            ))

        regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return regions

    def _infer_facades_from_elements(self, image_shape: tuple,
                                     elements: list) -> list:
        """Infer facade zones from detected windows/doors when contour is missing."""
        if not elements:
            return []

        h, w = image_shape[:2]
        row_tol = max(60, h // 14)
        gap_tol = max(140, w // 7)
        margin_x = max(70, w // 30)
        margin_top = max(110, h // 8)
        margin_bottom = max(80, h // 12)

        centers_sorted = sorted(elements, key=lambda r: (r.center[1], r.center[0]))
        rows = []
        for region in centers_sorted:
            cy = region.center[1]
            attached = False
            for row in rows:
                if abs(cy - row["cy"]) <= row_tol:
                    row["items"].append(region)
                    row["cy"] = int(
                        sum(i.center[1] for i in row["items"]) / len(row["items"])
                    )
                    attached = True
                    break
            if not attached:
                rows.append({"cy": cy, "items": [region]})

        inferred = []
        next_idx = 1
        for row in rows:
            row_items = sorted(row["items"], key=lambda r: r.center[0])
            groups = []
            for item in row_items:
                if not groups:
                    groups.append([item])
                    continue
                prev = groups[-1][-1]
                if abs(item.center[0] - prev.center[0]) <= gap_tol:
                    groups[-1].append(item)
                else:
                    groups.append([item])

            for group in groups:
                xs = []
                ys = []
                for item in group:
                    x, y, bw, bh = item.bbox
                    xs.extend([x, x + bw])
                    ys.extend([y, y + bh])

                x1 = max(0, min(xs) - margin_x)
                x2 = min(w, max(xs) + margin_x)
                y1 = max(0, min(ys) - margin_top)
                y2 = min(h, max(ys) + margin_bottom)

                if (x2 - x1) < 120 or (y2 - y1) < 90:
                    continue

                inferred.append(DetectedRegion(
                    label=f"FATADA_AUTO_{next_idx}",
                    region_type="facade",
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    area_px=float((x2 - x1) * (y2 - y1)),
                    color_detected="inferred-from-elements",
                ))
                next_idx += 1

        inferred.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return self._merge_overlapping_facades(inferred)

    def detect_markup_regions(self, image: np.ndarray) -> tuple:
        """Detect explicit red/yellow user markup contours."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_area = image.shape[0] * image.shape[1]

        red_low = cv2.inRange(
            hsv, np.array((0, 120, 80), dtype=np.uint8),
            np.array((10, 255, 255), dtype=np.uint8)
        )
        red_high = cv2.inRange(
            hsv, np.array((170, 120, 80), dtype=np.uint8),
            np.array((180, 255, 255), dtype=np.uint8)
        )
        red_mask = cv2.bitwise_or(red_low, red_high)
        red_mask = self._cleanup_mask(red_mask, close_kernel=(9, 9),
                                      open_kernel=(3, 3), dilate_iterations=1)

        yellow_mask = cv2.inRange(
            hsv, np.array((18, 120, 80), dtype=np.uint8),
            np.array((40, 255, 255), dtype=np.uint8)
        )
        yellow_mask = self._cleanup_mask(yellow_mask, close_kernel=(7, 7),
                                         open_kernel=(3, 3), dilate_iterations=1)

        facades = []
        contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_red:
            area = cv2.contourArea(c)
            if area < image_area * 0.02:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w < 100 or h < 80:
                continue
            rect_area = max(1, w * h)
            extent = area / rect_area
            # Explicit user markup is line-like; broad filled wall colors from raw photos
            # should not enter markup-priority mode.
            if extent > 0.18:
                continue
            facades.append(DetectedRegion(
                label=f"FATADA_MARKUP_{len(facades) + 1}",
                region_type="facade",
                bbox=(x, y, w, h),
                contour=c,
                area_px=float(area),
                color_detected="markup-red",
            ))

        windows = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contours_y, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_y:
            area = cv2.contourArea(c)
            if area < image_area * 0.0004:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w < 18 or h < 18:
                continue
            ratio = w / h if h else 0
            if ratio < 0.25 or ratio > 4.5:
                continue
            rect_area = max(1, w * h)
            extent = area / rect_area
            if extent > 0.30:
                continue
            roi_gray = gray[y:y + h, x:x + w]
            if roi_gray.size > 0:
                mean_val = float(np.mean(roi_gray))
                std_val = float(np.std(roi_gray))
                if mean_val > 165 and std_val < 45:
                    continue
            windows.append(DetectedRegion(
                label=f"F_MARKUP_{len(windows) + 1}",
                region_type="window",
                bbox=(x, y, w, h),
                contour=c,
                area_px=float(area),
                color_detected="markup-yellow",
            ))

        facades.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        windows.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return facades, windows

    def detect_all(self, image: np.ndarray) -> dict:
        """Run facade/window/door color detection in one pass."""
        markup_facades, markup_windows = self.detect_markup_regions(image)
        # Enter markup-priority mode only when a facade contour is explicitly marked.
        if markup_facades:
            facades = markup_facades
            windows = markup_windows
            doors = []
            self._assign_parent_facades(facades, windows)
            facade_by_label = {f.label: f for f in facades}
            filtered_windows = []
            for w in windows:
                parent = facade_by_label.get(w.parent_facade)
                if parent is None:
                    continue
                fx, fy, fw, fh = parent.bbox
                facade_area = max(1, fw * fh)
                ratio = w.area_px / facade_area
                _, cy = w.center
                if cy > fy + fh * 0.72:
                    continue
                if ratio < 0.006 or ratio > 0.30:
                    continue
                filtered_windows.append(w)
            windows = filtered_windows
            return {"facades": facades, "windows": windows, "doors": doors}

        facades = self.detect_facades(image)
        windows = self.detect_windows(image)
        doors = self.detect_doors_by_color(image)
        image_area = image.shape[0] * image.shape[1]

        if facades:
            largest = max((f.area_px for f in facades), default=0.0)
            if largest < image_area * 0.02:
                facades = []

        annotation_signal = len(windows) + len(doors)
        # Prefer photo pipeline when no color facades (point cloud / dark background).
        if not facades:
            photo_facades = self.detect_photo_facade_region(image)
            if photo_facades:
                facades = photo_facades
                windows = self.detect_photo_windows(image, facades)
                doors = self.detect_photo_doors(image, facades, windows)
                windows, doors = self._refine_photo_openings(image, facades, windows, doors)

                # When the facade geometry is merged from a lower body + upper blocks,
                # the base body often yields better window/door separation than the
                # merged ROI. Keep the merged facade, but union openings from the base body.
                if any((f.color_detected or "").find("merged") >= 0 for f in facades):
                    base_facades = self._detect_photo_facade_region_by_foreground(
                        image, float(getattr(config, "PHOTO_FACADE_MIN_AREA_RATIO", 0.08))
                    )
                    if base_facades:
                        base_windows = self.detect_photo_windows(image, base_facades)
                        base_doors = self.detect_photo_doors(image, base_facades, base_windows)
                        base_windows, base_doors = self._refine_photo_openings(
                            image, base_facades, base_windows, base_doors
                        )
                        if base_windows or base_doors:
                            windows = self._nms_regions(windows + base_windows, iou_threshold=0.25)
                            doors = self._nms_regions(doors + base_doors, iou_threshold=0.22)
                            doors = self._recover_composite_gap_doors(
                                image, facades, windows, doors
                            )
                            windows = self._recover_photo_door_sidelights(
                                image, facades, windows, doors
                            )
                            windows, doors = self._refine_composite_opening_clusters(
                                facades, windows, doors
                            )
                            windows, doors = self._align_composite_photo_door_clusters(
                                image, facades, windows, doors
                            )
                            windows, doors = self._regularize_composite_opening_geometry(
                                image, facades, windows, doors
                            )
                            windows = self._nms_regions(windows, iou_threshold=0.25)
                            doors = self._nms_regions(doors, iou_threshold=0.22)
        # Fallback: we have facades but 0 doors and many windows on dark background -
        # likely photo/point cloud where color detection missed the door. Try photo pipeline.
        elif (not doors and len(windows) >= 4 and self._has_dark_background(image)):
            photo_facades = self.detect_photo_facade_region(image)
            if photo_facades:
                photo_windows = self.detect_photo_windows(image, photo_facades)
                photo_doors = self.detect_photo_doors(image, photo_facades, photo_windows)
                photo_windows, photo_doors = self._refine_photo_openings(
                    image, photo_facades, photo_windows, photo_doors
                )
                if photo_doors:
                    windows = photo_windows
                    doors = photo_doors
                    facades = photo_facades

        if not facades:
            facades = self._infer_facades_from_elements(
                image.shape, windows + doors
            )

        self._assign_parent_facades(facades, windows)
        self._assign_parent_facades(facades, doors)
        return {"facades": facades, "windows": windows, "doors": doors}

    def draw_detections(self, image: np.ndarray, results: dict) -> np.ndarray:
        """Draw detection results as colored overlays for visualization."""
        viz = image.copy()

        style_map = {
            "facades": ((255, 180, 0), 3, 0.12),
            "windows": ((0, 255, 255), 2, 0.20),
            "doors":   ((0, 100, 255), 2, 0.20),
            "missing_zones": ((255, 170, 40), 2, 0.18),
            "socles":  ((255, 0, 180), 2, 0.22),
            "socle_profiles": ((220, 0, 255), 3, 0.0),
            "window_perimeters": ((0, 255, 0), 2, 0.0),
            "door_perimeters": ((180, 80, 255), 3, 0.0),
        }
        line_only_keys = {"socle_profiles", "window_perimeters", "door_perimeters"}

        for key, (color, thickness, alpha) in style_map.items():
            for region in results.get(key, []):
                x, y, w, h = region.bbox
                has_contour = isinstance(region.contour, np.ndarray) and region.contour.size >= 6
                draw_contour = region.contour

                overlay = viz.copy()
                if has_contour and key not in line_only_keys:
                    if key == "facades":
                        perim = float(cv2.arcLength(region.contour, True))
                        eps = max(1.0, perim * 0.0018)
                        draw_contour = cv2.approxPolyDP(region.contour, eps, True)
                    cv2.drawContours(overlay, [draw_contour], -1, color, -1)
                    cv2.addWeighted(overlay, alpha, viz, 1 - alpha, 0, viz)
                    cv2.drawContours(viz, [draw_contour], -1, color, thickness)
                    x, y, w, h = cv2.boundingRect(draw_contour)
                elif has_contour and key in line_only_keys:
                    cv2.polylines(
                        viz,
                        [draw_contour],
                        isClosed=not bool(region.is_open_path),
                        color=color,
                        thickness=thickness,
                    )
                    x, y, w, h = cv2.boundingRect(draw_contour)
                else:
                    if key in line_only_keys:
                        if key == "socle_profiles":
                            cv2.line(
                                viz,
                                (x, y + max(0, h // 2)),
                                (x + w, y + max(0, h // 2)),
                                color,
                                thickness,
                            )
                        else:
                            cv2.rectangle(viz, (x, y), (x + w, y + h), color, thickness)
                    else:
                        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
                        cv2.addWeighted(overlay, alpha, viz, 1 - alpha, 0, viz)
                        cv2.rectangle(viz, (x, y), (x + w, y + h), color, thickness)

                label = region.label
                if region.length_m is not None:
                    label += f" {region.length_m:.2f}m"
                elif region.area_m2 is not None:
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















