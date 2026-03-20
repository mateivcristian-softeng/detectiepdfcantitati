"""
DrawQuantPDF - Analysis Pipeline
Orchestrates OCR-first approach with color verification:
1. OCR extracts all text and positions
2. Parser finds FATADA labels, area values, F/U element labels
3. Color detection finds yellow window rectangles
4. Spatial logic groups everything together
5. Calculator produces final report
"""

import os
import re
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import config
from core.color_detector import ColorDetector, DetectedRegion
from core.ocr_engine import OCREngine
from core.area_calculator import AreaCalculator, FacadeReport, ProjectReport

# Feature flags for photo detection refinement (fallback to current flow if no signal)
USE_PHOTO_WINDOW_VALIDATOR = True   # filter roof/oversized FPs (on by default)
USE_PHOTO_ROI_REFINER = False       # optional GrabCut/heuristic refinement (opt-in)


@dataclass
class ParsedFacade:
    name: str
    position: tuple  # (x, y) of the label
    total_area: float = 0.0
    net_area: float = 0.0
    region_bbox: Optional[tuple] = None  # estimated bounding box of this facade view
    contour: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    seed_region_bbox: Optional[tuple] = None  # original detector bbox kept for scene-specific rebuilds
    seed_contour: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    socle_bbox: Optional[tuple] = None
    socle_contour: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    socle_profile: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    area_px: float = 0.0
    source: str = "ocr"
    socle_excluded_area: float = 0.0
    scene_type: str = "unknown"


@dataclass
class ParsedElement:
    label: str
    element_type: str  # "window" or "door"
    position: tuple  # (x, y) center
    bbox: Optional[tuple] = None  # (x, y, w, h)
    contour: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    area_px: float = 0.0
    area: float = 0.0
    width: float = 0.0
    height: float = 0.0
    parent_facade: Optional[str] = None
    source: str = "ocr"


class AnalysisPipeline:
    def __init__(self):
        self.ocr_engine = OCREngine()
        self.color_detector = ColorDetector()
        self.calculator = AreaCalculator()

        self.parsed_facades = []
        self.parsed_elements = []
        self.ocr_results = []
        self.detection_results = {"facades": [], "windows": [], "doors": []}
        self.report = None
        self.area_scale_px_per_m2 = None
        self.linear_scale_px_per_m = None
        self.current_image = None
        self.source_dpi = None
        self.warnings = []
        self.signal_metrics = {}
        self.scale_ratio_used = None
        self.scale_ratio_source = ""
        self.linear_scale_source = ""

    def run(self, image: np.ndarray,
            progress_callback=None,
            manual_linear_scale_px_per_m: Optional[float] = None,
            source_dpi: Optional[float] = None) -> ProjectReport:
        """Run the full analysis pipeline."""
        self.current_image = image
        self.area_scale_px_per_m2 = None
        self.linear_scale_px_per_m = (
            manual_linear_scale_px_per_m
            if manual_linear_scale_px_per_m and manual_linear_scale_px_per_m > 0
            else None
        )
        self.source_dpi = (
            float(source_dpi) if source_dpi and source_dpi > 0 else None
        )
        self.warnings = []
        self.signal_metrics = {}
        self.scale_ratio_used = None
        self.scale_ratio_source = ""
        self.linear_scale_source = (
            "manual_ref"
            if self.linear_scale_px_per_m and self.linear_scale_px_per_m > 0
            else ""
        )
        self._ml_detection_used = False
        self._ml_socles = []
        self._ml_guard_reason = ""

        if progress_callback:
            progress_callback("Pas 1/5: Extragere text (OCR)...", 0.1)
        self.ocr_results = self.ocr_engine.extract_all_text(image)

        if progress_callback:
            progress_callback("Pas 2/5: Parsare date din text...", 0.3)
        self.parsed_facades = self._parse_facades_from_ocr()
        self.parsed_elements = self._parse_elements_from_ocr()

        if progress_callback:
            progress_callback("Pas 3/5: Detectie culori (fatade/ferestre/usi)...",
                              0.5)
        color_results = self.color_detector.detect_all(image)
        color_facades = color_results["facades"]
        color_windows = color_results["windows"]
        color_doors = color_results["doors"]
        photo_facades = list(color_facades)
        photo_windows = list(color_windows)
        photo_doors = list(color_doors)

        # Hybrid detection: if no markup colors found (photo mode) and ML is
        # enabled, replace photo-detected openings with ML detections.
        use_ml = os.environ.get("DQP_USE_ML", "").strip().lower() in ("1", "true", "yes")
        has_markup = any(
            "photo" not in (r.color_detected or "").lower()
            and "inferred" not in (r.color_detected or "").lower()
            for r in color_facades
        )
        # Also check raw pixel colors: if significant cyan/blue content exists,
        # this is a markup image even if the detector classified it as "photo".
        if not has_markup and image is not None and use_ml:
            hsv_check = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h_ch, s_ch = hsv_check[:, :, 0], hsv_check[:, :, 1]
            cyan_blue = ((h_ch >= 75) & (h_ch <= 130) & (s_ch > 30))
            if cyan_blue.sum() / max(1, cyan_blue.size) > 0.05:
                has_markup = True  # >5% cyan/blue pixels = markup image
        if use_ml and not has_markup:
            try:
                from core.foundation_segmentation import is_available, run_foundation_segmentation
                if is_available():
                    ml_results = run_foundation_segmentation(image, device="auto")
                    ml_facades = ml_results.get("facades", [])
                    ml_windows = ml_results.get("windows", [])
                    ml_doors = ml_results.get("doors", [])
                    ml_socles = ml_results.get("socles", [])
                    keep_photo_openings = self._should_keep_photo_openings_for_flat_scene(
                        photo_facades,
                        photo_windows,
                        photo_doors,
                        ml_windows,
                        ml_doors,
                    )
                    # Replace photo-detected openings with ML detections only when
                    # flat-facade routing does not depend on the photo central door cluster.
                    if (ml_windows or ml_doors) and not keep_photo_openings:
                        color_windows = ml_windows
                        color_doors = ml_doors
                        self._ml_detection_used = True
                    elif keep_photo_openings:
                        self._ml_guard_reason = "flat_long_facade_kept_photo_openings"
                    # Use ML facades only if rule-based found none
                    if not color_facades and ml_facades:
                        color_facades = ml_facades
                    # Store ML socles for later use
                    self._ml_socles = ml_socles
            except Exception:
                pass  # ML unavailable — fall back to rules silently

        if (USE_PHOTO_WINDOW_VALIDATOR or USE_PHOTO_ROI_REFINER) and not getattr(self, '_ml_detection_used', False):
            color_windows = self._apply_photo_refinement(
                color_windows, color_facades, image
            )

        element_regions = color_windows + color_doors
        if element_regions:
            self.ocr_engine.enrich_regions(image, element_regions)

        if progress_callback:
            progress_callback("Pas 4/5: Corelare OCR + detectii...", 0.7)

        self._merge_facade_regions(color_facades, image.shape)
        self._merge_color_with_ocr(color_windows, "window")
        self._merge_color_with_ocr(color_doors, "door")
        self._assign_parents()
        self._classify_facade_scenes()
        self._route_photo_scene_refinements()
        self._assign_parents()
        self._classify_facade_scenes()
        self._prune_scene_specific_openings()
        self._filter_elements_by_facade_proximity()
        self._propagate_known_areas()
        self._calibrate_linear_scale_from_ocr()
        self._apply_default_scale_assumption()
        self._estimate_element_dimensions_from_scale()
        self._infer_missing_areas_from_scale()
        self._estimate_element_dimensions_from_scale()
        self._propagate_known_areas()
        self._infer_facade_areas_from_scale()
        self._exclude_socle_from_facade_areas()
        self._clip_elements_to_socle_top()
        self._clip_elements_to_facade_bottom()
        self._evaluate_signal_quality(color_facades, color_windows, color_doors)

        if progress_callback:
            progress_callback("Pas 5/5: Calcul raport final...", 0.9)
        self.report = self._build_report()

        self._build_detection_results()

        if progress_callback:
            progress_callback("Analiza completa.", 1.0)

        return self.report

    def _apply_photo_refinement(
        self,
        color_windows: list,
        color_facades: list,
        image: np.ndarray,
    ) -> list:
        """Apply validator and optionally refiner to photo window candidates. Fallback to original if no signal."""
        from core.window_validator import validate
        from core.photo_refiner import refine

        has_photo = any(
            (r.color_detected or "").lower().find("photo") >= 0
            for r in color_windows
        )
        if not has_photo:
            return color_windows

        try:
            if USE_PHOTO_WINDOW_VALIDATOR:
                validated = validate(color_windows, image, color_facades)
                if len(validated) == 0 and len(color_windows) > 0:
                    return color_windows
                color_windows = validated

            if USE_PHOTO_ROI_REFINER:
                refined_list = []
                for r in color_windows:
                    refined = refine(image, r, method="grabcut")
                    refined_list.append(refined if refined else r)
                color_windows = refined_list
        except Exception:
            return color_windows

        return color_windows

    def _evaluate_signal_quality(self, color_facades: list,
                                 color_windows: list, color_doors: list):
        """Assess if the current input has enough signal for reliable automation."""
        facade_count = len(self.parsed_facades)
        facade_area_count = sum(1 for f in self.parsed_facades if f.total_area > 0)

        ocr_window_count = sum(
            1 for e in self.parsed_elements if e.element_type == "window"
        )
        ocr_door_count = sum(
            1 for e in self.parsed_elements if e.element_type == "door"
        )
        ocr_elements_with_area = sum(1 for e in self.parsed_elements if e.area > 0)
        elements_with_dimensions = sum(
            1 for e in self.parsed_elements if e.width > 0 and e.height > 0
        )
        facades_with_socle_exclusion = sum(
            1 for f in self.parsed_facades if f.socle_excluded_area > 0
        )

        color_facade_count = len(color_facades)
        color_window_count = len(color_windows)
        color_door_count = len(color_doors)

        self.signal_metrics = {
            "facades_total": facade_count,
            "facades_with_area": facade_area_count,
            "ocr_windows": ocr_window_count,
            "ocr_doors": ocr_door_count,
            "ocr_elements_with_area": ocr_elements_with_area,
            "elements_with_dimensions": elements_with_dimensions,
            "facades_with_socle_exclusion": facades_with_socle_exclusion,
            "color_facades": color_facade_count,
            "color_windows": color_window_count,
            "color_doors": color_door_count,
            "linear_scale_px_per_m": (
                round(self.linear_scale_px_per_m, 3)
                if self.linear_scale_px_per_m else 0
            ),
            "linear_scale_source": self.linear_scale_source or "none",
            "area_scale_px_per_m2": (
                round(self.area_scale_px_per_m2, 3)
                if self.area_scale_px_per_m2 else 0
            ),
            "scale_ratio_used": (
                round(self.scale_ratio_used, 2) if self.scale_ratio_used else 0
            ),
            "scale_ratio_source": self.scale_ratio_source or "none",
        }

        if facade_area_count == 0 and ocr_elements_with_area == 0:
            self.warnings.append(
                "Semnal automat insuficient: nu au fost detectate cote/suprafete "
                "de incredere. Pentru imagini foto simple folositi modul asistat "
                "(referinta + completare manuala)."
            )

        if facade_count == 0 and color_facade_count == 0:
            self.warnings.append(
                "Nu a fost identificata nicio fatada valida in inputul curent."
            )

        if facade_count > 0 and facade_area_count == 0 and not self.linear_scale_px_per_m:
            self.warnings.append(
                "Fatada a fost detectata geometric, dar nu exista calibrare scara."
                " Introduceti Ref. (m) + Ref. (px) pentru arii in m²."
            )

        if (ocr_window_count + ocr_door_count) > 0 and elements_with_dimensions == 0:
            self.warnings.append(
                "Cantitatile liniare (glaf/picurator/coltare) pot fi incomplete "
                "fara calibrare liniara px/m."
            )

    def _parse_facades_from_ocr(self) -> list:
        """Find FATADA labels and associated area values from OCR results.

        Two types of FATADA text exist:
        1. Section titles: "FATADA PRINCIPALA", "FATADA POSTERIOARA" etc.
           These identify the facade but don't have area values nearby.
        2. Area labels: Just "FATADA" in red text with area values below.
           These contain the actual measurements.

        Strategy: find section titles first, then find area labels and
        associate them with the nearest section title.
        """
        section_titles = []
        area_labels = []

        for r in self.ocr_results:
            text_up = r.text.strip().upper()
            if not re.search(r"FATAD", text_up):
                continue

            x_c = (r.bbox[0] + r.bbox[2]) // 2
            y_c = (r.bbox[1] + r.bbox[3]) // 2

            has_qualifier = any(
                kw in text_up for kw in
                ["PRINCIPALA", "PRINCIPAL", "POSTERIOARA", "POSTERIOR",
                 "LATERALA", "LATERAL", "STANGA", "DREAPTA", "STINGA"]
            )

            if has_qualifier:
                section_titles.append({
                    "name": r.text.strip(),
                    "position": (x_c, y_c),
                    "bbox": r.bbox,
                })
            else:
                nearby = self._find_nearby_numbers(
                    r.bbox, max_distance=100, prefer_below=True
                )
                total_area = 0.0
                net_area = 0.0
                if len(nearby) >= 2:
                    s = sorted(nearby, reverse=True)
                    total_area = s[0]
                    net_area = s[1]
                elif len(nearby) == 1:
                    total_area = nearby[0]

                if total_area > 0:
                    area_labels.append({
                        "position": (x_c, y_c),
                        "total_area": total_area,
                        "net_area": net_area,
                        "bbox": r.bbox,
                    })

        # If no section titles found, use area labels directly
        if not section_titles:
            facades = []
            for i, al in enumerate(area_labels):
                facades.append(ParsedFacade(
                    name=f"Fatada {i+1}",
                    position=al["position"],
                    total_area=al["total_area"],
                    net_area=al["net_area"],
                ))
            return facades

        # Global optimal matching: compute all distances, assign closest pairs
        # to avoid greedy mismatch
        if area_labels:
            pairs = []
            for si, st in enumerate(section_titles):
                for ai, al in enumerate(area_labels):
                    sx, sy = st["position"]
                    ax, ay = al["position"]
                    dist = ((sx - ax) ** 2 + (sy - ay) ** 2) ** 0.5
                    if dist < 500:
                        pairs.append((dist, si, ai))

            pairs.sort(key=lambda x: x[0])
            used_st = set()
            used_al = set()
            assignments = {}

            for dist, si, ai in pairs:
                if si in used_st or ai in used_al:
                    continue
                assignments[si] = ai
                used_st.add(si)
                used_al.add(ai)

            facades = []
            for si, st in enumerate(section_titles):
                if si in assignments:
                    al = area_labels[assignments[si]]
                    total = al["total_area"]
                    net = al["net_area"]
                else:
                    total = 0.0
                    net = 0.0

                facades.append(ParsedFacade(
                    name=st["name"],
                    position=st["position"],
                    total_area=total,
                    net_area=net,
                ))

            for ai, al in enumerate(area_labels):
                if ai not in used_al:
                    facades.append(ParsedFacade(
                        name="Fatada (neasociata)",
                        position=al["position"],
                        total_area=al["total_area"],
                        net_area=al["net_area"],
                    ))
        else:
            facades = [
                ParsedFacade(name=st["name"], position=st["position"])
                for st in section_titles
            ]

        return facades

    def _merge_section_titles(self, titles: list) -> list:
        """Merge section title fragments that span multiple OCR blocks."""
        if len(titles) <= 1:
            return titles

        merged = []
        used = set()
        for i, t1 in enumerate(titles):
            if i in used:
                continue
            combined = t1["name"]
            pos = t1["position"]
            for j, t2 in enumerate(titles):
                if j <= i or j in used:
                    continue
                dx = abs(t1["position"][0] - t2["position"][0])
                dy = abs(t1["position"][1] - t2["position"][1])
                if dx < 250 and dy < 30:
                    combined += " " + t2["name"]
                    used.add(j)
            merged.append({"name": combined, "position": pos, "bbox": t1["bbox"]})
            used.add(i)
        return merged

    def _parse_elements_from_ocr(self) -> list:
        """Find F2, U2 type labels and their associated dimensions."""
        elements = []

        for r in self.ocr_results:
            text_upper = r.text.strip().upper()

            # Window patterns: F2, F2-1, F 2, etc.
            w_match = re.match(r"^F\s*(\d+)(?:\s*[-_]\s*(\d+))?$", text_upper)
            if w_match:
                x_c = (r.bbox[0] + r.bbox[2]) // 2
                y_c = (r.bbox[1] + r.bbox[3]) // 2
                nearby = self._find_nearby_numbers(
                    r.bbox, max_distance=120, max_value=20
                )
                area = nearby[0] if nearby else 0.0

                elements.append(ParsedElement(
                    label=r.text.strip(),
                    element_type="window",
                    position=(x_c, y_c),
                    area=area,
                    source="ocr",
                ))
                continue

            # Door patterns: U2, U2-1, U 2, etc.
            d_match = re.match(r"^U\s*(\d+)(?:\s*[-_]\s*(\d+))?$", text_upper)
            if d_match:
                x_c = (r.bbox[0] + r.bbox[2]) // 2
                y_c = (r.bbox[1] + r.bbox[3]) // 2
                nearby = self._find_nearby_numbers(
                    r.bbox, max_distance=80, max_value=8
                )
                area = nearby[0] if nearby else 0.0

                elements.append(ParsedElement(
                    label=r.text.strip(),
                    element_type="door",
                    position=(x_c, y_c),
                    area=area,
                    source="ocr",
                ))

        return elements

    def _find_nearby_numbers(self, ref_bbox: tuple, max_distance: int = 150,
                             prefer_below: bool = False,
                             max_value: float = 500) -> list:
        """Find numeric values from OCR results near a reference bbox."""
        ref_cx = (ref_bbox[0] + ref_bbox[2]) // 2
        ref_cy = (ref_bbox[1] + ref_bbox[3]) // 2

        candidates = []
        for r in self.ocr_results:
            if r.parsed_value is None:
                continue
            if r.parsed_value < 0.01 or r.parsed_value > max_value:
                continue

            cx = (r.bbox[0] + r.bbox[2]) // 2
            cy = (r.bbox[1] + r.bbox[3]) // 2
            dx = abs(cx - ref_cx)
            dy = cy - ref_cy if prefer_below else abs(cy - ref_cy)

            if prefer_below and dy < -30:
                continue

            dist = (dx ** 2 + abs(dy) ** 2) ** 0.5
            if dist < max_distance:
                candidates.append((dist, r.parsed_value))

        candidates.sort(key=lambda x: x[0])
        return [val for _, val in candidates]

    def _find_nearby_values_to_point(self, point: tuple, max_distance: int = 140,
                                     max_value: float = 80) -> list:
        """Find OCR numeric values around a point, preferring area-like values."""
        px, py = point
        candidates = []
        for r in self.ocr_results:
            if r.parsed_value is None:
                continue
            if r.parsed_value < 0.01 or r.parsed_value > max_value:
                continue

            cx = (r.bbox[0] + r.bbox[2]) // 2
            cy = (r.bbox[1] + r.bbox[3]) // 2
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if dist > max_distance:
                continue

            # Favor values explicitly recognized as area.
            score = dist * (0.8 if r.unit in ("m²", "m2", "mp") else 1.0)
            candidates.append((score, r.parsed_value))

        candidates.sort(key=lambda x: x[0])
        return [val for _, val in candidates]

    def _merge_facade_regions(self, color_facades: list, image_shape: tuple):
        """Attach color-detected facade regions to OCR-parsed facades."""
        if not self.parsed_facades and color_facades:
            for idx, cf in enumerate(color_facades, 1):
                self.parsed_facades.append(ParsedFacade(
                    name=f"Fatada {idx}",
                    position=cf.center,
                    total_area=cf.area_m2 or 0.0,
                    net_area=0.0,
                    region_bbox=cf.bbox,
                    contour=cf.contour if isinstance(cf.contour, np.ndarray) else np.array([]),
                    seed_region_bbox=cf.bbox,
                    seed_contour=(cf.contour if isinstance(cf.contour, np.ndarray) else np.array([])),
                    area_px=cf.area_px or 0.0,
                    source=f"color:{cf.color_detected or 'unknown'}",
                ))
            return

        if self.parsed_facades and color_facades:
            pairs = []
            for fi, facade in enumerate(self.parsed_facades):
                fx, fy = facade.position
                for ci, cfac in enumerate(color_facades):
                    cx, cy = cfac.center
                    dist = ((fx - cx) ** 2 + (fy - cy) ** 2) ** 0.5
                    pairs.append((dist, fi, ci))

            pairs.sort(key=lambda x: x[0])
            used_facades = set()
            used_color = set()
            for dist, fi, ci in pairs:
                if fi in used_facades or ci in used_color:
                    continue
                if dist > 600:
                    continue
                self.parsed_facades[fi].region_bbox = color_facades[ci].bbox
                self.parsed_facades[fi].contour = (
                    color_facades[ci].contour
                    if isinstance(color_facades[ci].contour, np.ndarray)
                    else np.array([])
                )
                self.parsed_facades[fi].seed_region_bbox = color_facades[ci].bbox
                self.parsed_facades[fi].seed_contour = (
                    color_facades[ci].contour
                    if isinstance(color_facades[ci].contour, np.ndarray)
                    else np.array([])
                )
                self.parsed_facades[fi].area_px = color_facades[ci].area_px or 0.0
                detected_src = color_facades[ci].color_detected or ""
                if detected_src:
                    self.parsed_facades[fi].source = f"color:{detected_src}"
                used_facades.add(fi)
                used_color.add(ci)

        self._estimate_facade_regions(image_shape, missing_only=True)

    def _estimate_facade_regions(self, image_shape: tuple,
                                 missing_only: bool = False):
        """Estimate facade bounding boxes from label distribution."""
        h, w = image_shape[:2]

        if not self.parsed_facades:
            return

        for facade in self.parsed_facades:
            if missing_only and facade.region_bbox:
                continue

            fx, fy = facade.position
            left = max(0, fx - w // 3)
            right = min(w, fx + w // 3)
            top = max(0, fy - h // 6)
            bottom = min(h, fy + h // 4)

            for other in self.parsed_facades:
                if other is facade:
                    continue
                ox, oy = other.position
                if abs(oy - fy) < 80:
                    if ox > fx:
                        right = min(right, (fx + ox) // 2)
                    else:
                        left = max(left, (fx + ox) // 2)
                else:
                    if oy > fy:
                        bottom = min(bottom, (fy + oy) // 2)
                    elif oy < fy:
                        top = max(top, (fy + oy) // 2)

            facade.region_bbox = (left, top, max(1, right - left),
                                  max(1, bottom - top))

    def _merge_color_with_ocr(self, color_regions: list, element_type: str):
        """Cross-reference color-detected regions with OCR-parsed elements."""
        ocr_elements = [e for e in self.parsed_elements
                        if e.element_type == element_type]
        if not color_regions:
            return

        max_dist = 220 if element_type == "window" else 240
        pairs = []
        for ci, cr in enumerate(color_regions):
            if element_type == "door":
                if self._contains_facade_text(cr):
                    continue
                if not self._has_door_signal(cr):
                    nearest = self._distance_to_nearest_ocr_element(
                        cr.center, ocr_elements
                    )
                    if nearest > 90:
                        continue

            cx, cy = cr.center
            ck = self._label_key(cr.label)
            for ei, elem in enumerate(ocr_elements):
                ex, ey = elem.position
                dist = ((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5
                if dist > max_dist:
                    continue
                ek = self._label_key(elem.label)
                score = dist
                if ck and ek and ck != ek and dist > 60:
                    score += 120
                pairs.append((score, ci, ei))

        matched_colors = set()
        matched_ocr = set()
        pairs.sort(key=lambda x: x[0])
        for _, ci, ei in pairs:
            if ci in matched_colors or ei in matched_ocr:
                continue
            cr = color_regions[ci]
            elem = ocr_elements[ei]
            matched_colors.add(ci)
            matched_ocr.add(ei)

            if elem.area <= 0 and cr.area_m2:
                elem.area = cr.area_m2
            if cr.area_m2 is None and elem.area > 0:
                cr.area_m2 = elem.area

            if not elem.label:
                elem.label = cr.label or elem.label
            cr.label = elem.label or cr.label
            elem.position = cr.center
            elem.bbox = cr.bbox
            elem.contour = (
                cr.contour if isinstance(cr.contour, np.ndarray) else np.array([])
            )
            elem.area_px = cr.area_px
            color_tag = cr.color_detected or "unknown"
            if "color" not in elem.source:
                elem.source = f"ocr+color:{color_tag}"
            elif elem.source == "color":
                elem.source = f"color:{color_tag}"
            if cr.width_m and cr.height_m:
                elem.width = cr.width_m
                elem.height = cr.height_m

        for ci, cr in enumerate(color_regions):
            if ci in matched_colors:
                continue

            if element_type == "door":
                if self._contains_facade_text(cr):
                    continue
                if not self._has_door_signal(cr):
                    continue

            inferred_max = 12 if element_type == "window" else 6.5
            inferred = self._find_nearby_values_to_point(
                cr.center,
                max_distance=130,
                max_value=inferred_max,
            )
            self.parsed_elements.append(ParsedElement(
                label=cr.label,
                element_type=element_type,
                position=cr.center,
                bbox=cr.bbox,
                contour=cr.contour if isinstance(cr.contour, np.ndarray) else np.array([]),
                area_px=cr.area_px,
                area=cr.area_m2 or (inferred[0] if inferred else 0.0),
                width=cr.width_m or 0.0,
                height=cr.height_m or 0.0,
                source=f"color:{cr.color_detected or 'unknown'}",
            ))

    def _build_local_facade_mask(self, facade: ParsedFacade):
        if self.current_image is None or not facade.region_bbox:
            return None, None
        x, y, w, h = facade.region_bbox
        if w < 8 or h < 8:
            return None, None
        mask = np.zeros((h, w), dtype=np.uint8)
        if isinstance(facade.contour, np.ndarray) and facade.contour.size >= 6:
            c_local = facade.contour.astype(np.int32).copy()
            c_local[:, 0, 0] -= x
            c_local[:, 0, 1] -= y
            cv2.drawContours(mask, [c_local], -1, 255, -1)
        else:
            mask[:, :] = 255
        return mask, facade.region_bbox

    @staticmethod
    def _keep_bottom_connected_mask(mask: np.ndarray,
                                    min_area_ratio: float = 0.06,
                                    bottom_band_ratio: float = 0.05) -> np.ndarray:
        """Keep only meaningful facade components that remain connected to the lower body."""
        if mask is None or mask.size == 0:
            return mask

        roi_h, roi_w = mask.shape[:2]
        if roi_h < 20 or roi_w < 20:
            return mask

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n_labels <= 1:
            return mask

        keep = np.zeros_like(mask)
        min_area = max(40, int(round(roi_h * roi_w * float(min_area_ratio))))
        band_h = max(3, int(round(roi_h * float(bottom_band_ratio))))
        band_y0 = max(0, roi_h - band_h)
        kept_any = False

        for idx in range(1, n_labels):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            if np.any(labels[band_y0:, :] == idx):
                keep[labels == idx] = 255
                kept_any = True

        return keep if kept_any else mask

    def _recover_gable_silhouette_from_seed(self, facade: ParsedFacade):
        """Recover a full gable silhouette from long diagonal roof lines in the seed bbox."""
        if self.current_image is None or not facade.seed_region_bbox:
            return None

        x, y, w, h = facade.seed_region_bbox
        if w < 140 or h < 140:
            return None
        roi = self.current_image[y:y + h, x:x + w]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        edges = cv2.Canny(gray, 40, 140)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=max(25, int(w * 0.03)),
            minLineLength=max(40, int(w * 0.12)),
            maxLineGap=max(18, int(w * 0.03)),
        )
        if lines is None:
            return None

        neg_candidates = []
        pos_candidates = []
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, line)
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < max(25, int(w * 0.08)):
                continue
            slope = dy / float(dx)
            abs_slope = abs(slope)
            if abs_slope < 0.18 or abs_slope > 2.6:
                continue
            length = float((dx * dx + dy * dy) ** 0.5)
            mean_y = (y1 + y2) * 0.5
            if mean_y > h * 0.78:
                continue
            line_min_x = min(x1, x2)
            line_max_x = max(x1, x2)
            line_min_y = min(y1, y2)
            if slope < 0:
                if line_min_x > w * 0.18 or line_max_x < w * 0.18 or line_max_x > w * 0.55:
                    continue
                if line_min_y > h * 0.45:
                    continue
                neg_candidates.append((length, (x1, y1, x2, y2)))
            else:
                if line_max_x < w * 0.78 or line_min_x > w * 0.45:
                    continue
                if line_min_y > h * 0.45:
                    continue
                pos_candidates.append((length, (x1, y1, x2, y2)))

        if not neg_candidates or not pos_candidates:
            return None

        neg_candidates.sort(key=lambda item: item[0], reverse=True)
        pos_candidates.sort(key=lambda item: item[0], reverse=True)
        best = None
        for _, neg in neg_candidates[:8]:
            nx1, ny1, nx2, ny2 = neg
            an = (ny2 - ny1) / float(nx2 - nx1)
            bn = ny1 - an * nx1
            for _, pos in pos_candidates[:8]:
                px1, py1, px2, py2 = pos
                ap = (py2 - py1) / float(px2 - px1)
                bp = py1 - ap * px1
                if abs(an - ap) < 1e-4:
                    continue
                apex_x = (bp - bn) / (an - ap)
                apex_y = an * apex_x + bn
                if not (w * 0.25 <= apex_x <= w * 0.75):
                    continue
                if not (0 <= apex_y <= h * 0.42):
                    continue
                score = float(_ + _)
                best = (score, neg, pos, apex_x, apex_y)
                break
            if best is not None:
                break

        if best is None:
            return None

        _, neg, pos, apex_x, apex_y = best
        nx1, ny1, nx2, ny2 = neg
        px1, py1, px2, py2 = pos
        an = (ny2 - ny1) / float(nx2 - nx1)
        bn = ny1 - an * nx1
        ap = (py2 - py1) / float(px2 - px1)
        bp = py1 - ap * px1

        left_x = max(0, min(nx1, nx2))
        right_x = min(w - 1, max(px1, px2))
        if right_x - left_x < max(80, int(w * 0.45)):
            return None

        left_y = int(round(an * left_x + bn))
        right_y = int(round(ap * right_x + bp))
        apex_x_i = int(round(apex_x))
        apex_y_i = int(round(apex_y))
        left_y = max(0, min(h - 1, left_y))
        right_y = max(0, min(h - 1, right_y))
        apex_x_i = max(0, min(w - 1, apex_x_i))
        apex_y_i = max(0, min(h - 1, apex_y_i))
        bottom_y = h - 1

        contour_local = np.array(
            [
                [left_x, left_y],
                [apex_x_i, apex_y_i],
                [right_x, right_y],
                [right_x, bottom_y],
                [left_x, bottom_y],
            ],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        area = float(cv2.contourArea(contour_local))
        if area < float(w * h) * 0.22:
            return None

        contour_global = contour_local.copy()
        contour_global[:, 0, 0] += x
        contour_global[:, 0, 1] += y
        bx, by, bw, bh = cv2.boundingRect(contour_local)
        return contour_global, (x + bx, y + by, bw, bh), area

    def _classify_single_photo_scene(self, facade: ParsedFacade) -> str:
        src = (facade.source or '').lower()
        if 'merged' in src:
            return 'composite_stepped_facade'
        if self.current_image is None or not facade.region_bbox:
            return 'unknown'

        mask, bbox = self._build_local_facade_mask(facade)
        if mask is None or bbox is None:
            return 'unknown'
        _, _, w, h = bbox
        if w < 60 or h < 60:
            return 'unknown'

        openings = [
            e for e in self.parsed_elements
            if e.parent_facade == facade.name and e.bbox
        ]
        opening_count = len(openings)

        top_profile = []
        for cx in range(w):
            ys = np.where(mask[:, cx] > 0)[0]
            if ys.size > 0:
                top_profile.append(int(ys[0]))
        if len(top_profile) < max(18, int(w * 0.12)):
            if opening_count <= int(getattr(config, 'PHOTO_SCENE_SPARSE_MAX_OPENINGS', 1)):
                return 'sparse_openings_flat'
            return 'unknown'

        top_profile = np.array(top_profile, dtype=np.float32)
        top_iqr = float(np.percentile(top_profile, 75) - np.percentile(top_profile, 25))
        top_span = float(np.percentile(top_profile, 95) - np.percentile(top_profile, 5))
        aspect = float(w) / max(1.0, float(h))

        edge_n = max(4, int(len(top_profile) * 0.18))
        edge_vals = np.concatenate([top_profile[:edge_n], top_profile[-edge_n:]])
        center_start = max(0, int(len(top_profile) * 0.35))
        center_end = min(len(top_profile), max(center_start + 4, int(len(top_profile) * 0.65)))
        center_vals = top_profile[center_start:center_end]
        gable_peak = 0.0
        if edge_vals.size > 0 and center_vals.size > 0:
            gable_peak = float(np.median(edge_vals) - np.median(center_vals))

        opening_top_spread = 0.0
        opening_center_spread = 0.0
        if openings:
            opening_tops = np.array([max(0, e.bbox[1] - bbox[1]) for e in openings], dtype=np.float32)
            opening_centers = np.array([
                max(0.0, (e.bbox[1] + e.bbox[3] * 0.5) - bbox[1])
                for e in openings
            ], dtype=np.float32)
            if opening_tops.size > 0:
                opening_top_spread = float(np.percentile(opening_tops, 90) - np.percentile(opening_tops, 10))
            if opening_centers.size > 0:
                opening_center_spread = float(np.percentile(opening_centers, 90) - np.percentile(opening_centers, 10))

        if (
            aspect <= float(getattr(config, 'PHOTO_SCENE_GABLE_MAX_ASPECT', 2.15))
            and top_span >= h * float(getattr(config, 'PHOTO_SCENE_GABLE_TOP_SPAN_RATIO', 0.28))
            and (
                top_iqr >= h * float(getattr(config, 'PHOTO_SCENE_GABLE_TOP_IQR_RATIO', 0.12))
                or gable_peak >= h * float(getattr(config, 'PHOTO_SCENE_GABLE_PEAK_DELTA_RATIO', 0.10))
            )
        ):
            return 'gable_facade'

        if (
            opening_count >= int(getattr(config, 'PHOTO_SCENE_COMPOSITE_MIN_OPENINGS', 4))
            and aspect >= float(getattr(config, 'PHOTO_SCENE_FLAT_MIN_ASPECT', 2.0))
            and (
                opening_top_spread >= h * float(getattr(config, 'PHOTO_SCENE_COMPOSITE_OPENING_TOP_SPREAD_RATIO', 0.12))
                or opening_center_spread >= h * float(getattr(config, 'PHOTO_SCENE_COMPOSITE_OPENING_CENTER_SPREAD_RATIO', 0.16))
            )
        ):
            return 'composite_stepped_facade'

        if self._has_flat_central_opening_cluster(facade, openings):
            return 'flat_long_facade'

        if (
            aspect >= float(getattr(config, 'PHOTO_SCENE_FLAT_MIN_ASPECT', 2.0))
            and opening_count <= int(getattr(config, 'PHOTO_SCENE_SPARSE_MAX_OPENINGS', 1))
        ):
            return 'sparse_openings_flat'

        if (
            top_span >= h * float(getattr(config, 'PHOTO_SCENE_BORDERLINE_TOP_SPAN_RATIO', 0.08))
            and opening_count <= int(getattr(config, 'PHOTO_SCENE_BORDERLINE_MAX_OPENINGS', 4))
        ):
            return 'sparse_openings_flat'

        if (
            aspect >= float(getattr(config, 'PHOTO_SCENE_FLAT_MIN_ASPECT', 2.0))
            and top_iqr <= h * float(getattr(config, 'PHOTO_SCENE_FLAT_TOP_IQR_RATIO', 0.10))
        ):
            return 'flat_long_facade'

        if opening_count <= int(getattr(config, 'PHOTO_SCENE_BORDERLINE_MAX_OPENINGS', 4)):
            return 'sparse_openings_flat'

        return 'flat_long_facade'

    def _classify_facade_scenes(self):
        for facade in self.parsed_facades:
            scene = 'unknown'
            src = (facade.source or '').lower()
            if 'photo' in src or 'inferred' in src:
                scene = self._classify_single_photo_scene(facade)
            facade.scene_type = scene

    def _prune_scene_specific_openings(self):
        if not self.parsed_facades or not self.parsed_elements:
            return

        kept = []
        for elem in self.parsed_elements:
            keep_elem = True
            src = (elem.source or '').lower()
            parent = next((f for f in self.parsed_facades if f.name == elem.parent_facade), None)
            if parent is not None and elem.bbox:
                scene = parent.scene_type
                ex, ey, ew, eh = elem.bbox
                near = []
                for other in self.parsed_elements:
                    if other is elem or other.parent_facade != elem.parent_facade or not other.bbox:
                        continue
                    ox, oy, ow, oh = other.bbox
                    overlap_y = max(0, min(ey + eh, oy + oh) - max(ey, oy))
                    gap_x = max(0, max(ox - (ex + ew), ex - (ox + ow)))
                    if overlap_y < min(eh, oh) * 0.20:
                        continue
                    if gap_x > max(90, int(max(ew, ow) * 1.4)):
                        continue
                    near.append(other)

                if scene == 'gable_facade':
                    if elem.element_type == 'door' and 'ocr' not in src and 'photo-door' in src:
                        near_windows = [o for o in near if o.element_type == 'window']
                        if len(near_windows) <= 1:
                            keep_elem = False
                    elif elem.element_type == 'window' and 'ocr' not in src:
                        cy_ratio = ((ey + eh * 0.5) - parent.region_bbox[1]) / max(1.0, float(parent.region_bbox[3]))
                        facade_windows = [o for o in self.parsed_elements if o.parent_facade == elem.parent_facade and o.element_type == 'window']
                        area_small = (elem.area > 0 and elem.area < 0.45)
                        if parent.region_bbox:
                            _, _, pw, ph = parent.region_bbox
                            area_small = area_small or ((ew * eh) <= (pw * ph * 0.015))
                        if area_small:
                            near_doors = [o for o in near if o.element_type == 'door' and 'ocr' not in (o.source or '').lower()]
                            if near_doors:
                                keep_elem = False
                            elif len(facade_windows) <= 1 and cy_ratio >= 0.48:
                                keep_elem = False

                elif scene == 'sparse_openings_flat':
                    if elem.element_type == 'door' and 'ocr' not in src and 'photo-door' in src:
                        side_windows = [o for o in near if o.element_type == 'window' and 'sidelight' in (o.source or '').lower()]
                        opposite_windows = [o for o in near if o.element_type == 'window' and 'sidelight' not in (o.source or '').lower()]
                        left_hits = [o for o in side_windows if (o.bbox[0] + o.bbox[2]) <= ex]
                        right_hits = [o for o in side_windows if o.bbox[0] >= (ex + ew)]
                        if side_windows and (not left_hits or not right_hits):
                            if len(opposite_windows) <= 1:
                                keep_elem = False
                    elif elem.element_type == 'window' and 'sidelight' in src:
                        near_doors = [o for o in near if o.element_type == 'door' and 'photo-door' in (o.source or '').lower() and 'ocr' not in (o.source or '').lower()]
                        if near_doors:
                            left_hits = [o for o in near if o.element_type == 'window' and 'sidelight' in (o.source or '').lower() and (o.bbox[0] + o.bbox[2]) <= near_doors[0].bbox[0]]
                            right_hits = [o for o in near if o.element_type == 'window' and 'sidelight' in (o.source or '').lower() and o.bbox[0] >= (near_doors[0].bbox[0] + near_doors[0].bbox[2])]
                            if not left_hits or not right_hits:
                                keep_elem = False

            if keep_elem:
                kept.append(elem)

        self.parsed_elements = kept

    def _refine_gable_photo_facades(self):
        """Conservative cleanup for gable/fronton facades without flat cap-line logic."""
        if self.current_image is None:
            return

        for facade in self.parsed_facades:
            src = (facade.source or "").lower()
            if "photo" not in src:
                continue
            if facade.scene_type != "gable_facade":
                continue
            if not facade.region_bbox:
                continue

            mask, bbox = self._build_local_facade_mask(facade)
            if mask is None or bbox is None:
                continue

            seed_bbox = facade.seed_region_bbox if facade.seed_region_bbox else bbox
            if seed_bbox and bbox and seed_bbox[2] >= int(bbox[2] * 1.35):
                recovered = self._recover_gable_silhouette_from_seed(facade)
                if recovered is not None:
                    contour_global, recovered_bbox, recovered_area = recovered
                    facade.contour = contour_global
                    facade.region_bbox = recovered_bbox
                    facade.area_px = recovered_area
                    continue

            x, y, w, h = bbox
            if w < 80 or h < 80:
                continue

            work = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            )
            work = cv2.morphologyEx(
                work,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            )
            work = self._keep_bottom_connected_mask(work)
            work = self.color_detector._trim_side_spikes(work)

            contours, _ = cv2.findContours(
                work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            base = max(contours, key=cv2.contourArea)
            base_area = float(cv2.contourArea(base))
            if base_area < max(600.0, float(np.count_nonzero(mask)) * 0.45):
                continue

            draw_contour = base
            smooth_window = max(7, int(round(w * 0.014)))
            if smooth_window % 2 == 0:
                smooth_window += 1
            profile_contour = self._profile_contour_from_mask(
                work,
                support_mask=work,
                smooth_window=smooth_window,
                prefer_thick_columns=False,
            )
            if profile_contour is not None:
                profile_area = float(cv2.contourArea(profile_contour))
                if base_area * 0.82 <= profile_area <= base_area * 1.18:
                    draw_contour = profile_contour

            bx, by, bw, bh = cv2.boundingRect(draw_contour)
            center_shift = abs((bx + bw * 0.5) - (w * 0.5))
            if bw < int(round(w * 0.72)) or bh < int(round(h * 0.82)):
                continue
            if center_shift > w * 0.18:
                continue

            contour_global = draw_contour.copy()
            contour_global[:, 0, 0] += x
            contour_global[:, 0, 1] += y
            facade.contour = contour_global
            facade.region_bbox = (x + bx, y + by, bw, bh)
            facade.area_px = float(cv2.contourArea(draw_contour))

            if facade.seed_region_bbox and facade.seed_region_bbox[2] >= int(facade.region_bbox[2] * 1.35):
                recovered = self._recover_gable_silhouette_from_seed(facade)
                if os.environ.get("DQP_DEBUG_GEOM"):
                    print("DEBUG_GABLE_FALLBACK", "seed", facade.seed_region_bbox, "cur", facade.region_bbox, "has_recovered", recovered is not None)
                if recovered is not None:
                    recovered_contour, recovered_bbox, recovered_area = recovered
                    if os.environ.get("DQP_DEBUG_GEOM"):
                        print("DEBUG_GABLE_FALLBACK_BBOX", recovered_bbox)
                    if recovered_bbox[2] > int(facade.region_bbox[2] * 1.20):
                        facade.contour = recovered_contour
                        facade.region_bbox = recovered_bbox
                        facade.area_px = recovered_area

    def _extend_composite_stepped_facade_top(self):
        """Extend composite stepped facade upward to capture upper line-drawn volumes.

        The photo-segmentation seed selects the dominant wall body which may
        miss upper stepped volumes whose rows have low full-width occupancy
        (they are mostly white with thin outlines in the right portion only).
        For composite_stepped_facade scenes we scan above the current facade top
        in the right half of the facade width and, when significant non-white
        content is found, merge it into the facade mask and refit the contour.
        """
        if self.current_image is None:
            return
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        img_h, img_w = gray.shape[:2]

        for facade in self.parsed_facades:
            if facade.scene_type != "composite_stepped_facade":
                continue
            src = (facade.source or "").lower()
            if "photo" not in src and "inferred" not in src:
                continue
            if not facade.region_bbox:
                continue

            fx, fy, fw, fh = facade.region_bbox
            if fy <= 0 or fw < 200 or fh < 100:
                continue

            # Only scan upward when the facade top has room to extend.
            max_extend_px = max(50, int(round(fh * 0.35)))
            scan_y0 = max(0, fy - max_extend_px)
            scan_y1 = fy  # exclusive (we want rows above current top)
            if scan_y0 >= scan_y1:
                continue

            # Look in the RIGHT HALF of the facade – upper stepped volumes are
            # typically on the right of a composite stepped facade.
            right_x0 = fx + fw // 2
            right_x1 = min(img_w, fx + fw)
            right_w = max(1, right_x1 - right_x0)

            # Find the topmost row with >= threshold non-white pixels in the
            # right half (non-white = gray < 230).
            occ_threshold = float(getattr(config, "COMPOSITE_EXTEND_OCC_THRESHOLD", 0.04))
            new_top_y = None
            for y in range(scan_y0, scan_y1):
                row = gray[y, right_x0:right_x1]
                non_white = int(np.count_nonzero(row < 230))
                if non_white >= int(right_w * occ_threshold):
                    new_top_y = y
                    break

            if new_top_y is None:
                continue

            # Build extension mask: non-white pixels in the strip above facade top
            # covering the full facade x range (not just the right half so that
            # the bounding box stays consistent).
            strip_h = scan_y1 - new_top_y
            if strip_h < 4:
                continue
            strip = gray[new_top_y:scan_y1, max(0, fx):min(img_w, fx + fw)]
            ext_mask = (strip < 230).astype(np.uint8) * 255
            # Close small gaps in the structural drawing lines.
            ext_mask = cv2.morphologyEx(
                ext_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)),
            )

            ext_area = int(np.count_nonzero(ext_mask))
            if ext_area < max(50, int(strip_h * right_w * 0.05)):
                continue

            # Build the original facade mask in its local coordinate space.
            x0 = max(0, fx)
            y0 = max(0, fy)
            x1 = min(img_w, fx + fw)
            y1 = min(img_h, fy + fh)
            roi_h_orig = y1 - y0
            roi_w_orig = x1 - x0
            orig_mask = np.zeros((roi_h_orig, roi_w_orig), dtype=np.uint8)
            base_contour = facade.contour
            if isinstance(base_contour, np.ndarray) and base_contour.size >= 6:
                c_local = base_contour.astype(np.int32).copy()
                c_local[:, 0, 0] -= x0
                c_local[:, 0, 1] -= y0
                cv2.drawContours(orig_mask, [c_local], -1, 255, -1)
            else:
                orig_mask[:, :] = 255

            # Pad the original mask upward to accommodate the extension.
            combined_h = strip_h + roi_h_orig
            combined_w = roi_w_orig
            combined_mask = np.zeros((combined_h, combined_w), dtype=np.uint8)
            # Extension goes in the top strip_h rows; align x.
            ext_x_offset = max(0, fx) - max(0, fx)  # both start at same x, = 0
            combined_mask[:strip_h, ext_x_offset:ext_x_offset + ext_mask.shape[1]] = ext_mask
            combined_mask[strip_h:, :] = orig_mask

            # Find contour of the combined mask.
            combined_mask_closed = cv2.morphologyEx(
                combined_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
            )
            contours, _ = cv2.findContours(
                combined_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            best = max(contours, key=cv2.contourArea)
            new_area = float(cv2.contourArea(best))
            orig_area_check = float(np.count_nonzero(orig_mask))
            if new_area < orig_area_check * 0.85:
                continue  # Extension shrank the facade — something went wrong.

            # Convert back to global coordinates (offset = (x0, new_top_y)).
            contour_global = best.copy()
            contour_global[:, 0, 0] += x0
            contour_global[:, 0, 1] += new_top_y

            bx, by, bw, bh = cv2.boundingRect(best)
            new_bbox = (x0 + bx, new_top_y + by, bw, bh)

            # Sanity-check: the new bbox must be meaningfully taller.
            if new_bbox[3] <= fh + max(8, int(fh * 0.04)):
                continue

            facade.contour = contour_global
            facade.region_bbox = new_bbox
            facade.area_px = new_area
            # Also update the seed so that later stages use the extended region.
            facade.seed_region_bbox = new_bbox
            facade.seed_contour = contour_global

    def _route_photo_scene_refinements(self):
        self._refine_flat_photo_facades_from_openings()
        self._refine_gable_photo_facades()
        self._extend_composite_stepped_facade_top()

    def _refine_flat_photo_facades_from_openings(self):
        """Refine flat photo facades using a strong cap line above detected openings."""
        if self.current_image is None:
            return

        image = self.current_image
        for facade in self.parsed_facades:
            src = (facade.source or "").lower()
            if "photo" not in src:
                continue
            if facade.scene_type != "flat_long_facade":
                continue
            if not facade.region_bbox:
                continue

            probe_facades = self.color_detector.detect_photo_facade_region(image)
            if probe_facades:
                probe = probe_facades[0]
                if (probe.color_detected or "").find("merged") >= 0:
                    px, py, pw, ph = probe.bbox
                    fx0, fy0, fw0, fh0 = facade.region_bbox
                    if py < fy0 - max(20, int(fh0 * 0.08)) and pw >= int(fw0 * 0.90):
                        continue

            openings = [
                e for e in self.parsed_elements
                if e.parent_facade == facade.name and e.bbox
            ]
            if len(openings) < 2:
                continue

            base_bbox = facade.region_bbox
            base_contour_override = None
            if (
                facade.scene_type in {"gable_facade", "sparse_openings_flat"}
                and facade.seed_region_bbox
            ):
                seed_x, seed_y, seed_w, seed_h = facade.seed_region_bbox
                if seed_w >= (facade.region_bbox[2] if facade.region_bbox else 0):
                    base_bbox = facade.seed_region_bbox
                if (
                    facade.scene_type == "gable_facade"
                    and seed_w >= int((facade.region_bbox[2] if facade.region_bbox else seed_w) * 1.35)
                ):
                    recovered = self._recover_gable_silhouette_from_seed(facade)
                    if recovered is not None:
                        recovered_contour, recovered_bbox, _ = recovered
                        if os.environ.get("DQP_DEBUG_GEOM"):
                            print("DEBUG_EXCLUDE_RECOVERED", facade.scene_type, "recovered_bbox", recovered_bbox)
                        base_bbox = recovered_bbox
                        base_contour_override = recovered_contour
            x, y, w, h = base_bbox
            if w < 120 or h < 120:
                continue

            if isinstance(facade.contour, np.ndarray) and facade.contour.size >= 6:
                c_local = facade.contour.astype(np.int32).copy()
                c_local[:, 0, 0] -= x
                c_local[:, 0, 1] -= y
                facade_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(facade_mask, [c_local], -1, 255, -1)
            else:
                facade_mask = np.ones((h, w), dtype=np.uint8) * 255

            top_profile = []
            for cx in range(w):
                ys = np.where(facade_mask[:, cx] > 0)[0]
                if ys.size > 0:
                    top_profile.append(int(ys[0]))
            if len(top_profile) < max(32, int(w * 0.20)):
                continue

            top_profile = np.array(top_profile, dtype=np.float32)
            top_iqr = float(np.percentile(top_profile, 75) - np.percentile(top_profile, 25))
            if top_iqr > max(24.0, h * 0.10):
                # Skip stepped facades; they need separate handling.
                continue

            opening_tops = np.array([max(0, e.bbox[1] - y) for e in openings], dtype=np.float32)
            ref_top = int(round(float(np.percentile(opening_tops, 35))))
            search_min = max(6, ref_top - int(h * 0.42))
            search_max = max(search_min + 8, ref_top - max(6, int(h * 0.02)))
            if search_max <= search_min + 4:
                continue

            roi = image[y:y + h, x:x + w]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            edges = cv2.Canny(gray, 60, 150)
            lines = cv2.HoughLinesP(
                edges,
                1,
                np.pi / 180.0,
                threshold=max(36, int(w * 0.05)),
                minLineLength=max(60, int(w * 0.45)),
                maxLineGap=max(18, int(w * 0.04)),
            )
            if lines is None:
                continue

            candidates = []
            for line in lines[:, 0, :]:
                x1, y1, x2, y2 = map(int, line)
                dx = x2 - x1
                dy = y2 - y1
                if abs(dx) < max(24, int(w * 0.12)):
                    continue
                slope = abs(dy / max(1.0, float(dx)))
                if slope > 0.12:
                    continue
                mean_y = (y1 + y2) * 0.5
                if mean_y < search_min or mean_y > search_max:
                    continue
                if min(y1, y2) >= ref_top - 4:
                    continue
                length = float((dx * dx + dy * dy) ** 0.5)
                candidates.append((mean_y, length, (x1, y1, x2, y2)))

            if not candidates:
                continue

            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            x1, y1, x2, y2 = candidates[0][2]
            if x2 == x1:
                continue

            cap_bias = max(
                int(getattr(config, "PHOTO_FACADE_CAP_LINE_BIAS_MIN_PX", 8)),
                int(round(h * float(getattr(config, "PHOTO_FACADE_CAP_LINE_BIAS_RATIO", 0.035))))
            )
            max_cap_y = int(ref_top - max(4, int(h * 0.01)))
            y1 = min(max_cap_y, y1 + cap_bias)
            y2 = min(max_cap_y, y2 + cap_bias)
            if min(y1, y2) <= search_min:
                continue

            a = (y2 - y1) / float(x2 - x1)
            b = y1 - a * x1

            capped_mask = facade_mask.copy()
            for cx in range(w):
                cy = int(round(a * cx + b))
                cy = max(0, min(h - 1, cy))
                capped_mask[:cy, cx] = 0

            contours, _ = cv2.findContours(capped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(contour))
            if area <= 0:
                continue
            area_ratio = area / max(1.0, float(facade.area_px or 1.0))
            if area_ratio < 0.42 or area_ratio > 0.92:
                continue

            contour = cv2.approxPolyDP(contour, max(1.0, cv2.arcLength(contour, True) * 0.0035), True)
            contour[:, 0, 0] += x
            contour[:, 0, 1] += y
            bx, by, bw, bh = cv2.boundingRect(contour)

            kept_centers = 0
            for elem in openings:
                ex, ey = elem.position
                if cv2.pointPolygonTest(contour.astype(np.float32), (float(ex), float(ey)), False) >= 0:
                    kept_centers += 1
            if kept_centers < max(1, len(openings) - 1):
                continue

            facade.region_bbox = (int(bx), int(by), int(bw), int(bh))
            facade.contour = contour
            facade.area_px = area
            if "capped" not in src:
                facade.source = f"{facade.source}+capped"

    def _assign_parents(self):
        """Assign each element to its parent facade based on spatial proximity."""
        for elem in self.parsed_elements:
            elem.parent_facade = None
            best_facade = None
            best_dist = float("inf")

            for facade in self.parsed_facades:
                if facade.region_bbox:
                    fx, fy, fw, fh = facade.region_bbox
                    ex, ey = elem.position
                    if fx <= ex <= fx + fw and fy <= ey <= fy + fh:
                        elem.parent_facade = facade.name
                        best_facade = facade
                        break

                dx = abs(elem.position[0] - facade.position[0])
                dy = abs(elem.position[1] - facade.position[1])
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_facade = facade

            if elem.parent_facade is None and best_facade:
                elem.parent_facade = best_facade.name

    def _filter_elements_by_facade_proximity(self):
        """Remove OCR artifacts that are too far from all facade regions."""
        facade_boxes = [f.region_bbox for f in self.parsed_facades if f.region_bbox]
        if not facade_boxes:
            return

        filtered = []
        for elem in self.parsed_elements:
            point = elem.position
            inside = False
            min_dist = float("inf")
            for bbox in facade_boxes:
                x, y, w, h = bbox
                if x <= point[0] <= x + w and y <= point[1] <= y + h:
                    inside = True
                    min_dist = 0.0
                    break
                d = self._outside_distance_to_bbox(point, bbox)
                if d < min_dist:
                    min_dist = d

            if inside or min_dist <= 80:
                filtered.append(elem)

        self.parsed_elements = filtered

    @staticmethod
    def _label_key(label: str) -> str:
        text = (label or "").strip().upper().replace(" ", "")
        match = re.match(r"^([FU])[_-]?(\d+)", text)
        if match:
            return f"{match.group(1)}{match.group(2)}"
        return text

    @staticmethod
    def _distance_to_nearest_ocr_element(point: tuple, elements: list) -> float:
        if not elements:
            return float("inf")
        px, py = point
        return min(((px - e.position[0]) ** 2 + (py - e.position[1]) ** 2) ** 0.5
                   for e in elements)

    @staticmethod
    def _contains_facade_text(region: DetectedRegion) -> bool:
        txt = f"{region.label} {(region.ocr_text or '')}".upper()
        return "FATADA" in txt

    @staticmethod
    def _has_door_signal(region: DetectedRegion) -> bool:
        txt = f"{region.label} {(region.ocr_text or '')}".upper()
        if "FATADA" in txt:
            return False
        if re.search(r"\bU(?:SA)?[_ -]*PHOTO[_ -]*\d+\b", txt):
            return True
        if re.search(r"\bU(?:SA)?[_ -]*\d+\b", txt):
            return True
        if re.search(r"\bU\s*[-_ ]?\d+\b", (region.ocr_text or "").upper()):
            return True
        if region.area_m2 is not None and 0.5 <= region.area_m2 <= 6.5:
            return True
        return False

    @staticmethod
    def _area_bounds_for_type(element_type: str) -> tuple:
        if element_type == "window":
            return (0.15, 15.0)
        if element_type == "door":
            return (0.50, 6.50)
        return (0.01, 5000.0)

    @staticmethod
    def _dimension_bounds_for_type(element_type: str) -> tuple:
        if element_type == "window":
            return (0.20, 8.00, 0.20, 5.00)
        if element_type == "door":
            return (0.50, 5.00, 1.40, 5.00)
        return (0.05, 50.0, 0.05, 50.0)

    def _propagate_known_areas(self):
        """Copy known template areas to identical element labels with missing area."""
        known = {}
        for elem in self.parsed_elements:
            if elem.area <= 0:
                continue
            key = (elem.element_type, self._label_key(elem.label))
            known.setdefault(key, []).append(elem.area)

        for elem in self.parsed_elements:
            if elem.area > 0:
                continue
            key = (elem.element_type, self._label_key(elem.label))
            vals = known.get(key, [])
            if vals:
                elem.area = float(np.median(vals))

    @staticmethod
    def _outside_distance_to_bbox(point: tuple, bbox: tuple) -> float:
        px, py = point
        x, y, w, h = bbox
        dx = max(x - px, 0, px - (x + w))
        dy = max(y - py, 0, py - (y + h))
        return (dx ** 2 + dy ** 2) ** 0.5

    @staticmethod
    def _region_bbox(region) -> Optional[tuple]:
        if region is None:
            return None
        bbox = getattr(region, "region_bbox", None)
        if bbox:
            return bbox
        return getattr(region, "bbox", None)

    @staticmethod
    def _region_kind(region) -> str:
        kind = getattr(region, "element_type", "")
        if kind:
            return str(kind)
        return str(getattr(region, "region_type", ""))

    @staticmethod
    def _bbox_center(bbox: tuple) -> tuple[float, float]:
        x, y, w, h = bbox
        return (float(x) + float(w) * 0.5, float(y) + float(h) * 0.5)

    def _region_area(self, region) -> float:
        bbox = self._region_bbox(region)
        if not bbox:
            return 0.0
        return float(getattr(region, "area_px", 0.0) or (bbox[2] * bbox[3]))

    def _regions_inside_facade(self, facade, regions: list) -> list:
        facade_bbox = self._region_bbox(facade)
        if not facade_bbox:
            return []
        fx, fy, fw, fh = facade_bbox
        tol_x = max(8, int(round(fw * 0.025)))
        tol_y = max(8, int(round(fh * 0.025)))
        local = []
        for region in regions:
            bbox = self._region_bbox(region)
            if not bbox:
                continue
            cx, cy = self._bbox_center(bbox)
            if (
                (fx - tol_x) <= cx <= (fx + fw + tol_x)
                and (fy - tol_y) <= cy <= (fy + fh + tol_y)
            ):
                local.append(region)
        return local

    def _has_flat_central_opening_cluster(self, facade, openings: list) -> bool:
        facade_bbox = self._region_bbox(facade)
        if not facade_bbox:
            return False
        fx, fy, fw, fh = facade_bbox
        if fw < 160 or fh < 90:
            return False

        aspect = float(fw) / max(1.0, float(fh))
        if aspect < float(getattr(config, "PHOTO_SCENE_FLAT_MIN_ASPECT", 2.0)):
            return False

        local_openings = self._regions_inside_facade(facade, openings)
        windows = [
            region for region in local_openings
            if self._region_kind(region) == "window" and self._region_bbox(region)
        ]
        doors = [
            region for region in local_openings
            if self._region_kind(region) == "door" and self._region_bbox(region)
        ]
        if len(windows) < 2 or not doors:
            return False

        facade_center_x = fx + fw * 0.5
        best_door = None
        best_score = None
        for door in doors:
            bbox = self._region_bbox(door)
            if not bbox:
                continue
            dx, dy, dw, dh = bbox
            door_center_x = dx + dw * 0.5
            center_delta = abs(door_center_x - facade_center_x)
            top_ratio = (dy - fy) / max(1.0, float(fh))
            bottom_ratio = (dy + dh - fy) / max(1.0, float(fh))
            if center_delta > fw * 0.17:
                continue
            if not (fw * 0.020 <= dw <= fw * 0.18):
                continue
            if not (fh * 0.18 <= dh <= fh * 0.82):
                continue
            if not (0.18 <= top_ratio <= 0.78):
                continue
            if not (0.55 <= bottom_ratio <= 0.98):
                continue
            score = center_delta - dh * 0.03
            if best_score is None or score < best_score:
                best_score = score
                best_door = bbox

        if best_door is None:
            return False

        dx, dy, dw, dh = best_door
        door_cy = dy + dh * 0.5
        left_hits = []
        right_hits = []
        max_gap = max(44, int(round(fw * 0.14)))
        max_vertical_delta = max(42, int(round(fh * 0.18)))

        for window in windows:
            bbox = self._region_bbox(window)
            if not bbox:
                continue
            wx, wy, ww, wh = bbox
            win_cx = wx + ww * 0.5
            win_cy = wy + wh * 0.5
            gap_x = max(0, max(wx - (dx + dw), dx - (wx + ww)))
            overlap_y = max(0, min(wy + wh, dy + dh) - max(wy, dy))
            vertical_delta = abs(win_cy - door_cy)
            if ww > fw * 0.24 or wh > fh * 0.52:
                continue
            if gap_x > max_gap:
                continue
            if overlap_y < min(wh, dh) * 0.12 and vertical_delta > max_vertical_delta:
                continue
            if not (fy + fh * 0.16 <= win_cy <= fy + fh * 0.78):
                continue
            if wx + ww <= dx and win_cx < facade_center_x:
                left_hits.append((gap_x, vertical_delta, bbox))
            elif wx >= dx + dw and win_cx > facade_center_x:
                right_hits.append((gap_x, vertical_delta, bbox))

        if not left_hits or not right_hits:
            return False

        left_hits.sort(key=lambda item: (item[0], item[1]))
        right_hits.sort(key=lambda item: (item[0], item[1]))
        _, _, left_box = left_hits[0]
        _, _, right_box = right_hits[0]
        left_gap = max(0, dx - (left_box[0] + left_box[2]))
        right_gap = max(0, right_box[0] - (dx + dw))
        if abs(left_gap - right_gap) > max(36, int(round(fw * 0.08))):
            return False

        left_top = left_box[1]
        right_top = right_box[1]
        top_delta = abs(left_top - right_top)
        if top_delta > max(36, int(round(fh * 0.15))):
            return False

        return True

    def _should_keep_photo_openings_for_flat_scene(self,
                                                   photo_facades: list,
                                                   photo_windows: list,
                                                   photo_doors: list,
                                                   ml_windows: list,
                                                   ml_doors: list) -> bool:
        if not photo_facades or not photo_windows or not photo_doors:
            return False
        if ml_doors:
            return False
        if not ml_windows:
            return False

        candidate_facades = [
            facade for facade in photo_facades
            if "merged" not in ((getattr(facade, "color_detected", "") or "").lower())
        ]
        if not candidate_facades:
            return False
        primary_facade = max(candidate_facades, key=self._region_area)

        photo_openings = self._regions_inside_facade(
            primary_facade,
            list(photo_windows) + list(photo_doors),
        )
        if not self._has_flat_central_opening_cluster(primary_facade, photo_openings):
            return False

        local_photo_doors = [
            region for region in photo_openings if self._region_kind(region) == "door"
        ]
        if not local_photo_doors:
            return False

        local_ml_windows = self._regions_inside_facade(primary_facade, ml_windows)
        return bool(local_ml_windows or ml_windows)

    def _calibrate_linear_scale_from_ocr(self):
        """Estimate global px/m scale from OCR linear dimensions near facades."""
        facade_boxes = [f.region_bbox for f in self.parsed_facades if f.region_bbox]
        if not facade_boxes:
            return

        candidates = []
        for r in self.ocr_results:
            if r.parsed_value is None or r.unit != "m":
                continue
            value_m = float(r.parsed_value)
            if value_m < 0.4 or value_m > 30:
                continue

            cx = (r.bbox[0] + r.bbox[2]) // 2
            cy = (r.bbox[1] + r.bbox[3]) // 2
            point = (cx, cy)

            best_bbox = None
            best_dist = float("inf")
            for bbox in facade_boxes:
                d = self._outside_distance_to_bbox(point, bbox)
                if d < best_dist:
                    best_dist = d
                    best_bbox = bbox

            if best_bbox is None or best_dist > 140:
                continue

            x, y, w, h = best_bbox
            if w <= 0 or h <= 0:
                continue

            top_d = abs(cy - y)
            bottom_d = abs(cy - (y + h))
            left_d = abs(cx - x)
            right_d = abs(cx - (x + w))
            near_h = min(top_d, bottom_d) <= max(30, int(h * 0.12))
            near_v = min(left_d, right_d) <= max(30, int(w * 0.12))

            if near_h:
                candidates.append(w / value_m)
            if near_v:
                candidates.append(h / value_m)

            if not near_h and not near_v and best_dist <= 30:
                if min(top_d, bottom_d) <= min(left_d, right_d):
                    candidates.append(w / value_m)
                else:
                    candidates.append(h / value_m)

        candidates = [c for c in candidates if 5 <= c <= 3000]
        if len(candidates) < 2:
            return

        med = float(np.median(candidates))
        filtered = [c for c in candidates if abs(c - med) <= med * 0.35]
        # Require at least two consistent linear cues; otherwise fallback path is safer.
        if len(filtered) < 2:
            return
        calibrated = float(np.median(filtered))

        if self.linear_scale_px_per_m and self.linear_scale_px_per_m > 0:
            # Blend manual/reference scale with OCR-derived scale.
            self.linear_scale_px_per_m = (
                self.linear_scale_px_per_m * 0.7 + calibrated * 0.3
            )
            if self.linear_scale_source == "manual_ref":
                self.linear_scale_source = "manual_ref+ocr"
            elif not self.linear_scale_source:
                self.linear_scale_source = "ocr_blended"
        else:
            self.linear_scale_px_per_m = calibrated
            self.linear_scale_source = "ocr"

    def _apply_default_scale_assumption(self):
        """Fallback scale when input likely comes from white-background exported facade."""
        if self.linear_scale_px_per_m and self.linear_scale_px_per_m > 0:
            return
        if self.current_image is None:
            return
        if not getattr(config, "AUTO_ASSUME_SCALE_ENABLED", False):
            return

        scale_ratio = self._extract_scale_ratio_from_ocr()
        if scale_ratio and scale_ratio > 0:
            self.scale_ratio_used = float(scale_ratio)
            self.scale_ratio_source = "ocr"
        else:
            scale_ratio = float(getattr(config, "SCALE", 100.0))
            self.scale_ratio_used = float(scale_ratio)
            self.scale_ratio_source = "config"

        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        border = max(8, min(h, w) // 35)
        border_pixels = np.concatenate([
            gray[:border, :].ravel(),
            gray[-border:, :].ravel(),
            gray[:, :border].ravel(),
            gray[:, -border:].ravel(),
        ])
        white_level = float(np.percentile(border_pixels, 70))
        white_min = float(getattr(config, "AUTO_ASSUME_SCALE_WHITE_BORDER_MIN", 240.0))
        looks_white_export = white_level >= white_min
        has_detected_facade_geometry = any(
            bool(f.region_bbox) for f in self.parsed_facades
        )
        allow_geom_fallback = bool(
            getattr(config, "AUTO_ASSUME_SCALE_GEOMETRY_FALLBACK", True)
        )
        used_geom_fallback = False
        if not looks_white_export:
            if self.scale_ratio_source == "ocr":
                # If OCR sees a direct 1:xx scale marker, allow fallback anyway.
                pass
            elif allow_geom_fallback and has_detected_facade_geometry:
                used_geom_fallback = True
            else:
                return
        if not looks_white_export and not (self.scale_ratio_source == "ocr" or used_geom_fallback):
            return

        cfg_dpi = float(getattr(config, "PDF_DPI", 254.0))
        max_dev = float(getattr(config, "AUTO_ASSUME_SCALE_DPI_DEVIATION_MAX_RATIO", 0.15))
        dpi = cfg_dpi
        dpi_src = "config"
        if self.source_dpi and 30 <= self.source_dpi <= 2400 and cfg_dpi > 0:
            rel_dev = abs(float(self.source_dpi) - cfg_dpi) / cfg_dpi
            if rel_dev <= max_dev:
                dpi = float(self.source_dpi)
                dpi_src = "metadate imagine"
            else:
                self.warnings.append(
                    "DPI din metadate a fost ignorat (deviatie mare fata de "
                    f"DPI configurat: {self.source_dpi:.1f} vs {cfg_dpi:.1f})."
                )
        if dpi <= 0 or scale_ratio <= 0:
            return

        assumed_px_per_m = dpi * 39.37007874 / scale_ratio
        if assumed_px_per_m <= 0:
            return
        self.linear_scale_px_per_m = float(assumed_px_per_m)
        self.linear_scale_source = (
            "auto_assumed_geom" if used_geom_fallback else "auto_assumed"
        )

        ratio_msg = (
            f"1:{int(round(scale_ratio))} (din OCR)"
            if self.scale_ratio_source == "ocr"
            else f"1:{int(round(scale_ratio))} (din setari)"
        )
        confidence_msg = (
            "Estimare aproximativa (fara fond alb validat); "
            if used_geom_fallback else ""
        )
        self.warnings.append(
            confidence_msg
            + "Scara liniara a fost estimata automat "
            f"din {ratio_msg} + DPI ({dpi_src}). "
            "Pentru precizie maxima puteti introduce Ref. (m) + Ref. (px)."
        )

    def _extract_scale_ratio_from_ocr(self) -> Optional[float]:
        """Try to detect drawing scale ratio from OCR text blocks (e.g. 1:100)."""
        if not self.ocr_results:
            return None

        weighted = {}
        preferred = {20, 25, 50, 75, 100, 125, 200, 250}

        for r in self.ocr_results:
            text = (r.text or "").upper()
            if not text:
                continue

            # OCR often confuses I/l with 1 in small texts.
            text = text.replace("I", "1").replace("L", "1")
            has_scale_kw = re.search(r"SCA?R|SC\.", text) is not None
            base_score = float(max(0.05, min(1.0, r.confidence)))

            # Common form: "1:100", "1/100", optionally with "SCARA".
            for m in re.finditer(r"(?:SCARA|SC\.?|SCA)?\s*1\s*[:/\\|]\s*(\d{2,4})", text):
                den = int(m.group(1))
                if den < 10 or den > 1000:
                    continue
                score = base_score
                if has_scale_kw:
                    score += 0.50
                if den in preferred:
                    score += 0.15
                weighted[den] = weighted.get(den, 0.0) + score

            # Less common OCR order: "100:1" (convert to same ratio denominator).
            for m in re.finditer(r"(?:SCARA|SC\.?|SCA)?\s*(\d{2,4})\s*[:/\\|]\s*1", text):
                den = int(m.group(1))
                if den < 10 or den > 1000:
                    continue
                score = base_score * 0.8
                if has_scale_kw:
                    score += 0.40
                if den in preferred:
                    score += 0.15
                weighted[den] = weighted.get(den, 0.0) + score

        if not weighted:
            return None
        best_ratio = max(weighted.items(), key=lambda kv: kv[1])[0]
        return float(best_ratio)

    def _estimate_element_dimensions_from_scale(self):
        """Estimate element width/height in meters from bbox and px/m scale."""
        if not self.linear_scale_px_per_m or self.linear_scale_px_per_m <= 0:
            return

        for elem in self.parsed_elements:
            if not elem.bbox:
                continue
            _, _, bw, bh = elem.bbox
            if bw <= 0 or bh <= 0:
                continue

            est_w = float(bw / self.linear_scale_px_per_m)
            est_h = float(bh / self.linear_scale_px_per_m)
            if est_w <= 0 or est_h <= 0:
                continue
            min_w, max_w, min_h, max_h = self._dimension_bounds_for_type(
                elem.element_type
            )
            if not (min_w <= est_w <= max_w and min_h <= est_h <= max_h):
                continue

            if elem.width <= 0:
                elem.width = round(est_w, 3)
            if elem.height <= 0:
                elem.height = round(est_h, 3)

            if elem.area <= 0:
                est_area = est_w * est_h
                min_a, max_a = self._area_bounds_for_type(elem.element_type)
                if min_a <= est_area <= max_a:
                    elem.area = round(float(est_area), 3)

    def _infer_missing_areas_from_scale(self):
        """Infer missing element areas using calibrated px/m² from known elements."""
        ratios = []
        for elem in self.parsed_elements:
            if "ocr" not in elem.source:
                continue
            if elem.area <= 0 or elem.area_px <= 0:
                continue
            ratio = elem.area_px / elem.area
            if 10 <= ratio <= 500000:
                ratios.append(ratio)

        area_from_elements = None
        if len(ratios) >= 2:
            area_from_elements = float(np.median(ratios))

        area_from_linear = None
        if self.linear_scale_px_per_m and self.linear_scale_px_per_m > 0:
            area_from_linear = float(self.linear_scale_px_per_m ** 2)

        if area_from_elements and area_from_linear:
            low = area_from_elements * 0.25
            high = area_from_elements * 4.0
            if low <= area_from_linear <= high:
                self.area_scale_px_per_m2 = (
                    area_from_elements * 0.7 + area_from_linear * 0.3
                )
            else:
                self.area_scale_px_per_m2 = area_from_elements
        elif area_from_elements:
            self.area_scale_px_per_m2 = area_from_elements
        elif area_from_linear:
            self.area_scale_px_per_m2 = area_from_linear
        else:
            return

        if self.area_scale_px_per_m2 <= 0:
            return

        for elem in self.parsed_elements:
            if elem.area > 0 or elem.area_px <= 0:
                continue
            if elem.element_type == "door" and "ocr" not in elem.source:
                continue
            est = elem.area_px / self.area_scale_px_per_m2
            min_a, max_a = self._area_bounds_for_type(elem.element_type)
            if min_a <= est <= max_a:
                elem.area = round(float(est), 3)

    def _infer_facade_areas_from_scale(self):
        """Fill facade total area from detected facade pixels and scale."""
        if not self.area_scale_px_per_m2 or self.area_scale_px_per_m2 <= 0:
            return

        for facade in self.parsed_facades:
            if facade.total_area > 0:
                continue
            if facade.area_px <= 0:
                continue
            est = facade.area_px / self.area_scale_px_per_m2
            if 2.0 <= est <= 5000:
                facade.total_area = round(float(est), 3)

    @staticmethod
    def _facade_bottom_profile(facade_mask: np.ndarray) -> np.ndarray:
        """Return bottom-most facade pixel for each column; -1 where unsupported."""
        roi_h, roi_w = facade_mask.shape[:2]
        bottoms = np.full(roi_w, -1, dtype=np.int32)
        for cx in range(roi_w):
            ys = np.where(facade_mask[:, cx] > 0)[0]
            if ys.size > 0:
                bottoms[cx] = int(ys[-1])
        return bottoms

    @staticmethod
    def _build_socle_band_from_bottoms(facade_mask: np.ndarray,
                                       tops: np.ndarray) -> np.ndarray:
        """Fill a socle mask between a top profile and the facade bottom profile."""
        roi_h, roi_w = facade_mask.shape[:2]
        socle_mask = np.zeros_like(facade_mask)
        bottoms = AnalysisPipeline._facade_bottom_profile(facade_mask)
        for cx in range(roi_w):
            bottom = int(bottoms[cx])
            if bottom < 0:
                continue
            top = int(round(float(tops[cx])))
            top = max(0, min(bottom, top))
            socle_mask[top:bottom + 1, cx] = facade_mask[top:bottom + 1, cx]
        return socle_mask


    @staticmethod
    def _mask_top_profile(mask: np.ndarray) -> np.ndarray:
        """Return top-most nonzero pixel for each column; -1 where unsupported."""
        roi_h, roi_w = mask.shape[:2]
        tops = np.full(roi_w, -1, dtype=np.int32)
        for cx in range(roi_w):
            ys = np.where(mask[:, cx] > 0)[0]
            if ys.size > 0:
                tops[cx] = int(ys[0])
        return tops

    @staticmethod
    def _interpolate_profile(profile: np.ndarray) -> np.ndarray:
        """Linearly interpolate a sparse per-column profile across supported span."""
        if profile is None or profile.size == 0:
            return profile
        out = profile.astype(np.float32).copy()
        valid = np.where(profile >= 0)[0]
        if valid.size == 0:
            return out
        if valid.size == 1:
            out[:] = float(profile[int(valid[0])])
            return out
        left = int(valid[0])
        right = int(valid[-1])
        span = np.arange(left, right + 1, dtype=np.float32)
        out[left:right + 1] = np.interp(span, valid.astype(np.float32), profile[valid].astype(np.float32))
        out[:left] = float(profile[left])
        out[right + 1:] = float(profile[right])
        return out

    @staticmethod
    def _clip_band_mask_to_support_columns(band_mask: np.ndarray,
                                           support_mask: np.ndarray) -> np.ndarray:
        """Keep a band-like mask only where the support mask has facade columns."""
        if band_mask is None or support_mask is None or band_mask.size == 0 or support_mask.size == 0:
            return band_mask
        out = band_mask.copy()
        support_cols = np.sum(support_mask > 0, axis=0) > 0
        if support_cols.size != out.shape[1]:
            return out
        out[:, ~support_cols] = 0
        return out

    @staticmethod
    def _cut_mask_below_profile(mask: np.ndarray,
                                top_profile: np.ndarray,
                                pad_px: int = 0) -> np.ndarray:
        """Zero everything below a top profile, column by column."""
        if mask is None or mask.size == 0 or top_profile is None or top_profile.size == 0:
            return mask
        roi_h, roi_w = mask.shape[:2]
        out = mask.copy()
        for cx in range(min(roi_w, int(top_profile.size))):
            top = float(top_profile[cx])
            if top < 0:
                continue
            cut_y = max(0, min(roi_h, int(round(top)) + int(pad_px)))
            out[cut_y:, cx] = 0
        return out

    @staticmethod
    def _clip_mask_to_x_span(mask: np.ndarray,
                             left: Optional[int],
                             right: Optional[int]) -> np.ndarray:
        """Zero mask columns outside an explicit horizontal span."""
        if mask is None or mask.size == 0:
            return mask
        if left is None or right is None:
            return mask
        roi_h, roi_w = mask.shape[:2]
        left_i = max(0, min(roi_w - 1, int(left)))
        right_i = max(0, min(roi_w - 1, int(right)))
        if right_i <= left_i:
            return mask
        out = mask.copy()
        out[:, :left_i] = 0
        out[:, right_i + 1:] = 0
        return out

    @staticmethod
    def _profile_from_line_contour(line_contour: np.ndarray,
                                   width: int) -> np.ndarray:
        """Expand an open line/polyline contour into a per-column y profile."""
        if (
            line_contour is None
            or not isinstance(line_contour, np.ndarray)
            or line_contour.size < 4
            or width <= 0
        ):
            return np.array([], dtype=np.float32)
        pts = line_contour.reshape(-1, 2).astype(np.float32)
        if pts.shape[0] < 2:
            return np.array([], dtype=np.float32)

        samples_x = []
        samples_y = []
        for (x_a, y_a), (x_b, y_b) in zip(pts[:-1], pts[1:]):
            x_a_i = int(round(float(x_a)))
            x_b_i = int(round(float(x_b)))
            y_a_f = float(y_a)
            y_b_f = float(y_b)
            if x_a_i == x_b_i:
                samples_x.append(float(x_a_i))
                samples_y.append(float(min(y_a_f, y_b_f)))
                continue
            left_x = min(x_a_i, x_b_i)
            right_x = max(x_a_i, x_b_i)
            xs_seg = np.arange(left_x, right_x + 1, dtype=np.float32)
            interp = np.interp(
                xs_seg,
                np.array([float(x_a_i), float(x_b_i)], dtype=np.float32),
                np.array([y_a_f, y_b_f], dtype=np.float32),
            )
            samples_x.extend(xs_seg.tolist())
            samples_y.extend(interp.astype(np.float32).tolist())

        if not samples_x:
            return np.array([], dtype=np.float32)

        sample_x = np.round(np.array(samples_x, dtype=np.float32)).astype(np.int32)
        sample_y = np.array(samples_y, dtype=np.float32)
        uniq_x = np.unique(sample_x)
        if uniq_x.size < 2:
            return np.array([], dtype=np.float32)

        x_nodes = []
        y_nodes = []
        for x_i in uniq_x.tolist():
            same = sample_y[sample_x == int(x_i)]
            x_nodes.append(float(x_i))
            y_nodes.append(float(np.median(same)))

        x_nodes = np.array(x_nodes, dtype=np.float32)
        y_nodes = np.array(y_nodes, dtype=np.float32)
        left = max(0, min(width - 1, int(round(float(x_nodes[0])))))
        right = max(0, min(width - 1, int(round(float(x_nodes[-1])))))
        profile = np.full(width, -1.0, dtype=np.float32)
        if right <= left:
            profile[left] = float(np.median(y_nodes))
            return profile
        span = np.arange(left, right + 1, dtype=np.float32)
        interp = np.interp(span, x_nodes, y_nodes)
        profile[left:right + 1] = interp.astype(np.float32)
        return profile

    @staticmethod
    def _build_mask_from_profiles(top_profile: np.ndarray,
                                  bottom_profile: np.ndarray,
                                  support_mask: np.ndarray) -> np.ndarray:
        """Fill a mask between explicit top and bottom profiles across support columns."""
        if (
            top_profile is None
            or bottom_profile is None
            or support_mask is None
            or top_profile.size == 0
            or bottom_profile.size == 0
            or support_mask.size == 0
        ):
            return np.zeros_like(support_mask)

        roi_h, roi_w = support_mask.shape[:2]
        out = np.zeros_like(support_mask)
        support_bottoms = AnalysisPipeline._facade_bottom_profile(support_mask)
        support_tops = AnalysisPipeline._mask_top_profile(support_mask)
        for cx in range(min(roi_w, int(top_profile.size), int(bottom_profile.size))):
            if support_bottoms[cx] < 0 or support_tops[cx] < 0:
                continue
            top = int(round(float(top_profile[cx])))
            bottom = int(round(float(bottom_profile[cx])))
            support_top = int(support_tops[cx])
            support_bottom = int(support_bottoms[cx])
            top = max(0, max(support_top, min(top, support_bottom)))
            bottom = max(top, min(support_bottom, bottom))
            out[top:bottom + 1, cx] = 255
        return out

    @staticmethod
    def _polyline_contour_from_profile(profile: np.ndarray,
                                       left: Optional[int] = None,
                                       right: Optional[int] = None) -> np.ndarray:
        """Compress a per-column profile into an open polyline that preserves step breaks."""
        if profile is None or profile.size == 0:
            return np.array([])
        valid = np.where(profile >= 0)[0]
        if valid.size < 2:
            return np.array([])
        left_i = int(valid[0] if left is None else max(valid[0], int(left)))
        right_i = int(valid[-1] if right is None else min(valid[-1], int(right)))
        if right_i <= left_i:
            return np.array([])

        points = []
        prev_x = None
        prev_y = None
        for cx in range(left_i, right_i + 1):
            y = float(profile[cx])
            if y < 0:
                continue
            y_i = int(round(y))
            if prev_y is None:
                points.append([int(cx), y_i])
                prev_x = int(cx)
                prev_y = y_i
                continue
            if y_i == prev_y:
                prev_x = int(cx)
                continue
            if prev_x is not None and points[-1][0] != int(prev_x):
                points.append([int(prev_x), int(prev_y)])
            points.append([int(cx), y_i])
            prev_x = int(cx)
            prev_y = y_i
        end_y = int(round(float(profile[right_i])))
        if prev_x is not None and points[-1][0] != int(prev_x):
            points.append([int(prev_x), int(prev_y)])
        if not points or points[-1][0] != right_i or points[-1][1] != end_y:
            points.append([right_i, end_y])
        contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        if contour.shape[0] < 2:
            return np.array([])
        return contour

    @staticmethod
    def _closed_contour_from_profiles(top_profile: np.ndarray,
                                      bottom_profile: np.ndarray,
                                      support_mask: np.ndarray) -> Optional[np.ndarray]:
        """Build a closed contour directly from explicit top/bottom profiles."""
        if (
            top_profile is None
            or bottom_profile is None
            or support_mask is None
            or top_profile.size == 0
            or bottom_profile.size == 0
            or support_mask.size == 0
        ):
            return None

        valid_cols = np.where(np.sum(support_mask > 0, axis=0) > 0)[0]
        if valid_cols.size < 2:
            return None
        left = int(valid_cols[0])
        right = int(valid_cols[-1])
        top_pts = []
        bottom_pts = []
        for cx in range(left, right + 1):
            if cx >= int(top_profile.size) or cx >= int(bottom_profile.size):
                break
            top_y = float(top_profile[cx])
            bottom_y = float(bottom_profile[cx])
            if top_y < 0 or bottom_y < 0:
                continue
            top_i = int(round(top_y))
            bottom_i = int(round(max(top_y, bottom_y)))
            top_pts.append([int(cx), top_i])
            bottom_pts.append([int(cx), bottom_i])

        if len(top_pts) < 2 or len(bottom_pts) < 2:
            return None
        contour = np.array(top_pts + bottom_pts[::-1], dtype=np.int32).reshape((-1, 1, 2))
        if contour.shape[0] < 6:
            return None
        perim = float(cv2.arcLength(contour, True))
        eps = max(1.0, perim * 0.0025)
        approx = cv2.approxPolyDP(contour, eps, True)
        return approx if approx is not None and approx.size >= 6 else contour

    @staticmethod
    def _line_span(line_contour: np.ndarray) -> tuple[Optional[int], Optional[int]]:
        """Return min/max x covered by an open line contour."""
        if (
            line_contour is None
            or not isinstance(line_contour, np.ndarray)
            or line_contour.size < 4
        ):
            return (None, None)
        pts = line_contour.reshape(-1, 2).astype(np.int32)
        return (int(np.min(pts[:, 0])), int(np.max(pts[:, 0])))

    @staticmethod
    def _derive_socle_profile_line_local(socle_contour: np.ndarray,
                                         facade_contour: Optional[np.ndarray],
                                         roi_h: int) -> np.ndarray:
        """Derive the visual socle drip-profile line from local contours."""
        if (
            socle_contour is None
            or not isinstance(socle_contour, np.ndarray)
            or socle_contour.size < 6
        ):
            return np.array([])

        sx, sy, sw, _ = cv2.boundingRect(socle_contour)
        x_a, y_a = sx, sy
        x_b, y_b = sx + sw, sy

        socle_pts = socle_contour.reshape(-1, 2).astype(np.float32)
        y_thr = np.percentile(socle_pts[:, 1], 35)
        top_pts = socle_pts[socle_pts[:, 1] <= y_thr]
        if top_pts.shape[0] >= 6 and len(np.unique(top_pts[:, 0])) >= 2:
            a, b = np.polyfit(top_pts[:, 0], top_pts[:, 1], 1)
            x_a = sx
            x_b = sx + sw
            y_a = int(round(a * x_a + b))
            y_b = int(round(a * x_b + b))

        if (
            abs(int(y_b) - int(y_a)) <= 1
            and isinstance(facade_contour, np.ndarray)
            and facade_contour.size >= 6
        ):
            f_pts = facade_contour.reshape(-1, 2).astype(np.float32)
            fy_thr = np.percentile(f_pts[:, 1], 78)
            bot_pts = f_pts[f_pts[:, 1] >= fy_thr]
            if bot_pts.shape[0] >= 8 and len(np.unique(bot_pts[:, 0])) >= 2:
                fa, fb = np.polyfit(bot_pts[:, 0], bot_pts[:, 1], 1)
                y_a = int(round(fa * x_a + fb))
                y_b = int(round(fa * x_b + fb))

        max_dy = max(4, int(roi_h * 0.08))
        dy = int(y_b - y_a)
        if abs(dy) > max_dy:
            mid = int(round((y_a + y_b) * 0.5))
            half = int(round(max_dy * 0.5))
            y_a = mid - half
            y_b = mid + half

        return np.array(
            [[[int(x_a), int(y_a)]], [[int(x_b), int(y_b)]]],
            dtype=np.int32,
        )

    @staticmethod
    def _rolling_mean(signal: np.ndarray, window: int) -> np.ndarray:
        if signal.size == 0 or window <= 1:
            return signal.astype(np.float32)
        kernel = np.ones(int(window), dtype=np.float32) / float(window)
        return np.convolve(signal.astype(np.float32), kernel, mode="same")

    def _trim_flat_photo_facade_body_support(self,
                                             facade_mask: np.ndarray,
                                             facade: ParsedFacade,
                                             roi_gray: np.ndarray,
                                             allow_side_trim: bool = True) -> np.ndarray:
        """Suppress edge noise on flat photo facades using stable wall-body support."""
        if facade_mask is None or facade_mask.size == 0 or roi_gray is None or roi_gray.size == 0:
            return facade_mask

        roi_h, roi_w = facade_mask.shape[:2]
        if roi_h < 60 or roi_w < 80 or not facade.region_bbox:
            return facade_mask

        work = facade_mask.copy()
        _, fy, _, _ = facade.region_bbox
        openings = [
            e for e in self.parsed_elements
            if e.parent_facade == facade.name and e.bbox
        ]

        local_tops = []
        local_bottoms = []
        for elem in openings:
            _, ey, _, eh = elem.bbox
            local_tops.append(max(0, ey - fy))
            local_bottoms.append(min(roi_h - 1, ey - fy + eh))

        if local_tops:
            body_y0 = max(0, int(min(local_tops) - max(10, roi_h * 0.10)))
            body_y1 = min(roi_h, int(max(local_bottoms) + max(12, roi_h * 0.10)))
        else:
            body_y0 = int(roi_h * 0.12)
            body_y1 = int(roi_h * 0.76)
        if body_y1 <= body_y0 + 20:
            return work

        body_support = np.zeros_like(work)
        body_support[body_y0:body_y1, :] = work[body_y0:body_y1, :]
        col_occ = np.sum(body_support > 0, axis=0).astype(np.float32)
        nz_cols = col_occ[col_occ > 0]
        valid_cols = np.where(col_occ > 0)[0]
        if len(nz_cols) >= 8:
            occ_thr = max(6.0, float(np.percentile(nz_cols, 60)) * 0.72)
            col_mean = self._rolling_mean(col_occ, 9)
            valid_cols = np.where((col_occ >= occ_thr) & (col_mean >= occ_thr * 0.92))[0]
            if len(valid_cols) < max(18, int(roi_w * 0.20)):
                body_support = self.color_detector._trim_side_spikes(body_support)
                valid_cols = np.where(np.sum(body_support > 0, axis=0) > 0)[0]
        max_trim_ratio = float(getattr(config, "PHOTO_FACADE_BODY_EDGE_TRIM_MAX_RATIO", 0.08))
        if allow_side_trim and len(valid_cols) >= max(24, int(roi_w * 0.40)):
            left = int(valid_cols[0])
            right = int(valid_cols[-1])
            if 4 <= left <= int(roi_w * max_trim_ratio):
                work[:, :left] = 0
            trim_right = roi_w - 1 - right
            if 4 <= trim_right <= int(roi_w * max_trim_ratio):
                work[:, right + 1:] = 0

        row_occ = np.sum(work > 0, axis=1).astype(np.float32)
        nz_rows = row_occ[row_occ > 0]
        if len(nz_rows) >= 8:
            max_occ = float(np.percentile(nz_rows, 85))
            row_mean = self._rolling_mean(row_occ, 7)
            stable_thr = max(
                6.0,
                max_occ * float(getattr(config, "PHOTO_FACADE_BODY_ROW_OCC_RATIO", 0.58)),
            )
            mean_thr = max_occ * float(getattr(config, "PHOTO_FACADE_BODY_ROW_MEAN_RATIO", 0.62))
            top_scan_end = min(max(body_y0 + 12, int(roi_h * 0.42)), roi_h - 1)
            stable_top = None
            for r in range(0, max(1, top_scan_end)):
                if row_occ[r] >= stable_thr and row_mean[r] >= mean_thr:
                    stable_top = r
                    break
            if stable_top is not None and 2 <= stable_top <= int(roi_h * 0.28):
                relax_px = int(getattr(config, "PHOTO_FACADE_BODY_TOP_RELAX_PX", 4))
                trim_top = max(0, stable_top - relax_px)
                work[:trim_top, :] = 0

        return work


    @staticmethod
    def _fit_mask_edge_line(mask: np.ndarray,
                            edge: str = "top",
                            support_mask: Optional[np.ndarray] = None,
                            prefer_thick_columns: bool = False,
                            min_cols_ratio: float = 0.18,
                            max_slope: float = 0.08):
        """Fit a stable linear top/bottom edge for flat facade-style masks."""
        if mask is None or mask.size == 0:
            return None

        roi_h, roi_w = mask.shape[:2]
        span_mask = support_mask if support_mask is not None else mask
        span_cols = np.where(np.sum(span_mask > 0, axis=0) > 0)[0]
        if len(span_cols) < max(12, int(roi_w * min_cols_ratio)):
            return None

        xs = []
        ys_fit = []
        heights = []
        support_bottoms = (
            AnalysisPipeline._facade_bottom_profile(span_mask)
            if support_mask is not None
            else None
        )
        for cx in span_cols.tolist():
            ys = np.where(mask[:, cx] > 0)[0]
            if ys.size == 0:
                continue
            xs.append(float(cx))
            if edge == "bottom":
                ys_fit.append(float(ys[-1]))
            else:
                ys_fit.append(float(ys[0]))
            if support_bottoms is not None and support_bottoms[cx] >= 0:
                heights.append(float(support_bottoms[cx] - ys[0] + 1))
            else:
                heights.append(float(ys[-1] - ys[0] + 1))

        if len(xs) < max(12, int(roi_w * min_cols_ratio)):
            return None

        xs = np.array(xs, dtype=np.float32)
        ys_fit = np.array(ys_fit, dtype=np.float32)
        heights = np.array(heights, dtype=np.float32)
        if prefer_thick_columns and heights.size >= 8:
            thick_thr = max(3.0, float(np.percentile(heights, 65)) * 0.75)
            strong = heights >= thick_thr
            if int(np.count_nonzero(strong)) >= max(8, int(xs.size * 0.28)):
                xs = xs[strong]
                ys_fit = ys_fit[strong]

        if len(np.unique(xs.astype(np.int32))) < 6:
            return None

        a, b = np.polyfit(xs, ys_fit, 1)
        a = float(np.clip(a, -max_slope, max_slope))
        return int(span_cols[0]), int(span_cols[-1]), a, float(b)

    @staticmethod
    def _linear_band_contour_from_lines(left: int,
                                        right: int,
                                        top_line: tuple,
                                        bottom_line: tuple,
                                        roi_h: int) -> Optional[np.ndarray]:
        """Build a closed band contour from two linear edges."""
        if top_line is None or bottom_line is None:
            return None
        if right - left + 1 < 12:
            return None

        xs = np.arange(int(left), int(right) + 1, dtype=np.int32)
        top_pts = []
        bottom_pts = []
        a_top, b_top = top_line
        a_bottom, b_bottom = bottom_line
        for cx in xs.tolist():
            top_y = int(round(a_top * cx + b_top))
            bottom_y = int(round(a_bottom * cx + b_bottom))
            top_y = max(0, min(roi_h - 1, top_y))
            bottom_y = max(top_y, min(roi_h - 1, bottom_y))
            top_pts.append([cx, top_y])
            bottom_pts.append([cx, bottom_y])

        contour = np.array(top_pts + bottom_pts[::-1], dtype=np.int32).reshape((-1, 1, 2))
        if contour.shape[0] < 6:
            return None
        perim = float(cv2.arcLength(contour, True))
        eps = max(1.0, perim * 0.002)
        return cv2.approxPolyDP(contour, eps, True)

    @staticmethod
    def _bridge_socle_edge_gaps(socle_mask: np.ndarray,
                                facade_mask: np.ndarray,
                                max_gap_ratio: float = 0.06) -> np.ndarray:
        """Extend short missing socle spans at facade edges using nearby band geometry."""
        if socle_mask is None or socle_mask.size == 0 or facade_mask is None or facade_mask.size == 0:
            return socle_mask

        roi_h, roi_w = socle_mask.shape[:2]
        if roi_h < 12 or roi_w < 24:
            return socle_mask

        out = socle_mask.copy()
        col_occ = np.sum(out > 0, axis=0)
        valid_cols = np.where(col_occ > 0)[0]
        if len(valid_cols) < max(12, int(roi_w * 0.12)):
            return out

        max_gap = max(3, int(round(roi_w * max_gap_ratio)))
        bottoms = AnalysisPipeline._facade_bottom_profile(facade_mask)

        def _fill_edge(start_col: int, stop_col: int, ref_cols: np.ndarray):
            if start_col > stop_col or ref_cols.size == 0:
                return
            ref_occ = col_occ[ref_cols].astype(np.float32)
            occ_thr = max(2.0, float(np.percentile(ref_occ, 65)))
            strong_cols = ref_cols[ref_occ >= occ_thr]
            if strong_cols.size < 2:
                strong_cols = ref_cols
            ref_tops = []
            for rc in strong_cols.tolist():
                ys = np.where(out[:, int(rc)] > 0)[0]
                if ys.size > 0:
                    ref_tops.append(int(ys[0]))
            if len(ref_tops) < 2:
                return
            fill_top = int(round(float(np.percentile(np.array(ref_tops, dtype=np.float32), 35))))
            for cx in range(start_col, stop_col + 1):
                bottom = int(bottoms[cx])
                if bottom < 0:
                    continue
                top = max(0, min(bottom, fill_top))
                out[top:bottom + 1, cx] = facade_mask[top:bottom + 1, cx]

        left = int(valid_cols[0])
        if 1 <= left <= max_gap:
            scan_end = min(roi_w - 1, left + max(12, max_gap * 2))
            ref_cols = valid_cols[(valid_cols >= left + max(2, max_gap // 3)) & (valid_cols <= scan_end)]
            if ref_cols.size < 2:
                ref_cols = valid_cols[valid_cols <= scan_end]
            _fill_edge(0, left - 1, ref_cols)

        right = int(valid_cols[-1])
        right_gap = (roi_w - 1) - right
        if 1 <= right_gap <= max_gap:
            scan_start = max(0, right - max(12, max_gap * 2))
            ref_cols = valid_cols[(valid_cols <= right - max(2, max_gap // 3)) & (valid_cols >= scan_start)]
            if ref_cols.size < 2:
                ref_cols = valid_cols[valid_cols >= scan_start]
            _fill_edge(right + 1, roi_w - 1, ref_cols)

        return out


    @staticmethod
    def _profile_contour_from_mask(mask: np.ndarray,
                                   support_mask: Optional[np.ndarray] = None,
                                   smooth_window: int = 9,
                                   prefer_thick_columns: bool = False) -> Optional[np.ndarray]:
        """Build a cleaner closed contour from top/bottom profiles of a band-like mask."""
        if mask is None or mask.size == 0:
            return None

        roi_h, roi_w = mask.shape[:2]
        src_cols = []
        src_top = []
        src_bottom = []
        heights = []
        for cx in range(roi_w):
            ys = np.where(mask[:, cx] > 0)[0]
            if ys.size == 0:
                continue
            src_cols.append(int(cx))
            src_top.append(float(ys[0]))
            src_bottom.append(float(ys[-1]))
            heights.append(float(ys[-1] - ys[0] + 1))
        if len(src_cols) < max(12, int(roi_w * 0.08)):
            return None

        src_x = np.array(src_cols, dtype=np.float32)
        top_y = np.array(src_top, dtype=np.float32)
        bottom_y = np.array(src_bottom, dtype=np.float32)
        heights_y = np.array(heights, dtype=np.float32)
        if prefer_thick_columns and heights_y.size >= 8:
            thick_thr = max(3.0, float(np.percentile(heights_y, 65)) * 0.55)
            strong = heights_y >= thick_thr
            if int(np.count_nonzero(strong)) >= max(6, int(len(src_cols) * 0.20)):
                src_x = src_x[strong]
                top_y = top_y[strong]
                bottom_y = bottom_y[strong]

        span_mask = support_mask if support_mask is not None else mask
        span_cols = np.where(np.sum(span_mask > 0, axis=0) > 0)[0]
        if len(span_cols) == 0:
            return None
        left = int(span_cols[0])
        right = int(span_cols[-1])
        span = np.arange(left, right + 1, dtype=np.float32)
        if span.size < 4:
            return None

        top_interp = np.interp(span, src_x, top_y)
        if support_mask is not None:
            support_bottoms = AnalysisPipeline._facade_bottom_profile(support_mask)
            bottom_interp = np.array([
                float(support_bottoms[int(x)]) if support_bottoms[int(x)] >= 0 else float(np.interp(x, src_x, bottom_y))
                for x in span
            ], dtype=np.float32)
        else:
            bottom_interp = np.interp(span, src_x, bottom_y)

        smooth_window = max(3, int(smooth_window))
        if smooth_window % 2 == 0:
            smooth_window += 1
        if span.size >= smooth_window:
            top_interp = AnalysisPipeline._rolling_mean(top_interp, smooth_window)
            bottom_interp = AnalysisPipeline._rolling_mean(bottom_interp, smooth_window)

        top_points = []
        bottom_points = []
        for x_f, top_f, bottom_f in zip(span.tolist(), top_interp.tolist(), bottom_interp.tolist()):
            x_i = int(round(float(x_f)))
            top_i = int(round(float(np.clip(top_f, 0, roi_h - 1))))
            bottom_i = int(round(float(np.clip(bottom_f, top_i, roi_h - 1))))
            top_points.append([x_i, top_i])
            bottom_points.append([x_i, bottom_i])

        contour = np.array(top_points + bottom_points[::-1], dtype=np.int32).reshape((-1, 1, 2))
        if contour.shape[0] < 6:
            return None
        perim = float(cv2.arcLength(contour, True))
        eps = max(1.0, perim * 0.0025)
        return cv2.approxPolyDP(contour, eps, True)

    @staticmethod
    def _has_right_side_profile_break(contour: np.ndarray) -> bool:
        """Detect a meaningful extra right-side corner beyond a flat 4-point band."""
        if contour is None or not isinstance(contour, np.ndarray) or contour.size < 10:
            return False
        pts = contour.reshape(-1, 2).astype(np.int32)
        if pts.shape[0] < 5:
            return False

        x_min = int(np.min(pts[:, 0]))
        x_max = int(np.max(pts[:, 0]))
        y_min = int(np.min(pts[:, 1]))
        y_max = int(np.max(pts[:, 1]))
        span_w = x_max - x_min
        span_h = y_max - y_min
        if span_w < 80 or span_h < 60:
            return False

        right_band_start = x_max - max(12, int(round(span_w * 0.05)))
        min_drop = max(8, int(round(span_h * 0.12)))
        upper_limit = y_max - max(10, int(round(span_h * 0.08)))
        for px, py in pts.tolist():
            if px < right_band_start:
                continue
            if (y_min + min_drop) <= py <= upper_limit:
                return True
        return False

    @staticmethod
    def _simplified_contour_vertex_count(contour: np.ndarray,
                                         simplify_ratio: float = 0.004) -> int:
        """Count vertices after the same contour simplification used in regression output."""
        if contour is None or not isinstance(contour, np.ndarray) or contour.size < 6:
            return 0
        contour_i = contour.astype(np.int32).reshape((-1, 1, 2))
        perim = float(cv2.arcLength(contour_i, True))
        eps = max(1.0, perim * float(simplify_ratio))
        approx = cv2.approxPolyDP(contour_i, eps, True)
        if approx is None or approx.size < 6:
            return 0
        return int(approx.reshape(-1, 2).shape[0])

    def _contour_keeps_facade_openings(self,
                                       facade: ParsedFacade,
                                       contour_local: np.ndarray,
                                       x0: int,
                                       y0: int,
                                       element_types: Optional[set[str]] = None) -> bool:
        """Reject facade regularization candidates that cut through retained openings."""
        if contour_local is None or not isinstance(contour_local, np.ndarray) or contour_local.size < 6:
            return False

        contour_global = contour_local.astype(np.int32).copy()
        contour_global[:, 0, 0] += x0
        contour_global[:, 0, 1] += y0
        openings = [
            e for e in self.parsed_elements
            if e.parent_facade == facade.name and e.bbox
        ]
        if not openings:
            return True

        kept_centers = 0
        for elem in openings:
            if element_types is not None and elem.element_type not in element_types:
                continue
            ex, ey, ew, eh = elem.bbox
            cx = float(ex + ew * 0.5)
            cy = float(ey + eh * 0.5)
            if cv2.pointPolygonTest(contour_global.astype(np.float32), (cx, cy), False) >= 0:
                kept_centers += 1
        relevant = len([e for e in openings if element_types is None or e.element_type in element_types])
        if relevant == 0:
            return True
        return kept_centers >= max(1, relevant - 1)

    def _opening_rows_for_facade(self,
                                 facade: ParsedFacade,
                                 x0: int,
                                 y0: int,
                                 element_type: str = "window") -> list[dict]:
        """Group facade openings into horizontal rows using bottom-aligned clustering."""
        if not facade.region_bbox:
            return []

        openings = []
        for elem in self.parsed_elements:
            if elem.parent_facade != facade.name or elem.element_type != element_type or not elem.bbox:
                continue
            ex, ey, ew, eh = elem.bbox
            openings.append({
                "left": int(ex - x0),
                "right": int(ex + ew - x0),
                "top": int(ey - y0),
                "bottom": int(ey + eh - y0),
                "center_x": float(ex + ew * 0.5 - x0),
            })
        if not openings:
            return []

        openings.sort(key=lambda item: (item["bottom"], item["left"]))
        row_tol = max(14, int(round(float(facade.region_bbox[3]) * 0.05)))
        rows: list[dict] = []
        for item in openings:
            if not rows:
                rows.append({
                    "items": [item],
                    "bottoms": [float(item["bottom"])],
                })
                continue

            row_bottom = float(np.median(np.array(rows[-1]["bottoms"], dtype=np.float32)))
            if abs(float(item["bottom"]) - row_bottom) > row_tol:
                rows.append({
                    "items": [item],
                    "bottoms": [float(item["bottom"])],
                })
            else:
                rows[-1]["items"].append(item)
                rows[-1]["bottoms"].append(float(item["bottom"]))

        out: list[dict] = []
        for row in rows:
            items = row["items"]
            out.append({
                "bottom": float(np.median(np.array(row["bottoms"], dtype=np.float32))),
                "left": int(min(item["left"] for item in items)),
                "right": int(max(item["right"] for item in items)),
                "center_x": float(np.median(np.array([item["center_x"] for item in items], dtype=np.float32))),
                "count": int(len(items)),
            })
        return out

    @staticmethod
    def _profile_step_break_index(profile: np.ndarray) -> Optional[int]:
        """Find the dominant descending step in a piecewise socle/facade profile."""
        if profile is None or profile.size < 4:
            return None
        valid = np.where(profile >= 0)[0]
        if valid.size < 4:
            return None
        values = profile[valid].astype(np.float32)
        span = float(np.max(values) - np.min(values))
        if span <= 6.0:
            return None
        diffs = np.diff(values)
        jump_thr = max(10.0, span * 0.18)
        hits = np.where(diffs <= -jump_thr)[0]
        if hits.size == 0:
            return None
        best = int(hits[int(np.argmin(diffs[hits]))] + 1)
        return int(valid[best])

    def _estimate_composite_socle_top_profile(self,
                                              facade: ParsedFacade,
                                              facade_mask: np.ndarray,
                                              x0: int,
                                              y0: int) -> np.ndarray:
        """Infer a stepped facade cutoff/profile for composite photo facades from top breaks and opening rows."""
        if facade.scene_type != "composite_stepped_facade" or not facade.region_bbox:
            return np.array([], dtype=np.float32)
        if facade_mask is None or facade_mask.size == 0:
            return np.array([], dtype=np.float32)

        roi_h, roi_w = facade_mask.shape[:2]
        if roi_h < 80 or roi_w < 120:
            return np.array([], dtype=np.float32)

        top_profile = self._interpolate_profile(self._mask_top_profile(facade_mask))
        support_bottoms = self._facade_bottom_profile(facade_mask).astype(np.float32)
        valid_cols = np.where((top_profile >= 0) & (support_bottoms >= 0))[0]
        if valid_cols.size < max(18, int(round(roi_w * 0.20))):
            return np.array([], dtype=np.float32)

        window_rows = self._opening_rows_for_facade(facade, x0, y0, element_type="window")
        if len(window_rows) < 2:
            return np.array([], dtype=np.float32)
        upper_row = min(window_rows, key=lambda row: row["bottom"])
        lower_row = max(window_rows, key=lambda row: row["bottom"])
        min_row_gap = max(18.0, float(facade.region_bbox[3]) * 0.10)
        if (lower_row["bottom"] - upper_row["bottom"]) < min_row_gap:
            return np.array([], dtype=np.float32)
        if lower_row["center_x"] >= upper_row["center_x"]:
            return np.array([], dtype=np.float32)

        gap_between_rows = float(upper_row["left"] - lower_row["right"])
        min_row_gap_x = max(36.0, float(roi_w) * 0.05)
        if gap_between_rows < min_row_gap_x:
            return np.array([], dtype=np.float32)

        smooth_window = max(9, int(round(float(roi_w) * 0.004)))
        if smooth_window % 2 == 0:
            smooth_window += 1
        top_smooth = self._rolling_mean(top_profile.astype(np.float32), smooth_window)
        diff = np.diff(top_smooth)
        jump_thr = max(14.0, float(facade.region_bbox[3]) * 0.04)
        jump_candidates = np.where(diff <= -jump_thr)[0] + 1

        side_pad = max(18, int(round(float(roi_w) * 0.03)))
        prefer_left = max(int(valid_cols[0]) + side_pad, int(round(float(lower_row["right"]) + side_pad * 0.75)))
        prefer_right = min(int(valid_cols[-1]) - side_pad, int(round(float(upper_row["left"]) - side_pad * 0.50)))
        lead_in = int(round(float(np.clip(gap_between_rows * 0.20, 32.0, 140.0))))
        anchor_break = int(round(float(upper_row["left"]) - float(lead_in)))
        anchor_break = max(int(valid_cols[0]) + side_pad, min(int(valid_cols[-1]) - side_pad, anchor_break))
        if prefer_right > prefer_left:
            in_gap = jump_candidates[(jump_candidates >= prefer_left) & (jump_candidates <= prefer_right)]
        else:
            in_gap = np.array([], dtype=np.int32)
        central = jump_candidates[
            (jump_candidates >= int(valid_cols[0]) + max(24, int(round(float(roi_w) * 0.14))))
            & (jump_candidates <= int(valid_cols[-1]) - max(24, int(round(float(roi_w) * 0.14))))
        ]
        chosen_pool = in_gap if in_gap.size > 0 else central
        if chosen_pool.size > 0:
            deltas = np.abs(chosen_pool.astype(np.float32) - float(anchor_break))
            best_idx = int(np.argmin(deltas))
            break_idx = int(chosen_pool[best_idx])
        else:
            break_idx = int(anchor_break)

        facade_y_shift = max(0, int(facade.region_bbox[1]) - y0)
        fh_ref = max(1.0, float(facade.region_bbox[3]))

        def _row_target(row_bottom_local: float) -> float:
            bottom_in_facade = max(0.0, float(row_bottom_local) - float(facade_y_shift))
            remaining_ratio = max(0.0, min(1.0, (fh_ref - bottom_in_facade) / fh_ref))
            margin = float(np.clip((remaining_ratio ** 2) * fh_ref * 0.48, 8.0, 48.0))
            return float(row_bottom_local) + margin

        upper_target = _row_target(float(upper_row["bottom"]))
        lower_target = _row_target(float(lower_row["bottom"]))
        right_side_doors = []
        for elem in self.parsed_elements:
            if elem.parent_facade != facade.name or elem.element_type != "door" or not elem.bbox:
                continue
            ex, ey, ew, eh = elem.bbox
            right_side_doors.append({
                "left": int(ex - x0),
                "right": int(ex + ew - x0),
                "top": float(ey - y0),
                "bottom": float(ey + eh - y0),
                "center_x": float(ex + ew * 0.5 - x0),
            })
        if right_side_doors:
            bottom_tol = max(26.0, fh_ref * 0.06)
            top_ceiling = float(upper_row["bottom"]) - max(10.0, fh_ref * 0.03)
            upper_right_doors = [
                door for door in right_side_doors
                if door["center_x"] >= float(break_idx) - max(12.0, float(side_pad) * 0.35)
                and abs(door["bottom"] - float(lower_row["bottom"])) <= bottom_tol
                and door["top"] <= top_ceiling
            ]
            if upper_right_doors:
                anchor_door = max(upper_right_doors, key=lambda door: door["center_x"])
                door_pad = float(np.clip((float(upper_row["bottom"]) - anchor_door["top"]) * 0.04, 0.0, 6.0))
                upper_target = min(upper_target, anchor_door["top"] + door_pad)
        min_step_drop = max(18.0, fh_ref * 0.08)
        upper_target = min(upper_target, lower_target - min_step_drop)
        if upper_target <= 0 or lower_target <= upper_target:
            return np.array([], dtype=np.float32)

        min_body_height = max(28.0, fh_ref * 0.14)
        profile = np.full(roi_w, -1.0, dtype=np.float32)
        for cx in valid_cols.tolist():
            target_y = lower_target if cx <= break_idx else upper_target
            min_bottom = float(top_profile[cx]) + min_body_height
            profile[cx] = float(max(min_bottom, min(float(support_bottoms[cx]), target_y)))

        return profile

    def _estimate_composite_socle_bottom_profile(self,
                                                 facade: ParsedFacade,
                                                 top_profile: np.ndarray,
                                                 support_mask: np.ndarray,
                                                 x0: int,
                                                 y0: int) -> np.ndarray:
        """Build a scenario-aware socle bottom profile under the stepped facade cutoff."""
        if (
            self.current_image is None
            or facade.scene_type != "composite_stepped_facade"
            or top_profile is None
            or top_profile.size == 0
            or support_mask is None
            or support_mask.size == 0
            or not facade.region_bbox
        ):
            return np.array([], dtype=np.float32)

        roi_h, roi_w = support_mask.shape[:2]
        valid_cols = np.where(np.sum(support_mask > 0, axis=0) > 0)[0]
        if valid_cols.size < max(18, int(round(roi_w * 0.20))):
            return np.array([], dtype=np.float32)

        break_idx = self._profile_step_break_index(top_profile)
        if break_idx is None:
            window_rows = self._opening_rows_for_facade(facade, x0, y0, element_type="window")
            if len(window_rows) >= 2:
                upper_row = min(window_rows, key=lambda row: row["bottom"])
                lower_row = max(window_rows, key=lambda row: row["bottom"])
                gap_between_rows = float(upper_row["left"] - lower_row["right"])
                if gap_between_rows > max(36.0, float(roi_w) * 0.05):
                    break_idx = int(round(float(upper_row["left"]) - float(np.clip(gap_between_rows * 0.20, 32.0, 140.0))))
        if break_idx is None:
            break_idx = int(round(float(valid_cols[0] + valid_cols[-1]) * 0.5))
        break_idx = max(int(valid_cols[0]), min(int(valid_cols[-1]), int(break_idx)))

        fh_ref = max(1.0, float(facade.region_bbox[3]))
        image_bottom_local = float(self.current_image.shape[0] - 1 - y0)
        support_bottoms = self._facade_bottom_profile(support_mask).astype(np.float32)
        lower_thickness = float(np.clip(fh_ref * 0.29, 92.0, 220.0))
        upper_thickness = float(np.clip(fh_ref * 0.31, 108.0, 240.0))

        right_side_doors = []
        for elem in self.parsed_elements:
            if elem.parent_facade != facade.name or elem.element_type != "door" or not elem.bbox:
                continue
            ex, ey, ew, eh = elem.bbox
            cx = float(ex + ew * 0.5 - x0)
            if cx < float(break_idx):
                continue
            right_side_doors.append({
                "top": float(ey - y0),
                "bottom": float(ey + eh - y0),
                "height": float(eh),
                "center_x": cx,
            })
        if right_side_doors:
            right_anchor = max(right_side_doors, key=lambda door: door["center_x"])
            upper_thickness = max(
                upper_thickness,
                float(right_anchor["bottom"] - right_anchor["top"]) - max(6.0, fh_ref * 0.02),
            )

        bottom_profile = np.full(roi_w, -1.0, dtype=np.float32)
        for cx in valid_cols.tolist():
            top_y = float(top_profile[cx])
            if top_y < 0:
                continue
            base_thickness = lower_thickness if cx <= break_idx else upper_thickness
            support_floor = float(support_bottoms[cx]) if 0 <= cx < support_bottoms.size and support_bottoms[cx] >= 0 else -1.0
            target = top_y + base_thickness
            if support_floor >= 0:
                target = max(target, support_floor + (10.0 if cx <= break_idx else 4.0))
            bottom_profile[cx] = float(min(image_bottom_local, max(top_y + 24.0, target)))

        return bottom_profile

    def _regularize_flat_facade_against_socle(self, facade: ParsedFacade):
        """Rebuild flat facade contour while preserving the existing top/eave geometry."""
        if self.current_image is None or not facade.region_bbox:
            return
        if facade.scene_type != "flat_long_facade":
            return
        if not isinstance(facade.contour, np.ndarray) or facade.contour.size < 6:
            return
        if not isinstance(facade.socle_contour, np.ndarray) or facade.socle_contour.size < 6:
            return

        fx, fy, fw, fh = facade.region_bbox
        sx, sy, sw, sh = facade.socle_bbox if facade.socle_bbox else facade.region_bbox
        x0 = max(0, min(fx, sx))
        y0 = max(0, min(fy, sy))
        x1 = min(self.current_image.shape[1], max(fx + fw, sx + sw))
        y1 = min(self.current_image.shape[0], max(fy + fh, sy + sh))
        if x1 - x0 < 60 or y1 - y0 < 60:
            return

        roi_h = y1 - y0
        roi_w = x1 - x0
        roi_gray = cv2.cvtColor(self.current_image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        facade_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        socle_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        f_local = facade.contour.astype(np.int32).copy()
        f_local[:, 0, 0] -= x0
        f_local[:, 0, 1] -= y0
        s_local = facade.socle_contour.astype(np.int32).copy()
        s_local[:, 0, 0] -= x0
        s_local[:, 0, 1] -= y0
        cv2.drawContours(facade_mask, [f_local], -1, 255, -1)
        cv2.drawContours(socle_mask, [s_local], -1, 255, -1)

        socle_top_profile = np.array([], dtype=np.float32)
        if isinstance(facade.socle_profile, np.ndarray) and facade.socle_profile.size >= 4:
            profile_local = facade.socle_profile.astype(np.int32).copy()
            profile_local[:, 0, 0] -= x0
            profile_local[:, 0, 1] -= y0
            socle_top_profile = self._profile_from_line_contour(profile_local, roi_w)
        if socle_top_profile.size == 0:
            socle_top_profile = self._interpolate_profile(self._mask_top_profile(socle_mask))
        if socle_top_profile.size == 0:
            return

        final_mask = self._cut_mask_below_profile(facade_mask, socle_top_profile, pad_px=0)
        span_left, span_right = self._line_span(facade.socle_profile)
        span_left_local = (span_left - x0) if span_left is not None else None
        span_right_local = (span_right - x0) if span_right is not None else None
        final_mask = self._clip_mask_to_x_span(final_mask, span_left_local, span_right_local)
        final_mask = self._trim_flat_photo_facade_body_support(
            final_mask,
            facade,
            roi_gray,
            allow_side_trim=False,
        )
        final_mask = self._clip_mask_to_x_span(final_mask, span_left_local, span_right_local)
        final_mask = cv2.morphologyEx(
            final_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )
        final_mask = cv2.morphologyEx(
            final_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        )

        base_area = max(1.0, float(cv2.contourArea(f_local)))
        chosen_contour = None
        chosen_area = 0.0
        linear_contour = None
        linear_area = 0.0

        top_fit = self._fit_mask_edge_line(
            final_mask,
            edge="top",
            support_mask=final_mask,
            prefer_thick_columns=False,
            min_cols_ratio=0.30,
            max_slope=0.10,
        )
        if (
            top_fit
            and isinstance(facade.socle_profile, np.ndarray)
            and facade.socle_profile.size >= 4
            and span_left is not None
            and span_right is not None
        ):
            pts = facade.socle_profile.reshape(-1, 2).astype(np.float32).copy()
            pts[:, 0] -= x0
            pts[:, 1] -= y0
            order = np.argsort(pts[:, 0], kind="mergesort")
            pts = pts[order]
            if pts.shape[0] >= 2:
                x_a, y_a = pts[0]
                x_b, y_b = pts[-1]
                dx = max(1.0, float(x_b - x_a))
                a_bottom = float(np.clip((y_b - y_a) / dx, -0.12, 0.12))
                b_bottom = float(y_a - a_bottom * x_a)
                linear_contour = self._linear_band_contour_from_lines(
                    int(span_left_local),
                    int(span_right_local),
                    (top_fit[2], top_fit[3]),
                    (a_bottom, b_bottom),
                    roi_h,
                )
                if linear_contour is not None:
                    linear_area = float(cv2.contourArea(linear_contour))
                    if (
                        base_area * 0.70 <= linear_area <= base_area * 1.05
                        and self._contour_keeps_facade_openings(facade, linear_contour, x0, y0)
                    ):
                        chosen_contour = linear_contour
                        chosen_area = linear_area

        profile_contour = self._profile_contour_from_mask(
            final_mask,
            support_mask=final_mask,
            smooth_window=max(9, int(round(roi_w * 0.016)) | 1),
            prefer_thick_columns=False,
        )
        if profile_contour is not None:
            prof_area = float(cv2.contourArea(profile_contour))
            profile_ok = base_area * 0.55 <= prof_area <= base_area * 1.02
            if profile_ok:
                bx, by, bw, bh = cv2.boundingRect(profile_contour)
                if span_left is not None and span_right is not None:
                    span_width = max(1, int(span_right - span_left + 1))
                    if span_width >= int(roi_w * 0.70) and bw < int(span_width * 0.97):
                        profile_ok = False
                if profile_ok and not self._contour_keeps_facade_openings(facade, profile_contour, x0, y0):
                    profile_ok = False
            if profile_ok:
                if chosen_contour is None:
                    chosen_contour = profile_contour
                    chosen_area = prof_area
                elif self._has_right_side_profile_break(profile_contour):
                    chosen_contour = profile_contour
                    chosen_area = prof_area

        if chosen_contour is None:
            return

        for _ in range(6):
            chosen_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            cv2.drawContours(chosen_mask, [chosen_contour.astype(np.int32)], -1, 255, -1)
            refined_profile = self._profile_contour_from_mask(
                chosen_mask,
                support_mask=chosen_mask,
                smooth_window=max(9, int(round(roi_w * 0.016)) | 1),
                prefer_thick_columns=False,
            )
            if refined_profile is None:
                break

            refined_area = float(cv2.contourArea(refined_profile))
            refined_ok = base_area * 0.55 <= refined_area <= base_area * 1.02
            if refined_ok:
                bx, by, bw, bh = cv2.boundingRect(refined_profile)
                if span_left is not None and span_right is not None:
                    span_width = max(1, int(span_right - span_left + 1))
                    if span_width >= int(roi_w * 0.70) and bw < int(span_width * 0.97):
                        refined_ok = False
                if refined_ok and not self._contour_keeps_facade_openings(facade, refined_profile, x0, y0):
                    refined_ok = False
                if refined_ok and not self._has_right_side_profile_break(refined_profile):
                    refined_ok = False
            if not refined_ok:
                break

            current_simplified_vertices = self._simplified_contour_vertex_count(chosen_contour)
            refined_simplified_vertices = self._simplified_contour_vertex_count(refined_profile)
            replace_profile = refined_simplified_vertices > current_simplified_vertices
            if not replace_profile and not self._has_right_side_profile_break(chosen_contour):
                replace_profile = True
            if not replace_profile:
                shape_delta = cv2.matchShapes(
                    chosen_contour.astype(np.float32),
                    refined_profile.astype(np.float32),
                    cv2.CONTOURS_MATCH_I1,
                    0.0,
                )
                replace_profile = shape_delta > 0.0005 and refined_simplified_vertices >= current_simplified_vertices
            if not replace_profile:
                break

            chosen_contour = refined_profile
            chosen_area = refined_area
            if refined_simplified_vertices > 4:
                break

        if (
            not self._has_right_side_profile_break(chosen_contour)
            or self._simplified_contour_vertex_count(chosen_contour) <= 4
        ):
            tapered = self._recover_flat_right_eave_taper(
                facade,
                chosen_contour,
                x0,
                y0,
                roi_w,
                roi_h,
                socle_top_profile,
            )
            if tapered is not None:
                chosen_contour, chosen_area = tapered

        bx, by, bw, bh = cv2.boundingRect(chosen_contour)
        contour_global = chosen_contour.copy()
        contour_global[:, 0, 0] += x0
        contour_global[:, 0, 1] += y0
        facade.contour = contour_global
        facade.region_bbox = (x0 + bx, y0 + by, bw, bh)
        facade.area_px = chosen_area
        if self.area_scale_px_per_m2 and self.area_scale_px_per_m2 > 0:
            facade.total_area = round(float(chosen_area / self.area_scale_px_per_m2), 3)

    def _recover_flat_right_eave_taper(self,
                                       facade: ParsedFacade,
                                       contour_local: np.ndarray,
                                       x0: int,
                                       y0: int,
                                       roi_w: int,
                                       roi_h: int,
                                       socle_top_profile: np.ndarray):
        """Rebuild a robust right-side taper that survives downstream contour simplification."""
        if (
            self.current_image is None
            or contour_local is None
            or not isinstance(contour_local, np.ndarray)
            or contour_local.size < 6
            or socle_top_profile is None
            or socle_top_profile.size == 0
        ):
            return None

        bx, by, bw, bh = cv2.boundingRect(contour_local)
        if bw < 120 or bh < 80:
            return None

        contour_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour_local.astype(np.int32)], -1, 255, -1)
        top_profile = self._mask_top_profile(contour_mask)
        valid_cols = np.where(top_profile >= 0)[0]
        if valid_cols.size < max(40, int(roi_w * 0.28)):
            return None

        left = int(valid_cols[0])
        right = int(valid_cols[-1])
        if right <= left + 40:
            return None

        current_simplified_vertices = self._simplified_contour_vertex_count(contour_local)
        core_end = min(right, left + int(round((right - left) * 0.62)))
        core_top = top_profile[left:core_end + 1]
        core_top = core_top[core_top >= 0]
        if core_top.size < max(24, int((right - left) * 0.18)):
            return None
        base_top = int(round(float(np.percentile(core_top.astype(np.float32), 35))))

        openings = [
            e for e in self.parsed_elements
            if e.parent_facade == facade.name and e.bbox
        ]
        right_windows = []
        for elem in openings:
            if elem.element_type != "window":
                continue
            ex, ey, ew, eh = elem.bbox
            cx_global = ex + ew * 0.5
            if cx_global < (x0 + left + (right - left) * 0.55):
                continue
            right_windows.append((ex, ey, ew, eh))
        if not right_windows:
            return None

        right_window = max(right_windows, key=lambda item: item[0] + item[2])
        rw_right_local = int(right_window[0] + right_window[2] - x0)
        taper_start = max(
            int(left + (right - left) * 0.78),
            rw_right_local + max(16, int(round((right - left) * 0.015))),
        )
        min_remaining = max(96, int(round((right - left) * 0.08)))
        if taper_start >= right - min_remaining:
            return None

        remaining = right - taper_start
        if remaining < min_remaining:
            return None

        bottom_left = socle_top_profile[left] if left < socle_top_profile.size else -1
        bottom_right = socle_top_profile[right] if right < socle_top_profile.size else -1
        if bottom_left < 0 or bottom_right < 0:
            return None
        bottom_left = int(round(float(bottom_left)))
        bottom_right = int(round(float(bottom_right)))
        if bottom_left <= base_top + 20 or bottom_right <= base_top + 40:
            return None

        shoulder_y = min(
            bottom_right - max(38, int(round(roi_h * 0.08))),
            base_top + max(48, int(round(roi_h * 0.24))),
        )
        if shoulder_y <= base_top + 18:
            return None

        shoulder_x = max(
            taper_start + max(42, int(round(remaining * 0.34))),
            int(left + (right - left) * 0.86),
        )
        shoulder_x = min(shoulder_x, right - max(78, int(round((right - left) * 0.05))))
        if shoulder_x <= taper_start + 24:
            return None

        top_chain = [
            (left, base_top),
            (taper_start, base_top),
            (shoulder_x, shoulder_y),
            (right, shoulder_y),
        ]
        tapered_contour = np.array(
            [
                [px, py] for px, py in top_chain
            ] + [
                [right, bottom_right],
                [left, bottom_left],
            ],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        tapered_area = float(cv2.contourArea(tapered_contour))
        base_area = max(1.0, float(cv2.contourArea(contour_local)))
        if not (base_area * 0.68 <= tapered_area <= base_area * 1.12):
            return None
        tapered_simplified_vertices = self._simplified_contour_vertex_count(tapered_contour)
        if tapered_simplified_vertices <= max(4, current_simplified_vertices):
            return None

        tapered_global = tapered_contour.astype(np.int32).copy()
        tapered_global[:, 0, 0] += x0
        tapered_global[:, 0, 1] += y0

        def _top_y_at(cx_local: int) -> float:
            if cx_local <= top_chain[1][0]:
                return float(base_top)
            for (x_a, y_a), (x_b, y_b) in zip(top_chain, top_chain[1:]):
                if x_a <= cx_local <= x_b:
                    alpha = (cx_local - x_a) / max(1.0, float(x_b - x_a))
                    return float(y_a + (y_b - y_a) * alpha)
            return float(top_chain[-1][1])

        for elem in openings:
            ex, ey, ew, eh = elem.bbox
            cx = int(round(ex + ew * 0.5)) - x0
            if not (0 <= cx < roi_w):
                continue
            allowed_top = int(round(ey - y0 + eh * 0.18))
            top_here = int(round(_top_y_at(cx)))
            if top_here > allowed_top:
                return None
            if cv2.pointPolygonTest(
                tapered_global.astype(np.float32),
                (float(ex + ew * 0.5), float(ey + eh * 0.5)),
                False,
            ) < 0:
                return None

        return tapered_contour, tapered_area

    def _regularize_sparse_facade_from_seed_hull(self, facade: ParsedFacade):
        if self.current_image is None or not facade.region_bbox:
            return
        base_contour = None
        if isinstance(facade.seed_contour, np.ndarray) and facade.seed_contour.size >= 6:
            base_contour = facade.seed_contour.astype(np.int32)
        elif isinstance(facade.contour, np.ndarray) and facade.contour.size >= 6:
            base_contour = facade.contour.astype(np.int32)
        if base_contour is None:
            return

        hull = cv2.convexHull(base_contour)
        if hull is None or len(hull) < 4:
            return

        chosen = None
        hull_area = max(1.0, float(cv2.contourArea(hull)))
        for eps_ratio in (0.03, 0.02, 0.015, 0.012):
            eps = cv2.arcLength(hull, True) * float(eps_ratio)
            approx = cv2.approxPolyDP(hull, eps, True)
            if approx is None or len(approx) < 4 or len(approx) > 6:
                continue
            ax, ay, aw, ah = cv2.boundingRect(approx)
            if aw < int(facade.region_bbox[2] * 0.88):
                continue
            if ah < int(facade.region_bbox[3] * 0.95):
                continue
            area = float(cv2.contourArea(approx))
            if not (hull_area * 0.72 <= area <= hull_area * 1.02):
                continue
            chosen = approx.astype(np.int32)
            break
        if chosen is None:
            return

        pts = chosen.reshape(-1, 2).astype(np.int32)
        bottom_thr = np.percentile(pts[:, 1], 80)
        bottom_pts = pts[pts[:, 1] >= bottom_thr]
        if bottom_pts.shape[0] < 2:
            return
        left_bottom = bottom_pts[np.argmin(bottom_pts[:, 0])]
        right_bottom = bottom_pts[np.argmax(bottom_pts[:, 0])]
        left_x = int(left_bottom[0])
        right_x = int(right_bottom[0])
        top_left_y = int(left_bottom[1])
        top_right_y = int(right_bottom[1])
        span_w = max(1, right_x - left_x)

        facade.contour = chosen
        fx, fy, fw, fh = cv2.boundingRect(chosen)
        facade.region_bbox = (fx, fy, fw, fh)
        facade.area_px = float(cv2.contourArea(chosen))
        if self.area_scale_px_per_m2 and self.area_scale_px_per_m2 > 0:
            facade.total_area = round(float(facade.area_px / self.area_scale_px_per_m2), 3)

        socle_h_ref = facade.socle_bbox[3] if facade.socle_bbox else max(22, int(round(fh * 0.08)))
        band_left = max(16, int(round(socle_h_ref * 0.28)))
        band_right = max(band_left + 2, int(round(socle_h_ref * 0.40)))
        x_l1 = min(right_x, left_x + max(2, int(round(span_w * 0.003))))
        x_l2 = min(right_x, left_x + max(4, int(round(span_w * 0.006))))
        x_r1 = right_x + max(8, int(round(span_w * 0.012)))
        x_r2 = right_x + max(7, int(round(span_w * 0.011)))
        x_r3 = right_x + max(5, int(round(span_w * 0.008)))
        img_w = self.current_image.shape[1]
        img_h = self.current_image.shape[0]
        x_r1 = min(img_w - 1, x_r1)
        x_r2 = min(img_w - 1, x_r2)
        x_r3 = min(img_w - 1, x_r3)
        bot_left = min(img_h - 1, top_left_y + band_left)
        bot_right = min(img_h - 1, top_right_y + band_right)

        socle = np.array(
            [
                [left_x, top_left_y],
                [x_l1, bot_left],
                [x_r1, bot_right],
                [x_r2, min(img_h - 1, top_right_y + 1)],
                [x_r3, max(min(img_h - 1, bot_right - 1), top_right_y + 1)],
                [x_l2, max(top_left_y + 1, bot_left - 2)],
            ],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        facade.socle_contour = socle
        sx, sy, sw, sh = cv2.boundingRect(socle)
        facade.socle_bbox = (sx, sy, sw, sh)
        facade.socle_profile = np.array(
            [[left_x, top_left_y], [x_r2, min(img_h - 1, top_right_y + 1)]],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        socle_area_px = float(cv2.contourArea(socle))
        facade.socle_excluded_area = round(
            float(socle_area_px / self.area_scale_px_per_m2), 3
        ) if self.area_scale_px_per_m2 and self.area_scale_px_per_m2 > 0 else 0.0

    def _regularize_composite_facade_bottom_against_socle(self, facade: ParsedFacade):
        if self.current_image is None or not facade.region_bbox or not facade.socle_bbox:
            return
        if not isinstance(facade.contour, np.ndarray) or facade.contour.size < 6:
            return
        if not isinstance(facade.socle_contour, np.ndarray) or facade.socle_contour.size < 6:
            return

        fx, fy, fw, fh = facade.region_bbox
        sx, sy, sw, sh = facade.socle_bbox
        x0 = max(0, min(fx, sx))
        y0 = max(0, min(fy, sy))
        x1 = min(self.current_image.shape[1], max(fx + fw, sx + sw))
        extra_bottom = max(40, int(round(float(fh) * 0.34)))
        y1 = min(self.current_image.shape[0], max(fy + fh, sy + sh, fy + fh + extra_bottom))
        if x1 - x0 < 60 or y1 - y0 < 60:
            return

        roi_h = y1 - y0
        roi_w = x1 - x0
        facade_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        socle_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        f_local = facade.contour.astype(np.int32).copy()
        f_local[:, 0, 0] -= x0
        f_local[:, 0, 1] -= y0
        s_local = facade.socle_contour.astype(np.int32).copy()
        s_local[:, 0, 0] -= x0
        s_local[:, 0, 1] -= y0
        cv2.drawContours(facade_mask, [f_local], -1, 255, -1)
        cv2.drawContours(socle_mask, [s_local], -1, 255, -1)
        base_area = max(1.0, float(cv2.contourArea(f_local)))

        stepped_profile = self._estimate_composite_socle_top_profile(facade, facade_mask, x0, y0)
        top_profile = self._interpolate_profile(self._mask_top_profile(facade_mask))
        if stepped_profile.size > 0 and top_profile.size > 0:
            valid_cols = np.where(np.sum(facade_mask > 0, axis=0) > 0)[0]
            rebuilt_facade_mask = self._build_mask_from_profiles(top_profile, stepped_profile, facade_mask)
            rebuilt_facade_mask = self._clip_band_mask_to_support_columns(rebuilt_facade_mask, facade_mask)
            rebuilt_facade_mask = cv2.morphologyEx(
                rebuilt_facade_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            )
            socle_bottom_profile = self._estimate_composite_socle_bottom_profile(
                facade,
                stepped_profile,
                socle_mask,
                x0,
                y0,
            )

            contour_candidate = self._closed_contour_from_profiles(
                top_profile,
                stepped_profile,
                rebuilt_facade_mask,
            )
            socle_candidate = self._closed_contour_from_profiles(
                stepped_profile,
                socle_bottom_profile,
                facade_mask,
            )
            left_bound = int(valid_cols[0]) if valid_cols.size > 0 else None
            right_bound = int(valid_cols[-1]) if valid_cols.size > 0 else None
            profile_line_local = self._polyline_contour_from_profile(
                stepped_profile,
                left=left_bound,
                right=right_bound,
            )
            if (
                contour_candidate is not None
                and socle_candidate is not None
                and isinstance(profile_line_local, np.ndarray)
                and profile_line_local.size >= 4
            ):
                contour_area = float(cv2.contourArea(contour_candidate))
                if (
                    base_area * 0.38 <= contour_area <= base_area * 1.02
                    and self._contour_keeps_facade_openings(
                        facade,
                        contour_candidate,
                        x0,
                        y0,
                        element_types={"window"},
                    )
                ):
                    contour_global = contour_candidate.copy()
                    contour_global[:, 0, 0] += x0
                    contour_global[:, 0, 1] += y0
                    bx, by, bw, bh = cv2.boundingRect(contour_candidate)
                    facade.contour = contour_global
                    facade.region_bbox = (x0 + bx, y0 + by, bw, bh)
                    facade.area_px = contour_area
                    if self.area_scale_px_per_m2 and self.area_scale_px_per_m2 > 0:
                        facade.total_area = round(float(contour_area / self.area_scale_px_per_m2), 3)

                    socle_global = socle_candidate.copy()
                    socle_global[:, 0, 0] += x0
                    socle_global[:, 0, 1] += y0
                    sx, sy, sw, sh = cv2.boundingRect(socle_candidate)
                    facade.socle_contour = socle_global
                    facade.socle_bbox = (x0 + sx, y0 + sy, sw, sh)

                    profile_global = profile_line_local.copy()
                    profile_global[:, 0, 0] += x0
                    profile_global[:, 0, 1] += y0
                    facade.socle_profile = profile_global

                    socle_area_px = float(cv2.contourArea(socle_candidate))
                    facade.socle_excluded_area = round(
                        float(socle_area_px / self.area_scale_px_per_m2), 3
                    ) if self.area_scale_px_per_m2 and self.area_scale_px_per_m2 > 0 else 0.0
                    return

        # Fallback path still requires an explicit socle profile line. The
        # _mask_top_profile fallback alone can lock onto a mis-positioned band
        # and destroy the composite facade body.
        if not (isinstance(facade.socle_profile, np.ndarray) and facade.socle_profile.size >= 4):
            return

        socle_top_profile = np.array([], dtype=np.float32)
        if isinstance(facade.socle_profile, np.ndarray) and facade.socle_profile.size >= 4:
            profile_local = facade.socle_profile.astype(np.int32).copy()
            profile_local[:, 0, 0] -= x0
            profile_local[:, 0, 1] -= y0
            socle_top_profile = self._profile_from_line_contour(profile_local, roi_w)
        if socle_top_profile.size == 0:
            socle_top_profile = self._interpolate_profile(self._mask_top_profile(socle_mask))
        if socle_top_profile.size == 0:
            return

        final_mask = self._cut_mask_below_profile(facade_mask, socle_top_profile, pad_px=0)
        span_left, span_right = self._line_span(facade.socle_profile)
        span_left_local = (span_left - x0) if span_left is not None else None
        span_right_local = (span_right - x0) if span_right is not None else None
        final_mask = self._clip_mask_to_x_span(final_mask, span_left_local, span_right_local)
        final_mask = cv2.morphologyEx(
            final_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )
        final_mask = cv2.morphologyEx(
            final_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        )

        smooth_window = max(7, int(round(roi_w * 0.012)))
        if smooth_window % 2 == 0:
            smooth_window += 1
        profile_contour = self._profile_contour_from_mask(
            final_mask,
            support_mask=final_mask,
            smooth_window=smooth_window,
            prefer_thick_columns=False,
        )
        if profile_contour is None:
            return

        prof_area = float(cv2.contourArea(profile_contour))
        if not (base_area * 0.55 <= prof_area <= base_area * 1.02):
            return

        bx, by, bw, bh = cv2.boundingRect(profile_contour)
        contour_global = profile_contour.copy()
        contour_global[:, 0, 0] += x0
        contour_global[:, 0, 1] += y0
        facade.contour = contour_global
        facade.region_bbox = (x0 + bx, y0 + by, bw, bh)
        facade.area_px = prof_area
        if self.area_scale_px_per_m2 and self.area_scale_px_per_m2 > 0:
            facade.total_area = round(float(prof_area / self.area_scale_px_per_m2), 3)

    def _estimate_socle_mask_from_magenta(self, roi_bgr: np.ndarray,
                                          facade_mask: np.ndarray) -> Optional[np.ndarray]:
        """Estimate socle band from explicit low guide strokes when present."""
        if roi_bgr is None or roi_bgr.size == 0 or facade_mask is None:
            return None

        roi_h, roi_w = facade_mask.shape[:2]
        if roi_h < 20 or roi_w < 20:
            return None

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mag_mask = cv2.inRange(
            hsv,
            np.array((132, 70, 80), dtype=np.uint8),
            np.array((179, 255, 255), dtype=np.uint8),
        )
        red_low = cv2.inRange(
            hsv,
            np.array((0, 140, 110), dtype=np.uint8),
            np.array((8, 255, 255), dtype=np.uint8),
        )
        red_high = cv2.inRange(
            hsv,
            np.array((170, 140, 110), dtype=np.uint8),
            np.array((179, 255, 255), dtype=np.uint8),
        )
        guide_mask = cv2.bitwise_or(mag_mask, red_low)
        guide_mask = cv2.bitwise_or(guide_mask, red_high)
        guide_mask = cv2.morphologyEx(
            guide_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
        )
        guide_mask = cv2.morphologyEx(
            guide_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )
        guide_mask[:int(roi_h * 0.55), :] = 0
        guide_mask = cv2.bitwise_and(guide_mask, facade_mask)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(guide_mask, 8)
        if n_labels <= 1:
            return None

        min_top = int(roi_h * float(getattr(config, "PHOTO_SOCLE_GUIDE_MIN_TOP_RATIO", 0.68)))
        min_width = int(roi_w * float(getattr(config, "PHOTO_SOCLE_GUIDE_MIN_WIDTH_RATIO", 0.12)))
        max_height = max(18, int(roi_h * float(getattr(config, "PHOTO_SOCLE_GUIDE_MAX_HEIGHT_RATIO", 0.22))))
        keep_mask = np.zeros_like(guide_mask)
        top_candidates = []
        for idx in range(1, n_labels):
            x, y, w, h, area = stats[idx]
            if area < max(18, int(roi_w * 0.008)):
                continue
            if y < min_top:
                continue
            if w < min_width and (w / max(1.0, float(h))) < 4.0:
                continue
            if h > max_height:
                continue
            keep_mask[labels == idx] = 255
            top_candidates.append(int(y))

        if not top_candidates:
            return None

        q_top = int(np.percentile(np.array(top_candidates, dtype=np.float32), 60))
        keep_mask[:q_top, :] = 0
        col_cov = np.count_nonzero(np.any(keep_mask > 0, axis=0)) / max(1, roi_w)
        if col_cov < float(getattr(config, "PHOTO_SOCLE_GUIDE_MIN_COVERAGE", 0.25)):
            return None

        ys, xs = np.where(keep_mask > 0)
        if len(xs) < 30 or len(np.unique(xs)) < 8:
            return None

        top_pts = {}
        for x_i, y_i in zip(xs.tolist(), ys.tolist()):
            if x_i not in top_pts or y_i < top_pts[x_i]:
                top_pts[x_i] = y_i
        fit_x = np.array(sorted(top_pts.keys()), dtype=np.float32)
        fit_y = np.array([top_pts[int(x)] for x in fit_x], dtype=np.float32)
        if fit_x.size < 8:
            return None

        a, b = np.polyfit(fit_x, fit_y, 1)
        max_slope = float(getattr(config, "PHOTO_SOCLE_GUIDE_MAX_SLOPE", 0.12))
        a = float(np.clip(a, -max_slope, max_slope))
        tops = np.empty(roi_w, dtype=np.float32)
        for cx in range(roi_w):
            cy = int(round(a * cx + b))
            tops[cx] = float(max(min_top, min(roi_h - 1, cy)))

        socle_mask = self._build_socle_band_from_bottoms(facade_mask, tops)
        if np.count_nonzero(socle_mask) < max(120, int(roi_w * roi_h * 0.015)):
            return None
        return socle_mask

    def _exclude_socle_from_facade_areas(self):
        """Estimate and subtract socle band from auto/photo facades."""
        if self.current_image is None:
            return
        if not self.area_scale_px_per_m2 or self.area_scale_px_per_m2 <= 0:
            return

        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        img_h, img_w = gray.shape[:2]
        for facade in self.parsed_facades:
            # Reset per-run socle geometry; recomputed below when valid.
            facade.socle_bbox = None
            facade.socle_contour = np.array([])
            facade.socle_profile = np.array([])
            facade.socle_excluded_area = 0.0

            if facade.total_area <= 0:
                continue
            if not facade.region_bbox:
                continue
            # Apply only on automatically segmented facade-like regions.
            src = (facade.source or "").lower()
            if "photo" not in src and "inferred" not in src:
                continue
            # If OCR already supplied a reliable secondary area, keep original area.
            if facade.net_area > 0:
                continue

            base_bbox = facade.region_bbox
            base_contour_override = None
            if (
                facade.scene_type in {"gable_facade", "sparse_openings_flat"}
                and facade.seed_region_bbox
            ):
                seed_x, seed_y, seed_w, seed_h = facade.seed_region_bbox
                if seed_w >= (facade.region_bbox[2] if facade.region_bbox else 0):
                    base_bbox = facade.seed_region_bbox
                direct_geometry_applied = False
                direct_socle_profile_global = np.array([])
                if facade.scene_type == "gable_facade":
                    recovered = self._recover_gable_silhouette_from_seed(facade)
                    if os.environ.get("DQP_DEBUG_GEOM"):
                        print("DEBUG_EXCLUDE_RECOVERED", facade.scene_type, recovered is not None)
                    if recovered is not None:
                        recovered_contour, recovered_bbox, _ = recovered
                        if os.environ.get("DQP_DEBUG_GEOM"):
                            print("DEBUG_EXCLUDE_RECOVERED_BBOX", recovered_bbox)
                        base_bbox = recovered_bbox
                        base_contour_override = recovered_contour

            x, y, w, h = base_bbox
            if w < 80 or h < 80:
                continue
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(img_w, x + w)
            y1 = min(img_h, y + h)
            if x1 - x0 < 40 or y1 - y0 < 40:
                continue

            roi = gray[y0:y1, x0:x1]
            if roi.size == 0:
                continue

            roi_h, roi_w = roi.shape[:2]
            if os.environ.get("DQP_DEBUG_GEOM"):
                print("DEBUG_EXCLUDE_BASE", facade.scene_type, "base_bbox", base_bbox, "override", base_contour_override is not None)
            facade_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            base_contour = base_contour_override if base_contour_override is not None else facade.contour
            if (
                facade.scene_type in {"gable_facade", "sparse_openings_flat"}
                and isinstance(facade.seed_contour, np.ndarray)
                and facade.seed_contour.size >= 6
            ):
                if not (isinstance(base_contour, np.ndarray) and base_contour.size >= 6):
                    base_contour = facade.seed_contour
                else:
                    cur_x, cur_y, cur_w, cur_h = cv2.boundingRect(base_contour.astype(np.int32))
                    seed_x, seed_y, seed_w, seed_h = cv2.boundingRect(facade.seed_contour.astype(np.int32))
                    if seed_w > cur_w * 1.12 or (seed_w >= cur_w * 0.98 and seed_h > cur_h * 1.08):
                        base_contour = facade.seed_contour
            if isinstance(base_contour, np.ndarray) and base_contour.size >= 6:
                c_local = base_contour.astype(np.int32).copy()
                c_local[:, 0, 0] -= x0
                c_local[:, 0, 1] -= y0
                cv2.drawContours(facade_mask, [c_local], -1, 255, -1)
            else:
                facade_mask[:, :] = 255

            if np.count_nonzero(facade_mask) < 200:
                continue

            # Robust horizontal transition detector in lower facade area,
            # constrained strictly inside the facade contour.
            blur = cv2.GaussianBlur(roi, (5, 5), 0)
            grad_y = np.abs(cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3))
            support = (facade_mask > 0).astype(np.float32)
            row_occ = np.sum(support, axis=1)
            max_occ = float(max(1.0, np.max(row_occ)))
            row_score = np.sum(grad_y * support, axis=1) / (row_occ + 1e-6)

            start = int(roi_h * float(getattr(config, "PHOTO_SOCLE_Y_START", 0.58)))
            end = int(roi_h * float(getattr(config, "PHOTO_SOCLE_Y_END", 0.96)))
            if end <= start + 5:
                continue

            zone = row_score[start:end]
            zone_occ = row_occ[start:end]
            if zone.size == 0:
                continue
            occ_ratio = float(getattr(config, "PHOTO_SOCLE_MIN_ROW_OCC_RATIO", 0.28))
            valid_rows = zone_occ >= (max_occ * occ_ratio)
            if not np.any(valid_rows):
                continue
            zone_valid = np.where(valid_rows, zone, 0.0)
            peak_idx = int(np.argmax(zone_valid))
            peak_score = float(zone_valid[peak_idx])
            median_score = float(np.median(zone[valid_rows])) + 1e-6
            peak_ratio = float(getattr(config, "PHOTO_SOCLE_PEAK_RATIO", 1.22))
            if peak_score < median_score * peak_ratio:
                continue

            first_ratio = float(getattr(config, "PHOTO_SOCLE_FIRST_EDGE_RATIO", 0.72))
            first_thr = max(median_score * peak_ratio, peak_score * first_ratio)
            first_candidates = np.where((zone_valid >= first_thr) & valid_rows)[0]
            min_frac = float(getattr(config, "PHOTO_SOCLE_MIN_BAND_RATIO", 0.04))
            max_frac = float(getattr(config, "PHOTO_SOCLE_MAX_BAND_RATIO", 0.30))
            candidate_idxs = [int(i) for i in first_candidates.tolist()]
            if peak_idx not in candidate_idxs:
                candidate_idxs.append(int(peak_idx))
            # For composite stepped facades, prefer the peak (strongest gradient)
            # over the first ascending candidate — the first candidate often
            # selects a step boundary instead of the actual socle-facade edge.
            if facade.scene_type == "composite_stepped_facade":
                candidate_idxs = sorted(candidate_idxs, key=lambda i: abs(i - peak_idx))
            cut = None
            for c_idx in candidate_idxs:
                c_cut = start + c_idx
                c_band_px = roi_h - c_cut
                c_frac = c_band_px / max(1, roi_h)
                if min_frac <= c_frac <= max_frac:
                    cut = c_cut
                    band_px = c_band_px
                    frac = c_frac
                    break
            if cut is None:
                continue

            band_px = max(1, roi_h - cut)
            bottoms = self._facade_bottom_profile(facade_mask)
            tops = np.empty(roi_w, dtype=np.float32)
            for cx in range(roi_w):
                bottom = int(bottoms[cx])
                if bottom < 0:
                    tops[cx] = float(max(0, cut))
                    continue
                tops[cx] = float(max(0, bottom - band_px + 1))
            socle_mask = self._build_socle_band_from_bottoms(facade_mask, tops)
            socle_mask = cv2.morphologyEx(
                socle_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
            )

            # Prefer explicit magenta socle guides when available.
            roi_bgr = self.current_image[y0:y1, x0:x1]
            magenta_socle = self._estimate_socle_mask_from_magenta(
                roi_bgr, facade_mask
            )
            used_magenta_socle = magenta_socle is not None
            if used_magenta_socle:
                socle_mask = cv2.bitwise_or(socle_mask, magenta_socle)
                socle_mask = cv2.morphologyEx(
                    socle_mask,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)),
                )
            else:
                valid_bottoms = bottoms[bottoms >= 0]
                if valid_bottoms.size >= max(24, int(roi_w * 0.20)):
                    bottom_iqr = float(
                        np.percentile(valid_bottoms, 75) - np.percentile(valid_bottoms, 25)
                    )
                    flat_bottom_iqr = float(
                        getattr(config, "PHOTO_SOCLE_FLAT_BOTTOM_MAX_IQR", 14.0)
                    )
                    # For composite stepped facades the bottom profile varies
                    # strongly between portions (stepped volumes at different
                    # heights).  The IQR filter would incorrectly restrict the
                    # socle to the lowest-bottom columns only, so skip it.
                    if bottom_iqr > flat_bottom_iqr and facade.scene_type != "composite_stepped_facade":
                        bottom_cut = float(np.percentile(valid_bottoms, 70))
                        bottom_tol = max(8, int(band_px * 0.20))
                        active_cols = bottoms >= (bottom_cut - bottom_tol)
                        if np.count_nonzero(active_cols) >= int(roi_w * 0.18):
                            col_mask = np.zeros_like(socle_mask)
                            col_mask[:, active_cols] = 255
                            socle_mask = cv2.bitwise_and(socle_mask, col_mask)
                            socle_mask = cv2.morphologyEx(
                                socle_mask,
                                cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3)),
                            )

            socle_mask = self._bridge_socle_edge_gaps(
                socle_mask,
                facade_mask,
                float(getattr(config, "PHOTO_SOCLE_EDGE_BRIDGE_MAX_RATIO", 0.06)),
            )

            # Keep only socle components connected to facade bottom edge.
            # For composite stepped facades the band is near the local facade
            # bottom for each column, which may be well above the ROI bottom
            # (e.g. the right upper volume).  Use per-column proximity instead.
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(socle_mask, 8)
            if n_labels > 1:
                keep = np.zeros_like(socle_mask)
                min_comp_area = max(24, int(roi_w * 0.01))
                kept_any = False
                if facade.scene_type == "composite_stepped_facade":
                    # Build a per-column facade-bottom proximity mask covering
                    # the socle band window so that upper stepped volumes are
                    # included even when they don't reach the ROI bottom row.
                    bottom_proximity = np.zeros_like(socle_mask)
                    for cx in range(roi_w):
                        bot = int(bottoms[cx])
                        if bot < 0:
                            continue
                        r0 = max(0, bot - band_px)
                        r1 = min(roi_h - 1, bot)
                        if r0 <= r1:
                            bottom_proximity[r0:r1 + 1, cx] = 255
                    for idx in range(1, n_labels):
                        area = int(stats[idx, cv2.CC_STAT_AREA])
                        if area < min_comp_area:
                            continue
                        comp_mask = (labels == idx).astype(np.uint8) * 255
                        if np.any(cv2.bitwise_and(comp_mask, bottom_proximity) > 0):
                            keep[labels == idx] = 255
                            kept_any = True
                else:
                    bottom_idx = max(0, socle_mask.shape[0] - 1)
                    bottom_band = max(3, int(socle_mask.shape[0] * 0.03))
                    row0 = max(0, bottom_idx - bottom_band + 1)
                    for idx in range(1, n_labels):
                        area = int(stats[idx, cv2.CC_STAT_AREA])
                        if area < min_comp_area:
                            continue
                        if np.any(labels[row0:bottom_idx + 1, :] == idx):
                            keep[labels == idx] = 255
                            kept_any = True
                if kept_any:
                    socle_mask = keep

            socle_area_px = float(np.count_nonzero(socle_mask))
            if socle_area_px <= 0:
                continue
            socle_area_m2 = socle_area_px / self.area_scale_px_per_m2
            if socle_area_m2 <= 0:
                continue
            max_share = float(getattr(config, "PHOTO_SOCLE_MAX_AREA_SHARE", 0.50))
            if used_magenta_socle:
                max_share = min(max_share, 0.22)
            if socle_area_m2 > facade.total_area * max_share:
                continue

            facade_area_without_socle_px = None
            contours, _ = cv2.findContours(
                socle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                c = max(contours, key=cv2.contourArea)
                chosen_socle = c
                if facade.scene_type == "flat_long_facade":
                    socle_top_fit = self._fit_mask_edge_line(
                        socle_mask,
                        edge="top",
                        support_mask=facade_mask,
                        prefer_thick_columns=True,
                        min_cols_ratio=0.28,
                        max_slope=float(getattr(config, "PHOTO_SOCLE_GUIDE_MAX_SLOPE", 0.12)),
                    )
                    socle_bottom_fit = self._fit_mask_edge_line(
                        facade_mask,
                        edge="bottom",
                        support_mask=facade_mask,
                        prefer_thick_columns=False,
                        min_cols_ratio=0.28,
                        max_slope=float(getattr(config, "PHOTO_SOCLE_GUIDE_MAX_SLOPE", 0.12)),
                    )
                    if socle_top_fit and socle_bottom_fit:
                        left = max(0, min(int(socle_top_fit[0]), int(socle_bottom_fit[0])))
                        right = min(roi_w - 1, max(int(socle_top_fit[1]), int(socle_bottom_fit[1])))
                        linear_socle_contour = self._linear_band_contour_from_lines(
                            left,
                            right,
                            (socle_top_fit[2], socle_top_fit[3]),
                            (socle_bottom_fit[2], socle_bottom_fit[3]),
                            roi_h,
                        )
                        if linear_socle_contour is not None:
                            linear_area = float(cv2.contourArea(linear_socle_contour))
                            base_area = max(1.0, float(cv2.contourArea(c)))
                            if base_area * 0.78 <= linear_area <= base_area * 1.28:
                                chosen_socle = linear_socle_contour
                    smooth_window = max(9, int(round(roi_w * 0.018)))
                    if smooth_window % 2 == 0:
                        smooth_window += 1
                    profile_socle = self._profile_contour_from_mask(
                        socle_mask,
                        support_mask=facade_mask,
                        smooth_window=smooth_window,
                        prefer_thick_columns=True,
                    )
                    if profile_socle is not None:
                        prof_area = float(cv2.contourArea(profile_socle))
                        base_area = max(1.0, float(cv2.contourArea(chosen_socle)))
                        bx0, by0, bw0, bh0 = cv2.boundingRect(chosen_socle)
                        bx1, by1, bw1, bh1 = cv2.boundingRect(profile_socle)
                        if (
                            prof_area >= base_area * 0.70
                            and bh1 <= max(bh0 * 1.35, bh0 + 12)
                            and by1 <= by0 + max(10, int(bh0 * 0.20))
                        ):
                            chosen_socle = profile_socle
                elif facade.scene_type == "gable_facade":
                    smooth_window = max(7, int(round(roi_w * 0.014)))
                    if smooth_window % 2 == 0:
                        smooth_window += 1
                    profile_socle = self._profile_contour_from_mask(
                        socle_mask,
                        support_mask=facade_mask,
                        smooth_window=smooth_window,
                        prefer_thick_columns=True,
                    )
                    if profile_socle is not None:
                        prof_area = float(cv2.contourArea(profile_socle))
                        base_area = max(1.0, float(cv2.contourArea(chosen_socle)))
                        if base_area * 0.72 <= prof_area <= base_area * 1.22:
                            chosen_socle = profile_socle

                direct_geometry_applied = False
                direct_socle_profile_global = np.array([])
                if facade.scene_type == "gable_facade":
                    recovered = self._recover_gable_silhouette_from_seed(facade)
                    if recovered is not None:
                        recovered_contour_global, _, _ = recovered
                        recovered_local = recovered_contour_global.astype(np.int32).copy()
                        recovered_local[:, 0, 0] -= x0
                        recovered_local[:, 0, 1] -= y0
                        recovered_pts = recovered_local.reshape(-1, 2)
                        if recovered_pts.shape[0] >= 5:
                            left_roof = recovered_pts[0].astype(np.int32)
                            apex = recovered_pts[1].astype(np.int32)
                            right_roof = recovered_pts[2].astype(np.int32)
                            right_bottom = recovered_pts[3].astype(np.int32)
                            left_bottom = recovered_pts[4].astype(np.int32)
                            bottom_y = int(
                                np.clip(
                                    max(int(left_bottom[1]), int(right_bottom[1])),
                                    0,
                                    roi_h - 1,
                                )
                            )
                            left_x = int(np.clip(min(int(left_roof[0]), int(left_bottom[0])), 0, roi_w - 1))
                            right_x = int(np.clip(max(int(right_roof[0]), int(right_bottom[0])), 0, roi_w - 1))
                            if right_x - left_x >= max(80, int(roi_w * 0.42)):
                                band_ratio = float(
                                    getattr(config, "PHOTO_GABLE_SOCLE_BAND_RATIO", 0.08)
                                )
                                band_h = max(12, int(round(roi_h * band_ratio)))
                                socle_top_y = int(
                                    np.clip(
                                        bottom_y - band_h,
                                        0,
                                        max(0, bottom_y - 4),
                                    )
                                )
                                facade_local = np.array(
                                    [
                                        [left_x, int(np.clip(left_roof[1], 0, roi_h - 1))],
                                        [int(np.clip(apex[0], 0, roi_w - 1)), int(np.clip(apex[1], 0, roi_h - 1))],
                                        [right_x, int(np.clip(right_roof[1], 0, roi_h - 1))],
                                        [right_x, socle_top_y],
                                        [left_x, socle_top_y],
                                    ],
                                    dtype=np.int32,
                                ).reshape((-1, 1, 2))
                                span_w = max(1, right_x - left_x)
                                left_inset = max(1, int(round(span_w * 0.022)))
                                right_inset = max(1, int(round(span_w * 0.016)))
                                lip = max(1, int(round(span_w * 0.002)))
                                top_left_x = min(right_x, left_x + left_inset)
                                top_right_x = max(left_x, right_x - right_inset)
                                outer_bottom_y = int(max(socle_top_y + 12, bottom_y - 3))
                                inner_bottom_y = int(max(socle_top_y + 10, outer_bottom_y))
                                socle_local = np.array(
                                    [
                                        [top_right_x, socle_top_y],
                                        [max(top_left_x, top_right_x - lip), inner_bottom_y],
                                        [min(top_right_x, top_left_x + lip), inner_bottom_y],
                                        [top_left_x, min(roi_h - 1, socle_top_y + 1)],
                                        [top_left_x, outer_bottom_y],
                                        [top_right_x, outer_bottom_y],
                                    ],
                                    dtype=np.int32,
                                ).reshape((-1, 1, 2))
                                facade_area_px = float(cv2.contourArea(facade_local))
                                socle_area_px = float(cv2.contourArea(socle_local))
                                if facade_area_px > 0 and socle_area_px > 0:
                                    facade_global = facade_local.copy()
                                    facade_global[:, 0, 0] += x0
                                    facade_global[:, 0, 1] += y0
                                    socle_global = socle_local.copy()
                                    socle_global[:, 0, 0] += x0
                                    socle_global[:, 0, 1] += y0
                                    socle_profile_local = np.array(
                                        [
                                            [top_left_x, min(roi_h - 1, socle_top_y + 1)],
                                            [top_right_x, socle_top_y],
                                        ],
                                        dtype=np.int32,
                                    ).reshape((-1, 1, 2))
                                    socle_profile_global = socle_profile_local.copy()
                                    socle_profile_global[:, 0, 0] += x0
                                    socle_profile_global[:, 0, 1] += y0
                                    chosen_socle = socle_local

                                    facade.contour = facade_global
                                    fx, fy, fw, fh = cv2.boundingRect(facade_local)
                                    facade.region_bbox = (x0 + fx, y0 + fy, fw, fh)
                                    facade.socle_contour = socle_global
                                    sx, sy, sw, sh = cv2.boundingRect(socle_local)
                                    facade.socle_bbox = (x0 + sx, y0 + sy, sw, sh)
                                    facade.socle_profile = socle_profile_global
                                    socle_area_m2 = (
                                        socle_area_px / self.area_scale_px_per_m2
                                        if socle_area_px > 0
                                        else 0.0
                                    )
                                    direct_socle_profile_global = socle_profile_global
                                    facade_area_without_socle_px = facade_area_px
                                    direct_geometry_applied = True
                                    if os.environ.get("DQP_DEBUG_GEOM"):
                                        print(
                                            "DEBUG_GEOM_APPLIED_GABLE_POLY",
                                            facade.region_bbox,
                                            facade.socle_bbox,
                                        )

                if not direct_geometry_applied and facade.scene_type == "gable_facade":
                    socle_top_fit = self._fit_mask_edge_line(
                        socle_mask,
                        edge="top",
                        support_mask=facade_mask,
                        prefer_thick_columns=True,
                        min_cols_ratio=0.22,
                        max_slope=0.18,
                    )
                    socle_bottom_fit = self._fit_mask_edge_line(
                        facade_mask,
                        edge="bottom",
                        support_mask=facade_mask,
                        prefer_thick_columns=False,
                        min_cols_ratio=0.22,
                        max_slope=0.18,
                    )
                    if socle_top_fit and socle_bottom_fit:
                        support_cols = np.where(np.sum(facade_mask > 0, axis=0) > 0)[0]
                        if support_cols.size >= max(12, int(roi_w * 0.22)):
                            left = int(support_cols[0])
                            right = int(support_cols[-1])
                        else:
                            left = max(0, min(int(socle_top_fit[0]), int(socle_bottom_fit[0])))
                            right = min(roi_w - 1, max(int(socle_top_fit[1]), int(socle_bottom_fit[1])))
                        a_bottom, b_bottom = float(socle_bottom_fit[2]), float(socle_bottom_fit[3])
                        sample_x = np.array([left, max(left, (left + right) // 2), right], dtype=np.float32)
                        top_vals = np.array([float(socle_top_fit[2]) * x + float(socle_top_fit[3]) for x in sample_x], dtype=np.float32)
                        bottom_vals = np.array([a_bottom * x + b_bottom for x in sample_x], dtype=np.float32)
                        mean_band_h = float(np.mean(np.maximum(4.0, bottom_vals - top_vals)))
                        max_band_h = float(max(12, int(round(roi_h * 0.10))))
                        min_band_h = float(max(10, int(round(roi_h * 0.05))))
                        target_band_h = float(np.clip(mean_band_h, min_band_h, max_band_h))
                        linear_socle_contour = self._linear_band_contour_from_lines(
                            left,
                            right,
                            (a_bottom, b_bottom - target_band_h),
                            (a_bottom, b_bottom),
                            roi_h,
                        )
                        if linear_socle_contour is not None:
                            linear_area = float(cv2.contourArea(linear_socle_contour))
                            base_area = max(1.0, float(cv2.contourArea(chosen_socle)))
                            if base_area * 0.35 <= linear_area <= base_area * 1.65:
                                chosen_socle = linear_socle_contour
                c = chosen_socle
                c_global = c.copy()
                c_global[:, 0, 0] += x0
                c_global[:, 0, 1] += y0
                facade.socle_contour = c_global
                sx, sy, sw, sh = cv2.boundingRect(c)
                facade.socle_bbox = (x0 + sx, y0 + sy, sw, sh)
                facade.socle_profile = np.array([])

                facade_without_socle = cv2.bitwise_and(
                    facade_mask, cv2.bitwise_not(socle_mask)
                )
                if sw >= int(roi_w * 0.88):
                    socle_pts = c.reshape(-1, 2).astype(np.float32)
                    top_band_thr = np.percentile(socle_pts[:, 1], 35)
                    top_pts = socle_pts[socle_pts[:, 1] <= top_band_thr]
                    if top_pts.shape[0] >= 6:
                        top_band = top_pts[:, 1]
                        top_band_iqr = float(
                            np.percentile(top_band, 75) - np.percentile(top_band, 25)
                        )
                        if top_band_iqr <= 16.0:
                            if len(np.unique(top_pts[:, 0])) >= 2:
                                a, b = np.polyfit(top_pts[:, 0], top_pts[:, 1], 1)
                                a = float(np.clip(a, -0.08, 0.08))
                                for cx in range(roi_w):
                                    cut_y = int(round(a * cx + b))
                                    cut_y = max(0, min(roi_h - 1, cut_y))
                                    facade_without_socle[cut_y:, cx] = 0
                            else:
                                flat_socle_top = int(round(np.percentile(top_band, 15)))
                                facade_without_socle[flat_socle_top:, :] = 0
                if not direct_geometry_applied and facade.scene_type == "gable_facade":
                    support_cols = np.where(np.sum(facade_mask > 0, axis=0) > 0)[0]
                    socle_top_profile = np.array([], dtype=np.float32)
                    if support_cols.size >= max(12, int(roi_w * 0.22)):
                        bottoms = self._facade_bottom_profile(facade_mask).astype(np.float32)
                        target_band_h = float(
                            max(
                                12,
                                int(
                                    round(
                                        roi_h
                                        * (
                                            0.08
                                            if facade.scene_type == "gable_facade"
                                            else 0.07
                                        )
                                    )
                                ),
                            )
                        )
                        adjusted_socle_top = np.full(roi_w, -1.0, dtype=np.float32)
                        for cx in support_cols.tolist():
                            bottom = float(bottoms[int(cx)])
                            if bottom < 0:
                                continue
                            adjusted_socle_top[int(cx)] = bottom - target_band_h
                        socle_top_profile = self._interpolate_profile(adjusted_socle_top)

                    if (
                        support_cols.size >= max(12, int(roi_w * 0.22))
                        and socle_top_profile.size != 0
                    ):
                        left = int(support_cols[0])
                        right = int(support_cols[-1])
                        final_socle_mask = self._build_socle_band_from_bottoms(
                            facade_mask,
                            socle_top_profile,
                        )
                        final_socle_mask = self._clip_mask_to_x_span(
                            final_socle_mask,
                            left,
                            right,
                        )
                        final_socle_mask = self._clip_band_mask_to_support_columns(
                            final_socle_mask,
                            facade_mask,
                        )
                        final_socle_mask = self._bridge_socle_edge_gaps(
                            final_socle_mask,
                            facade_mask,
                            max_gap_ratio=0.10,
                        )

                        final_facade_mask = self._cut_mask_below_profile(
                            facade_mask.copy(),
                            socle_top_profile,
                            pad_px=0,
                        )
                        final_facade_mask = self._clip_mask_to_x_span(
                            final_facade_mask,
                            left,
                            right,
                        )
                        final_facade_mask = cv2.morphologyEx(
                            final_facade_mask,
                            cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                        )
                        final_facade_mask = cv2.morphologyEx(
                            final_facade_mask,
                            cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
                        )

                        smooth_window = max(
                            7,
                            int(
                                round(
                                    roi_w
                                    * (
                                        0.012
                                        if facade.scene_type == "gable_facade"
                                        else 0.015
                                    )
                                )
                            ),
                        )
                        if smooth_window % 2 == 0:
                            smooth_window += 1

                        profile_facade = self._profile_contour_from_mask(
                            final_facade_mask,
                            support_mask=final_facade_mask,
                            smooth_window=smooth_window,
                            prefer_thick_columns=False,
                        )
                        if os.environ.get("DQP_DEBUG_GEOM"):
                            print(
                                "DEBUG_GEOM",
                                facade.scene_type,
                                "cut", cut,
                                "support_cols", int(support_cols.size),
                                "facade_nonzero", int(np.count_nonzero(final_facade_mask)),
                                "socle_nonzero", int(np.count_nonzero(final_socle_mask)),
                                "profile_facade", profile_facade is not None,
                            )
                        if profile_facade is not None:
                            prof_area = float(cv2.contourArea(profile_facade))
                            base_area = max(1.0, float(cv2.contourArea(f_local)))
                            if os.environ.get("DQP_DEBUG_GEOM"):
                                print("DEBUG_GEOM_AREA", facade.scene_type, "prof", round(prof_area,1), "base", round(base_area,1))
                            min_ratio = 0.35 if facade.scene_type == "gable_facade" else 0.45
                            max_ratio = 1.45 if facade.scene_type == "gable_facade" else 1.25
                            if base_area * min_ratio <= prof_area <= base_area * max_ratio:
                                profile_socle = self._profile_contour_from_mask(
                                    final_socle_mask,
                                    support_mask=final_socle_mask,
                                    smooth_window=max(5, smooth_window - 2),
                                    prefer_thick_columns=True,
                                )
                                if profile_socle is None:
                                    final_socle_contours, _ = cv2.findContours(
                                        final_socle_mask,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE,
                                    )
                                    profile_socle = (
                                        max(final_socle_contours, key=cv2.contourArea)
                                        if final_socle_contours
                                        else None
                                    )

                                if os.environ.get("DQP_DEBUG_GEOM"):
                                    print("DEBUG_GEOM_SOCLE", facade.scene_type, "profile_socle", profile_socle is not None)
                                if profile_socle is not None:
                                    c = profile_socle
                                    draw_contour = profile_facade
                                    clipped_area = prof_area
                                    clipped_global = draw_contour.copy()
                                    clipped_global[:, 0, 0] += x0
                                    clipped_global[:, 0, 1] += y0
                                    facade.contour = clipped_global
                                    cx, cy, cw, ch = cv2.boundingRect(draw_contour)
                                    facade.region_bbox = (x0 + cx, y0 + cy, cw, ch)

                                    c_global = c.copy()
                                    c_global[:, 0, 0] += x0
                                    c_global[:, 0, 1] += y0
                                    facade.socle_contour = c_global
                                    sx, sy, sw, sh = cv2.boundingRect(c)
                                    facade.socle_bbox = (x0 + sx, y0 + sy, sw, sh)

                                    socle_profile_local = self._derive_socle_profile_line_local(
                                        c,
                                        draw_contour,
                                        roi_h,
                                    )
                                    if (
                                        isinstance(socle_profile_local, np.ndarray)
                                        and socle_profile_local.size >= 4
                                    ):
                                        socle_profile_global = socle_profile_local.copy()
                                        socle_profile_global[:, 0, 0] += x0
                                        socle_profile_global[:, 0, 1] += y0
                                        facade.socle_profile = socle_profile_global
                                    else:
                                        facade.socle_profile = np.array([])

                                    socle_area_px = float(np.count_nonzero(final_socle_mask))
                                    socle_area_m2 = (
                                        socle_area_px / self.area_scale_px_per_m2
                                        if socle_area_px > 0
                                        else 0.0
                                    )
                                    facade_area_without_socle_px = clipped_area
                                    direct_geometry_applied = True
                                    if os.environ.get("DQP_DEBUG_GEOM"):
                                        print("DEBUG_GEOM_APPLIED", facade.scene_type, facade.region_bbox, facade.socle_bbox)

                if direct_geometry_applied:
                    facade.socle_excluded_area = round(float(socle_area_m2), 3)
                    if isinstance(direct_socle_profile_global, np.ndarray) and direct_socle_profile_global.size >= 4:
                        facade.socle_profile = direct_socle_profile_global
                    if facade_area_without_socle_px and facade_area_without_socle_px > 0:
                        facade.area_px = float(facade_area_without_socle_px)
                        facade.total_area = round(
                            float(facade_area_without_socle_px / self.area_scale_px_per_m2), 3
                        )
                    else:
                        facade.total_area = round(
                            float(max(0.0, facade.total_area - socle_area_m2)), 3
                        )
                    continue

                if facade.scene_type == "flat_long_facade":
                    facade_without_socle = self._trim_flat_photo_facade_body_support(
                        facade_without_socle,
                        facade,
                        roi,
                        allow_side_trim=False,
                    )

                facade_without_socle = cv2.morphologyEx(
                    facade_without_socle,
                    cv2.MORPH_OPEN,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                )
                facade_without_socle = cv2.morphologyEx(
                    facade_without_socle,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                )
                clipped_contours, _ = cv2.findContours(
                    facade_without_socle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if clipped_contours:
                    clipped = max(clipped_contours, key=cv2.contourArea)
                    clipped_area = float(cv2.contourArea(clipped))
                    min_keep_area = max(180.0, float(np.count_nonzero(facade_mask)) * 0.35)
                    if clipped_area >= min_keep_area:
                        clipped_mask_single = np.zeros_like(facade_without_socle)
                        cv2.drawContours(clipped_mask_single, [clipped], -1, 255, -1)
                        draw_contour = clipped
                        final_socle_mask = socle_mask.copy()
                        if facade.scene_type == "flat_long_facade":
                            linear_locked = False
                            facade_top_fit = self._fit_mask_edge_line(
                                facade_without_socle,
                                edge="top",
                                support_mask=facade_without_socle,
                                prefer_thick_columns=False,
                                min_cols_ratio=0.30,
                                max_slope=0.08,
                            )
                            facade_bottom_fit = self._fit_mask_edge_line(
                                socle_mask,
                                edge="top",
                                support_mask=facade_mask,
                                prefer_thick_columns=True,
                                min_cols_ratio=0.30,
                                max_slope=float(getattr(config, "PHOTO_SOCLE_GUIDE_MAX_SLOPE", 0.12)),
                            )
                            if facade_top_fit and facade_bottom_fit:
                                left = max(0, min(int(facade_top_fit[0]), int(facade_bottom_fit[0])))
                                right = min(roi_w - 1, max(int(facade_top_fit[1]), int(facade_bottom_fit[1])))
                                linear_facade_contour = self._linear_band_contour_from_lines(
                                    left,
                                    right,
                                    (facade_top_fit[2], facade_top_fit[3]),
                                    (facade_bottom_fit[2], facade_bottom_fit[3]),
                                    roi_h,
                                )
                                if linear_facade_contour is not None:
                                    linear_area = float(cv2.contourArea(linear_facade_contour))
                                    linear_min_ratio = 0.55 if "capped" in src else 0.80
                                    if clipped_area * linear_min_ratio <= linear_area <= clipped_area * 1.18:
                                        draw_contour = linear_facade_contour
                                        linear_locked = "capped" in src
                            smooth_window = max(9, int(round(roi_w * 0.016)))
                            if smooth_window % 2 == 0:
                                smooth_window += 1
                            profile_facade = self._profile_contour_from_mask(
                                clipped_mask_single,
                                support_mask=clipped_mask_single,
                                smooth_window=smooth_window,
                                prefer_thick_columns=False,
                            )
                            if profile_facade is not None:
                                prof_area = float(cv2.contourArea(profile_facade))
                                if prof_area >= clipped_area * 0.82 and not linear_locked:
                                    draw_contour = profile_facade

                            final_facade_mask = np.zeros_like(facade_without_socle)
                            cv2.drawContours(final_facade_mask, [draw_contour], -1, 255, -1)
                            socle_profile_local = self._derive_socle_profile_line_local(
                                c,
                                draw_contour,
                                roi_h,
                            )
                            span_left, span_right = self._line_span(socle_profile_local)
                            final_facade_mask = self._clip_mask_to_x_span(
                                final_facade_mask,
                                span_left,
                                span_right,
                            )
                            final_socle_mask = self._clip_band_mask_to_support_columns(
                                socle_mask,
                                final_facade_mask,
                            )
                            final_socle_mask = self._clip_mask_to_x_span(
                                final_socle_mask,
                                span_left,
                                span_right,
                            )
                            socle_top_profile = self._profile_from_line_contour(
                                socle_profile_local,
                                roi_w,
                            )
                            if socle_top_profile.size == 0:
                                socle_top_profile = self._mask_top_profile(final_socle_mask)
                                socle_top_profile = self._interpolate_profile(socle_top_profile)
                            final_facade_mask = self._cut_mask_below_profile(
                                final_facade_mask,
                                socle_top_profile,
                                pad_px=0,
                            )
                            final_facade_mask = self._trim_flat_photo_facade_body_support(
                                final_facade_mask,
                                facade,
                                roi,
                                allow_side_trim=False,
                            )
                            final_facade_mask = self._clip_mask_to_x_span(
                                final_facade_mask,
                                span_left,
                                span_right,
                            )
                            reclipped_contours, _ = cv2.findContours(
                                final_facade_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                            )
                            if reclipped_contours:
                                reclipped = max(reclipped_contours, key=cv2.contourArea)
                                reclipped_area = float(cv2.contourArea(reclipped))
                                if reclipped_area >= max(clipped_area * 0.60, min_keep_area * 0.55):
                                    draw_contour = reclipped
                                    clipped_area = reclipped_area
                        elif facade.scene_type == "gable_facade":
                            smooth_window = max(7, int(round(roi_w * 0.014)))
                            if smooth_window % 2 == 0:
                                smooth_window += 1
                            profile_facade = self._profile_contour_from_mask(
                                clipped_mask_single,
                                support_mask=clipped_mask_single,
                                smooth_window=smooth_window,
                                prefer_thick_columns=False,
                            )
                            if profile_facade is not None:
                                prof_area = float(cv2.contourArea(profile_facade))
                                if clipped_area * 0.78 <= prof_area <= clipped_area * 1.16:
                                    draw_contour = profile_facade
                        clipped_global = draw_contour.copy()
                        clipped_global[:, 0, 0] += x0
                        clipped_global[:, 0, 1] += y0
                        facade.contour = clipped_global
                        cx, cy, cw, ch = cv2.boundingRect(draw_contour)
                        facade.region_bbox = (x0 + cx, y0 + cy, cw, ch)

                        if facade.scene_type == "flat_long_facade":
                            final_facade_mask = np.zeros_like(facade_without_socle)
                            cv2.drawContours(final_facade_mask, [draw_contour], -1, 255, -1)
                            final_socle_mask = self._clip_band_mask_to_support_columns(
                                socle_mask,
                                final_facade_mask,
                            )
                            span_left, span_right = self._line_span(facade.socle_profile)
                            final_socle_mask = self._clip_mask_to_x_span(
                                final_socle_mask,
                                span_left,
                                span_right,
                            )
                            final_socle_contours, _ = cv2.findContours(
                                final_socle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                            )
                            if final_socle_contours:
                                c = max(final_socle_contours, key=cv2.contourArea)
                                c_global = c.copy()
                                c_global[:, 0, 0] += x0
                                c_global[:, 0, 1] += y0
                                facade.socle_contour = c_global
                                sx, sy, sw, sh = cv2.boundingRect(c)
                                facade.socle_bbox = (x0 + sx, y0 + sy, sw, sh)
                                socle_profile_local = self._derive_socle_profile_line_local(
                                    c,
                                    draw_contour,
                                    roi_h,
                                )
                                if isinstance(socle_profile_local, np.ndarray) and socle_profile_local.size >= 4:
                                    socle_profile_global = socle_profile_local.copy()
                                    socle_profile_global[:, 0, 0] += x0
                                    socle_profile_global[:, 0, 1] += y0
                                    facade.socle_profile = socle_profile_global
                                socle_area_px = float(np.count_nonzero(final_socle_mask))
                                socle_area_m2 = socle_area_px / self.area_scale_px_per_m2 if socle_area_px > 0 else 0.0

                        facade_area_without_socle_px = clipped_area

            facade.socle_excluded_area = round(float(socle_area_m2), 3)
            if facade_area_without_socle_px and facade_area_without_socle_px > 0:
                facade.area_px = float(facade_area_without_socle_px)
                facade.total_area = round(
                    float(facade_area_without_socle_px / self.area_scale_px_per_m2), 3
                )
            else:
                facade.total_area = round(
                    float(max(0.0, facade.total_area - socle_area_m2)), 3
                )

            if facade.scene_type == "flat_long_facade":
                self._regularize_flat_facade_against_socle(facade)
            elif facade.scene_type == "composite_stepped_facade":
                self._regularize_composite_facade_bottom_against_socle(facade)
            elif facade.scene_type == "sparse_openings_flat":
                self._regularize_sparse_facade_from_seed_hull(facade)

    def _clip_elements_to_socle_top(self):
        """Clip flat-facade doors at socle top/profile when they extend too low."""
        if self.current_image is None:
            return
        for facade in self.parsed_facades:
            if facade.scene_type != "flat_long_facade":
                continue
            if not facade.socle_bbox or not isinstance(facade.socle_contour, np.ndarray) or facade.socle_contour.size < 6:
                continue

            fx, fy, fw, fh = facade.region_bbox if facade.region_bbox else facade.socle_bbox
            sx, sy, sw, sh = facade.socle_bbox
            x0 = max(0, min(fx, sx))
            y0 = max(0, min(fy, sy))
            x1 = min(self.current_image.shape[1], max(fx + fw, sx + sw))
            y1 = min(self.current_image.shape[0], max(fy + fh, sy + sh))
            if x1 - x0 < 20 or y1 - y0 < 20:
                continue

            roi_h = y1 - y0
            roi_w = x1 - x0
            socle_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            s_local = facade.socle_contour.astype(np.int32).copy()
            s_local[:, 0, 0] -= x0
            s_local[:, 0, 1] -= y0
            cv2.drawContours(socle_mask, [s_local], -1, 255, -1)
            top_profile = np.array([], dtype=np.float32)
            if isinstance(facade.socle_profile, np.ndarray) and facade.socle_profile.size >= 4:
                profile_local = facade.socle_profile.astype(np.int32).copy()
                profile_local[:, 0, 0] -= x0
                profile_local[:, 0, 1] -= y0
                top_profile = self._profile_from_line_contour(profile_local, roi_w)
            if top_profile.size == 0:
                top_profile = self._interpolate_profile(self._mask_top_profile(socle_mask))
            if top_profile.size == 0:
                continue

            for elem in self.parsed_elements:
                if elem.parent_facade != facade.name or elem.element_type != "door" or not elem.bbox:
                    continue
                ex, ey, ew, eh = elem.bbox
                cx = int(round(ex + ew * 0.5)) - x0
                if cx < 0 or cx >= roi_w:
                    continue
                cut_y = float(top_profile[cx])
                if cut_y < 0:
                    continue
                cut_y_global = int(round(cut_y)) + y0
                bottom = ey + eh
                if bottom <= cut_y_global + 1:
                    continue
                new_h = cut_y_global - ey
                if new_h < int(round(eh * 0.60)):
                    continue
                elem.bbox = (ex, ey, ew, int(new_h))
                elem.area_px = float(ew * int(new_h))
                if self.linear_scale_px_per_m and self.linear_scale_px_per_m > 0:
                    elem.width = round(float(ew / self.linear_scale_px_per_m), 3)
                    elem.height = round(float(int(new_h) / self.linear_scale_px_per_m), 3)
                if self.area_scale_px_per_m2 and self.area_scale_px_per_m2 > 0:
                    elem.area = round(float(elem.area_px / self.area_scale_px_per_m2), 3)

    def _clip_elements_to_facade_bottom(self):
        """Clip openings that extend below the detected facade bottom for composite facades."""
        if self.current_image is None:
            return
        for facade in self.parsed_facades:
            if facade.scene_type != "composite_stepped_facade":
                continue
            if not facade.region_bbox:
                continue
            _fx, fy, _fw, fh = facade.region_bbox
            facade_bottom = fy + fh
            for elem in self.parsed_elements:
                if elem.parent_facade != facade.name or not elem.bbox:
                    continue
                if elem.element_type not in ("door", "window"):
                    continue
                ex, ey, ew, eh = elem.bbox
                bottom = ey + eh
                if bottom <= facade_bottom:
                    continue
                new_h = facade_bottom - ey
                if new_h < int(round(eh * 0.30)):
                    continue
                if new_h < 40:
                    continue
                elem.bbox = (ex, ey, ew, int(new_h))
                elem.area_px = float(ew * int(new_h))
                if self.linear_scale_px_per_m and self.linear_scale_px_per_m > 0:
                    elem.width = round(float(ew / self.linear_scale_px_per_m), 3)
                    elem.height = round(float(int(new_h) / self.linear_scale_px_per_m), 3)
                if self.area_scale_px_per_m2 and self.area_scale_px_per_m2 > 0:
                    elem.area = round(float(elem.area_px / self.area_scale_px_per_m2), 3)

    def _build_report(self) -> ProjectReport:
        """Build the final ProjectReport from parsed data."""
        report = ProjectReport()

        if not self.parsed_facades and self.parsed_elements:
            self.parsed_facades.append(ParsedFacade(
                name="Fatada detectata",
                position=self.parsed_elements[0].position,
                total_area=0.0,
                net_area=0.0,
            ))
            self._assign_parents()

        for pf in self.parsed_facades:
            fr = FacadeReport(
                name=pf.name,
                total_area_m2=pf.total_area,
                socle_excluded_area_m2=pf.socle_excluded_area,
            )
            if self.linear_scale_px_per_m and self.linear_scale_px_per_m > 0:
                if pf.region_bbox:
                    _, _, fw, fh = pf.region_bbox
                    corner = 2.0 * (fh / self.linear_scale_px_per_m)
                    if 0.2 <= corner <= 200:
                        fr.corner_length_m = round(float(corner), 3)
                    socle_w_px = fw
                    if pf.socle_bbox:
                        _, _, sw, _ = pf.socle_bbox
                        socle_w_px = sw
                    socle_len = float(socle_w_px / self.linear_scale_px_per_m)
                    if 0.2 <= socle_len <= 500:
                        fr.socle_drip_profile_length_m = round(socle_len, 3)

            for elem in self.parsed_elements:
                if elem.parent_facade != pf.name:
                    continue
                width_m = elem.width if elem.width > 0 else 0.0
                if width_m <= 0 and self.linear_scale_px_per_m and elem.bbox:
                    _, _, bw, _ = elem.bbox
                    width_m = float(bw / self.linear_scale_px_per_m)
                min_w, max_w, _, _ = self._dimension_bounds_for_type(
                    elem.element_type
                )
                if not (min_w <= width_m <= max_w):
                    width_m = 0.0
                height_m = elem.height if elem.height > 0 else 0.0
                if height_m <= 0 and self.linear_scale_px_per_m and elem.bbox:
                    _, _, _, bh = elem.bbox
                    height_m = float(bh / self.linear_scale_px_per_m)

                if elem.element_type == "window":
                    fr.windows.append((elem.label, elem.area))
                    if width_m > 0:
                        fr.sill_length_m += width_m
                        fr.drip_profile_length_m += width_m
                    if width_m > 0 and height_m > 0:
                        fr.window_perimeter_length_m += 2.0 * (width_m + height_m)
                elif elem.element_type == "door":
                    fr.doors.append((elem.label, elem.area))
                    if width_m > 0:
                        fr.drip_profile_length_m += width_m
                    if width_m > 0 and height_m > 0:
                        fr.door_perimeter_length_m += (2.0 * height_m + width_m)

            fr.sill_length_m = round(float(fr.sill_length_m), 3)
            fr.drip_profile_length_m = round(float(fr.drip_profile_length_m), 3)
            fr.socle_drip_profile_length_m = round(
                float(fr.socle_drip_profile_length_m), 3
            )
            fr.window_perimeter_length_m = round(
                float(fr.window_perimeter_length_m), 3
            )
            fr.door_perimeter_length_m = round(
                float(fr.door_perimeter_length_m), 3
            )

            report.facades.append(fr)

        self.calculator.report = report
        return report

    def _build_detection_results(self):
        """Build detection_results dict for visualization."""
        self.detection_results = {
            "facades": [],
            "windows": [],
            "doors": [],
            "socles": [],
            "socle_profiles": [],
            "window_perimeters": [],
            "door_perimeters": [],
        }

        for pf in self.parsed_facades:
            if pf.region_bbox:
                x, y, w, h = pf.region_bbox
            else:
                x, y = pf.position
                w, h = 200, 100

            facade_height_m = None
            facade_width_m = None
            if self.linear_scale_px_per_m and self.linear_scale_px_per_m > 0:
                facade_width_m = float(w / self.linear_scale_px_per_m)
                facade_height_m = float(h / self.linear_scale_px_per_m)

            self.detection_results["facades"].append(DetectedRegion(
                label=pf.name,
                region_type="facade",
                bbox=(x, y, w, h),
                contour=pf.contour if isinstance(pf.contour, np.ndarray) else np.array([]),
                area_m2=pf.total_area,
                width_m=facade_width_m,
                height_m=facade_height_m,
            ))
            if pf.socle_excluded_area > 0:
                if pf.socle_bbox:
                    sx, sy, sw, sh = pf.socle_bbox
                else:
                    # Fallback: thin lower band inside facade bbox.
                    sh = max(1, int(h * 0.10))
                    sx, sy, sw = x, y + h - sh, w
                self.detection_results["socles"].append(DetectedRegion(
                    label=f"SOCLE_{pf.name}",
                    region_type="socle",
                    bbox=(sx, sy, sw, sh),
                    contour=(
                        pf.socle_contour
                        if isinstance(pf.socle_contour, np.ndarray)
                        else np.array([])
                    ),
                    area_m2=pf.socle_excluded_area,
                    parent_facade=pf.name,
                    color_detected="auto-socle",
                ))
                if self.linear_scale_px_per_m and self.linear_scale_px_per_m > 0:
                    line_pts = None
                    if isinstance(pf.socle_profile, np.ndarray) and pf.socle_profile.size >= 4:
                        line_pts = pf.socle_profile.astype(np.int32).copy()
                    else:
                        x_a, y_a = sx, sy
                        x_b, y_b = sx + sw, sy

                        if isinstance(pf.socle_contour, np.ndarray) and pf.socle_contour.size >= 6:
                            pts = pf.socle_contour.reshape(-1, 2).astype(np.float32)
                            y_thr = np.percentile(pts[:, 1], 35)
                            top_pts = pts[pts[:, 1] <= y_thr]
                            if top_pts.shape[0] >= 6 and len(np.unique(top_pts[:, 0])) >= 2:
                                a, b = np.polyfit(top_pts[:, 0], top_pts[:, 1], 1)
                                x_a = sx
                                x_b = sx + sw
                                y_a = int(round(a * x_a + b))
                                y_b = int(round(a * x_b + b))

                        if abs(int(y_b) - int(y_a)) <= 1 and isinstance(pf.contour, np.ndarray) and pf.contour.size >= 6:
                            f_pts = pf.contour.reshape(-1, 2).astype(np.float32)
                            fy_thr = np.percentile(f_pts[:, 1], 78)
                            bot_pts = f_pts[f_pts[:, 1] >= fy_thr]
                            if bot_pts.shape[0] >= 8 and len(np.unique(bot_pts[:, 0])) >= 2:
                                fa, fb = np.polyfit(bot_pts[:, 0], bot_pts[:, 1], 1)
                                y_a = int(round(fa * x_a + fb))
                                y_b = int(round(fa * x_b + fb))

                        max_dy = max(4, int(h * 0.08))
                        dy = int(y_b - y_a)
                        if abs(dy) > max_dy:
                            mid = int(round((y_a + y_b) * 0.5))
                            half = int(round(max_dy * 0.5))
                            y_a = mid - half
                            y_b = mid + half

                        line_pts = np.array(
                            [[[int(x_a), int(y_a)]], [[int(x_b), int(y_b)]]], dtype=np.int32
                        )

                    x_a, y_a = line_pts[0, 0]
                    x_b, y_b = line_pts[-1, 0]
                    line_len_px = float(((x_b - x_a) ** 2 + (y_b - y_a) ** 2) ** 0.5)
                    self.detection_results["socle_profiles"].append(DetectedRegion(
                        label=f"PIC_SOCLE_{pf.name}",
                        region_type="socle_profile",
                        bbox=(int(min(x_a, x_b)), int(min(y_a, y_b)), int(abs(x_b - x_a)), max(1, int(abs(y_b - y_a)))),
                        contour=line_pts,
                        length_m=round(float(line_len_px / self.linear_scale_px_per_m), 3),
                        parent_facade=pf.name,
                        color_detected="auto-socle-profile",
                        is_open_path=True,
                    ))

        for elem in self.parsed_elements:
            if elem.bbox:
                x, y, w, h = elem.bbox
            else:
                x, y = elem.position
                w, h = 50, 30
            width_m = elem.width if elem.width > 0 else None
            height_m = elem.height if elem.height > 0 else None
            if (width_m is None or height_m is None) and self.linear_scale_px_per_m:
                width_m = float(w / self.linear_scale_px_per_m)
                height_m = float(h / self.linear_scale_px_per_m)
            region = DetectedRegion(
                label=elem.label,
                region_type=elem.element_type,
                bbox=(x, y, w, h),
                contour=(
                    elem.contour if isinstance(elem.contour, np.ndarray)
                    else np.array([])
                ),
                area_m2=elem.area,
                width_m=width_m,
                height_m=height_m,
                parent_facade=elem.parent_facade,
            )
            key = "windows" if elem.element_type == "window" else "doors"
            self.detection_results[key].append(region)
            if width_m and height_m:
                if elem.element_type == "window":
                    contour = np.array(
                        [[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]],
                        dtype=np.int32,
                    )
                    self.detection_results["window_perimeters"].append(DetectedRegion(
                        label=f"PERIM_{elem.label}",
                        region_type="window_perimeter",
                        bbox=(x, y, w, h),
                        contour=contour,
                        length_m=round(float(2.0 * (width_m + height_m)), 3),
                        parent_facade=elem.parent_facade,
                        color_detected="auto-window-perimeter",
                    ))
                elif elem.element_type == "door":
                    contour = np.array(
                        [[[x, y + h]], [[x, y]], [[x + w, y]], [[x + w, y + h]]],
                        dtype=np.int32,
                    )
                    self.detection_results["door_perimeters"].append(DetectedRegion(
                        label=f"PERIM_{elem.label}",
                        region_type="door_perimeter",
                        bbox=(x, y, w, h),
                        contour=contour,
                        length_m=round(float(2.0 * height_m + width_m), 3),
                        parent_facade=elem.parent_facade,
                        color_detected="auto-door-perimeter",
                        is_open_path=True,
                    ))

    def get_summary(self) -> str:
        """Get text summary of the report."""
        if self.report:
            self.calculator.report = self.report
            summary = self.calculator.summary_text()
            if self.warnings:
                header = ["ATENTIONARI CALITATE INPUT", "-" * 40]
                for w in self.warnings:
                    header.append(f"- {w}")
                return "\n".join(header) + "\n\n" + summary
            return summary
        return "Nu exista date de raport."

    def get_ocr_debug(self) -> str:
        """Get debug info about what OCR found."""
        lines = ["OCR RESULTS DEBUG", "=" * 50]

        lines.append(f"\nTotal text blocks: {len(self.ocr_results)}")
        for r in self.ocr_results:
            marker = ""
            if r.parsed_value is not None:
                marker = f" >>> {r.parsed_value} {r.unit}"
            lines.append(
                f"  [{r.confidence:.2f}] '{r.text}' at {r.bbox}{marker}"
            )

        lines.append(f"\nParsed facades: {len(self.parsed_facades)}")
        for f in self.parsed_facades:
            socle_bbox_txt = f" socle_bbox={f.socle_bbox}" if f.socle_bbox else ""
            lines.append(
                f"  {f.name} at {f.position} "
                f"total={f.total_area}m² net={f.net_area}m² "
                f"area_px={f.area_px:.1f} socle={f.socle_excluded_area:.3f}m² "
                f"source={f.source}{socle_bbox_txt}"
            )

        if self.area_scale_px_per_m2:
            lines.append(
                f"\nCalibrare automata arie: {self.area_scale_px_per_m2:.2f} px/m²"
            )
        if self.linear_scale_px_per_m:
            src = self.linear_scale_source or "auto"
            lines.append(
                "Calibrare automata liniara: "
                f"{self.linear_scale_px_per_m:.2f} px/m (sursa: {src})"
            )
        if self.scale_ratio_used:
            ratio_src = self.scale_ratio_source or "necunoscut"
            lines.append(
                f"Scara folosita: 1:{self.scale_ratio_used:.0f} (sursa: {ratio_src})"
            )
        if self.source_dpi:
            lines.append(f"DPI sursa imagine: {self.source_dpi:.2f}")

        if self.signal_metrics:
            lines.append("\nSignal metrics:")
            for k, v in self.signal_metrics.items():
                lines.append(f"  - {k}: {v}")

        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")

        lines.append(f"\nParsed elements: {len(self.parsed_elements)}")
        for e in self.parsed_elements:
            lines.append(
                f"  [{e.element_type}] {e.label} at {e.position} "
                f"area={e.area}m² area_px={e.area_px:.1f} "
                f"w={e.width:.2f}m h={e.height:.2f}m "
                f"source={e.source} parent={e.parent_facade}"
            )

        return "\n".join(lines)














