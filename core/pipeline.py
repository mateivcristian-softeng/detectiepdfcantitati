"""
DrawQuantPDF - Analysis Pipeline
Orchestrates OCR-first approach with color verification:
1. OCR extracts all text and positions
2. Parser finds FATADA labels, area values, F/U element labels
3. Color detection finds yellow window rectangles
4. Spatial logic groups everything together
5. Calculator produces final report
"""

import re
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from core.color_detector import ColorDetector, DetectedRegion
from core.ocr_engine import OCREngine
from core.area_calculator import AreaCalculator, FacadeReport, ProjectReport


@dataclass
class ParsedFacade:
    name: str
    position: tuple  # (x, y) of the label
    total_area: float = 0.0
    net_area: float = 0.0
    region_bbox: Optional[tuple] = None  # estimated bounding box of this facade view


@dataclass
class ParsedElement:
    label: str
    element_type: str  # "window" or "door"
    position: tuple  # (x, y) center
    area: float = 0.0
    width: float = 0.0
    height: float = 0.0
    parent_facade: Optional[str] = None


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

    def run(self, image: np.ndarray,
            progress_callback=None) -> ProjectReport:
        """Run the full analysis pipeline."""

        if progress_callback:
            progress_callback("Pas 1/4: Extragere text (OCR)...", 0.1)
        self.ocr_results = self.ocr_engine.extract_all_text(image)

        if progress_callback:
            progress_callback("Pas 2/4: Parsare date din text...", 0.4)
        self.parsed_facades = self._parse_facades_from_ocr()
        self.parsed_elements = self._parse_elements_from_ocr()

        if progress_callback:
            progress_callback("Pas 3/4: Detectie ferestre (culoare)...", 0.6)
        color_windows = self.color_detector.detect_windows(image)
        color_doors = self.color_detector.detect_doors_by_color(image)

        if progress_callback:
            progress_callback("Pas 4/4: Combinare si calcul...", 0.8)

        self._estimate_facade_regions(image.shape)
        self._merge_color_with_ocr(color_windows, "window")
        self._merge_color_with_ocr(color_doors, "door")
        self._assign_parents()

        self.report = self._build_report()

        self._build_detection_results()

        if progress_callback:
            progress_callback("Analiza completa.", 1.0)

        return self.report

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
                nearby = self._find_nearby_numbers(r.bbox, max_distance=120)
                area = nearby[0] if nearby else 0.0

                elements.append(ParsedElement(
                    label=r.text.strip(),
                    element_type="window",
                    position=(x_c, y_c),
                    area=area,
                ))
                continue

            # Door patterns: U2, U2-1, U 2, etc.
            d_match = re.match(r"^U\s*(\d+)(?:\s*[-_]\s*(\d+))?$", text_upper)
            if d_match:
                x_c = (r.bbox[0] + r.bbox[2]) // 2
                y_c = (r.bbox[1] + r.bbox[3]) // 2
                nearby = self._find_nearby_numbers(
                    r.bbox, max_distance=80, max_value=20
                )
                area = nearby[0] if nearby else 0.0

                elements.append(ParsedElement(
                    label=r.text.strip(),
                    element_type="door",
                    position=(x_c, y_c),
                    area=area,
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

    def _estimate_facade_regions(self, image_shape: tuple):
        """Estimate the bounding box of each facade view based on label positions."""
        h, w = image_shape[:2]

        if not self.parsed_facades:
            return

        positions = [(f.position[0], f.position[1]) for f in self.parsed_facades]
        positions.sort(key=lambda p: (p[1], p[0]))

        for facade in self.parsed_facades:
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
                    # Same row
                    if ox > fx:
                        right = min(right, (fx + ox) // 2)
                    else:
                        left = max(left, (fx + ox) // 2)
                else:
                    if oy > fy:
                        bottom = min(bottom, (fy + oy) // 2)
                    elif oy < fy:
                        top = max(top, (fy + oy) // 2)

            facade.region_bbox = (left, top, right - left, bottom - top)

    def _merge_color_with_ocr(self, color_regions: list, element_type: str):
        """Cross-reference color-detected regions with OCR-parsed elements."""
        ocr_elements = [e for e in self.parsed_elements
                        if e.element_type == element_type]

        for cr in color_regions:
            cx, cy = cr.center
            best_match = None
            best_dist = float("inf")

            for elem in ocr_elements:
                ex, ey = elem.position
                dist = ((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5
                if dist < best_dist and dist < 150:
                    best_dist = dist
                    best_match = elem

            if best_match:
                if best_match.area > 0 and cr.area_m2 is None:
                    cr.area_m2 = best_match.area
                cr.label = best_match.label
            else:
                # Color-detected region without OCR match: keep as new element
                self.parsed_elements.append(ParsedElement(
                    label=cr.label,
                    element_type=element_type,
                    position=cr.center,
                    area=cr.area_m2 or 0.0,
                ))

    def _assign_parents(self):
        """Assign each element to its parent facade based on spatial proximity."""
        for elem in self.parsed_elements:
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

    def _build_report(self) -> ProjectReport:
        """Build the final ProjectReport from parsed data."""
        report = ProjectReport()

        for pf in self.parsed_facades:
            fr = FacadeReport(
                name=pf.name,
                total_area_m2=pf.total_area,
            )

            for elem in self.parsed_elements:
                if elem.parent_facade != pf.name:
                    continue
                if elem.element_type == "window":
                    fr.windows.append((elem.label, elem.area))
                elif elem.element_type == "door":
                    fr.doors.append((elem.label, elem.area))

            report.facades.append(fr)

        self.calculator.report = report
        return report

    def _build_detection_results(self):
        """Build detection_results dict for visualization."""
        self.detection_results = {"facades": [], "windows": [], "doors": []}

        for pf in self.parsed_facades:
            if pf.region_bbox:
                x, y, w, h = pf.region_bbox
            else:
                x, y = pf.position
                w, h = 200, 100

            self.detection_results["facades"].append(DetectedRegion(
                label=pf.name,
                region_type="facade",
                bbox=(x, y, w, h),
                area_m2=pf.total_area,
            ))

        for elem in self.parsed_elements:
            x, y = elem.position
            region = DetectedRegion(
                label=elem.label,
                region_type=elem.element_type,
                bbox=(x - 25, y - 15, 50, 30),
                area_m2=elem.area,
            )
            key = "windows" if elem.element_type == "window" else "doors"
            self.detection_results[key].append(region)

    def get_summary(self) -> str:
        """Get text summary of the report."""
        if self.report:
            self.calculator.report = self.report
            return self.calculator.summary_text()
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
            lines.append(
                f"  {f.name} at {f.position} "
                f"total={f.total_area}m² net={f.net_area}m²"
            )

        lines.append(f"\nParsed elements: {len(self.parsed_elements)}")
        for e in self.parsed_elements:
            lines.append(
                f"  [{e.element_type}] {e.label} at {e.position} "
                f"area={e.area}m² parent={e.parent_facade}"
            )

        return "\n".join(lines)
