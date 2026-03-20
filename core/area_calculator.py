"""
DrawQuantPDF - Area Calculator
Computes facade, window, door and net thermosystem areas.
"""

from dataclasses import dataclass, field


@dataclass
class FacadeReport:
    name: str
    total_area_m2: float = 0.0
    windows: list = field(default_factory=list)  # list of (label, area_m2)
    doors: list = field(default_factory=list)     # list of (label, area_m2)
    reconstructed_area_m2: float = 0.0
    sill_length_m: float = 0.0
    corner_length_m: float = 0.0
    drip_profile_length_m: float = 0.0
    socle_drip_profile_length_m: float = 0.0
    window_perimeter_length_m: float = 0.0
    door_perimeter_length_m: float = 0.0
    socle_excluded_area_m2: float = 0.0

    @property
    def total_windows_area(self) -> float:
        return sum(a for _, a in self.windows)

    @property
    def total_doors_area(self) -> float:
        return sum(a for _, a in self.doors)

    @property
    def total_carpentry_area(self) -> float:
        """Total tamplarie = ferestre + usi."""
        return self.total_windows_area + self.total_doors_area

    @property
    def net_thermosystem_area(self) -> float:
        """Suprafata termosistem = fatada totala - ferestre - usi."""
        return max(0, self.total_area_m2 - self.total_carpentry_area)

    @property
    def gross_facade_area_m2(self) -> float:
        """Fatada bruta inainte de excluderea soclului."""
        return self.total_area_m2 + self.socle_excluded_area_m2


@dataclass
class ProjectReport:
    project_name: str = ""
    facades: list = field(default_factory=list)  # list of FacadeReport

    @property
    def total_gross_facade_area_m2(self) -> float:
        return sum(f.gross_facade_area_m2 for f in self.facades)

    @property
    def total_facade_area(self) -> float:
        return sum(f.total_area_m2 for f in self.facades)

    @property
    def total_windows_area(self) -> float:
        return sum(f.total_windows_area for f in self.facades)

    @property
    def total_doors_area(self) -> float:
        return sum(f.total_doors_area for f in self.facades)

    @property
    def total_carpentry_area(self) -> float:
        return sum(f.total_carpentry_area for f in self.facades)

    @property
    def total_thermosystem_area(self) -> float:
        return sum(f.net_thermosystem_area for f in self.facades)

    @property
    def total_sill_length_m(self) -> float:
        return sum(f.sill_length_m for f in self.facades)

    @property
    def total_corner_length_m(self) -> float:
        return sum(f.corner_length_m for f in self.facades)

    @property
    def total_drip_profile_length_m(self) -> float:
        return sum(f.drip_profile_length_m for f in self.facades)

    @property
    def total_socle_drip_profile_length_m(self) -> float:
        return sum(f.socle_drip_profile_length_m for f in self.facades)

    @property
    def total_window_perimeter_length_m(self) -> float:
        return sum(f.window_perimeter_length_m for f in self.facades)

    @property
    def total_door_perimeter_length_m(self) -> float:
        return sum(f.door_perimeter_length_m for f in self.facades)

    @property
    def total_socle_excluded_area_m2(self) -> float:
        return sum(f.socle_excluded_area_m2 for f in self.facades)

    @property
    def total_reconstructed_area_m2(self) -> float:
        return sum(f.reconstructed_area_m2 for f in self.facades)


class AreaCalculator:
    def __init__(self):
        self.report = ProjectReport()

    def compute_from_detections(self, detection_results: dict) -> ProjectReport:
        """Build a ProjectReport from detection results."""
        self.report = ProjectReport()

        facade_map = {}
        for facade in detection_results.get("facades", []):
            fr = FacadeReport(
                name=facade.label,
                total_area_m2=facade.area_m2 or 0.0,
            )
            if getattr(facade, "height_m", None):
                fr.corner_length_m = max(0.0, 2.0 * float(facade.height_m))
            facade_map[facade.label] = fr
            self.report.facades.append(fr)

        for window in detection_results.get("windows", []):
            parent = window.parent_facade
            area = window.area_m2 or 0.0
            if area == 0 and window.width_m and window.height_m:
                area = window.width_m * window.height_m
            width = float(window.width_m) if window.width_m else 0.0

            if parent and parent in facade_map:
                facade_map[parent].windows.append((window.label, area))
                if width > 0:
                    facade_map[parent].sill_length_m += width
                    facade_map[parent].drip_profile_length_m += width
                if width > 0 and window.height_m:
                    facade_map[parent].window_perimeter_length_m += (
                        2.0 * float(width) + 2.0 * float(window.height_m)
                    )
            else:
                if self.report.facades:
                    self.report.facades[-1].windows.append((window.label, area))
                    if width > 0:
                        self.report.facades[-1].sill_length_m += width
                        self.report.facades[-1].drip_profile_length_m += width
                    if width > 0 and window.height_m:
                        self.report.facades[-1].window_perimeter_length_m += (
                            2.0 * float(width) + 2.0 * float(window.height_m)
                        )

        for door in detection_results.get("doors", []):
            parent = door.parent_facade
            area = door.area_m2 or 0.0
            if area == 0 and door.width_m and door.height_m:
                area = door.width_m * door.height_m
            width = float(door.width_m) if door.width_m else 0.0

            if parent and parent in facade_map:
                facade_map[parent].doors.append((door.label, area))
                if width > 0:
                    facade_map[parent].drip_profile_length_m += width
                if width > 0 and door.height_m:
                    facade_map[parent].door_perimeter_length_m += (
                        2.0 * float(door.height_m) + float(width)
                    )
            else:
                if self.report.facades:
                    self.report.facades[-1].doors.append((door.label, area))
                    if width > 0:
                        self.report.facades[-1].drip_profile_length_m += width
                    if width > 0 and door.height_m:
                        self.report.facades[-1].door_perimeter_length_m += (
                            2.0 * float(door.height_m) + float(width)
                        )

        for zone in detection_results.get("missing_zones", []):
            parent = zone.parent_facade
            area = zone.area_m2 or 0.0
            if area == 0 and zone.width_m and zone.height_m:
                area = zone.width_m * zone.height_m
            if parent and parent in facade_map:
                facade_map[parent].reconstructed_area_m2 += area
                facade_map[parent].total_area_m2 += area
            elif self.report.facades:
                self.report.facades[-1].reconstructed_area_m2 += area
                self.report.facades[-1].total_area_m2 += area

        for line in detection_results.get("socle_profiles", []):
            parent = line.parent_facade
            length = float(line.length_m) if line.length_m else 0.0
            if parent and parent in facade_map:
                facade_map[parent].socle_drip_profile_length_m += length
            elif self.report.facades:
                self.report.facades[-1].socle_drip_profile_length_m += length

        for line in detection_results.get("window_perimeters", []):
            parent = line.parent_facade
            length = float(line.length_m) if line.length_m else 0.0
            if parent and parent in facade_map:
                facade_map[parent].window_perimeter_length_m += length
            elif self.report.facades:
                self.report.facades[-1].window_perimeter_length_m += length

        for line in detection_results.get("door_perimeters", []):
            parent = line.parent_facade
            length = float(line.length_m) if line.length_m else 0.0
            if parent and parent in facade_map:
                facade_map[parent].door_perimeter_length_m += length
            elif self.report.facades:
                self.report.facades[-1].door_perimeter_length_m += length

        return self.report

    def compute_manual(self, facade_data: list) -> ProjectReport:
        """Build report from manually entered data.

        facade_data: list of dicts with keys:
            name, total_area, windows: [(label, area)], doors: [(label, area)]
        """
        self.report = ProjectReport()
        for fd in facade_data:
            fr = FacadeReport(
                name=fd["name"],
                total_area_m2=fd["total_area"],
                windows=fd.get("windows", []),
                doors=fd.get("doors", []),
                reconstructed_area_m2=fd.get("reconstructed_area_m2", 0.0),
                sill_length_m=fd.get("sill_length_m", 0.0),
                corner_length_m=fd.get("corner_length_m", 0.0),
                drip_profile_length_m=fd.get("drip_profile_length_m", 0.0),
                socle_drip_profile_length_m=fd.get("socle_drip_profile_length_m", 0.0),
                window_perimeter_length_m=fd.get("window_perimeter_length_m", 0.0),
                door_perimeter_length_m=fd.get("door_perimeter_length_m", 0.0),
                socle_excluded_area_m2=fd.get("socle_excluded_area_m2", 0.0),
            )
            self.report.facades.append(fr)
        return self.report

    def summary_text(self) -> str:
        """Generate a readable summary of the report."""
        lines = []
        lines.append("=" * 60)
        lines.append("  RAPORT CANTITATI - DrawQuantPDF")
        lines.append("=" * 60)

        for i, f in enumerate(self.report.facades, 1):
            lines.append(f"\n{'─' * 50}")
            lines.append(f"  {i}. {f.name}")
            lines.append(f"{'─' * 50}")
            if f.socle_excluded_area_m2 > 0:
                lines.append(
                    f"  Fatada bruta:              {f.gross_facade_area_m2:>10.3f} m²"
                )
                lines.append(
                    f"  Soclu exclus:              {f.socle_excluded_area_m2:>10.3f} m²"
                )
            lines.append(
                f"  Suprafata fatada (fara soclu): {f.total_area_m2:>10.3f} m²"
            )
            if f.reconstructed_area_m2 > 0:
                lines.append(
                    f"  Completare asistata:        {f.reconstructed_area_m2:>10.3f} m²"
                )

            if f.windows:
                lines.append(f"  Ferestre:")
                for label, area in f.windows:
                    lines.append(f"    - {label:<20} {area:>10.3f} m²")
                lines.append(
                    f"  Total ferestre:           {f.total_windows_area:>10.3f} m²"
                )

            if f.doors:
                lines.append(f"  Usi:")
                for label, area in f.doors:
                    lines.append(f"    - {label:<20} {area:>10.3f} m²")
                lines.append(
                    f"  Total usi:                {f.total_doors_area:>10.3f} m²"
                )

            lines.append(
                f"  Total tamplarie:          {f.total_carpentry_area:>10.3f} m²"
            )
            lines.append(
                f"  TERMOSISTEM (net):        {f.net_thermosystem_area:>10.3f} m²"
            )
            if f.socle_excluded_area_m2 > 0:
                lines.append(
                    "  Formula net:              "
                    f"({f.gross_facade_area_m2:.3f} - {f.socle_excluded_area_m2:.3f})"
                    f" - {f.total_carpentry_area:.3f} = {f.net_thermosystem_area:.3f} m²"
                )
            else:
                lines.append(
                    "  Formula net:              "
                    f"{f.total_area_m2:.3f} - {f.total_carpentry_area:.3f}"
                    f" = {f.net_thermosystem_area:.3f} m²"
                )
            lines.append(
                f"  Glaf exterior:            {f.sill_length_m:>10.3f} m"
            )
            lines.append(
                f"  Picurator goluri:         {f.drip_profile_length_m:>10.3f} m"
            )
            lines.append(
                f"  Lungime coltare:          {f.corner_length_m:>10.3f} m"
            )
            lines.append(
                f"  Picurator soclu:          {f.socle_drip_profile_length_m:>10.3f} m"
            )
            lines.append(
                f"  Perimetru ferestre:       {f.window_perimeter_length_m:>10.3f} m"
            )
            lines.append(
                f"  Perimetru usi:            {f.door_perimeter_length_m:>10.3f} m"
            )

        lines.append(f"\n{'=' * 60}")
        lines.append("  TOTALURI PROIECT")
        lines.append(f"{'=' * 60}")
        lines.append(
            f"  Suprafata bruta fatade:   {self.report.total_gross_facade_area_m2:>10.3f} m²"
        )
        lines.append(
            f"  Suprafata fatade (fara soclu): {self.report.total_facade_area:>10.3f} m²"
        )
        lines.append(
            f"  Total ferestre:           {self.report.total_windows_area:>10.3f} m²"
        )
        lines.append(
            f"  Total usi:                {self.report.total_doors_area:>10.3f} m²"
        )
        lines.append(
            f"  Total tamplarie:          {self.report.total_carpentry_area:>10.3f} m²"
        )
        lines.append(
            f"  TOTAL TERMOSISTEM:        "
            f"{self.report.total_thermosystem_area:>10.3f} m²"
        )
        lines.append(
            "  Formula total net:        "
            f"({self.report.total_gross_facade_area_m2:.3f} - "
            f"{self.report.total_socle_excluded_area_m2:.3f}) - "
            f"{self.report.total_carpentry_area:.3f} = "
            f"{self.report.total_thermosystem_area:.3f} m²"
        )
        lines.append(
            f"  TOTAL SOCLU EXCLUS:       "
            f"{self.report.total_socle_excluded_area_m2:>10.3f} m²"
        )
        lines.append(
            f"  TOTAL COMPLETARE ASISTATA:{self.report.total_reconstructed_area_m2:>10.3f} m²"
        )
        lines.append(
            f"  TOTAL GLAF EXTERIOR:      "
            f"{self.report.total_sill_length_m:>10.3f} m"
        )
        lines.append(
            f"  TOTAL PICURATOR GOLURI:   "
            f"{self.report.total_drip_profile_length_m:>10.3f} m"
        )
        lines.append(
            f"  TOTAL LUNGIME COLTARE:    "
            f"{self.report.total_corner_length_m:>10.3f} m"
        )
        lines.append(
            f"  TOTAL PICURATOR SOCLU:    "
            f"{self.report.total_socle_drip_profile_length_m:>10.3f} m"
        )
        lines.append(
            f"  TOTAL PERIMETRU FERESTRE: "
            f"{self.report.total_window_perimeter_length_m:>10.3f} m"
        )
        lines.append(
            f"  TOTAL PERIMETRU USI:      "
            f"{self.report.total_door_perimeter_length_m:>10.3f} m"
        )
        lines.append("=" * 60)
        return "\n".join(lines)
