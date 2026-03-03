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


@dataclass
class ProjectReport:
    project_name: str = ""
    facades: list = field(default_factory=list)  # list of FacadeReport

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
            facade_map[facade.label] = fr
            self.report.facades.append(fr)

        for window in detection_results.get("windows", []):
            parent = window.parent_facade
            area = window.area_m2 or 0.0
            if area == 0 and window.width_m and window.height_m:
                area = window.width_m * window.height_m

            if parent and parent in facade_map:
                facade_map[parent].windows.append((window.label, area))
            else:
                if self.report.facades:
                    self.report.facades[-1].windows.append((window.label, area))

        for door in detection_results.get("doors", []):
            parent = door.parent_facade
            area = door.area_m2 or 0.0
            if area == 0 and door.width_m and door.height_m:
                area = door.width_m * door.height_m

            if parent and parent in facade_map:
                facade_map[parent].doors.append((door.label, area))
            else:
                if self.report.facades:
                    self.report.facades[-1].doors.append((door.label, area))

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
            lines.append(f"  Suprafata totala fatada:   {f.total_area_m2:>10.3f} m²")

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

        lines.append(f"\n{'=' * 60}")
        lines.append("  TOTALURI PROIECT")
        lines.append(f"{'=' * 60}")
        lines.append(
            f"  Suprafata totala fatade:  {self.report.total_facade_area:>10.3f} m²"
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
        lines.append("=" * 60)
        return "\n".join(lines)
