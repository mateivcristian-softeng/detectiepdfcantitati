import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.area_calculator import AreaCalculator
from core.color_detector import DetectedRegion


class TestMissingZoneReport(unittest.TestCase):
    def test_missing_zone_area_is_added_to_facade(self):
        calc = AreaCalculator()
        results = {
            "facades": [
                DetectedRegion(
                    label="F1",
                    region_type="facade",
                    bbox=(0, 0, 100, 100),
                    contour=np.array([]),
                    area_m2=10.0,
                )
            ],
            "windows": [
                DetectedRegion(
                    label="W1",
                    region_type="window",
                    bbox=(10, 10, 20, 20),
                    contour=np.array([]),
                    area_m2=1.5,
                    parent_facade="F1",
                )
            ],
            "doors": [],
            "missing_zones": [
                DetectedRegion(
                    label="ZL1",
                    region_type="missing_zone",
                    bbox=(100, 0, 20, 100),
                    contour=np.array([]),
                    area_m2=2.0,
                    parent_facade="F1",
                )
            ],
        }

        report = calc.compute_from_detections(results)
        self.assertEqual(len(report.facades), 1)
        facade = report.facades[0]
        self.assertAlmostEqual(facade.total_area_m2, 12.0, places=3)
        self.assertAlmostEqual(facade.reconstructed_area_m2, 2.0, places=3)
        self.assertAlmostEqual(facade.net_thermosystem_area, 10.5, places=3)
        self.assertAlmostEqual(report.total_reconstructed_area_m2, 2.0, places=3)


if __name__ == "__main__":
    unittest.main()
