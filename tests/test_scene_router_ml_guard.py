import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.color_detector import DetectedRegion
from core.pipeline import AnalysisPipeline, ParsedFacade


def _region(label: str, region_type: str, bbox: tuple, color_detected: str = "photo-test") -> DetectedRegion:
    x, y, w, h = bbox
    return DetectedRegion(
        label=label,
        region_type=region_type,
        bbox=bbox,
        contour=np.array([]),
        area_px=float(w * h),
        color_detected=color_detected,
    )


class TestSceneRouterMlGuard(unittest.TestCase):
    def setUp(self):
        self.pipeline = AnalysisPipeline()
        self.facade = ParsedFacade(
            name="ROMCEA",
            position=(700, 160),
            region_bbox=(40, 80, 1400, 320),
            source="photo",
        )
        self.photo_facade = _region(
            "FATADA_PHOTO_1",
            "facade",
            (40, 80, 1400, 320),
        )

    def test_detects_flat_central_door_cluster(self):
        openings = [
            _region("F1", "window", (230, 182, 120, 132)),
            _region("F2", "window", (612, 185, 72, 150)),
            _region("U1", "door", (710, 176, 92, 204)),
            _region("F3", "window", (826, 187, 74, 151)),
            _region("F4", "window", (1095, 180, 132, 128)),
        ]

        self.assertTrue(
            self.pipeline._has_flat_central_opening_cluster(self.facade, openings)
        )

    def test_keeps_photo_openings_when_ml_loses_flat_central_door(self):
        photo_windows = [
            _region("F1", "window", (230, 182, 120, 132)),
            _region("F2", "window", (612, 185, 72, 150)),
            _region("F3", "window", (826, 187, 74, 151)),
            _region("F4", "window", (1095, 180, 132, 128)),
        ]
        photo_doors = [
            _region("U1", "door", (710, 176, 92, 204)),
        ]
        ml_windows = [
            _region("ML_F1", "window", (232, 182, 118, 132), "ml-yolov8"),
            _region("ML_F2", "window", (612, 184, 73, 150), "ml-yolov8"),
            _region("ML_F3", "window", (709, 178, 93, 203), "ml-yolov8"),
            _region("ML_F4", "window", (826, 187, 75, 151), "ml-yolov8"),
        ]

        self.assertTrue(
            self.pipeline._should_keep_photo_openings_for_flat_scene(
                [self.photo_facade],
                photo_windows,
                photo_doors,
                ml_windows,
                [],
            )
        )

    def test_does_not_keep_photo_openings_without_balanced_side_windows(self):
        photo_windows = [
            _region("F1", "window", (230, 182, 120, 132)),
            _region("F2", "window", (612, 185, 72, 150)),
        ]
        photo_doors = [
            _region("U1", "door", (710, 176, 92, 204)),
        ]
        ml_windows = [
            _region("ML_F1", "window", (232, 182, 118, 132), "ml-yolov8"),
            _region("ML_F2", "window", (612, 184, 73, 150), "ml-yolov8"),
            _region("ML_F3", "window", (709, 178, 93, 203), "ml-yolov8"),
        ]

        self.assertFalse(
            self.pipeline._should_keep_photo_openings_for_flat_scene(
                [self.photo_facade],
                photo_windows,
                photo_doors,
                ml_windows,
                [],
            )
        )


if __name__ == "__main__":
    unittest.main()
