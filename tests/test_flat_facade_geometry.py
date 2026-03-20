import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import AnalysisPipeline, ParsedElement, ParsedFacade


def _local_contour(points: list[list[int]]) -> np.ndarray:
    return np.array(points, dtype=np.int32).reshape((-1, 1, 2))


class TestFlatFacadeGeometry(unittest.TestCase):
    def setUp(self):
        self.pipeline = AnalysisPipeline()
        self.facade = ParsedFacade(
            name="ROMCEA",
            position=(700, 500),
            region_bbox=(83, 426, 1400, 214),
            source="photo",
            scene_type="flat_long_facade",
        )
        self.pipeline.parsed_facades = [self.facade]
        self.pipeline.parsed_elements = [
            ParsedElement(
                label="F_LEFT",
                element_type="window",
                position=(383, 513),
                bbox=(295, 449, 176, 128),
                parent_facade="ROMCEA",
                source="photo",
            ),
            ParsedElement(
                label="F_SIDE_LEFT",
                element_type="window",
                position=(682, 500),
                bbox=(640, 425, 84, 151),
                parent_facade="ROMCEA",
                source="photo",
            ),
            ParsedElement(
                label="DOOR",
                element_type="door",
                position=(789, 547),
                bbox=(728, 442, 122, 210),
                parent_facade="ROMCEA",
                source="photo",
            ),
            ParsedElement(
                label="F_SIDE_RIGHT",
                element_type="window",
                position=(896, 495),
                bbox=(854, 411, 85, 168),
                parent_facade="ROMCEA",
                source="photo",
            ),
            ParsedElement(
                label="F_RIGHT",
                element_type="window",
                position=(1188, 501),
                bbox=(1105, 437, 167, 129),
                parent_facade="ROMCEA",
                source="photo",
            ),
        ]

    def test_detects_right_side_profile_break(self):
        contour = _local_contour(
            [
                [0, 0],
                [1399, 0],
                [1399, 62],
                [1378, 213],
                [21, 213],
            ]
        )

        self.assertTrue(self.pipeline._has_right_side_profile_break(contour))

    def test_rejects_plain_flat_band_without_profile_break(self):
        contour = _local_contour(
            [
                [0, 0],
                [1399, 0],
                [1399, 213],
                [0, 213],
            ]
        )

        self.assertFalse(self.pipeline._has_right_side_profile_break(contour))

    def test_profiled_contour_keeps_opening_centers(self):
        contour = _local_contour(
            [
                [0, 0],
                [1399, 0],
                [1399, 62],
                [1378, 213],
                [21, 213],
            ]
        )

        self.assertTrue(
            self.pipeline._contour_keeps_facade_openings(self.facade, contour, 83, 426)
        )

    def test_rejects_contour_that_cuts_right_openings(self):
        contour = _local_contour(
            [
                [0, 0],
                [740, 0],
                [740, 213],
                [0, 213],
            ]
        )

        self.assertFalse(
            self.pipeline._contour_keeps_facade_openings(self.facade, contour, 83, 426)
        )

    def test_profile_from_polyline_preserves_step_break(self):
        profile = self.pipeline._profile_from_line_contour(
            _local_contour(
                [
                    [0, 303],
                    [1241, 303],
                    [1242, 191],
                    [2230, 191],
                ]
            ),
            2231,
        )

        self.assertEqual(int(round(float(profile[100]))), 303)
        self.assertEqual(int(round(float(profile[1241]))), 303)
        self.assertEqual(int(round(float(profile[1242]))), 191)
        self.assertEqual(int(round(float(profile[1800]))), 191)


if __name__ == "__main__":
    unittest.main()
