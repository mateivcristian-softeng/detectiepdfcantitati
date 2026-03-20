import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import AnalysisPipeline, ParsedElement, ParsedFacade


class TestCompositeSocleGeometry(unittest.TestCase):
    def setUp(self):
        self.pipeline = AnalysisPipeline()
        self.pipeline.current_image = np.zeros((800, 900, 3), dtype=np.uint8)
        self.facade = ParsedFacade(
            name="ARION_SYNTH",
            position=(300, 250),
            region_bbox=(0, 0, 700, 520),
            source="photo",
            scene_type="composite_stepped_facade",
        )
        self.pipeline.parsed_facades = [self.facade]
        self.pipeline.parsed_elements = [
            ParsedElement(
                label="W_UPPER_A",
                element_type="window",
                position=(520, 210),
                bbox=(470, 140, 100, 120),
                parent_facade="ARION_SYNTH",
                source="photo",
            ),
            ParsedElement(
                label="W_UPPER_B",
                element_type="window",
                position=(640, 210),
                bbox=(600, 140, 80, 120),
                parent_facade="ARION_SYNTH",
                source="photo",
            ),
            ParsedElement(
                label="W_LOWER_A",
                element_type="window",
                position=(160, 420),
                bbox=(100, 340, 120, 100),
                parent_facade="ARION_SYNTH",
                source="photo",
            ),
            ParsedElement(
                label="W_LOWER_B",
                element_type="window",
                position=(295, 420),
                bbox=(250, 340, 90, 100),
                parent_facade="ARION_SYNTH",
                source="photo",
            ),
            ParsedElement(
                label="D_RIGHT",
                element_type="door",
                position=(555, 350),
                bbox=(500, 240, 110, 200),
                parent_facade="ARION_SYNTH",
                source="photo",
            ),
            ParsedElement(
                label="D_LEFT",
                element_type="door",
                position=(430, 390),
                bbox=(380, 320, 100, 120),
                parent_facade="ARION_SYNTH",
                source="photo",
            ),
        ]

    def test_estimated_composite_socle_top_profile_uses_upper_right_door_top(self):
        facade_mask = np.zeros((620, 760), dtype=np.uint8)
        facade_mask[60:540, 20:720] = 255

        profile = self.pipeline._estimate_composite_socle_top_profile(
            self.facade,
            facade_mask,
            0,
            0,
        )

        self.assertGreater(profile.size, 0)
        self.assertGreater(float(profile[120]), 420.0)
        self.assertLess(float(profile[620]), 280.0)

    def test_window_only_guard_allows_composite_socle_cut_above_doors(self):
        contour = np.array(
            [
                [20, 60],
                [720, 60],
                [720, 260],
                [380, 260],
                [380, 440],
                [20, 440],
            ],
            dtype=np.int32,
        ).reshape((-1, 1, 2))

        self.assertFalse(
            self.pipeline._contour_keeps_facade_openings(self.facade, contour, 0, 0)
        )
        self.assertTrue(
            self.pipeline._contour_keeps_facade_openings(
                self.facade,
                contour,
                0,
                0,
                element_types={"window"},
            )
        )


if __name__ == "__main__":
    unittest.main()
