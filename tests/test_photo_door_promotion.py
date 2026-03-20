import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.color_detector import ColorDetector, DetectedRegion


def _region(label: str, bbox: tuple) -> DetectedRegion:
    x, y, w, h = bbox
    return DetectedRegion(
        label=label,
        region_type="window",
        bbox=bbox,
        contour=np.array([]),
        area_px=float(w * h),
        color_detected="photo-test",
    )


class TestPhotoDoorPromotion(unittest.TestCase):
    def setUp(self):
        self.detector = ColorDetector()
        self.facade = DetectedRegion(
            label="FATADA_1",
            region_type="facade",
            bbox=(0, 0, 1000, 500),
            contour=np.array([]),
            area_px=float(1000 * 500),
            color_detected="photo-test",
        )

    def test_promotes_single_central_bottom_opening(self):
        windows = [
            _region("W1", (120, 150, 90, 95)),
            _region("W2", (300, 150, 92, 98)),
            _region("W3", (700, 150, 90, 95)),
            _region("W4", (840, 150, 88, 96)),
            _region("W5", (470, 185, 115, 235)),
        ]

        promoted = self.detector._select_promoted_photo_door([self.facade], windows)
        self.assertIsNotNone(promoted)
        self.assertEqual(promoted["bbox"], (470, 185, 115, 235))
        self.assertEqual(len(promoted["windows"]), 1)

    def test_promotes_vertical_pair_when_door_is_split(self):
        windows = [
            _region("W1", (120, 150, 90, 95)),
            _region("W2", (300, 150, 92, 98)),
            _region("W3", (700, 150, 90, 95)),
            _region("W4", (840, 150, 88, 96)),
            _region("W5", (470, 180, 112, 86)),
            _region("W6", (474, 272, 108, 155)),
        ]

        promoted = self.detector._select_promoted_photo_door([self.facade], windows)
        self.assertIsNotNone(promoted)
        self.assertEqual(promoted["bbox"], (470, 180, 112, 247))
        self.assertEqual(len(promoted["windows"]), 2)

    def test_merges_fragmented_window_before_door_selection(self):
        windows = [
            _region("W1", (205, 150, 141, 119)),
            _region("W2", (521, 150, 65, 119)),
            _region("W3", (586, 203, 84, 69)),
            _region("W4", (753, 155, 61, 134)),
            _region("W5", (919, 184, 149, 90)),
            _region("W6", (1169, 152, 103, 110)),
        ]

        merged = self.detector._merge_photo_window_fragments([self.facade], windows)
        merged_boxes = [w.bbox for w in merged]
        self.assertIn((521, 150, 149, 122), merged_boxes)
        self.assertEqual(len(merged), 5)

    def test_promotes_narrow_realistic_door(self):
        facade = DetectedRegion(
            label="FATADA_REAL",
            region_type="facade",
            bbox=(20, 265, 1400, 296),
            contour=np.array([]),
            area_px=float(1400 * 296),
            color_detected="photo-test",
        )
        windows = [
            _region("W1", (205, 312, 141, 119)),
            _region("W2", (521, 310, 149, 122)),
            _region("W3", (753, 317, 61, 134)),
            _region("W4", (919, 344, 149, 90)),
            _region("W5", (1169, 312, 103, 110)),
        ]

        promoted = self.detector._select_promoted_photo_door([facade], windows)
        self.assertIsNotNone(promoted)
        self.assertEqual(promoted["bbox"], (753, 317, 61, 134))
        self.assertEqual(len(promoted["windows"]), 1)

    def test_rejects_ambiguous_low_windows(self):
        windows = [
            _region("W1", (110, 285, 110, 120)),
            _region("W2", (300, 290, 110, 120)),
            _region("W3", (470, 292, 110, 118)),
            _region("W4", (650, 289, 110, 120)),
            _region("W5", (820, 286, 110, 119)),
        ]

        promoted = self.detector._select_promoted_photo_door([self.facade], windows)
        self.assertIsNone(promoted)

    def test_completes_door_hole_below_glazed_seed(self):
        gray = np.full((120, 160), 170, dtype=np.uint8)
        # Door opening: upper glazed part and lower opaque panel.
        gray[20:55, 60:90] = 105
        gray[55:86, 60:90] = 128

        bbox = self.detector._complete_photo_door_hole_bbox(
            gray, (0, 0, 160, 120), (60, 20, 30, 35)
        )
        self.assertGreaterEqual(bbox[3], 60)
        self.assertLessEqual(bbox[3], 70)

    def test_does_not_extend_when_lower_part_matches_wall(self):
        gray = np.full((120, 160), 170, dtype=np.uint8)
        gray[20:55, 60:90] = 105
        # Below the glazed part, wall resumes with same tone.
        gray[55:90, 60:90] = 170

        bbox = self.detector._complete_photo_door_hole_bbox(
            gray, (0, 0, 160, 120), (60, 20, 30, 35)
        )
        self.assertLessEqual(bbox[3], 42)

    def test_extends_when_jamb_edges_continue_downward(self):
        gray = np.full((140, 180), 180, dtype=np.uint8)
        gray[20:55, 70:100] = 110
        gray[55:92, 70:100] = 150
        # Vertical jamb edges remain visible below the glazed panel.
        gray[20:92, 68:70] = 40
        gray[20:92, 100:102] = 40

        bbox = self.detector._extend_door_by_jamb_continuity(
            gray, (0, 0, 180, 140), (70, 20, 30, 35)
        )
        self.assertGreaterEqual(bbox[3], 68)

    def test_stops_when_jamb_edges_end(self):
        gray = np.full((140, 180), 180, dtype=np.uint8)
        gray[20:55, 70:100] = 110
        gray[20:55, 68:70] = 40
        gray[20:55, 100:102] = 40

        bbox = self.detector._extend_door_by_jamb_continuity(
            gray, (0, 0, 180, 140), (70, 20, 30, 35)
        )
        self.assertLessEqual(bbox[3], 45)

    def test_expands_door_width_when_jambs_are_outside_seed(self):
        gray = np.full((140, 200), 180, dtype=np.uint8)
        gray[20:90, 78:108] = 120
        gray[20:90, 70:72] = 35
        gray[20:90, 118:120] = 35

        bbox = self.detector._expand_door_width_by_jambs(
            gray, (0, 0, 200, 140), (78, 20, 30, 70)
        )
        self.assertLessEqual(bbox[0], 70)
        self.assertGreaterEqual(bbox[2], 45)

    def test_keeps_width_when_no_clear_outer_jambs(self):
        gray = np.full((140, 200), 180, dtype=np.uint8)
        gray[20:90, 78:108] = 120
        gray[20:90, 78:80] = 35
        gray[20:90, 106:108] = 35

        bbox = self.detector._expand_door_width_by_jambs(
            gray, (0, 0, 200, 140), (78, 20, 30, 70)
        )
        self.assertLessEqual(bbox[2], 36)

    def test_fits_door_to_tight_composite_gap(self):
        bbox = self.detector._fit_door_to_window_gap(
            (131, 155, 2230, 658),
            (626, 588, 101, 225),
            (439, 543, 82, 128),
            (642, 543, 44, 126),
        )
        self.assertEqual(bbox, (522, 588, 119, 225))

    def test_recenters_door_in_wide_gap_without_overgrowing(self):
        bbox = self.detector._fit_door_to_window_gap(
            (131, 155, 2230, 658),
            (1262, 544, 122, 200),
            (833, 541, 182, 128),
            (1475, 403, 181, 134),
        )
        self.assertEqual(bbox, (1213, 544, 134, 200))


if __name__ == "__main__":
    unittest.main()
