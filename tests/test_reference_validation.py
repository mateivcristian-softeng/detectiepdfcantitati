import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.eval.reference_validation import analyze_opening_box
from tools.eval.validate_reference_geometry import validate_reference


ROOT = Path(__file__).resolve().parents[1]


class TestReferenceValidation(unittest.TestCase):
    def test_rejects_uniform_bright_wall_box(self):
        gray = np.full((120, 160), 225, dtype=np.uint8)

        metrics = analyze_opening_box(gray, (40, 20, 60, 80))
        self.assertFalse(metrics["likely_opening"])
        self.assertTrue(metrics["bright_wall"])

    def test_accepts_bright_box_with_frame_structure(self):
        gray = np.full((140, 180), 228, dtype=np.uint8)
        gray[26:114, 62:118] = 206
        gray[26:114, 62:64] = 158
        gray[26:114, 116:118] = 158
        gray[26:30, 62:118] = 165
        gray[58:82, 86:94] = 196

        metrics = analyze_opening_box(gray, (62, 26, 56, 88))
        self.assertTrue(metrics["likely_opening"])
        self.assertFalse(metrics["bright_wall"])
        self.assertGreater(metrics["jamb_score"], 0.05)

    def test_arion_reference_is_valid_against_raw_image(self):
        manifest = json.loads((ROOT / "docs/reference_cases/canonical_scenarios.json").read_text(encoding="utf-8-sig"))
        arion = next(case for case in manifest["scenarios"] if case["id"] == "ARION")
        report = validate_reference(ROOT / arion["reference_json_path"], arion)

        self.assertEqual(report["verdict"], "VALID")
        self.assertTrue(report["geometry_valid"])
        self.assertTrue(report["openings_valid"])
        self.assertEqual(report["reference_id"], "arion_reviewed_raw_v3_2026-03-19")

    def test_marginean_processed_reference_is_geometry_valid_against_raw_image(self):
        manifest = json.loads((ROOT / "docs/reference_cases/canonical_scenarios.json").read_text(encoding="utf-8-sig"))
        case = next(item for item in manifest["scenarios"] if item["id"] == "MARGINEAN_DREAPTA")
        report = validate_reference(ROOT / case["reference_json_path"], case)

        self.assertEqual(report["verdict"], "WARNING")
        self.assertTrue(report["geometry_valid"])
        self.assertTrue(report["openings_valid"])
        self.assertFalse(report["bounds_violations"])
        self.assertIn("reference image_size does not match raw PNG dimensions", report["issues"])

    def test_muntean_processed_reference_is_geometry_valid_against_raw_image(self):
        manifest = json.loads((ROOT / "docs/reference_cases/canonical_scenarios.json").read_text(encoding="utf-8-sig"))
        case = next(item for item in manifest["scenarios"] if item["id"] == "MUNTEAN_STANGA")
        report = validate_reference(ROOT / case["reference_json_path"], case)

        self.assertEqual(report["verdict"], "WARNING")
        self.assertTrue(report["geometry_valid"])
        self.assertTrue(report["openings_valid"])
        self.assertFalse(report["bounds_violations"])
        self.assertIn("reference image_size does not match raw PNG dimensions", report["issues"])


if __name__ == "__main__":
    unittest.main()
