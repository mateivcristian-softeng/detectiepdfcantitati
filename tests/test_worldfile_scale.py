import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.worldfile_scale import read_worldfile_px_per_m


class TestWorldfileScale(unittest.TestCase):
    def test_reads_pgw_sidecar_and_computes_px_per_m(self):
        with tempfile.TemporaryDirectory() as td:
            image_path = os.path.join(td, "sample.png")
            pgw_path = os.path.join(td, "sample.pgw")

            with open(image_path, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n")

            # 0.01 m / px => 100 px / m
            with open(pgw_path, "w", encoding="utf-8") as f:
                f.write("0.01\n0\n0\n-0.01\n1000\n2000\n")

            info = read_worldfile_px_per_m(image_path)
            self.assertIsNotNone(info)
            self.assertAlmostEqual(info["px_per_m"], 100.0, places=6)


if __name__ == "__main__":
    unittest.main()
