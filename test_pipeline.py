"""Test the full analysis pipeline on the sample image."""

import sys
import os
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from core.pipeline import AnalysisPipeline
from core.excel_exporter import ExcelExporter

image_path = os.path.join("assets", "sample_facade.png")
img = cv2.imread(image_path)
print(f"Image: {img.shape[1]}x{img.shape[0]}px\n")

pipeline = AnalysisPipeline()

def progress(msg, val):
    print(f"  [{val*100:.0f}%] {msg}")

report = pipeline.run(img, progress_callback=progress)

print("\n" + pipeline.get_ocr_debug())
print("\n" + pipeline.get_summary())

# Export Excel test
exporter = ExcelExporter()
excel_path = os.path.join("assets", "test_output.xlsx")
exporter.export(report, excel_path, "Test Fatade")
print(f"\nExcel saved: {excel_path}")

# Save visualization
viz = pipeline.color_detector.draw_detections(img, pipeline.detection_results)
cv2.imwrite(os.path.join("assets", "pipeline_result.png"), viz)
print("Visualization saved: assets/pipeline_result.png")

# Optional: run evaluator on fixtures if present
_fixture_img = os.path.join("tests", "fixtures", "images", "fixture_facade.png")
_fixture_gt = os.path.join("tests", "fixtures", "gt", "fixture_facade.json")
if os.path.exists(_fixture_img) and os.path.exists(_fixture_gt):
    from tools.eval.run_eval import run_eval
    _r = run_eval(_fixture_img, _fixture_gt, "fixture_facade")
    print(f"\n[Eval] fixture_facade: IoU={_r.facade_iou_mean:.3f} "
          f"P={_r.window_precision:.3f} R={_r.window_recall:.3f}")
