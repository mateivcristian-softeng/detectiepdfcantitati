"""Test the full analysis pipeline on the sample image."""

import sys, os
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
