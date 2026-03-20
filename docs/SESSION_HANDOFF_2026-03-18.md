# Session Handoff - 2026-03-18

## Scope
Continue the facade ML track from the discussion about annotated images in:
`C:\Users\Admin\Desktop\Antranare imagini`

## What Was Done
1. Fixed dataset import so YOLO labels are generated from Pascal VOC XML first, then JSON, and only then TXT fallback.
2. Re-generated `data/yolo_facade` from the source annotations.
3. Added regression tests for the importer.
4. Re-ran the project test suite.
5. Trained fresh YOLO models on the cleaned dataset.
6. Evaluated both validation metrics and canonical pipeline behavior.

## Files Touched
- `tools/eval/import_labelimg_annotations.py`
- `tests/test_import_labelimg_annotations.py`
- regenerated `data/yolo_facade/...`

## Key Dataset Findings
- Old numeric TXT mapping was unreliable across files.
- Safe import now prefers named annotations from XML/JSON.
- Re-generated dataset summary:
  - 134 annotated images
  - 114 train / 20 val
  - 624 total labels
  - class counts: facade 155, window 248, door 76, socle 145
  - annotation sources used: XML 124, JSON 10, TXT 0

## Verification
- `py -m unittest tests.test_import_labelimg_annotations -v` passed
- `py -m unittest discover -s tests -p "test_*.py"` passed
  - 21 tests

## Trained Models
- `models/yolov8n_facade_v7_clean_xml_best.pt`
- `models/yolov8s_facade_v2_clean_xml_best.pt`

## Standardized Val Metrics
Same `model.val(... split='val', workers=0)` evaluation for comparison.

- `yolov8n_facade_v4_best.pt`
  - mAP50: 0.7348
  - mAP50-95: 0.6134
- `yolov8n_facade_v6_best.pt`
  - mAP50: 0.5712
  - mAP50-95: 0.4273
- `yolov8n_facade_v7_clean_xml_best.pt`
  - mAP50: 0.5650
  - mAP50-95: 0.4090
- `yolov8s_facade_v2_clean_xml_best.pt`
  - mAP50: 0.6157
  - mAP50-95: 0.4132

## Canonical Behavior Summary
### `yolov8n_facade_v7_clean_xml_best.pt`
- `ROMCEA`: 5 windows, 0 doors -> still wrong
- `ARION`: 5 windows, 2 doors -> counts closer, geometry still weak
- `MARGINEAN_DREAPTA`: 0 windows -> improvement
- `MUNTEAN_STANGA`: acceptable counts

### `yolov8s_facade_v2_clean_xml_best.pt`
- `ROMCEA`: 4 windows, 0 doors -> still misses central door
- `ARION`: 5 windows, 3 doors -> still weak geometrically
- `MARGINEAN_DREAPTA`: 0 windows -> improvement
- `MUNTEAN_STANGA`: acceptable counts

## Important Conclusions
1. The systemic dataset import bug is fixed.
2. The remaining blocker is no longer the import path.
3. Both clean models still confuse the central `ROMCEA` door as window(s).
4. `ARION` is not present in `C:\Users\Admin\Desktop\Antranare imagini`, so the clean retrain does not cover that canonical scenario.
5. This is now a pipeline/routing problem plus a targeted data coverage problem, not just a larger-model problem.

## Exact Next Step For Tomorrow
Implement a guarded ML integration step in the pipeline for the canonical flat-facade scenario.

Target direction:
- `scene-router`: protect `ROMCEA` / `flat_long_facade`
- `opening-detector`: if ML returns a central opening cluster with `4 windows, 0 doors`, do not blindly replace the photo-based result
- keep the `window + door + window` structure when the flat facade pattern is detected
- do this as scenario routing, not as a global hack

Practical plan:
1. Inspect `core/pipeline.py` ML override path.
2. Add scenario guard before replacing photo windows/doors with ML output.
3. Preserve or re-promote the central door in `flat_long_facade` when ML fails to emit any door.
4. Re-run `tools/eval/run_canonical_regression.py` against canonical cases.
5. Only after that decide whether to add curated `ARION`-like training samples.

## Notes
- Do not promote the new YOLO models to default yet.
- `yolov8n_facade_v4_best.pt` is still stronger on raw val metrics, even though the clean models improved some false positives.
- The next change should be in `scene-router` / `opening-detector`, not another blind retrain.
