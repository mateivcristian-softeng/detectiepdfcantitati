# Session Handoff 2026-03-15 (updated 2026-03-16)

## Current Metrics (confirmed 2026-03-16)

| Case | facade_iou | socle_iou | window_iou | door_iou |
|------|-----------|-----------|-----------|---------|
| ARION | 0.8347 | 0.0125 | 0.1259 | 0.4196 |
| MARGINEAN_DREAPTA | 0.7731 | 0.6000 | — | — |
| MUNTEAN_STANGA | 0.7703 | 0.6667 | — | — |
| ROMCEA | null | null | null | null |

## Reference Validation (2026-03-16)

`tools/eval/validate_reference_geometry.py` validates ALL reference JSONs.

| Reference | Verdict | Key Issue |
|-----------|---------|-----------|
| ROMCEA | VALID | No geometry ref yet |
| **ARION** | **INVALID** | 3 OOB pts, degenerate socle (475px), bright wall box, image size mismatch |
| **MARGINEAN** | **INVALID** | 4 OOB pts, degenerate socle (4px), image size mismatch |
| **MUNTEAN** | **INVALID** | 5 OOB pts, degenerate socle (6px), image size mismatch |

**NOTE**: MARGINEAN/MUNTEAN socle IoU works despite degenerate raster because BOTH detected and reference
socle are thin bottom strips (~4-6px each). For ARION, detected socle is large (36,499px) vs reference (475px).

## Proposed ARION Reference

File: `docs/reference_cases/arion_expected_proposed.json`

### Changes vs official
| Field | Official | Proposed | Reason |
|-------|----------|----------|--------|
| image_size | [2408, 1007] | [2414, 908] | Official from rendering, not raw PNG |
| facade_bbox | [87,168,2268,641] | [135,176,2226,544] | Matched to raw image detection |
| socle_contour OOB | 3/6 points OOB | 0/6 OOB | Clipped to y_norm ≤ 1.0 |
| socle raster area | 475px | 35,593px | Fixed from degenerate |
| window_boxes | 6 (from rendering) | 6 (from raw detector) | Official win1 at bright wall |
| door_boxes | 2 (from rendering) | 2 (from raw detector) | Official door1 at bright wall |

### Experiment: official vs proposed
| Metric | Official | Proposed |
|--------|----------|----------|
| facade_shape_iou | 0.8347 | **1.0000** |
| socle_shape_iou | 0.0125 | **1.0000** |
| window_box_mean_iou | 0.1259 | **1.0000** |
| door_box_mean_iou | 0.4196 | **1.0000** |

Proposed is 1.0000 because boxes are detector-derived. **HUMAN MUST VERIFY** the proposed boxes.

## Artifacts for Human Review

### Validation
- `debug/reference_validation/reference_validation_report.md` — full validation of all 3 refs
- `debug/reference_validation/reference_validation_report.json` — structured

### Re-annotation package
- `debug/arion_reannotation/01_raw_with_official_expected.png` — raw image + official boxes (RED)
- `debug/arion_reannotation/02_raw_with_proposed_expected.png` — raw image + proposed boxes (GREEN)
- `debug/arion_reannotation/03_official_vs_proposed_overlay.png` — comparison overlay
- `debug/arion_reannotation/04_socle_official_raster.png` — official socle on 512×512 (475px)
- `debug/arion_reannotation/05_socle_proposed_raster.png` — proposed socle on 512×512 (35,593px)
- `debug/arion_reannotation/arion_expected_diff.md` — diff explanation
- `debug/arion_reannotation/regression_official_vs_proposed.md` — metric comparison

### Tools
- `tools/eval/validate_reference_geometry.py` — reference geometry validator

## Core Patches (unchanged from previous session)
1. IQR filter skip + per-column proximity mask for `composite_stepped_facade`
2. Guard in `_regularize_composite_facade_bottom_against_socle`
3. `_clip_elements_to_facade_bottom` for `composite_stepped_facade`

## Next Steps
1. **HUMAN REVIEW**: Inspect `debug/arion_reannotation/02_raw_with_proposed_expected.png`
   - Verify window/door boxes match real openings in the raw photo
   - Adjust boxes manually if needed
   - Approve or modify `docs/reference_cases/arion_expected_proposed.json`
2. Once approved: replace `arion_expected.json` with the reviewed version
3. Re-run regression to establish new baseline
4. If metrics < 1.0 after correction → resume detector work on the actual gaps
5. Consider applying similar fixes to MARGINEAN/MUNTEAN socle references (lower priority — IoU already works)
