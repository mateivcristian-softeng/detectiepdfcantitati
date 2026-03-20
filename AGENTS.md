# Project Agents

## Canonical Reference Cases

When tuning facade/window/door/socle detection, always use these references first:

1. `ROMCEA` flat facade with central door + sidelights
- Raw image: `detector cantitati/raw/romcea_raw/ROMCEA MARIA_colorized_fatada_spate.png`
- World file: `detector cantitati/raw/romcea_raw/ROMCEA MARIA_colorized_fatada_spate.pgw`
- Canonical spec: `docs/reference_cases/romcea_spate_reference.md`
- Canonical JSON target: `docs/reference_cases/romcea_spate_expected.json`
- Reserved path for the exact user-provided reference screenshot: `assets/reference/romcea_spate_expected_reference_chat_2026-03-11.png`

2. `ARION` door + adjacent windows regression
- Raw image: `detector cantitati/raw/usa + fereastra alaturate_raw/ARION SIMION_colorized_fatada_fata.png`
- Processed comparison image: `detector cantitati/procesate/ușă + ferestre alăturate/ARION SIMION_colorized_fatada_fata_procesat.png`

## Required Comparison Rules

- Do not tune only by one screenshot of current output.
- Treat `ROMCEA` as the canonical flat-facade reference for:
  - facade upper contour under the eave,
  - facade lower cutoff exactly at socle top,
  - socle surface reaching left bottom edge,
  - socle drip profile as a line segment, not a filled area,
  - central cluster separation into `window + door + window`.
- Treat `ARION` as the canonical merged-volume / adjacent-openings regression.
- Do not claim general robustness. Improve by scenario and keep regressions explicit.

## Working Roles

Use these roles explicitly when improving the detector. They are logical workstreams, not separate codebases.

1. `scene-router`
- Goal: classify each facade into the correct scenario before refinement.
- Owns: `core/pipeline.py` scene classification and routing.
- Must protect: `ROMCEA` from regressions caused by `gable` or `composite` logic.

2. `opening-detector`
- Goal: detect windows and doors per scenario, not globally.
- Owns: `core/color_detector.py` opening detection, promotion, split, and pruning.
- Must protect: central `window + door + window` on `ROMCEA`.

3. `facade-geometry`
- Goal: produce facade and socle contours that follow the real structure, not noise.
- Owns: facade contour cleanup, socle exclusion, socle profile line.
- Must protect: facade lower cutoff at socle top and drip profile as a line segment.

4. `regression-judge`
- Goal: compare each change against canonical scenarios and reject regressions.
- Owns: scenario matrix, benchmark scripts, expected geometry specs.
- Must protect: no claim of robustness without explicit scenario coverage.

## Scenario Classes

Every new tuning pass must target one of these classes:

1. `flat_long_facade`
- Canonical: `ROMCEA`
- Expected: straight/near-straight upper body under eave, lower cutoff at socle top, central cluster split.

2. `composite_stepped_facade`
- Canonical: `ARION`
- Expected: merged facade body with stepped roofline and multiple vertical opening bands.

3. `gable_facade`
- Canonical: `MARGINEAN ANA_colorized_fatada_dreapta`
- Expected: triangular/pentagonal facade contour, no fake door/window cluster in the lower-right dark area.

4. `sparse_openings_flat`
- Canonical: `MUNTEAN LUCRETIA_colorized_fatada_stanga`
- Expected: keep the one real window, suppress fake door + sidelight pair.

## Required Artifacts Before Claiming Improvement

For each scenario class, keep all of these:
- raw image path
- scale source (`PGW` when available, otherwise explicit fallback)
- expected contour sketch or processed reference image
- current screenshot from UI
- short note: `what improved / what is still wrong`

## Missing Skills To Add

These are the next useful custom skills for this project:

1. `facade-scene-routing`
- Focus: deterministic scene classification, routing rules, scenario guards, acceptance checks.

2. `facade-openings-regression`
- Focus: door/window detection by scenario, split/merge rules, false-positive pruning, batch regression summaries.

3. `facade-geometry-regularization`
- Focus: contour cleanup, socle band extraction, drip-profile line generation, polygon regularization.

4. `facade-dataset-triage`
- Focus: turn screenshots/raw/expected overlays into an indexed regression set with issue tags.

## Parallel Orchestration Rules

Use parallel workstreams, but never parallel blind edits.

1. Shared contract
- Every workstream must read from `docs/reference_cases/canonical_scenarios.json`.
- Every workstream must validate against `tools/eval/run_canonical_regression.py`.

2. Allowed parallel lanes
- `scene-router`
  - may change only scene classification / routing in `core/pipeline.py`
- `opening-detector`
  - may change only opening detection / split / prune in `core/color_detector.py`
- `facade-geometry`
  - may change only facade/socle/profile geometry in `core/pipeline.py`
- `regression-judge`
  - may change only manifests, reports, and evaluation scripts under `docs/` and `tools/eval/`

3. Merge discipline
- Do not edit the same function from two lanes in the same pass.
- Every pass must produce one canonical regression report before claiming improvement.
- If a change helps one canonical case and regresses another, it is rejected until routed by scenario.

4. Canonical first-pass coverage
- `ROMCEA` -> owner: `facade-geometry`
- `ARION` -> owner: `opening-detector`
- `MARGINEAN` -> owner: `facade-geometry`
- `MUNTEAN` -> owner: `scene-router`
