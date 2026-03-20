# Execution Plan - 2026-03-11

## Objective

Stop optimizing by isolated screenshots. Improve the detector by scenario class with explicit regression gates.

## Current Assessment

1. `ROMCEA`
- Windows and door are close to target.
- Main remaining issues: facade upper-right contour, noise on right edge, socle left extension and terrain-following lower edge.

2. `ARION`
- Some progress, but still far from target.
- Main remaining issues: composite facade contour, missing/incorrect second door cluster, incomplete adjacent openings recovery.

3. `MARGINEAN`
- False openings are mostly suppressed.
- Main remaining issue: facade contour should cover the full wall body, not stop at the bright/white transition.

4. `MUNTEAN`
- False door and false sidelight are suppressed.
- Main remaining issue: gable + right attachment contour should match the real facade silhouette.

## Workstreams

### Workstream A - Scene Router

Goal:
- Route images to the right scenario before geometry or opening refinement.

Current focus:
- stabilize `composite_stepped_facade` vs `flat_long_facade`
- stabilize `gable_facade` vs `sparse_openings_flat`

Acceptance:
- `ROMCEA` -> `flat_long_facade`
- `ARION` -> `composite_stepped_facade`
- `MARGINEAN` -> `gable_facade`
- `MUNTEAN` -> `sparse_openings_flat` or `gable-with-attachment` if added later

### Workstream B - Opening Detector

Goal:
- Detect windows and doors by scenario, not by one shared fallback path.

Current focus:
- `ROMCEA`: keep the current central split stable
- `ARION`: recover `double brown door` + `door with sidelights`
- `MARGINEAN`: keep zero false openings
- `MUNTEAN`: keep the one real window only

Acceptance:
- no new false door/window on `MARGINEAN`
- no fake door+sidelight pair on `MUNTEAN`
- `ARION` must add the missing door cluster without collapsing windows into one giant door

### Workstream C - Facade Geometry

Goal:
- Build facade and socle contours from structural lines, not from residual noise.

Current focus:
- `ROMCEA`: upper-right eave-following contour and right edge cutoff
- `ARION`: stepped roofline and shared lower cutoff above socle
- `MARGINEAN`: full pentagonal body, not truncated by bright/white region
- `MUNTEAN`: right attachment included correctly in final silhouette

Acceptance:
- facade lower cutoff aligns with socle top
- socle drip profile remains a line segment
- no facade edge should follow isolated point-cloud noise outside the wall body

## No More Ping-Pong Rule

Every iteration must produce all of these, not just a new screenshot:
- scenario class touched
- exact code area changed
- expected gain
- quick regression result on the 4 canonical cases
- residual failure list

## What To Pull From Context7

Use official OpenCV docs only for these patterns:
- `connectedComponentsWithStats` for component filtering and structure-preserving pruning
- `morphologyEx` open/close for mask cleanup
- `approxPolyDP` and contour simplification for regularized facade polygons
- `HoughLinesP` for roofline/eave/drip-profile extraction
- `distanceTransform` + `watershed` only if component separation inside dark clusters becomes necessary

## Recommended Next 3 Steps

1. Add a dedicated `composite-openings` route for `ARION`
- Separate lower-left door+sidelights from middle single door.

2. Add `gable-body completion`
- Extend facade contour through bright/white gaps using edge-consistent side walls and roof slopes.

3. Add a tiny regression script for the 4 canonical cases
- Output: scene type, element count, facade bbox, socle bbox, and warnings.
