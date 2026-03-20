# ROMCEA Spate Reference

This is the canonical target case for the flat-facade photo pipeline.

## Source Inputs
- Raw image: `detector cantitati/raw/romcea_raw/ROMCEA MARIA_colorized_fatada_spate.png`
- World file: `detector cantitati/raw/romcea_raw/ROMCEA MARIA_colorized_fatada_spate.pgw`

## Canonical User Reference
- Reference date: `2026-03-11`
- Source: user-provided clean annotated screenshot in chat
- Reserved asset path for the exact screenshot: `assets/reference/romcea_spate_expected_reference_chat_2026-03-11.png`
- Note: the exact binary screenshot attached in chat is not directly exportable from this environment, so this document and the companion JSON are the canonical persisted reference until the exact PNG is placed at the reserved path.

## Expected Interpretation

### Facade
- One single flat facade body.
- Upper contour must follow the eave line, without including roof noise.
- Left and right margins must stay inside the actual wall body and avoid exterior noise columns.
- Lower facade boundary must stop at the top line of the socle, not continue to the ground.
- Target net facade area: `24.81 m²`.

### Socle
- One socle surface band under the facade.
- The socle should reach the left bottom edge of the building footprint.
- The lower edge is slightly oblique.
- Target socle area: `5.24 m²`.

### Socle Drip Profile
- This is not a surface.
- It must be modeled and displayed as a line segment / linear profile measured in meters.
- It follows the top edge of the socle.

### Openings
Expected openings on this facade:
- Left window: `1.80 m²`
- Left sidelight near door: `1.15 m²`
- Central door: `2.88 m²`
- Right sidelight near door: `1.15 m²`
- Right window: `1.95 m²`

## Geometry Priorities
1. Preserve the central `window + door + window` separation.
2. Do not merge either sidelight into the door.
3. Do not inflate the door width into the sidelights.
4. Do not lose the left sidelight because of roof/eave noise.
5. Keep the right standalone window independent from the central cluster.

## Practical Use
When changing code in `core/pipeline.py` or `core/color_detector.py`, compare against this reference before accepting the patch.
