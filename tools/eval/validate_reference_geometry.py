"""Validate reference geometry JSONs for structural issues.

Checks:
- normalized coords outside the expected contour envelope
- rasterized polygon areas (degenerate on 512x512 canvas)
- self-intersecting polygons
- window/door box consistency with raw image pixel content
- image_size mismatch between reference and raw PNG (warning unless geometry/content fail)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval.reference_validation import validate_reference_file


def validate_reference(ref_path: Path, scenario: dict) -> dict:
    raw_rel = scenario.get("raw_path")
    raw_path = ROOT / raw_rel if raw_rel else None
    return validate_reference_file(ref_path, raw_path)


def main():
    parser = argparse.ArgumentParser(description="Validate reference geometry JSONs.")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "docs" / "reference_cases" / "canonical_scenarios.json")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "debug" / "reference_validation")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    scenarios = manifest.get("scenarios", [])

    all_reports = []
    for scenario in scenarios:
        ref_rel = scenario.get("reference_json_path")
        if not ref_rel:
            continue
        ref_path = ROOT / ref_rel
        if not ref_path.exists():
            continue
        report = validate_reference(ref_path, scenario)
        all_reports.append(report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_out = args.output_dir / "reference_validation_report.json"
    json_out.write_text(json.dumps(all_reports, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = ["# Reference Geometry Validation Report\n"]
    for rep in all_reports:
        md_lines.append(f"## {rep['reference_id']}")
        md_lines.append(f"**Path:** `{rep['path']}`")
        md_lines.append(f"**Verdict:** `{rep['verdict']}`\n")

        if rep["image_size_check"]:
            sc = rep["image_size_check"]
            md_lines.append(f"### Image size")
            md_lines.append(f"- Reference: {sc.get('reference_image_size')}")
            md_lines.append(f"- Raw PNG: {sc.get('raw_image_size')}")
            if sc.get('match'):
                md_lines.append("- Match: YES")
            else:
                md_lines.append(f"- Match: **NO** (scale x={sc.get('width_ratio')}, y={sc.get('height_ratio')})")
            md_lines.append("")

        if rep["bounds_violations"]:
            md_lines.append("### Normalized bounds violations")
            md_lines.append("| # | polygon | norm_x | norm_y | pixel | outside_canvas |")
            md_lines.append("|---|---------|--------|--------|-------|----------------|")
            for v in rep["bounds_violations"]:
                md_lines.append(
                    f"| {v['point_idx']} | {v['polygon'][:30]} | {v['norm_x']:.4f} | {v['norm_y']:.4f} | ({v['pixel_x']},{v['pixel_y']}) | {'YES' if v['outside_canvas'] else 'no'} |"
                )
            md_lines.append("")

        if rep["raster_checks"]:
            md_lines.append("### Raster area checks")
            for r in rep["raster_checks"]:
                status = "**DEGENERATE**" if r["degenerate"] else "OK"
                md_lines.append(f"- `{r['polygon'][:40]}`: area={r['raster_area_px']}px ({r['area_fraction']*100:.2f}%) — {status}")
            md_lines.append("")

        if rep["box_content_checks"]:
            md_lines.append("### Opening boxes vs raw image content")
            md_lines.append("| type | idx | scaled_box | mean_gray | dark_frac | std | edge | jamb | contrast | verdict |")
            md_lines.append("|------|-----|------------|-----------|-----------|-----|------|------|----------|---------|")
            for c in rep["box_content_checks"]:
                verdict = "**BRIGHT WALL**" if c["bright_wall"] else ("opening" if c["likely_opening"] else "ambiguous")
                md_lines.append(
                    f"| {c['box_type']} | {c['box_idx']} | {c['scaled_box']} | {c['mean_gray']} | {c['dark_frac']} | {c.get('std_gray')} | {c.get('edge_density')} | {c.get('jamb_score')} | {c.get('context_contrast')} | {verdict} |"
                )
            md_lines.append("")

        if rep["issues"]:
            md_lines.append("### Issues summary")
            for issue in rep["issues"]:
                md_lines.append(f"- {issue}")
            md_lines.append("")
        md_lines.append("---\n")

    md_out = args.output_dir / "reference_validation_report.md"
    md_out.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Validation report: {json_out}")
    print(f"Markdown report: {md_out}")
    for rep in all_reports:
        status = rep['verdict']
        issues = "; ".join(rep["issues"]) if rep["issues"] else "none"
        print(f"  {rep['reference_id']}: {status} — {issues}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
