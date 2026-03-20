"""
World-file scale helpers (PGW/PNGW/JGW/GFW/WLD).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def _parse_float(raw: str) -> Optional[float]:
    txt = (raw or "").strip().replace(",", ".")
    if not txt:
        return None
    try:
        return float(txt)
    except Exception:
        return None


def _candidate_worldfiles(image_path: Path) -> list[Path]:
    stem = image_path.with_suffix("")
    ext = image_path.suffix.lower().lstrip(".")
    candidates: list[Path] = []

    if ext:
        if len(ext) >= 2:
            candidates.append(stem.with_suffix(f".{ext[0]}{ext[-1]}w"))
        candidates.append(stem.with_suffix(f".{ext}w"))

    candidates.append(stem.with_suffix(".wld"))
    return candidates


def read_worldfile_px_per_m(image_path: str) -> Optional[dict]:
    """
    Read sidecar world file and return scale in px/m.

    Returns dict with keys:
      - path
      - px_per_m
      - meters_per_pixel_x
      - meters_per_pixel_y
      - axis_scale_diff_ratio
    Returns None when no valid world file exists.
    """
    img_path = Path(image_path)
    for wf in _candidate_worldfiles(img_path):
        if not wf.exists() or not wf.is_file():
            continue

        lines = wf.read_text(encoding="utf-8", errors="ignore").splitlines()
        if len(lines) < 6:
            continue

        # World file convention:
        # 1: A pixel size in x direction
        # 4: E pixel size in y direction (usually negative)
        a = _parse_float(lines[0])
        e = _parse_float(lines[3])
        if a is None or e is None:
            continue

        mpp_x = abs(float(a))
        mpp_y = abs(float(e))
        if mpp_x <= 0 and mpp_y <= 0:
            continue

        if mpp_x > 0 and mpp_y > 0:
            mpp = (mpp_x + mpp_y) / 2.0
            diff_ratio = abs(mpp_x - mpp_y) / max(mpp_x, mpp_y)
        else:
            mpp = mpp_x if mpp_x > 0 else mpp_y
            diff_ratio = 0.0

        if mpp <= 0:
            continue

        px_per_m = 1.0 / mpp
        if px_per_m <= 0:
            continue

        return {
            "path": str(wf),
            "px_per_m": float(px_per_m),
            "meters_per_pixel_x": float(mpp_x),
            "meters_per_pixel_y": float(mpp_y),
            "axis_scale_diff_ratio": float(diff_ratio),
        }

    return None
