"""
DrawQuantPDF - Foundation Model Adapter
Converts segmentation model output to project-compatible DetectedRegion format.
Output: bbox/mask regions for facade, window, door (aligned with color_detector).
OPT-IN: No impact on default pipeline; used only when foundation_segmentation is active.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class RawPrediction:
    """Raw prediction from a segmentation model (agnostic format)."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in pixels
    mask: Optional[np.ndarray] = None  # binary mask, same size as crop or full image
    class_id: int = 0
    class_name: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


def _safe_area_px(mask: Optional[np.ndarray], bbox: Tuple[int, int, int, int]) -> float:
    """Compute area_px from mask or bbox."""
    if mask is not None and mask.size > 0:
        return float(np.count_nonzero(mask))
    x, y, w, h = bbox
    return float(w * h)


def _ensure_bbox_format(bbox: Union[tuple, list]) -> Tuple[int, int, int, int]:
    """Normalize bbox to (x, y, w, h) assuming input is already xywh."""
    b = tuple(bbox)
    if len(b) != 4:
        raise ValueError(f"bbox must have 4 values, got {len(b)}")
    x, y, w, h = b
    return (int(x), int(y), max(1, int(w)), max(1, int(h)))


def _xyxy_to_xywh(bbox: Union[tuple, list]) -> Tuple[int, int, int, int]:
    """Convert (x1, y1, x2, y2) to (x, y, w, h)."""
    b = tuple(bbox)
    if len(b) != 4:
        raise ValueError(f"bbox must have 4 values, got {len(b)}")
    x1, y1, x2, y2 = b
    return (int(x1), int(y1), max(1, int(x2 - x1)), max(1, int(y2 - y1)))


# Class name mapping for project region_type and output key
CLASS_TO_REGION = {
    "facade": ("facade", "facades"),
    "window": ("window", "windows"),
    "door": ("door", "doors"),
    "socle": ("socle", "socles"),
    "fatada": ("facade", "facades"),
    "fereastra": ("window", "windows"),
    "usa": ("door", "doors"),
    "soclu": ("socle", "socles"),
}


def adapt_predictions(
    predictions: List[RawPrediction],
    image_shape: Optional[Tuple[int, int]] = None,
) -> Dict[str, List["DetectedRegion"]]:
    """
    Adapt raw segmentation predictions to project format.

    Returns dict: {"facades": [...], "windows": [...], "doors": [...]}
    Each element is a DetectedRegion compatible with color_detector / pipeline.
    """
    # Lazy import to avoid circular deps; DetectedRegion lives in color_detector
    from core.color_detector import DetectedRegion

    result: Dict[str, List[DetectedRegion]] = {
        "facades": [],
        "windows": [],
        "doors": [],
        "socles": [],
    }

    counters = {"facade": 0, "window": 0, "door": 0, "socle": 0}

    for pred in predictions:
        try:
            bbox = _ensure_bbox_format(pred.bbox)
        except (ValueError, TypeError):
            continue

        name_lower = (pred.class_name or "").strip().lower()
        region_type, key = CLASS_TO_REGION.get(
            name_lower,
            (["facade", "window", "door"][pred.class_id % 3],
             ["facades", "windows", "doors"][pred.class_id % 3]),
        )

        area_px = _safe_area_px(pred.mask, bbox)
        counters[region_type] = counters.get(region_type, 0) + 1
        prefix = {"facade": "FATADA", "window": "F", "door": "U", "socle": "SOCLU"}[region_type]
        label = f"{prefix}_{counters[region_type]}"

        region = DetectedRegion(
            label=label,
            region_type=region_type,
            bbox=bbox,
            contour=np.array([]),
            area_px=area_px,
            area_m2=None,
            width_m=None,
            height_m=None,
            ocr_text="",
            parent_facade=None,
            color_detected="foundation-model",
        )
        result[key].append(region)

    return result


def adapt_from_dict_batch(
    raw_output: Dict[str, Any],
    image_shape: Optional[Tuple[int, int]] = None,
) -> Dict[str, List["DetectedRegion"]]:
    """
    Adapt from a generic dict format (e.g. model batch output).

    Expected keys: "boxes" (Nx4), "labels"/"class_names" (N,),
    optional "masks" (N,H,W), "scores" (N,), "bbox_format" in {"xyxy", "xywh"}.
    """
    boxes = raw_output.get("boxes") or raw_output.get("bboxes") or []
    labels = raw_output.get("labels") or raw_output.get("class_ids") or []
    class_names = raw_output.get("class_names") or []
    masks = raw_output.get("masks") or []
    scores = raw_output.get("scores") or []
    bbox_format = str(raw_output.get("bbox_format", "xyxy")).strip().lower()

    predictions = []
    n = len(boxes) if hasattr(boxes, "__len__") else 0
    for i in range(n):
        box = boxes[i] if i < len(boxes) else (0, 0, 1, 1)
        if hasattr(box, "__iter__"):
            box_values = tuple(box)
        else:
            box_values = (0, 0, 1, 1)
        try:
            if bbox_format == "xywh":
                norm_box = _ensure_bbox_format(box_values)
            else:
                norm_box = _xyxy_to_xywh(box_values)
        except (TypeError, ValueError):
            norm_box = (0, 0, 1, 1)
        label = labels[i] if i < len(labels) else 0
        name = class_names[i] if i < len(class_names) else ""
        mask = masks[i] if i < len(masks) else None
        score = scores[i] if i < len(scores) else 1.0
        predictions.append(RawPrediction(
            bbox=norm_box,
            mask=mask,
            class_id=int(label) if isinstance(label, (int, float)) else 0,
            class_name=str(name) if name else "",
            confidence=float(score) if score is not None else 1.0,
        ))
    return adapt_predictions(predictions, image_shape)
