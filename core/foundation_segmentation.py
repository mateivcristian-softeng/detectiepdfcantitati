"""
DrawQuantPDF - Foundation Segmentation (YOLOv8)
Detects facades, windows, doors, and socles using a fine-tuned YOLOv8 model.
OPT-IN: Activated via environment variable or explicit call. No impact on
default pipeline when not activated.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import numpy as np

_FOUNDATION_AVAILABLE = False
_FOUNDATION_ERROR: Optional[str] = None
_MODEL = None
_MODEL_PATH: Optional[str] = None

# Default model path (relative to project root)
_DEFAULT_MODEL = str(Path(__file__).resolve().parent.parent / "models" / "yolov8n_facade_v4_best.pt")

# Class names must match training order
CLASS_NAMES = ["facade", "window", "door", "socle"]

try:
    from ultralytics import YOLO  # noqa: F401
    import numpy as np  # noqa: F401
    _FOUNDATION_AVAILABLE = True
except ImportError as e:
    _FOUNDATION_ERROR = (
        f"Foundation segmentation requires ultralytics: {e}. "
        "Install with: pip install ultralytics"
    )


def is_available() -> bool:
    """Return True if foundation segmentation backend is usable."""
    if not _FOUNDATION_AVAILABLE:
        return False
    # Also check that the model file exists
    model_path = os.environ.get("DQP_YOLO_MODEL", _DEFAULT_MODEL)
    return Path(model_path).exists()


def get_availability_error() -> Optional[str]:
    """Return error message when backend is not available."""
    if _FOUNDATION_ERROR:
        return _FOUNDATION_ERROR
    model_path = os.environ.get("DQP_YOLO_MODEL", _DEFAULT_MODEL)
    if not Path(model_path).exists():
        return f"Model file not found: {model_path}"
    return None


def _get_model():
    """Lazy-load the YOLO model (singleton)."""
    global _MODEL, _MODEL_PATH
    from ultralytics import YOLO

    model_path = os.environ.get("DQP_YOLO_MODEL", _DEFAULT_MODEL)
    if _MODEL is None or _MODEL_PATH != model_path:
        _MODEL = YOLO(model_path)
        _MODEL_PATH = model_path
    return _MODEL


def segment_image(
    image: "np.ndarray",
    model_id: Optional[str] = None,
    conf: float = 0.40,
    imgsz: int = 2048,
    device: str = "cpu",
    **kwargs: Any,
) -> Dict[str, list]:
    """
    Run YOLOv8 detection on a facade image.

    Args:
        image: BGR numpy array (H, W, 3).
        model_id: Optional path to a specific model file.
        conf: Confidence threshold (default 0.40).
        imgsz: Inference image size (default 2048 for high detail).
        device: 'cpu' or '0' for GPU.
        **kwargs: Additional args passed to model.predict().

    Returns:
        Dict with "facades", "windows", "doors", "socles" lists of
        DetectedRegion objects.
    """
    if not _FOUNDATION_AVAILABLE:
        return {"facades": [], "windows": [], "doors": [], "socles": []}

    from core.foundation_adapter import RawPrediction, adapt_predictions

    # Override model path if provided
    if model_id:
        from ultralytics import YOLO
        model = YOLO(model_id)
    else:
        model = _get_model()

    # Detect CUDA availability for auto device selection
    if device == "auto":
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    # Run inference
    results = model.predict(
        image, conf=conf, imgsz=imgsz, verbose=False, device=device, **kwargs
    )

    # Convert YOLO output to RawPrediction list
    predictions: List[RawPrediction] = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox_xywh = (int(x1), int(y1), max(1, int(x2 - x1)), max(1, int(y2 - y1)))

            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"

            predictions.append(RawPrediction(
                bbox=bbox_xywh,
                class_id=cls_id,
                class_name=cls_name,
                confidence=confidence,
            ))

    # Adapt to DetectedRegion format
    adapted = adapt_predictions(predictions, image.shape[:2])

    # Add socles (not in original adapter CLASS_TO_REGION)
    socle_regions = []
    from core.color_detector import DetectedRegion
    import numpy as np
    socle_idx = 0
    for pred in predictions:
        if pred.class_name == "socle":
            socle_idx += 1
            socle_regions.append(DetectedRegion(
                label=f"SOCLU_ML_{socle_idx}",
                region_type="socle",
                bbox=pred.bbox,
                contour=np.array([]),
                area_px=float(pred.bbox[2] * pred.bbox[3]),
                color_detected="ml-yolov8",
            ))
    adapted["socles"] = socle_regions

    # Tag all regions with ml source
    for key in ["facades", "windows", "doors"]:
        for region in adapted.get(key, []):
            region.color_detected = "ml-yolov8"

    return adapted


def run_foundation_segmentation(
    image: "np.ndarray",
    model_id: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, list]:
    """
    Public entry point: run YOLOv8 facade detection.

    Usage:
        from core.foundation_segmentation import run_foundation_segmentation, is_available
        if is_available():
            results = run_foundation_segmentation(image)
    """
    return segment_image(image, model_id=model_id, **kwargs)
