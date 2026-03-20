"""
DrawQuantPDF - Metrics computation for evaluation.
IoU facade, precision/recall windows, area error.
"""

from dataclasses import dataclass, field
from typing import Optional


def bbox_iou(a: tuple, b: tuple) -> float:
    """Compute IoU of two bboxes (x, y, w, h). Returns 0 if invalid."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def bbox_center_distance(a: tuple, b: tuple) -> float:
    """Euclidean distance between bbox centers."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    cx_a = ax + aw / 2
    cy_a = ay + ah / 2
    cx_b = bx + bw / 2
    cy_b = by + bh / 2
    return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5


@dataclass
class EvalResult:
    """Aggregated evaluation metrics."""

    # Facade
    facade_iou_mean: float = 0.0
    facade_iou_per_item: list = field(default_factory=list)

    # Windows: precision, recall, F1
    window_precision: float = 0.0
    window_recall: float = 0.0
    window_f1: float = 0.0
    window_tp: int = 0
    window_fp: int = 0
    window_fn: int = 0

    # Area error (m²): mean absolute error, max error
    area_mae: float = 0.0
    area_max_error: float = 0.0
    area_errors: list = field(default_factory=list)

    # Metadata
    sample_id: str = ""
    num_facades_gt: int = 0
    num_facades_pred: int = 0
    num_windows_gt: int = 0
    num_windows_pred: int = 0

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "facade_iou_mean": round(self.facade_iou_mean, 4),
            "facade_iou_per_item": [round(x, 4) for x in self.facade_iou_per_item],
            "window_precision": round(self.window_precision, 4),
            "window_recall": round(self.window_recall, 4),
            "window_f1": round(self.window_f1, 4),
            "window_tp": self.window_tp,
            "window_fp": self.window_fp,
            "window_fn": self.window_fn,
            "num_windows_gt": self.num_windows_gt,
            "num_windows_pred": self.num_windows_pred,
            "area_mae_m2": round(self.area_mae, 4),
            "area_max_error_m2": round(self.area_max_error, 4),
        }


def compute_facade_iou(
    pred_facades: list,
    gt_facades: list,
    iou_threshold: float = 0.1,
) -> tuple[float, list]:
    """
    Match predicted facades to GT by best IoU. Returns (mean_iou, per_item_ious).
    pred_facades / gt_facades: list of objects with .bbox (x,y,w,h)
    """
    if not gt_facades:
        return (1.0 if not pred_facades else 0.0, [])

    used_gt = [False] * len(gt_facades)
    ious = []

    for pf in pred_facades:
        pbox = getattr(pf, "bbox", None) or getattr(pf, "region_bbox", None)
        if not pbox:
            continue
        best_iou = 0.0
        best_idx = -1
        for gi, gf in enumerate(gt_facades):
            if used_gt[gi]:
                continue
            gbox = getattr(gf, "bbox", None)
            if not gbox:
                continue
            iou = bbox_iou(pbox, gbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = gi
        if best_idx >= 0 and best_iou >= iou_threshold:
            used_gt[best_idx] = True
            ious.append(best_iou)
        # Unmatched pred -> IoU 0 (penalize FP)
        elif best_idx < 0 or best_iou < iou_threshold:
            ious.append(0.0)

    # Unmatched GT -> IoU 0
    for u in used_gt:
        if not u:
            ious.append(0.0)

    mean_iou = sum(ious) / len(ious) if ious else 0.0
    return (mean_iou, ious)


def compute_window_precision_recall(
    pred_windows: list,
    gt_windows: list,
    match_distance_px: float = 80.0,
) -> tuple[float, float, float, int, int, int]:
    """
    Match windows by center distance. Returns (precision, recall, f1, tp, fp, fn).
    pred/gt: list of objects with .bbox and optionally .position or .center
    """
    if not gt_windows and not pred_windows:
        return (1.0, 1.0, 1.0, 0, 0, 0)
    if not gt_windows:
        return (0.0, 1.0, 0.0, 0, len(pred_windows), 0)
    if not pred_windows:
        return (1.0, 0.0, 0.0, 0, 0, len(gt_windows))

    def get_box(obj):
        b = getattr(obj, "bbox", None)
        if b:
            return b
        p = getattr(obj, "position", None) or getattr(obj, "center", None)
        if p:
            return (p[0] - 10, p[1] - 10, 20, 20)
        return None

    used_gt = [False] * len(gt_windows)
    tp = 0

    for pw in pred_windows:
        pbox = get_box(pw)
        if not pbox:
            continue
        best_d = float("inf")
        best_gi = -1
        for gi, gw in enumerate(gt_windows):
            if used_gt[gi]:
                continue
            gbox = get_box(gw)
            if not gbox:
                continue
            d = bbox_center_distance(pbox, gbox)
            if d < best_d:
                best_d = d
                best_gi = gi
        if best_gi >= 0 and best_d <= match_distance_px:
            used_gt[best_gi] = True
            tp += 1

    fp = len(pred_windows) - tp
    fn = sum(1 for u in used_gt if not u)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return (precision, recall, f1, tp, fp, fn)


def compute_area_errors(
    pred_windows: list,
    gt_windows: list,
    pred_facades: list,
    gt_facades: list,
    match_distance_px: float = 80.0,
) -> tuple[float, float, list]:
    """
    Pair pred with GT by bbox proximity, compute |pred_area - gt_area|.
    Returns (mae, max_error, errors_list).
    pred/gt: objects with .bbox and .area_m2 or .area or total_area_m2
    """
    errors = []

    def get_area(obj, is_facade=False):
        a = getattr(obj, "area_m2", None) or getattr(obj, "area", None)
        if a is not None:
            return float(a)
        if is_facade:
            return float(getattr(obj, "total_area_m2", 0) or getattr(obj, "total_area", 0))
        return 0.0

    def get_box(obj):
        b = getattr(obj, "bbox", None) or getattr(obj, "region_bbox", None)
        if b:
            return b
        p = getattr(obj, "position", None) or getattr(obj, "center", None)
        if p:
            return (p[0] - 10, p[1] - 10, 20, 20)
        return None

    # Windows
    used_gt_w = [False] * len(gt_windows)
    for pw in pred_windows:
        pbox = get_box(pw)
        if not pbox:
            continue
        best_d = float("inf")
        best_gw = None
        for gi, gw in enumerate(gt_windows):
            if used_gt_w[gi]:
                continue
            gbox = get_box(gw)
            if not gbox:
                continue
            d = bbox_center_distance(pbox, gbox)
            if d < best_d and d <= match_distance_px:
                best_d = d
                best_gw = gw
        if best_gw is not None:
            for gi, gw in enumerate(gt_windows):
                if gw is best_gw:
                    used_gt_w[gi] = True
                    break
            pa = get_area(pw)
            ga = get_area(best_gw)
            if ga > 0 or pa > 0:
                err = abs(pa - ga)
                errors.append(err)

    # Facades
    used_gt_f = [False] * len(gt_facades)
    for pf in pred_facades:
        pbox = get_box(pf)
        if not pbox:
            continue
        best_d = float("inf")
        best_gf = None
        for gi, gf in enumerate(gt_facades):
            if used_gt_f[gi]:
                continue
            gbox = get_box(gf)
            if not gbox:
                continue
            d = bbox_center_distance(pbox, gbox)
            if d < best_d and d <= match_distance_px * 2:  # facades larger
                best_d = d
                best_gf = gf
        if best_gf is not None:
            for gi, gf in enumerate(gt_facades):
                if gf is best_gf:
                    used_gt_f[gi] = True
                    break
            pa = get_area(pf, is_facade=True)
            ga = get_area(best_gf, is_facade=True)
            if ga > 0 or pa > 0:
                err = abs(pa - ga)
                errors.append(err)

    mae = sum(errors) / len(errors) if errors else 0.0
    max_err = max(errors) if errors else 0.0
    return (mae, max_err, errors)


def evaluate_sample(
    pred_facades: list,
    pred_windows: list,
    gt_facades: list,
    gt_windows: list,
    sample_id: str = "",
) -> EvalResult:
    """Full evaluation of one sample."""
    r = EvalResult(sample_id=sample_id)
    r.num_facades_gt = len(gt_facades)
    r.num_facades_pred = len(pred_facades)
    r.num_windows_gt = len(gt_windows)
    r.num_windows_pred = len(pred_windows)

    r.facade_iou_mean, r.facade_iou_per_item = compute_facade_iou(
        pred_facades, gt_facades
    )
    (
        r.window_precision,
        r.window_recall,
        r.window_f1,
        r.window_tp,
        r.window_fp,
        r.window_fn,
    ) = compute_window_precision_recall(pred_windows, gt_windows)
    r.area_mae, r.area_max_error, r.area_errors = compute_area_errors(
        pred_windows, gt_windows, pred_facades, gt_facades
    )
    return r
