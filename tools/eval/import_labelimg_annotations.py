"""Import LabelImg annotations into the YOLO training dataset.

The source folder may contain Pascal VOC XML, JSON exports, and YOLO TXT files.
We prefer XML/JSON because they store class names explicitly. TXT import remains a
fallback for legacy cases and resolves numeric IDs from the folder's classes.txt.
"""
from __future__ import annotations

import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

OUR_CLASSES = ["facade", "window", "door", "socle"]
REQUIRED_SOURCE_CLASSES = set(OUR_CLASSES)
CLASS_NAME_TO_ID = {
    "facade": 0,
    "window": 1,
    "door": 2,
    "socle": 3,
    "fereastra": 1,
    "usa": 2,
    "ușa": 2,
    "soclu": 3,
}


def _normalize_class_name(name: str) -> str:
    return (name or "").strip().lower()


def _class_id_from_name(name: str) -> int:
    key = _normalize_class_name(name)
    if key not in CLASS_NAME_TO_ID:
        raise ValueError(f"Unsupported class name: {name!r}")
    return CLASS_NAME_TO_ID[key]


def load_labelimg_to_ours(source_dir: Path) -> dict[int, int]:
    """Build a class-id remap from the source folder's classes.txt."""
    classes_path = source_dir / "classes.txt"
    if not classes_path.exists():
        raise FileNotFoundError(
            f"Missing classes.txt in source folder: {classes_path}"
        )

    raw_names = [
        _normalize_class_name(line)
        for line in classes_path.read_text(encoding="utf-8").splitlines()
    ]

    labelimg_to_ours: dict[int, int] = {}
    found_names: set[str] = set()
    for idx, class_name in enumerate(raw_names):
        if class_name in REQUIRED_SOURCE_CLASSES:
            labelimg_to_ours[idx] = _class_id_from_name(class_name)
            found_names.add(class_name)

    missing = sorted(REQUIRED_SOURCE_CLASSES - found_names)
    if missing:
        raise ValueError(
            "classes.txt does not define all required classes. "
            f"Missing: {missing}. File: {classes_path}"
        )

    return labelimg_to_ours


def _clamp01(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _bbox_to_yolo_line(class_id: int, bbox: tuple[float, float, float, float],
                       img_w: float, img_h: float) -> str | None:
    x, y, w, h = bbox
    if img_w <= 0 or img_h <= 0 or w <= 0 or h <= 0:
        return None
    cx = _clamp01((x + w / 2.0) / img_w)
    cy = _clamp01((y + h / 2.0) / img_h)
    nw = _clamp01(w / img_w, minimum=0.001)
    nh = _clamp01(h / img_h, minimum=0.001)
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def annotation_items_from_voc_xml(xml_path: Path) -> tuple[tuple[int, int], list[tuple[str, tuple[float, float, float, float]]]]:
    """Parse Pascal VOC XML and return image size plus named bbox items."""
    root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    size_node = root.find("size")
    if size_node is None:
        raise ValueError(f"Missing <size> in XML: {xml_path}")

    img_w = int(float(size_node.findtext("width", default="0")))
    img_h = int(float(size_node.findtext("height", default="0")))
    items: list[tuple[str, tuple[float, float, float, float]]] = []
    for obj in root.findall("object"):
        label = obj.findtext("name", default="").strip()
        box = obj.find("bndbox")
        if not label or box is None:
            continue
        xmin = float(box.findtext("xmin", default="0"))
        ymin = float(box.findtext("ymin", default="0"))
        xmax = float(box.findtext("xmax", default="0"))
        ymax = float(box.findtext("ymax", default="0"))
        items.append((label, (xmin, ymin, max(0.0, xmax - xmin), max(0.0, ymax - ymin))))
    return (img_w, img_h), items


def annotation_items_from_json(json_path: Path) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Parse named bbox items from the JSON export found in the source folder."""
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else [payload]
    items: list[tuple[str, tuple[float, float, float, float]]] = []
    for record in records:
        for ann in record.get("annotations", []):
            label = ann.get("label", "")
            coords = ann.get("coordinates", {})
            w = float(coords.get("width", 0.0))
            h = float(coords.get("height", 0.0))
            cx = float(coords.get("x", 0.0))
            cy = float(coords.get("y", 0.0))
            items.append((label, (cx - w / 2.0, cy - h / 2.0, w, h)))
    return items


def build_lines_from_named_items(items: list[tuple[str, tuple[float, float, float, float]]],
                                 img_w: int, img_h: int) -> list[str]:
    lines: list[str] = []
    for label, bbox in items:
        class_id = _class_id_from_name(label)
        line = _bbox_to_yolo_line(class_id, bbox, float(img_w), float(img_h))
        if line:
            lines.append(line)
    return lines


def remap_label_file_lines(src_path: Path,
                           labelimg_to_ours: dict[int, int]) -> list[str]:
    """Remap numeric LabelImg TXT labels using the resolved classes.txt mapping."""
    lines = src_path.read_text(encoding="utf-8").strip().split("\n")
    remapped: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            old_cls = int(parts[0])
        except ValueError:
            continue
        if old_cls not in labelimg_to_ours:
            continue
        new_cls = labelimg_to_ours[old_cls]
        remapped.append(f"{new_cls} {' '.join(parts[1:])}")
    return remapped


def annotation_lines_for_image(image_path: Path,
                               labelimg_to_ours: dict[int, int]) -> tuple[list[str], str]:
    """Resolve labels for one image, preferring named annotations over numeric TXT."""
    xml_path = image_path.with_suffix(".xml")
    if xml_path.exists():
        (img_w, img_h), items = annotation_items_from_voc_xml(xml_path)
        return build_lines_from_named_items(items, img_w, img_h), "xml"

    json_path = image_path.with_suffix(".json")
    if json_path.exists():
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "JSON import fallback requires opencv-python and numpy"
            ) from exc
        data = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot load image for JSON import: {image_path}")
        img_h, img_w = image.shape[:2]
        items = annotation_items_from_json(json_path)
        return build_lines_from_named_items(items, img_w, img_h), "json"

    txt_path = image_path.with_suffix(".txt")
    if txt_path.exists():
        return remap_label_file_lines(txt_path, labelimg_to_ours), "txt"

    return [], "missing"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str,
                        default=r"C:\Users\Admin\Desktop\Antranare imagini")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    source = Path(args.source)
    out_dir = ROOT / "data" / "yolo_facade"
    labelimg_to_ours = load_labelimg_to_ours(source)

    pairs = []
    for png in sorted(source.glob("*.png")):
        if any((png.with_suffix(ext)).exists() for ext in (".xml", ".json", ".txt")):
            pairs.append(png)

    print(f"Found {len(pairs)} annotated images in {source}")
    print(f"Resolved TXT fallback class remap: {labelimg_to_ours}")

    if not pairs:
        print("No annotated images found. Exiting.")
        return

    random.seed(42)
    random.shuffle(pairs)
    val_count = max(3, int(len(pairs) * args.val_ratio))
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    total_labels = 0
    class_counts = {c: 0 for c in OUR_CLASSES}
    source_counts = {"xml": 0, "json": 0, "txt": 0, "missing": 0}

    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        img_dir = out_dir / "images" / split_name
        lbl_dir = out_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for f in img_dir.glob("*"):
            f.unlink()
        for f in lbl_dir.glob("*"):
            f.unlink()

        for png in split_pairs:
            shutil.copy2(png, img_dir / png.name)
            lines, ann_source = annotation_lines_for_image(png, labelimg_to_ours)
            source_counts[ann_source] = source_counts.get(ann_source, 0) + 1
            dst_txt = lbl_dir / png.with_suffix(".txt").name
            dst_txt.write_text("\n".join(lines), encoding="utf-8")
            total_labels += len(lines)

            for line in lines:
                parts = line.strip().split()
                if parts:
                    cls = int(parts[0])
                    if 0 <= cls < len(OUR_CLASSES):
                        class_counts[OUR_CLASSES[cls]] += 1

        print(f"  {split_name}: {len(split_pairs)} images")

    (out_dir / "classes.txt").write_text("\n".join(OUR_CLASSES), encoding="utf-8")
    yaml_content = f"""# DrawQuantPDF Facade Detection Dataset
# {len(pairs)} images, {total_labels} labels

path: {out_dir.as_posix()}
train: images/train
val: images/val

nc: {len(OUR_CLASSES)}
names: {OUR_CLASSES}
"""
    (out_dir / "dataset.yaml").write_text(yaml_content, encoding="utf-8")

    print(f"\nDataset created: {out_dir}")
    print(f"  Train: {len(train_pairs)} images")
    print(f"  Val: {len(val_pairs)} images")
    print(f"  Total labels: {total_labels}")
    print(f"  Per class: {class_counts}")
    print(f"  Annotation sources used: {source_counts}")


if __name__ == "__main__":
    main()
