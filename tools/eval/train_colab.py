"""
DrawQuantPDF — YOLOv8 Training Script for Google Colab GPU

USAGE (in Colab):
1. Upload the entire data/yolo_facade/ folder to Colab (or mount Google Drive)
2. Run this script:

    !pip install ultralytics
    !python train_colab.py --data /content/yolo_facade/dataset.yaml --device 0

Or locally on GPU:
    python tools/eval/train_colab.py --data data/yolo_facade/dataset.yaml --device 0

On CPU (slow, ~12h):
    python tools/eval/train_colab.py --data data/yolo_facade/dataset.yaml --device cpu
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 facade detector")
    parser.add_argument("--data", type=str, default="data/yolo_facade/dataset.yaml")
    parser.add_argument("--device", type=str, default="0", help="0 for GPU, cpu for CPU")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--model-base", type=str, default="yolov8n.pt")
    parser.add_argument("--name", type=str, default="yolov8n_facade_v4")
    args = parser.parse_args()

    from ultralytics import YOLO

    data_path = str(Path(args.data).resolve())
    print(f"Dataset: {data_path}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch}, ImgSz: {args.imgsz}")

    model = YOLO(args.model_base)
    results = model.train(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project="runs/facade_detect",
        name=args.name,
        exist_ok=True,
        verbose=True,
        device=args.device,
        augment=True,
        mosaic=0.5,
        flipud=0.3,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.3,
    )

    # Save best model
    save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else Path(f"runs/facade_detect/{args.name}")
    best_src = save_dir / "weights" / "best.pt"

    # Try alternative path
    if not best_src.exists():
        best_src = Path(f"runs/detect/runs/facade_detect/{args.name}/weights/best.pt")

    if best_src.exists():
        out_name = f"models/{args.name}_best.pt"
        Path("models").mkdir(exist_ok=True)
        shutil.copy2(best_src, out_name)
        print(f"\nSAVED: {out_name}")
    else:
        print(f"\nWARNING: best.pt not found at expected paths")

    print("TRAINING_COMPLETE")


if __name__ == "__main__":
    main()
