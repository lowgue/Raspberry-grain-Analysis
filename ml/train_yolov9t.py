#!/usr/bin/env python3
"""
Treina YOLOv9t no Coffee Bean Dataset.

Uso:
  python ml/download_dataset.py   # uma vez
  python ml/train_yolov9t.py
  python ml/train_yolov9t.py --epochs 50 --imgsz 416 --device cpu
"""
from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_DATA = os.path.join(PROJECT_ROOT, "data", "coffee_beans", "data.yaml")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "models", "coffee_beans_yolov9t")
DEFAULT_PRETRAINED = os.path.join(PROJECT_ROOT, "models", "pretrained", "yolov9t.pt")


def parse_args():
    p = argparse.ArgumentParser(description="Treino YOLOv9t — Coffee Bean Dataset")
    p.add_argument("--data", default=DEFAULT_DATA, help="Caminho do data.yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="", help="cuda, cpu ou vazio (auto)")
    p.add_argument("--project", default=os.path.join(PROJECT_ROOT, "runs", "detect"))
    p.add_argument("--name", default="coffee_beans_yolov9t")
    p.add_argument("--export-onnx", action="store_true", help="Exporta best.pt para ONNX após treino")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.data):
        print(
            f"Dataset não encontrado: {args.data}\n"
            "Execute primeiro: python ml/download_dataset.py",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "Ultralytics não instalado. Execute:\n"
            "  pip install -r requirements-ml.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(DEFAULT_OUTPUT, exist_ok=True)

    weights = DEFAULT_PRETRAINED if os.path.isfile(DEFAULT_PRETRAINED) else "yolov9t.pt"
    print(f"Carregando YOLOv9t: {weights}")
    model = YOLO(weights)

    train_kw = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )
    if args.device:
        train_kw["device"] = args.device

    print(f"Treinando com {args.data} ...")
    results = model.train(**train_kw)

    run_dir = getattr(results, "save_dir", None) or os.path.join(
        args.project, args.name
    )
    best_pt = os.path.join(run_dir, "weights", "best.pt")
    if not os.path.isfile(best_pt):
        print(f"Aviso: best.pt não encontrado em {run_dir}", file=sys.stderr)
        sys.exit(1)

    weights_dir = os.path.join(DEFAULT_OUTPUT, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    dest_pt = os.path.join(weights_dir, "best.pt")
    import shutil

    shutil.copy2(best_pt, dest_pt)
    print(f"Modelo copiado para: {dest_pt}")

    if args.export_onnx:
        from ml.export_model import export_onnx

        export_onnx(dest_pt)

    print("\nTreino concluído. Para usar no servidor:")
    print(f"  export YOLO_MODEL_PATH={dest_pt}")
    print("  python main.py")


if __name__ == "__main__":
    main()
