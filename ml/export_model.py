#!/usr/bin/env python3
"""Exporta weights YOLOv9t para ONNX (inferência leve no Raspberry Pi)."""
from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PT = os.path.join(
    PROJECT_ROOT, "models", "coffee_beans_yolov9t", "weights", "best.pt"
)


def export_onnx(weights_path: str, imgsz: int = 640) -> str:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("pip install -r requirements-ml.txt", file=sys.stderr)
        sys.exit(1)

    model = YOLO(weights_path)
    out = model.export(format="onnx", imgsz=imgsz, simplify=True)
    print(f"ONNX exportado: {out}")
    return str(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=DEFAULT_PT)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    if not os.path.isfile(args.weights):
        print(f"Pesos não encontrados: {args.weights}", file=sys.stderr)
        sys.exit(1)

    export_onnx(args.weights, args.imgsz)


if __name__ == "__main__":
    main()
