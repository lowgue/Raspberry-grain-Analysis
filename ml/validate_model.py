#!/usr/bin/env python3
"""Valida mAP do modelo treinado no split de teste."""
from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_WEIGHTS = os.path.join(
    PROJECT_ROOT, "models", "coffee_beans_yolov9t", "weights", "best.pt"
)
DEFAULT_DATA = os.path.join(PROJECT_ROOT, "data", "coffee_beans", "data.yaml")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--data", default=DEFAULT_DATA)
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("pip install -r requirements-ml.txt", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.weights):
        print(f"Modelo não encontrado: {args.weights}", file=sys.stderr)
        sys.exit(1)

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split="test")
    print(metrics)


if __name__ == "__main__":
    main()
