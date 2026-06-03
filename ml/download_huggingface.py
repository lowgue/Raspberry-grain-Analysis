#!/usr/bin/env python3
"""
Baixa Coffee Bean Dataset do Hugging Face e converte LabelMe -> YOLO.

Dataset: SamruddhK/coffee-bean-grading-dataset
Classes: defect (CGD), premium (CGA/CGB/CGC)
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "coffee_beans"
HF_REPO = "SamruddhK/coffee-bean-grading-dataset"
HF_CACHE = PROJECT_ROOT / "data" / "hf_coffee_bean_raw"

# Pastas de grau -> classe YOLO (alinhado ao CoffeeBeansGradingV3)
GRADE_TO_CLASS = {
    "CGD": "defect",
    "CGA": "premium",
    "CGB": "premium",
    "CGC": "premium",
}
CLASS_IDS = {"defect": 0, "premium": 1}


def _polygon_to_bbox(points: list, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_c = (x_min + x_max) / 2 / img_w
    y_c = (y_min + y_max) / 2 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return x_c, y_c, w, h


def _convert_labelme_json(json_path: Path, class_name: str) -> list[str]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    img_w = data.get("imageWidth") or 0
    img_h = data.get("imageHeight") or 0
    if not img_w or not img_h:
        return []

    cls_id = CLASS_IDS[class_name]
    lines = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "polygon":
            continue
        points = shape.get("points", [])
        if len(points) < 3:
            continue
        x_c, y_c, w, h = _polygon_to_bbox(points, img_w, img_h)
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
    return lines


def _ensure_hf_deps():
    try:
        import huggingface_hub  # noqa: F401
        return
    except ImportError:
        print(
            "Erro: pacote 'huggingface_hub' não encontrado neste Python.\n\n"
            "Use o ambiente virtual do projeto:\n"
            "  cd Raspberry-grain-Analysis\n"
            "  source .venv/bin/activate\n"
            "  pip install -r requirements-ml.txt\n"
            "  python ml/download_dataset.py --source huggingface\n\n"
            "Ou em uma linha:\n"
            "  .venv/bin/pip install huggingface_hub\n"
            "  .venv/bin/python ml/download_dataset.py --source huggingface",
            file=sys.stderr,
        )
        sys.exit(1)


def download_hf_raw() -> Path:
    _ensure_hf_deps()
    from huggingface_hub import snapshot_download

    print(f"Baixando {HF_REPO} ... (pode levar vários minutos)", flush=True)
    path = snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=str(HF_CACHE),
        local_dir_use_symlinks=False,
    )
    return Path(path)


def build_yolo_dataset(raw_dir: Path, val_ratio: float = 0.2, seed: int = 42) -> Path:
    random.seed(seed)
    pairs: list[tuple[Path, Path, str]] = []

    for grade_dir, class_name in GRADE_TO_CLASS.items():
        grade_path = raw_dir / grade_dir
        img_dir = grade_path / "images"
        json_dir = grade_path / "json"
        if not img_dir.is_dir():
            continue
        for json_path in json_dir.glob("*.json"):
            stem = json_path.stem
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
                candidate = img_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                continue
            lines = _convert_labelme_json(json_path, class_name)
            if lines:
                pairs.append((img_path, json_path, class_name))

    if not pairs:
        raise RuntimeError(f"Nenhuma anotação encontrada em {raw_dir}")

    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True)

    for split_name, split_pairs in (("train", train_pairs), ("valid", val_pairs)):
        img_out = DATA_DIR / split_name / "images"
        lbl_out = DATA_DIR / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, json_path, class_name in split_pairs:
            dest_img = img_out / img_path.name
            shutil.copy2(img_path, dest_img)
            lines = _convert_labelme_json(json_path, class_name)
            dest_lbl = lbl_out / f"{img_path.stem}.txt"
            dest_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")

    yaml_path = DATA_DIR / "data.yaml"
    yaml_path.write_text(
        f"""# Coffee Bean Dataset (Hugging Face -> YOLO)
path: {DATA_DIR}
train: train/images
val: valid/images

nc: 2
names:
  0: defect
  1: premium
""",
        encoding="utf-8",
    )

    print(f"Imagens: {len(pairs)} (train={len(train_pairs)}, val={len(val_pairs)})")
    print(f"Dataset YOLO: {DATA_DIR}")
    print(f"Config: {yaml_path}")
    return yaml_path


def main():
    raw = download_hf_raw()
    yaml_path = build_yolo_dataset(raw)
    print("\nPróximo passo:")
    print(f"  python ml/train_yolov9t.py --data {yaml_path}")


if __name__ == "__main__":
    main()
