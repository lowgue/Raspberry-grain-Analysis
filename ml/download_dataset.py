#!/usr/bin/env python3
"""
Baixa o Coffee Bean Dataset em formato YOLO.

Fontes:
  - roboflow (padrão): CoffeeBeansGradingV3 — requer ROBOFLOW_API_KEY
  - huggingface (fallback): SamruddhK/coffee-bean-grading-dataset

Uso:
  python ml/download_dataset.py
  python ml/download_dataset.py --source huggingface
  export ROBOFLOW_API_KEY=seu_token && python ml/download_dataset.py --source roboflow
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "coffee_beans")

# CoffeeBeansGradingV3 — defect / premium (compatível com healthy/damaged)
ROBOFLOW_WORKSPACE = "jiingyi"
ROBOFLOW_PROJECT = "coffeebeansgradingv3"
ROBOFLOW_VERSION = 1


def _patch_data_yaml(download_dir: str) -> str:
    """Garante data.yaml com paths relativos corretos para o treino."""
    src_yaml = os.path.join(download_dir, "data.yaml")
    if not os.path.exists(src_yaml):
        for root, _, files in os.walk(download_dir):
            if "data.yaml" in files:
                src_yaml = os.path.join(root, "data.yaml")
                break
        else:
            raise FileNotFoundError(f"data.yaml não encontrado em {download_dir}")

    dest = os.path.join(DATA_DIR, "data.yaml")
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(src_yaml, encoding="utf-8") as f:
        content = f.read()

    # Normaliza path raiz para o diretório de destino
    content = content.replace("path: ../datasets/", f"path: {DATA_DIR}\n# ")
    if "path:" not in content.split("\n")[0:5]:
        header = f"path: {DATA_DIR}\n"
        content = header + content

    with open(dest, "w", encoding="utf-8") as f:
        f.write(content)

    # Copia splits se o download ficou em subpasta
    for split in ("train", "valid", "test"):
        for base in (download_dir, os.path.dirname(download_dir)):
            src_split = os.path.join(base, split)
            if os.path.isdir(src_split):
                dest_split = os.path.join(DATA_DIR, split)
                if os.path.abspath(src_split) != os.path.abspath(dest_split):
                    if os.path.exists(dest_split):
                        shutil.rmtree(dest_split)
                    shutil.copytree(src_split, dest_split)
                break

    return dest


def download_with_roboflow() -> str:
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        print(
            "Erro: defina ROBOFLOW_API_KEY.\n"
            "  export ROBOFLOW_API_KEY=<sua_chave>\n"
            "  https://app.roboflow.com/settings/api",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("Instale: pip install roboflow", file=sys.stderr)
        sys.exit(1)

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Baixando {ROBOFLOW_PROJECT} (YOLOv9)...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)
    dataset = version.download("yolov9", location=DATA_DIR)

    download_path = getattr(dataset, "location", None) or str(dataset)
    yaml_path = _patch_data_yaml(download_path if os.path.isdir(download_path) else DATA_DIR)
    print(f"Dataset salvo em: {DATA_DIR}")
    print(f"Config: {yaml_path}")
    return yaml_path


def _load_dotenv():
    env_file = Path(PROJECT_ROOT) / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _check_python_hint():
    venv_python = Path(PROJECT_ROOT) / ".venv" / "bin" / "python"
    if venv_python.is_file() and Path(sys.executable).resolve() != venv_python.resolve():
        print(
            "Aviso: você não está usando o Python do .venv.\n"
            f"  Atual:  {sys.executable}\n"
            f"  Use:    source .venv/bin/activate\n"
            "  ou:     .venv/bin/python ml/download_dataset.py --source huggingface\n",
            file=sys.stderr,
        )


def main():
    _load_dotenv()
    _check_python_hint()
    p = argparse.ArgumentParser(description="Download Coffee Bean Dataset")
    p.add_argument(
        "--source",
        choices=("auto", "roboflow", "huggingface"),
        default="auto",
        help="auto: Roboflow se ROBOFLOW_API_KEY existir, senão Hugging Face",
    )
    args = p.parse_args()

    source = args.source
    if source == "auto":
        source = "roboflow" if os.environ.get("ROBOFLOW_API_KEY", "").strip() else "huggingface"

    if source == "roboflow":
        yaml_path = download_with_roboflow()
    else:
        from ml.download_huggingface import build_yolo_dataset, download_hf_raw

        raw = download_hf_raw()
        yaml_path = str(build_yolo_dataset(raw))

    print("\nPróximo passo:")
    print(f"  python ml/train_yolov9t.py --data {yaml_path}")


if __name__ == "__main__":
    main()
