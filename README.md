# Sistema de Reconhecimento de Grãos

Detecção e contagem de grãos de café em tempo real (YOLOv9t), dashboard web, métricas SQLite e jato de ar para grãos defeituosos. Roda em PC ou Raspberry Pi.

---

## Início rápido — rodar agora (modelo já treinado)

Se o projeto já tem `models/coffee_beans_yolov9t/weights/best.pt`:

```bash
cd Raspberry-grain-Analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-ml.txt
export YOLO_MODEL_PATH=models/coffee_beans_yolov9t/weights/best.pt
python main.py
```

Abra **http://localhost:8000**

> Use sempre `source .venv/bin/activate` antes de `python main.py`. O `python3` do sistema não tem as dependências instaladas.

---

## Setup completo — do zero ao painel (copiar e colar)

Primeira vez no projeto (instala tudo, baixa dataset, treina e sobe o servidor):

```bash
cd Raspberry-grain-Analysis

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements-ml.txt

python ml/download_dataset.py --source huggingface

python ml/train_yolov9t.py \
  --data data/coffee_beans/data.yaml \
  --epochs 50 \
  --imgsz 416 \
  --batch 4 \
  --device cpu

export YOLO_MODEL_PATH=models/coffee_beans_yolov9t/weights/best.pt
python main.py
```

Com **GPU NVIDIA** (treino muito mais rápido):

```bash
python ml/train_yolov9t.py \
  --data data/coffee_beans/data.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --export-onnx
```

---

## Comandos do dia a dia

Ativar ambiente e subir o servidor:

```bash
cd Raspberry-grain-Analysis
source .venv/bin/activate
export YOLO_MODEL_PATH=models/coffee_beans_yolov9t/weights/best.pt
python main.py
```

Só servidor, **sem YOLO** (detector clássico por contornos):

```bash
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Retreinar o modelo:

```bash
source .venv/bin/activate
python ml/train_yolov9t.py --data data/coffee_beans/data.yaml --epochs 100 --device cpu
```

Exportar ONNX para Raspberry Pi:

```bash
source .venv/bin/activate
python ml/export_model.py
export YOLO_ONNX_PATH=models/coffee_beans_yolov9t/weights/best.onnx
python main.py
```

---

## Variáveis opcionais

```bash
export YOLO_MODEL_PATH=models/coffee_beans_yolov9t/weights/best.pt
export YOLO_CONF=0.5          # confiança mínima (padrão 0.5)
export YOLO_DEVICE=cpu        # cpu ou 0 (GPU)
```

---

## Dataset próprio

Estrutura YOLO:

```
data/meu_cafe/
├── data.yaml
├── train/images/  train/labels/
└── valid/images/  valid/labels/
```

Treinar:

```bash
source .venv/bin/activate
python ml/train_yolov9t.py --data data/meu_cafe/data.yaml --epochs 100
export YOLO_MODEL_PATH=models/coffee_beans_yolov9t/weights/best.pt
python main.py
```

Classes customizadas: edite `ml/class_mapping.py` (`defect` → estragado, `premium` → saudável).

Download alternativo (Roboflow):

```bash
export ROBOFLOW_API_KEY=sua_chave
python ml/download_dataset.py --source roboflow
```

---

## Hardware (Raspberry Pi)

- **GPIO válvula:** pino **18** (BCM)
- **Ligação:** GPIO 18 → 220Ω → MOSFET (IRLZ44N) → solenoide + diodo flyback

---

## Problemas comuns

| Erro | Comando / solução |
|------|-------------------|
| `No module named 'uvicorn'` | `source .venv/bin/activate` |
| YOLO não carrega | Confirme: `ls models/coffee_beans_yolov9t/weights/best.pt` |
| Treino lento | Use `--device 0` com GPU ou `--epochs 5 --imgsz 416 --batch 4` |

---

## Funcionalidades

- Câmera USB / Raspberry Pi / modo simulado
- YOLOv9t + contagem saudável / estragado
- Dashboard em http://localhost:8000
- Gravação de vídeo e jato de ar (GPIO)
