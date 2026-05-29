# Sistema de Reconhecimento de Grãos

Detecção e contagem de grãos de café em tempo real (YOLOv9t), dashboard web, métricas SQLite e jato de ar para grãos defeituosos. Roda em PC ou Raspberry Pi.

---

## Início rápido — passo a passo

### 1. Entrar na pasta do projeto

```bash
cd Raspberry-grain-Analysis
```

### 2. Criar o ambiente virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

O terminal deve mostrar `(.venv)` no início da linha. Para sair depois: `deactivate`.

### 3. Instalar dependências

```bash
pip install -r requirements-ml.txt
```

### 4. Criar a pasta `data/` e baixar o dataset

O script cria `data/coffee_beans/` com imagens, anotações e o arquivo `data.yaml`:

```bash
python ml/download_dataset.py --source huggingface
```

Isso pode demorar alguns minutos (download de ~2284 imagens).

Se preferir criar as pastas vazias à mão antes:

```bash
mkdir -p data/coffee_beans/train/images
mkdir -p data/coffee_beans/train/labels
mkdir -p data/coffee_beans/valid/images
mkdir -p data/coffee_beans/valid/labels
```

Depois coloque suas fotos e labels YOLO nessas pastas e rode o download **ou** monte o `data.yaml` (veja seção [Dataset próprio](#dataset-próprio)).

### 5. Treinar o YOLOv9t

Sem placa de vídeo (CPU — mais lento):

```bash
python ml/train_yolov9t.py \
  --data data/coffee_beans/data.yaml \
  --epochs 50 \
  --imgsz 416 \
  --batch 4 \
  --device cpu
```

Com GPU NVIDIA (mais rápido):

```bash
python ml/train_yolov9t.py \
  --data data/coffee_beans/data.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

Ao terminar, o modelo fica em `models/coffee_beans_yolov9t/weights/best.pt`.

### 6. Rodar o sistema

```bash
export YOLO_MODEL_PATH=models/coffee_beans_yolov9t/weights/best.pt
python main.py
```

Abra **http://localhost:8000** no navegador.

> Sempre use `source .venv/bin/activate` antes dos comandos `python`. O `python3` do sistema não tem as bibliotecas instaladas.

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

Use outro nome de pasta, por exemplo `data/meu_cafe/`:

```bash
mkdir -p data/meu_cafe/train/images data/meu_cafe/train/labels
mkdir -p data/meu_cafe/valid/images data/meu_cafe/valid/labels
```

Estrutura:

```
data/meu_cafe/
├── data.yaml
├── train/images/   train/labels/
└── valid/images/   valid/labels/
```

Cada imagem `foto.jpg` precisa de um `foto.txt` na pasta `labels/` (coordenadas normalizadas 0–1):

```
0 0.5 0.5 0.12 0.18
```

Crie `data/meu_cafe/data.yaml`:

```yaml
path: /caminho/absoluto/Raspberry-grain-Analysis/data/meu_cafe
train: train/images
val: valid/images
nc: 2
names:
  0: defect
  1: premium
```

Treinar e rodar:

```bash
source .venv/bin/activate
python ml/train_yolov9t.py --data data/meu_cafe/data.yaml --epochs 100
export YOLO_MODEL_PATH=models/coffee_beans_yolov9t/weights/best.pt
python main.py
```

Classes no painel: edite `ml/class_mapping.py`.

Download via Roboflow em vez do Hugging Face:

```bash
export ROBOFLOW_API_KEY=sua_chave
python ml/download_dataset.py --source roboflow
```

**Nota:** `data/` é para treino YOLO. A pasta `dataset/healthy` e `dataset/damaged` é onde o sistema salva recortes durante o uso da câmera.

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
