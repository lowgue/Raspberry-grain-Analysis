# Sistema de Reconhecimento de Grãos

Este projeto é um sistema leve de detecção e classificação de grãos projetado para rodar em Raspberry Pi ou em PCs locais para desenvolvimento. Ele identifica grãos saudáveis e estragados em tempo real utilizando Visão Computacional / IA, registra métricas em um banco SQLite, gerencia a gravação de vídeo e aciona uma válvula solenoide (sinal digital para jato de ar) para ejetar grãos defeituosos.

## Funcionalidades

1. **Reconhecimento de Câmera Universal**: Suporta câmeras USB genéricas e câmeras nativas do Raspberry Pi.
2. **Classificação por Inteligência Artificial**: Detecção de grãos usando OpenCV DNN ou fallback simulado para fins de teste.
3. **Métricas Completas**: Contagem acumulada e por minuto de grãos processados, agrupados por estado (Saudável, Estragado).
4. **Controle de Válvula (Jato de Ar)**: Aciona um pino GPIO para soltar um jato de ar sempre que um grão estragado é identificado.
5. **Dashboard Web Responsivo**: Painel de visualização em tempo real (desktop e mobile) com gráficos e controle de gravação.
6. **Acesso Remoto Seguro**: Métodos de autenticação embutidos e diretrizes de rede.

## Instalação e Execução

### Requisitos
- Python 3.9 ou superior
- OpenCV (`opencv-python-headless` ou `opencv-python`)
- FastAPI & Uvicorn
- SQLite3 (nativo do Python)

### Instalação das Dependências
```bash
pip install -r requirements.txt
```

### Executando o Servidor
```bash
python main.py
```
O painel estará disponível em `http://localhost:8000`.

## YOLOv9t + Coffee Bean Dataset

O sistema suporta detecção com **YOLOv9t** treinado no dataset **CoffeeBeansGradingV3** (Roboflow: classes `defect` e `premium`, mapeadas para `damaged` e `healthy`).

### 1. Instalar dependências de ML (PC com GPU recomendado)

```bash
pip install -r requirements-ml.txt
```

### 2. Baixar o dataset

**Opção A — Hugging Face (sem API key, padrão):**

```bash
python ml/download_dataset.py --source huggingface
```

Usa [SamruddhK/coffee-bean-grading-dataset](https://huggingface.co/datasets/SamruddhK/coffee-bean-grading-dataset) (~2284 imagens anotadas), convertido para YOLO com classes `defect` e `premium`.

**Opção B — Roboflow (CoffeeBeansGradingV3):**

```bash
export ROBOFLOW_API_KEY=sua_chave_aqui
python ml/download_dataset.py --source roboflow
```

### 3. Treinar YOLOv9t

```bash
python ml/train_yolov9t.py --epochs 100 --export-onnx
```

O modelo final fica em `models/coffee_beans_yolov9t/weights/best.pt`.

### 4. Usar no servidor de detecção

```bash
export YOLO_MODEL_PATH=models/coffee_beans_yolov9t/weights/best.pt
python main.py
```

No Raspberry Pi, exporte ONNX para inferência mais leve:

```bash
python ml/export_model.py
export YOLO_ONNX_PATH=models/coffee_beans_yolov9t/weights/best.onnx
```

Sem modelo treinado, o sistema continua usando o detector clássico (contornos/cor).

## Configuração do Hardware (Raspberry Pi)
- **Pino GPIO da Válvula Solenoide (Jato de Ar)**: Padrão no pino físico GPIO 18 (Bcm).
- **Esquema de Ligação**: GPIO 18 -> Resistor (220Ω) -> Gate do MOSFET de Potência (ex: IRLZ44N) -> Solenoide (com diodo de flyback).
