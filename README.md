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

## Configuração do Hardware (Raspberry Pi)
- **Pino GPIO da Válvula Solenoide (Jato de Ar)**: Padrão no pino físico GPIO 18 (Bcm).
- **Esquema de Ligação**: GPIO 18 -> Resistor (220Ω) -> Gate do MOSFET de Potência (ex: IRLZ44N) -> Solenoide (com diodo de flyback).
