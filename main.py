import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import time
import logging

# Configuração de logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("main")

# Importa os módulos criados
from camera import CameraManager
from ai_detector import GrainDetector
from database import db_manager
from gpio_control import gpio_controller

app = FastAPI(title="Ceres AI - Grain Recognition System")

# Servindo os arquivos estáticos (Frontend)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Inicialização de instâncias globais
camera_manager = CameraManager(camera_index=0)
detector = GrainDetector()

# Estado global da aplicação
current_group = "Grupo Principal"
last_ejection_time = 0.0
last_healthy_log_time = 0.0

# Intervalos de Cooldown (evitar contagem duplicada do mesmo grão em frames consecutivos)
COOLDOWN_DAMAGED_SEC = 0.6
COOLDOWN_HEALTHY_SEC = 0.6

class GroupSelection(BaseModel):
    group_name: str

class EjectionRequest(BaseModel):
    group_name: str

@app.get("/")
def read_root():
    """Entrega a página principal do dashboard."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Painel Ceres AI ativo. Frontend não encontrado em /static/index.html"}

def gen_frames():
    """Gerador do stream de vídeo MJPEG com detecção de IA acoplada."""
    global last_ejection_time, last_healthy_log_time
    
    while True:
        frame = camera_manager.get_frame()
        if frame is None:
            time.sleep(0.03)
            continue
        
        # Executa a inteligência artificial para detecção e classificação
        annotated_frame, detections = detector.detect_grains(frame, is_simulated=camera_manager.is_simulated)
        
        now = time.time()
        for det in detections:
            status = det["status"]
            confidence = det["confidence"]
            
            if status == "damaged":
                # Verifica se saiu do tempo de cooldown para disparar e registrar
                if now - last_ejection_time > COOLDOWN_DAMAGED_SEC:
                    last_ejection_time = now
                    logger.info("Grão estragado detectado! Acionando jato de ar...")
                    
                    # 1. Envia o sinal elétrico para o solenoide
                    gpio_controller.trigger_air_jet()
                    
                    # 2. Registra o evento de ejeção no banco de dados
                    db_manager.log_grain(status, current_group, confidence)
            
            elif status == "healthy":
                # Registro com cooldown para evitar inflar contagem do mesmo grão
                if now - last_healthy_log_time > COOLDOWN_HEALTHY_SEC:
                    last_healthy_log_time = now
                    logger.info("Grão saudável registrado.")
                    db_manager.log_grain(status, current_group, confidence)
        
        # Converte o frame anotado para formato JPEG
        import cv2
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        # Retorna o frame no formato Multipart para o browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Limita o loop para cerca de 30 FPS
        time.sleep(0.03)

@app.get("/api/video_feed")
def video_feed():
    """Endpoint para streaming MJPEG em tempo real."""
    return StreamingResponse(
        gen_frames(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/metrics")
def get_metrics():
    """Retorna métricas consolidadas de grãos, histórico e estado de gravação."""
    summary = db_manager.get_summary_metrics()
    groups = db_manager.get_group_metrics()
    return {
        "summary": summary,
        "groups": groups,
        "recording": camera_manager.is_recording
    }

@app.get("/api/recent_history")
def get_recent_history():
    """Retorna a lista de logs das últimas 10 detecções."""
    return db_manager.get_recent_history(limit=10)

@app.post("/api/set_group")
def set_group(group: GroupSelection):
    """Muda o lote/grupo ativo de contagem de grãos."""
    global current_group
    current_group = group.group_name
    logger.info(f"Lote/Grupo ativo alterado para: {current_group}")
    return {"status": "success", "active_group": current_group}

@app.post("/api/start_recording")
def start_recording():
    """Inicia a gravação do feed de vídeo."""
    path = camera_manager.start_recording()
    if path:
        return {"status": "success", "filepath": path}
    raise HTTPException(status_code=500, detail="Não foi possível iniciar a gravação.")

@app.post("/api/stop_recording")
def stop_recording():
    """Para a gravação atual de vídeo."""
    path = camera_manager.stop_recording()
    return {"status": "success", "filepath": path}

@app.post("/api/trigger_ejection")
def trigger_ejection(req: EjectionRequest):
    """Acionamento manual do solenoide para teste de maquinário e calibração de jato."""
    logger.info(f"Disparo de teste manual do jato de ar solicitado para o grupo: {req.group_name}")
    
    # Aciona a válvula solenoide (GPIO 18)
    gpio_controller.trigger_air_jet()
    
    # Registra como grão danificado simulado para auditoria do teste
    db_manager.log_grain("damaged", req.group_name, 1.0)
    
    return {"status": "success", "message": "Jato de ar acionado manualmente."}

@app.post("/api/clear_metrics")
def clear_metrics():
    """Reseta todos os contadores e limpa o banco de dados SQLite."""
    db_manager.clear_metrics()
    return {"status": "success", "message": "Banco de dados de métricas resetado."}

@app.on_event("shutdown")
def shutdown_event():
    """Garante o desligamento limpo dos recursos de hardware e threads."""
    camera_manager.stop()
    gpio_controller.cleanup()
    logger.info("Recursos liberados e servidor encerrado.")

if __name__ == "__main__":
    # Inicia o servidor uvicorn na porta 8000 para acesso local e remoto
    uvicorn.run(app, host="0.0.0.0", port=8000)
