import cv2
import numpy as np
import math
import threading
import time
import os
import logging
from datetime import datetime


logger = logging.getLogger("camera")

RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "recordings")

class CameraManager:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.running = False
        self.frame = None
        self._lock = threading.Lock()
        
        # Estado de gravação
        self.is_recording = False
        self.video_writer = None
        self.recording_filepath = None
        
        # Garante a existência do diretório de gravações
        if not os.path.exists(RECORDINGS_DIR):
            os.makedirs(RECORDINGS_DIR)

        # Inicia a captura em thread de background
        self.start()

    @property
    def is_simulated(self):
        """Retorna True se estivermos em modo simulado (sem câmera física)."""
        return self.cap is None

    def start(self):
        """Inicia a captura da câmera na thread."""
        if self.running:
            return
        
        # Tenta inicializar a câmera (indices 0, 1, 2...)
        for idx in [self.camera_index, 1, 2, 0]:
            self.cap = cv2.VideoCapture(idx)
            if self.cap.isOpened():
                logger.info(f"Câmera inicializada com sucesso no índice {idx}")
                self.camera_index = idx
                break
        
        if not self.cap or not self.cap.isOpened():
            logger.warning("Nenhuma câmera física encontrada. Iniciando em modo SIMULADO.")
            self.cap = None
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        """Loop de captura contínuo para manter latência zero."""
        simulated_frame_count = 0
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self._lock:
                        self.frame = frame
                        
                        # Se estiver gravando, grava o frame atual
                        if self.is_recording and self.video_writer:
                            try:
                                self.video_writer.write(frame)
                            except Exception as e:
                                logger.error(f"Erro ao gravar frame: {e}")
                else:
                    logger.warning("Falha ao capturar frame físico da câmera.")
                    time.sleep(0.1)
            else:
                # Simula uma imagem de teste caso não haja câmera disponível
                # Cria uma imagem colorida que se move ligeiramente para demonstrar atividade
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                # Adiciona grid ou animação de fundo
                simulated_frame_count += 1
                bg_color = int((10 + abs(10 * math.sin(simulated_frame_count * 0.05))))
                frame[:] = [bg_color, bg_color, bg_color]
                
                # Desenha esteiras transportadoras simuladas
                cv2.rectangle(frame, (100, 0), (540, 480), (40, 40, 40), -1) # Esteira principal
                cv2.line(frame, (320, 0), (320, 480), (80, 80, 80), 2) # Divisória
                
                # Exibe um texto "Sem Sinal de Câmera - Demo Ativa"
                cv2.putText(
                    frame, 
                    "CAMERA DEMO (SEM DISPOSITIVO)", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                )
                
                with self._lock:
                    self.frame = frame
                    if self.is_recording and self.video_writer:
                        try:
                            self.video_writer.write(frame)
                        except Exception as e:
                            logger.error(f"Erro ao gravar frame simulado: {e}")
                
                time.sleep(1/30.0) # 30 FPS simulados

    def get_frame(self):
        """Retorna uma cópia do frame mais recente."""
        with self._lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

    def start_recording(self):
        """Inicia a gravação de vídeo para um arquivo no disco."""
        with self._lock:
            if self.is_recording:
                return self.recording_filepath
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"record_{timestamp}.mp4"
            filepath = os.path.join(RECORDINGS_DIR, filename)
            
            # Pega dimensões do frame atual para configurar o codec
            h, w = 480, 640
            if self.frame is not None:
                h, w = self.frame.shape[:2]
                
            # Codec MP4V para portabilidade
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filepath, fourcc, 20.0, (w, h))
            
            if self.video_writer.isOpened():
                self.is_recording = True
                self.recording_filepath = filepath
                logger.info(f"Gravação iniciada: {filepath}")
                return filepath
            else:
                logger.error("Falha ao abrir VideoWriter.")
                self.video_writer = None
                return None

    def stop_recording(self):
        """Para a gravação atual e fecha o arquivo."""
        with self._lock:
            if not self.is_recording:
                return None
            
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                
            path = self.recording_filepath
            logger.info(f"Gravação finalizada: {path}")
            self.recording_filepath = None
            return path

    def stop(self):
        """Para o loop de leitura e libera a câmera."""
        self.running = False
        self.stop_recording()
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("Câmera desligada.")

# Limpeza concluída

