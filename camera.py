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
        
        self.picam2 = None
        # Tenta inicializar rpicam via Picamera2 (para Raspberry Pi 4 Bookworm)
        try:
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            config = self.picam2.create_video_configuration({"size": (640, 480)})
            self.picam2.configure(config)
            self.picam2.start()
            self.cap = "picamera2"
            logger.info("Câmera inicializada com sucesso via rpicam (Picamera2)")
        except Exception as e:
            logger.warning(f"Não foi possível iniciar rpicam (Picamera2): {e}. Tentando OpenCV padrão...")
            # Tenta inicializar a câmera (indices 0, 1, 2...)
            for idx in [self.camera_index, 1, 2, 0]:
                self.cap = cv2.VideoCapture(idx)
                if self.cap.isOpened():
                    logger.info(f"Câmera inicializada com sucesso no índice {idx}")
                    self.camera_index = idx
                    break
            
            if not self.cap or (isinstance(self.cap, cv2.VideoCapture) and not self.cap.isOpened()):
                logger.warning("Nenhuma câmera física encontrada. Iniciando em modo SIMULADO.")
                self.cap = None
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        """Loop de captura contínuo para manter latência zero."""
        import glob
        import random
        while self.running:
            if self.picam2 is not None:
                try:
                    frame = self.picam2.capture_array()
                    with self._lock:
                        self.frame = frame
                        if self.is_recording and self.video_writer:
                            try:
                                self.video_writer.write(frame)
                            except Exception as e:
                                logger.error(f"Erro ao gravar frame: {e}")
                except Exception as e:
                    logger.warning(f"Falha ao capturar frame do rpicam: {e}")
                    time.sleep(0.1)
            elif self.cap and isinstance(self.cap, cv2.VideoCapture) and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self._lock:
                        self.frame = frame
                        if self.is_recording and self.video_writer:
                            try:
                                self.video_writer.write(frame)
                            except Exception as e:
                                logger.error(f"Erro ao gravar frame: {e}")
                else:
                    logger.warning("Falha ao capturar frame físico da câmera.")
                    time.sleep(0.1)
            else:
                # Modo simulado: lê imagens do dataset
                dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
                images = glob.glob(os.path.join(dataset_dir, "healthy", "*.*")) + \
                         glob.glob(os.path.join(dataset_dir, "damaged", "*.*"))
                
                # Filtra apenas extensões de imagem comuns
                images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if not images:
                    # Se não tiver nenhuma, não faz nada
                    time.sleep(1)
                    continue
                
                img_path = random.choice(images)
                frame = cv2.imread(img_path)
                
                if frame is not None:
                    # Redimensiona para garantir compatibilidade com o modelo
                    frame = cv2.resize(frame, (640, 480))
                    with self._lock:
                        self.frame = frame
                        if self.is_recording and self.video_writer:
                            try:
                                self.video_writer.write(frame)
                            except Exception as e:
                                logger.error(f"Erro ao gravar frame simulado: {e}")
                
                time.sleep(1.0) # Atualiza a imagem a cada 1 segundo no modo dataset

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

