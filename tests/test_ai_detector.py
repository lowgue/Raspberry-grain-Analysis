import numpy as np
import cv2
from unittest.mock import MagicMock, patch
import pytest
from ai_detector import GrainDetector

def test_grain_detector_initialization_fallback():
    # Sem arquivos e sem variáveis de ambiente, o detector deve inicializar
    # com net e yolo como None, caindo no modo Visão Computacional Clássica
    with patch("os.path.isfile", return_value=False):
        detector = GrainDetector(model_path="dummy.pt")
        assert detector.net is None
        assert detector.yolo is None

def test_detect_cv_fallback_simulated():
    # Cria uma imagem preta de teste
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    detector = GrainDetector(model_path="dummy.pt")
    # Limpa referências a redes neurais se houver no ambiente
    detector.net = None
    detector.yolo = None
    
    # Com frame todo preto, o detector por visão clássica não deve encontrar grãos
    annotated_frame, detections = detector.detect_grains(frame, is_simulated=True)
    assert len(detections) == 0

def test_detect_dnn_coordinates_scaling():
    # Mock do método de inicialização e arquivo ONNX
    with patch("os.path.isfile", return_value=True):
        with patch("cv2.dnn.readNetFromONNX") as mock_read_net:
            mock_net = MagicMock()
            mock_read_net.return_value = mock_net
            
            # Inicializa o detector que carregará o mock_net
            detector = GrainDetector(model_path="dummy.pt")
            detector.net = mock_net
            detector.yolo = None
            detector.conf_threshold = 0.5

            # Cria um mock da saída da rede YOLOv9t
            # Formato: [1, 4 + nc, num_anchors] -> ex: [1, 6, 5]
            # nc = 2 (classes: 0: defect, 1: premium)
            # Vamos simular 5 âncoras
            mock_output = np.zeros((1, 6, 5), dtype=np.float32)
            
            # Âncora 0 (Ignorada por baixa confiança):
            # cx=320, cy=320, w=100, h=100, class_0=0.1, class_1=0.2
            mock_output[0, :, 0] = [320, 320, 100, 100, 0.1, 0.2]
            
            # Âncora 1 (Detecta Premium - Saudável):
            # cx=320, cy=320, w=100, h=100, class_0=0.1, class_1=0.9
            mock_output[0, :, 1] = [320, 320, 100, 100, 0.1, 0.9]
            
            # Âncora 2 (Detecta Defect - Danificado):
            # cx=100, cy=100, w=50, h=50, class_0=0.8, class_1=0.0
            mock_output[0, :, 2] = [100, 100, 50, 50, 0.8, 0.0]

            mock_net.forward.return_value = mock_output

            # Frame real da câmera 640x480 (escala y é 480/640 = 0.75, escala x é 640/640 = 1.0)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Executa detecção ONNX
            annotated_frame, detections = detector.detect_grains(frame)
            
            # Deve detectar 2 grãos (Âncora 1 e Âncora 2)
            assert len(detections) == 2
            
            # Primeiro grão (Premium -> class_id 1 -> healthy)
            # cx=320, cy=320, w=100, h=100
            # x1 = (320 - 100/2) * 1.0 = 270
            # y1 = (320 - 100/2) * 0.75 = 202.5 -> 202
            # bw = 100 * 1.0 = 100
            # bh = 100 * 0.75 = 75
            det_healthy = [d for d in detections if d["status"] == "healthy"][0]
            assert det_healthy["confidence"] == pytest.approx(0.9)
            assert det_healthy["box"] == (270, 202, 100, 75)
            assert det_healthy["class_name"] == "premium"

            # Segundo grão (Defect -> class_id 0 -> damaged)
            # cx=100, cy=100, w=50, h=50
            # x1 = (100 - 50/2) * 1.0 = 75
            # y1 = (100 - 50/2) * 0.75 = 56.25 -> 56
            # bw = 50 * 1.0 = 50
            # bh = 50 * 0.75 = 37.5 -> 37
            det_damaged = [d for d in detections if d["status"] == "damaged"][0]
            assert det_damaged["confidence"] == pytest.approx(0.8)
            assert det_damaged["box"] == (75, 56, 50, 37)
            assert det_damaged["class_name"] == "defect"
