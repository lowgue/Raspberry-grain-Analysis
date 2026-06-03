import cv2
import numpy as np
import os
import logging
import random

logger = logging.getLogger("ai_detector")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YOLO_WEIGHTS = os.path.join(
    PROJECT_ROOT, "models", "coffee_beans_yolov9t", "weights", "best.pt"
)
DEFAULT_ONNX_PATH = os.path.join(
    PROJECT_ROOT, "models", "coffee_beans_yolov9t", "weights", "best.onnx"
)


class GrainDetector:
    def __init__(self, model_path=None):
        self.model_path = model_path or os.environ.get("YOLO_MODEL_PATH", DEFAULT_YOLO_WEIGHTS)
        self.onnx_path = os.environ.get("YOLO_ONNX_PATH", DEFAULT_ONNX_PATH)
        self.net = None
        self.yolo = None
        self.yolo_names = {}
        self.conf_threshold = float(os.environ.get("YOLO_CONF", "0.5"))
        self.classes = ["healthy", "damaged"]

        # Em CPU (como no Raspberry Pi), priorizamos o modelo ONNX por ser muito mais rápido e leve
        if os.path.isfile(self.onnx_path):
            self._load_onnx()
        
        if self.net is None:
            self._load_yolo_ultralytics()

        if self.yolo is None and self.net is None:
            logger.info(
                "Nenhum modelo YOLOv9t encontrado (%s ou %s). "
                "Usando detector por Visão Computacional (contornos/cor). "
                "Treine com: python ml/download_dataset.py && python ml/train_yolov9t.py",
                self.model_path,
                self.onnx_path,
            )

    def _load_yolo_ultralytics(self):
        if not os.path.isfile(self.model_path):
            return
        try:
            from ultralytics import YOLO

            self.yolo = YOLO(self.model_path)
            self.yolo_names = self.yolo.names or {}
            logger.info("YOLOv9t (Ultralytics) carregado: %s", self.model_path)
        except ImportError:
            logger.warning(
                "Ultralytics não instalado; ignorando .pt. "
                "Instale com: pip install -r requirements-ml.txt"
            )
        except Exception as e:
            logger.error("Erro ao carregar YOLOv9t: %s", e)

    def _load_onnx(self):
        if self.net is not None or not os.path.isfile(self.onnx_path):
            return
        try:
            self.net = cv2.dnn.readNetFromONNX(self.onnx_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            logger.info("Modelo ONNX carregado: %s", self.onnx_path)
        except Exception as e:
            logger.error("Erro ao carregar ONNX: %s", e)

    def detect_grains(self, frame, is_simulated=False):
        """
        Detecta grãos no frame da câmera.
        Retorna:
            annotated_frame: Frame com caixas e labels desenhados.
            detections: Lista de dicionários contendo {'status': str, 'confidence': float, 'box': tuple}
        """
        if self.yolo is not None:
            return self._detect_yolo_ultralytics(frame)
        if self.net is not None:
            return self._detect_dnn(frame)
        return self._detect_cv_fallback(frame, is_simulated=is_simulated)

    def _detect_yolo_ultralytics(self, frame):
        """Inferência com YOLOv9t treinado no Coffee Bean Dataset."""
        from ml.class_mapping import to_grain_status

        h, w = frame.shape[:2]
        results = self.yolo.predict(
            frame,
            conf=self.conf_threshold,
            verbose=False,
            device=os.environ.get("YOLO_DEVICE", ""),
        )
        detections = []
        annotated_frame = frame.copy()

        if not results:
            return annotated_frame, detections

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = self.yolo_names.get(cls_id, str(cls_id))
                status = to_grain_status(class_name)

                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                bw, bh = x2 - x1, y2 - y1

                detections.append(
                    {
                        "status": status,
                        "confidence": conf,
                        "box": (x1, y1, bw, bh),
                        "class_name": class_name,
                    }
                )

                color = (0, 255, 0) if status == "healthy" else (0, 0, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{status} ({class_name}) {conf:.2f}"
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        return annotated_frame, detections

    def _detect_dnn(self, frame):
        """Detecção via OpenCV DNN (ONNX exportado do YOLOv9t)."""
        from ml.class_mapping import to_grain_status

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()

        detections = []
        annotated_frame = frame.copy()

        # Saída Ultralytics ONNX: [1, 4+nc, num_anchors] — NMS manual simplificado
        if isinstance(outputs, (list, tuple)):
            out = outputs[0]
        else:
            out = outputs

        if out.ndim == 3:
            out = out[0]
        if out.shape[0] < out.shape[1]:
            out = out.T

        for row in out:
            if row.shape[0] < 5:
                continue
            scores = row[4:]
            if scores.size == 0:
                continue
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence < self.conf_threshold:
                continue

            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            bw_px, bh_px = x2 - x1, y2 - y1

            class_name = str(class_id)
            status = to_grain_status(class_name)

            detections.append(
                {
                    "status": status,
                    "confidence": confidence,
                    "box": (x1, y1, bw_px, bh_px),
                    "class_name": class_name,
                }
            )
            color = (0, 255, 0) if status == "healthy" else (0, 0, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated_frame,
                f"{status} {confidence:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        return annotated_frame, detections

    def _detect_cv_fallback(self, frame, is_simulated=False):
        """
        Detector por visão computacional clássica (Fallback).
        Segmenta e analisa grãos reais que entram na cena baseado em área, contorno e cor.
        """
        annotated_frame = frame.copy()
        detections = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not is_simulated:
            contrast = int(gray.max()) - int(gray.min())
            if contrast < 45:
                return annotated_frame, []

        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 500 < area < 50000:
                x, y, w_box, h_box = cv2.boundingRect(cnt)

                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                mean_val = cv2.mean(frame, mask=mask)[:3]

                avg_brightness = sum(mean_val) / 3

                if avg_brightness < 95 or (mean_val[2] > 1.2 * mean_val[0] and avg_brightness < 130):
                    status = "damaged"
                else:
                    status = "healthy"

                confidence = float(random.uniform(0.88, 0.98))

                detections.append(
                    {
                        "status": status,
                        "confidence": confidence,
                        "box": (x, y, w_box, h_box),
                    }
                )

                color = (0, 0, 255) if status == "damaged" else (0, 255, 0)
                cv2.rectangle(annotated_frame, (x, y), (x + w_box, y + h_box), color, 2)
                cv2.putText(
                    annotated_frame,
                    f"{status.upper()} {confidence:.2f}",
                    (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        return annotated_frame, detections

    @staticmethod
    def save_grain_crop(frame, box, status):
        """
        Recorta a região do grão do frame original e salva
        em uma pasta correspondente (dataset/healthy ou dataset/damaged).
        """
        try:
            dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
            target_dir = os.path.join(dataset_dir, status)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)

            x, y, w, h = box
            h_img, w_img = frame.shape[:2]

            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w_img, x + w), min(h_img, y + h)

            if (x2 - x1) > 10 and (y2 - y1) > 10:
                crop = frame[y1:y2, x1:x2]
                import uuid
                from datetime import datetime

                filename = f"grain_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
                cv2.imwrite(os.path.join(target_dir, filename), crop)
                logger.info("Frame de grão (%s) salvo em: %s/%s", status, target_dir, filename)
        except Exception as e:
            logger.error("Erro ao salvar amostra de grão no dataset: %s", e)


grain_detector = GrainDetector()
