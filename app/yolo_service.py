import io
import time
from PIL import Image
from ultralytics import YOLO

from .schemas import DetectionResponse, Detection, BoundingBox

class YoloObjectDetection:
    def __init__(self, model_name: str = 'yolo26s.pt'):
        self.model = YOLO(model_name)

    def detect_objects(self, image_bytes: bytes) -> DetectionResponse:
        # Cargar la imagen desde los bytes
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Realizar la inferencia y medir el tiempo (inicialmente se mide en milisegundos, luego se convierte a segundos)
        start_time = time.time()
        results = self.model(image)
        inference_time = round((time.time() - start_time) * 1000, 2)  # Tiempo en milisegundos con 2 decimales
        
        # Procesar los resultados para crear la respuesta
        detections = []
        for result in results:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                class_id = int(box.cls[0].item())
                c_n = self.model.names[class_id]
                
                detection = Detection(
                    class_name=c_n,
                    confidence=conf,
                    box=BoundingBox(
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max
                    )
                )
                detections.append(detection)
        
        return DetectionResponse(inference_time=inference_time, detections=detections)