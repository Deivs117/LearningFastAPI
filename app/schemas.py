from pydantic import BaseModel, Field
from typing import List

class BoundingBox(BaseModel):
    x_min: float = Field(..., description="Coordenada x mínima del cuadro delimitador")
    y_min: float = Field(..., description="Coordenada y mínima del cuadro delimitador")
    x_max: float = Field(..., description="Coordenada x máxima del cuadro delimitador")
    y_max: float = Field(..., description="Coordenada y máxima del cuadro delimitador")

class Detection(BaseModel):
    class_name: str = Field(..., description="Nombre de la clase detectada (vehículo, peatón, ciclista)")
    confidence: float = Field(..., description="Confianza de la detección")
    box: BoundingBox = Field(..., description="Cuadro delimitador de la detección")

class DetectionResponse(BaseModel):
    inference_time: float = Field(..., description="Tiempo que tomó realizar la inferencia en segundos")
    detections: List[Detection] = Field(..., description="Lista de detecciones en la imagen")