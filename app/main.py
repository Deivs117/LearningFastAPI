from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, HTTPException
from app.schemas import DetectionResponse
from app.yolo_service import YoloObjectDetection

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicializar el modelo YOLO al iniciar la aplicación
    ml_models['yolo'] = YoloObjectDetection(model_name='yolo26s.pt')
    yield
    # ----------------------------
    # Ejecución de shutdown
    # ----------------------------
    ml_models.clear()
    print("Modelos descargados y recursos liberados.")

#Instancia de FastAPI con el contexto de vida útil definido
app = FastAPI(
    title       = "API de Detección de Objetos con YOLOv26s",
    description = "Una API para detectar vehículos, peatones y ciclistas en imágenes utilizando el modelo YOLOv26s.",
    version     = "1.0.0",
    lifespan    = lifespan
)

@app.post("/api/v1/detect", response_model=DetectionResponse, summary="Detectar objetos en una imagen")
async def detect_objects(file: UploadFile) -> DetectionResponse:
    # CORRECCIÓN: validar correctamente el content_type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen jpeg o png.")
    try:
        image_bytes = await file.read()  # leer bytes de la imagen
        yolo_model = ml_models.get('yolo')
        if yolo_model is None:
            raise HTTPException(status_code=500, detail="El modelo de detección no está disponible.")
        response = yolo_model.detect_objects(image_bytes)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")