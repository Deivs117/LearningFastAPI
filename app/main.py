from urllib import response
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from app.schemas import DetectionResponse, TaskResponse, TaskIDResponse
from app.yolo_service import YoloObjectDetection

ml_models = {}
task_db = {}

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

def process_image_task(task_id: str, image_bytes: bytes):
    try:
        task_db[task_id]['status'] = 'Processing'
        yolo_model = ml_models.get('yolo')
        if yolo_model is None:
            task_db[task_id]['status'] = 'failed'
            task_db[task_id]['error'] = "El modelo de detección no está disponible."
            return
        #Inferencia del modelo con los bytes de la imagen
        DetectionResponse = yolo_model.detect_objects(image_bytes)
        task_db[task_id]['status'] = 'completed'
        task_db[task_id]['result'] =  DetectionResponse
    except Exception as e:
        task_db[task_id]['status'] = 'failed'
        task_db[task_id]['error'] = f"Error al procesar la imagen: {str(e)}"

@app.post("/api/v1/detect", response_model=TaskResponse, tags=["Detección de Objetos"])
async def detect_objects(file: UploadFile) -> TaskResponse:
    # CORRECCIÓN: validar correctamente el content_type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen jpeg o png.")
    try:
        image_bytes = await file.read()  # leer bytes de la imagen
        task_id = str(uuid.uuid4())  # Generar un ID único para la tarea
        task_db[task_id] = {
            'status': 'pending',
            'result': None,
            'error': None
        } # Crear una entrada en la base de datos de tareas
        BackgroundTasks().add_task(process_image_task, task_id, image_bytes) # Ejecutar la tarea en segundo plano
        return TaskResponse(
            task_id=task_id,
            status='pending',
            message="La tarea ha sido iniciada. Use el endpoint /api/v1/tasks/{task_id} para verificar el estado.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")
    
@app.get("/api/v1/tasks/{task_id}", response_model=TaskIDResponse, tags=["Tareas"])
async def get_task_status(task_id: str) -> TaskIDResponse:
    # revisar esta parte por task_info y task_db
    task_info = task_db.get(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail="Tarea no encontrada.")
    return TaskIDResponse(
        task_id=task_id,
        status=task_info['status'],
        result=task_info['result'],
        error=task_info['error'],
        message="Estado de la tarea recuperado correctamente."
    )