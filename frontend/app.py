import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io
from pathlib import Path

# Configuration of the website
st.set_page_config(
    page_title="Dectecting Images using YOLO26",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="expanded"
)
# Title and description
st.title("Dectecting Images using YOLO26")
st.markdown("""
Upload an image to detect objects using the YOLO26 model. The detected objects will be highlighted with bounding boxes and labels.
""")

API_URL = "http://127.0.0.1:8000/api/v1/detect"  # Change this to your backend URL

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def get_font(size=100):
    """
    Intentar cargar una fuente TTF en orden:
      1. Arial Black (Windows)
      2. DejaVuSans (común en Linux/PIL)
      3. Fuente incluida en el repo: ./frontend/fonts/DejaVuSans.ttf  (si la añades)
      4. Fallback a ImageFont.load_default()
    """
    # 1) Intento cargar fuentes del sistema
    try:
        return ImageFont.truetype("arialblack.ttf", size=size)
    except Exception:
        pass
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        pass

    # 2) Intento cargar una fuente incluida en el repo (opcional)
    local_font = Path(__file__).parent / "fonts" / "DejaVuSans.ttf"
    if local_font.exists():
        try:
            return ImageFont.truetype(str(local_font), size=size)
        except Exception:
            pass

    # 3) Fallback
    return ImageFont.load_default()

if uploaded_file is not None:
    # mostrar las columnas para comparar Original y Procesada
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file).convert("RGB")
    with col1:
        st.header("Original Image")
        st.image(image, width="stretch")
    if st.button("Detect Objects"):
        with st.spinner("Processing..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(API_URL, files=files)
                response.raise_for_status()
                data = response.json()
                # Cambiar color de la fuentes a rojo para mejor visibilidad
                draw = ImageDraw.Draw(image)
                font = get_font(size=100)  # fuente segura con fallback
                for detection in data.get("detections", []):
                    box = detection.get("box", {})
                    x1 = int(box.get("x_min", 0))
                    y1 = int(box.get("y_min", 0))
                    x2 = int(box.get("x_max", 0))
                    y2 = int(box.get("y_max", 0))
                    label = detection.get("class_name", "obj")
                    draw.rectangle([x1, y1, x2, y2], outline="black", width=10)
                    draw.text((x1, max(0, y1 - 10)), label, fill="cyan", font=font)
                with col2:
                    st.header("Processed Image")
                    st.image(image, width="stretch")
            except requests.exceptions.RequestException as e:
                # Mostrar texto de la respuesta para debug si existe
                err_text = ""
                try:
                    err_text = response.text
                except Exception:
                    err_text = ""
                st.error(f"Error connecting to the API: {e}\n{err_text}")