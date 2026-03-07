import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io

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

if uploaded_file is not None:
    # mostrar las columnas para comprar Origial y Procesada
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
                # Dibujar las detecciones en la imagen
                draw = ImageDraw.Draw(image)
                font = ImageFont.load_default()
                for detection in data["detections"]:
                    x1, y1, x2, y2 = detection["bbox"]
                    label = detection["label"]
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                    draw.text((x1, y1 - 10), label, fill="red", font=font)
                with col2:
                    st.header("Processed Image")
                    st.image(image, use_column_width=True)
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to the API: {e}")