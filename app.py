import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="AI Object Detector", layout="centered")

st.title("📸 Fast AI Object Detector")
st.write("Upload an image to detect objects using YOLOv8!")

# Load YOLOv8 Model (Pre-trained, Free, No API)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt") 

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Detect Objects'):
        results = model(image)
        # Show results
        res_plotted = results[0].plot()
        st.image(res_plotted, caption='Detected Objects', use_column_width=True)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                label = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                st.write(f"✅ Found: **{label}** (Confidence: {conf*100:.1f}%)")
else:
    st.info("Aap image upload karein, AI khud detect kar lega.")
