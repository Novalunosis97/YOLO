import streamlit as st
from PIL import Image
import torch
import os
import sys

# Add the YOLOv5 directory to the Python path
yolov5_dir = os.path.abspath("yolov5")
if yolov5_dir not in sys.path:
    sys.path.append(yolov5_dir)

st.set_page_config(layout="wide")

@st.cache_resource
def load_model(path, device):
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        return None

    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=path)
        model.to(device)
        print("Model loaded to", device)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def infer_image(model, img, confidence):
    model.conf = confidence
    results = model(img)
    results.render()
    return Image.fromarray(results.ims[0])

def main():
    st.title("YOLO Object Detection")

    # Model loading
    model_path = 'models/best.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_path, device)

    if model is None:
        st.error("Failed to load the model. Please check the model file and try again.")
        return

    # Confidence slider
    confidence = st.slider('Confidence', min_value=0.1, max_value=1.0, value=0.45)

    # Image upload
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button('Run Detection'):
            result_image = infer_image(model, image, confidence)
            st.image(result_image, caption="Detection Result", use_column_width=True)

if __name__ == "__main__":
    main()