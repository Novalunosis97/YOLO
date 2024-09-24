import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
import os
import sys

# Add the YOLOv5 directory to the Python path
yolov5_dir = os.path.abspath("yolov5")
if yolov5_dir not in sys.path:
    sys.path.append(yolov5_dir)

# Now we can import the necessary modules from YOLOv5
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.plots import Annotator, colors

st.set_page_config(layout="wide")

@st.cache_resource
def load_model(path, device):
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        return None

    try:
        model = attempt_load(path, device=device)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def infer_image(model, img, confidence, device):
    img = np.array(img)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]
    pred = non_max_suppression(pred, confidence, 0.45)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = det[:, :4].clone()  # xyxy
            
            annotator = Annotator(img.squeeze().permute(1, 2, 0).cpu().numpy(), line_width=3, example="best")
            
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = f'{model.names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))

    result_img = annotator.result()
    return Image.fromarray(result_img)

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
            result_image = infer_image(model, image, confidence, device)
            st.image(result_image, caption="Detection Result", use_column_width=True)

if __name__ == "__main__":
    main()