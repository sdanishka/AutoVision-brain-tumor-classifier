import streamlit as st
import torch
import cv2
from PIL import Image
import numpy as np
import os

from roboflow import Roboflow

# Roboflow configuration
rf = Roboflow(api_key="Pqs2Di4XkKfl9U7avfkn")
project = rf.workspace("sara-ghazi-qj73m").project("bt-cohqx")
version = project.version(1)
dataset = version.download("yolov5")

# Define model path
model_path = dataset.model_path

# Load YOLOv5 model directly
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

st.title("Brain Tumor Classification")
st.write("Upload an image to detect brain tumors.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Convert image to OpenCV format
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Perform inference
    results = model(image)
    
    # Draw bounding boxes
    results.render()
    st.image(results.imgs[0], caption='Processed Image.', use_column_width=True)
    
    # Print results
    st.write(results.pandas().xyxy[0])
