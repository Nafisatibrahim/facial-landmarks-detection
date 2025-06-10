import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.title("🧠 Facial Landmark Detection")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image with PIL and convert to RGB
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    # Convert to Numpy array and OpenCV format (BRG)
    image_array = np.array(image)
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    st.write("Image shape:", image_rgb.shape)
    st.write("Image channels:", image_rgb.shape[2] if len(image_rgb.shape) > 2 else 1)
    st.write("✅ Image successfully loaded and preprocessed.")