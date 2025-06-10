import streamlit as st
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp

st.set_page_config(page_title="Facial Landmarks", layout="centered")
st.title("🧠 Facial Landmark Detection with MediaPipe")

# Step 1: Load and Preprocess Image
st.write("This app allows you to upload an image and preprocess it for facial landmark detection.")
st.markdown("""
Upload a **portrait image**, and this app will draw a facial mesh on the detected face using **MediaPipe Face Mesh**.  
✅ Works best on **clear, front-facing** images.  
""")

# Set up the sidebar
st.sidebar.title("Instructions")
st.sidebar.markdown("""
1. **Upload an image**: Click on the "Upload an image" button to select a portrait image from your device.
2. **Image requirements**: Ensure the image is clear and front-facing for best results.
3. **View results**: The app will display the original image and the image with facial landmarks drawn on it.
4. **Troubleshooting**: If no landmarks are detected, try uploading a different image.
""")


# File uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image with PIL and convert to RGB
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    st.success("Image loaded! Scroll down to view the detected landmarks.")

    # Convert to Numpy array and OpenCV format (BRG)
    image_array = np.array(image)
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    st.write("Image shape:", image_rgb.shape)
    st.write("Image channels:", image_rgb.shape[2] if len(image_rgb.shape) > 2 else 1)
    st.write("✅ Image successfully loaded and preprocessed.")

    # Step 2: Apply Face Mesh
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    # Drawing spec - only lines (no dots)
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, color=(0, 255, 0))

    # Process image with face mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image = image_array,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_spec
                )

            st.image(image_array, caption="Facial Landmarks Detected", use_container_width=True)
        else:
            st.write("No facial landmarks detected in the image. Please try another image.")
            st.image(image_array, caption="No Landmarks Detected", use_container_width=True)
        