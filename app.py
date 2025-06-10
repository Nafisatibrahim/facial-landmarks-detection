import streamlit as st
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp

# Set page config (must be first Streamlit command)
st.set_page_config(page_title="Facial Landmarks", layout="centered")
st.title("🧠 Facial Landmark Detection with MediaPipe")

# Description
st.write("This app allows you to upload an image and detect facial landmarks using MediaPipe.")
st.markdown("""
Upload a **portrait image**, and this app will draw a facial mesh on the detected face using **MediaPipe Face Mesh**.  
✅ Works best on **clear, front-facing** images.  
""")

# Sidebar
st.sidebar.title("Instructions")
st.sidebar.markdown("""
1. **Upload an image**: Click on the button to select a portrait image.
2. **Image requirements**: Make sure it's clear and front-facing.
3. **View results**: The app will display the image with facial landmarks.
4. **Troubleshooting**: If no landmarks are detected, try another image.
""")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image and convert to RGB
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    st.success("Image loaded! Scroll down to view the detected landmarks.")

    # Convert to NumPy array and OpenCV format (BGR)
    image_array = np.array(image)
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    st.write("Image shape:", image_bgr.shape)
    st.write("Image channels:", image_bgr.shape[2] if len(image_bgr.shape) > 2 else 1)
    st.write("✅ Image successfully loaded and preprocessed.")

    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, color=(0, 255, 0))

    # Detect facial landmarks
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image_bgr,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,  # Hide dots
                    connection_drawing_spec=drawing_spec  # Show mesh lines
                )

            # Convert back to RGB for display
            output_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            st.image(output_image, caption="Facial Landmarks Detected", use_container_width=True)

        else:
            st.warning("No facial landmarks detected in the image.")
            st.image(image, caption="No Landmarks Detected", use_container_width=True)
