# 🧠 Facial Landmark Detection Using MediaPipe  
A Simple Computer Vision Web App for Visualizing Facial Mesh

---

## 🧭 Overview  
This project implements real-time facial landmark detection using **MediaPipe Face Mesh** and displays facial mesh lines on portrait images via a **Streamlit web app**. Rather than predicting expressions or performing face recognition, the focus is on understanding facial structure through mesh overlay.

Key steps included:

- Image upload and preprocessing using Pillow and NumPy
- Facial landmark detection using MediaPipe's pre-trained FaceMesh model
- Visualization of facial mesh (lines only, no dots) using OpenCV
- Web app creation and deployment using Streamlit
- Optional enhancements: region highlighting (eyes, lips, etc.) and real-time webcam extension

---

## 📦 Input
- **Image input**: User-uploaded `.jpg`, `.jpeg`, or `.png` portrait image
- **Face detection model**: MediaPipe's built-in face mesh estimator
- **Output**: Original image with facial mesh lines drawn

---

## 🔍 Highlights
- Detects **468 facial landmarks** per face
- Supports **1 face per image** (can be extended)
- Draws only **mesh lines** (tesselation), without cluttering with dots
- Works well on **clear, front-facing images**
- Fully deployable as a **Streamlit web app**

---

## 🤖 How It Works
1. User uploads an image
2. Image is converted to RGB and passed to MediaPipe FaceMesh
3. If a face is detected:
   - Facial landmarks are returned
   - Connections (lines) are drawn using OpenCV
4. Final image is displayed using Streamlit

---

## 💡 Key Takeaways
- Facial mesh detection is possible without training any model
- MediaPipe is efficient and accurate for landmark extraction
- Streamlit is a powerful tool for quick deployment of ML/vision prototypes
- OpenCV and PIL can be used together for both preprocessing and overlaying

---

## 📊 Tools Used
- Python (NumPy, OpenCV, Pillow)
- MediaPipe
- Streamlit
- Git & GitHub

---

## 🚀 Live Demo
> Coming soon — hosted on Streamlit Cloud

---

## 👤 Author  
**Nafisat Ibrahim**

---

## 📥 Resources & Downloads
- 📄 [View Source Code (`app.py`)](./app.py)
- 📦 [View Requirements (`requirements.txt`)](./requirements.txt)
- 🛠️ [System Dependencies (`apt-packages`)](./apt-packages)

---
