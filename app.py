import streamlit as st
import cv2
import numpy as np
import os

st.title("Fun AR Filters with Webcam")

st.write("""
Experience **Augmented Reality (AR)** in real-time!
Apply Crowns, Glasses, Mustaches, Funny Face Images, and Cartoon Overlays.
""")

st.sidebar.title("Select Filters / Overlays")

filters = st.sidebar.multiselect(
    "Choose Filters / Effects",
    ["Glasses", "Mustache", "Crown", "Funny Face Image", "Cartoon Overlay"]
)

show_face_outline = st.sidebar.checkbox("Highlight My Face")

# Image selections
funny_face_img_path = None
cartoon_img_path = None

if "Funny Face Image" in filters:
    funny_face_img_path = st.sidebar.selectbox(
        "Choose Funny Image",
        ["overwhelming.png", "funny.png", "serious.png"]
    )

if "Cartoon Overlay" in filters:
    cartoon_img_path = st.sidebar.selectbox(
        "Choose Cartoon Overlay",
        ["mickey.png", "motto.png"]
    )

# ---------------- Overlay Function ----------------
def overlay_filter(frame, filter_img, x, y, w, h):

    filter_img = cv2.resize(filter_img, (w, h))

    if filter_img.shape[2] < 4:
        return frame

    for i in range(h):
        for j in range(w):

            if y+i >= frame.shape[0] or x+j >= frame.shape[1]:
                continue

            alpha = filter_img[i, j, 3] / 255.0

            frame[y+i, x+j] = (
                alpha * filter_img[i, j, :3] +
                (1 - alpha) * frame[y+i, x+j]
            )

    return frame


# ---------------- Face Detection ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ---------------- Camera Capture ----------------
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:

    bytes_data = img_file_buffer.getvalue()
    np_array = np.frombuffer(bytes_data, np.uint8)

    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        if show_face_outline:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        if "Glasses" in filters:
            img = cv2.imread("filters/glasses.png", cv2.IMREAD_UNCHANGED)
            if img is not None:
                frame = overlay_filter(frame, img, x, y+int(h/4), w, int(h/4))

        if "Mustache" in filters:
            img = cv2.imread("filters/mustache.png", cv2.IMREAD_UNCHANGED)
            if img is not None:
                frame = overlay_filter(frame, img, x+int(w/4), y+int(h*0.6), int(w/2), int(h/5))

        if "Crown" in filters:
            img = cv2.imread("filters/crown.png", cv2.IMREAD_UNCHANGED)
            if img is not None:
                frame = overlay_filter(frame, img, x, y-int(h/2), w, int(h/2))

        if "Funny Face Image" in filters and funny_face_img_path:
            img = cv2.imread(f"funny_images/{funny_face_img_path}", cv2.IMREAD_UNCHANGED)
            if img is not None:
                frame = overlay_filter(frame, img, x, y, w, h)

        if "Cartoon Overlay" in filters and cartoon_img_path:
            img = cv2.imread(f"cartoons/{cartoon_img_path}", cv2.IMREAD_UNCHANGED)
            if img is not None:
                frame = overlay_filter(frame, img, x, y, w, h)

    st.image(frame, channels="BGR")
