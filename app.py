import streamlit as st
import cv2
import numpy as np
import os

st.title("Fun AR Filters with Webcam")
st.write("""
Experience **Augmented Reality (AR)** in real-time! 
Apply Crowns, Glasses, Mustaches, Funny Face Images, and Cartoon Overlays directly on your face using your webcam.
""")
st.markdown("""
## Why Try Our AR Filters?

Experience the magic of **Augmented Reality (AR)** right from your webcam! Here are some benefits:

<ul>
<li><strong>Fun & Engagement:</strong> Instantly add Crowns, Glasses, Mustaches, or Cartoon overlays for a playful experience.</li>
<li><strong>Creativity Boost:</strong> Experiment with different looks and funny faces, perfect for social media content or selfies.</li>
<li><strong>Real-Time Interaction:</strong> See AR effects applied live, improving your understanding of AR technology.</li>
<li><strong>Learning Opportunity:</strong> Explore computer vision and AR in a hands-on, visual way.</li>
<li><strong>Memory Capture:</strong> Snap and save your creative moments automatically.</li>
</ul>
""", unsafe_allow_html=True)


st.sidebar.title("Select Filters / Overlays")
filters = st.sidebar.multiselect(
    "Choose Filters / Effects",
    ["Glasses", "Mustache", "Crown", "Funny Face Image", "Cartoon Overlay"]
)

save_snapshots = st.sidebar.checkbox("Enable Auto-Save Snapshots")
show_face_outline = st.sidebar.checkbox(
    "Highlight My Face",
    help="Draws a green outline around your face in real-time for better AR visualization."
)


if "Funny Face Image" in filters:
    funny_face_img_path = st.sidebar.selectbox("Choose Funny Image", ["overwhelming.png", "funny.png", "serious.png"])
else:
    funny_face_img_path = None

if "Cartoon Overlay" in filters:
    cartoon_img_path = st.sidebar.selectbox("Choose Cartoon Overlay", ["mickey.png", "motto.png"])
else:
    cartoon_img_path = None

if not os.path.exists("snapshots"):
    os.makedirs("snapshots")

# ---------------- Helper Functions ----------------
def overlay_filter(frame, filter_img, x, y, w, h):
    filter_img = cv2.resize(filter_img, (w, h))
    if filter_img.shape[2] < 4:
        return frame
    for i in range(h):
        for j in range(w):
            if y+i >= frame.shape[0] or x+j >= frame.shape[1]:
                continue
            alpha = filter_img[i,j,3]/255.0
            frame[y+i, x+j] = alpha*filter_img[i,j,:3] + (1-alpha)*frame[y+i, x+j]
    return frame

# ---------------- Load Haar Cascade ----------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---------------- Streamlit Image Placeholder ----------------
FRAME_WINDOW = st.image([])

# ---------------- Webcam Loop ----------------
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        st.error("Failed to access camera")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Predefined Filters
         if show_face_outline:
            color = (0, 255, 0)  # Green
            thickness = 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            # Optional: smoother ellipse contour
            center = (x + w // 2, y + h // 2)
            axes = (w // 2, h // 2)
            cv2.ellipse(frame, center, axes, 0, 0, 360, color, thickness)
         if "Glasses" in filters:
            f_img = cv2.imread("filters/glasses.png", cv2.IMREAD_UNCHANGED)
            if f_img is not None:
                y1 = y + int(h/4)
                frame = overlay_filter(frame, f_img, x, y1, w, int(h/4))

         if "Mustache" in filters:
            f_img = cv2.imread("filters/mustache.png", cv2.IMREAD_UNCHANGED)
            if f_img is not None:
                y1 = y + int(h*0.6)
                frame = overlay_filter(frame, f_img, x + int(w/4), y1, int(w/2), int(h/5))

         if "Crown" in filters:
            f_img = cv2.imread("filters/crown.png", cv2.IMREAD_UNCHANGED)
            if f_img is not None:
                y1 = max(0, y - int(h/2.5))
                frame = overlay_filter(frame, f_img, x, y1, w, int(h/2))

        # Funny Face Image Overlay
         if "Funny Face Image" in filters and funny_face_img_path is not None:
            f_img = cv2.imread(f"funny_images/{funny_face_img_path}", cv2.IMREAD_UNCHANGED)
            if f_img is not None:
                frame = overlay_filter(frame, f_img, x, y, w, h)

        # Cartoon Overlay
         if "Cartoon Overlay" in filters and cartoon_img_path is not None:
            f_img = cv2.imread(f"cartoons/{cartoon_img_path}", cv2.IMREAD_UNCHANGED)
            if f_img is not None:
                frame = overlay_filter(frame, f_img, x, y, w, h)

    FRAME_WINDOW.image(frame, channels="BGR")

    if save_snapshots:
        cv2.imwrite("snapshots/webcam_snapshot.png", frame)