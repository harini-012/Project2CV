import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("Fun AR Filters with Webcam")

# Sidebar
filters = st.sidebar.multiselect(
    "Choose Filters",
    ["Glasses", "Mustache", "Crown"]
)

show_face_outline = st.sidebar.checkbox("Highlight Face")


# Overlay function
def overlay_filter(frame, filter_img, x, y, w, h):

    filter_img = cv2.resize(filter_img, (w, h))

    if filter_img.shape[2] != 4:
        return frame

    alpha = filter_img[:, :, 3] / 255.0

    for c in range(3):
        frame[y:y+h, x:x+w, c] = (
            alpha * filter_img[:, :, c] +
            (1 - alpha) * frame[y:y+h, x:x+w, c]
        )

    return frame


# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


class VideoProcessor(VideoTransformerBase):

    def __init__(self):
        self.filters = filters
        self.show_face_outline = show_face_outline

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:

            if self.show_face_outline:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

            if "Glasses" in self.filters:

                glasses = cv2.imread("filters/glasses.png", cv2.IMREAD_UNCHANGED)

                if glasses is not None:
                    img = overlay_filter(img, glasses, x, y+int(h/4), w, int(h/4))

            if "Mustache" in self.filters:

                mustache = cv2.imread("filters/mustache.png", cv2.IMREAD_UNCHANGED)

                if mustache is not None:
                    img = overlay_filter(img, mustache, x+int(w/4), y+int(h*0.6), int(w/2), int(h/5))

            if "Crown" in self.filters:

                crown = cv2.imread("filters/crown.png", cv2.IMREAD_UNCHANGED)

                if crown is not None:
                    y1 = max(0, y-int(h/2))
                    img = overlay_filter(img, crown, x, y1, w, int(h/2))

        return img


webrtc_streamer(
    key="ar-camera",
    video_processor_factory=VideoProcessor
)
