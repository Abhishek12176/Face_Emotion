#%%
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
import av
import tensorflow as tf
model = tf.keras.models.load_model("emotion_model.h5", compile=False)

@st.cache_resource
def load_emotion_model():
    return load_model("emotion_model.h5", compile=False)

model = load_emotion_model()

emotion_names = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
    'Surprise'
]

face_detect = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

st.title("Face Detection Project")

user_name = st.text_input("Enter Your Name")

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }
)

if user_name:

    st.success(f"Welcome {user_name}")

    class EmotionDetector(VideoProcessorBase):

        def recv(self, frame):

            img = frame.to_ndarray(format="bgr24")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_detect.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:

                face = gray[y:y+h, x:x+w]

                face = cv2.resize(face, (48,48))

                face = face / 255.0

                face = np.reshape(face, (1,48,48,1))

                result = model.predict(face, verbose=0)

                emotion_index = np.argmax(result)

                emotion = emotion_names[emotion_index]

                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

                cv2.putText(
                    img,
                    emotion,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,0,0),
                    2
                )

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="emotion-detection",
        video_processor_factory=EmotionDetector,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )
