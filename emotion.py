# %%

import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

model = load_model("emotion_model.h5")

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

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Webcam Not Opened")
    exit()

# Stability variables define here (before loop)
current_emotion = None
last_update_time = time.time()

while True:
    check, frame = camera.read()
    if not check:
        print("Frame Not Captured")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48))
        face = face / 255.0
        face = np.reshape(face, (1,48,48,1))

        result = model.predict(face)
        emotion_index = np.argmax(result)
        emotion = emotion_names[emotion_index]

        #  Stability logic
        if current_emotion is None or (emotion == current_emotion):
            current_emotion = emotion
            last_update_time = time.time()
        else:
            if time.time() - last_update_time > 1:  # 1 sec hold
                current_emotion = emotion
                last_update_time = time.time()

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, current_emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

# %%
