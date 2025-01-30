import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/face_recognition_model.h5")
label_map = np.load("models/label_map.npy", allow_pickle=True).item()

def recognize_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (100, 100)).reshape(1, 100, 100, 1) / 255.0

        prediction = model.predict(face_img)
        class_index = np.argmax(prediction)
        name = label_map[class_index]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame, name if faces != () else "Unknown"

