import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

def load_data():
    images, labels = [], []
    label_map = {}
    label_count = 0

    for folder in os.listdir("dataset"):
        label_map[label_count] = folder
        folder_path = os.path.join("dataset", folder)
        
        for image_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, image_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (100, 100))
            images.append(img)
            labels.append(label_count)

        label_count += 1

    images = np.array(images).reshape(-1, 100, 100, 1) / 255.0
    labels = np.array(labels)

    return train_test_split(images, labels, test_size=0.2), label_map

(X_train, X_test, y_train, y_test), label_map = load_data()

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100,100,1)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

model.save("models/face_recognition_model.h5")
np.save("models/label_map.npy", label_map)
print("Model trained and saved!")
