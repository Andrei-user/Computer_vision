"""
Применение обученной модели для классификации изображения
"""

import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.preprocessing import image

IMG_SIZE = (128, 128)

# Загружаем модель
model = tf.keras.models.load_model("model.keras")

# Классы (важно: порядок как в train/)
class_names = ["elephant", "lion"]

def predict(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        print(f"{img_path} → LION 🦁 ({prediction:.2f})")
    else:
        print(f"{img_path} → ELEPHANT 🐘 ({1-prediction:.2f})")


# Тест
predict("inf/lion.jpg")
predict("inf/elephant.jpg")