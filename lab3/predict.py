import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # отключаем GPU

MODEL_FILE = 'model.h5'
INF_DIR = 'inf'  # папка с новыми изображениями для предсказания

# -----------------------
# Загружаем модель
# -----------------------
model = tf.keras.models.load_model(MODEL_FILE)

# -----------------------
# Функция предсказания
# -----------------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128,128))
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)[0][0]
    label = 'LION' if pred>0.5 else 'ELEPHANT'
    print(f"{img_path} → {label} ({pred:.2f})")

# -----------------------
# Прогон всех изображений из папки
# -----------------------
for fname in os.listdir(INF_DIR):
    path = os.path.join(INF_DIR, fname)
    if os.path.isfile(path):
        predict_image(path)