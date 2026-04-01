"""
Обучение полносвязной нейронной сети для классификации изображений
(лев vs слон)

Этапы:
1. Загрузка изображений
2. Предобработка (изменение размера, нормализация)
3. Построение модели Dense
4. Обучение
5. Сохранение модели в .keras
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import os
import matplotlib
matplotlib.use('Agg')   
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import matplotlib.pyplot as plt

# =========================
# ПАРАМЕТРЫ
# =========================
IMG_SIZE = (128, 128)   # размер изображений
BATCH_SIZE = 32

TRAIN_DIR = "train"
VAL_DIR = "val"

# =========================
# ЗАГРУЗКА ДАННЫХ
# =========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

# Нормализация (0..255 → 0..1)
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# =========================
# СОЗДАНИЕ МОДЕЛИ
# =========================
model = models.Sequential([
    layers.Flatten(input_shape=(128, 128, 3)),  # "сплющиваем" картинку
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # бинарная классификация
])

# Компиляция модели
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# ОБУЧЕНИЕ
# =========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30
)

# =========================
# СОХРАНЕНИЕ МОДЕЛИ
# =========================
model.save("model.keras")

print("Модель сохранена в model.keras")

# =========================
# ВИЗУАЛИЗАЦИЯ ОБУЧЕНИЯ
# =========================
plt.figure(figsize=(12, 5))

# Точность
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title("Accuracy")
plt.legend()

# Ошибка
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title("Loss")
plt.legend()

plt.savefig("training_plot.png")
print("График сохранен в training_plot.png")