
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Отключает GPU

# -----------------------
# Настройки
# -----------------------
IMG_SIZE = (128, 128)       # размер изображений
BATCH_SIZE = 32
EPOCHS = 20                 # увеличено для плавного графика
TRAIN_DIR = 'train' # путь к обучающей выборке
VAL_DIR = 'val'     # путь к валидационной выборке
MODEL_FILE = 'model.h5'     # куда сохраняем модель

# -----------------------
# Аугментация данных
# -----------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,          # нормализация
    rotation_range=20,       # случайный поворот
    width_shift_range=0.2,   # сдвиг по ширине
    height_shift_range=0.2,  # сдвиг по высоте
    zoom_range=0.2,          # случайный зум
    horizontal_flip=True     # зеркальное отражение
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_ds = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_ds = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# -----------------------
# Создание CNN
# -----------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # бинарная классификация
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# -----------------------
# EarlyStopping для предотвращения переобучения
# -----------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# -----------------------
# Обучение
# -----------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# -----------------------
# Сохранение модели
# -----------------------
model.save(MODEL_FILE)
print(f"Модель сохранена в {MODEL_FILE}")

# -----------------------
# Визуализация результатов
# -----------------------
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Точность обучения')
plt.xlabel('Эпоха')
plt.ylabel('Accuracy')
plt.legend()

plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Training Metrics')
plt.xlabel('Epoch')
plt.ylabel('Accuracy / Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_plot.png')  # сохраняем график
#plt.show()  # покажет график, если есть GUI, иначе можно закомментировать