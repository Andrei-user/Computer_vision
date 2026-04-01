"""
Детекция ежевики (YOLOv8 + цветовая сегментация)

Основная идея:
- YOLO ищет кандидатов (любые объекты)
- затем фильтруем по цвету и яркости
- строим маску только для ягод

Установка:
    pip install ultralytics opencv-python numpy
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO


# ─────────────────────────────────────────────
# НАСТРОЙКИ
# ─────────────────────────────────────────────

MODEL_NAME = "yolov8n.pt"

# Порог уверенности YOLO (низкий → больше кандидатов)
DEFAULT_CONF = 0.15

# Минимальная доля "ягодных" пикселей в bbox
MIN_BERRY_MASK_RATIO = 0.08

# HSV диапазон для ежевики
BERRY_HSV_LOWER = np.array([100, 60, 20])
BERRY_HSV_UPPER = np.array([160, 255, 140])


# ─────────────────────────────────────────────
# ЗАГРУЗКА МОДЕЛИ
# ─────────────────────────────────────────────

def load_model():
    """Загрузка YOLO модели"""
    print("Загрузка YOLO...")
    return YOLO(MODEL_NAME)


# ─────────────────────────────────────────────
# ФИЛЬТР ТЁМНЫХ ОБЛАСТЕЙ
# ─────────────────────────────────────────────

def dark_mask(hsv):
    """
    Возвращает маску тёмных пикселей.
    
    Почему это важно:
    Ежевика почти чёрная → низкая яркость (V канал)
    """
    v = hsv[:, :, 2]
    return cv2.inRange(v, 0, 120)


# ─────────────────────────────────────────────
# ГЛОБАЛЬНАЯ МАСКА (без YOLO)
# ─────────────────────────────────────────────

def berry_mask_full(bgr):
    """
    Строит маску по всему изображению.
    Используется если YOLO ничего не нашла.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Цветовая фильтрация
    color = cv2.inRange(hsv, BERRY_HSV_LOWER, BERRY_HSV_UPPER)

    # Фильтр тёмных областей
    dark = dark_mask(hsv)

    # Объединение условий
    mask = cv2.bitwise_and(color, dark)

    # Удаление шума
    mask = cv2.medianBlur(mask, 5)

    # Морфология
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=2)

    return mask


# ─────────────────────────────────────────────
# ОЦЕНКА BBOX
# ─────────────────────────────────────────────

def bbox_berry_ratio(bgr, x1, y1, x2, y2):
    """
    Считает долю пикселей, похожих на ягоду, внутри bbox.
    
    Используется чтобы отфильтровать ложные детекции YOLO.
    """
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    color = cv2.inRange(hsv, BERRY_HSV_LOWER, BERRY_HSV_UPPER)
    dark = dark_mask(hsv)

    mask = cv2.bitwise_and(color, dark)

    # сглаживание шума
    mask = cv2.medianBlur(mask, 5)

    return np.count_nonzero(mask) / mask.size


# ─────────────────────────────────────────────
# МАСКА ВНУТРИ BBOX
# ─────────────────────────────────────────────

def build_berry_mask(bgr, x1, y1, x2, y2):
    """
    Строит точную маску ягод внутри bbox.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    color = cv2.inRange(hsv, BERRY_HSV_LOWER, BERRY_HSV_UPPER)
    dark = dark_mask(hsv)

    mask = cv2.bitwise_and(color, dark)

    # ограничиваем область bbox
    roi = np.zeros_like(mask)
    roi[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

    # очистка
    roi = cv2.medianBlur(roi, 5)

    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, k1, iterations=2)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, k2, iterations=2)

    return roi


# ─────────────────────────────────────────────
# ОСНОВНАЯ ДЕТЕКЦИЯ
# ─────────────────────────────────────────────

def detect_frame(model, bgr, conf=DEFAULT_CONF):
    """
    Основная функция детекции.

    Возвращает:
    - изображение с bbox
    - маску ягод
    - список найденных объектов
    """
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    results = model.predict(bgr, conf=conf, device="cpu", verbose=False)[0]

    annotated = bgr.copy()
    detections = []

    # если YOLO ничего не нашла
    if results.boxes is None or len(results.boxes) == 0:
        mask = berry_mask_full(bgr)
        return annotated, mask, detections

    # обработка bbox
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        ratio = bbox_berry_ratio(bgr, x1, y1, x2, y2)

        # фильтр
        if ratio < MIN_BERRY_MASK_RATIO:
            continue

        bbox_mask = build_berry_mask(bgr, x1, y1, x2, y2)
        mask = cv2.bitwise_or(mask, bbox_mask)

        detections.append((x1, y1, x2, y2))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # fallback
    if len(detections) == 0:
        mask = berry_mask_full(bgr)

    return annotated, mask, detections


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str)
    args = parser.parse_args()

    model = load_model()

    bgr = cv2.imread(args.image)
    ann, mask, dets = detect_frame(model, bgr)

    print("Найдено объектов:", len(dets))

    cv2.imshow("Result", np.hstack([bgr, ann, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))
    cv2.waitKey(0)


if __name__ == "__main__":
    main()