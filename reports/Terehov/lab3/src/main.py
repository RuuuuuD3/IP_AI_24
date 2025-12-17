import os
import cv2
import shutil
import matplotlib.pyplot as plt
import numpy as np
from roboflow import Roboflow
from ultralytics import YOLO
import torch

ROBOFLOW_API_KEY = "C2T3vnzfNqLPVi0C7GRf"
WORKSPACE_ID = "leo-ueno"
PROJECT_ID = "people-detection-o4rdr"
VERSION_NUMBER = 10
DATASET_FORMAT = "yolov8"
DATASET_FOLDER = 'People-Detection-10'
data_yaml_path = ""

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

print("--- 1. Подготовка и Загрузка Данных Roboflow ---")
print(f"Используемое устройство для обучения: {device}")
try:
    full_path_to_delete = os.path.join(os.getcwd(), DATASET_FOLDER)
    if os.path.exists(full_path_to_delete):
        print(f"🧹 Удаление старой папки: {full_path_to_delete}")
        shutil.rmtree(full_path_to_delete)

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)

    dataset = project.version(VERSION_NUMBER).download(DATASET_FORMAT)

    data_yaml_path = os.path.join(os.getcwd(), DATASET_FOLDER, "data.yaml")

    if not os.path.exists(data_yaml_path):
        data_yaml_path = os.path.join(os.getcwd(), DATASET_FOLDER, "yolov8", "data.yaml")

    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"Файл data.yaml не найден даже после повторного скачивания.")

    print(f"✅ Путь к конфигу датасета: {data_yaml_path}")

except Exception as e:
    print(f"❌ Критическая ошибка при подготовке данных: {e}")
    exit(1)

print("\n--- 2. Инициализация и Обучение YOLOv10n ---")
model = YOLO('yolov10n.pt')

print(f"🚀 Начинаем обучение на устройстве: {device}")
try:
    results = model.train(
        data=data_yaml_path,
        epochs=20,
        imgsz=640,
        batch=16,
        name='yolov10_m1_final_run',
        device=device
    )
except Exception as e:
    print(f"❌ Ошибка обучения. Попробуйте заменить 'device={device}' на 'device='cpu'.")
    print(e)
    exit(1)

print("\n--- 3. Валидация модели ---")
metrics = model.val()
print(f"mAP@50: {metrics.box.map50:.4f}")
print(f"mAP@50-95: {metrics.box.map:.4f}")

print("\n--- 4. Визуализация на вашем фото ---")

test_image_path = input("📸 Введите полный путь к вашему тестовому фото (например: /Users/user/Desktop/my_photo.jpg): ")

try:
    if not os.path.exists(test_image_path):
        raise FileNotFoundError(f"Файл не найден по пути: {test_image_path}")

    output_name = 'local_predict_custom'

    print(f"🔍 Запускаем предсказание для фото: {test_image_path}")

    results_predict = model.predict(source=test_image_path, conf=0.25, save=True, name=output_name, exist_ok=True)

    if results_predict and results_predict[0].save_dir:
        image_filename = os.path.basename(test_image_path)
        output_image_path = os.path.join(results_predict[0].save_dir, image_filename)

        result_img_bgr = cv2.imread(output_image_path)
        if result_img_bgr is not None:
            result_img_rgb = cv2.cvtColor(result_img_bgr, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(12, 8))
            plt.imshow(result_img_rgb)
            plt.axis('off')
            plt.title("Результат детекции YOLOv10n (Ваше фото)")
            plt.show()

    print(f"✅ Результаты детекции сохранены в runs/detect/{output_name}")

except FileNotFoundError as e:
    print(f"❌ Ошибка: {e}. Пожалуйста, убедитесь, что путь указан правильно.")
except Exception as e:
    print(f"❌ Неизвестная ошибка при визуализации: {e}")