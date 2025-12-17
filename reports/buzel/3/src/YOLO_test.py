from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

# --- ШАГ 1: ЗАГРУЗКА НАШЕЙ ОБУЧЕННОЙ МОДЕЛИ ---
model_path = 'yolov10s_vehicles_lab3/weights/best.pt'
model = YOLO(model_path)

# --- ШАГ 2: УКАЗЫВАЕМ ПУТЬ К ИЗОБРАЖЕНИЮ ДЛЯ ДЕТЕКЦИИ ---
image_path = 'D:/REALOIIS/lab3/images.jpg'

# --- ШАГ 3: ЗАПУСК ОБНАРУЖЕНИЯ ОБЪЕКТОВ ---
# Модель обработает изображение и найдет на нем объекты
# conf=0.25 - порог уверенности. 
results = model.predict(source=image_path, conf=0.25, save=True)

print("\nОбнаружение завершено.")
print("Результат сохранен.")

# --- ДОПОЛНИТЕЛЬНО: Показываем результат прямо в скрипте ---
if results and results[0].save_dir:
    result_image_path = f"{results[0].save_dir}/{results[0].path.split('/')[-1].split('\\')[-1]}"
    
    try:
        # Открываем сохраненное изображение и показываем его
        result_image = Image.open(result_image_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(result_image)
        plt.axis('off') # Убираем оси
        plt.show()
    except FileNotFoundError:
        print(f"Не удалось найти сохраненное изображение по пути: {result_image_path}")
