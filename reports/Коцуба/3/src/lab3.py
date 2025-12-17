from roboflow import Roboflow
import os
import requests
from io import BytesIO

rf = Roboflow(api_key="2OFRWtLUJqdrNuwZhhPk")
project = rf.workspace("koer3741-gmail-com").project("watermeteramrv2")
dataset = project.version(1).download("yolov8")
dataset_path = dataset.location
data_yaml_path = os.path.join(dataset_path, "data.yaml")
with open(data_yaml_path, 'r') as f:
    print("Содержимое data.yaml:")
    print(f.read())
train_images = len(os.listdir(os.path.join(dataset_path, "train", "images")))
val_images = len(os.listdir(os.path.join(dataset_path, "valid", "images")))
test_images = len(os.listdir(os.path.join(dataset_path, "test", "images"))) if os.path.exists(os.path.join(dataset_path, "test")) else 0
print(f"\nРазмеры датасета:")
print(f"Train: {train_images} изображений")
print(f"Valid: {val_images} изображений")
print(f"Test: {test_images} изображений")
print("\nКлассы: '0','1','2','3','4','5','6','7','8','9','counter','liters'")

from ultralytics import YOLO

model = YOLO("yolov10m.pt")
results = model.train(
    data=data_yaml_path,
    epochs=10,
    imgsz=640,
    batch=16,
    name="yolov10m_water_meter_detection",
    device=0
)

metrics = model.val(data=data_yaml_path)
print("Метрики валидации:")
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print("\nПолные метрики:")
print(metrics)

import supervision as sv
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO

internet_image_url = "https://baumarket.by/statics/catalog/product/img/6824/img/645ce7dc86114.jpg"
response = requests.get(internet_image_url)
img = Image.open(BytesIO(response.content))
img.save("internet_water_meter.jpg")
test_image_path = "internet_water_meter.jpg"

results = model(test_image_path)

plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')

for result in results:
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            class_name = model.names[cls]
            label = f"{class_name} {conf:.2f}"
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2))
            plt.text(x1, y1-10, label, color='red', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.title("Визуализация детекций на изображении из интернета")
plt.show()
result.save("detection_result.jpg")
print("Результат сохранен как detection_result.jpg")