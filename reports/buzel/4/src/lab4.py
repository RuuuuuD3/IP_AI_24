from ultralytics import YOLO

# Этот блок нужен для корректной работы на Windows
if __name__ == '__main__':
    # --- ШАГ 1: ЗАГРУЗКА ВАШЕЙ ОБУЧЕННОЙ МОДЕЛИ ---
    # Указываем путь к файлу best.pt, который мы получили в ЛР №3
    model_path = 'D:/REALOIIS/lab3/runs/detect/yolov10s_vehicles_lab3/weights/best.pt'
    model = YOLO(model_path)

    # --- ШАГ 2: ВЫБОР ИСХОДНОГО ВИДЕО ---
    # Замените 'traffic.mp4' на имя вашего видеофайла
    video_path = 'D:/REALOIIS/lab4/traffic.mp4'

    # --- ШАГ 3: ЗАПУСК ОТСЛЕЖИВАНИЯ ---
    # Используем метод model.track()
    # tracker='botsort.yaml' или 'bytetrack.yaml' - выбор алгоритма трекинга

    print("Начинаем отслеживание с помощью BoT-SORT...")
    results_botsort = model.track(
        source=video_path,
        tracker='botsort.yaml',  # Указываем конфигурационный файл трекера
        conf=0.3,                # Порог уверенности для детекции
        iou=0.5,                 # Порог IoU для non-max suppression
        device='cpu',            # Явно указываем использование CPU
        show=True,               # Отображать видео в реальном времени (может тормозить на CPU)
        save=True                # Сохранить видео с результатами трекинга
    )

    print("\nНачинаем отслеживание с помощью ByteTrack...")
    results_bytetrack = model.track(
        source=video_path,
        tracker='bytetrack.yaml', # Меняем трекер на ByteTrack
        conf=0.3,
        iou=0.5,
        device='cpu',
        show=True,
        save=True
    )

    print("\nОбработка завершена.")
    print("Результаты сохранены в папку 'runs/detect/track...'")