from ultralytics import YOLO

if __name__ == '__main__':
    # --- ШАГ 1: ЗАГРУЗКА МОДЕЛИ ---
    model = YOLO('yolov10s.pt')

    # --- ШАГ 2: НАСТРОЙКА И ЗАПУСК ОБУЧЕНИЯ ---
    # model.train() - главная функция для обучения
    results = model.train(
        # --- Основные параметры ---
        data='D:/REALOIIS/lab3/yolo/dataset.yaml', # !Путь к файлу конфигурации
        
        # --- Параметры для potato ---
        epochs=15,         
        batch=4,          
        imgsz=320,          
        device='cpu',      
        
        # --- Дополнительные параметры ---
        name='yolov10s_vehicles_lab3', 
        exist_ok=True,      
        verbose=True     
    )

    # --- ШАГ 3: ОЦЕНКА МОДЕЛИ ---
    # Это и есть "Оценить эффективность обучения на тестовой выборке (mAP)"
    print("Запуск оценки на тестовом наборе данных...")
    results = model.val(data='D:/REALOIIS/lab3/yolo/dataset.yaml', split='test')
    
    print("Обучение и оценка завершены.")
    print("Результаты сохранены в папке 'yolov10s_vehicles_lab3'")
