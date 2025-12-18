from datetime import datetime
import numpy as np
from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import yaml

possible_model_paths = [
    "runs/detect/yolov12s_cats_gpu/weights/best.pt",
    "runs/detect/train/weights/best.pt",
    "best.pt",
    "yolov12s.pt"
]

BEST_MODEL_PATH = None
for model_path in possible_model_paths:
    if Path(model_path).exists():
        BEST_MODEL_PATH = model_path
        print(f"Модель найдена: {BEST_MODEL_PATH}")
        break

if not BEST_MODEL_PATH:
    possible_paths = list(Path(".").rglob("best.pt"))
    if possible_paths:
        BEST_MODEL_PATH = str(possible_paths[0])
        print(f"Модель найдена автоматически: {BEST_MODEL_PATH}")
    else:
        BEST_MODEL_PATH = "yolov12s.pt"
        print(f"Собственная модель не найдена, используем: {BEST_MODEL_PATH}")

try:
    model = YOLO(BEST_MODEL_PATH)
    print(f"Модель загружена успешно")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    model = YOLO("yolov12s.pt")

VIDEO_SOURCES = [
    "videos/cats1.mp4",
    "videos/cats2.mp4",
    "videos/cats3.mp4",
]

EXISTING_VIDEOS = []
for video in VIDEO_SOURCES:
    if Path(video).exists():
        EXISTING_VIDEOS.append(video)
        print(f"✓ Видео найдено: {video}")

trackers_dir = Path("custom_trackers")
trackers_dir.mkdir(exist_ok=True)

ultralytics_dir = Path("yolov12") if Path("yolov12").exists() else Path(".")
original_botsort = ultralytics_dir / "ultralytics" / "cfg" / "trackers" / "botsort.yaml"
original_bytetrack = ultralytics_dir / "ultralytics" / "cfg" / "trackers" / "bytetrack.yaml"


CONFIGS = {
    "botsort_default.yaml": {
        "source": str(original_botsort) if original_botsort.exists() else None,
        "config": {
            "tracker_type": "botsort",
            "track_high_thresh": 0.5,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.6,
            "track_buffer": 30,
            "match_thresh": 0.8,
            "fuse_score": True
        }
    },
    "botsort_reid.yaml": {
        "source": str(original_botsort) if original_botsort.exists() else None,
        "config": {
            "tracker_type": "botsort",
            "track_high_thresh": 0.5,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.6,
            "track_buffer": 60,
            "match_thresh": 0.8,
            "fuse_score": True,
            "with_reid": True,
            "proximity_thresh": 0.5,
            "appearance_thresh": 0.25
        }
    },
    "botsort_long_buffer.yaml": {
        "source": str(original_botsort) if original_botsort.exists() else None,
        "config": {
            "tracker_type": "botsort",
            "track_high_thresh": 0.5,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.6,
            "track_buffer": 120,
            "match_thresh": 0.8,
            "fuse_score": True
        }
    },
    "bytetrack_default.yaml": {
        "source": str(original_bytetrack) if original_bytetrack.exists() else None,
        "config": {
            "tracker_type": "bytetrack",
            "track_high_thresh": 0.6,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.7,
            "track_buffer": 30,
            "match_thresh": 0.9,
            "fuse_score": False,
            "min_box_area": 10
        }
    },
    "bytetrack_strict.yaml": {
        "source": str(original_bytetrack) if original_bytetrack.exists() else None,
        "config": {
            "tracker_type": "bytetrack",
            "track_high_thresh": 0.6,
            "track_low_thresh": 0.2,
            "new_track_thresh": 0.7,
            "track_buffer": 30,
            "match_thresh": 0.85,
            "fuse_score": False,
            "min_box_area": 50
        }
    }
}
for config_name, config_data in CONFIGS.items():
    config_path = trackers_dir / config_name

    if config_data["source"] and Path(config_data["source"]).exists():
        with open(config_data["source"], 'r') as f:
            original_config = yaml.safe_load(f)

        if original_config:
            original_config.update(config_data["config"])
            config_data["config"] = original_config

    with open(config_path, 'w') as f:
        yaml.dump(config_data["config"], f, default_flow_style=False)


results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

for video_path in EXISTING_VIDEOS:
    video_name = Path(video_path).stem
    print(f"\n{'=' * 50}")
    print(f"Обрабатываем видео: {video_path}")
    print(f"{'=' * 50}")

    for config_name in CONFIGS.keys():
        config_path = trackers_dir / config_name
        tracker_name = config_name.replace('.yaml', '')
        output_name = f"{video_name}_{tracker_name}"

        print(f"\n→ Трекер: {tracker_name}")

        try:
            results = model.track(
                source=video_path,
                tracker=str(config_path),
                save=True,
                project="results",
                name=output_name,
                exist_ok=True,
                imgsz=640,
                conf=0.4,
                iou=0.5,
                show=False,
                stream=False,
                persist=True,
                verbose=False,
                save_txt=True,
                save_conf=True
            )

            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    if result.boxes.id is not None:
                        unique_ids = len(set(result.boxes.id.cpu().numpy()))
                        print(f"   Обнаружено уникальных треков: {unique_ids}")
                    else:
                        print(f"   Обнаружено объектов: {len(result.boxes)}")

            print(f"   Результаты сохранены в: results/{output_name}/")

        except Exception as e:
            print(f"   Ошибка при трекинге: {e}")
            try:
                print(f"   Пробуем с дефолтным трекером...")
                results = model.track(
                    source=video_path,
                    save=True,
                    project="results",
                    name=f"{video_name}_default",
                    exist_ok=True,
                    imgsz=640,
                    conf=0.4,
                    iou=0.5,
                    show=False,
                    stream=False,
                    persist=True,
                    verbose=False
                )
            except Exception as e2:
                print(f"   Ошибка с дефолтным трекером: {e2}")