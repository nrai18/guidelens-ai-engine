# export_chair_engine.py
from ultralytics import YOLO
from pathlib import Path
import config

print("Loading original YOLO-World .pt model...")
model = YOLO("models/yolov8s-worldv2.pt")

print(f"Setting target class to: ['{config.TARGET_OBJECT}']")
model.set_classes([config.TARGET_OBJECT])

engine_path = Path(config.BASE_DIR) / "models" / f"{config.TARGET_OBJECT}.engine"

print(f"Exporting to TensorRT engine at {engine_path}...")
model.export(format="engine", half=True, imgsz=config.PROCESSING_SIZE[0])

# --- FIX ---
# The Ultralytics library saves the engine in the project's root `models` folder,
# not relative to this script's location.
default_engine_name = "yolov8s-worldv2.engine"
source_engine_path = config.BASE_DIR.parent / "models" / default_engine_name

print(f"Renaming {source_engine_path} to {engine_path}")
source_engine_path.rename(engine_path)

print(f"✅ Engine for '{config.TARGET_OBJECT}' created successfully.")