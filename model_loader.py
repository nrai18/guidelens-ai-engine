# model_loader.py (depth removed, return 3)
from ultralytics import YOLO
from mobile_sam import SamPredictor, sam_model_registry
import easyocr
import config

def load_models():
    """Loads the YOLO engine, the SAM model, and the EasyOCR reader."""
    print(f"Loading models to {config.DEVICE}...")
    yolo_model = YOLO(config.YOLO_MODEL_PATH)

    sam_model = sam_model_registry["vit_t"](checkpoint=config.SAM_CHECKPOINT_PATH)
    sam_model.to(device=config.DEVICE)
    sam_predictor = SamPredictor(sam_model)

    print("Loading EasyOCR reader...")
    use_gpu = config.DEVICE == "cuda"
    ocr_reader = easyocr.Reader(config.OCR_LANGUAGE, gpu=use_gpu)

    print("All models loaded successfully.")
    return yolo_model, sam_predictor, ocr_reader
