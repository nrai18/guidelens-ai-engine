# perception.py

import numpy as np
import time
from collections import deque
import cv2
import config

class ObjectMemory:
    """Stores and retrieves information about detected objects over time."""
    def __init__(self, memory_duration=30):
        self.memory_duration = memory_duration
        self.stored_objects = {}
        self.last_seen = {}

    def update_object(self, obj_name, bbox, confidence):
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        current_time = time.time()
        self.stored_objects[obj_name] = { 'center': center, 'bbox': bbox, 'confidence': confidence, 'timestamp': current_time }
        self.last_seen[obj_name] = current_time

    def get_object(self, obj_name):
        self.cleanup_old_objects()
        if obj_name in self.stored_objects: return self.stored_objects[obj_name]
        return None

    def cleanup_old_objects(self):
        current_time = time.time()
        expired_objects = [obj for obj, last_time in self.last_seen.items() if current_time - last_time > self.memory_duration]
        for obj in expired_objects:
            if obj in self.stored_objects: del self.stored_objects[obj]
            if obj in self.last_seen: del self.last_seen[obj]

class PerceptionSystem:
    """Handles object detection and floor segmentation (no depth)."""
    def __init__(self, yolo_model, sam_predictor, ocr_reader, object_memory):
        self.yolo_model = yolo_model
        self.sam_predictor = sam_predictor
        self.ocr_reader = ocr_reader
        self.object_memory = object_memory
        self.prev_floor_mask = None
        self.mask_ema = None
        self.bbox_hist = deque(maxlen=getattr(config, "DETECTION_HISTORY", 5))

    def _median_box(self):
        if not self.bbox_hist:
            return None
        arr = np.array(self.bbox_hist)
        return np.median(arr, axis=0).tolist()

    def _postprocess_floor(self, mask, H, W):
        # Ensure mask is 2D (H, W)
        if mask.ndim == 3:
            mask = mask[0] if mask.shape[0] == 1 else mask.squeeze()
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
        mask_u8 = (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask
        num, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
        if num <= 1:
            return (mask_u8 > 0)
        band_y0 = int(H * (1 - config.MASK_BAND_HEIGHT_FRAC))
        best_label, best_area = 0, -1
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if (y + h >= band_y0) and area > best_area:
                best_area, best_label = area, i
        floor = (labels == best_label)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        floor = cv2.morphologyEx(floor.astype(np.uint8), cv2.MORPH_CLOSE, k, iterations=2).astype(bool)
        floor = cv2.morphologyEx(floor.astype(np.uint8), cv2.MORPH_OPEN, k, iterations=1).astype(bool)
        return floor

    def _ema_mask(self, mask_bool):
        if self.mask_ema is None:
            self.mask_ema = mask_bool.astype(np.float32)
        else:
            a = config.MASK_EMA_ALPHA
            self.mask_ema = a * mask_bool.astype(np.float32) + (1 - a) * self.mask_ema
        return (self.mask_ema > 0.5)

    def _floor_confidence(self, mask_bool, H):
        band_h = int(H * config.MASK_BAND_HEIGHT_FRAC)
        band = mask_bool[H - band_h : H, :]
        return float(band.mean()) if band_h > 0 else float(mask_bool.mean())

    def detect_floor(self, image_np, negative_box=None):
        """MobileSAM with positive bottom points and optional negative point at target center."""
        self.sam_predictor.set_image(image_np)
        H, W, _ = image_np.shape
        input_points = np.array([[W//2, H-5], [int(W*0.25), H-5], [int(W*0.75), H-5]])
        input_labels = np.array([1, 1, 1])

        if negative_box is not None:
            x1, y1, x2, y2 = negative_box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            input_points = np.concatenate([input_points, np.array([[cx, cy]])], axis=0)
            input_labels = np.concatenate([input_labels, np.array([0])], axis=0)  # negative point

        masks, _, _ = self.sam_predictor.predict(
            point_coords=input_points, point_labels=input_labels, multimask_output=False
        )
        return masks[0]  # always return 2D mask

    def process_frame(self, image_pil, last_command=None):
        target_box = None
        image_np = np.array(image_pil)  # RGB
        H, W, _ = image_np.shape

        # 1) YOLO detection with temporal smoothing
        results = self.yolo_model.predict(image_pil, verbose=False)
        results = self.yolo_model.predict(image_pil, verbose=False)
        r = results[0]
        if len(r.boxes) > 0:
            best_idx = int(r.boxes.conf.argmax().item())
            confidence = float(r.boxes.conf[best_idx].item())
            if confidence >= config.CONFIDENCE_THRESHOLD:
                box = r.boxes.xyxy[best_idx].cpu().numpy().tolist()
                self.bbox_hist.append(box)
                if len(self.bbox_hist) >= config.YOLO_STABLE_FRAMES:
                    target_box = self._median_box()
                    self.object_memory.update_object(config.TARGET_OBJECT, target_box, confidence)

        # 2) OCR
        ocr_results = self.ocr_reader.readtext(image_np)

        # 3) Floor segmentation + post-processing + EMA
        raw_floor = self.detect_floor(image_np, negative_box=target_box)
        floor_pp = self._postprocess_floor(raw_floor, H, W)
        floor_mask = self._ema_mask(floor_pp)
        floor_conf = self._floor_confidence(floor_mask, H)

        return {
            "target_box": target_box,
            "floor_mask": floor_mask,
            "floor_confidence": floor_conf,
            "last_command": last_command,
            "text_results": ocr_results,
        }