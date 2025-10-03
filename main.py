# main.py

import cv2
import threading
import queue
import time
import numpy as np
from PIL import Image

import config
from model_loader import load_models
from perception import PerceptionSystem, ObjectMemory
from navigation import NavigationSystem
from visualization import visualize_output

def inference_thread_func(frame_queue, result_queue, perception_system, navigation_system, stop_event):
    """Thread for running inference and navigation logic."""
    last_command = None

    while not stop_event.is_set():
        try:
            frame_bgr = frame_queue.get(timeout=1)
            
            frame_resized = cv2.resize(frame_bgr, config.PROCESSING_SIZE)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)

            H, W, _ = frame_resized.shape
            user_pos = (W // 2, H - 20)
            
            # --- Perception ---
            perception_data = perception_system.process_frame(frame_pil, last_command)
            
            # --- Navigation ---
            command, system_state, _, path = navigation_system.get_navigation_command(perception_data, user_pos)
            
            last_command = command
            
            result = {
                "frame": frame_resized,
                "perception_data": perception_data,
                "command": command,
                "system_state": system_state,
                "a_star_path": path,
                # "inflated_mask": inflated_mask,  # if exposed from nav
            }

            if result_queue.full():
                result_queue.get_nowait()
            result_queue.put(result)

        except queue.Empty:
            continue

def main():
    """Main function to initialize and run the application."""
    yolo_model, sam_predictor, ocr_reader = load_models()
    
    object_memory = ObjectMemory(config.OBJECT_MEMORY_DURATION)
    perception_system = PerceptionSystem(yolo_model, sam_predictor, ocr_reader, object_memory)
    navigation_system = NavigationSystem(object_memory)

    frame_queue = queue.Queue(maxsize=config.CAMERA_BUFFER_SIZE)
    result_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    print(f"Attempting to connect to camera at {config.CAMERA_URL}...")
    cap = cv2.VideoCapture(config.CAMERA_URL)
    if not cap.isOpened():
        print(f"Error: Could not open camera stream.")
        return

    print("Camera connected successfully.")
    
    inf_thread = threading.Thread(
        target=inference_thread_func,
        args=(frame_queue, result_queue, perception_system, navigation_system, stop_event),
        daemon=True
    )
    inf_thread.start()

    latest_result = {}
    last_frame_time = time.time()
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                time.sleep(0.1)
                continue
            
            if not frame_queue.full():
                frame_queue.put(frame)
            
            try:
                latest_result = result_queue.get_nowait()
            except queue.Empty:
                pass
            
            if "frame" in latest_result:
                current_time = time.time()
                fps = 1.0 / (current_time - last_frame_time)
                last_frame_time = current_time

                viz_data = {
                    "command": latest_result.get("command", "Initializing..."),
                    "system_state": latest_result.get("system_state", "INIT"),
                    "target_box": latest_result["perception_data"].get("target_box"),
                    "floor_mask": latest_result["perception_data"].get("floor_mask"),
                    "unsafe_mask": latest_result["perception_data"].get("unsafe_mask"),
                    "inflated_mask": latest_result.get("inflated_mask"),
                    "a_star_path": latest_result.get("a_star_path"),
                    "fps": fps,
                    "text_results": latest_result["perception_data"].get("text_results"),
                    "floor_confidence": latest_result["perception_data"].get("floor_confidence"),
                }
                visualize_output(latest_result["frame"], viz_data)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print("Stopping application...")
        stop_event.set()
        inf_thread.join()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()