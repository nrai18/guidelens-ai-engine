=========================================================
 GuideLens: A Real-Time Visual Assistance System
=========================================================

GuideLens is an AI-powered wearable system that helps visually 
impaired individuals navigate indoor environments safely. 
It detects target objects, identifies walkable floor areas, 
and provides real-time audio/haptic feedback for guidance.

---------------------------------------------------------
 Features
---------------------------------------------------------
- Target-Oriented Navigation: Guides user to specific objects 
  (e.g., door, chair).
- Safe Pathfinding: Uses floor segmentation and A* algorithm 
  for obstacle-free routes.
- Object Permanence: Remembers last-seen target locations.
- Multi-Modal Feedback: Clear voice commands + haptic vibrations.
- OCR Integration: Reads signs and labels from the environment.
- Optimized for real-time on Raspberry Pi 5 + Coral Accelerator.

---------------------------------------------------------
 Technology Stack
---------------------------------------------------------
Core AI Models:
- YOLO-World (zero-shot object detection, TensorRT optimized)
- MobileSAM (walkable floor segmentation)
- EasyOCR (text recognition)

Pathfinding:
- A* Algorithm for shortest safe paths
- Pure Pursuit Control for smooth guidance

Software & Tools:
- Python, PyTorch, OpenCV, Ultralytics, NumPy

Hardware:
- Raspberry Pi 5 (8GB RAM)
- Google Coral USB Accelerator
- Camera module (USB/IP)
- Audio + haptic output devices

---------------------------------------------------------
 Installation & Setup
---------------------------------------------------------
1. Clone this repository:
   git clone https://github.com/your-username/GuideLens.git
   cd GuideLens

2. Install dependencies:
   pip install -r requirements.txt

3. Download models:
   - Place yolov8s-worldv2.pt into models/
   - Place mobile_sam.pt into MobileSAM/weights/

4. Configure system:
   - Edit config.py to set TARGET_OBJECT and CAMERA_URL

5. Convert YOLO model to TensorRT:
   python export_chair_engine.py

6. Run application:
   python main.py

---------------------------------------------------------
 Demo
---------------------------------------------------------
- When started, the system detects the target object 
  and guides the user with voice and haptic feedback.
- Example: "Move Forward", "Turn Slightly Left".

---------------------------------------------------------
 Future Work
---------------------------------------------------------
- Outdoor navigation (GPS + Visual SLAM)
- Wearable AR glasses integration
- Cloud-based remote assistance

---------------------------------------------------------
 License
---------------------------------------------------------
This project is licensed under the MIT License.
