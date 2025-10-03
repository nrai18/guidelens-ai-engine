# config.py

import torch
from pathlib import Path

# --- Project Root Configuration ---
BASE_DIR = Path(__file__).parent.resolve()

# --- System Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROCESSING_SIZE = (640, 480)
CAMERA_URL = "http://10.51.142.11:8080/videofeed" # Your camera URL

# --- Model & Detection Configuration ---
YOLO_MODEL_PATH = BASE_DIR / "models" / "chair.engine"
SAM_CHECKPOINT_PATH = BASE_DIR / "MobileSAM/weights/mobile_sam.pt"
TARGET_OBJECT = "chair" # The navigation goal is now an object.
CONFIDENCE_THRESHOLD = 0.05

# --- OCR Configuration ---
# OCR is now for general text recognition, not targeting.
OCR_CONFIDENCE_THRESHOLD = 0.4
OCR_LANGUAGE = ['en']

# --- Performance & Threading Configuration ---
MAX_FPS = 30
CAMERA_BUFFER_SIZE = 1
INFERENCE_INTERVAL = 0.5

# --- Navigation & Logic Configuration ---
PATH_LOOKAHEAD_INDEX = 4 # Aim for the 4th point in the path
OBJECT_MEMORY_DURATION = 15
PLANNING_DURATION = 2.0
STUCK_TIME_THRESHOLD = 3.0
PATHFINDING_GRID_SCALE = 10
PATHFINDING_FALLBACK_RADIUS = 75
INSTRUCTION_LOCK_DURATION = 1.5
MIN_PATH_LENGTH_FOR_MOVE = 3
APPROACH_DISTANCE = 30

# --- Safety & inflation ---
SAFETY_CLEARANCE_PX = 20         # required min clearance along path
INFLATION_RADIUS_PX = 15         # shrink walkable area near obstacles (via distance transform threshold)
PATH_SMOOTHING_ITERS = 1         # Chaikin iterations for path smoothing

# --- Perception smoothing ---
MASK_EMA_ALPHA = 0.6             # EMA for floor mask stability
MASK_BAND_HEIGHT_FRAC = 0.15     # bottom band for floor confidence
MASK_MIN_CONFIDENCE = 0.55       # stop if floor confidence below this
DETECTION_HISTORY = 5            # median of last N boxes
YOLO_STABLE_FRAMES = 2           # require N stable frames before memory update

# # --- Depth hazard gating (MiDaS) ---
# DEPTH_ENABLE = True
# MIDAS_MODEL_PATH = "models/midas_v21_small.onnx"
# MIDAS_INPUT_SIZE = (256, 256)    # MiDaS small typical input
# DEPTH_GRADIENT_THRESH = 0.12     # stop if forward gradient exceeds this
# DEPTH_NEAR_OBST_Z = 0.35         # relative inverse-depth threshold for near obstacle
# FORWARD_ROI_WIDTH_FRAC = 0.35    # forward corridor width for hazard check
# FORWARD_ROI_HEIGHT_FRAC = 0.25   # forward corridor height for hazard check

# --- Pure Pursuit ---
PP_LOOKAHEAD_PX_OPEN = 90
PP_LOOKAHEAD_PX_TIGHT = 40
PP_TIGHT_CLEARANCE = 28          # use tight lookahead when min clearance < this
PP_SLIGHT_TURN_K = 0.005         # curvature thresholds
PP_SHARP_TURN_K  = 0.015

# --- Instruction timing ---
INSTRUCTION_LOCK_DURATION = 0.8  # existing; keep adaptive in code

# Radius in pixels around the target where we consider it "reached"
TARGET_RADIUS_PX = 30   # adjust depending on your grid resolution
MIN_PATH_LENGTH_FOR_MOVE = 5  # also make sure this exists