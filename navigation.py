# navigation.py

import cv2
import numpy as np
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from collections import deque
import config
import time
from scipy import interpolate

# --- helpers ---

def spline_smooth(path, num_points=100):
    """Smooth path with cubic B-spline interpolation."""
    if path is None or len(path) < 3:
        return path

    x = [p[0] for p in path]
    y = [p[1] for p in path]

    t = range(len(path))
    t_new = np.linspace(0, len(path)-1, num_points)

    spl_x = interpolate.CubicSpline(t, x)
    spl_y = interpolate.CubicSpline(t, y)

    smoothed = [(float(spl_x(tt)), float(spl_y(tt))) for tt in t_new]
    return smoothed

def path_length(path):
    if not path or len(path) < 2:
        return 0.0
    return float(sum(
    np.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
    for i in range(len(path)-1)
    ))


def pick_lookahead(path, user_pos, L):
    if not path:
        return None

    dists = [0.0]
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        dists.append(dists[-1] + np.hypot(dx, dy))

    for i in range(len(dists)):
        if dists[i] >= L:
            return path[i]

    return path[-1]


def curvature_cmd(user_pos, lookahead_pt, slight_k, sharp_k):
    dx = lookahead_pt[0] - user_pos[0]
    dy = lookahead_pt[1] - user_pos[1]

    y_body = -(dy)  # forward
    x_body = dx     # right
    if y_body <= 1e-6:
        return "Stop"
    L = np.hypot(x_body, y_body)
    kappa = (2.0 * x_body) / (L*L + 1e-6)
    ak = abs(kappa)
    if ak > sharp_k:
        return "Turn sharp right" if kappa > 0 else "Turn sharp left"
    elif ak > slight_k:
        return "Turn slightly right" if kappa > 0 else "Turn slightly left"
    else:
        return "Move Forward"

class StuckDetector:
    def __init__(self, time_threshold, pos_threshold=5):
        self.time_threshold = time_threshold
        self.pos_threshold = pos_threshold
        self.history = []
    def update(self, command, target_box):
        current_time = time.time()
        target_y = target_box[3] if target_box else -1
        self.history.append((current_time, command, target_y))
        self.history = [e for e in self.history if current_time - e[0] < self.time_threshold]
    def is_stuck(self):
        if not self.history:
            return False
        first_time = self.history[0][0]
        if time.time() - first_time < self.time_threshold:
            return False
        if not all(e[1] == "Move Forward" for e in self.history):
            return False
        start_y = self.history[0][2]
        end_y = self.history[-1][2]
        return (end_y - start_y) < self.pos_threshold

def find_nearest_walkable_node(target_pos, grid, max_radius=50):
    x0, y0 = int(target_pos[0]), int(target_pos[1])
    if grid.inside(x0, y0) and grid.node(x0, y0).walkable:
        return grid.node(x0, y0)

    visited = set()
    queue = deque([(x0, y0, 0)])
    while queue:
        x, y, dist = queue.popleft()
        if (x, y) in visited or dist > max_radius:
            continue
        visited.add((x, y))
        if grid.inside(x, y) and grid.node(x, y).walkable:
            return grid.node(x, y)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited:
                    queue.append((nx, ny, dist + 1))


class NavigationSystem:
    """Determines navigation commands based on perception data with safety & Pure Pursuit."""
    def __init__(self, object_memory):
        self.object_memory = object_memory
        self.stuck_detector = StuckDetector(config.STUCK_TIME_THRESHOLD)
        self.global_path = None
        self.last_command_time = 0
        self.last_command = "Initializing..."

    def _inflate_and_grid(self, floor_mask):
        # obstacle = ~floor; distance from obstacle (Euclidean)
        obst = np.logical_not(floor_mask).astype(np.uint8)
        dist = cv2.distanceTransform(obst, cv2.DIST_L2, 3)
        # walkable if sufficiently far from obstacle
        walkable = (dist >= config.INFLATION_RADIUS_PX).astype(np.uint8)
        return walkable, dist

    def get_navigation_command(self, perception_data, user_pos):
        target_box = perception_data.get("target_box")
        floor_mask = perception_data.get("floor_mask")
        floor_conf = perception_data.get("floor_confidence", 0.0)
        hazard = perception_data.get("hazard", False)

        system_state = "SEARCHING"
        target_pos = None

        # Immediate safety gates
        if floor_mask is None or floor_conf < config.MASK_MIN_CONFIDENCE or hazard:
            # Lock a 'Stop' with reason
            return "Stop", "UNSAFE", None, None

        if target_box:
            system_state = "GUIDING"
            x1, y1, x2, y2 = target_box
            target_pos = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        else:
            stored_obj = self.object_memory.get_object(config.TARGET_OBJECT)
            if stored_obj:
                system_state = "REACHING"
                target_pos = stored_obj['center']
                target_box = stored_obj['bbox']

        if not target_pos:
            return "Searching for target...", "SEARCHING", None, None

        # Build inflated grid
        scale = config.PATHFINDING_GRID_SCALE
        h, w = floor_mask.shape
        walkable_u8, dist = self._inflate_and_grid(floor_mask)
        small = cv2.resize(walkable_u8, (w // scale, h // scale), interpolation=cv2.INTER_NEAREST)
        grid = Grid(matrix=small.astype(int))

        # unpack user_pos and target_pos
        ux, uy = user_pos
        tx, ty = target_pos

        start_node = find_nearest_walkable_node((ux / scale, uy / scale), grid)
        if not start_node:
            return "Adjust camera to find floor", system_state, target_pos, None

        end_node = find_nearest_walkable_node((tx / scale, ty / scale), grid)
        if not end_node:
            return "Target is unreachable", system_state, target_pos, None

        finder = AStarFinder()
        path_scaled, _ = finder.find_path(start_node, end_node, grid)
        path = [(node.x * scale, node.y * scale) for node in path_scaled] if path_scaled else None
        if not path or len(path) < 2:
            return "Path blocked", system_state, target_pos, None

        # Clearance check along path (min distance in original resolution)
        pts = np.array(path, dtype=np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w-1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h-1)
        min_clear = float(np.min(dist[pts[:,1], pts[:,0]])) if dist.size else 0.0
        if min_clear < config.SAFETY_CLEARANCE_PX:
            return "Stop", "UNSAFE", target_pos, path

        # Smooth path
        path_sm = spline_smooth(path, num_points=20 * config.PATH_SMOOTHING_ITERS)

        # Pick lookahead based on clearance
        L = config.PP_LOOKAHEAD_PX_TIGHT if min_clear < config.PP_TIGHT_CLEARANCE else config.PP_LOOKAHEAD_PX_OPEN
        look_pt = pick_lookahead(path_sm, user_pos, L)

        # Pure Pursuit to command
        if look_pt is None:
            return "Path blocked", system_state, target_pos, path_sm
        new_command = curvature_cmd(user_pos, look_pt, config.PP_SLIGHT_TURN_K, config.PP_SHARP_TURN_K)

        dx = target_pos[0] - user_pos[0]
        dy = target_pos[1] - user_pos[1]
        dist_to_target = np.hypot(dx, dy)

        if path_length(path_sm) <= config.MIN_PATH_LENGTH_FOR_MOVE or dist_to_target < config.TARGET_RADIUS_PX:
            new_command = "Target Reached"


        # Adaptive instruction lock (longer lock when confidence low)
        now = time.time()
        lock = config.INSTRUCTION_LOCK_DURATION * (1.2 if floor_conf < 0.7 else 1.0)
        if new_command == "Target Reached" or (now - self.last_command_time > lock):
            self.last_command = new_command
            self.last_command_time = now

        return self.last_command, system_state, target_pos, path_sm