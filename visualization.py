# visualization.py
import cv2
import numpy as np
import config

def visualize_output(image_bgr, viz_data):
    """Draws all visualization elements on the frame."""
    vis_image = image_bgr.copy()

    # Unpack data
    command = viz_data.get("command")
    system_state = viz_data.get("system_state")
    target_box = viz_data.get("target_box")
    floor_mask = viz_data.get("floor_mask")
    unsafe_mask = viz_data.get("unsafe_mask")
    inflated_mask = viz_data.get("inflated_mask")
    floor_conf = viz_data.get("floor_confidence")
    a_star_path = viz_data.get("a_star_path")
    fps = viz_data.get("fps")
    text_results = viz_data.get("text_results")

    # Draw floor
    if floor_mask is not None:
        floor_overlay = np.zeros_like(vis_image, dtype=np.uint8)
        floor_overlay[floor_mask] = (0, 200, 0)
        vis_image = cv2.addWeighted(vis_image, 0.7, floor_overlay, 0.3, 0)
    # Draw unsafe
    if unsafe_mask is not None:
        unsafe_overlay = np.zeros_like(vis_image, dtype=np.uint8)
        unsafe_overlay[unsafe_mask] = (0, 0, 255)
        vis_image = cv2.addWeighted(vis_image, 0.8, unsafe_overlay, 0.2, 0)
    # Optionally draw inflated walkable (cyan tint)
    if inflated_mask is not None:
        infl_overlay = np.zeros_like(vis_image, dtype=np.uint8)
        infl_overlay[inflated_mask.astype(bool)] = (255, 255, 0)
        vis_image = cv2.addWeighted(vis_image, 0.9, infl_overlay, 0.1, 0)

    # status text
    if floor_conf is not None:
        cv2.putText(vis_image, f"Floor Conf: {floor_conf:.2f}", (15, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
        cv2.putText(vis_image, f"Floor Conf: {floor_conf:.2f}", (15, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)

    # Draw all detected text boxes in green
    if text_results:
        for (bbox, text, conf) in text_results:
            if conf > config.OCR_CONFIDENCE_THRESHOLD:
                pts = np.array(bbox, dtype=np.int32)
                cv2.polylines(vis_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(vis_image, text, (pts[0][0], pts[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw navigation target box (from YOLO) in blue
    if target_box:
        x1, y1, x2, y2 = [int(c) for c in target_box]
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 100, 100), 3)
        # Label the target object
        label = f"TARGET: {config.TARGET_OBJECT}"
        cv2.putText(vis_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
    
    # Draw A* path
    if a_star_path and len(a_star_path) > 1:
        # Draw circles at each waypoint for clarity
        for point in a_star_path:
            pt = (int(point[0]), int(point[1]))
            cv2.circle(vis_image, pt, 5, (0, 255, 255), -1) # Yellow circles

        # Draw connecting lines
        for i in range(len(a_star_path) - 1):
            pt1 = (int(a_star_path[i][0]), int(a_star_path[i][1]))
            pt2 = (int(a_star_path[i+1][0]), int(a_star_path[i+1][1]))
            cv2.line(vis_image, pt1, pt2, (255, 255, 0), 3) # Cyan lines

    # Display status text
    cv2.putText(vis_image, f"STATE: {system_state}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
    cv2.putText(vis_image, f"STATE: {system_state}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    cv2.putText(vis_image, f"CMD: {command}", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
    cv2.putText(vis_image, f"CMD: {command}", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(vis_image, f"FPS: {fps:.1f}", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
    cv2.putText(vis_image, f"FPS: {fps:.1f}", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    
    cv2.imshow("Navigation Prototype", vis_image)