import cv2

def estimate_color(vehicle_crop):
    """Very rough dominant-color heuristic; use as optional metadata, not identity."""
    if vehicle_crop is None or vehicle_crop.size == 0:
        return "unknown"
    small = cv2.resize(vehicle_crop, (32, 32))
    b, g, r = [float(x) for x in cv2.mean(small)[:3]]
    if max(r, g, b) < 55:
        return "black/dark"
    if min(r, g, b) > 190:
        return "white/light"
    if abs(r-g) < 18 and abs(g-b) < 18:
        return "gray/silver"
    return max([(r, "red"), (g, "green"), (b, "blue")])[1]
