import yaml
import cv2
import numpy as np


def load_config(path="config.yaml"):
    """Load and validate configuration from YAML file.

    Args:
        path: path to config.yaml

    Returns:
        dict with camera, counting_line, staff_zones, settings
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Validate required fields
    if "camera" not in config or "source" not in config["camera"]:
        raise ValueError("Config missing camera.source")
    if "counting_line" not in config and "counting_lines" not in config:
        raise ValueError("Config missing counting_line/counting_lines (run setup_tool.py first)")
    if "counting_line" in config:
        if "start" not in config["counting_line"] or "end" not in config["counting_line"]:
            raise ValueError("Config missing counting_line.start or counting_line.end")

    # Defaults for settings
    settings = config.get("settings", {})
    settings.setdefault("confidence", 0.5)
    settings.setdefault("process_every_n_frames", 3)
    settings.setdefault("staff_threshold_minutes", 60)
    config["settings"] = settings

    # Ensure staff_zones is a list
    config.setdefault("staff_zones", [])

    return config


def save_config(path, data):
    """Save configuration to YAML file.

    Args:
        path: output file path
        data: dict to save
    """
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"Config saved to {path}")


def draw_counts_overlay(frame, counts, staff_excluded=0, fps=0):
    """Draw count information overlay on frame.

    Args:
        frame: BGR numpy array (modified in place)
        counts: dict with entries, exits, occupancy
        staff_excluded: number of staff excluded
        fps: current processing FPS

    Returns:
        frame with overlay drawn
    """
    h, w = frame.shape[:2]

    # Semi-transparent background bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Count text
    text = (
        f"IN: {counts['entries']}  |  "
        f"OUT: {counts['exits']}  |  "
        f"INSIDE: {counts['occupancy']}"
    )
    cv2.putText(frame, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Status bar at bottom
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 30), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)

    status = f"FPS: {fps:.0f}  |  Staff Excluded: {staff_excluded}"
    cv2.putText(frame, status, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


def draw_staff_zones(frame, staff_zones, alpha=0.2):
    """Draw staff zone polygons on frame.

    Args:
        frame: BGR numpy array (modified in place)
        staff_zones: list of dicts with 'polygon' key (list of [x,y] points)
        alpha: transparency of zone overlay

    Returns:
        frame with zones drawn
    """
    for zone_data in staff_zones:
        polygon = np.array(zone_data["polygon"], dtype=np.int32)
        # Draw filled polygon with transparency
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], (255, 100, 0))
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        # Draw border
        cv2.polylines(frame, [polygon], True, (255, 100, 0), 2)
        # Label
        if "name" in zone_data:
            cx = int(polygon[:, 0].mean())
            cy = int(polygon[:, 1].mean())
            cv2.putText(frame, zone_data["name"], (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame
