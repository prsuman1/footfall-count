"""Visual Setup Tool - Draw counting lines and staff zones on camera feed.

Usage:
    python src/setup_tool.py --source <video_file_or_rtsp_url>
    python src/setup_tool.py --source sample_videos/test.mp4

Controls:
    Click 2 points      → Draw counting line (up to 2 lines)
    Click up to 4 more  → Mark the INSIDE (store) side of that line
    N                   → Finish current line, start next line
    Z + click corners   → Draw staff zone polygon
    Enter               → Close current zone polygon
    S                   → Save config to config.yaml
    R                   → Reset all lines and zones
    Q                   → Quit
"""

import argparse
import sys
import os

import cv2
import numpy as np

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import save_config, load_config

# Line colors for visual distinction
LINE_COLORS = [(0, 255, 0), (255, 0, 255)]  # green, magenta
INSIDE_COLORS = [(0, 255, 255), (255, 255, 0)]  # cyan, yellow


class SetupTool:
    def __init__(self, source):
        self.source = source
        self.frame = None
        self.display_frame = None

        self.max_lines = 2
        self.max_in_points = 4

        # Completed lines: list of {line_points, line_points_full, in_points, in_points_display}
        self.completed_lines = []

        # Current line being drawn
        self.current_line_points = []       # display coords
        self.current_line_points_full = []  # full-res coords
        self.current_in_points = []         # full-res
        self.current_in_points_display = [] # display coords

        # Staff zone state
        self.zone_mode = False
        self.current_zone_points = []
        self.current_zone_points_full = []
        self.completed_zones = []
        self.zone_counter = 0

        # Load existing settings to preserve them
        self.existing_settings = None
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
        if os.path.exists(config_path):
            try:
                existing = load_config(config_path)
                self.existing_settings = existing.get("settings", {})
            except Exception:
                pass

        self.window_name = "Footfall Counter - Setup Tool"

    def _current_line_index(self):
        """Index of the line currently being drawn."""
        return len(self.completed_lines)

    def _finalize_current_line(self):
        """Save current line to completed_lines and reset for next."""
        if len(self.current_line_points) < 2:
            print("  Need 2 points for a line first.")
            return False
        self.completed_lines.append({
            "line_points": list(self.current_line_points),
            "line_points_full": list(self.current_line_points_full),
            "in_points": list(self.current_in_points),
            "in_points_display": list(self.current_in_points_display),
        })
        idx = len(self.completed_lines)
        print(f"  Line {idx} saved with {len(self.current_in_points)} inside points.")
        self.current_line_points = []
        self.current_line_points_full = []
        self.current_in_points = []
        self.current_in_points_display = []
        return True

    def run(self):
        """Main setup tool loop."""
        if isinstance(self.source, str) and self.source.startswith(("rtsp://", "rtsps://")):
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"Error: Cannot open source: {self.source}")
            return

        ret, self.frame = cap.read()
        if not ret:
            print("Error: Cannot read frame from source")
            cap.release()
            return

        h, w = self.frame.shape[:2]
        self.frame_w = w
        self.frame_h = h
        self.display_w = w
        self.display_h = h
        max_display_w = 960
        if w > max_display_w:
            scale = max_display_w / w
            self.display_w = int(w * scale)
            self.display_h = int(h * scale)
        self.scale_x = w / self.display_w
        self.scale_y = h / self.display_h
        print(f"Source opened: {w}x{h}, display: {self.display_w}x{self.display_h}")
        print()
        print("=== SETUP TOOL ===")
        print(f"1. Click 2 points to draw COUNTING LINE 1 (up to {self.max_lines} lines)")
        print(f"2. Click up to {self.max_in_points} points on the INSIDE (store) side")
        print("3. Press N to finish current line and start the next line")
        print("4. Press Z for zone mode, click corners for STAFF ZONE, Enter to close")
        print("5. Press S to save config")
        print("6. Press R to reset | Press Q to quit")
        print()

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        while True:
            self._redraw()
            cv2.imshow(self.window_name, self.display_frame)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save_config()
            elif key == ord('r'):
                self._reset()
                print("Reset: all lines and zones cleared")
            elif key == ord('n'):
                # Finalize current line, start next
                if len(self.current_line_points) >= 2:
                    if self._current_line_index() < self.max_lines:
                        self._finalize_current_line()
                        remaining = self.max_lines - len(self.completed_lines)
                        if remaining > 0:
                            print(f"  Draw line {len(self.completed_lines) + 1} (click 2 points)")
                        else:
                            print("  Max lines reached. Press S to save.")
                    else:
                        print("  Max lines reached. Press S to save.")
                else:
                    print("  Draw 2 points for the line first.")
            elif key == ord('z'):
                self.zone_mode = not self.zone_mode
                if self.zone_mode:
                    self.current_zone_points = []
                    self.current_zone_points_full = []
                    print("Zone mode ON - Click corners of staff zone, then press Enter")
                else:
                    print("Zone mode OFF")
            elif key == 13:  # Enter
                if self.zone_mode and len(self.current_zone_points) >= 3:
                    self.zone_counter += 1
                    zone = {
                        "name": f"staff_zone_{self.zone_counter}",
                        "polygon": [list(p) for p in self.current_zone_points_full],
                    }
                    self.completed_zones.append(zone)
                    print(f"Zone '{zone['name']}' saved with {len(self.current_zone_points)} points")
                    self.current_zone_points = []
                    self.current_zone_points_full = []
                elif self.zone_mode:
                    print("Need at least 3 points for a zone polygon")

            ret, new_frame = cap.read()
            if ret:
                self.frame = new_frame
            elif isinstance(self.source, str) and os.path.isfile(self.source):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        cap.release()
        cv2.destroyAllWindows()

    def _mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        dx, dy = x, y
        fx, fy = int(x * self.scale_x), int(y * self.scale_y)

        if self.zone_mode:
            self.current_zone_points.append((dx, dy))
            self.current_zone_points_full.append((fx, fy))
            print(f"  Zone point: ({fx}, {fy}) [{len(self.current_zone_points)} points]")
            return

        line_idx = self._current_line_index()
        if line_idx >= self.max_lines:
            print("  Max lines reached. Press S to save or R to reset.")
            return

        # Drawing the 2 line points
        if len(self.current_line_points) < 2:
            self.current_line_points.append((dx, dy))
            self.current_line_points_full.append((fx, fy))
            n = len(self.current_line_points)
            print(f"  Line {line_idx + 1} point {n}: ({fx}, {fy})")
            if n == 2:
                print(f"  Line {line_idx + 1} set! Click up to {self.max_in_points} points on INSIDE side.")
        # Drawing inside points
        elif len(self.current_in_points) < self.max_in_points:
            self.current_in_points.append((fx, fy))
            self.current_in_points_display.append((dx, dy))
            n = len(self.current_in_points)
            print(f"  Line {line_idx + 1} inside point {n}/{self.max_in_points}: ({fx}, {fy})")
            if n >= self.max_in_points:
                if line_idx + 1 < self.max_lines:
                    print(f"  Max inside points. Press N for next line, or S to save.")
                else:
                    print(f"  Max inside points. Press S to save.")
            else:
                print(f"  Click more inside points, press N for next line, or S to save.")
        else:
            if line_idx + 1 < self.max_lines:
                print(f"  Press N to start next line, or S to save.")
            else:
                print(f"  Press S to save config.")

    def _redraw(self):
        """Redraw the frame with all annotations."""
        self.display_frame = cv2.resize(self.frame, (self.display_w, self.display_h))

        # Draw completed lines
        for i, line_data in enumerate(self.completed_lines):
            self._draw_line(line_data, i)

        # Draw current line being drawn
        if self.current_line_points:
            current_data = {
                "line_points": self.current_line_points,
                "in_points_display": self.current_in_points_display,
            }
            self._draw_line(current_data, self._current_line_index())

        # Draw completed zones (scale full-res to display coords)
        for zone in self.completed_zones:
            polygon_full = np.array(zone["polygon"], dtype=np.float64)
            polygon_display = polygon_full.copy()
            polygon_display[:, 0] /= self.scale_x
            polygon_display[:, 1] /= self.scale_y
            polygon_display = polygon_display.astype(np.int32)
            overlay = self.display_frame.copy()
            cv2.fillPoly(overlay, [polygon_display], (255, 100, 0))
            cv2.addWeighted(overlay, 0.3, self.display_frame, 0.7, 0, self.display_frame)
            cv2.polylines(self.display_frame, [polygon_display], True, (255, 100, 0), 2)
            cx = int(polygon_display[:, 0].mean())
            cy = int(polygon_display[:, 1].mean())
            cv2.putText(self.display_frame, zone["name"], (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw current zone-in-progress
        if self.zone_mode and self.current_zone_points:
            for pt in self.current_zone_points:
                cv2.circle(self.display_frame, pt, 5, (255, 0, 0), -1)
            if len(self.current_zone_points) >= 2:
                pts = np.array(self.current_zone_points, dtype=np.int32)
                cv2.polylines(self.display_frame, [pts], False, (255, 0, 0), 2)

        # Draw mode indicator
        if self.zone_mode:
            mode_text = "MODE: ZONE (press Z to toggle)"
        else:
            line_num = len(self.completed_lines) + 1
            mode_text = f"MODE: LINE {line_num}/{self.max_lines}"
        cv2.putText(self.display_frame, mode_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _draw_line(self, line_data, index):
        """Draw a single counting line with its inside points."""
        pts = line_data["line_points"]
        in_pts = line_data.get("in_points_display", [])
        color = LINE_COLORS[index % len(LINE_COLORS)]
        in_color = INSIDE_COLORS[index % len(INSIDE_COLORS)]
        label = f"L{index + 1}"

        if len(pts) >= 1:
            cv2.circle(self.display_frame, pts[0], 6, (0, 0, 255), -1)
        if len(pts) == 2:
            cv2.line(self.display_frame, pts[0], pts[1], color, 2)
            cv2.circle(self.display_frame, pts[0], 6, (0, 0, 255), -1)
            cv2.circle(self.display_frame, pts[1], 6, (0, 0, 255), -1)
            mid_x = (pts[0][0] + pts[1][0]) // 2
            mid_y = (pts[0][1] + pts[1][1]) // 2
            cv2.putText(self.display_frame, label, (mid_x - 10, mid_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if in_pts:
                for ip in in_pts:
                    cv2.circle(self.display_frame, ip, 8, in_color, -1)
                    cv2.arrowedLine(self.display_frame, (mid_x, mid_y), ip, in_color, 2, tipLength=0.3)
                cx = sum(p[0] for p in in_pts) // len(in_pts)
                cy = sum(p[1] for p in in_pts) // len(in_pts)
                cv2.putText(self.display_frame, f"{label} INSIDE", (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, in_color, 2)
                if len(in_pts) >= 3:
                    poly = np.array(in_pts, dtype=np.int32)
                    overlay = self.display_frame.copy()
                    cv2.fillPoly(overlay, [poly], in_color)
                    cv2.addWeighted(overlay, 0.15, self.display_frame, 0.85, 0, self.display_frame)
            else:
                cv2.putText(self.display_frame, "Click INSIDE", (mid_x - 40, mid_y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, in_color, 2)

    def _save_config(self):
        """Save current setup to config.yaml, preserving existing settings."""
        # Finalize current line if in progress
        if len(self.current_line_points) >= 2:
            self._finalize_current_line()

        if not self.completed_lines:
            print("Error: No counting lines set. Click 2 points first.")
            return

        # Build counting_lines list
        counting_lines = []
        for line_data in self.completed_lines:
            entry = {
                "start": list(line_data["line_points_full"][0]),
                "end": list(line_data["line_points_full"][1]),
            }
            if line_data["in_points"]:
                pts = line_data["in_points"]
                cx = sum(p[0] for p in pts) // len(pts)
                cy = sum(p[1] for p in pts) // len(pts)
                entry["in_direction_point"] = [cx, cy]
            counting_lines.append(entry)

        # Preserve existing settings, only override what's needed
        settings = {
            "confidence": 0.3,
            "model": "yolov8n.pt",
            "process_every_n_frames": 1,
            "minimum_crossing_threshold": 1,
            "staff_threshold_minutes": 60,
        }
        if self.existing_settings:
            settings.update(self.existing_settings)

        config = {
            "camera": {
                "source": self.source if isinstance(self.source, str) else int(self.source),
                "resolution": [self.frame_w, self.frame_h],
            },
            "counting_lines": counting_lines,
            "staff_zones": self.completed_zones,
            "settings": settings,
        }

        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
        save_config(config_path, config)

    def _reset(self):
        self.completed_lines = []
        self.current_line_points = []
        self.current_line_points_full = []
        self.current_in_points = []
        self.current_in_points_display = []
        self.current_zone_points = []
        self.current_zone_points_full = []
        self.completed_zones = []
        self.zone_counter = 0
        self.zone_mode = False


def main():
    parser = argparse.ArgumentParser(description="Footfall Counter Setup Tool")
    parser.add_argument("--source", required=True,
                        help="Video file path, RTSP URL, or webcam index (0)")
    args = parser.parse_args()

    source = args.source
    if source.isdigit():
        source = int(source)

    tool = SetupTool(source)
    tool.run()


if __name__ == "__main__":
    main()
