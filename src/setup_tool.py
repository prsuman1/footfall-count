"""Visual Setup Tool - Draw counting line and staff zones on camera feed.

Usage:
    python src/setup_tool.py --source <video_file_or_rtsp_url>
    python src/setup_tool.py --source sample_videos/test.mp4

Controls:
    Click 2 points      → Draw counting line
    Z + click corners    → Draw staff zone polygon
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import save_config


class SetupTool:
    def __init__(self, source):
        self.source = source
        self.frame = None
        self.display_frame = None

        # Counting line state
        self.line_points = []  # [(x,y), (x,y)]

        # Staff zone state
        self.zone_mode = False
        self.current_zone_points = []
        self.completed_zones = []  # list of {"name": str, "polygon": [[x,y],...]}
        self.zone_counter = 0

        self.window_name = "Footfall Counter - Setup Tool"

    def run(self):
        """Main setup tool loop."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"Error: Cannot open source: {self.source}")
            return

        # Read first frame
        ret, self.frame = cap.read()
        if not ret:
            print("Error: Cannot read frame from source")
            cap.release()
            return

        h, w = self.frame.shape[:2]
        self.frame_w = w
        self.frame_h = h
        # Display at a fixed size that fits screen, track scale for coordinate mapping
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
        print("1. Click 2 points to draw the COUNTING LINE (across the doorway)")
        print("2. Press Z to enter zone mode, click corners for STAFF ZONE")
        print("3. Press Enter to close a zone polygon")
        print("4. Press S to save config")
        print("5. Press R to reset | Press Q to quit")
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
                self._save_config(w, h)
            elif key == ord('r'):
                self._reset()
                print("Reset: all lines and zones cleared")
            elif key == ord('z'):
                self.zone_mode = not self.zone_mode
                if self.zone_mode:
                    self.current_zone_points = []
                    print("Zone mode ON - Click corners of staff zone, then press Enter")
                else:
                    print("Zone mode OFF")
            elif key == 13:  # Enter
                if self.zone_mode and len(self.current_zone_points) >= 3:
                    self.zone_counter += 1
                    zone = {
                        "name": f"staff_zone_{self.zone_counter}",
                        "polygon": [list(p) for p in getattr(self, 'current_zone_points_full', self.current_zone_points)],
                    }
                    self.completed_zones.append(zone)
                    print(f"Zone '{zone['name']}' saved with {len(self.current_zone_points)} points")
                    self.current_zone_points = []
                    self.current_zone_points_full = []
                elif self.zone_mode:
                    print("Need at least 3 points for a zone polygon")

            # Try to advance video frame periodically for live feeds
            ret, new_frame = cap.read()
            if ret:
                self.frame = new_frame
            elif isinstance(self.source, str) and os.path.isfile(self.source):
                # Video file ended, loop back
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        cap.release()
        cv2.destroyAllWindows()

    def _mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Display coordinates (for drawing on resized frame)
        dx, dy = x, y
        # Full resolution coordinates (for saving to config)
        fx, fy = int(x * self.scale_x), int(y * self.scale_y)

        if self.zone_mode:
            self.current_zone_points.append((dx, dy))
            self.current_zone_points_full = getattr(self, 'current_zone_points_full', [])
            self.current_zone_points_full.append((fx, fy))
            print(f"  Zone point: ({fx}, {fy}) [{len(self.current_zone_points)} points]")
        else:
            if len(self.line_points) < 2:
                self.line_points.append((dx, dy))
                self.line_points_full = getattr(self, 'line_points_full', [])
                self.line_points_full.append((fx, fy))
                print(f"  Line point {len(self.line_points)}: ({fx}, {fy}) [full res]")
                if len(self.line_points) == 2:
                    print("  Counting line set! Press S to save.")
            else:
                # Reset line and start over
                self.line_points = [(dx, dy)]
                self.line_points_full = [(fx, fy)]
                print(f"  Line reset. Point 1: ({fx}, {fy})")

    def _redraw(self):
        """Redraw the frame with all annotations."""
        self.display_frame = cv2.resize(self.frame, (self.display_w, self.display_h))

        # Draw counting line
        if len(self.line_points) >= 1:
            cv2.circle(self.display_frame, self.line_points[0], 6, (0, 0, 255), -1)
        if len(self.line_points) == 2:
            cv2.line(self.display_frame, self.line_points[0], self.line_points[1], (0, 255, 0), 2)
            cv2.circle(self.display_frame, self.line_points[0], 6, (0, 0, 255), -1)
            cv2.circle(self.display_frame, self.line_points[1], 6, (0, 0, 255), -1)
            # Draw direction arrow (IN direction = left side of line)
            mid_x = (self.line_points[0][0] + self.line_points[1][0]) // 2
            mid_y = (self.line_points[0][1] + self.line_points[1][1]) // 2
            cv2.putText(self.display_frame, "IN ->", (mid_x - 20, mid_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw completed zones
        for zone in self.completed_zones:
            polygon = np.array(zone["polygon"], dtype=np.int32)
            overlay = self.display_frame.copy()
            cv2.fillPoly(overlay, [polygon], (255, 100, 0))
            cv2.addWeighted(overlay, 0.3, self.display_frame, 0.7, 0, self.display_frame)
            cv2.polylines(self.display_frame, [polygon], True, (255, 100, 0), 2)
            cx = int(polygon[:, 0].mean())
            cy = int(polygon[:, 1].mean())
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
        mode_text = "MODE: ZONE (press Z to toggle)" if self.zone_mode else "MODE: LINE (click 2 points)"
        cv2.putText(self.display_frame, mode_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _save_config(self, width, height):
        """Save current setup to config.yaml. Coordinates are at full resolution."""
        line_full = getattr(self, 'line_points_full', self.line_points)
        if len(line_full) < 2:
            print("Error: No counting line set. Click 2 points first.")
            return

        config = {
            "camera": {
                "source": self.source if isinstance(self.source, str) else int(self.source),
                "resolution": [self.frame_w, self.frame_h],
            },
            "counting_line": {
                "start": list(line_full[0]),
                "end": list(line_full[1]),
            },
            "staff_zones": self.completed_zones,
            "settings": {
                "confidence": 0.5,
                "process_every_n_frames": 3,
                "staff_threshold_minutes": 60,
            },
        }

        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
        save_config(config_path, config)

    def _reset(self):
        self.line_points = []
        self.line_points_full = []
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

    # Handle webcam index
    source = args.source
    if source.isdigit():
        source = int(source)

    tool = SetupTool(source)
    tool.run()


if __name__ == "__main__":
    main()
