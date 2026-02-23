"""Footfall Counter - Main Pipeline

Connects all components: camera → detector → staff filter → counter → display.

Usage:
    python src/main.py                          # Use config.yaml defaults
    python src/main.py --config config.yaml     # Specify config
    python src/main.py --source video.mp4       # Override camera source
"""

import argparse
import signal
import sys
import os
import time

import cv2
import supervision as sv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.camera import RTSPCamera
from src.detector import PersonDetector
from src.counter import FootfallCounter, StaffExcluder
from src.utils import load_config, draw_counts_overlay, draw_staff_zones


class FootfallPipeline:
    def __init__(self, config, save_video=None):
        self.config = config
        self.running = True
        self.save_video = save_video
        self.video_writer = None

        # Camera
        source = config["camera"]["source"]
        print(f"[Pipeline] Source: {source}")
        self.camera = RTSPCamera(source, camera_id="cam_0")

        # Detector
        self.detector = PersonDetector(
            model_path=config["settings"].get("model", "yolov8s.pt"),
            confidence=config["settings"]["confidence"],
            min_bbox_area=config["settings"].get("min_bbox_area", 0),
        )

        # Counter
        line = config["counting_line"]
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(project_root, "footfall_log.csv")
        self.counter = FootfallCounter(
            line_start=line["start"],
            line_end=line["end"],
            minimum_crossing_threshold=config["settings"].get("minimum_crossing_threshold", 2),
            csv_path=csv_path,
        )

        # Staff excluder
        staff_zones = [z["polygon"] for z in config.get("staff_zones", []) if "polygon" in z]
        self.staff_excluder = StaffExcluder(
            threshold_minutes=config["settings"]["staff_threshold_minutes"],
            staff_zones=staff_zones if staff_zones else None,
        )

        # Annotators
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)
        self.line_annotator = sv.LineZoneAnnotator(display_in_count=False, display_out_count=False)

        # Frame skip
        self.process_every_n = config["settings"]["process_every_n_frames"]
        self.frame_count = 0
        self.none_frame_count = 0
        self.last_detections = sv.Detections.empty()
        self.last_counts = {"entries": 0, "exits": 0, "occupancy": 0}

        # FPS tracking
        self.fps = 0
        self.fps_frames = 0
        self.fps_start = time.time()

        # Logging
        self.last_log_time = time.time()

    def run(self):
        """Main processing loop."""
        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)

        # Start camera
        self.camera.start()
        print("[Pipeline] Running... Press Q to quit.")
        print()

        while self.running:
            frame = self.camera.read()
            if frame is None:
                self.none_frame_count += 1
                if self.camera.stopped:
                    print("[Pipeline] Camera permanently stopped, exiting")
                    break
                if self.none_frame_count >= 5:
                    print("[Pipeline] Stream lost, forcing reconnect...")
                    self.camera.connected = False
                    self.none_frame_count = 0
                else:
                    print("[Pipeline] Waiting for frame...")
                continue

            self.none_frame_count = 0

            self.frame_count += 1

            # Process every Nth frame
            if self.frame_count % self.process_every_n == 0:
                detections = self.detector.detect_and_track(frame)

                # Staff exclusion
                self.staff_excluder.update(detections)
                customer_detections = self.staff_excluder.filter_customers(detections)

                # Count line crossings (use customer detections only)
                self.last_counts = self.counter.update(customer_detections, frame=frame)
                self.last_detections = customer_detections

                # Periodically cleanup old tracks
                if self.frame_count % 300 == 0:
                    self.staff_excluder.cleanup_old_tracks(detections.tracker_id)

            # Annotate frame
            annotated = self._annotate(frame)

            # Write to video file or show on screen
            if self.save_video:
                if self.video_writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(self.save_video, fourcc, 25, (w, h))
                    print(f"[Pipeline] Writing annotated video to {self.save_video}")
                self.video_writer.write(annotated)
            else:
                cv2.imshow("Footfall Counter", annotated)

            # FPS calculation
            self.fps_frames += 1
            elapsed = time.time() - self.fps_start
            if elapsed >= 1.0:
                self.fps = self.fps_frames / elapsed
                self.fps_frames = 0
                self.fps_start = time.time()

            # Console log every second
            now = time.time()
            if now - self.last_log_time >= 1.0:
                self._log_counts()
                self.last_log_time = now

            # Key handling (only when showing window)
            if not self.save_video:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[Pipeline] Quit requested")
                    break

        self._shutdown()

    def _annotate(self, frame):
        """Draw all annotations on frame."""
        annotated = frame.copy()

        # Draw staff zones
        if self.config.get("staff_zones"):
            annotated = draw_staff_zones(annotated, self.config["staff_zones"])

        # Draw counting line
        annotated = self.line_annotator.annotate(
            frame=annotated,
            line_counter=self.counter.line_zone,
        )

        # Draw bounding boxes and labels on customers
        if len(self.last_detections) > 0:
            labels = []
            for i in range(len(self.last_detections)):
                tid = self.last_detections.tracker_id[i] if self.last_detections.tracker_id is not None else "?"
                conf = self.last_detections.confidence[i] if self.last_detections.confidence is not None else 0
                labels.append(f"#{tid} {conf:.1%}")

            annotated = self.box_annotator.annotate(
                scene=annotated,
                detections=self.last_detections,
            )
            annotated = self.label_annotator.annotate(
                scene=annotated,
                detections=self.last_detections,
                labels=labels,
            )

        # Draw counts overlay
        annotated = draw_counts_overlay(
            annotated,
            self.last_counts,
            staff_excluded=self.staff_excluder.staff_count,
            fps=self.fps,
        )

        return annotated

    def _log_counts(self):
        """Print counts to console."""
        ts = time.strftime("%H:%M:%S")
        c = self.last_counts
        staff = self.staff_excluder.staff_count
        print(
            f"[{ts}] Entries: {c['entries']:>4} | "
            f"Exits: {c['exits']:>4} | "
            f"Inside: {c['occupancy']:>4} | "
            f"Staff: {staff} excluded"
        )

    def _signal_handler(self, sig, frame):
        print("\n[Pipeline] Ctrl+C received, shutting down...")
        self.running = False

    def _shutdown(self):
        """Clean shutdown of all components."""
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"[Pipeline] Annotated video saved to {self.save_video}")
        self.camera.stop()
        cv2.destroyAllWindows()
        print("[Pipeline] Shutdown complete")
        print()
        print(f"=== FINAL COUNTS ===")
        print(f"  Entries:  {self.last_counts['entries']}")
        print(f"  Exits:    {self.last_counts['exits']}")
        print(f"  Inside:   {self.last_counts['occupancy']}")
        print(f"  Staff:    {self.staff_excluder.staff_count} excluded")


def main():
    parser = argparse.ArgumentParser(description="Footfall Counter")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--source", default=None, help="Override camera source (video file or RTSP URL)")
    parser.add_argument("--save-video", default=None, help="Save annotated video to file instead of displaying")
    args = parser.parse_args()

    # Find config
    config_path = args.config
    if config_path is None:
        # Look in project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "config.yaml")

    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        print("Run setup_tool.py first to create config.yaml:")
        print("  python src/setup_tool.py --source <video_or_rtsp>")
        sys.exit(1)

    config = load_config(config_path)

    # Override source if provided
    if args.source:
        config["camera"]["source"] = args.source

    pipeline = FootfallPipeline(config, save_video=args.save_video)
    pipeline.run()


if __name__ == "__main__":
    main()
