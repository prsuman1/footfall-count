import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


class PersonDetector:
    """YOLOv8n person detector with ByteTrack tracking.

    Uses MPS (Apple Silicon GPU) when available, falls back to CPU.
    Only detects persons (class 0). Integrates ByteTrack via supervision
    for persistent tracking IDs across frames.
    """

    def __init__(self, model_path="yolov8n.pt", confidence=0.5, device=None, min_bbox_area=0, detection_zone=None):
        self.confidence = confidence
        self.min_bbox_area = min_bbox_area  # Filter out small detections (people far away/through glass)

        # Detection zone polygon — only detections with bottom-center inside are kept
        if detection_zone is not None:
            self.detection_zone = np.array(detection_zone, dtype=np.int32)
            print(f"[Detector] Detection zone set with {len(detection_zone)} vertices")
        else:
            self.detection_zone = None

        # Auto-select device
        if device is None:
            import torch
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"[Detector] Loading {model_path} on {self.device}")
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=90,
            minimum_matching_threshold=0.7,
            frame_rate=30,
        )

        # Warm up the model
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, classes=[0], conf=self.confidence, device=self.device, verbose=False)
        print("[Detector] Model loaded and warmed up")

    def detect_and_track(self, frame):
        """Run detection + tracking on a frame.

        Args:
            frame: BGR numpy array

        Returns:
            sv.Detections with tracker_id assigned
        """
        results = self.model.predict(
            frame,
            classes=[0],
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        if len(detections) == 0:
            return detections

        # Filter out small bboxes (people far away, seen through glass, etc.)
        if self.min_bbox_area > 0:
            areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * \
                    (detections.xyxy[:, 3] - detections.xyxy[:, 1])
            size_mask = areas >= self.min_bbox_area
            detections = detections[size_mask]

        if len(detections) == 0:
            return detections

        detections = self.tracker.update_with_detections(detections)

        # Filter detections outside the detection zone (e.g. through-glass sidewalk people)
        if self.detection_zone is not None and len(detections) > 0:
            # Bottom-center of each bbox: ((x1+x2)/2, y2)
            bottom_centers_x = (detections.xyxy[:, 0] + detections.xyxy[:, 2]) / 2
            bottom_centers_y = detections.xyxy[:, 3]
            zone_mask = np.array([
                cv2.pointPolygonTest(self.detection_zone, (float(x), float(y)), False) >= 0
                for x, y in zip(bottom_centers_x, bottom_centers_y)
            ])
            detections = detections[zone_mask]

        return detections

    def reset_tracker(self):
        """Reset the tracker (e.g., on camera reconnect)."""
        self.tracker.reset()
