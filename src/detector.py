import numpy as np
import supervision as sv
from ultralytics import YOLO


class PersonDetector:
    """YOLOv8n person detector with ByteTrack tracking.

    Uses MPS (Apple Silicon GPU) when available, falls back to CPU.
    Only detects persons (class 0). Integrates ByteTrack via supervision
    for persistent tracking IDs across frames.
    """

    def __init__(self, model_path="yolov8n.pt", confidence=0.5, device=None, min_bbox_area=0):
        self.confidence = confidence
        self.min_bbox_area = min_bbox_area  # Filter out small detections (people far away/through glass)

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
            frame_rate=10,
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
        return detections

    def reset_tracker(self):
        """Reset the tracker (e.g., on camera reconnect)."""
        self.tracker.reset()
