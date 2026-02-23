import csv
import os
import time
from datetime import datetime

import cv2

import numpy as np
import supervision as sv


class FootfallCounter:
    """Line-crossing counter using supervision LineZone.

    Counts entries and exits based on people crossing a virtual line.
    Direction is determined by the line orientation (start → end).
    Each tracker ID can only trigger one IN and one OUT crossing to
    prevent false counts from jittery detections near the line.
    """

    def __init__(self, line_start, line_end, minimum_crossing_threshold=2, csv_path=None):
        """
        Args:
            line_start: (x, y) tuple for line start point
            line_end: (x, y) tuple for line end point
            minimum_crossing_threshold: consecutive frames on one side before counting
            csv_path: path to CSV log file (None to disable logging)
        """
        self.line_zone = sv.LineZone(
            start=sv.Point(line_start[0], line_start[1]),
            end=sv.Point(line_end[0], line_end[1]),
            triggering_anchors=[sv.Position.BOTTOM_CENTER],
            minimum_crossing_threshold=minimum_crossing_threshold,
        )
        self._prev_entries = 0
        self._prev_exits = 0
        self._csv_path = csv_path
        self._screenshots_dir = None
        if csv_path:
            self._screenshots_dir = os.path.join(os.path.dirname(csv_path), "screenshots")
            os.makedirs(self._screenshots_dir, exist_ok=True)
            write_header = not os.path.exists(csv_path)
            self._csv_file = open(csv_path, "a", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            if write_header:
                self._csv_writer.writerow(["timestamp", "event", "count", "screenshot"])

    def update(self, detections, frame=None):
        """Process detections and update entry/exit counts.

        Uses supervision's built-in LineZone counting directly.
        LineZone already tracks per-ID crossings internally.

        Args:
            detections: sv.Detections with tracker_id
            frame: optional video frame for screenshot capture on entry

        Returns:
            dict with entries, exits, occupancy
        """
        self.line_zone.trigger(detections)

        entries = self.line_zone.out_count
        exits = self.line_zone.in_count
        occupancy = max(0, entries - exits)

        # Log new entry events to CSV + capture screenshots
        if self._csv_path and entries > self._prev_entries:
            for i in range(entries - self._prev_entries):
                count = self._prev_entries + i + 1
                ts = datetime.now()
                screenshot_name = ""
                if frame is not None and self._screenshots_dir:
                    screenshot_name = f"entry_{count}_{ts.strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(
                        os.path.join(self._screenshots_dir, screenshot_name),
                        frame,
                    )
                self._csv_writer.writerow([
                    ts.isoformat(),
                    "entry",
                    count,
                    screenshot_name,
                ])
            self._csv_file.flush()
        self._prev_entries = entries

        # Log new exit events to CSV
        if self._csv_path and exits > self._prev_exits:
            for i in range(exits - self._prev_exits):
                count = self._prev_exits + i + 1
                ts = datetime.now()
                self._csv_writer.writerow([
                    ts.isoformat(),
                    "exit",
                    count,
                    "",
                ])
            self._csv_file.flush()
        self._prev_exits = exits

        return {
            "entries": entries,
            "exits": exits,
            "occupancy": occupancy,
        }

    @property
    def line_annotator(self):
        """Get a LineZoneAnnotator for visualization."""
        return sv.LineZoneAnnotator()


class StaffExcluder:
    """Excludes staff from customer count using duration and zone heuristics.

    - Duration-based: anyone tracked for longer than threshold = staff
    - Zone-based: anyone spending significant time in staff zones = staff
    """

    def __init__(self, threshold_minutes=60, staff_zones=None):
        """
        Args:
            threshold_minutes: minutes before a person is classified as staff
            staff_zones: list of polygon arrays (each Nx2 numpy array) defining staff-only areas
        """
        self.threshold_seconds = threshold_minutes * 60
        self.first_seen = {}  # track_id -> timestamp
        self.staff_ids = set()
        self.zone_time = {}  # track_id -> cumulative seconds in staff zones
        self.staff_zones = []

        if staff_zones:
            for polygon in staff_zones:
                poly = np.array(polygon, dtype=np.int32)
                zone = sv.PolygonZone(polygon=poly)
                self.staff_zones.append(zone)

    def update(self, detections):
        """Update staff tracking state. Call before filter_customers().

        Args:
            detections: sv.Detections with tracker_id
        """
        if detections.tracker_id is None or len(detections) == 0:
            return

        current_time = time.time()

        for i, track_id in enumerate(detections.tracker_id):
            tid = int(track_id)

            # Duration-based check
            if tid not in self.first_seen:
                self.first_seen[tid] = current_time

            duration = current_time - self.first_seen[tid]
            if duration > self.threshold_seconds:
                self.staff_ids.add(tid)

            # Zone-based check
            if self.staff_zones:
                # Get this detection's bottom center point
                bbox = detections.xyxy[i]
                cx = (bbox[0] + bbox[2]) / 2
                cy = bbox[3]  # bottom of bbox

                for zone in self.staff_zones:
                    # Check if point is inside zone polygon
                    single_det = sv.Detections(
                        xyxy=detections.xyxy[i:i+1],
                        confidence=detections.confidence[i:i+1] if detections.confidence is not None else None,
                    )
                    in_zone = zone.trigger(single_det)
                    if in_zone.any():
                        self.zone_time[tid] = self.zone_time.get(tid, 0) + 0.1
                        # If spent >10 min in staff zone, mark as staff
                        if self.zone_time[tid] > 600:
                            self.staff_ids.add(tid)

    def filter_customers(self, detections):
        """Remove staff from detections, returning only customers.

        Args:
            detections: sv.Detections with tracker_id

        Returns:
            sv.Detections with staff filtered out
        """
        if detections.tracker_id is None or len(detections) == 0:
            return detections

        customer_mask = np.array([
            int(tid) not in self.staff_ids
            for tid in detections.tracker_id
        ])

        return detections[customer_mask]

    @property
    def staff_count(self):
        return len(self.staff_ids)

    def cleanup_old_tracks(self, active_ids):
        """Remove tracking data for IDs no longer active."""
        active = set(int(x) for x in active_ids) if active_ids is not None else set()
        stale = set(self.first_seen.keys()) - active - self.staff_ids
        # Keep staff IDs forever within a session, clean up old customer tracks
        for tid in stale:
            if tid in self.first_seen:
                # Only remove if not seen for a while
                if time.time() - self.first_seen[tid] > 300:  # 5 min
                    del self.first_seen[tid]
                    self.zone_time.pop(tid, None)
