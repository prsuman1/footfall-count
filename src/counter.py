import csv
import os
import time
from datetime import datetime

import cv2

import numpy as np
import supervision as sv


class _LineData:
    """Internal data for a single counting line."""
    def __init__(self, line_start, line_end, minimum_crossing_threshold, in_direction_point, index):
        self.line_zone = sv.LineZone(
            start=sv.Point(line_start[0], line_start[1]),
            end=sv.Point(line_end[0], line_end[1]),
            triggering_anchors=[sv.Position.BOTTOM_CENTER],
            minimum_crossing_threshold=minimum_crossing_threshold,
        )
        start = np.array(line_start, dtype=np.float64)
        end = np.array(line_end, dtype=np.float64)
        line_vec = end - start
        self.normal = np.array([-line_vec[1], line_vec[0]])
        self.normal_len = np.linalg.norm(self.normal)
        self.start = start
        self.index = index

        if in_direction_point is not None:
            dist = np.dot(np.array(in_direction_point, dtype=np.float64) - start, self.normal) / self.normal_len
            self.in_is_crossed_in = dist < 0
            side = "LEFT (crossed_in=ENTRY)" if self.in_is_crossed_in else "RIGHT (crossed_out=ENTRY)"
            print(f"[Counter] Line {index+1}: in_direction_point {in_direction_point} on {side} side (dist={dist:.1f})")
        else:
            self.in_is_crossed_in = True
            print(f"[Counter] Line {index+1}: no in_direction_point, defaulting crossed_in=ENTRY")


class FootfallCounter:
    """Multi-line crossing counter using supervision LineZone with displacement validation.

    Supports 1 or more counting lines. A person crossing ANY line gets counted.
    Cooldown is shared across all lines to prevent double-counting.
    """

    MIN_DISPLACEMENT = 0.5  # pixels — must be seen on both sides of at least one line
    COOLDOWN_FRAMES = 150          # ignore same tracker for ~12s after counting

    def __init__(self, lines=None, line_start=None, line_end=None,
                 minimum_crossing_threshold=2, csv_path=None, in_direction_point=None):
        """
        Args:
            lines: list of dicts with keys: start, end, in_direction_point (optional).
                   If provided, line_start/line_end/in_direction_point are ignored.
            line_start: (x, y) for single line (backward compat)
            line_end: (x, y) for single line (backward compat)
            minimum_crossing_threshold: consecutive frames on one side before counting
            csv_path: path to CSV log file
            in_direction_point: (x, y) for single line (backward compat)
        """
        # Build line list — support both new multi-line and old single-line API
        if lines is None:
            lines = [{"start": line_start, "end": line_end, "in_direction_point": in_direction_point}]

        self._lines = []
        for i, line_cfg in enumerate(lines):
            ld = _LineData(
                line_start=line_cfg["start"],
                line_end=line_cfg["end"],
                minimum_crossing_threshold=minimum_crossing_threshold,
                in_direction_point=line_cfg.get("in_direction_point"),
                index=i,
            )
            self._lines.append(ld)

        # Per-line per-tracker displacement: line_idx -> {tid -> (max_pos, min_neg)}
        self._tracker_extremes: list[dict[int, tuple[float, float]]] = [{} for _ in self._lines]
        # Shared cooldown across all lines: tid -> frame_count when last counted
        self._tracker_cooldown: dict[int, int] = {}
        self._frame_count = 0

        self._validated_entries = 0
        self._validated_exits = 0
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

    def _signed_distance(self, point, line_data):
        """Compute signed perpendicular distance from a point to a line."""
        return np.dot(point - line_data.start, line_data.normal) / line_data.normal_len

    def _update_tracker_extremes(self, detections):
        """Update displacement extremes for each tracker on each line."""
        if detections.tracker_id is None or len(detections) == 0:
            return
        for line_idx, ld in enumerate(self._lines):
            extremes = self._tracker_extremes[line_idx]
            for i, tid in enumerate(detections.tracker_id):
                tid = int(tid)
                bbox = detections.xyxy[i]
                bottom_center = np.array([(bbox[0] + bbox[2]) / 2, bbox[3]])
                dist = self._signed_distance(bottom_center, ld)
                if tid in extremes:
                    max_pos, min_neg = extremes[tid]
                    extremes[tid] = (max(max_pos, dist), min(min_neg, dist))
                else:
                    extremes[tid] = (dist, dist)

    def _has_real_displacement(self, tid, line_idx):
        """Check if tracker has been seen on both sides of a specific line."""
        extremes = self._tracker_extremes[line_idx]
        if tid not in extremes:
            return False
        max_pos, min_neg = extremes[tid]
        return max_pos >= self.MIN_DISPLACEMENT and min_neg <= -self.MIN_DISPLACEMENT

    def _prune_stale_trackers(self, active_ids):
        """Remove displacement and cooldown data for trackers no longer active."""
        if active_ids is None:
            return
        active = set(int(x) for x in active_ids)
        for extremes in self._tracker_extremes:
            stale = set(extremes.keys()) - active
            for tid in stale:
                del extremes[tid]
        stale_cd = set(self._tracker_cooldown.keys()) - active
        for tid in stale_cd:
            del self._tracker_cooldown[tid]

    def update(self, detections, frame=None):
        """Process detections and update entry/exit counts across all lines.

        Args:
            detections: sv.Detections with tracker_id
            frame: optional video frame for screenshot capture on entry

        Returns:
            dict with entries, exits, occupancy
        """
        self._frame_count += 1
        self._update_tracker_extremes(detections)

        # Collect validated crossings across all lines, dedup by tracker ID
        # Only the first crossing per tracker per frame wins (prevents X-line double-count)
        entry_tids = set()
        exit_tids = set()

        for line_idx, ld in enumerate(self._lines):
            crossed_in, crossed_out = ld.line_zone.trigger(detections)

            if detections.tracker_id is None or len(detections) == 0:
                continue

            for i, tid in enumerate(detections.tracker_id):
                tid = int(tid)
                # Shared cooldown — if counted on ANY line, skip
                if tid in self._tracker_cooldown:
                    if self._frame_count - self._tracker_cooldown[tid] < self.COOLDOWN_FRAMES:
                        continue
                    else:
                        del self._tracker_cooldown[tid]

                # Already counted this frame on another line
                if tid in entry_tids or tid in exit_tids:
                    continue

                is_entry = crossed_in[i] if ld.in_is_crossed_in else crossed_out[i]
                is_exit = crossed_out[i] if ld.in_is_crossed_in else crossed_in[i]

                if is_entry:
                    has_disp = self._has_real_displacement(tid, line_idx)
                    extremes = self._tracker_extremes[line_idx].get(tid, (0, 0))
                    print(f"[CROSS] L{line_idx+1} tid={tid} ENTRY candidate disp={has_disp} extremes={extremes}")
                    if has_disp:
                        entry_tids.add(tid)
                if is_exit:
                    has_disp = self._has_real_displacement(tid, line_idx)
                    extremes = self._tracker_extremes[line_idx].get(tid, (0, 0))
                    print(f"[CROSS] L{line_idx+1} tid={tid} EXIT candidate disp={has_disp} extremes={extremes}")
                    if has_disp:
                        exit_tids.add(tid)

        # Apply cooldown for all counted tids
        for tid in entry_tids | exit_tids:
            for ex in self._tracker_extremes:
                ex.pop(tid, None)
            self._tracker_cooldown[tid] = self._frame_count

        new_entries = len(entry_tids)
        new_exits = len(exit_tids)

        self._validated_entries += new_entries
        self._validated_exits += new_exits
        entries = self._validated_entries
        exits = self._validated_exits
        occupancy = max(0, entries - exits)

        # Log new entry events to CSV + capture screenshots
        if self._csv_path and new_entries > 0:
            for i in range(new_entries):
                count = entries - new_entries + i + 1
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

        # Log new exit events to CSV
        if self._csv_path and new_exits > 0:
            for i in range(new_exits):
                count = exits - new_exits + i + 1
                ts = datetime.now()
                self._csv_writer.writerow([
                    ts.isoformat(),
                    "exit",
                    count,
                    "",
                ])
            self._csv_file.flush()

        # Prune stale tracker data every ~300 frames (was 30 — too aggressive with process_every_n_frames)
        if self._frame_count % 300 == 0:
            active_ids = detections.tracker_id if detections.tracker_id is not None else []
            self._prune_stale_trackers(active_ids)

        return {
            "entries": entries,
            "exits": exits,
            "occupancy": occupancy,
        }

    @property
    def line_zones(self):
        """Get all LineZone objects for annotation."""
        return [ld.line_zone for ld in self._lines]

    @property
    def line_zone(self):
        """Get first LineZone (backward compat for annotation)."""
        return self._lines[0].line_zone if self._lines else None

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
