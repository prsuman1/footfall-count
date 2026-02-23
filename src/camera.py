import cv2
import threading
import queue
import time
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;30000000"


class RTSPCamera:
    """Thread-safe camera reader that always provides the latest frame.

    Supports RTSP URLs, video files, and webcam (device index).
    Uses a background thread to continuously grab frames, dropping old ones
    to avoid latency buildup.
    """

    def __init__(self, source, camera_id="cam_0"):
        self.source = source
        self.camera_id = camera_id
        self.frame_queue = queue.Queue(maxsize=10)
        self.stopped = False
        self.connected = False
        self.consecutive_failures = 0
        self.max_failures = 100
        self.cap = None
        self.fps = 30
        self.width = 0
        self.height = 0
        self._is_file = False
        self._lock = threading.Lock()

    def start(self):
        """Connect and start the background reader thread."""
        self._detect_source_type()
        self._connect()
        t = threading.Thread(target=self._reader_loop, daemon=True)
        t.start()
        return self

    def _detect_source_type(self):
        """Determine if source is a file, webcam index, or RTSP URL."""
        if isinstance(self.source, int):
            self._is_file = False
        elif isinstance(self.source, str):
            if self.source.startswith("rtsp://") or self.source.startswith("rtsps://"):
                self._is_file = False
            elif os.path.isfile(self.source):
                self._is_file = True
            else:
                # Assume it could be a URL or device path
                self._is_file = False

    def _connect(self):
        """Connect or reconnect to the camera/video source."""
        with self._lock:
            if self.cap:
                self.cap.release()

            if isinstance(self.source, int):
                self.cap = cv2.VideoCapture(self.source)
            elif self.source.startswith("rtsp://") or self.source.startswith("rtsps://"):
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                self.cap = cv2.VideoCapture(self.source)

            self.connected = self.cap.isOpened()
            if self.connected:
                self.consecutive_failures = 0
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[{self.camera_id}] Connected: {self.width}x{self.height} @ {self.fps:.0f}fps")
            else:
                print(f"[{self.camera_id}] Failed to connect to {self.source}")

    def _reconnect(self):
        """Reconnect with exponential backoff. Returns True on success."""
        if self._is_file:
            # For video files, stop at end
            print(f"[{self.camera_id}] Video file ended")
            self.stopped = True
            return False

        max_retries = 10
        for attempt in range(max_retries):
            if self.stopped:
                return False
            delay = min(2 ** attempt, 30)
            print(f"[{self.camera_id}] Reconnecting in {delay}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
            self._connect()
            if self.connected:
                return True
        return False

    def _reader_loop(self):
        """Background thread: continuously grab frames, keep only latest."""
        while not self.stopped:
            if not self.connected:
                if not self._reconnect():
                    print(f"[{self.camera_id}] Giving up after max retries")
                    self.stopped = True
                    break
                continue

            with self._lock:
                if self.cap is None:
                    continue
                ret = self.cap.grab()

            if not ret:
                self.consecutive_failures += 1
                if self.consecutive_failures > self.max_failures:
                    self.connected = False
                    print(f"[{self.camera_id}] Lost connection ({self.consecutive_failures} failures)")
                continue

            self.consecutive_failures = 0

            with self._lock:
                if self.cap is None:
                    continue
                ret, frame = self.cap.retrieve()

            if ret and frame is not None:
                if self._is_file:
                    # For video files, block until space available (no dropping)
                    self.frame_queue.put(frame)
                else:
                    # For live streams, drop old frame, keep only latest
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.frame_queue.put(frame)

            # For video files, pace reading to approximate real-time
            if self._is_file and self.fps > 0:
                time.sleep(1.0 / self.fps)
            elif not self._is_file:
                # For live streams, small sleep to prevent CPU spin
                time.sleep(0.001)

    def read(self):
        """Get the latest frame. Returns None on timeout."""
        try:
            return self.frame_queue.get(timeout=5)
        except queue.Empty:
            return None

    def stop(self):
        """Stop the reader thread and release resources."""
        self.stopped = True
        with self._lock:
            if self.cap:
                self.cap.release()
                self.cap = None
        print(f"[{self.camera_id}] Stopped")

    @property
    def is_running(self):
        return not self.stopped and self.connected
