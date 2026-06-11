"""
ROS2 Image Subscriber with Skeleton Fallback.

When rclpy is available, subscribes to a ROS2 image topic and converts
incoming sensor_msgs/Image messages to JPEG frames using cv_bridge.

When rclpy is NOT available (or FORCE_SKELETON=true), generates synthetic
test frames so the frontend can be developed independently.

Reference: https://github.com/marcos-moura97/video_stream_ros2/blob/main/video_stream_ros2/app.py
"""

import os
import time
import logging
from threading import Thread, Lock, Event
from datetime import datetime

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread-safe frame buffer
# ---------------------------------------------------------------------------

class FrameBuffer:
    """Thread-safe container for the latest camera frame (JPEG bytes)."""

    def __init__(self):
        self._frame: bytes | None = None
        self._lock = Lock()
        self._event = Event()
        self._width: int = 0
        self._height: int = 0
        self._encoding: str = "unknown"
        self._connected: bool = False
        self._frame_count: int = 0

    def update(self, jpeg_bytes: bytes, width: int, height: int, encoding: str = "bgr8"):
        with self._lock:
            self._frame = jpeg_bytes
            self._width = width
            self._height = height
            self._encoding = encoding
            self._connected = True
            self._frame_count += 1
        self._event.set()

    @property
    def frame(self) -> bytes | None:
        with self._lock:
            return self._frame

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    @connected.setter
    def connected(self, value: bool):
        with self._lock:
            self._connected = value

    @property
    def info(self) -> dict:
        with self._lock:
            return {
                "width": self._width,
                "height": self._height,
                "encoding": self._encoding,
                "connected": self._connected,
                "frame_count": self._frame_count,
            }

    def wait(self, timeout: float = 1.0) -> bool:
        """Block until a new frame is available. Returns True if frame arrived."""
        result = self._event.wait(timeout=timeout)
        self._event.clear()
        return result


# Global buffer shared between the subscriber and the FastAPI server
frame_buffer = FrameBuffer()


# ---------------------------------------------------------------------------
# ROS2 subscriber (real mode)
# ---------------------------------------------------------------------------

def _start_ros2_subscriber(topic: str, quality: int):
    """Start the ROS2 node and subscribe to the image topic."""
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge

    bridge = CvBridge()

    rclpy.init(args=None)
    node = rclpy.create_node("visor_camera_subscriber")
    logger.info(f"ROS2 node created, subscribing to topic: {topic}")

    def on_image(msg: Image):
        try:
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            # Convert to BGR if needed for JPEG encoding
            if msg.encoding == "rgb8":
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            elif msg.encoding == "mono8":
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)

            h, w = cv_image.shape[:2]
            _, jpeg = cv2.imencode(".jpg", cv_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            frame_buffer.update(jpeg.tobytes(), w, h, msg.encoding)
        except Exception as e:
            logger.error(f"Failed to process image: {e}")

    node.create_subscription(Image, topic, on_image, 10)
    logger.info("ROS2 subscriber active — spinning…")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        frame_buffer.connected = False


# ---------------------------------------------------------------------------
# Skeleton / demo mode (no ROS2)
# ---------------------------------------------------------------------------

def _start_skeleton_stream(quality: int):
    """Generate synthetic test frames for development without ROS2."""
    logger.info("Starting SKELETON mode — generating synthetic frames")
    width, height = 640, 480
    frame_idx = 0

    while True:
        # Create a visually interesting test pattern
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Animated gradient background
        t = time.time()
        for y in range(height):
            hue = int((y / height * 180 + t * 20) % 180)
            img[y, :] = [hue, 180, 60]
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        # Grid overlay
        for x in range(0, width, 40):
            cv2.line(img, (x, 0), (x, height), (0, 60, 0), 1)
        for y in range(0, height, 40):
            cv2.line(img, (0, y), (width, y), (0, 60, 0), 1)

        # Center crosshair
        cx, cy = width // 2, height // 2
        cv2.line(img, (cx - 30, cy), (cx + 30, cy), (0, 230, 118), 1)
        cv2.line(img, (cx, cy - 30), (cx, cy + 30), (0, 230, 118), 1)
        cv2.circle(img, (cx, cy), 20, (0, 230, 118), 1)

        # Timestamp text
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        cv2.putText(img, f"SKELETON MODE", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 118), 1, cv2.LINE_AA)
        cv2.putText(img, now, (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 180, 200), 1, cv2.LINE_AA)
        cv2.putText(img, f"FRAME {frame_idx:06d}", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 140, 160), 1, cv2.LINE_AA)

        # Moving scan line
        scan_y = int((t * 100) % height)
        cv2.line(img, (0, scan_y), (width, scan_y), (0, 230, 118), 1)

        _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        frame_buffer.update(jpeg.tobytes(), width, height, "skeleton")
        frame_idx += 1

        time.sleep(1 / 30)  # ~30 FPS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_subscriber_thread: Thread | None = None


def start_subscriber():
    """Start the image subscriber in a background thread."""
    global _subscriber_thread

    topic = os.getenv("ROS2_IMAGE_TOPIC", "/image")
    quality = int(os.getenv("STREAM_QUALITY", "80"))
    force_skeleton = os.getenv("FORCE_SKELETON", "false").lower() == "true"

    use_ros2 = False
    if not force_skeleton:
        try:
            import rclpy  # noqa: F401
            use_ros2 = True
        except ImportError:
            logger.warning("rclpy not found — falling back to skeleton mode")

    if use_ros2:
        target = _start_ros2_subscriber
        args = (topic, quality)
    else:
        target = _start_skeleton_stream
        args = (quality,)

    _subscriber_thread = Thread(target=target, args=args, daemon=True)
    _subscriber_thread.start()
    logger.info(f"Subscriber thread started (mode={'ros2' if use_ros2 else 'skeleton'})")


def get_mode() -> str:
    """Return the current operating mode."""
    force_skeleton = os.getenv("FORCE_SKELETON", "false").lower() == "true"
    if force_skeleton:
        return "skeleton"
    try:
        import rclpy  # noqa: F401
        return "ros2"
    except ImportError:
        return "skeleton"
