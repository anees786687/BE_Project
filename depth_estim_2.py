#!/usr/bin/env python3
"""
Waste-Sorting Robot — Stereo Depth + YOLO Perception Node
Serves live MJPEG feeds over FastAPI (replaces cv2.imshow windows).

Streams:
  GET /api/stream/stereo  — rectified left + right side-by-side with YOLO boxes
  GET /api/stream/depth   — TURBO colormap depth map with scale bar
  GET /api/status         — JSON status (resolution, frame count, mode)
  GET /api/snapshot/stereo|depth — single JPEG download
  GET /api/health         — health check

Run: python depth_estim_4.py
     Then open http://localhost:8000 in a browser (or the React frontend).
"""

import os
import time
import logging
import threading
from contextlib import asynccontextmanager
from threading import Thread, Lock, Event
from collections import deque

import cv2
import numpy as np
import yaml
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, Response
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CALIB_FILE   = '/home/anees/rdk_yolo/stereo_calib_640x352.yaml'
DEPTH_OFFSET = 0.025   # metres
DEPTH_THRESH = 0.25    # metres — arm trigger threshold
BOX_PERSIST_S = 0.5
DEPTH_CACHE_MAX = 50

WEB_HOST = '0.0.0.0'
WEB_PORT = 8000
STREAM_QUALITY = int(os.getenv('STREAM_QUALITY', '75'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thread-safe dual frame buffer
# ---------------------------------------------------------------------------

class DualFrameBuffer:
    """Holds the two latest JPEG-encoded frames (stereo view + depth map)."""

    def __init__(self):
        self._frames   = {'stereo': None, 'depth': None}
        self._lock     = Lock()
        self._events   = {'stereo': Event(), 'depth': Event()}
        self._counts   = {'stereo': 0, 'depth': 0}
        self._width    = 0
        self._height   = 0
        self._connected = False

    def update(self, key: str, bgr_frame: np.ndarray):
        """Encode a BGR numpy frame to JPEG and store it."""
        _, buf = cv2.imencode(
            '.jpg', bgr_frame,
            [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY]
        )
        jpeg = buf.tobytes()
        h, w = bgr_frame.shape[:2]
        with self._lock:
            self._frames[key] = jpeg
            self._counts[key] += 1
            if key == 'stereo':
                self._width  = w
                self._height = h
            self._connected = True
        self._events[key].set()

    def get_jpeg(self, key: str):
        with self._lock:
            return self._frames.get(key)

    def wait(self, key: str, timeout: float = 0.1) -> bool:
        result = self._events[key].wait(timeout=timeout)
        self._events[key].clear()
        return result

    @property
    def info(self) -> dict:
        with self._lock:
            return {
                'connected':    self._connected,
                'width':        self._width,
                'height':       self._height,
                'stereo_count': self._counts['stereo'],
                'depth_count':  self._counts['depth'],
            }


# Global buffer — shared between ImgViewer and FastAPI
frame_buffer = DualFrameBuffer()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('VISOR backend ready')
    yield
    logger.info('VISOR backend shutting down')


app = FastAPI(title='VISOR — Waste Robot Camera Dashboard', version='2.0.0', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


def _mjpeg_generator(key: str):
    while True:
        frame = frame_buffer.get_jpeg(key)
        if frame is not None:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + frame
                + b'\r\n'
            )
        frame_buffer.wait(key, timeout=0.1)


@app.get('/api/stream/{key}')
async def video_stream(key: str):
    if key not in ('stereo', 'depth'):
        return JSONResponse({'error': 'Unknown stream key'}, status_code=404)
    return StreamingResponse(
        _mjpeg_generator(key),
        media_type='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'Connection': 'close',
        },
    )

# Keep /api/stream as an alias for the stereo feed (backward-compat with VISOR frontend)
@app.get('/api/stream')
async def video_stream_default():
    return await video_stream('stereo')


@app.get('/api/status')
async def get_status():
    info = frame_buffer.info
    w, h = info['width'], info['height']
    return JSONResponse({
        'connected':    info['connected'],
        'width':        w,
        'height':       h,
        'encoding':     'bgr8',
        'frame_count':  info['stereo_count'],
        'stereo_count': info['stereo_count'],
        'depth_count':  info['depth_count'],
        'mode':         'ros2',
        'topic':        '/image_combine_jpeg',
        'resolution':   f'{w}×{h}' if w > 0 else '—',
        'device':       'SC230AI Stereo — ROS2 /image_combine_jpeg',
    })


@app.get('/api/snapshot/{key}')
async def snapshot(key: str):
    if key not in ('stereo', 'depth'):
        return JSONResponse({'error': 'Unknown stream key'}, status_code=404)
    frame = frame_buffer.get_jpeg(key)
    if frame is None:
        return JSONResponse({'error': 'No frame available yet'}, status_code=503)
    return Response(
        content=frame,
        media_type='image/jpeg',
        headers={
            'Content-Disposition':
                f'attachment; filename=visor_{key}_{int(time.time())}.jpg',
        },
    )

# Alias for default snapshot (VISOR frontend compat)
@app.get('/api/snapshot')
async def snapshot_default():
    return await snapshot('stereo')


@app.get('/api/health')
async def health():
    return {'status': 'ok', 'mode': 'ros2'}


# ---------------------------------------------------------------------------
# ROS2 perception node
# ---------------------------------------------------------------------------

class ImgViewer(Node):
    def __init__(self):
        super().__init__('img_viewer')

        lK, lD, lR, lP, rK, rD, rR, rP, self.Q, w, h = self.load_calib(CALIB_FILE)
        self.get_logger().info(f'Calibration loaded: {w}x{h}')
        self.get_logger().info(
            f'Q[2][3](focal)={self.Q[2][3]:.4f}  Q[3][2](1/Tx)={self.Q[3][2]:.4f}')

        self.img_w = w
        self.img_h = h
        self._size_logged = False

        # ── Rectification maps ────────────────────────────────────────────────
        self.map_lx, self.map_ly = cv2.initUndistortRectifyMap(
            lK, lD, lR, lP, (w, h), cv2.CV_32FC1)
        self.map_rx, self.map_ry = cv2.initUndistortRectifyMap(
            rK, rD, rR, rP, (w, h), cv2.CV_32FC1)

        # ── SGBM + WLS ────────────────────────────────────────────────────────
        scale    = w / 640.0
        num_disp = int(128 * scale / 16) * 16
        block    = 5
        self.get_logger().info(f'SGBM numDisparities={num_disp} scale={scale:.2f}x')

        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity      = 0,
            numDisparities    = num_disp,
            blockSize         = block,
            P1                = 8  * 3 * block ** 2,
            P2                = 32 * 3 * block ** 2,
            disp12MaxDiff     = 1,
            uniquenessRatio   = 10,
            speckleWindowSize = 150,
            speckleRange      = 2,
            preFilterCap      = 63,
            mode              = cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
        self.wls_filter    = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
        self.wls_filter.setLambda(4000)
        self.wls_filter.setSigmaColor(0.8)

        # ── YOLO ──────────────────────────────────────────────────────────────
        self.yolo_model = YOLO('recycle_detector_v6_best.pt')
        self.yolo_model.to('cuda')
        dummy = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        self.yolo_model(dummy, classes=[0, 1], conf=0.35, verbose=False)
        self.get_logger().info('YOLO warmed up')

        # ── Shared state ──────────────────────────────────────────────────────
        self.latest_frame      = None
        self.latest_depth      = None
        self.latest_left_rect  = None
        self.latest_right_rect = None
        self.bb_box_list       = []
        self._bb_last_updated  = 0.0
        self.last_depth        = {}

        # ── Detection publisher ───────────────────────────────────────────────
        self.det_pub = self.create_publisher(String, '/detections', 10)

        self.img_lock    = threading.Lock()
        self.depth_lock  = threading.Lock()
        self.bb_lock     = threading.Lock()
        self.rect_lock   = threading.Lock()
        self.img_event   = threading.Event()

        self.depth_buffer    = deque(maxlen=5)
        self._left_for_yolo  = None
        self._left_yolo_lock = threading.Lock()
        self._yolo_event     = threading.Event()

        threading.Thread(target=self._depth_thread, daemon=True).start()
        threading.Thread(target=self._yolo_thread,  daemon=True).start()

        self.create_subscription(
            CompressedImage, '/image_combine_jpeg', self.cb, 10)

    # ── Calibration loader ────────────────────────────────────────────────────
    def load_calib(self, path):
        with open(path) as f:
            c = yaml.safe_load(f)

        def mat(data, shape):
            flat = [v for row in data for v in
                    (row if hasattr(row, '__iter__') else [row])]
            return np.array(flat, dtype=np.float64).reshape(shape)

        lK = mat(c['left_camera']['K'],  (3, 3))
        lD = mat(c['left_camera']['D'],  (1, 5))
        lR = mat(c['left_camera']['R'],  (3, 3))
        lP = mat(c['left_camera']['P'],  (3, 4))
        rK = mat(c['right_camera']['K'], (3, 3))
        rD = mat(c['right_camera']['D'], (1, 5))
        rR = mat(c['right_camera']['R'], (3, 3))
        rP = mat(c['right_camera']['P'], (3, 4))
        Q  = mat(c['Q'],                 (4, 4))
        w  = c['image_width']
        h  = c['image_height']

        if w == 640 and h == 352:
            Q[2][3] = 591.4976270904475 * 0.5

        return lK, lD, lR, lP, rK, rD, rR, rP, Q, w, h

    # ── ROS callback ──────────────────────────────────────────────────────────
    def cb(self, msg: CompressedImage):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            with self.img_lock:
                self.latest_frame = img
            self.img_event.set()

    # ── Depth thread ─────────────────────────────────────────────────────────
    def _depth_thread(self):
        while True:
            self.img_event.wait()
            self.img_event.clear()

            with self.img_lock:
                frame = self.latest_frame
            if frame is None:
                continue

            half  = frame.shape[0] // 2
            left  = frame[:half, :]
            right = frame[half:, :]

            if not self._size_logged:
                self.get_logger().info(
                    f'Received combined={frame.shape[1]}x{frame.shape[0]} '
                    f'→ each half={left.shape[1]}x{left.shape[0]}')
                self._size_logged = True

            if left.shape[1] != self.img_w or left.shape[0] != self.img_h:
                self.get_logger().warn(
                    f'Frame {left.shape[1]}x{left.shape[0]} != '
                    f'calib {self.img_w}x{self.img_h} — skipping')
                continue

            if np.mean(right) < 5.0:
                self.depth_buffer.clear()
                self.get_logger().warn('Right camera dead — depth buffer flushed')
                continue

            # ── Rectify ───────────────────────────────────────────────────────
            left_rect  = cv2.remap(left,  self.map_lx, self.map_ly, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right, self.map_rx, self.map_ry, cv2.INTER_LINEAR)

            with self.rect_lock:
                self.latest_left_rect  = left_rect
                self.latest_right_rect = right_rect

            with self._left_yolo_lock:
                self._left_for_yolo = left_rect
            self._yolo_event.set()

            # ── SGBM + WLS ────────────────────────────────────────────────────
            left_gray  = cv2.cvtColor(left_rect,  cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

            disp_left  = self.left_matcher.compute(left_gray, right_gray)
            disp_right = self.right_matcher.compute(right_gray, left_gray)
            disp_filt  = self.wls_filter.filter(
                disp_left, left_rect, disparity_map_right=disp_right)
            disp = disp_filt.astype(np.float32) / 16.0

            points_3d = cv2.reprojectImageTo3D(disp, self.Q)
            depth = points_3d[:, :, 2].astype(np.float32)

            depth[disp  <= 0]          = np.nan
            depth[depth <= 0]          = np.nan
            depth[depth >  5.0]        = np.nan
            depth[~np.isfinite(depth)] = np.nan
            depth = depth - DEPTH_OFFSET
            depth[depth <= 0] = np.nan

            self.depth_buffer.append(depth)
            if len(self.depth_buffer) < 3:
                continue
            with np.errstate(all='ignore'):
                avg_depth = np.nanmedian(
                    np.stack(list(self.depth_buffer), axis=0), axis=0)

            with self.depth_lock:
                self.latest_depth = avg_depth

    # ── YOLO thread ──────────────────────────────────────────────────────────
    def _yolo_thread(self):
        while True:
            self._yolo_event.wait()
            self._yolo_event.clear()

            with self._left_yolo_lock:
                left = self._left_for_yolo
            if left is None:
                continue

            with self.depth_lock:
                dep = self.latest_depth

            results = self.yolo_model(
                left,
                classes = [0, 1],
                conf    = 0.55,
                iou     = 0.45,
                half    = True,
                imgsz   = (self.img_h, self.img_w),
                verbose = False
            )

            boxes        = []
            current_keys = set()

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls   = int(box.cls[0])
                    conf  = float(box.conf[0])
                    label = 'plastic_bottle' if cls == 0 else 'aluminum_can'

                    d_val = None
                    if dep is not None:
                        dh, dw = dep.shape
                        bx1 = int(np.clip(x1, 0, dw - 1))
                        bx2 = int(np.clip(x2, 0, dw - 1))
                        by1 = int(np.clip(y1, 0, dh - 1))
                        by2 = int(np.clip(y2, 0, dh - 1))

                        if label == 'aluminum_can':
                            ibx1 = bx1 + (bx2 - bx1) // 5
                            ibx2 = bx2 - (bx2 - bx1) // 5
                            iby1 = by1 + (by2 - by1) // 8
                            iby2 = by2 - (by2 - by1) // 8
                            min_valid = 3
                            pct = 20
                        else:
                            ibx1 = bx1 + (bx2 - bx1) // 4
                            ibx2 = bx2 - (bx2 - bx1) // 4
                            iby1 = by1 + (by2 - by1) // 4
                            iby2 = by2 - (by2 - by1) // 4
                            min_valid = 5
                            pct = 10

                        roi   = dep[iby1:iby2, ibx1:ibx2]
                        valid = roi[np.isfinite(roi)]

                        if len(valid) >= min_valid:
                            d_val = float(np.percentile(valid, pct))
                        else:
                            pad  = 10
                            ry1  = max(by1 - pad, 0)
                            ry2  = min(by2 + pad, dh)
                            rx1  = max(bx1 - pad, 0)
                            rx2  = min(bx2 + pad, dw)
                            ring = dep[ry1:ry2, rx1:rx2].copy()
                            ring[by1-ry1:by2-ry1, bx1-rx1:bx2-rx1] = np.nan
                            valid_ring = ring[np.isfinite(ring)]
                            if len(valid_ring) >= 5:
                                d_val = float(np.percentile(valid_ring, 10))

                    cx  = (x1 + x2) // 2
                    cy  = (y1 + y2) // 2
                    key = f'{label}_{cx // 50}_{cy // 50}'
                    current_keys.add(key)

                    bbox_area = (x2 - x1) * (y2 - y1)
                    if d_val is not None and key in self.last_depth:
                        prev       = self.last_depth[key]
                        depth_jump = abs(d_val - prev['depth']) > 0.40
                        area_ratio = bbox_area / max(prev['area'], 1)
                        area_jump  = area_ratio < 0.5 or area_ratio > 2.0
                        if depth_jump and area_jump:
                            self.last_depth[key] = {'depth': d_val, 'area': bbox_area}
                        elif depth_jump:
                            d_val = prev['depth']
                        else:
                            self.last_depth[key] = {'depth': d_val, 'area': bbox_area}
                    elif d_val is not None:
                        self.last_depth[key] = {'depth': d_val, 'area': bbox_area}

                    if d_val is None and key in self.last_depth:
                        d_val = self.last_depth[key]['depth']

                    boxes.append((x1, y1, x2, y2, label, conf, d_val))

            for k in [k for k in self.last_depth if k not in current_keys]:
                del self.last_depth[k]
            if len(self.last_depth) > DEPTH_CACHE_MAX:
                self.last_depth.clear()

            now_ts = time.time()
            with self.bb_lock:
                if boxes:
                    self.bb_box_list      = boxes
                    self._bb_last_updated = now_ts
                elif (now_ts - self._bb_last_updated) > BOX_PERSIST_S:
                    self.bb_box_list = []

            if boxes:
                parts = []
                for (_, _, _, _, label, _, d_val) in boxes:
                    depth_str = f'{d_val:.3f}' if d_val is not None else 'none'
                    parts.append(f'{label}:{depth_str}')
                msg = String()
                msg.data = ','.join(parts)
                self.det_pub.publish(msg)


# ---------------------------------------------------------------------------
# Render loop — builds display frames and pushes to frame_buffer
# ---------------------------------------------------------------------------

def render_loop(node: ImgViewer):
    """
    Mirrors the old main() cv2.imshow logic but pushes frames into
    frame_buffer instead of displaying them on-screen.
    Runs in its own daemon thread.
    """
    while rclpy.ok():
        with node.img_lock:
            frame = node.latest_frame

        if frame is None:
            time.sleep(0.01)
            continue

        half = frame.shape[0] // 2

        with node.rect_lock:
            left = node.latest_left_rect
        if left is None:
            left = frame[:half, :].copy()
        else:
            left = left.copy()

        with node.bb_lock:
            boxes = list(node.bb_box_list)

        # ── Draw YOLO boxes on rectified left ─────────────────────────────────
        closest = None
        if boxes:
            valid_depth = [(i, b[6]) for i, b in enumerate(boxes) if b[6] is not None]
            if valid_depth:
                closest = min(valid_depth, key=lambda x: x[1])[0]

        for i, (x1, y1, x2, y2, label, conf, d_val) in enumerate(boxes):
            is_closest = (i == closest)
            in_range   = d_val is not None and d_val < DEPTH_THRESH

            if in_range:
                color, thickness = (0, 0, 255), 3
            elif is_closest:
                color, thickness = (0, 215, 255), 2
            else:
                color, thickness = (0, 255, 0), 2

            depth_str = f'{d_val:.2f}m' if d_val is not None else 'no depth'
            cv2.rectangle(left, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(left, f'{label} {conf:.2f} {depth_str}',
                        (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(left, ((x1+x2)//2, (y1+y2)//2), 4, (0, 200, 255), -1)

        # ── Stereo frame (left camera only, no epipolar lines) ────────────────
        display = cv2.resize(left, (1280, 720))

        frame_buffer.update('stereo', display)

        # ── Depth frame ───────────────────────────────────────────────────────
        with node.depth_lock:
            depth = node.latest_depth

        if depth is not None:
            d_vis = np.nan_to_num(depth, nan=0.0)

            valid_vals = d_vis[d_vis > 0.05]
            if len(valid_vals) > 100:
                lo = float(np.percentile(valid_vals, 5))
                hi = float(np.percentile(valid_vals, 95))
            else:
                lo, hi = 0.1, 3.0

            d_norm   = np.clip((d_vis - lo) / max(hi - lo, 0.01), 0, 1)
            d_scaled = (d_norm * 255).astype(np.uint8)

            invalid_mask = (d_vis <= 0.05).astype(np.uint8)
            if invalid_mask.any():
                d_scaled = cv2.inpaint(d_scaled, invalid_mask, 12, cv2.INPAINT_NS)

            d_smooth = cv2.bilateralFilter(d_scaled, d=7, sigmaColor=18, sigmaSpace=18)
            d_color  = cv2.applyColorMap(d_smooth, cv2.COLORMAP_TURBO)
            d_color  = cv2.resize(d_color, (1280, 720))

            bar_h    = 20
            gradient = np.linspace(0, 255, 1280, dtype=np.uint8)
            bar_raw  = np.tile(gradient, (bar_h, 1))
            bar_col  = cv2.applyColorMap(bar_raw, cv2.COLORMAP_TURBO)
            mid = (lo + hi) / 2
            for val, x in [(lo, 10), (mid, 610), (hi, 1210)]:
                cv2.putText(bar_col, f'{val:.2f}m', (x, bar_h - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                            cv2.LINE_AA)
            cv2.putText(d_color, 'DEPTH MAP', (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                        cv2.LINE_AA)
            depth_display = np.vstack([d_color, bar_col])
            frame_buffer.update('depth', depth_display)

        time.sleep(0.033)   # ~30 fps render cap


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    rclpy.init()
    node = ImgViewer()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    # ROS2 spin thread
    Thread(target=executor.spin, daemon=True).start()

    # Render loop thread (builds frames, feeds frame_buffer)
    Thread(target=render_loop, args=(node,), daemon=True).start()

    logger.info(f'Starting VISOR web server on http://{WEB_HOST}:{WEB_PORT}')
    logger.info('  Stereo stream : /api/stream/stereo')
    logger.info('  Depth stream  : /api/stream/depth')
    logger.info('  Status        : /api/status')

    # FastAPI runs on the main thread (blocks until Ctrl-C)
    uvicorn.run(app, host=WEB_HOST, port=WEB_PORT, log_level='warning')

    rclpy.shutdown()


if __name__ == '__main__':
    main()