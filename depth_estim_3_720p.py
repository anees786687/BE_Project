#!/usr/bin/env python3
import time
import rclpy
import cv2
import numpy as np
import threading
import yaml
from ultralytics import YOLO
from collections import deque
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import CompressedImage

# ── Config ─────────────────────────────────────────────────────────────────────
# Updated to native 720p calibration file — no focal-length hack needed
CALIB_FILE   = '/home/anees/rdk_yolo/stereo_calib.yaml'
DEPTH_OFFSET = 0.025   # metres — tune per ground truth test
DEPTH_THRESH = 0.25    # metres — arm trigger threshold

BOX_PERSIST_S  = 0.5
DEPTH_CACHE_MAX = 50


class ImgViewer(Node):
    def __init__(self):
        super().__init__('img_viewer')

        lK, lD, lR, lP, rK, rD, rR, rP, self.Q, w, h = self.load_calib(CALIB_FILE)
        self.get_logger().info(f"Calibration loaded: {w}x{h}")
        self.get_logger().info(
            f"Q[2][3](focal)={self.Q[2][3]:.4f}  Q[3][2](1/Tx)={self.Q[3][2]:.4f}")

        self.img_w = w   # 1280
        self.img_h = h   # 720
        self._size_logged = False

        # ── Rectification maps ────────────────────────────────────────────────
        self.map_lx, self.map_ly = cv2.initUndistortRectifyMap(
            lK, lD, lR, lP, (w, h), cv2.CV_32FC1)
        self.map_rx, self.map_ry = cv2.initUndistortRectifyMap(
            rK, rD, rR, rP, (w, h), cv2.CV_32FC1)

        # ── SGBM + WLS ────────────────────────────────────────────────────────
        # At 1280px: scale=2.0 → numDisparities=192
        # blockSize reduced 9→7 to compensate for 4× pixel count at 720p —
        # keeps per-frame SGBM time roughly comparable to 640p with block=9.
        # SGBM_3WAY kept for quality; if CPU can't keep up switch to SGBM_MODE_HH.
        scale    = w / 640.0                      # 2.0 at 1280px
        num_disp = int(96 * scale / 16) * 16      # 192
        block    = 7                               # was 9 — faster at higher res
        self.get_logger().info(f"SGBM numDisparities={num_disp} scale={scale:.2f}x block={block}")

        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity      = 0,
            numDisparities    = num_disp,
            blockSize         = block,
            P1                = 8  * 3 * block ** 2,
            P2                = 64 * 3 * block ** 2,
            disp12MaxDiff     = 2,
            uniquenessRatio   = 8,
            speckleWindowSize = 150,   # scaled up from 100 — 720p has more pixels
            speckleRange      = 2,
            preFilterCap      = 63,
            mode              = cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
        self.wls_filter    = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
        self.wls_filter.setLambda(8000)
        self.wls_filter.setSigmaColor(1.5)

        # ── YOLO ──────────────────────────────────────────────────────────────
        self.yolo_model = YOLO('recycle_detector_v6_best.pt')
        self.yolo_model.to('cuda')
        dummy = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        self.yolo_model(dummy, classes=[39, 41], conf=0.35, verbose=False)
        self.get_logger().info("YOLO warmed up")

        # ── Shared state ──────────────────────────────────────────────────────
        self.latest_frame      = None
        self.latest_depth      = None
        self.latest_left_rect  = None
        self.latest_right_rect = None
        self.bb_box_list       = []
        self._bb_last_updated  = 0.0

        self.last_depth = {}

        self.img_lock        = threading.Lock()
        self.depth_lock      = threading.Lock()
        self.bb_lock         = threading.Lock()
        self.rect_lock       = threading.Lock()
        self.img_event       = threading.Event()
        self._left_yolo_lock = threading.Lock()
        self._yolo_event     = threading.Event()

        self.depth_buffer    = deque(maxlen=5)
        self._left_for_yolo  = None

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

        # No focal-length hack needed — this calib file is native 1280x720
        # so Q is already correct. The 640x352 patch is intentionally removed.

        return lK, lD, lR, lP, rK, rD, rR, rP, Q, w, h

    # ── ROS callback ──────────────────────────────────────────────────────────
    def cb(self, msg: CompressedImage):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            with self.img_lock:
                self.latest_frame = img
            self.img_event.set()

    # ── Depth thread — rectify + SGBM + WLS (CPU) ────────────────────────────
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
                    f"Received combined={frame.shape[1]}x{frame.shape[0]} "
                    f"→ each half={left.shape[1]}x{left.shape[0]}")
                self._size_logged = True

            if left.shape[1] != self.img_w or left.shape[0] != self.img_h:
                self.get_logger().warn(
                    f"Frame {left.shape[1]}x{left.shape[0]} != "
                    f"calib {self.img_w}x{self.img_h} — skipping")
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

            # ── Depth ─────────────────────────────────────────────────────────
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

    # ── YOLO thread — GPU inference (parallel to SGBM) ───────────────────────
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
                classes = [39, 41],
                conf    = 0.35,
                iou     = 0.45,
                half    = True,
                imgsz   = (self.img_h, self.img_w),   # (720, 1280)
                verbose = False
            )

            boxes        = []
            current_keys = set()

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls   = int(box.cls[0])
                    conf  = float(box.conf[0])
                    label = 'bottle' if cls == 39 else 'cup'

                    d_val = None
                    if dep is not None:
                        dh, dw = dep.shape
                        bx1 = int(np.clip(x1, 0, dw - 1))
                        bx2 = int(np.clip(x2, 0, dw - 1))
                        by1 = int(np.clip(y1, 0, dh - 1))
                        by2 = int(np.clip(y2, 0, dh - 1))
                        ibx1 = bx1 + (bx2 - bx1) // 4
                        ibx2 = bx2 - (bx2 - bx1) // 4
                        iby1 = by1 + (by2 - by1) // 4
                        iby2 = by2 - (by2 - by1) // 4
                        roi   = dep[iby1:iby2, ibx1:ibx2]
                        valid = roi[np.isfinite(roi)]

                        if len(valid) >= 5:
                            d_val = float(np.percentile(valid, 10))
                        else:
                            # Transparent bottle fallback — border ring
                            pad  = 15   # slightly larger pad at 720p
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
                    key = f"{label}_{cx // 50}_{cy // 50}"
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


def main():
    rclpy.init()
    node = ImgViewer()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    while rclpy.ok():
        with node.img_lock:
            frame = node.latest_frame

        if frame is not None:
            half = frame.shape[0] // 2

            with node.rect_lock:
                left       = node.latest_left_rect
                right_disp = node.latest_right_rect
            if left is None:
                left = frame[:half, :].copy()
            else:
                left = left.copy()
            if right_disp is None:
                right_disp = frame[half:, :]

            with node.bb_lock:
                boxes = list(node.bb_box_list)

            closest = None
            if boxes:
                valid_depth = [(i, b[6]) for i, b in enumerate(boxes)
                               if b[6] is not None]
                if valid_depth:
                    closest = min(valid_depth, key=lambda x: x[1])[0]

            for i, (x1, y1, x2, y2, label, conf, d_val) in enumerate(boxes):
                is_closest = (i == closest)
                in_range   = d_val is not None and d_val < DEPTH_THRESH

                if in_range and is_closest:
                    color, thickness = (0, 0, 255), 3
                elif is_closest:
                    color, thickness = (0, 215, 255), 2
                else:
                    color, thickness = (0, 255, 0), 2

                depth_str = f"{d_val:.2f}m" if d_val is not None else "no depth"
                cv2.rectangle(left, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(left, f"{label} {conf:.2f} {depth_str}",
                            (x1, max(y1 - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(left, ((x1+x2)//2, (y1+y2)//2), 4, (0, 200, 255), -1)

            # ── Display resize ────────────────────────────────────────────────
            # At 720p: hconcat gives 2560×720.
            # Resize to 1920×270 to keep both frames side-by-side on screen.
            combined = cv2.hconcat([left, right_disp])
            combined = cv2.resize(combined, (1920, 270))
            for y in range(0, combined.shape[0], 40):
                cv2.line(combined, (0, y), (combined.shape[1], y), (0, 0, 255), 1)

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
                    d_scaled = cv2.inpaint(d_scaled, invalid_mask, 5, cv2.INPAINT_NS)

                d_color = cv2.applyColorMap(d_scaled, cv2.COLORMAP_TURBO)
                # Resize depth strip to match the 1920px wide combined frame
                d_color = cv2.resize(d_color, (1920, 180))

                cv2.putText(d_color, f"{lo:.2f}m", (5, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(d_color, f"{hi:.2f}m", (1880, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                display = np.vstack([combined, d_color])
            else:
                display = combined

            cv2.putText(display, "LEFT",  (10,  20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "RIGHT", (970, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Stereo + Depth + YOLO", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
