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
from std_msgs.msg import String

CALIB_FILE   = '/home/anees/rdk_yolo/stereo_calib_640x352.yaml'
DEPTH_OFFSET = 0.025   # metres — tune per ground truth test
DEPTH_THRESH = 0.25    # metres — arm trigger threshold

# Bbox persistence: keep last good boxes for this many seconds when conf dips
BOX_PERSIST_S = 0.5

# Max entries in last_depth cache — guards against unbounded growth
DEPTH_CACHE_MAX = 50


class ImgViewer(Node):
    def __init__(self):
        super().__init__('img_viewer')

        lK, lD, lR, lP, rK, rD, rR, rP, self.Q, w, h = self.load_calib(CALIB_FILE)
        self.get_logger().info(f"Calibration loaded: {w}x{h}")
        self.get_logger().info(
            f"Q[2][3](focal)={self.Q[2][3]:.4f}  Q[3][2](1/Tx)={self.Q[3][2]:.4f}")

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
        # Increase numDisparities for better close-range coverage (<0.5m)
        num_disp = int(128 * scale / 16) * 16
        # Smaller block = finer detail, less blurring on object edges
        block    = 5
        self.get_logger().info(f"SGBM numDisparities={num_disp} scale={scale:.2f}x")

        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity      = 0,
            numDisparities    = num_disp,
            blockSize         = block,
            P1                = 8  * 3 * block ** 2,
            P2                = 32 * 3 * block ** 2,  # lower P2/P1 ratio = sharper depth discontinuities
            disp12MaxDiff     = 1,                    # stricter L-R consistency
            uniquenessRatio   = 10,                   # reject more ambiguous matches
            speckleWindowSize = 150,                  # larger speckle filter kills isolated noise blobs
            speckleRange      = 2,
            preFilterCap      = 63,
            mode              = cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
        self.wls_filter    = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
        # Lower lambda = less smoothing = sharper edges; lower sigma = edge-aware
        self.wls_filter.setLambda(4000)
        self.wls_filter.setSigmaColor(0.8)

        # ── YOLO — load once, warm up once ────────────────────────────────────
        self.yolo_model = YOLO('recycle_detector_v6_best.pt')
        self.yolo_model.to('cuda')
        dummy = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        self.yolo_model(dummy, classes=[0, 1], conf=0.35, verbose=False)
        self.get_logger().info("YOLO warmed up")

        # ── Shared state ──────────────────────────────────────────────────────
        self.latest_frame      = None
        self.latest_depth      = None
        self.latest_left_rect  = None          # rectified left frame for display
        self.latest_right_rect = None          # rectified right frame for display
        self.bb_box_list       = []
        self._bb_last_updated  = 0.0

        # last_depth: key → {'depth': float, 'area': int}
        self.last_depth = {}

        # ── Detection result publisher ───────────────────────────────────────
        self.det_pub = self.create_publisher(String, '/detections', 10)

        self.img_lock    = threading.Lock()
        self.depth_lock  = threading.Lock()
        self.bb_lock     = threading.Lock()
        self.rect_lock   = threading.Lock()    # guards latest_right_rect
        self.img_event   = threading.Event()
        self.frame_event = threading.Event()   # signals new frame to YOLO thread

        # Temporal depth buffer — median over last 5 frames reduces noise
        self.depth_buffer = deque(maxlen=5)

        # ── Two decoupled worker threads ──────────────────────────────────────
        # depth_thread: rectify + SGBM + WLS (CPU-bound, ~80-120ms)
        # yolo_thread:  YOLO inference         (GPU-bound, ~15-30ms)
        # They run in parallel — depth_thread signals yolo_thread when a new
        # left frame is ready so YOLO always works on the most recent frame.
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
            Q[2][3] = 591.4976270904475 * 0.5   # focal fix for non-uniform downscale

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

            # Dead camera guard — flush stale depth buffer on disconnect
            if np.mean(right) < 5.0:
                self.depth_buffer.clear()
                self.get_logger().warn('Right camera dead — depth buffer flushed')
                continue

            # ── Rectify ───────────────────────────────────────────────────────
            left_rect  = cv2.remap(left,  self.map_lx, self.map_ly, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right, self.map_rx, self.map_ry, cv2.INTER_LINEAR)

            # Share rectified frames for display
            with self.rect_lock:
                self.latest_left_rect  = left_rect
                self.latest_right_rect = right_rect

            # Signal YOLO thread with the rectified left frame
            with self._left_yolo_lock:
                self._left_for_yolo = left_rect
            self._yolo_event.set()

            # ── SGBM + WLS disparity ──────────────────────────────────────────
            left_gray  = cv2.cvtColor(left_rect,  cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

            disp_left  = self.left_matcher.compute(left_gray, right_gray)
            disp_right = self.right_matcher.compute(right_gray, left_gray)
            disp_filt  = self.wls_filter.filter(
                disp_left, left_rect, disparity_map_right=disp_right)
            disp = disp_filt.astype(np.float32) / 16.0

            # ── Disparity → metric depth ──────────────────────────────────────
            points_3d = cv2.reprojectImageTo3D(disp, self.Q)
            depth = points_3d[:, :, 2].astype(np.float32)

            depth[disp  <= 0]          = np.nan
            depth[depth <= 0]          = np.nan
            depth[depth >  5.0]        = np.nan
            depth[~np.isfinite(depth)] = np.nan

            # Apply calibration offset (was a no-op comment in original)
            depth = depth - DEPTH_OFFSET
            depth[depth <= 0] = np.nan

            # Temporal median
            self.depth_buffer.append(depth)
            if len(self.depth_buffer) < 3:
                continue
            with np.errstate(all='ignore'):
                avg_depth = np.nanmedian(
                    np.stack(list(self.depth_buffer), axis=0), axis=0)

            with self.depth_lock:
                self.latest_depth = avg_depth

    # ── YOLO thread — GPU inference (runs parallel to SGBM) ──────────────────
    def _yolo_thread(self):
        while True:
            self._yolo_event.wait()
            self._yolo_event.clear()

            with self._left_yolo_lock:
                left = self._left_for_yolo
            if left is None:
                continue

            # Acquire depth snapshot once before the box loop — not per box
            with self.depth_lock:
                dep = self.latest_depth

            # Skip disparity computation if no depth available yet
            # but still run YOLO so boxes appear from frame 1
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

                    # ── Depth sampling — 10th percentile of inner 50% ROI ─────
                    d_val = None
                    if dep is not None:
                        dh, dw = dep.shape
                        bx1 = int(np.clip(x1, 0, dw - 1))
                        bx2 = int(np.clip(x2, 0, dw - 1))
                        by1 = int(np.clip(y1, 0, dh - 1))
                        by2 = int(np.clip(y2, 0, dh - 1))
                        # ── ROI strategy depends on class ─────────────────────
                        # aluminum_can: shiny surface → SGBM gives sparse hits
                        #   → use a wide vertical centre strip (inner ~60% W,
                        #     inner ~75% H) to maximise valid pixel count, and
                        #     lower the valid threshold to 3 pixels.
                        # plastic_bottle: semi-transparent → keep existing
                        #   inner-50% crop with 5-pixel threshold.
                        if label == 'aluminum_can':
                            ibx1 = bx1 + (bx2 - bx1) // 5      # inner 60% W
                            ibx2 = bx2 - (bx2 - bx1) // 5
                            iby1 = by1 + (by2 - by1) // 8      # inner 75% H
                            iby2 = by2 - (by2 - by1) // 8
                            min_valid = 3
                            pct = 20   # less aggressive near-edge bias on shiny surface
                        else:
                            ibx1 = bx1 + (bx2 - bx1) // 4      # inner 50%
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
                            # Transparent bottle / specular can fallback —
                            # sample border ring around the bbox
                            pad   = 10
                            ry1   = max(by1 - pad, 0)
                            ry2   = min(by2 + pad, dh)
                            rx1   = max(bx1 - pad, 0)
                            rx2   = min(bx2 + pad, dw)
                            ring  = dep[ry1:ry2, rx1:rx2].copy()
                            # Blank out interior so we only sample the border
                            ring[by1-ry1:by2-ry1, bx1-rx1:bx2-rx1] = np.nan
                            valid_ring = ring[np.isfinite(ring)]
                            if len(valid_ring) >= 5:
                                d_val = float(np.percentile(valid_ring, 10))

                    # ── Depth key ─────────────────────────────────────────────
                    cx     = (x1 + x2) // 2
                    cy     = (y1 + y2) // 2
                    key    = f"{label}_{cx // 50}_{cy // 50}"
                    current_keys.add(key)

                    # ── Depth continuity — new-object-aware ───────────────────
                    bbox_area = (x2 - x1) * (y2 - y1)
                    if d_val is not None and key in self.last_depth:
                        prev       = self.last_depth[key]
                        depth_jump = abs(d_val - prev['depth']) > 0.40
                        area_ratio = bbox_area / max(prev['area'], 1)
                        area_jump  = area_ratio < 0.5 or area_ratio > 2.0
                        if depth_jump and area_jump:
                            # New object entered cell — reset cache
                            self.last_depth[key] = {'depth': d_val, 'area': bbox_area}
                        elif depth_jump:
                            # Noise spike — reject, keep cached value
                            d_val = prev['depth']
                        else:
                            self.last_depth[key] = {'depth': d_val, 'area': bbox_area}
                    elif d_val is not None:
                        self.last_depth[key] = {'depth': d_val, 'area': bbox_area}

                    # If live depth failed but we have a cached value, use it
                    # so the red-trigger works even during cold-start or sparse frames
                    if d_val is None and key in self.last_depth:
                        d_val = self.last_depth[key]['depth']

                    boxes.append((x1, y1, x2, y2, label, conf, d_val))

            # Prune stale cache entries
            for k in [k for k in self.last_depth if k not in current_keys]:
                del self.last_depth[k]

            # Guard against unbounded cache growth in long sessions
            if len(self.last_depth) > DEPTH_CACHE_MAX:
                self.last_depth.clear()

            # ── Bbox persistence — prevents flicker ───────────────────────────
            now_ts = time.time()
            with self.bb_lock:
                if boxes:
                    self.bb_box_list      = boxes
                    self._bb_last_updated = now_ts
                elif (now_ts - self._bb_last_updated) > BOX_PERSIST_S:
                    self.bb_box_list = []

            # ── Publish detection results ─────────────────────────────────────
            # Format: "label:depth_m" per detection, comma-separated
            # e.g. "plastic_bottle:0.23,aluminum_can:0.41"
            # Detections with no depth emit "label:none"
            if boxes:
                parts = []
                for (_, _, _, _, label, _, d_val) in boxes:
                    depth_str = f"{d_val:.3f}" if d_val is not None else "none"
                    parts.append(f"{label}:{depth_str}")
                msg = String()
                msg.data = ",".join(parts)
                self.det_pub.publish(msg)


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
            half  = frame.shape[0] // 2

            # Draw on rectified left so bbox coords (computed in rect space) align
            with node.rect_lock:
                left       = node.latest_left_rect
                right_disp = node.latest_right_rect
            if left is None:
                left = frame[:half, :].copy()   # fallback before first rectification
            else:
                left = left.copy()
            if right_disp is None:
                right_disp = frame[half:, :]

            with node.bb_lock:
                boxes = list(node.bb_box_list)

            # Pick closest detection for trigger highlight
            closest = None
            if boxes:
                valid_depth = [(i, b[6]) for i, b in enumerate(boxes)
                               if b[6] is not None]
                if valid_depth:
                    closest = min(valid_depth, key=lambda x: x[1])[0]

            for i, (x1, y1, x2, y2, label, conf, d_val) in enumerate(boxes):
                is_closest = (i == closest)
                in_range   = d_val is not None and d_val < DEPTH_THRESH

                if in_range:
                    color     = (0, 0, 255)    # red — within trigger range
                    thickness = 3
                elif is_closest:  # closest but outside threshold
                    color     = (0, 215, 255)  # yellow — closest but not in range
                    thickness = 2
                else:
                    color     = (0, 255, 0)    # green — other detections
                    thickness = 2

                # Mark stale cached depth with asterisk
                depth_str = (f"{d_val:.2f}m" if d_val is not None else "no depth")

                cv2.rectangle(left, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(left, f"{label} {conf:.2f} {depth_str}",
                            (x1, max(y1 - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(left, ((x1+x2)//2, (y1+y2)//2), 4, (0, 200, 255), -1)

            combined = cv2.hconcat([left, right_disp])
            combined = cv2.resize(combined, (1280, 360))
            for y in range(0, combined.shape[0], 40):
                cv2.line(combined, (0, y), (combined.shape[1], y), (0, 0, 255), 1)

            with node.depth_lock:
                depth = node.latest_depth

            if depth is not None:
                d_vis = np.nan_to_num(depth, nan=0.0)

                # Adaptive range — stretch colourmap to actual scene depth
                valid_vals = d_vis[d_vis > 0.05]
                if len(valid_vals) > 100:
                    lo = float(np.percentile(valid_vals, 5))
                    hi = float(np.percentile(valid_vals, 95))
                else:
                    lo, hi = 0.1, 3.0

                d_norm   = np.clip((d_vis - lo) / max(hi - lo, 0.01), 0, 1)
                d_scaled = (d_norm * 255).astype(np.uint8)

                # Inpaint invalid holes — larger radius fills bigger voids
                invalid_mask = (d_vis <= 0.05).astype(np.uint8)
                if invalid_mask.any():
                    d_scaled = cv2.inpaint(d_scaled, invalid_mask, 12, cv2.INPAINT_NS)

                # Bilateral filter — smooths flat regions while keeping depth edges sharp
                d_smooth = cv2.bilateralFilter(d_scaled, d=7, sigmaColor=18, sigmaSpace=18)
                d_color  = cv2.applyColorMap(d_smooth, cv2.COLORMAP_TURBO)
                d_color = cv2.resize(d_color, (1280, 360))

                # ── Gradient scale bar (bottom 20px strip) ────────────────────
                bar_h    = 20
                gradient = np.linspace(0, 255, 1280, dtype=np.uint8)
                bar_raw  = np.tile(gradient, (bar_h, 1))
                bar_col  = cv2.applyColorMap(bar_raw, cv2.COLORMAP_TURBO)
                mid = (lo + hi) / 2
                for val, x in [(lo, 10), (mid, 610), (hi, 1210)]:
                    cv2.putText(bar_col, f"{val:.2f}m", (x, bar_h - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                                cv2.LINE_AA)
                cv2.putText(d_color, "DEPTH MAP", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                            cv2.LINE_AA)
                depth_display = np.vstack([d_color, bar_col])
                cv2.imshow("Depth Map", depth_display)

                display = combined
            else:
                display = combined

            cv2.putText(display, "LEFT",  (10,  20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "RIGHT", (650, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Stereo + Depth + YOLO v3", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()