#!/usr/bin/env python3
"""
CargaBot Vision App v7 — Alignment + Ignore brush + Cube-aware planning
========================================================================
Changes vs v6:

  ── MULTI-POINT FRONT-LINE ALIGNMENT ──
  - Goal-reached for targets now checks 3 points along Claudio's front edge:
    center, left-quarter, right-quarter.
  - ALL 3 must be within FRONT_LINE_REACH_CM of the target box edges.
  - This ensures the robot is actually aligned with a face, not just
    touching a corner.

  ── IGNORE BRUSH ──
  - Middle-click (or E + left-click) places a circular ignore zone on the
    overhead image. Anything inside is masked out of ALL semantic layers
    (obstacles, cubes, zones).
  - 'X' key clears all ignore zones.
  - Useful for false-positive obstacles or phantom detections.
  - Zones persist across frames, stored as list of (cx_cm, cy_cm, r_cm).

  ── TWO-PASS A* (CUBE-AWARE PLANNING) ──
  - Pass 1: Plan with cubes added to the costmap as obstacles.
    Blocks are 3D — they are real physical obstacles even though
    perception doesn't classify them as "black obstacles".
  - Pass 2 (fallback): If pass 1 finds no route, plan with the
    original costmap (cubes ignored). This handles cases where the
    only path goes through a cube field.
  - The SELECTED target's cube is always excluded from the costmap
    (you need to reach it, not avoid it).

Usage:
  uv run vision_cenital/cargabot_vision_app.py \
    --params resource/camera_params.yaml \
    --homography resource/homography_retry.yaml \
    --ros
"""

# ── CRITICO: forzar X11 ANTES de que Qt se inicialice ──
import os
os.environ['WAYLAND_DISPLAY'] = ''
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['DISPLAY']         = os.environ.get('DISPLAY', ':0')

import sys
import json
import math
import time
import argparse
import threading
from collections import deque
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vision_cenital.perception import OverheadPerception, ARUCO_REAL_CM
from vision_cenital.planning   import GridNavigator

ROS_AVAILABLE = False
try:
    import rclpy
    from rclpy.node         import Node
    from geometry_msgs.msg  import PoseStamped
    from nav_msgs.msg       import Path, Odometry
    from sensor_msgs.msg    import Image as ROSImage
    from std_msgs.msg       import String
    from cv_bridge          import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    pass


# ── Paleta ────────────────────────────────────────────────────────────────────
_W  = (255, 255, 255)
_BK = (  0,   0,   0)
_YL = (  0, 220, 255)
_CY = (255, 210,   0)
_GR = ( 50, 220,  50)
_RD = ( 50,  50, 220)
_OR = (  0, 160, 255)
_MG = (220,   0, 220)
_GY = (140, 140, 140)
_DG = ( 40,  40,  40)
_TL = (180, 200,   0)

LAYER_PALETTE = {
    'red':   (( 50,  50, 220), 'ROJO'),
    'green': (( 50, 220,  50), 'VERDE'),
    'blue':  ((220,  80,  50), 'AZUL'),
}

def _layer_style(name: str):
    for k, (color, label) in LAYER_PALETTE.items():
        if k in name:
            return color, label
    return _GY, name.upper()


# ── Helpers de dibujo (optimizados — from v6) ────────────────────────────────

_FONT = cv2.FONT_HERSHEY_SIMPLEX

def _txt(img, msg, pos, scale=0.55, color=_W, thick=1, shadow=True):
    if shadow:
        cv2.putText(img, msg, pos, _FONT, scale, _BK, thick+2, cv2.LINE_AA)
    cv2.putText(img, msg, pos, _FONT, scale, color, thick, cv2.LINE_AA)

def _txt_fast(img, msg, pos, scale=0.50, color=_W, thick=1):
    cv2.putText(img, msg, pos, _FONT, scale, color, thick, cv2.LINE_4)

def _dashed(img, p1, p2, color, thick=1, dash=12, gap=6):
    dx = p2[0]-p1[0]; dy = p2[1]-p1[1]
    d = max(1, int(math.hypot(dx, dy)))
    inv_d = 1.0 / d
    step = dash + gap
    for i in range(0, d, step):
        t0 = i * inv_d; t1 = min((i+dash) * inv_d, 1.0)
        a = (int(p1[0]+dx*t0), int(p1[1]+dy*t0))
        b = (int(p1[0]+dx*t1), int(p1[1]+dy*t1))
        cv2.line(img, a, b, color, thick)

def _tint_mask(img, mask_bool, color_bgr, alpha_int=102):
    if not np.any(mask_bool): return
    inv = 256 - alpha_int
    pix = img[mask_bool].astype(np.uint16)
    pix[:, 0] = (pix[:, 0] * inv + color_bgr[0] * alpha_int) >> 8
    pix[:, 1] = (pix[:, 1] * inv + color_bgr[1] * alpha_int) >> 8
    pix[:, 2] = (pix[:, 2] * inv + color_bgr[2] * alpha_int) >> 8
    img[mask_bool] = pix.astype(np.uint8)


# ── Target ────────────────────────────────────────────────────────────────────

class Target:
    __slots__ = ('id', 'kind', 'color_name', 'color_bgr',
                 'center', 'size', 'angle_rad', 'corners',
                 'approach_pose')
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k, None))

    def label(self) -> str:
        pfx = "DROP-" if self.kind == 'zone' else ""
        return f"{pfx}{self.color_name} #{self.id}"


# ── ROS2 Bridge (unchanged from v6) ──────────────────────────────────────────

class ROSBridge:
    def __init__(self):
        rclpy.init()
        self.node         = rclpy.create_node('cargabot_vision_app')
        self.bridge       = CvBridge()
        self.latest_frame = None
        self.odom_pose:   Optional[Tuple] = None
        self.ext_goal:    Optional[Tuple] = None
        self._lock        = threading.Lock()
        self._mc_busy     = False
        self._mc_lock     = threading.Lock()
        self._last_mc_response: Optional[dict] = None

        self.pub_path  = self.node.create_publisher(Path,     '/cargabot/global_path',   10)
        self.pub_exec  = self.node.create_publisher(String,   '/cargabot/execute_req',   10)
        self.pub_debug = self.node.create_publisher(ROSImage, '/cargabot/overhead_debug', 4)

        self.node.create_subscription(Odometry,    '/odom',                      self._odom_cb,        10)
        self.node.create_subscription(PoseStamped, '/cargabot/goal_pose',        self._goal_cb,        10)
        self.node.create_subscription(ROSImage,    '/cargabot/camera/image_raw', self._cam_cb,         4)
        self.node.create_subscription(String,      '/cargabot/execute_res',      self._mc_response_cb, 10)

        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):              rclpy.spin(self.node)
    def _odom_cb(self, msg):
        p = msg.pose.pose
        with self._lock:
            self.odom_pose = (p.position.x, p.position.y,
                              2.0 * math.atan2(p.orientation.z, p.orientation.w))
    def _goal_cb(self, msg):
        with self._lock:
            self.ext_goal = (msg.pose.position.x * 100.0, msg.pose.position.y * 100.0)
    def _cam_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with self._lock: self.latest_frame = frame
    def _mc_response_cb(self, msg):
        try:    resp = json.loads(msg.data)
        except: resp = {"success": False, "message": "bad JSON"}
        with self._mc_lock:
            self._mc_busy = False; self._last_mc_response = resp
    def get_frame(self):
        with self._lock: return self.latest_frame
    def get_odom(self):
        with self._lock: return self.odom_pose
    def get_ext_goal(self):
        with self._lock:
            g = self.ext_goal; self.ext_goal = None; return g
    def is_mc_busy(self) -> bool:
        with self._mc_lock: return self._mc_busy
    def send_mc_command(self, cmd: dict) -> bool:
        with self._mc_lock:
            if self._mc_busy: return False
            self._mc_busy = True
        msg = String(); msg.data = json.dumps(cmd)
        self.pub_exec.publish(msg); return True
    def send_stop(self):
        msg = String(); msg.data = '{"cmd":"stop"}'
        self.pub_exec.publish(msg)
        with self._mc_lock: self._mc_busy = False
    def publish_path(self, path_cm):
        msg = Path(); msg.header.frame_id = 'map'
        msg.header.stamp = self.node.get_clock().now().to_msg()
        for x, y in path_cm:
            p = PoseStamped(); p.header = msg.header
            p.pose.position.x = x / 100.0; p.pose.position.y = y / 100.0
            msg.poses.append(p)
        self.pub_path.publish(msg)
    def publish_debug(self, frame):
        try: self.pub_debug.publish(self.bridge.cv2_to_imgmsg(frame, encoding='bgr8'))
        except: pass
    def shutdown(self):
        self.send_stop(); rclpy.shutdown()


# ── App ───────────────────────────────────────────────────────────────────────

class CargaBotVisionApp:

    # ── Claudio geometry ──────────────────────────────────────────────────────
    CLAUDIO_HALF_LENGTH_CM      = 21.5
    CLAUDIO_HALF_WIDTH_CM       = 12.0   # NEW: for multi-point front-line
    CLAUDIO_EXCLUSION_DILATE_CM = 8.0

    # ── Approach ──────────────────────────────────────────────────────────────
    APPROACH_MARGIN_CM     = 5.0
    FRONT_LINE_REACH_CM    = 6.0
    FINAL_APPROACH_CM      = 35.0
    FINAL_MOVE_CAP_CM      = 8.0
    FINAL_ROT_TOL_DEG      = 20.0
    DECEL_FACTOR           = 0.45

    # ── Tracking ──────────────────────────────────────────────────────────────
    LOOKAHEAD_CM        = 22.0
    DEVIATION_THRESHOLD = 15.0
    TRAIL_MAX           = 300
    MIN_CUBE_PX         = 100
    REPLAN_COOLDOWN     = 2.0
    TARGET_TRACK_DIST_CM = 25.0
    GOAL_REACHED_CM      = 8.0

    # ── Motion ────────────────────────────────────────────────────────────────
    ROTATE_THRESHOLD_DEG = 8.0
    MOVE_THRESHOLD_CM    = 3.0

    # ── Ignore brush ──────────────────────────────────────────────────────────
    IGNORE_RADIUS_CM     = 12.0   # radius of each ignore zone

    # ── Performance intervals ─────────────────────────────────────────────────
    LAYER_INTERVAL     = 3
    DEBUG_PUB_INTERVAL = 2
    _EXCL_RECOMPUTE_CM = 2.0

    # ── Cube costmap dilation ─────────────────────────────────────────────────
    CUBE_DILATION_CM   = 5.0    # extra dilation for cube obstacles in pass-1

    def __init__(self, args):
        self.args = args
        self.perception = OverheadPerception(
            args.params, args.homography,
            sim_mode=getattr(args, 'sim', False))
        self.navigator = GridNavigator(
            self.perception.pista_w_cm,
            self.perception.pista_h_cm,
            args.grid_res, args.robot_radius)

        self.ros: Optional[ROSBridge] = None
        if getattr(args, 'ros', False) and ROS_AVAILABLE:
            self.ros = ROSBridge()
        elif getattr(args, 'ros', False):
            print("[WARN] --ros pero rclpy no disponible")

        # ── State ─────────────────────────────────────────────────────────────
        self.start_cm:     Optional[Tuple] = None
        self.goal_cm:      Optional[Tuple] = None
        self.goal_heading: Optional[float] = None
        self.active_path:  List[Tuple]     = []
        self._path_arr:    Optional[np.ndarray] = None
        self._cost_ready   = False
        self.robot_trail   = deque(maxlen=self.TRAIL_MAX)
        self.deviation_cm  = 0.0
        self.progress      = 0.0
        self.replan_count  = 0
        self._last_replan  = 0.0
        self._fps_buf      = deque(maxlen=30)
        self._debug_mode   = False
        self._heading_debug = False

        # ── Targets + locked approach ─────────────────────────────────────────
        self.targets: List[Target] = []
        self.selected_target_id: Optional[int] = None
        self._locked_approach: Optional[Tuple[float, float, float]] = None
        self._locked_target_center: Optional[Tuple[float, float]] = None
        self._in_final_approach = False

        # ── Ignore zones (NEW v7) ─────────────────────────────────────────────
        self._ignore_zones: List[Tuple[float, float, float]] = []  # (cx_cm, cy_cm, r_cm)
        self._ignore_mode = False  # toggled with E key

        # ── Layers cache ──────────────────────────────────────────────────────
        self._layers_cache: Dict = {}
        self._frame_count = 0

        # ── Cube obstacle mask for two-pass planning (NEW v7) ─────────────────
        self._cube_obstacle_mask: Optional[np.ndarray] = None

        # ── Exclusion cache ───────────────────────────────────────────────────
        self._excl_mask: Optional[np.ndarray] = None
        self._excl_pose: Optional[Tuple[float, float]] = None

        # ── Grid overlay ──────────────────────────────────────────────────────
        self._grid_overlay: Optional[np.ndarray] = None
        self._grid_shape:   Optional[Tuple]       = None

        # ── Obstacle mask resized cache ───────────────────────────────────────
        self._obs_resized:  Optional[np.ndarray] = None
        self._wall_resized: Optional[np.ndarray] = None

        self.WIN = "CargaBot Vision"
        self._init_window()
        self._print_banner()

    # ── Window init ───────────────────────────────────────────────────────────

    def _init_window(self):
        _d = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(_d, "Iniciando...", (20, 240), _FONT, 1.0, _YL, 2)
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.imshow(self.WIN, _d)
        for _ in range(15): cv2.waitKey(40)
        try:
            cv2.setMouseCallback(self.WIN, self._mouse)
            cv2.resizeWindow(self.WIN, 1400, 800)
            return
        except cv2.error: pass
        cv2.destroyAllWindows()
        for _ in range(5): cv2.waitKey(40)
        cv2.namedWindow(self.WIN, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.WIN, _d)
        for _ in range(15): cv2.waitKey(40)
        cv2.setMouseCallback(self.WIN, self._mouse)

    def _print_banner(self):
        r = "ROS2 ON" if self.ros else "STANDALONE"
        print(f"\n  CargaBot v7 [{r}]  |  1-9:target  R:clr  P:plan  D:dbg  S:stop  E:ignore  X:clr-ignore\n")

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def _mouse(self, ev, x, y, flags, param):
        ppc = max(self.perception.px_per_cm, 0.01)
        pt_cm = (x / ppc, y / ppc)

        if ev == cv2.EVENT_RBUTTONDOWN:
            if self._ignore_mode:
                # Right-click in ignore mode: place ignore zone
                self._ignore_zones.append((pt_cm[0], pt_cm[1], self.IGNORE_RADIUS_CM))
                print(f"[IGN] Added ignore zone at ({pt_cm[0]:.1f},{pt_cm[1]:.1f}) r={self.IGNORE_RADIUS_CM}cm "
                      f"({len(self._ignore_zones)} total)")
                return
            # Normal right-click: set goal
            self.goal_cm = pt_cm
            self.goal_heading = None
            self.selected_target_id = None
            self._locked_approach = None
            self._in_final_approach = False
            self._plan()

        elif ev == cv2.EVENT_MBUTTONDOWN:
            # Middle-click: always place ignore zone
            self._ignore_zones.append((pt_cm[0], pt_cm[1], self.IGNORE_RADIUS_CM))
            print(f"[IGN] Added ignore zone at ({pt_cm[0]:.1f},{pt_cm[1]:.1f}) r={self.IGNORE_RADIUS_CM}cm "
                  f"({len(self._ignore_zones)} total)")

    # ── Ignore mask ───────────────────────────────────────────────────────────

    def _build_ignore_mask(self, shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Build a mask from all ignore zones. Returns None if no zones."""
        if not self._ignore_zones:
            return None
        ppc = max(self.perception.px_per_cm, 0.01)
        mask = np.zeros(shape[:2], dtype=np.uint8)
        for cx, cy, r in self._ignore_zones:
            cv2.circle(mask, (int(cx * ppc), int(cy * ppc)), int(r * ppc), 255, -1)
        return mask

    def _apply_ignore(self, layers: Dict, img_shape: Tuple[int, int]):
        """Zero out all layer pixels inside ignore zones."""
        ign = self._build_ignore_mask(img_shape)
        if ign is None:
            return
        for name, mask in list(layers.items()):
            if mask is None or 'raw' in name:
                continue
            m = mask if mask.dtype == np.uint8 else mask.astype(np.uint8)
            if ign.shape != m.shape[:2]:
                ig = cv2.resize(ign, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                ig = ign
            m[ig > 0] = 0
            layers[name] = m

    # ── Exclusion (cached — from v6) ──────────────────────────────────────────

    def _get_exclusion_mask(self, img_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        corners = self.perception.get_robot_hitbox_corners_px()
        if corners is None: return self._excl_mask
        pose = self.perception.robot_pose_cm
        if pose is not None and self._excl_pose is not None:
            d = math.hypot(pose[0] - self._excl_pose[0], pose[1] - self._excl_pose[1])
            if d < self._EXCL_RECOMPUTE_CM and self._excl_mask is not None:
                if self._excl_mask.shape == img_shape: return self._excl_mask
        ppc = max(self.perception.px_per_cm, 0.01)
        dpx = max(1, int(self.CLAUDIO_EXCLUSION_DILATE_CM * ppc))
        exc = np.zeros(img_shape, dtype=np.uint8)
        cv2.fillPoly(exc, [corners.astype(np.int32)], 255)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dpx*2+1, dpx*2+1))
        exc = cv2.dilate(exc, k)
        self._excl_mask = exc
        if pose: self._excl_pose = (pose[0], pose[1])
        return exc

    def _apply_exclusion(self, layers: Dict, img_shape: Tuple[int, int]):
        exc = self._get_exclusion_mask(img_shape)
        if exc is None: return
        for name, mask in list(layers.items()):
            if mask is None or 'raw' in name: continue
            m = mask if mask.dtype == np.uint8 else mask.astype(np.uint8)
            if exc.shape != m.shape[:2]:
                e = cv2.resize(exc, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
            else: e = exc
            m[e > 0] = 0
            layers[name] = m

    # ── Walls + cubes -> obstacles ────────────────────────────────────────────

    @staticmethod
    def _merge_walls(layers: Dict) -> Optional[np.ndarray]:
        obs   = layers.get('obstacles')
        walls = layers.get('walls')
        if obs is None and walls is None: return None
        if walls is None: return obs if obs.dtype == np.uint8 else obs.astype(np.uint8)
        w8 = walls if walls.dtype == np.uint8 else walls.astype(np.uint8)
        if obs is None:
            layers['obstacles'] = w8.copy(); return layers['obstacles']
        o8 = obs if obs.dtype == np.uint8 else obs.astype(np.uint8)
        if w8.shape != o8.shape:
            w8 = cv2.resize(w8, (o8.shape[1], o8.shape[0]), interpolation=cv2.INTER_NEAREST)
        merged = cv2.bitwise_or(o8, w8)
        layers['obstacles'] = merged
        return merged

    def _build_cube_obstacle_mask(self, layers: Dict, img_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Build a mask of all cube 'solid' detections (3D obstacles).
        Excludes the currently selected target's cube.
        Used for two-pass A* planning.
        """
        ppc = max(self.perception.px_per_cm, 0.01)
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        has_any = False

        for name, layer_mask in layers.items():
            if 'solid' not in name or layer_mask is None:
                continue
            m = layer_mask if layer_mask.dtype == np.uint8 else layer_mask.astype(np.uint8)
            if m.shape[:2] != img_shape[:2]:
                m = cv2.resize(m, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = cv2.bitwise_or(mask, m)
            if np.any(m > 0):
                has_any = True

        if not has_any:
            self._cube_obstacle_mask = None
            return None

        # Exclude the selected target's region so we can actually reach it
        sel = self._selected_target()
        if sel is not None and sel.kind == 'cube' and sel.corners is not None:
            pts = np.array([[int(c[0]*ppc), int(c[1]*ppc)] for c in sel.corners], np.int32)
            # Dilate the exclusion slightly to ensure the approach path is clear
            excl = np.zeros_like(mask)
            cv2.fillPoly(excl, [pts], 255)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            excl = cv2.dilate(excl, k)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(excl))

        # Dilate cube obstacles by robot radius + margin
        dil_px = max(1, int(self.CUBE_DILATION_CM * ppc))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_px*2+1, dil_px*2+1))
        mask = cv2.dilate(mask, k)

        self._cube_obstacle_mask = mask
        return mask

    # ── Target detection (from v6) ────────────────────────────────────────────

    def _detect_targets(self, layers: Dict) -> List[Target]:
        ppc = max(self.perception.px_per_cm, 0.01)
        inv_ppc = 1.0 / ppc
        new: List[Target] = []
        for name, mask in layers.items():
            if mask is None or 'raw' in name: continue
            if 'solid' in name:   kind = 'cube'
            elif 'zone' in name:  kind = 'zone'
            else: continue
            color_bgr, color_label = _layer_style(name)
            m = mask if mask.dtype == np.uint8 else mask.astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            for cnt in cnts:
                a = cv2.contourArea(cnt)
                if a < self.MIN_CUBE_PX: continue
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (w, h), ang = rect
                if w < 5 or h < 5: continue
                box = cv2.boxPoints(rect)
                corners_cm = [(float(p[0]*inv_ppc), float(p[1]*inv_ppc)) for p in box]
                new.append(Target(
                    id=None, kind=kind, color_name=color_label, color_bgr=color_bgr,
                    center=(cx*inv_ppc, cy*inv_ppc), size=(w*inv_ppc, h*inv_ppc),
                    angle_rad=math.radians(ang), corners=corners_cm))
        self._assign_ids(new)
        return new

    def _assign_ids(self, new: List[Target]):
        prev = {t.id: t for t in self.targets if t.id is not None}
        used = set(); unmatched = []
        for nt in new:
            bid = None; bd = self.TARGET_TRACK_DIST_CM
            for pid, pt in prev.items():
                if pid in used or pt.kind != nt.kind: continue
                d = math.hypot(nt.center[0]-pt.center[0], nt.center[1]-pt.center[1])
                if d < bd: bd = d; bid = pid
            if bid is not None: nt.id = bid; used.add(bid)
            else: unmatched.append(nt)
        nxt = 1
        for nt in unmatched:
            while nxt in used: nxt += 1
            nt.id = nxt; used.add(nxt)

    # ── Approach (from v6) ────────────────────────────────────────────────────

    def _compute_approach_pose(self, target: Target,
                                claudio_pos: Tuple[float, float]
                                ) -> Optional[Tuple[float, float, float]]:
        cx, cy = target.center
        corners = target.corners
        standoff = self.CLAUDIO_HALF_LENGTH_CM + self.APPROACH_MARGIN_CM
        best = None; best_s = float('-inf')
        for i in range(4):
            p1 = corners[i]; p2 = corners[(i+1) % 4]
            mx = (p1[0]+p2[0])*0.5; my = (p1[1]+p2[1])*0.5
            ex = p2[0]-p1[0]; ey = p2[1]-p1[1]
            el = math.hypot(ex, ey)
            if el < 1e-6: continue
            nx, ny = -ey, ex
            if (mx-cx)*nx + (my-cy)*ny < 0: nx, ny = -nx, -ny
            nl = math.hypot(nx, ny); nx /= nl; ny /= nl
            ax = mx + nx*standoff; ay = my + ny*standoff
            ah = math.atan2(-ny, -nx)
            s = -math.hypot(ax-claudio_pos[0], ay-claudio_pos[1])
            if not (0 <= ax <= self.perception.pista_w_cm and
                    0 <= ay <= self.perception.pista_h_cm): s -= 1e6
            if self._cost_ready:
                gx, gy = self.navigator.cm_to_grid(ax, ay)
                if 0 <= gx < self.navigator.cols and 0 <= gy < self.navigator.rows:
                    if self.navigator.cost_map[gy, gx] != 0: s -= 200
            if s > best_s: best_s = s; best = (ax, ay, ah)
        return best

    def _select_target(self, tid: int):
        target = next((t for t in self.targets if t.id == tid), None)
        if target is None: print(f"[T] #{tid} no existe"); return
        pose = self.perception.robot_pose_cm
        if pose is None: print("[T] Robot no detectado"); return
        ap = self._compute_approach_pose(target, pose[:2])
        if ap is None: print(f"[T] #{tid} sin approach valido"); return
        self._locked_approach = ap
        self._locked_target_center = target.center
        self.selected_target_id = tid
        self.goal_cm = (ap[0], ap[1])
        self.goal_heading = ap[2]
        self._in_final_approach = False
        target.approach_pose = ap
        print(f"[T] #{tid} LOCKED approach=({ap[0]:.1f},{ap[1]:.1f}) hdg={math.degrees(ap[2]):.1f}")
        self._plan()

    def _selected_target(self) -> Optional[Target]:
        if self.selected_target_id is None: return None
        return next((t for t in self.targets if t.id == self.selected_target_id), None)

    # ── Multi-point front-line (NEW v7) ───────────────────────────────────────

    def _front_line_points(self, pose) -> List[Tuple[float, float]]:
        """
        Returns 3 points along Claudio's front edge:
          center, left-quarter, right-quarter.
        """
        rx, ry, yaw = pose
        cos_y = math.cos(yaw); sin_y = math.sin(yaw)
        # Forward vector (along heading)
        fx = self.CLAUDIO_HALF_LENGTH_CM * cos_y
        fy = self.CLAUDIO_HALF_LENGTH_CM * sin_y
        # Lateral vector (perpendicular, half-width * 0.5 for quarter points)
        quarter_w = self.CLAUDIO_HALF_WIDTH_CM * 0.5
        lx = -quarter_w * sin_y
        ly =  quarter_w * cos_y

        center = (rx + fx, ry + fy)
        left   = (rx + fx + lx, ry + fy + ly)
        right  = (rx + fx - lx, ry + fy - ly)
        return [center, left, right]

    def _front_line_midpoint(self, pose):
        """Legacy: single center point for compatibility."""
        return self._front_line_points(pose)[0]

    @staticmethod
    def _dist_pt_seg(px, py, x1, y1, x2, y2) -> float:
        dx = x2-x1; dy = y2-y1
        L2 = dx*dx + dy*dy
        if L2 < 1e-9: return math.hypot(px-x1, py-y1)
        t = max(0.0, min(1.0, ((px-x1)*dx + (py-y1)*dy) / L2))
        return math.hypot(px - (x1+t*dx), py - (y1+t*dy))

    def _dist_to_box(self, pt, target) -> float:
        px, py = pt; c = target.corners; best = 1e9
        for i in range(4):
            d = self._dist_pt_seg(px, py, c[i][0], c[i][1],
                                  c[(i+1)%4][0], c[(i+1)%4][1])
            if d < best: best = d
        return best

    def _check_front_line_aligned(self, pose, target_or_approach) -> Tuple[bool, float]:
        pts = self._front_line_points(pose)
        sel = self._selected_target()

        distances = []
        for pt in pts:
            if sel is not None:
                d = self._dist_to_box(pt, sel)
            else:
                ax, ay = target_or_approach[:2]
                d = math.hypot(pt[0] - ax, pt[1] - ay) - self.CLAUDIO_HALF_LENGTH_CM
            distances.append(d)

        max_d = max(distances)
        all_aligned = all(d < self.FRONT_LINE_REACH_CM for d in distances)
        return all_aligned, max_d

    # ── Planning (TWO-PASS A* — NEW v7) ───────────────────────────────────────

    def _find_free(self, x, y, cost_map=None):
        """Find nearest free cell. Uses provided cost_map or navigator's default."""
        cm = cost_map if cost_map is not None else self.navigator.cost_map
        gx, gy = self.navigator.cm_to_grid(x, y)
        if cm[gy, gx] == 0: return (x, y)
        for r in range(1, 15):
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    if abs(dx) != r and abs(dy) != r: continue
                    nx, ny = gx+dx, gy+dy
                    if not (0 <= nx < self.navigator.cols and 0 <= ny < self.navigator.rows): continue
                    if cm[ny, nx] == 0:
                        return self.navigator.grid_to_cm(nx, ny)
        return None

    def _plan(self):
        pose = self.perception.robot_pose_cm
        if not (pose and self.goal_cm and self._cost_ready): return

        # ── Pass 1: Try planning with cubes as obstacles ──────────────────────
        path = self._plan_with_cubes(pose)

        # ── Pass 2: Fallback to normal costmap ────────────────────────────────
        if not path:
            path = self._plan_normal(pose)

        if path:
            self.active_path = path
            self._path_arr = np.array(path, dtype=np.float64)
            self.progress = 0.0; self.replan_count = 0
            if self.ros: self.ros.publish_path(path)
        else:
            print(f"[A*] sin ruta (ambos passes fallaron)")

    def _plan_with_cubes(self, pose) -> List[Tuple[float, float]]:
        """Pass 1: costmap with cube solids merged in as obstacles."""
        if self._cube_obstacle_mask is None:
            return []  # no cubes detected, skip to pass 2

        # Merge cube mask into base obstacle mask
        base_obs = self._layers_cache.get('obstacles')
        if base_obs is None:
            return []

        base = base_obs if base_obs.dtype == np.uint8 else base_obs.astype(np.uint8)
        cube_m = self._cube_obstacle_mask
        if cube_m.shape != base.shape:
            cube_m = cv2.resize(cube_m, (base.shape[1], base.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        merged = cv2.bitwise_or(base, cube_m)

        # Generate temporary costmap
        old_cost = self.navigator.cost_map.copy()
        self.navigator.generate_cost_map(merged)

        start = self._find_free(pose[0], pose[1])
        goal = self._find_free(self.goal_cm[0], self.goal_cm[1])
        path = []
        if start and goal:
            path = self.navigator.astar_search(start, goal)
            if path:
                self.start_cm = start
                print(f"[A*] Pass 1 (cube-aware): {len(path)} nodos")

        # Restore original costmap (pass 2 or tracking needs it)
        self.navigator.cost_map = old_cost
        # Regenerate from base obstacles so tracking uses the normal map
        self.navigator.generate_cost_map(base_obs)

        return path

    def _plan_normal(self, pose) -> List[Tuple[float, float]]:
        """Pass 2: normal costmap (cubes ignored)."""
        start = self._find_free(pose[0], pose[1])
        if start is None: return []
        self.start_cm = start
        goal = self._find_free(self.goal_cm[0], self.goal_cm[1])
        if goal is None: return []
        path = self.navigator.astar_search(start, goal)
        if path:
            print(f"[A*] Pass 2 (normal): {len(path)} nodos")
        else:
            print(f"[A*] sin ruta start=({start[0]:.0f},{start[1]:.0f}) "
                  f"goal=({goal[0]:.0f},{goal[1]:.0f})")
        return path

    def _auto_replan(self, rx, ry):
        if self._in_final_approach: return
        now = time.time()
        if now - self._last_replan < self.REPLAN_COOLDOWN: return
        if not self.goal_cm: return
        start = self._find_free(rx, ry)
        if start is None: return
        goal = self._find_free(self.goal_cm[0], self.goal_cm[1])
        if goal is None: return
        path = self.navigator.astar_search(start, goal)
        if path:
            self.active_path = path
            self._path_arr = np.array(path, dtype=np.float64)
            self.start_cm = start
            self.replan_count += 1
            self._last_replan = now
            if self.ros: self.ros.publish_path(path)

    # ── Tracking + motion (from v6, with multi-point alignment) ───────────────

    def _update_tracking(self) -> Optional[Tuple]:
        pose = self.perception.robot_pose_cm
        pa = self._path_arr
        if pose is None or pa is None or len(pa) < 2: return None
        rx, ry, yaw = pose

        # ── Multi-point front-line check (NEW v7) ────────────────────────────
        if self._locked_approach is not None:
            sel = self._selected_target()
            if sel is not None:
                aligned, max_d = self._check_front_line_aligned(pose, sel)
            else:
                aligned, max_d = self._check_front_line_aligned(pose, self._locked_approach)

            if aligned:
                print(f">>> ALIGNED — all 3 front points within {self.FRONT_LINE_REACH_CM}cm "
                      f"(max_d={max_d:.1f}cm) <<<")
                self._clear_nav()
                if self.ros: self.ros.send_stop()
                return None

            # Final approach check
            d_to_ap = math.hypot(rx - self._locked_approach[0],
                                 ry - self._locked_approach[1])
            if d_to_ap < self.FINAL_APPROACH_CM and not self._in_final_approach:
                self._in_final_approach = True
                print(f"[FA] Entrando final approach d={d_to_ap:.1f}cm")

        # ── Vectorized closest point ──────────────────────────────────────────
        dists = np.hypot(pa[:, 0] - rx, pa[:, 1] - ry)
        idx = int(np.argmin(dists))
        self.deviation_cm = float(dists[idx])
        self.progress = idx / max(len(pa) - 1, 1)

        # ── Euclidean fallback (right-click) ──────────────────────────────────
        if self._locked_approach is None:
            gx, gy = pa[-1]
            if math.hypot(rx - gx, ry - gy) < self.GOAL_REACHED_CM:
                self._clear_nav()
                if self.ros: self.ros.send_stop()
                return None

        # ── Replan si desviado ────────────────────────────────────────────────
        if self.deviation_cm > self.DEVIATION_THRESHOLD:
            self._auto_replan(rx, ry)
            pa = self._path_arr
            if pa is None or len(pa) < 2: return None
            dists = np.hypot(pa[:, 0] - rx, pa[:, 1] - ry)
            idx = int(np.argmin(dists))

        # ── Vectorized lookahead ──────────────────────────────────────────────
        tail = pa[idx:]
        if len(tail) < 2:
            lh = tuple(pa[-1])
        else:
            segs = np.diff(tail, axis=0)
            lens = np.hypot(segs[:, 0], segs[:, 1])
            cum = np.cumsum(lens)
            li = np.searchsorted(cum, self.LOOKAHEAD_CM)
            lh = tuple(pa[idx + li + 1]) if li < len(cum) else tuple(pa[-1])

        self._send_motion(lh, pose)
        return lh

    def _send_motion(self, target_cm, pose):
        if not self.ros or self.ros.is_mc_busy(): return
        rx, ry, yaw = pose
        dx = target_cm[0] - rx; dy = target_cm[1] - ry
        ta = math.atan2(dy, dx)
        delta = (ta - yaw + math.pi) % (2*math.pi) - math.pi
        delta_deg = math.degrees(delta)
        dist_cm = math.hypot(dx, dy)

        if self._in_final_approach:
            if abs(delta_deg) > self.FINAL_ROT_TOL_DEG:
                self.ros.send_mc_command({"cmd": "rotate", "angle": round(delta_deg, 1)})
                return
            if dist_cm > self.MOVE_THRESHOLD_CM:
                step = min(dist_cm * self.DECEL_FACTOR, self.FINAL_MOVE_CAP_CM)
                step = max(step, 2.0)
                self.ros.send_mc_command({"cmd": "move", "distance": round(step / 100.0, 4)})
            return

        if abs(delta_deg) > self.ROTATE_THRESHOLD_DEG:
            self.ros.send_mc_command({"cmd": "rotate", "angle": round(delta_deg, 1)})
            return
        if dist_cm > self.MOVE_THRESHOLD_CM:
            if self.goal_cm:
                d_goal = math.hypot(rx - self.goal_cm[0], ry - self.goal_cm[1])
                cap = max(d_goal * self.DECEL_FACTOR, 3.0) if d_goal < self.LOOKAHEAD_CM * 2 else self.LOOKAHEAD_CM
            else:
                cap = self.LOOKAHEAD_CM
            self.ros.send_mc_command({"cmd": "move", "distance": round(min(dist_cm, cap) / 100.0, 4)})

    def _clear_nav(self):
        self.active_path = []; self._path_arr = None
        self.selected_target_id = None
        self._locked_approach = None; self._locked_target_center = None
        self.goal_heading = None; self._in_final_approach = False

    # ── Frame pipeline ────────────────────────────────────────────────────────

    def _process_frame(self, raw: np.ndarray) -> np.ndarray:
        t0 = time.monotonic()
        self._fps_buf.append(t0)
        self._frame_count += 1

        warped = self.perception.warp_to_overhead(raw)
        pose = self.perception.detect_robot_pose(raw)
        if pose is None and self.ros:
            od = self.ros.get_odom()
            if od: self.perception.inject_pose_from_odom(*od)

        if self.ros:
            eg = self.ros.get_ext_goal()
            if eg:
                self.goal_cm = eg; self.goal_heading = None
                self.selected_target_id = None
                self._locked_approach = None
                self._in_final_approach = False
                self._plan()

        do_layers = (self._frame_count % self.LAYER_INTERVAL == 0) or not self._cost_ready

        if do_layers:
            layers = self.perception.extract_semantic_layers(warped)
            self._apply_exclusion(layers, warped.shape[:2])
            self._apply_ignore(layers, warped.shape[:2])
            self._layers_cache = layers
            obs = self._merge_walls(layers)
            if obs is not None:
                self.navigator.generate_cost_map(obs)
                self._cost_ready = True
            self.targets = self._detect_targets(layers)
            # Build cube obstacle mask for two-pass planning
            self._build_cube_obstacle_mask(layers, warped.shape[:2])

            sel = self._selected_target()
            if sel is not None and self._locked_approach is not None:
                sel.approach_pose = self._locked_approach
            self._cache_obs_masks(layers, warped.shape[:2])

        pose = self.perception.robot_pose_cm
        if pose: self.robot_trail.append(pose[:2])

        display = warped
        self._render(display)

        if self.ros and self._frame_count % self.DEBUG_PUB_INTERVAL == 0:
            self.ros.publish_debug(display)
        return display

    def _cache_obs_masks(self, layers, shape):
        h, w = shape
        obs = layers.get('obstacles')
        if obs is not None:
            m = obs if obs.dtype == np.uint8 else obs.astype(np.uint8)
            if m.shape[:2] != (h, w): m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            self._obs_resized = m
        walls = layers.get('walls')
        if walls is not None:
            m = walls if walls.dtype == np.uint8 else walls.astype(np.uint8)
            if m.shape[:2] != (h, w): m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            self._wall_resized = m

    # ── Render ────────────────────────────────────────────────────────────────

    def _render(self, img):
        ppc = max(self.perception.px_per_cm, 0.01)
        if self._debug_mode: self._draw_grid(img, ppc)
        self._draw_obs_fast(img)
        self._draw_ignore_zones(img, ppc)
        self._draw_targets_fast(img, ppc)
        lh = None
        if self.active_path:
            lh = self._update_tracking()
            self._draw_path_fast(img, ppc)
        self._draw_trail_fast(img, ppc)
        self._draw_robot_fast(img, ppc, lh)
        if self._debug_mode: self._draw_debug(img, ppc)
        if self._heading_debug: self._draw_heading(img, ppc)
        self._draw_panel_fast(img)

    def _draw_grid(self, img, ppc):
        h, w = img.shape[:2]
        if self._grid_shape != (h, w):
            ov = np.zeros((h, w, 3), dtype=np.uint8)
            step = max(1, int(5.0 * ppc))
            for x in range(0, w, step): cv2.line(ov, (x, 0), (x, h), (30,30,30), 1)
            for y in range(0, h, step): cv2.line(ov, (0, y), (w, y), (30,30,30), 1)
            self._grid_overlay = ov; self._grid_shape = (h, w)
        cv2.add(img, self._grid_overlay, img)

    def _draw_obs_fast(self, img):
        if self._obs_resized is not None:
            _tint_mask(img, self._obs_resized > 0, (0, 0, 180), 102)
        if self._wall_resized is not None:
            _tint_mask(img, self._wall_resized > 0, (0, 0, 180), 102)

    def _draw_ignore_zones(self, img, ppc):
        """Draw ignore zones as semi-transparent gray circles with X."""
        for cx, cy, r in self._ignore_zones:
            cpx = int(cx * ppc); cpy = int(cy * ppc); rpx = int(r * ppc)
            # Semi-transparent fill
            y1 = max(0, cpy - rpx); y2 = min(img.shape[0], cpy + rpx)
            x1 = max(0, cpx - rpx); x2 = min(img.shape[1], cpx + rpx)
            if x2 > x1 and y2 > y1:
                roi = img[y1:y2, x1:x2]
                m = np.zeros(roi.shape[:2], dtype=np.uint8)
                cv2.circle(m, (cpx - x1, cpy - y1), rpx, 255, -1)
                _tint_mask(roi, m > 0, (80, 80, 80), 140)
            cv2.circle(img, (cpx, cpy), rpx, _GY, 2)
            # X mark
            d = int(rpx * 0.5)
            cv2.line(img, (cpx-d, cpy-d), (cpx+d, cpy+d), _RD, 2)
            cv2.line(img, (cpx-d, cpy+d), (cpx+d, cpy-d), _RD, 2)
            _txt_fast(img, "IGN", (cpx - 12, cpy - rpx - 4), 0.35, _GY, 1)

    def _draw_targets_fast(self, img, ppc):
        for t in self.targets:
            cp = np.array([[int(c[0]*ppc), int(c[1]*ppc)] for c in t.corners], np.int32)
            is_sel = (t.id == self.selected_target_id)
            thick = 3 if is_sel else 2; col = t.color_bgr
            xn, yn = cp.min(axis=0); xx, yx = cp.max(axis=0)
            xn = max(0, xn); yn = max(0, yn)
            xx = min(img.shape[1], xx+1); yx = min(img.shape[0], yx+1)
            if xx > xn and yx > yn:
                roi = img[yn:yx, xn:xx]; lc = cp - np.array([xn, yn])
                m = np.zeros(roi.shape[:2], dtype=np.uint8)
                cv2.fillPoly(m, [lc], 255)
                _tint_mask(roi, m > 0, col, 76 if t.kind == 'cube' else 46)
            cv2.polylines(img, [cp], True, _BK, thick+2)
            cv2.polylines(img, [cp], True, col, thick)
            if is_sel and self._locked_approach is not None:
                ax, ay, _ = self._locked_approach
                bi = 0; bd = 1e9
                for i in range(4):
                    mx = (t.corners[i][0]+t.corners[(i+1)%4][0])*0.5
                    my = (t.corners[i][1]+t.corners[(i+1)%4][1])*0.5
                    d = math.hypot(mx-ax, my-ay)
                    if d < bd: bd = d; bi = i
                cv2.line(img, tuple(cp[bi]), tuple(cp[(bi+1)%4]), _OR, thick+2)
            cx_px = int(t.center[0]*ppc); cy_px = int(t.center[1]*ppc)
            rb = 15 if is_sel else 12
            cv2.circle(img, (cx_px, cy_px), rb, _BK, -1)
            cv2.circle(img, (cx_px, cy_px), rb, col, 2)
            off = -5 if t.id < 10 else -10
            _txt_fast(img, str(t.id), (cx_px+off, cy_px+5), 0.55, _W, 2)
            ly = max(int(cp[:, 1].min()) - 6, 14)
            _txt(img, t.label(), (int(cp[:, 0].min()), ly), 0.45, col, 1, shadow=True)
            if is_sel and self._locked_approach is not None:
                ax, ay, ah = self._locked_approach
                apx = int(ax*ppc); apy = int(ay*ppc)
                cv2.drawMarker(img, (apx, apy), _OR, cv2.MARKER_TRIANGLE_UP, 16, 2)
                al = max(8, int(10*ppc))
                cv2.arrowedLine(img, (apx, apy),
                                (int(apx+al*math.cos(ah)), int(apy+al*math.sin(ah))),
                                _OR, 2, tipLength=0.3)
                _txt_fast(img, "LOCK", (apx+8, apy-6), 0.4, _OR, 1)

    def _draw_path_fast(self, img, ppc):
        pa = self._path_arr
        if pa is None or len(pa) < 2: return
        pts = (pa * ppc).astype(np.int32)
        cv2.polylines(img, [pts], False, _YL, 2)
        cv2.circle(img, tuple(pts[0]), 6, _GR, -1)
        cv2.circle(img, tuple(pts[-1]), 9, _MG, -1)

    def _draw_trail_fast(self, img, ppc):
        trail = self.robot_trail; n = len(trail)
        if n < 2: return
        pts = list(trail)
        for i in range(0, n-1, 3):
            j = min(i+3, n-1)
            p1 = (int(pts[i][0]*ppc), int(pts[i][1]*ppc))
            p2 = (int(pts[j][0]*ppc), int(pts[j][1]*ppc))
            a = int(60 + 190 * i / max(n-1, 1))
            cv2.line(img, p1, p2, (a//4, a, a//4), 1)

    def _draw_robot_fast(self, img, ppc, lookahead):
        pose = self.perception.robot_pose_cm
        if pose is None: return
        rx = int(pose[0]*ppc); ry = int(pose[1]*ppc); yaw = pose[2]

        corners = self.perception.get_robot_hitbox_corners_px()
        if corners is not None:
            c = corners.astype(np.int32)
            xn, yn = c.min(axis=0); xx, yx = c.max(axis=0)
            xn = max(0, xn); yn = max(0, yn)
            xx = min(img.shape[1], xx+1); yx = min(img.shape[0], yx+1)
            if xx > xn and yx > yn:
                roi = img[yn:yx, xn:xx]; lc = c - np.array([xn, yn])
                m = np.zeros(roi.shape[:2], dtype=np.uint8)
                cv2.fillPoly(m, [lc], 255)
                _tint_mask(roi, m > 0, (40, 40, 40), 128)
            cv2.polylines(img, [c], True, _BK, 3)
            cv2.polylines(img, [c], True, _CY, 1)
        else:
            r = max(10, int(ARUCO_REAL_CM * ppc * 0.65))
            cv2.rectangle(img, (rx-r, ry-r), (rx+r, ry+r), _BK, -1)
            cv2.rectangle(img, (rx-r, ry-r), (rx+r, ry+r), _CY, 2)

        cy_ = math.cos(yaw); sy_ = math.sin(yaw)
        cl = int(ARUCO_REAL_CM * ppc * 0.5); cw = int(ARUCO_REAL_CM * ppc * 0.3)
        tip = (int(rx+cl*cy_), int(ry+cl*sy_))
        cv2.fillPoly(img, [np.array([tip,
            (int(rx-cw*sy_), int(ry+cw*cy_)),
            (int(rx+cw*sy_), int(ry-cw*cy_))], np.int32)], _CY)
        al = int(ARUCO_REAL_CM * ppc * 0.9)
        cv2.arrowedLine(img, (rx, ry), (int(rx+al*cy_), int(ry+al*sy_)), _CY, 2, tipLength=0.3)

        # Front-line 3 points (NEW v7 — visual)
        fl_pts = self._front_line_points(pose)
        for i, pt in enumerate(fl_pts):
            px = int(pt[0]*ppc); py = int(pt[1]*ppc)
            col = _OR if i == 0 else _YL
            cv2.circle(img, (px, py), 4 if i == 0 else 3, col, -1)

        _txt_fast(img, "CLAUDIO",
                  (rx - int(ARUCO_REAL_CM*ppc*0.7), max(ry - int(ARUCO_REAL_CM*ppc*0.9), 16)),
                  0.6, _CY, 2)
        _txt_fast(img, f"{math.degrees(yaw)%360:.0f}",
                  (rx + int(ARUCO_REAL_CM*ppc*0.8), ry-3), 0.45, _YL, 1)

        if lookahead:
            lx = int(lookahead[0]*ppc); ly = int(lookahead[1]*ppc)
            _dashed(img, (rx, ry), (lx, ly), _GR, 1)
            cv2.circle(img, (lx, ly), 7, _GR, 2)
        if self.goal_cm and self._locked_approach is None:
            gx = int(self.goal_cm[0]*ppc); gy = int(self.goal_cm[1]*ppc)
            cv2.drawMarker(img, (gx, gy), _MG, cv2.MARKER_TILTED_CROSS, 20, 2)
        if self._in_final_approach:
            _txt_fast(img, "FINAL APPROACH", (rx-60, ry+int(ARUCO_REAL_CM*ppc*1.2)), 0.5, _OR, 2)

    def _draw_heading(self, img, ppc):
        pose = self.perception.robot_pose_cm
        if pose is None: return
        rx_cm, ry_cm, yaw = pose
        rx = int(rx_cm*ppc); ry = int(ry_cm*ppc)
        cr = int(ARUCO_REAL_CM*ppc*1.8)
        cv2.circle(img, (rx, ry), cr, _DG, 1)
        for ad, lab in [(0,'E'),(90,'S'),(180,'W'),(270,'N')]:
            a = math.radians(ad)
            cv2.line(img, (int(rx+(cr-6)*math.cos(a)), int(ry+(cr-6)*math.sin(a))),
                     (int(rx+cr*math.cos(a)), int(ry+cr*math.sin(a))), _GY, 2)
            _txt_fast(img, lab, (int(rx+(cr+10)*math.cos(a))-4,
                                 int(ry+(cr+10)*math.sin(a))+4), 0.4, _GY, 1)
        cv2.line(img, (rx, ry), (int(rx+cr*math.cos(yaw)), int(ry+cr*math.sin(yaw))), _CY, 2)

    def _draw_debug(self, img, ppc):
        raw = self._layers_cache.get('obstacles_raw')
        if raw is not None:
            h, w = img.shape[:2]; tw = w//5; th = h//5
            t = cv2.resize(raw.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST)
            tb = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR); tb[t > 0] = (50, 50, 220)
            img[h-th-4:h-4, w-tw-4:w-4] = tb
            _txt_fast(img, "OBS RAW", (w-tw-4, h-th-8), 0.4, _OR)
        if self._excl_mask is not None:
            h, w = img.shape[:2]; tw = w//5; th = h//5
            e = cv2.resize(self._excl_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
            eb = cv2.cvtColor(e, cv2.COLOR_GRAY2BGR); eb[e > 0] = (255, 200, 0)
            y0 = h - 2*th - 12
            img[y0:y0+th, w-tw-4:w-4] = eb
            _txt_fast(img, "EXCL", (w-tw-4, y0-4), 0.4, _CY)
        # Cube obstacle mask debug
        if self._cube_obstacle_mask is not None and self._debug_mode:
            h, w = img.shape[:2]; tw = w//5; th = h//5
            cm = cv2.resize(self._cube_obstacle_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
            cb = cv2.cvtColor(cm, cv2.COLOR_GRAY2BGR); cb[cm > 0] = (0, 140, 255)
            y0 = h - 3*th - 20
            if y0 >= 0:
                img[y0:y0+th, w-tw-4:w-4] = cb
                _txt_fast(img, "CUBE OBS", (w-tw-4, y0-4), 0.4, _OR)

    def _draw_panel_fast(self, img):
        L = []; C_ = []
        if len(self._fps_buf) > 1:
            fps = len(self._fps_buf) / (self._fps_buf[-1] - self._fps_buf[0] + 1e-6)
            L.append(f"FPS:{fps:.0f}"); C_.append(_GY)
        pose = self.perception.robot_pose_cm
        if pose:
            L.append(f"({pose[0]:.0f},{pose[1]:.0f}) {math.degrees(pose[2])%360:.0f}deg"); C_.append(_CY)
        else: L.append("NO POSE"); C_.append(_RD)
        if self.ros:
            busy = self.ros.is_mc_busy()
            L.append(f"MC:{'BUSY' if busy else 'IDLE'}"); C_.append(_OR if busy else _GR)
        if self.active_path:
            L.append(f"Path:{len(self.active_path)} {self.progress*100:.0f}%"); C_.append(_W)
            ok = self.deviation_cm <= self.DEVIATION_THRESHOLD
            L.append(f"Dev:{self.deviation_cm:.0f}cm{'!' if not ok else ''}"); C_.append(_GR if ok else _RD)
        if self._in_final_approach:
            L.append(">> FINAL APPROACH <<"); C_.append(_OR)
        if self._ignore_zones:
            L.append(f"Ignore zones: {len(self._ignore_zones)}"); C_.append(_GY)
        if self._ignore_mode:
            L.append(">> IGNORE MODE (R-click to place) <<"); C_.append(_RD)
        for t in sorted(self.targets, key=lambda x: x.id)[:9]:
            s = ">>" if t.id == self.selected_target_id else "  "
            L.append(f"{s}{t.id}:{t.label()}")
            C_.append(_OR if t.id == self.selected_target_id else t.color_bgr)
        if not L: return
        pad = 6; lh = 18; pw = 310; ph = len(L)*lh + pad*2
        h, w = img.shape[:2]; ph = min(ph, h)
        roi = img[0:ph, 0:pw]
        roi[:] = (roi.astype(np.uint16) * 77 >> 8).astype(np.uint8)
        for i, (line, col) in enumerate(zip(L, C_)):
            cv2.putText(img, line, (pad, pad + (i+1)*lh), _FONT, 0.42, col, 1, cv2.LINE_4)

    # ── Run modes ─────────────────────────────────────────────────────────────

    def run_image(self, img_path: str):
        raw = cv2.imread(img_path)
        if raw is None: return
        while True:
            display = self._process_frame(raw)
            cv2.imshow(self.WIN, display)
            if not self._key(cv2.waitKey(30) & 0xFF): break
        cv2.destroyAllWindows()

    def run_live(self):
        from vision_cenital.camera import CargaCam
        with CargaCam() as cam:
            while True:
                ret, frame = cam.read()
                if not ret: continue
                display = self._process_frame(frame)
                cv2.imshow(self.WIN, display)
                if not self._key(cv2.waitKey(1) & 0xFF): break
        cv2.destroyAllWindows()

    def run_ros_sim(self):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        _txt_fast(blank, "Esperando camera ...", (20, 240), 0.8, _YL)
        while True:
            frame = self.ros.get_frame() if self.ros else None
            display = self._process_frame(frame) if frame is not None else blank
            cv2.imshow(self.WIN, display)
            if not self._key(cv2.waitKey(1) & 0xFF): break
        cv2.destroyAllWindows()

    def _key(self, k: int) -> bool:
        if k in (27, ord('q'), ord('Q')): return False
        if k in (ord('r'), ord('R')):
            self._clear_nav()
            self.start_cm = self.goal_cm = None
            self.robot_trail.clear(); self.replan_count = 0
            if self.ros: self.ros.send_stop()
        elif k in (ord('p'), ord('P')):
            if self.selected_target_id is not None:
                self._select_target(self.selected_target_id)
            else: self._plan()
        elif k in (ord('d'), ord('D')):
            self._debug_mode = not self._debug_mode
        elif k in (ord('h'), ord('H')):
            self._heading_debug = not self._heading_debug
        elif k in (ord('s'), ord('S')):
            if self.ros: self.ros.send_stop()
        elif k in (ord('e'), ord('E')):
            self._ignore_mode = not self._ignore_mode
            print(f"Ignore mode: {'ON — right-click to place zones' if self._ignore_mode else 'OFF'}")
        elif k in (ord('x'), ord('X')):
            n = len(self._ignore_zones)
            self._ignore_zones.clear()
            print(f"[IGN] Cleared {n} ignore zones")
        elif ord('1') <= k <= ord('9'):
            self._select_target(k - ord('0'))
        return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="CargaBot Vision v7")
    ap.add_argument('--params',       default='resource/camera_params.yaml')
    ap.add_argument('--homography',   default='resource/homography_retry.yaml')
    ap.add_argument('--image',        default='')
    ap.add_argument('--grid-res',     type=float, default=5.0,  dest='grid_res')
    ap.add_argument('--robot-radius', type=float, default=15.0, dest='robot_radius')
    ap.add_argument('--sim',          action='store_true')
    ap.add_argument('--ros',          action='store_true')
    ap.add_argument('--ros-sim',      action='store_true', dest='ros_sim')
    ap.add_argument('--layer-interval', type=int, default=3, dest='layer_interval')
    args = ap.parse_args()
    if args.ros_sim: args.ros = True; args.sim = True

    app = CargaBotVisionApp(args)
    if hasattr(args, 'layer_interval'):
        app.LAYER_INTERVAL = args.layer_interval

    if args.image:       app.run_image(args.image)
    elif args.ros_sim:   app.run_ros_sim()
    else:                app.run_live()
    if app.ros:          app.ros.shutdown()


if __name__ == '__main__':
    main()