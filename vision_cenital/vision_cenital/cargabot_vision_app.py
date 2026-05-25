#!/usr/bin/env python3
"""
CargaBot Vision App v3 — Rotated hitbox + heading debug
========================================================
Cambios vs v2:
  - _draw_robot: rotated hitbox via get_robot_hitbox_corners_px()
    with beveled corners, fill, and directional chevron
  - _draw_heading_debug: compass + angle-to-origin overlay (toggle H)
  - ROBOT_ARUCO_ID ahora es 0 (perception.py v7)
  - D: debug overlay | H: heading debug | R: clear | P: plan

  uv run vision_cenital/cargabot_vision_app.py   --params resource/camera_params.yaml   --homography resource/homography_retry.yaml

  Comando para correr con ROS:
  uv run vision_cenital/cargabot_vision_app.py   --params resource/camera_params.yaml   --homography resource/homography_retry.yaml   --ros
  """

# ── CRÍTICO: forzar X11 ANTES de que Qt se inicialice (antes de import cv2) ──
import os
os.environ['WAYLAND_DISPLAY'] = ''
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['DISPLAY']         = os.environ.get('DISPLAY', ':0')
# ─────────────────────────────────────────────────────────────────────────────

import sys
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
    from cv_bridge          import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    pass


# ── Paleta ────────────────────────────────────────────────────────────────────
C = {
    'white':   (255, 255, 255),
    'black':   (  0,   0,   0),
    'yellow':  (  0, 220, 255),
    'cyan':    (255, 210,   0),
    'green':   ( 50, 220,  50),
    'red':     ( 50,  50, 220),
    'orange':  (  0, 160, 255),
    'magenta': (220,   0, 220),
    'gray':    (140, 140, 140),
    'darkgray':( 40,  40,  40),
    'teal':    (180, 200,   0),
}

LAYER_PALETTE = {
    'red':   (( 50,  50, 220), 'ROJO'),
    'green': (( 50, 220,  50), 'VERDE'),
    'blue':  ((220,  80,  50), 'AZUL'),
}

def layer_style(name: str):
    for k, (color, label) in LAYER_PALETTE.items():
        if k in name.lower():
            return color, label
    return C['gray'], name.upper()


# ── Helpers de dibujo ─────────────────────────────────────────────────────────

def text(img, msg, pos, scale=0.6, color=C['white'], thick=1):
    cv2.putText(img, msg, pos, cv2.FONT_HERSHEY_DUPLEX, scale, C['black'], thick+3)
    cv2.putText(img, msg, pos, cv2.FONT_HERSHEY_DUPLEX, scale, color,    thick)

def dashed_line(img, p1, p2, color, thick=1, dash=12, gap=6):
    dx = p2[0]-p1[0]; dy = p2[1]-p1[1]
    d  = max(1, int(np.hypot(dx, dy)))
    for i in range(0, d, dash+gap):
        t0 = i/d; t1 = min((i+dash)/d, 1.0)
        a  = (int(p1[0]+dx*t0), int(p1[1]+dy*t0))
        b  = (int(p1[0]+dx*t1), int(p1[1]+dy*t1))
        cv2.line(img, a, b, color, thick)

def panel_bg(img, x, y, w, h, alpha=0.60):
    sub = img[y:y+h, x:x+w]
    black = np.zeros_like(sub)
    img[y:y+h, x:x+w] = cv2.addWeighted(sub, 1-alpha, black, alpha, 0)


# ── ROS2 Bridge ───────────────────────────────────────────────────────────────

class ROSBridge:
    def __init__(self):
        rclpy.init()
        self.node         = rclpy.create_node('cargabot_vision_app')
        self.bridge       = CvBridge()
        self.latest_frame = None
        self.odom_pose:   Optional[Tuple] = None
        self.ext_goal:    Optional[Tuple] = None
        self._lock        = threading.Lock()

        self.pub_path  = self.node.create_publisher(Path,        '/cargabot/global_path',   10)
        self.pub_goto  = self.node.create_publisher(PoseStamped, '/cargabot/cmd_goto',       10)
        self.pub_debug = self.node.create_publisher(ROSImage,    '/cargabot/overhead_debug', 10)

        self.node.create_subscription(Odometry,    '/odom',                      self._odom_cb, 10)
        self.node.create_subscription(PoseStamped, '/cargabot/goal_pose',        self._goal_cb, 10)
        self.node.create_subscription(ROSImage,    '/cargabot/camera/image_raw', self._cam_cb,  10)

        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        print("[ROS2] Bridge online")

    def _spin(self):   rclpy.spin(self.node)

    def _odom_cb(self, msg):
        qz = msg.pose.pose.orientation.z; qw = msg.pose.pose.orientation.w
        with self._lock:
            self.odom_pose = (msg.pose.pose.position.x,
                              msg.pose.pose.position.y,
                              2.0 * math.atan2(qz, qw))

    def _goal_cb(self, msg):
        with self._lock:
            self.ext_goal = (msg.pose.position.x * 100.0,
                             msg.pose.position.y * 100.0)

    def _cam_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with self._lock: self.latest_frame = frame

    def get_frame(self):
        with self._lock: return self.latest_frame

    def get_odom(self):
        with self._lock: return self.odom_pose

    def get_ext_goal(self):
        with self._lock:
            g = self.ext_goal; self.ext_goal = None; return g

    def publish_path(self, path_cm):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp    = self.node.get_clock().now().to_msg()
        for x, y in path_cm:
            p = PoseStamped(); p.header = msg.header
            p.pose.position.x = x / 100.0; p.pose.position.y = y / 100.0
            msg.poses.append(p)
        self.pub_path.publish(msg)

    def publish_goto(self, target_cm):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp    = self.node.get_clock().now().to_msg()
        msg.pose.position.x = target_cm[0] / 100.0
        msg.pose.position.y = target_cm[1] / 100.0
        self.pub_goto.publish(msg)

    def publish_debug(self, frame):
        try:
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(frame, encoding='bgr8'))
        except Exception: pass

    def shutdown(self): rclpy.shutdown()


# ── Aplicación principal ──────────────────────────────────────────────────────

class CargaBotVisionApp:

    LOOKAHEAD_CM        = 22.0
    DEVIATION_THRESHOLD = 15.0
    TRAIL_MAX           = 500
    MIN_CUBE_PX         = 100
    REPLAN_COOLDOWN     = 2.0

    def __init__(self, args):
        self.args = args

        self.perception = OverheadPerception(
            args.params, args.homography,
            sim_mode=getattr(args, 'sim', False))
        self.navigator = GridNavigator(
            self.perception.pista_w_cm,
            self.perception.pista_h_cm,
            args.grid_res,
            args.robot_radius)

        self.ros: Optional[ROSBridge] = None
        if getattr(args, 'ros', False) and ROS_AVAILABLE:
            self.ros = ROSBridge()
        elif getattr(args, 'ros', False):
            print("[WARN] --ros solicitado pero rclpy no disponible")

        self.start_cm:    Optional[Tuple] = None
        self.goal_cm:     Optional[Tuple] = None
        self.active_path: List[Tuple]     = []
        self._cost_ready  = False
        self.robot_trail  = deque(maxlen=self.TRAIL_MAX)
        self.deviation_cm = 0.0
        self.progress     = 0.0
        self.replan_count = 0
        self._last_replan = 0.0
        self._fps_buf     = deque(maxlen=30)
        self._debug_mode  = False
        self._heading_debug = False
        self._layers_cache: Dict = {}

        # Cache del grid
        self._grid_overlay: Optional[np.ndarray] = None
        self._grid_shape:   Optional[Tuple]       = None

        self.WIN = "CargaBot Vision"
        self._init_window()
        self._print_banner()

    # ── Inicialización ventana ────────────────────────────────────────────────

    def _init_window(self):
        _dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(_dummy, "Iniciando...", (20, 240),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 220, 255), 2)

        print("[Window] Inicializando ventana Qt5/xcb...")
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.imshow(self.WIN, _dummy)
        for _ in range(20): cv2.waitKey(50)

        try:
            cv2.setMouseCallback(self.WIN, self._mouse)
            cv2.resizeWindow(self.WIN, 1400, 800)
            print("[Window] OK (intento 1 - WINDOW_NORMAL)")
            return
        except cv2.error as e:
            print(f"[Window] Intento 1 fallo: {e}")

        cv2.destroyAllWindows()
        for _ in range(5): cv2.waitKey(50)
        cv2.namedWindow(self.WIN, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.WIN, _dummy)
        for _ in range(20): cv2.waitKey(50)

        try:
            cv2.setMouseCallback(self.WIN, self._mouse)
            print("[Window] OK (intento 2 - WINDOW_AUTOSIZE)")
            return
        except cv2.error as e:
            print(f"[Window] Intento 2 fallo: {e}")
            raise RuntimeError(f"No se pudo inicializar la ventana: {e}")

    def _print_banner(self):
        ros_str = "ROS2 ONLINE" if self.ros else "STANDALONE"
        print(f"""
+================================================+
|   CargaBot Vision App  [{ros_str:^16}]    |
+================================================+
|  L-click  -> ORIGEN                            |
|  R-click  -> DESTINO                           |
|  R        -> limpiar ruta                      |
|  P        -> replanificar                      |
|  D        -> debug overlay                     |
|  H        -> heading debug (grados vs origen)  |
|  ESC/Q    -> salir                             |
+================================================+
""")

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def _mouse(self, event, x, y, flags, param):
        ppc = max(self.perception.px_per_cm, 0.01)
        pt  = (x / ppc, y / ppc)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_cm = pt
            print(f"Origen -> ({pt[0]:.1f}, {pt[1]:.1f}) cm")
            self._plan()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.goal_cm = pt
            print(f"Destino -> ({pt[0]:.1f}, {pt[1]:.1f}) cm")
            self._plan()

    # ── Planificación ─────────────────────────────────────────────────────────

    def _plan(self):
        if not (self.start_cm and self.goal_cm and self._cost_ready): return
        path = self.navigator.astar_search(self.start_cm, self.goal_cm)
        if path:
            self.active_path  = path
            self.progress     = 0.0
            self.replan_count = 0
            if self.ros: self.ros.publish_path(path)
            print(f"A* -> {len(path)} nodos")
        else:
            print("A* -> sin ruta (obstruida)")

    def _auto_replan(self, rx, ry):
        now = time.time()
        if now - self._last_replan < self.REPLAN_COOLDOWN: return
        if not self.goal_cm: return
        path = self.navigator.astar_search((rx, ry), self.goal_cm)
        if path:
            self.active_path  = path
            self.start_cm     = (rx, ry)
            self.replan_count += 1
            self._last_replan = now
            if self.ros: self.ros.publish_path(path)
            print(f"Replan #{self.replan_count}: desv={self.deviation_cm:.1f}cm")

    def _update_tracking(self) -> Optional[Tuple]:
        pose = self.perception.robot_pose_cm
        if pose is None or len(self.active_path) < 2: return None
        rx, ry = pose[0], pose[1]
        dists  = [np.hypot(rx-px, ry-py) for px, py in self.active_path]
        idx    = int(np.argmin(dists))
        self.deviation_cm = dists[idx]
        self.progress     = idx / max(len(self.active_path)-1, 1)

        if self.deviation_cm > self.DEVIATION_THRESHOLD:
            self._auto_replan(rx, ry)

        accum = 0.0
        lh    = self.active_path[-1]
        for i in range(idx, len(self.active_path)-1):
            dx = self.active_path[i+1][0] - self.active_path[i][0]
            dy = self.active_path[i+1][1] - self.active_path[i][1]
            accum += np.hypot(dx, dy)
            if accum >= self.LOOKAHEAD_CM:
                lh = self.active_path[i+1]; break

        if self.ros and self.perception.robot_pose_cm:
            self.ros.publish_goto(lh)
        return lh

    # ── Frame ─────────────────────────────────────────────────────────────────

    def _process_frame(self, raw: np.ndarray) -> np.ndarray:
        self._fps_buf.append(time.time())

        warped = self.perception.warp_to_overhead(raw)

        pose = self.perception.detect_robot_pose(raw)
        if pose is None and self.ros and self.ros.get_odom():
            ox, oy, oyaw = self.ros.get_odom()
            self.perception.inject_pose_from_odom(ox, oy, oyaw)

        if self.ros:
            eg = self.ros.get_ext_goal()
            if eg:
                self.goal_cm = eg; self._plan()

        layers = self.perception.extract_semantic_layers(warped)
        self._layers_cache = layers
        print(f"[DBG] layers keys: {list(layers.keys())}, cost_ready={self._cost_ready}")  # ADD THIS
        self._layers_cache = layers

        obs = layers.get('obstacles')
        if obs is not None:
            self.navigator.generate_cost_map(obs)
            self._cost_ready = True

        pose = self.perception.robot_pose_cm
        if pose:
            self.robot_trail.append(pose[:2])
            if not self.start_cm:
                self.start_cm = pose[:2]

        display = warped.copy()
        self._render(display, layers)

        if self.ros:
            self.ros.publish_debug(display)

        return display

    # ── Render ────────────────────────────────────────────────────────────────

    def _render(self, img, layers):
        ppc = max(self.perception.px_per_cm, 0.01)
        self._draw_grid(img, ppc)
        self._draw_obstacles(img, layers, ppc)
        self._draw_zones(img, layers, ppc)
        self._draw_cubes(img, layers)

        lh = None
        if self.active_path:
            lh = self._update_tracking()
            self._draw_path(img, ppc)

        self._draw_trail(img, ppc)
        self._draw_robot(img, ppc, lh)
        self._draw_markers(img, ppc)
        self._draw_walls(img, layers)

        if self._debug_mode:
            self._draw_debug(img, layers, ppc)

        if self._heading_debug:
            self._draw_heading_debug(img, ppc)

        self._draw_panel(img)

    def _draw_grid(self, img, ppc):
        h, w = img.shape[:2]
        shape = (h, w)

        if self._grid_shape != shape:
            overlay = np.zeros((h, w, 3), dtype=np.uint8)
            step = max(1, int(5.0 * ppc))
            for x in range(0, w, step):
                cv2.line(overlay, (x, 0), (x, h), (30, 30, 30), 1)
            for y in range(0, h, step):
                cv2.line(overlay, (0, y), (w, y), (30, 30, 30), 1)
            self._grid_overlay = overlay
            self._grid_shape   = shape

        cv2.add(img, self._grid_overlay, img)

    def _draw_obstacles(self, img, layers, ppc):
        obs = layers.get('obstacles')
        if obs is None or not self._cost_ready: return
        try:
            m = obs.astype(np.uint8) if obs.dtype != np.uint8 else obs
            m_resized = cv2.resize(m, (img.shape[1], img.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            red = np.zeros_like(img); red[:,:] = (0, 0, 180)
            blended = cv2.addWeighted(img, 0.6, red, 0.4, 0)
            img[m_resized > 0] = blended[m_resized > 0]
        except Exception: pass

    def _draw_walls(self, img, layers):
        walls = layers.get('walls')
        if walls is None: return
        try:
            m = walls.astype(np.uint8) if walls.dtype != np.uint8 else walls
            m_resized = cv2.resize(m, (img.shape[1], img.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            blue = np.zeros_like(img); blue[:,:] = (140, 60, 0)
            blended = cv2.addWeighted(img, 0.5, blue, 0.5, 0)
            img[m_resized > 0] = blended[m_resized > 0]
        except Exception: pass

    def _draw_zones(self, img, layers, ppc):
        for name, mask in layers.items():
            if 'zone' not in name or mask is None: continue
            color, label = layer_style(name)
            m = mask.astype(np.uint8) if mask.dtype != np.uint8 else mask
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                if cv2.contourArea(cnt) < self.MIN_CUBE_PX: continue
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x+w, y+h), C['black'], 4)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 1)
                for i in range(0, w+h, 12):
                    p1 = (x + min(i, w), y + max(0, i-w))
                    p2 = (x + max(0, i-h), y + min(i, h))
                    cv2.line(img, p1, p2, color, 1)
                text(img, f"DROP-{label}", (x, max(y-6, 14)), 0.5, color)

    def _draw_cubes(self, img, layers):
        for name, mask in layers.items():
            if 'solid' not in name or mask is None: continue
            color, label = layer_style(name)
            m = mask.astype(np.uint8) if mask.dtype != np.uint8 else mask
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                if cv2.contourArea(cnt) < self.MIN_CUBE_PX: continue
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x-1, y-1), (x+w+1, y+h+1), C['black'], 5)
                cv2.rectangle(img, (x,   y),   (x+w,   y+h),   color,     2)
                overlay = img.copy()
                cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
                img[y:y+h, x:x+w] = cv2.addWeighted(
                    img[y:y+h, x:x+w], 0.80, overlay[y:y+h, x:x+w], 0.20, 0)
                text(img, label, (x+3, max(y-5, 14)), 0.6, color, thick=2)

    def _draw_path(self, img, ppc):
        if len(self.active_path) < 2: return
        pts = [(int(p[0]*ppc), int(p[1]*ppc)) for p in self.active_path]
        for i in range(len(pts)-1):
            cv2.line(img, pts[i], pts[i+1], C['yellow'], 2)
        cv2.circle(img, pts[0],   7, C['green'],   -1)
        cv2.circle(img, pts[-1], 10, C['magenta'], -1)
        cv2.circle(img, pts[-1], 10, C['white'],    1)

    def _draw_trail(self, img, ppc):
        trail = list(self.robot_trail)
        if len(trail) < 2: return
        n = len(trail)
        for i in range(n-1):
            p1 = (int(trail[i][0]*ppc),   int(trail[i][1]*ppc))
            p2 = (int(trail[i+1][0]*ppc), int(trail[i+1][1]*ppc))
            a  = int(60 + 190 * i / max(n-1, 1))
            cv2.line(img, p1, p2, (a//4, a, a//4), 2)

    def _draw_robot(self, img, ppc, lookahead):
        """Draw robot with rotated hitbox, directional chevron, and label."""
        pose = self.perception.robot_pose_cm
        if pose is None: return
        rx = int(pose[0]*ppc); ry = int(pose[1]*ppc); yaw = pose[2]

        # ── Rotated hitbox from perception ────────────────────────────────────
        corners = self.perception.get_robot_hitbox_corners_px()
        if corners is not None:
            # Semi-transparent fill
            overlay = img.copy()
            cv2.fillPoly(overlay, [corners], (40, 40, 40))
            cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

            # Border: outer black + inner cyan
            cv2.polylines(img, [corners], True, C['black'], 3)
            cv2.polylines(img, [corners], True, C['cyan'],  1)

            # Corner dots for visibility
            for pt in corners:
                cv2.circle(img, tuple(pt), 3, C['cyan'], -1)
        else:
            # Fallback: simple square if no hitbox available
            r = max(10, int(ARUCO_REAL_CM * ppc * 0.65))
            cv2.rectangle(img, (rx-r, ry-r), (rx+r, ry+r), C['black'], -1)
            cv2.rectangle(img, (rx-r, ry-r), (rx+r, ry+r), C['cyan'],   2)

        # ── Directional chevron (forward indicator) ───────────────────────────
        chevron_len = int(ARUCO_REAL_CM * ppc * 0.5)
        chevron_w   = int(ARUCO_REAL_CM * ppc * 0.3)
        cos_y = np.cos(yaw); sin_y = np.sin(yaw)

        # Tip of chevron (forward)
        tip = (int(rx + chevron_len * cos_y),
               int(ry + chevron_len * sin_y))
        # Two back points of chevron
        left  = (int(rx - chevron_w * sin_y),
                 int(ry + chevron_w * cos_y))
        right = (int(rx + chevron_w * sin_y),
                 int(ry - chevron_w * cos_y))

        chev_pts = np.array([tip, left, right], dtype=np.int32)
        cv2.fillPoly(img, [chev_pts], C['cyan'])
        cv2.polylines(img, [chev_pts], True, C['white'], 1)

        # ── Heading arrow ─────────────────────────────────────────────────────
        al = int(ARUCO_REAL_CM * ppc * 0.9)
        ex = int(rx + al*cos_y); ey = int(ry + al*sin_y)
        cv2.arrowedLine(img, (rx, ry), (ex, ey), C['cyan'], 2, tipLength=0.3)

        # ── Label ─────────────────────────────────────────────────────────────
        # Place label above the hitbox
        label_y = ry - int(ARUCO_REAL_CM * ppc * 0.9)
        text(img, "CLAUDIO", (rx - int(ARUCO_REAL_CM * ppc * 0.7),
                              max(label_y, 16)), 0.7, C['cyan'], thick=2)

        # ── Heading degrees text near robot ───────────────────────────────────
        deg = math.degrees(yaw) % 360
        text(img, f"{deg:.0f}deg", (rx + int(ARUCO_REAL_CM * ppc * 0.8),
                                     ry - 5), 0.5, C['yellow'])

        # ── Lookahead ─────────────────────────────────────────────────────────
        if lookahead:
            lx = int(lookahead[0]*ppc); ly = int(lookahead[1]*ppc)
            dashed_line(img, (rx, ry), (lx, ly), C['green'], 1)
            cv2.circle(img, (lx, ly), 9, C['green'],  2)
            cv2.circle(img, (lx, ly), 3, C['white'], -1)

    def _draw_heading_debug(self, img, ppc):
        """
        Heading debug overlay:
          - Compass rose at robot position
          - Angle from robot to origin (0,0)
          - Angle from origin to robot
          - Current heading in degrees
        """
        pose = self.perception.robot_pose_cm
        if pose is None: return
        rx_cm, ry_cm, yaw = pose
        rx = int(rx_cm * ppc); ry = int(ry_cm * ppc)

        # ── Compass rose around robot ─────────────────────────────────────────
        compass_r = int(ARUCO_REAL_CM * ppc * 1.8)

        # Draw compass circle
        cv2.circle(img, (rx, ry), compass_r, C['darkgray'], 1)

        # Cardinal directions
        for angle_deg, label_str in [(0, 'E'), (90, 'S'), (180, 'W'), (270, 'N')]:
            a = math.radians(angle_deg)
            ex = int(rx + compass_r * math.cos(a))
            ey = int(ry + compass_r * math.sin(a))
            # Tick mark
            ix = int(rx + (compass_r - 6) * math.cos(a))
            iy = int(ry + (compass_r - 6) * math.sin(a))
            cv2.line(img, (ix, iy), (ex, ey), C['gray'], 2)
            # Label
            lx = int(rx + (compass_r + 12) * math.cos(a)) - 5
            ly = int(ry + (compass_r + 12) * math.sin(a)) + 5
            text(img, label_str, (lx, ly), 0.45, C['gray'])

        # 45-degree ticks
        for angle_deg in [45, 135, 225, 315]:
            a = math.radians(angle_deg)
            ex = int(rx + compass_r * math.cos(a))
            ey = int(ry + compass_r * math.sin(a))
            ix = int(rx + (compass_r - 4) * math.cos(a))
            iy = int(ry + (compass_r - 4) * math.sin(a))
            cv2.line(img, (ix, iy), (ex, ey), C['darkgray'], 1)

        # Current heading line on compass (bright cyan)
        hx = int(rx + compass_r * math.cos(yaw))
        hy = int(ry + compass_r * math.sin(yaw))
        cv2.line(img, (rx, ry), (hx, hy), C['cyan'], 2)

        # ── Angle to origin (0,0) ────────────────────────────────────────────
        origin_px_x = 0; origin_px_y = 0

        # Draw origin marker
        cv2.drawMarker(img, (origin_px_x, origin_px_y),
                        C['yellow'], cv2.MARKER_DIAMOND, 16, 2)
        text(img, "ORIGIN", (origin_px_x + 10, origin_px_y + 16), 0.5, C['yellow'])

        # Angle from robot to origin
        dx = 0.0 - rx_cm
        dy = 0.0 - ry_cm
        angle_to_origin = math.atan2(dy, dx)
        angle_to_origin_deg = math.degrees(angle_to_origin) % 360

        # Relative angle (heading vs direction to origin)
        rel_angle = math.degrees(angle_to_origin - yaw) % 360
        if rel_angle > 180: rel_angle -= 360

        # Draw dashed line to origin
        dashed_line(img, (rx, ry), (origin_px_x, origin_px_y),
                    C['yellow'], 1, dash=8, gap=8)

        # Angle arc on compass showing direction to origin
        arc_r = compass_r - 10
        ox_c = int(rx + arc_r * math.cos(angle_to_origin))
        oy_c = int(ry + arc_r * math.sin(angle_to_origin))
        cv2.circle(img, (ox_c, oy_c), 4, C['yellow'], -1)

        # ── Info panel (bottom-right of compass) ──────────────────────────────
        heading_deg = math.degrees(yaw) % 360
        dist_to_origin = math.hypot(rx_cm, ry_cm)

        info_x = rx + compass_r + 20
        info_y = ry - 40

        # Ensure panel stays on screen
        h_img, w_img = img.shape[:2]
        if info_x + 200 > w_img: info_x = rx - compass_r - 220
        if info_y < 10: info_y = 10
        if info_y + 120 > h_img: info_y = h_img - 130

        pw, ph = 210, 110
        panel_bg(img, max(0, info_x), max(0, info_y), pw, ph, alpha=0.75)
        cv2.rectangle(img, (info_x, info_y), (info_x+pw, info_y+ph),
                      C['darkgray'], 1)

        lh = 22
        text(img, f"Hdg: {heading_deg:.1f}deg",
             (info_x+8, info_y + lh), 0.5, C['cyan'])
        text(img, f"->Origin: {angle_to_origin_deg:.1f}deg",
             (info_x+8, info_y + lh*2), 0.5, C['yellow'])
        text(img, f"Rel: {rel_angle:+.1f}deg",
             (info_x+8, info_y + lh*3), 0.5, C['orange'])
        text(img, f"Dist: {dist_to_origin:.1f}cm",
             (info_x+8, info_y + lh*4), 0.5, C['white'])

    def _draw_markers(self, img, ppc):
        if self.start_cm:
            sx = int(self.start_cm[0]*ppc); sy = int(self.start_cm[1]*ppc)
            cv2.drawMarker(img, (sx, sy), C['green'], cv2.MARKER_CROSS, 22, 2)
            text(img, "START", (sx+8, sy-8), 0.55, C['green'])
        if self.goal_cm:
            gx = int(self.goal_cm[0]*ppc); gy = int(self.goal_cm[1]*ppc)
            cv2.drawMarker(img, (gx, gy), C['magenta'], cv2.MARKER_TILTED_CROSS, 24, 2)
            text(img, "GOAL", (gx+8, gy-8), 0.55, C['magenta'])

    def _draw_debug(self, img, layers, ppc):
        raw = layers.get('obstacles_raw')
        if raw is None: return
        h, w = img.shape[:2]
        thumb_w = w // 5; thumb_h = h // 5
        thumb = cv2.resize(raw.astype(np.uint8),
                           (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST)
        thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        thumb_bgr[thumb > 0] = (50, 50, 220)
        img[h-thumb_h-4:h-4, w-thumb_w-4:w-4] = thumb_bgr
        text(img, "OBS RAW", (w-thumb_w-4, h-thumb_h-8), 0.5, C['orange'])

    def _draw_panel(self, img):
        lines  = []
        colors = []

        if len(self._fps_buf) > 1:
            fps = len(self._fps_buf) / (self._fps_buf[-1] - self._fps_buf[0] + 1e-6)
            lines.append(f"FPS: {fps:.1f}"); colors.append(C['gray'])
        else:
            lines.append("FPS: --"); colors.append(C['gray'])

        ros_str = "ROS2: OK" if self.ros else "ROS2: standalone"
        lines.append(ros_str); colors.append(C['teal'] if self.ros else C['gray'])

        pose = self.perception.robot_pose_cm
        if pose:
            lines.append(f"CLAUDIO: ({pose[0]:.1f},{pose[1]:.1f}) cm")
            colors.append(C['cyan'])
            deg = math.degrees(pose[2]) % 360
            lines.append(f"Hdg: {deg:.1f}deg")
            colors.append(C['cyan'])
        else:
            lines.append("CLAUDIO: no detectado"); colors.append(C['red'])

        if self.active_path:
            lines.append(f"Ruta: {len(self.active_path)} nodos"); colors.append(C['white'])
            lines.append(f"Progreso: {self.progress*100:.0f}%"); colors.append(C['white'])
            dev_ok  = self.deviation_cm <= self.DEVIATION_THRESHOLD
            dev_str = f"Desv: {self.deviation_cm:.1f}cm {'OK' if dev_ok else '!REPLAN'}"
            lines.append(dev_str); colors.append(C['green'] if dev_ok else C['red'])
            lines.append(f"Replans: {self.replan_count}"); colors.append(C['orange'])
        else:
            lines.append("Sin ruta"); colors.append(C['gray'])

        lines.append(f"Costmap: {'OK' if self._cost_ready else 'pendiente'}")
        colors.append(C['green'] if self._cost_ready else C['orange'])
        lines.append("D:debug H:heading R:clear P:plan")
        colors.append(C['gray'])

        pad = 10; lh = 26; pw = 330; ph = len(lines) * lh + pad * 2
        panel_bg(img, 0, 0, pw, ph, alpha=0.70)
        cv2.rectangle(img, (0, 0), (pw, ph), (60, 60, 60), 1)

        for i, (line, col) in enumerate(zip(lines, colors)):
            cv2.putText(img, line, (pad, pad + (i+1)*lh),
                        cv2.FONT_HERSHEY_DUPLEX, 0.58, col, 1)

    # ── Modos de ejecución ────────────────────────────────────────────────────

    def run_image(self, img_path: str):
        raw = cv2.imread(img_path)
        if raw is None:
            print(f"No se pudo abrir: {img_path}"); return
        while True:
            display = self._process_frame(raw)
            cv2.imshow(self.WIN, display)
            key = cv2.waitKey(30) & 0xFF
            if not self._handle_key(key): break
        cv2.destroyAllWindows()

    def run_live(self):
        from vision_cenital.camera import CargaCam
        with CargaCam() as cam:
            while True:
                ret, frame = cam.read()
                if not ret: continue
                display = self._process_frame(frame)
                cv2.imshow(self.WIN, display)
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key): break
        cv2.destroyAllWindows()

    def run_ros_sim(self):
        print("[ROS2] Esperando frames en /cargabot/camera/image_raw ...")
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        text(blank, "Esperando camera/image_raw ...", (20, 240), 0.8, C['yellow'])
        while True:
            frame = self.ros.get_frame() if self.ros else None
            display = self._process_frame(frame) if frame is not None else blank.copy()
            cv2.imshow(self.WIN, display)
            key = cv2.waitKey(1) & 0xFF
            if not self._handle_key(key): break
        cv2.destroyAllWindows()

    def _handle_key(self, key: int) -> bool:
        if key in (27, ord('q'), ord('Q')): return False
        if key in (ord('r'), ord('R')):
            self.start_cm = self.goal_cm = None
            self.active_path = []; self.robot_trail.clear()
            self.replan_count = 0; print("Ruta limpiada")
        elif key in (ord('p'), ord('P')):
            self._plan()
        elif key in (ord('d'), ord('D')):
            self._debug_mode = not self._debug_mode
            print(f"Debug: {'ON' if self._debug_mode else 'OFF'}")
        elif key in (ord('h'), ord('H')):
            self._heading_debug = not self._heading_debug
            print(f"Heading debug: {'ON' if self._heading_debug else 'OFF'}")
        return True


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CargaBot Vision App")
    parser.add_argument('--params',       default='resource/camera_params.yaml')
    parser.add_argument('--homography',   default='resource/homography_retry.yaml')
    parser.add_argument('--image',        default='')
    parser.add_argument('--grid-res',     type=float, default=5.0,  dest='grid_res')
    parser.add_argument('--robot-radius', type=float, default=15.0, dest='robot_radius')
    parser.add_argument('--sim',          action='store_true')
    parser.add_argument('--ros',          action='store_true')
    parser.add_argument('--ros-sim',      action='store_true', dest='ros_sim')
    args = parser.parse_args()

    if args.ros_sim:
        args.ros = True; args.sim = True

    app = CargaBotVisionApp(args)

    if args.image:
        print(f"Modo imagen: {args.image}")
        app.run_image(args.image)
    elif args.ros_sim:
        print("Modo ROS2 simulacion")
        app.run_ros_sim()
    else:
        print("Modo live (camara real)")
        app.run_live()

    if app.ros:
        app.ros.shutdown()


if __name__ == '__main__':
    main()