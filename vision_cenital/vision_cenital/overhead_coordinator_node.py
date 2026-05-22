"""
Nodo ROS 2 de Coordinación Cenital.
Integra los motores de Percepción y Planificación, gestiona suscripciones de meta
y transmite comandos secuenciales progresivos a la base motriz de CargaBot.
Soporta inyección de video por Hardware (GStreamer) o Simulación (Gemelo Digital).

(El día que vayas al lab, solo le pasas --ros-args -p use_sim:=false
y el cerebro volverá a encender la Logitech).
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import os
import math
from ament_index_python.packages import get_package_share_directory
import numpy as np
import cv2
from typing import List, Tuple, Optional

from vision_cenital.camera import CargaCam
from vision_cenital.perception import OverheadPerception
from vision_cenital.planning import GridNavigator

# ─── Constantes de visualización ─────────────────────────────────────────────
MAX_DISPLAY_W = 960
MAX_DISPLAY_H = 540


class OverheadCoordinatorNode(Node):

    def __init__(self):
        super().__init__('overhead_coordinator_node')

        # ── Parámetros configurables ──────────────────────────────────────────
        self.declare_parameter('pista_w_cm',          408.0)
        self.declare_parameter('pista_h_cm',          206.0)
        self.declare_parameter('grid_res_cm',           5.0)
        self.declare_parameter('robot_radius_cm',      15.0)
        self.declare_parameter('lookahead_dist_cm',    25.0)
        self.declare_parameter('max_lateral_drift_cm', 10.0)
        self.declare_parameter('use_sim',               True)

        pkg_share  = get_package_share_directory('vision_cenital')
        cam_params = os.path.join(pkg_share, 'resource', 'camera_params.yaml')
        homography = os.path.join(pkg_share, 'resource', 'homography_retry.yaml')

        p_w          = self.get_parameter('pista_w_cm').value
        p_h          = self.get_parameter('pista_h_cm').value
        g_res        = self.get_parameter('grid_res_cm').value
        r_rad        = self.get_parameter('robot_radius_cm').value
        self.use_sim = self.get_parameter('use_sim').value

        # perception.py ahora recibe sim_mode y ajusta px_per_cm internamente
        self.perception = OverheadPerception(cam_params, homography,
                                             sim_mode=self.use_sim)
        self.navigator  = GridNavigator(p_w, p_h, g_res, r_rad)
        self.bridge     = CvBridge()

        # ── Estado del planificador ───────────────────────────────────────────
        self.current_goal:   Optional[Tuple[float, float]] = None
        self.current_origin: Optional[Tuple[float, float]] = None
        self.planned_path:   List[Tuple[float, float]]     = []
        self.latest_frame                                  = None

        # ── Pose inyectada desde /odom (sim_mode) ────────────────────────────
        self._odom_pose: Optional[Tuple[float, float, float]] = None

        # ── Tópicos ───────────────────────────────────────────────────────────
        self.sub_goal = self.create_subscription(
            PoseStamped, '/cargabot/goal_pose', self.goal_callback, 10
        )
        self.pub_path     = self.create_publisher(Path,        '/cargabot/global_path',         10)
        self.pub_cmd_goto = self.create_publisher(PoseStamped, '/cargabot/cmd_goto',             10)
        self.pub_debug    = self.create_publisher(Image,       '/cargabot/overhead_debug_video', 10)

        cv2.namedWindow("CargaBot Digital Twin Dashboard", cv2.WINDOW_NORMAL)
        self.setup_mouse_callback()

        if self.use_sim:
            self.sub_cam = self.create_subscription(
                Image, '/cargabot/camera/image_raw', self.cam_callback, 10
            )
            # Suscripción a /odom para obtener la pose del robot en simulación
            self.sub_odom = self.create_subscription(
                Odometry, '/odom', self._odom_callback, 10
            )
            self.timer = self.create_timer(0.1, self.control_loop)
            self.get_logger().info(
                "🚀 Cerebro Cenital ONLINE (Modo SIMULACIÓN) — "
                f"px_per_cm={self.perception.px_per_cm}"
            )
        else:
            self.cam = CargaCam()
            if self.cam.start():
                self.timer = self.create_timer(0.1, self.control_loop)
                self.get_logger().info("🚀 Cerebro Cenital ONLINE (Modo HARDWARE).")
            else:
                self.get_logger().error("❌ Imposible inicializar hardware de video.")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def goal_callback(self, msg: PoseStamped):
        self.current_goal = (
            msg.pose.position.x * 100.0,
            msg.pose.position.y * 100.0,
        )
        self.get_logger().info(f"🎯 Nueva meta: {self.current_goal} cm")
        self.planned_path = []

    def cam_callback(self, msg: Image):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _odom_callback(self, msg: Odometry):
        """
        Convierte la pose de /odom a coordenadas cenitales (cm) y la almacena.
        Se usa en sim_mode en lugar de detectar el ArUco.
        """
        ox = msg.pose.pose.position.x
        oy = msg.pose.pose.position.y
        # Quaternion → yaw
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        yaw = 2.0 * math.atan2(qz, qw)
        self._odom_pose = self.perception.inject_pose_from_odom(ox, oy, yaw)

    # ── Mouse interactivo ──────────────────────────────────────────────────────

    def setup_mouse_callback(self):
        cv2.setMouseCallback("CargaBot Digital Twin Dashboard", self._mouse_event)

    def _mouse_event(self, event, x, y, flags, param):
        scale  = getattr(self, '_display_scale', 1.0)
        px_cm  = self.perception.px_per_cm
        x_cm   = (x / scale) / px_cm
        y_cm   = (y / scale) / px_cm

        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_origin = (x_cm, y_cm)
            self.planned_path   = []
            self.get_logger().info(f"📍 Origen manual: ({x_cm:.1f}, {y_cm:.1f}) cm")

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.current_goal = (x_cm, y_cm)
            self.planned_path = []
            self.get_logger().info(f"🎯 Meta manual:   ({x_cm:.1f}, {y_cm:.1f}) cm")

    # ── Bucle principal ────────────────────────────────────────────────────────

    def control_loop(self):

        # 0. Frame
        if self.use_sim:
            if self.latest_frame is None:
                return
            frame = self.latest_frame.copy()
        else:
            ret, frame = self.cam.read()
            if not ret:
                return

        # 1. Transformación cenital (no-op en sim_mode)
        warped  = self.perception.warp_to_overhead(frame)
        display = warped.copy()

        # 2. Segmentación y cost map
        layers = self.perception.extract_semantic_layers(warped)
        self.navigator.generate_cost_map(layers['obstacles'])

        # 3. Localización del robot
        if self.use_sim:
            # En simulación: pose viene de /odom (precisa), o del ArUco si el
            # simulador lo renderiza correctamente.
            robot_pose = self._odom_pose
            # Fallback: intentar ArUco igualmente (útil si se mejora el render)
            if robot_pose is None:
                robot_pose = self.perception.detect_robot_pose(warped)
        else:
            robot_pose = self.perception.detect_robot_pose(warped)

        # 3a. HUD del robot
        if robot_pose:
            rx, ry, ryaw = robot_pose
            r_px = (
                int(rx * self.perception.px_per_cm),
                int(ry * self.perception.px_per_cm),
            )
            radius_px = int(self.navigator.robot_radius_cm * self.perception.px_per_cm)
            cv2.circle(display, r_px, radius_px, (0, 255, 255), 2)
            dx = int(np.cos(ryaw) * 40)
            dy = int(np.sin(ryaw) * 40)
            cv2.line(display, r_px, (r_px[0] + dx, r_px[1] + dy), (0, 0, 255), 3)

        # 4. Planificación
        start_pose = self.current_origin if self.current_origin else (
            robot_pose[:2] if robot_pose else None
        )

        # ── Diagnóstico ────────────────────────────────────────────────────
        free_cells  = int(np.sum(self.navigator.cost_map == 0))
        total_cells = self.navigator.cost_map.size
        if not hasattr(self, '_diag_counter'):
            self._diag_counter = 0
        self._diag_counter += 1
        if self._diag_counter % 20 == 0:
            self.get_logger().info(
                f"🗺️  Cost map: {free_cells}/{total_cells} libres "
                f"({100*free_cells//total_cells}%) | "
                f"robot_pose={'OK' if robot_pose else 'NO DETECTADO'} | "
                f"origin={self.current_origin} | goal={self.current_goal}"
            )
        if start_pose and self.current_goal:
            sg = self.navigator.cm_to_grid(*start_pose)
            gg = self.navigator.cm_to_grid(*self.current_goal)
            sc = int(self.navigator.cost_map[sg[1], sg[0]])
            gc = int(self.navigator.cost_map[gg[1], gg[0]])
            if self._diag_counter % 20 == 0:
                self.get_logger().info(
                    f"🔍 start_grid={sg} costo={sc} | "
                    f"goal_grid={gg} costo={gc} | "
                    f"{'⚠️  BLOQUEADO' if sc > 0 or gc > 0 else '✅ celdas libres, A* puede correr'}"
                )

        # ── Cost map visual ────────────────────────────────────────────────
        cost_vis   = self._build_cost_map_vis(start_pose, self.current_goal)
        cost_h, cost_w = cost_vis.shape[:2]
        cost_scale = min(MAX_DISPLAY_W / cost_w, MAX_DISPLAY_H / cost_h, 1.0)
        cv2.imshow(
            "CargaBot Cost Map",
            cv2.resize(cost_vis,
                       (int(cost_w * cost_scale), int(cost_h * cost_scale)),
                       interpolation=cv2.INTER_NEAREST),
        )

        # Marcadores de origen y meta
        if self.current_origin:
            ox_px = int(self.current_origin[0] * self.perception.px_per_cm)
            oy_px = int(self.current_origin[1] * self.perception.px_per_cm)
            cv2.drawMarker(display, (ox_px, oy_px), (255, 0, 255),
                           cv2.MARKER_SQUARE, 14, 2)

        if self.current_goal:
            gx_px = int(self.current_goal[0] * self.perception.px_per_cm)
            gy_px = int(self.current_goal[1] * self.perception.px_per_cm)
            cv2.drawMarker(display, (gx_px, gy_px), (0, 215, 255),
                           cv2.MARKER_STAR, 20, 2)

        if self.current_goal and start_pose:
            if not self.planned_path:
                self.planned_path = self.navigator.astar_search(
                    start_pose, self.current_goal
                )
                self.get_logger().info(
                    f"{'✅ A* encontró' if self.planned_path else '❌ A* falló —'} "
                    f"{len(self.planned_path)} waypoints"
                )
                self._publish_ros_path(self.planned_path)

            for i in range(len(self.planned_path) - 1):
                p1 = (int(self.planned_path[i][0]     * self.perception.px_per_cm),
                      int(self.planned_path[i][1]     * self.perception.px_per_cm))
                p2 = (int(self.planned_path[i + 1][0] * self.perception.px_per_cm),
                      int(self.planned_path[i + 1][1] * self.perception.px_per_cm))
                cv2.line(display, p1, p2, (255, 0, 0), 3)

            if robot_pose:
                target_cmd = self.navigator.calculate_progressive_command(
                    robot_pose,
                    self.planned_path,
                    self.get_parameter('lookahead_dist_cm').value,
                    self.get_parameter('max_lateral_drift_cm').value,
                )
                if target_cmd:
                    self._publish_goto_cmd(target_cmd)
                    t_px = (int(target_cmd[0] * self.perception.px_per_cm),
                            int(target_cmd[1] * self.perception.px_per_cm))
                    cv2.circle(display, t_px, 8, (0, 255, 0), -1)

        # 5. Visualización
        h, w = display.shape[:2]
        self._display_scale = min(MAX_DISPLAY_W / w, MAX_DISPLAY_H / h, 1.0)
        display_resized = cv2.resize(
            display,
            (int(w * self._display_scale), int(h * self._display_scale)),
            interpolation=cv2.INTER_AREA,
        )
        cv2.imshow("CargaBot Digital Twin Dashboard", display_resized)
        cv2.waitKey(1)

        img_msg = self.bridge.cv2_to_imgmsg(display, encoding='bgr8')
        self.pub_debug.publish(img_msg)

    # ── Helpers de visualización ───────────────────────────────────────────────

    def _build_cost_map_vis(self,
                             start_pose: Optional[Tuple[float, float]],
                             goal:       Optional[Tuple[float, float]]) -> np.ndarray:
        cm  = self.navigator.cost_map
        vis = np.zeros((cm.shape[0], cm.shape[1], 3), dtype=np.uint8)
        vis[cm > 0] = (30, 30, 180)

        for x_cm, y_cm in self.planned_path:
            gx, gy = self.navigator.cm_to_grid(x_cm, y_cm)
            vis[gy, gx] = (200, 80, 0)

        if start_pose:
            sg    = self.navigator.cm_to_grid(*start_pose)
            color = (0, 60, 255) if cm[sg[1], sg[0]] > 0 else (255, 0, 255)
            cv2.drawMarker(vis, sg, color, cv2.MARKER_SQUARE, 6, 1)

        if goal:
            gg    = self.navigator.cm_to_grid(*goal)
            color = (0, 60, 255) if cm[gg[1], gg[0]] > 0 else (0, 215, 255)
            cv2.drawMarker(vis, gg, color, cv2.MARKER_STAR, 8, 1)

        # Escalar al tamaño de la pista manteniendo proporción de celdas
        scale_x = max(int(self.navigator.res * self.perception.px_per_cm), 1)
        scale_y = max(int(self.navigator.res * self.perception.px_per_cm), 1)
        vis_up  = cv2.resize(
            vis,
            (vis.shape[1] * scale_x, vis.shape[0] * scale_y),
            interpolation=cv2.INTER_NEAREST,
        )
        free_pct = 100 * int(np.sum(cm == 0)) // cm.size
        cv2.putText(vis_up, f"Libres: {free_pct}%  |  Clic izq=origen  Clic der=meta",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return vis_up

    # ── Helpers de publicación ─────────────────────────────────────────────────

    def _publish_ros_path(self, path_cm: List[Tuple[float, float]]):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp    = self.get_clock().now().to_msg()
        for x, y in path_cm:
            p = PoseStamped()
            p.header = msg.header
            p.pose.position.x = x / 100.0
            p.pose.position.y = y / 100.0
            msg.poses.append(p)
        self.pub_path.publish(msg)

    def _publish_goto_cmd(self, target_cm: Tuple[float, float]):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.pose.position.x = target_cm[0] / 100.0
        msg.pose.position.y = target_cm[1] / 100.0
        self.pub_cmd_goto.publish(msg)

    # ── Teardown ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        if not self.use_sim:
            self.cam.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OverheadCoordinatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()