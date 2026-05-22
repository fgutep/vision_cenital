#!/usr/bin/env python3
"""
Motor del Mundo Virtual (Digital Twin) para CargaBot.
Simula la física, la odometría con ruido y el feed de video de la cámara cenital.

FIX: El frame publicado es directamente el espacio cenital (pista sin márgenes,
escala px_per_cm consistente con OverheadPerception en sim_mode).

FIX ArUco: El robot ahora se renderiza con un marcador ArUco 4x4 ID=0 estático,
rotado según theta, evadiendo los problemas de versión de la librería cv2.aruco.
"""

import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import math
import random


# ─── Constantes de la pista ───────────────────────────────────────────────────
# CRÍTICO: PX_PER_CM debe coincidir con SIM_PX_PER_CM en perception.py
PISTA_W_CM  = 408.0
PISTA_H_CM  = 206.0
PX_PER_CM   = 2.5
FRAME_W     = int(PISTA_W_CM * PX_PER_CM)   # 1020 px
FRAME_H     = int(PISTA_H_CM * PX_PER_CM)   # 515 px

# Color de fondo (gris claro — no blanco ni negro para evitar falsos positivos)
BG_COLOR            = (180, 180, 180)

# Colores de objetos en BGR
COLOR_OBSTACLE_WALL = (40,  40,  40)
COLOR_CUBE_RED      = (0,   0,  200)
COLOR_CUBE_GREEN    = (0,  200,   0)
COLOR_CUBE_BLUE     = (200,  0,   0)
COLOR_ROBOT_FWD     = (0,    0, 255)   # flecha de dirección (rojo)

# Tamaño del marcador ArUco en cm (debe coincidir con el físico)
ARUCO_SIZE_CM = 11.2
ARUCO_ID      = 0    # Representativo, ya que cargamos la imagen estática


class VirtualWorldEngine(Node):
    def __init__(self):
        super().__init__("virtual_world_engine")

        # ─── Comunicaciones ROS 2 ────────────────────────────────────────
        self.sub_cmd  = self.create_subscription(
            Twist, 'turtlebot_cmdVel', self._cmd_cb, 10
        )
        self.pub_odom = self.create_publisher(Odometry, '/odom', 10)
        self.pub_cam  = self.create_publisher(Image, '/cargabot/camera/image_raw', 10)
        self.bridge   = CvBridge()

        # ─── Estado físico real (metros) ─────────────────────────────────
        self.x     = 0.5
        self.y     = 0.5
        self.theta = 0.0
        self.v     = 0.0
        self.w     = 0.0

        # ─── Estado de odometría (integración limpia) ─────────────────────
        self.odom_x     = 0.5
        self.odom_y     = 0.5
        self.odom_theta = 0.0

        # ─── Parámetros de física ─────────────────────────────────────────
        self.slip_factor = 0.95
        self.imu_noise   = 0.01

        # ─── Obstáculos estáticos (x_cm, y_cm, w_cm, h_cm) ───────────────
        self.obstacles = [
            (120.0, 40.0, 8.0, 126.0),   # Cinta vertical de exclusión
        ]

        # ─── Cubos (x_cm, y_cm, tipo) ────────────────────────────────────
        self.cubes = [
            (240.0,  80.0, 'red'),
            (320.0, 120.0, 'green'),
            (160.0, 140.0, 'blue'),
        ]

        # ─── Cargar imagen estática del marcador ArUco ────────────────────
        # Ruta absoluta "quemada" (hardcoded) directo a tu código fuente
        aruco_path = "/home/felipe/Desktop/Robotica_Proy/Vision_Cenital_Repo/vision_cenital/vision_cenital/vision_cenital/assets/aruco.png"
        
        self.get_logger().info(f"Intentando cargar ArUco desde: {aruco_path}")
        self._aruco_marker_img = self._load_aruco_asset(aruco_path)


        self.timer = self.create_timer(0.05, self._physics_step)  # 20 Hz
        self.get_logger().info(
            f"🌍 Motor Virtual ONLINE — frame cenital {FRAME_W}×{FRAME_H} px "
            f"({PISTA_W_CM}×{PISTA_H_CM} cm @ {PX_PER_CM} px/cm) | "
            f"ArUco estático ({ARUCO_SIZE_CM} cm)"
        )

    # ── Carga del marcador ArUco estático ─────────────────────────────────────

    def _load_aruco_asset(self, filepath: str) -> np.ndarray:
        """
        Carga el marcador ArUco desde un archivo de imagen estático.
        """
        aruco_px = int(ARUCO_SIZE_CM * PX_PER_CM)
        img = cv2.imread(filepath)

        if img is None:
            self.get_logger().error(f"¡No se pudo cargar {filepath}! Usando cuadrado magenta de fallback.")
            # Fallback brillante para que sepas inmediatamente si la ruta está mal
            return np.full((aruco_px, aruco_px, 3), (255, 0, 255), dtype=np.uint8)

        # Redimensionar la imagen estática a los píxeles calculados para la simulación
        return cv2.resize(img, (aruco_px, aruco_px), interpolation=cv2.INTER_AREA)

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def _cmd_cb(self, msg: Twist):
        self.v = msg.linear.x
        self.w = msg.angular.z

    # ── Bucle de física ───────────────────────────────────────────────────────

    def _physics_step(self):
        dt = 0.05

        actual_v = self.v * self.slip_factor
        actual_w = self.w + random.gauss(0, self.imu_noise * abs(self.w) + 1e-9)
        self.x     += actual_v * math.cos(self.theta) * dt
        self.y     += actual_v * math.sin(self.theta) * dt
        self.theta += actual_w * dt
        self.theta  = math.atan2(math.sin(self.theta), math.cos(self.theta))

        self.x = float(np.clip(self.x, 0.06, PISTA_W_CM / 100 - 0.06))
        self.y = float(np.clip(self.y, 0.06, PISTA_H_CM / 100 - 0.06))

        self.odom_x     += self.v * math.cos(self.odom_theta) * dt
        self.odom_y     += self.v * math.sin(self.odom_theta) * dt
        self.odom_theta += self.w * dt
        self.odom_theta  = math.atan2(math.sin(self.odom_theta), math.cos(self.odom_theta))

        self._publish_odom()
        self._render_and_publish_camera()

    # ── Publicación ───────────────────────────────────────────────────────────

    def _publish_odom(self):
        msg = Odometry()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.pose.pose.position.x = self.odom_x
        msg.pose.pose.position.y = self.odom_y
        msg.pose.pose.orientation.z = math.sin(self.odom_theta / 2.0)
        msg.pose.pose.orientation.w = math.cos(self.odom_theta / 2.0)
        self.pub_odom.publish(msg)

    def _render_and_publish_camera(self):
        """
        Genera el frame cenital sintético en coordenadas de pista.
        El robot se renderiza como un marcador ArUco cargado desde disco
        y rotado según theta.
        """
        # ── Fondo ─────────────────────────────────────────────────────────
        frame = np.full((FRAME_H, FRAME_W, 3), BG_COLOR, dtype=np.uint8)
        noise = np.random.randint(0, 8, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
        frame = cv2.subtract(frame, noise)

        # ── Paredes perimetrales ───────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (FRAME_W - 1, FRAME_H - 1),
                      COLOR_OBSTACLE_WALL, 6)

        # ── Obstáculos estáticos ───────────────────────────────────────────
        for ox, oy, ow, oh in self.obstacles:
            x1 = int(ox * PX_PER_CM)
            y1 = int(oy * PX_PER_CM)
            x2 = int((ox + ow) * PX_PER_CM)
            y2 = int((oy + oh) * PX_PER_CM)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_OBSTACLE_WALL, -1)

        # ── Cubos ─────────────────────────────────────────────────────────
        cube_sz = int(15 * PX_PER_CM)
        color_map = {
            'red':   COLOR_CUBE_RED,
            'green': COLOR_CUBE_GREEN,
            'blue':  COLOR_CUBE_BLUE,
        }
        for cx, cy, ctype in self.cubes:
            x1 = int(cx * PX_PER_CM)
            y1 = int(cy * PX_PER_CM)
            cv2.rectangle(frame, (x1, y1),
                          (x1 + cube_sz, y1 + cube_sz),
                          color_map[ctype], -1)

        # ── Robot: marcador estático rotado ───────────────────────────────
        aruco_px = int(ARUCO_SIZE_CM * PX_PER_CM)
        rx_px    = int(self.x * 100 * PX_PER_CM)
        ry_px    = int(self.y * 100 * PX_PER_CM)

        self._blit_rotated_marker(frame, self._aruco_marker_img, rx_px, ry_px, self.theta)

        # Flecha de dirección (ayuda visual, no afecta detección)
        # fx = int(rx_px + math.cos(self.theta) * (aruco_px / 1.5))
        # fy = int(ry_px + math.sin(self.theta) * (aruco_px / 1.5))
        # cv2.arrowedLine(frame, (rx_px, ry_px), (fx, fy), COLOR_ROBOT_FWD, 2, tipLength=0.3)

        # ── Publicar ───────────────────────────────────────────────────────
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.pub_cam.publish(img_msg)

    # ── Helper: pegar imagen rotada sobre frame (Anti-clipping) ───────────────

    @staticmethod
    def _blit_rotated_marker(frame: np.ndarray,
                              marker: np.ndarray,
                              cx: int, cy: int,
                              angle_rad: float) -> None:
        """
        Pega `marker` (imagen estática) centrada en (cx, cy) y rotada `angle_rad`.
        Calcula la nueva caja delimitadora para no recortar las esquinas de la imagen.
        """
        h, w = marker.shape[:2]
        center = (w // 2, h // 2)

        # Matriz de rotación
        M = cv2.getRotationMatrix2D(center, -math.degrees(angle_rad), 1.0)

        # Calcular nuevas dimensiones para que las esquinas no se recorten
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Ajustar la matriz de rotación para trasladar la imagen al nuevo centro
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Tomar el color de fondo para disimular los bordes del rotado
        if 0 <= cy < frame.shape[0] and 0 <= cx < frame.shape[1]:
            bg_color = tuple(int(c) for c in frame[cy, cx])
        else:
            bg_color = (0, 0, 0)

        rotated = cv2.warpAffine(marker, M, (new_w, new_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=bg_color)

        # Calcular ROI de destino con las nuevas dimensiones
        x1 = cx - new_w // 2
        y1 = cy - new_h // 2
        x2 = x1 + new_w
        y2 = y1 + new_h

        # Clamp
        fx1 = max(x1, 0);  fy1 = max(y1, 0)
        fx2 = min(x2, frame.shape[1]);  fy2 = min(y2, frame.shape[0])
        mx1 = fx1 - x1;  my1 = fy1 - y1
        mx2 = mx1 + (fx2 - fx1);  my2 = my1 + (fy2 - fy1)

        if fx2 <= fx1 or fy2 <= fy1:
            return

        frame[fy1:fy2, fx1:fx2] = rotated[my1:my2, mx1:mx2]


def main():
    rclpy.init()
    node = VirtualWorldEngine()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()