#!/usr/bin/env python3
"""
Motor del Mundo Virtual (Digital Twin) para CargaBot.
Simula la física, la odometría con ruido y el feed de video de la cámara cenital.

FIX: El frame publicado ahora es directamente el espacio cenital (pista sin márgenes,
escala px_per_cm consistente con OverheadPerception). Esto elimina la necesidad de
aplicar homografía sobre un frame sintético, que nunca fue calibrada para este renderer.
"""

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


# ─── Constantes de la pista (deben coincidir con los parámetros del coordinator) ──
PISTA_W_CM  = 408.0
PISTA_H_CM  = 206.0
PX_PER_CM   = 2.5          # resolución del frame cenital publicado
FRAME_W     = int(PISTA_W_CM * PX_PER_CM)   # 1020 px
FRAME_H     = int(PISTA_H_CM * PX_PER_CM)   # 515 px

# Color de fondo (gris claro, simula piso de laboratorio)
# IMPORTANTE: no usar blanco puro ni negro — evitar falsos positivos en HSV
BG_COLOR    = (180, 180, 180)

# Colores de objetos en BGR — calibrados para que extract_semantic_layers los detecte
# (ajusta estos si tus rangos HSV en perception.py son distintos)
COLOR_OBSTACLE_WALL = (40,  40,  40)   # gris muy oscuro → negro → obstáculo
COLOR_CUBE_RED      = (0,   0,  200)   # rojo puro
COLOR_CUBE_GREEN    = (0,  200,   0)   # verde puro
COLOR_CUBE_BLUE     = (200,  0,   0)   # azul puro
COLOR_ROBOT_WHITE   = (255, 255, 255)  # borde ArUco blanco
COLOR_ROBOT_BLACK   = (20,   20,  20)  # interior ArUco negro
COLOR_ROBOT_FWD     = (0,    0, 255)   # flecha de frente rojo


class VirtualWorldEngine(Node):
    def __init__(self):
        super().__init__("virtual_world_engine")

        # ─── Comunicaciones ROS 2 ────────────────────────────────────────
        self.sub_cmd = self.create_subscription(
            Twist, 'turtlebot_cmdVel', self._cmd_cb, 10
        )
        self.pub_odom = self.create_publisher(Odometry, '/odom', 10)
        self.pub_cam  = self.create_publisher(Image, '/cargabot/camera/image_raw', 10)
        self.bridge   = CvBridge()

        # ─── Estado físico real (metros) ────────────────────────────────
        self.x     = 0.5    # 50 cm desde el borde izquierdo
        self.y     = 0.5    # 50 cm desde el borde superior
        self.theta = 0.0
        self.v     = 0.0
        self.w     = 0.0

        # ─── Estado de odometría (lo que el robot "cree") ───────────────
        self.odom_x     = 0.5
        self.odom_y     = 0.5
        self.odom_theta = 0.0

        # ─── Parámetros de física ────────────────────────────────────────
        self.slip_factor = 0.95   # 5% pérdida de tracción
        self.imu_noise   = 0.01   # ruido en giro

        # ─── Obstáculos estáticos (x_cm, y_cm, w_cm, h_cm) ─────────────
        # Edita esta lista para agregar/mover obstáculos sin tocar nada más.
        self.obstacles = [
            (120.0, 40.0, 8.0, 126.0),   # Cinta vertical de exclusión
        ]

        # ─── Cubos (x_cm, y_cm, tipo) ────────────────────────────────────
        self.cubes = [
            (240.0, 80.0,  'red'),
            (320.0, 120.0, 'green'),
            (160.0, 140.0, 'blue'),
        ]

        self.timer = self.create_timer(0.05, self._physics_step)  # 20 Hz
        self.get_logger().info(
            f"🌍 Motor Virtual ONLINE — frame cenital {FRAME_W}×{FRAME_H} px "
            f"({PISTA_W_CM}×{PISTA_H_CM} cm @ {PX_PER_CM} px/cm)"
        )

    # ── ROS callbacks ────────────────────────────────────────────────────────

    def _cmd_cb(self, msg: Twist):
        self.v = msg.linear.x
        self.w = msg.angular.z

    # ── Bucle de física ──────────────────────────────────────────────────────

    def _physics_step(self):
        dt = 0.05

        # Cinemática real con derrape y ruido
        actual_v = self.v * self.slip_factor
        actual_w = self.w + random.gauss(0, self.imu_noise * abs(self.w) + 1e-9)
        self.x     += actual_v * math.cos(self.theta) * dt
        self.y     += actual_v * math.sin(self.theta) * dt
        self.theta += actual_w * dt
        self.theta  = math.atan2(math.sin(self.theta), math.cos(self.theta))

        # Limitar posición a los bordes de la pista (m)
        self.x = float(np.clip(self.x, 0.06, PISTA_W_CM / 100 - 0.06))
        self.y = float(np.clip(self.y, 0.06, PISTA_H_CM / 100 - 0.06))

        # Odometría perfecta (integración sin derrape)
        self.odom_x     += self.v * math.cos(self.odom_theta) * dt
        self.odom_y     += self.v * math.sin(self.odom_theta) * dt
        self.odom_theta += self.w * dt
        self.odom_theta  = math.atan2(math.sin(self.odom_theta), math.cos(self.odom_theta))

        self._publish_odom()
        self._render_and_publish_camera()

    # ── Publicación ──────────────────────────────────────────────────────────

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
        Genera el frame cenital sintético directamente en coordenadas de pista.
        Sin offset, sin homografía pendiente — lo que publica aquí es lo que
        OverheadPerception.warp_to_overhead() devolvería con la cámara real.
        """
        # ── Fondo: piso de lab con ruido leve ────────────────────────────
        frame = np.full((FRAME_H, FRAME_W, 3), BG_COLOR, dtype=np.uint8)
        noise = np.random.randint(0, 8, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
        frame = cv2.subtract(frame, noise)

        # ── Paredes de la pista (borde del frame = borde de la pista) ────
        cv2.rectangle(frame, (0, 0), (FRAME_W - 1, FRAME_H - 1),
                      COLOR_OBSTACLE_WALL, 6)

        # ── Obstáculos estáticos ──────────────────────────────────────────
        for ox, oy, ow, oh in self.obstacles:
            x1 = int(ox * PX_PER_CM)
            y1 = int(oy * PX_PER_CM)
            x2 = int((ox + ow) * PX_PER_CM)
            y2 = int((oy + oh) * PX_PER_CM)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_OBSTACLE_WALL, -1)

        # ── Cubos ─────────────────────────────────────────────────────────
        cube_sz = int(15 * PX_PER_CM)  # 15 cm × 15 cm
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

        # ── Robot (ArUco sintético) ───────────────────────────────────────
        aruco_cm = 11.2
        aruco_px = int(aruco_cm * PX_PER_CM)
        rx_px    = int(self.x * 100 * PX_PER_CM)
        ry_px    = int(self.y * 100 * PX_PER_CM)

        # Cuadrado rotado según theta
        rect = ((rx_px, ry_px), (aruco_px, aruco_px), math.degrees(self.theta))
        box  = cv2.boxPoints(rect).astype(np.int32)
        cv2.drawContours(frame, [box], 0, COLOR_ROBOT_WHITE, -1)
        cv2.drawContours(frame, [box], 0, COLOR_ROBOT_BLACK, max(1, int(aruco_px * 0.3)))

        # Flecha de frente
        fx = int(rx_px + math.cos(self.theta) * (aruco_px / 1.5))
        fy = int(ry_px + math.sin(self.theta) * (aruco_px / 1.5))
        cv2.line(frame, (rx_px, ry_px), (fx, fy), COLOR_ROBOT_FWD, 2)

        # ── Publicar ──────────────────────────────────────────────────────
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.pub_cam.publish(img_msg)


def main():
    rclpy.init()
    node = VirtualWorldEngine()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()