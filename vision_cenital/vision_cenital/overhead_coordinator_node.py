"""
Nodo ROS 2 de Coordinación Cenital.
Integra los motores de Percepción y Planificación, gestiona suscripciones de meta
y transmite comandos secuenciales progresivos a la base motriz de CargaBot.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import os
from ament_index_python.packages import get_package_share_directory
import numpy as np
import cv2

# Importaciones de los módulos del motor interno
from vision_cenital.camera import CargaCam
from vision_cenital.perception import OverheadPerception
from vision_cenital.planning import GridNavigator


class OverheadCoordinatorNode(Node):
    def __init__(self):
        super().__init__('overhead_coordinator_node')
        
        # Declarar parámetros configurables del nodo
        self.declare_parameter('pista_w_cm', 200.0)
        self.declare_parameter('pista_h_cm', 200.0)
        self.declare_parameter('grid_res_cm', 5.0)
        self.declare_parameter('robot_radius_cm', 15.0)
        self.declare_parameter('lookahead_dist_cm', 25.0)
        self.declare_parameter('max_lateral_drift_cm', 10.0)

        # Rutas de recursos de calibración
        pkg_share = get_package_share_directory('cargabot_overhead_vision')
        cam_params = os.path.join(pkg_share, 'resource', 'camera_params.yaml')
        homography = os.path.join(pkg_share, 'resource', 'homography.yaml')

        # Inicializar Motores Internos
        p_w = self.get_parameter('pista_w_cm').value
        p_h = self.get_parameter('pista_h_cm').value
        g_res = self.get_parameter('grid_res_cm').value
        r_rad = self.get_parameter('robot_radius_cm').value
        
        self.perception = OverheadPerception(cam_params, homography)
        self.navigator = GridNavigator(p_w, p_h, g_res, r_rad)
        self.cam = CargaCam()
        self.bridge = CvBridge()

        # Variables de estado del planificador
        self.current_goal: Optional[Tuple[float, float]] = None
        self.planned_path: List[Tuple[float, float]] = []

        # Tópicos de Comunicación
        self.sub_goal = self.create_subscription(
            PoseStamped, '/cargabot/goal_pose', self.goal_callback, 10
        )
        self.pub_path = self.create_publisher(Path, '/cargabot/global_path', 10)
        self.pub_cmd_goto = self.create_publisher(PoseStamped, '/cargabot/cmd_goto', 10)
        self.pub_debug = self.create_publisher(Image, '/cargabot/overhead_debug_video', 10)

        # Enlazar hardware e iniciar bucle principal a 10 Hz
        if self.cam.start():
            self.timer = self.create_timer(0.1, self.control_loop)
            self.get_logger().info("🚀 Sistema ROS 2 Cenital en línea y transmitiendo.")
        else:
            self.get_logger().error("❌ Imposible inicializar hardware de video.")

    def goal_callback(self, msg: PoseStamped):
        """Atrapa destinos enviados externamente y solicita re-planificación."""
        self.current_goal = (msg.pose.position.x * 100.0, msg.pose.position.y * 100.0) # a CM
        self.get_logger().info(f"🎯 Nueva meta de navegación fijada: {self.current_goal} cm")
        self.planned_path = [] # Forzar re-planificación limpia

    def control_loop(self):
        """Bucle de control principal del ciclo de vida del robot."""
        ret, frame = self.cam.read()
        if not ret:
            return

        # 1. Transformación Cenital
        warped = self.perception.warp_to_overhead(frame)
        display = warped.copy()

        # 2. Extracción Semántica y Mapeo
        layers = self.perception.extract_semantic_layers(warped)
        cost_grid = self.navigator.generate_cost_map(layers['obstacles'])

        # 3. Localización del Robot
        robot_pose = self.perception.detect_robot_pose(warped)
        
        # Superposición Visual HUD en el frame de depuración
        if robot_pose:
            rx, ry, ryaw = robot_pose
            r_px = (int(rx * self.perception.px_per_cm), int(ry * self.perception.px_per_cm))
            cv2.circle(display, r_px, int(self.navigator.robot_radius_cm * self.perception.px_per_cm), (0, 255, 255), 2)
            # Dibujar vector de orientación
            dx = int(np.cos(ryaw) * 40)
            dy = int(np.sin(ryaw) * 40)
            cv2.line(display, r_px, (r_px[0] + dx, r_px[1] + dy), (0, 0, 255), 3)

        # 4. Lógica A* y Emisión de Órdenes Go-To
        if self.current_goal and robot_pose:
            # Re-planificar dinámicamente si el camino anterior colapsa
            if not self.planned_path:
                self.planned_path = self.navigator.astar_search(robot_pose[:2], self.current_goal)
                self._publish_ros_path(self.planned_path)

            # Dibujar trazo de la ruta planeada en el HUD
            for i in range(len(self.planned_path) - 1):
                p1 = (int(self.planned_path[i][0] * self.perception.px_per_cm), int(self.planned_path[i][1] * self.perception.px_per_cm))
                p2 = (int(self.planned_path[i+1][0] * self.perception.px_per_cm), int(self.planned_path[i+1][1] * self.perception.px_per_cm))
                cv2.line(display, p1, p2, (255, 0, 0), 3)

            # Calcular el comando progresivo de corrección
            lookahead_cm = self.get_parameter('lookahead_dist_cm').value
            drift_limit_cm = self.get_parameter('max_lateral_drift_cm').value
            
            target_cmd = self.navigator.calculate_progressive_command(
                robot_pose, self.planned_path, lookahead_cm, drift_limit_cm
            )

            if target_cmd:
                self._publish_goto_cmd(target_cmd)
                t_px = (int(target_cmd[0] * self.perception.px_per_cm), int(target_cmd[1] * self.perception.px_per_cm))
                cv2.circle(display, t_px, 8, (0, 255, 0), -1) # Punto de acople verde en la ruta

        # Publicar depuración visual de la escena
        img_msg = self.bridge.cv2_to_imgmsg(display, encoding="bgr8")
        self.pub_debug.publish(img_msg)

    def _publish_ros_path(self, path_cm: List[Tuple[float, float]]):
        """Publica el camino completo de coordenadas en metros hacia RViz."""
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        
        for x, y in path_cm:
            p = PoseStamped()
            p.header = msg.header
            p.pose.position.x = x / 100.0 # a Metros
            p.pose.position.y = y / 100.0
            msg.poses.append(p)
        self.pub_path.publish(msg)

    def _publish_goto_cmd(self, target_cm: Tuple[float, float]):
        """Publica el punto específico al que debe avanzar el robot motrizmente."""
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = target_cm[0] / 100.0
        msg.pose.position.y = target_cm[1] / 100.0
        self.pub_cmd_goto.publish(msg)

    def destroy_node(self):
        self.cam.release()
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