"""
Herramienta de Auditoría Cenital Independiente (Standalone App).
Permite depurar el backend de visión cenital y A* sin iniciar la capa ROS 2.
Admite inyección de imágenes estáticas de archivo o conexión en vivo.
"""

import cv2
import numpy as np
import argparse
import sys
import os

# Ajustar path de ejecución para cargar librerías locales directamente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vision_cenital.perception import OverheadPerception
from vision_cenital.planning import GridNavigator
from vision_cenital.camera import CargaCam


class StandaloneMonitor:
    def __init__(self, args):
        self.args = args
        self.perception = OverheadPerception(args.params, args.homography)
        self.navigator = GridNavigator(
            self.perception.pista_w_cm, self.perception.pista_h_cm, args.grid_res, args.robot_radius
        )
        
        self.win_name = "CargaBot Standalone Planner Dashboard"
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_name, 1280, 720)
        cv2.setMouseCallback(self.win_name, self._mouse_event)

        self.start_cm = None
        self.goal_cm = None
        self.active_path = []

    def _mouse_event(self, event, x, y, flags, param):
        """Asigna punto de origen (Clic Izquierdo) o destino (Clic Derecho)."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_cm = (x / self.perception.px_per_cm, y / self.perception.px_per_cm)
            print(f"📍 Origen Manual fijado: {self.start_cm[0]:.1f}, {self.start_cm[1]:.1f} cm")
            self._update_plan()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.goal_cm = (x / self.perception.px_per_cm, y / self.perception.px_per_cm)
            print(f"🎯 Destino Manual fijado: {self.goal_cm[0]:.1f}, {self.goal_cm[1]:.1f} cm")
            self._update_plan()

    def _update_plan(self):
        if self.start_cm and self.goal_cm:
            self.active_path = self.navigator.astar_search(self.start_cm, self.goal_cm)
            if not self.active_path:
                print("⚠️ Planificación fallida: Ruta obstruida.")

    def run_image_mode(self, img_path: str):
        """Modo estático: Ejecuta auditoría visual sobre una foto de archivo."""
        raw = cv2.imread(img_path)
        if raw is None:
            print(f"❌ Imposible abrir foto: {img_path}")
            return

        warped = self.perception.warp_to_overhead(raw)
        layers = self.perception.extract_semantic_layers(warped)
        self.navigator.generate_cost_map(layers['obstacles'])

        # Autolocalizar si la foto contiene el robot físicamente
        robot_pose = self.perception.detect_robot_pose(warped)
        if robot_pose:
            self.start_cm = robot_pose[:2]
            print(f"🤖 ArUco detectado como origen automático: {self.start_cm}")

        while True:
            display = warped.copy()
            self._render_hud(display, layers, robot_pose)
            
            cv2.imshow(self.win_name, display)
            key = cv2.waitKey(30) & 0xFF
            if key == 27: # ESC
                break
            elif key == ord('r') or key == ord('R'):
                self.start_cm = None
                self.goal_cm = None
                self.active_path = []
                print("🧹 Waypoints reiniciados.")

        cv2.destroyAllWindows()

    def run_live_mode(self):
        """Modo dinámico: Ejecuta la tubería de hardware completa en el escritorio."""
        with CargaCam() as cam:
            while True:
                ret, frame = cam.read()
                if not ret:
                    continue

                warped = self.perception.warp_to_overhead(frame)
                layers = self.perception.extract_semantic_layers(warped)
                self.navigator.generate_cost_map(layers['obstacles'])

                robot_pose = self.perception.detect_robot_pose(warped)
                if robot_pose:
                    self.start_cm = robot_pose[:2]
                    self._update_plan()

                display = warped.copy()
                self._render_hud(display, layers, robot_pose)

                cv2.imshow(self.win_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break

        cv2.destroyAllWindows()

    def _render_hud(self, display, layers, robot_pose):
        """Inyecta interfaces e información semántica sobre el fotograma."""
        # ── 1. RENDERIZADO DE LA GRILLA DE COSTOS (Método Infalible OpenCV/NumPy) ──
        grid_overlay = cv2.resize(
            self.navigator.cost_map, (display.shape[1], display.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        
        # Creamos una imagen roja del mismo tamaño y mezclamos el 100% del frame de forma segura
        color_mask = np.zeros_like(display)
        color_mask[:, :] = (0, 0, 255) # Rojo
        blended_full = cv2.addWeighted(display, 0.7, color_mask, 0.3, 0)
        
        # Copiamos de vuelta solo los píxeles donde hay costo
        obstacle_indices = grid_overlay > 0
        display[obstacle_indices] = blended_full[obstacle_indices]

        # ── 2. DIBUJADO DE CAJAS DE CUBOS RGB ──
        colors = {'cube_red': (0, 0, 255), 'cube_green': (0, 255, 0), 'cube_blue': (255, 0, 0)}
        for name, mask in layers.items():
            if name == 'obstacles': 
                continue
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(display, (x, y), (x+w, y+h), colors[name], 3)
                cv2.putText(display, name.split('_')[1].upper(), (x, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[name], 2)

        # ── 3. DIBUJADO DEL CAMINO PLANEADO (A*) ──
        if self.active_path:
            for i in range(len(self.active_path) - 1):
                p1 = (int(self.active_path[i][0] * self.perception.px_per_cm), 
                      int(self.active_path[i][1] * self.perception.px_per_cm))
                p2 = (int(self.active_path[i+1][0] * self.perception.px_per_cm), 
                      int(self.active_path[i+1][1] * self.perception.px_per_cm))
                cv2.line(display, p1, p2, (255, 255, 0), 3)

            # Dibujar Punto Heurístico Lookahead
            if robot_pose:
                lookahead_cmd = self.navigator.calculate_progressive_command(robot_pose, self.active_path)
                if lookahead_cmd:
                    t_px = (int(lookahead_cmd[0] * self.perception.px_per_cm), 
                            int(lookahead_cmd[1] * self.perception.px_per_cm))
                    cv2.circle(display, t_px, 10, (0, 255, 0), -1)

        # ── 4. HUD TEXTUAL SUPERIOR ──
        cv2.putText(display, "L-Click: Fijar Origen | R-Click: Fijar Meta | 'R': Limpiar", 
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Módulo de Auditoría de Planificación de CargaBot")
    parser.add_argument('--params', type=str, default='resource/camera_params.yaml')
    parser.add_argument('--homography', type=str, default='resource/homography.yaml')
    parser.add_argument('--image', type=str, default='', help="Ruta a una imagen estática de pruebas offline")
    parser.add_argument('--grid-res', type=float, default=5.0, help="Tamaño métrico de celda en cm")
    parser.add_argument('--robot-radius', type=float, default=15.0, help="Radio de seguridad en cm")
    args = parser.parse_args()

    app = StandaloneMonitor(args)
    if args.image:
        print(f"Modo Estático activado sobre archivo: {args.image}")
        app.run_image_mode(args.image)
    else:
        print("Modo Transmisión Dinámica activado sobre GStreamer/GPU interna.")
        app.run_live_mode()