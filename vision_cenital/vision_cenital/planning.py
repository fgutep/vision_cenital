"""
Motor Algorítmico de Mapeo de Costos y Planificación de Rutas.
Implementa grillas discretas de 5x5 cm, búsqueda A* y lógicas heurísticas
de reconvergencia progresiva de trayectorias.
"""

import cv2
import numpy as np
import heapq
from typing import List, Tuple, Optional

class GridNavigator:
    def __init__(self, pista_w_cm: float, pista_h_cm: float, grid_res_cm: float = 5.0, robot_radius_cm: float = 15.0):
        self.res = grid_res_cm
        self.cols = int(np.ceil(pista_w_cm / self.res))
        self.rows = int(np.ceil(pista_h_cm / self.res))
        self.robot_radius_cm = robot_radius_cm
        self.cost_map = np.zeros((self.rows, self.cols), dtype=np.uint8)

    def generate_cost_map(self, raw_obstacle_mask: np.ndarray) -> np.ndarray:
        """
        Reduce la resolución de la capa métrica a celdas de 5 cm x 5 cm
        y aplica la dilatación de seguridad del radio del robot.
        """
        # Bajar escala preservando bordes críticos
        downsampled = cv2.resize(raw_obstacle_mask, (self.cols, self.rows), interpolation=cv2.INTER_AREA)
        _, binary_grid = cv2.threshold(downsampled, 127, 255, cv2.THRESH_BINARY)
        
        # Cálculo del kernel de dilatación en unidades de grilla
        radius_grid = int(np.ceil(self.robot_radius_cm / self.res))
        kernel_size = (radius_grid * 2) + 1
        # Kernel circular para una expansión isotrópica perfecta
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        self.cost_map = cv2.dilate(binary_grid, kernel, iterations=1)
        return self.cost_map

    def cm_to_grid(self, x_cm: float, y_cm: float) -> Tuple[int, int]:
        """Convierte coordenadas continuas a índices discretos de la matriz."""
        gx = int(np.clip(x_cm // self.res, 0, self.cols - 1))
        gy = int(np.clip(y_cm // self.res, 0, self.rows - 1))
        return gx, gy

    def grid_to_cm(self, gx: int, gy: int) -> Tuple[float, float]:
        """Devuelve el centro físico en cm de una celda de la grilla."""
        x_cm = (gx * self.res) + (self.res / 2.0)
        y_cm = (gy * self.res) + (self.res / 2.0)
        return x_cm, y_cm

    def astar_search(self, start_cm: Tuple[float, float], goal_cm: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Implementa A* optimizado en 8 direcciones sobre la matriz de costos."""
        start = self.cm_to_grid(*start_cm)
        goal = self.cm_to_grid(*goal_cm)

        if self.cost_map[start[1], start[0]] > 0 or self.cost_map[goal[1], goal[0]] > 0:
            # Origen o destino bloqueados por un obstáculo
            return []

        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
        close_set = set()
        came_from = {}
        
        # Búfer de costos g(n): distancia acumulada
        gscore = {start: 0.0}
        # Buffer F(n) = g(n) + h(n). Cola de prioridad: (fscore, (gx, gy))
        oheap = []
        heapq.heappush(oheap, (self._heuristic(start, goal), start))

        while oheap:
            _, current = heapq.heappop(oheap)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(self.grid_to_cm(*current))
                    current = came_from[current]
                path.append(self.grid_to_cm(*start))
                return path[::-1]

            close_set.add(current)

            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)

                if not (0 <= neighbor[0] < self.cols and 0 <= neighbor[1] < self.rows):
                    continue
                if self.cost_map[neighbor[1], neighbor[0]] > 0 or neighbor in close_set:
                    continue

                # Penalizar movimiento diagonal para reflejar distancia real
                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_gscore = gscore[current] + move_cost

                if neighbor not in gscore or tentative_gscore < gscore[neighbor]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_gscore
                    fscore = tentative_gscore + self._heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore, neighbor))

        return [] # Camino imposible de resolver

    @staticmethod
    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Distancia de Octil (óptima para movimientos conectados en 8-direcciones)."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return 1.0 * (dx + dy) + (1.414 - 2.0) * min(dx, dy)

    def calculate_progressive_command(self, current_pose: Tuple[float, float, float], 
                                      planned_path: List[Tuple[float, float]], 
                                      lookahead_cm: float = 20.0, 
                                      tolerance_cm: float = 8.0) -> Optional[Tuple[float, float]]:
        """
        Calcula un punto de avance (Go-to) progresivo. Si la trayectoria del robot
        difiere del plan más allá de la tolerancia permitida, el comando retrocede
        suavemente para re-acoplar el robot al trayecto original.
        """
        if not planned_path or len(planned_path) < 2:
            return None

        rx, ry, _ = current_pose
        path_arr = np.array(planned_path)
        
        # Encontrar el índice del punto más cercano al robot en la ruta planeada
        dists = np.linalg.norm(path_arr - np.array([rx, ry]), axis=1)
        closest_idx = int(np.argmin(dists))
        lateral_error = dists[closest_idx]

        # Si el robot supera la tolerancia de deriva, obligarlo a converger más cerca
        effective_lookahead = lookahead_cm if lateral_error <= tolerance_cm else max(5.0, lookahead_cm - lateral_error)
        
        # Buscar progresivamente hacia adelante en el array hasta cumplir la distancia lookahead
        target_idx = closest_idx
        accumulated_dist = 0.0
        
        for i in range(closest_idx, len(path_arr) - 1):
            accumulated_dist += np.linalg.norm(path_arr[i+1] - path_arr[i])
            if accumulated_dist >= effective_lookahead:
                target_idx = i + 1
                break
        else:
            target_idx = len(path_arr) - 1 # Extremo final de la ruta

        return float(path_arr[target_idx][0]), float(path_arr[target_idx][1])