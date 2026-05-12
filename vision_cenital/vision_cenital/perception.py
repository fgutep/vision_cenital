"""
Motor de Procesamiento Espacial Cenital.
Gestiona correcciones de lente, transformaciones top-down, segmentación semántica
y la estimación de pose global basada en ArUco (Compatible con OpenCV 4.6.0 y 4.7+).
"""

import cv2
import numpy as np
import yaml
from typing import Dict, Tuple, Optional


class OverheadPerception:
    def __init__(self, camera_params_path: str, homography_path: str):
        self.K, self.D, self.img_w, self.img_h = self._load_intrinsics(camera_params_path)
        self.H, self.pista_w_cm, self.pista_h_cm, self.px_per_cm = self._load_homography(homography_path)
        
        # Pre-computar mapas de corrección geométrica para máxima velocidad de bucle
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            self.K, self.D, None, self.K, (self.img_w, self.img_h), cv2.CV_32FC1
        )
        
        # Dimensiones de salida de la imagen con vista de pájaro
        self.warped_w = int(self.pista_w_cm * self.px_per_cm)
        self.warped_h = int(self.pista_h_cm * self.px_per_cm)
        
        # Ajustar H escalada a píxeles de salida
        self.H_scaled = self.H.copy()
        self.H_scaled[0] *= self.px_per_cm
        self.H_scaled[1] *= self.px_per_cm

        # ── INICIALIZACIÓN DE ARUCO (Agnóstica a la versión de OpenCV) ──
        # Compatibilidad con diccionarios en OpenCV <= 4.6 vs >= 4.7
        if hasattr(cv2.aruco, 'Dictionary_get'):
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        else:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            
        # Compatibilidad con parámetros
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        else:
            self.aruco_params = cv2.aruco.DetectorParameters()

        # Bandera para saber si usamos la API moderna (4.7+) o la clásica (4.6.0)
        self.has_new_api = hasattr(cv2.aruco, 'ArucoDetector')
        if self.has_new_api:
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    @staticmethod
    def _load_intrinsics(path: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
        with open(path, 'r') as f:
            d = yaml.safe_load(f)
        K = np.array(d['camera_matrix']['data']).reshape((3, 3))
        D = np.array(d['dist_coeffs']['data'])
        return K, D, d['image_width'], d['image_height']

    @staticmethod
    def _load_homography(path: str) -> Tuple[np.ndarray, float, float, int]:
        with open(path, 'r') as f:
                    d = yaml.safe_load(f)
        H = np.array(d['H']).reshape((3, 3))
        w = float(d.get('pista_w_cm', d.get('pista_size_cm', 200.0)))
        h = float(d.get('pista_h_cm', d.get('pista_size_cm', 200.0)))
        # Forzar entero estricto aquí para prevenir errores en remap/warp
        px_cm = int(float(d.get('px_per_cm', d.get('scale_px_per_cm', 5))))
        return H, w, h, px_cm

    def warp_to_overhead(self, frame: np.ndarray) -> np.ndarray:
        """Aplica corrección de lente y transformación proyectiva a vista de pájaro."""
        undistorted = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
        return cv2.warpPerspective(undistorted, self.H_scaled, (self.warped_w, self.warped_h))

    def detect_robot_pose(self, warped_frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Localiza el ArUco de 112 mm en el plano de vista de pájaro.
        Retorna (X_cm, Y_cm, Yaw_rad).
        """
        # Ejecutar detección según la API disponible en el sistema
        if self.has_new_api:
            corners, ids, _ = self.aruco_detector.detectMarkers(warped_frame)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(warped_frame, self.aruco_dict, parameters=self.aruco_params)
            
        if ids is not None and len(ids) > 0:
            c = corners[0][0]
            center_px = np.mean(c, axis=0)
            
            x_cm = center_px[0] / self.px_per_cm
            y_cm = center_px[1] / self.px_per_cm
            
            front_vec = ((c[0] + c[1]) / 2.0) - ((c[2] + c[3]) / 2.0)
            yaw_rad = np.arctan2(front_vec[1], front_vec[0])
            return float(x_cm), float(y_cm), float(yaw_rad)
        return None

    def extract_semantic_layers(self, warped_frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extrae máscaras puras para obstáculos negros y cubos de colores de 15x15 cm.
        """
        hsv = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2HSV)
        layers = {}

        # 1. Capa de Obstáculos (Negro intenso)
        gray = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
        _, thresh_black = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY_INV)
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        layers['obstacles'] = cv2.morphologyEx(thresh_black, cv2.MORPH_OPEN, kernel_clean)

        # 2. Capas de Cubos RGB (Objetivos de 15x15 cm)
        target_area_px = (15.0 * self.px_per_cm) ** 2
        min_area = target_area_px * 0.55
        max_area = target_area_px * 1.55

        color_ranges = {
            'cube_red': [(0, 130, 70), (10, 255, 255), (170, 130, 70), (180, 255, 255)],
            'cube_green': [(35, 100, 70), (85, 255, 255)],
            'cube_blue': [(100, 120, 70), (140, 255, 255)]
        }

        for name, ranges in color_ranges.items():
            if len(ranges) == 4:
                mask1 = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))
                mask2 = cv2.inRange(hsv, np.array(ranges[2]), np.array(ranges[3]))
                mask = mask1 | mask2
            else:
                mask = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))
            
            clean_mask = np.zeros_like(mask)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if min_area < cv2.contourArea(cnt) < max_area:
                    cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
            layers[name] = clean_mask

        return layers