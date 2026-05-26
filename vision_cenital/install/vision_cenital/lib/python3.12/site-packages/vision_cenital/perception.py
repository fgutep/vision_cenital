#!/usr/bin/env python3
"""
perception.py v7.1
===================
Fix v7.1 (vs v7):
  - Asymmetric hitbox: robot_hitbox_front_cm + robot_hitbox_back_cm
    (ArUco marker is offset, not centered on robot)
  - Center offset calculation: offset_cm moves the bbox center from ArUco to robot center
  - Improved exclusion mask with padding and dilation to eliminate false positives
  - Updated inject_pose_from_odom to account for ArUco offset
  - All rotated hitbox logic preserved from v7

Fix v7 (vs v6):
  - ROBOT_ARUCO_ID changed to 0
  - robot_bbox_px is now a rotated hitbox (cx, cy, hw, hh, yaw)
    instead of axis-aligned (x, y, w, h)
  - _robot_exclusion_mask draws a rotated filled polygon via fillPoly
  - Configurable hitbox half-dimensions via robot_hitbox_hw_cm
  - get_robot_hitbox_corners_px() for app-layer rotated rendering
  - Two-stage detection and bbox persistence from v6 intact
"""

import cv2
import numpy as np
import yaml
from typing import Dict, Tuple, Optional

SIM_PX_PER_CM:        float = 2.5
BLACK_THRESHOLD:      int   = 40
BORDER_MARGIN:        int   = 8
ROBOT_ARUCO_ID:       int   = 0
ARUCO_REAL_CM:        float = 15.0
OBSTACLE_DILATION_CM: float = 7.0

PROC_SCALE:          float = 0.5
COSTMAP_SKIP_FRAMES: int   = 4
ARUCO_DETECT_W:      int   = 1024   # ancho al que se reduce para detectar ArUco
BBOX_PERSIST_FRAMES: int   = 15     # frames que el bbox persiste sin detección


class OverheadPerception:
    def __init__(self, camera_params_path: str, homography_path: str,
                 sim_mode: bool = True):
        self.sim_mode = sim_mode

        self.K, self.D, self.img_w, self.img_h = self._load_intrinsics(camera_params_path)
        self.H, self.pista_w_cm, self.pista_h_cm, self._hw_px_per_cm = \
            self._load_homography(homography_path)

        self.px_per_cm: float = SIM_PX_PER_CM if sim_mode else float(self._hw_px_per_cm)
        self.warped_w = int(self.pista_w_cm * self.px_per_cm)
        self.warped_h = int(self.pista_h_cm * self.px_per_cm)

        if not self.sim_mode:
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                self.K, self.D, None, self.K,
                (self.img_w, self.img_h), cv2.CV_32FC1)
            self.H_scaled = self.H.copy()
            self.H_scaled[0] *= self.px_per_cm
            self.H_scaled[1] *= self.px_per_cm

        # ArUco
        if hasattr(cv2.aruco, 'Dictionary_get'):
            self.aruco_dict   = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.has_new_api  = False
        else:
            self.aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.has_new_api  = hasattr(cv2.aruco, 'ArucoDetector')
            if self.has_new_api:
                self.aruco_detector = cv2.aruco.ArucoDetector(
                    self.aruco_dict, self.aruco_params)

        # Parámetros ArUco más permisivos para imagen cenital
        if hasattr(self.aruco_params, 'adaptiveThreshWinSizeMin'):
            self.aruco_params.adaptiveThreshWinSizeMin  = 3
            self.aruco_params.adaptiveThreshWinSizeMax  = 53
            self.aruco_params.adaptiveThreshWinSizeStep = 10
            self.aruco_params.minMarkerPerimeterRate     = 0.02
            self.aruco_params.maxMarkerPerimeterRate     = 0.5
            # Muy permisivo: esquinas redondeadas por warpPerspective
            self.aruco_params.polygonalApproxAccuracyRate = 0.10
            self.aruco_params.cornerRefinementMethod     = \
                getattr(cv2.aruco, 'CORNER_REFINE_SUBPIX',
                        getattr(cv2.aruco, 'CORNER_REFINE_NONE', 0))
        if hasattr(self.aruco_params, 'perspectiveRemovePixelPerCell'):
            self.aruco_params.perspectiveRemovePixelPerCell = 3
        if hasattr(self.aruco_params, 'errorCorrectionRate'):
            self.aruco_params.errorCorrectionRate = 1.0

        self.robot_pose_cm:  Optional[Tuple[float, float, float]] = None
        # Rotated hitbox: (center_x_px, center_y_px, half_w_px, half_h_px, yaw_rad)
        self.robot_bbox_px:  Optional[Tuple[float, float, float, float, float]] = None
        self._bbox_age:      int = 0  # frames since last successful detection

        # Robot hitbox dimensions with offset ArUco (all in cm)
        # ArUco is NOT centered - it's closer to the back of the robot
        self.robot_hitbox_front_cm: float = 32   # distance from ArUco to robot front
        self.robot_hitbox_back_cm: float = 11.0     # distance from ArUco to robot back
        self.robot_hitbox_side_cm: float = 12.0    # half-width (side to side)

        # Precompute for compatibility with existing code
        self.robot_hitbox_hw_cm: Tuple[float, float] = (
            (self.robot_hitbox_front_cm + self.robot_hitbox_back_cm) / 2.0,  # half-length
            self.robot_hitbox_side_cm  # half-width
        )
        # Offset from ArUco center to robot geometric center (positive = forward)
        self.robot_hitbox_offset_cm: float = (
            self.robot_hitbox_front_cm - self.robot_hitbox_back_cm
        ) / 2.0

        dil_px = max(1, int(OBSTACLE_DILATION_CM * self.px_per_cm * PROC_SCALE))
        self._dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dil_px * 2 + 1, dil_px * 2 + 1))

        self._layers_cache:  Optional[Dict[str, np.ndarray]] = None
        self._costmap_frame: int = 0

        # Cache for warp_to_overhead result (reused by fallback detection)
        self._last_warped: Optional[np.ndarray] = None

    @staticmethod
    def _load_intrinsics(path):
        with open(path) as f: d = yaml.safe_load(f)
        K = np.array(d['camera_matrix']['data']).reshape((3, 3))
        D = np.array(d['dist_coeffs']['data'])
        return K, D, d['image_width'], d['image_height']

    @staticmethod
    def _load_homography(path):
        with open(path) as f: d = yaml.safe_load(f)
        H     = np.array(d['H']).reshape((3, 3))
        w     = float(d.get('pista_w_cm',  d.get('pista_size_cm', 408.0)))
        h     = float(d.get('pista_h_cm',  d.get('pista_size_cm', 206.0)))
        px_cm = int(float(d.get('px_per_cm', d.get('scale_px_per_cm', 5))))
        return H, w, h, px_cm

    def warp_to_overhead(self, frame: np.ndarray) -> np.ndarray:
        if self.sim_mode:
            self._last_warped = frame
            return frame
        undistorted = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
        warped = cv2.warpPerspective(undistorted, self.H_scaled,
                                      (self.warped_w, self.warped_h))
        self._last_warped = warped
        return warped

    # ── ArUco helpers ─────────────────────────────────────────────────────────

    def _detect_aruco_on(self, gray: np.ndarray, preprocess: bool = False):
        """
        Run ArUco detection on a grayscale image.
        Returns (corners, ids) or (None, None).
        """
        if preprocess:
            gray = self._preprocess_for_aruco(gray)

        if self.has_new_api:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None or len(ids) == 0:
            return None, None
        return corners, ids

    def _pick_robot_marker(self, ids):
        """Return index of preferred marker (ID 0), or 0 as fallback."""
        for i, mid in enumerate(ids.flatten()):
            if mid == ROBOT_ARUCO_ID:
                return i
        return 0

    def _corners_to_pose_and_bbox(self, corners_px):
        """
        Given 4 corner points already in overhead pixel coordinates,
        compute (x_cm, y_cm, yaw) and a rotated hitbox
        (cx_px, cy_px, half_w_px, half_h_px, yaw).
        
        Accounts for ArUco offset: the marker is not centered on the robot.
        """
        c = corners_px.reshape(4, 2)
        aruco_center_px = np.mean(c, axis=0)

        # Compute yaw from marker orientation
        fv   = ((c[0] + c[1]) / 2.0) - ((c[2] + c[3]) / 2.0)
        yaw  = float(np.arctan2(fv[1], fv[0]) + np.pi / 2)

        # Offset the robot center from ArUco center
        offset_cm = self.robot_hitbox_offset_cm
        robot_center_x = aruco_center_px[0] / self.px_per_cm + offset_cm * np.cos(yaw)
        robot_center_y = aruco_center_px[1] / self.px_per_cm + offset_cm * np.sin(yaw)

        x_cm = float(robot_center_x)
        y_cm = float(robot_center_y)

        # Hitbox half-dimensions
        hw_px = self.robot_hitbox_hw_cm[0] * self.px_per_cm
        hh_px = self.robot_hitbox_hw_cm[1] * self.px_per_cm

        # Store robot center (not ArUco center) for the bbox
        robot_cx_px = robot_center_x * self.px_per_cm
        robot_cy_px = robot_center_y * self.px_per_cm

        bbox = (float(robot_cx_px), float(robot_cy_px), hw_px, hh_px, yaw)
        return (x_cm, y_cm, yaw), bbox

    # ── Main detection ────────────────────────────────────────────────────────

    def detect_robot_pose(self, raw_frame: np.ndarray
                          ) -> Optional[Tuple[float, float, float]]:
        """
        Two-stage ArUco detection:
          Stage 1: undistorted frame (sharp corners, perspective transform via H)
          Stage 2: warped frame with CLAHE+sharpen (corners already in overhead px)
        Falls back to stage 2 when stage 1 misses.
        """
        if self.sim_mode:
            return None

        pose = self._detect_stage1_undistorted(raw_frame)
        if pose is not None:
            return pose

        pose = self._detect_stage2_warped()
        if pose is not None:
            return pose

        # Both stages failed — age the bbox but keep it alive for a while
        self._bbox_age += 1
        if self._bbox_age > BBOX_PERSIST_FRAMES:
            self.robot_bbox_px = None
            self.robot_pose_cm = None
        # else: keep last known bbox/pose so exclusion mask stays active

        return self.robot_pose_cm  # may be stale but non-None

    def _detect_stage1_undistorted(self, raw_frame: np.ndarray
                                    ) -> Optional[Tuple[float, float, float]]:
        """Stage 1: detect on undistorted (pre-warp) frame, project via H."""
        undistorted = cv2.remap(raw_frame, self.mapx, self.mapy,
                                cv2.INTER_LINEAR)

        fh, fw = undistorted.shape[:2]
        if fw > ARUCO_DETECT_W:
            scale = ARUCO_DETECT_W / fw
            small = cv2.resize(undistorted,
                               (ARUCO_DETECT_W, int(fh * scale)),
                               interpolation=cv2.INTER_LINEAR)
        else:
            small = undistorted
            scale = 1.0

        small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        small_gray = self._preprocess_for_aruco(small_gray)

        corners, ids = self._detect_aruco_on(small_gray, preprocess=False)
        if corners is None:
            return None

        chosen_idx = self._pick_robot_marker(ids)

        # Scale corners back to full-res undistorted coordinates
        c_und = corners[chosen_idx][0] / scale

        # Project to overhead via H_scaled
        pts = np.array(c_und, dtype=np.float32).reshape(-1, 1, 2)
        pts_overhead = cv2.perspectiveTransform(pts, self.H_scaled)

        pose, bbox = self._corners_to_pose_and_bbox(pts_overhead)
        self.robot_pose_cm = pose
        self.robot_bbox_px = bbox
        self._bbox_age = 0
        return pose

    def _detect_stage2_warped(self) -> Optional[Tuple[float, float, float]]:
        """
        Stage 2: detect directly on the warped (overhead) frame.
        Uses CLAHE + unsharp mask to recover blurred corners.
        Corners are already in overhead pixel coordinates — no H projection needed.
        """
        warped = self._last_warped
        if warped is None:
            return None

        fh, fw = warped.shape[:2]

        # Optionally downscale for speed (same logic as stage 1)
        if fw > ARUCO_DETECT_W:
            scale = ARUCO_DETECT_W / fw
            small = cv2.resize(warped,
                               (ARUCO_DETECT_W, int(fh * scale)),
                               interpolation=cv2.INTER_LINEAR)
        else:
            small = warped
            scale = 1.0

        small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Detect with preprocessing (CLAHE + sharpen) — key for warped frame
        corners, ids = self._detect_aruco_on(small_gray, preprocess=True)
        if corners is None:
            return None

        chosen_idx = self._pick_robot_marker(ids)

        # Corners are in small_gray coords → scale to full warped coords
        c_warped = corners[chosen_idx][0] / scale

        pose, bbox = self._corners_to_pose_and_bbox(
            c_warped.reshape(4, 2))
        self.robot_pose_cm = pose
        self.robot_bbox_px = bbox
        self._bbox_age = 0
        return pose

    # ── Odometry fallback ─────────────────────────────────────────────────────

    def inject_pose_from_odom(self, odom_x_m, odom_y_m, odom_yaw_rad,
                               frame_w_px=0, frame_h_px=0):
        """
        Inject pose from odometry, accounting for ArUco offset from robot center.
        """
        x_cm = float(odom_x_m * 100.0)
        y_cm = float(odom_y_m * 100.0)
        yaw  = float(odom_yaw_rad)
        self.robot_pose_cm = (x_cm, y_cm, yaw)

        # Offset from ArUco to robot center
        offset_cm = self.robot_hitbox_offset_cm
        cx_px = (x_cm + offset_cm * np.cos(yaw)) * self.px_per_cm
        cy_px = (y_cm + offset_cm * np.sin(yaw)) * self.px_per_cm

        hw_px = self.robot_hitbox_hw_cm[0] * self.px_per_cm
        hh_px = self.robot_hitbox_hw_cm[1] * self.px_per_cm
        self.robot_bbox_px = (cx_px, cy_px, hw_px, hh_px, yaw)
        self._bbox_age = 0
        return self.robot_pose_cm

    # ── Exclusion mask ────────────────────────────────────────────────────────

    def _robot_exclusion_mask(self, shape):
        """
        Returns a mask covering the robot area as a rotated rectangle.
        Uses persisted bbox even if detection missed this frame.
        Includes padding and dilation to eliminate false positives under the robot.
        """
        if self.robot_bbox_px is None:
            return None

        cx, cy, hw, hh, yaw = self.robot_bbox_px
        s = PROC_SCALE

        # Scale to proc resolution with EXTRA padding to ensure coverage
        padding_cm = 3.0  # Add 3cm padding around the robot
        cx_s = cx * s
        cy_s = cy * s
        hw_s = (hw * s) + (padding_cm * self.px_per_cm * s)
        hh_s = (hh * s) + (padding_cm * self.px_per_cm * s)

        # Expand more when bbox is aging (robot may have moved)
        if self._bbox_age > 0:
            expand = self._bbox_age * 3.0 * s  # Increased from 2.0
            hw_s += expand
            hh_s += expand

        # Compute rotated rectangle corners
        cos_a = np.cos(yaw)
        sin_a = np.sin(yaw)

        # Use the actual asymmetric corners
        local = np.array([
            [ hw_s,  hh_s],   # front-right
            [ hw_s, -hh_s],   # front-left
            [-hw_s, -hh_s],   # back-left
            [-hw_s,  hh_s],   # back-right
        ], dtype=np.float32)

        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        pts = (local @ rot.T) + np.array([cx_s, cy_s], dtype=np.float32)
        pts_int = pts.astype(np.int32)

        mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts_int], 255)

        # Dilate the mask slightly to ensure no edge artifacts
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def get_robot_hitbox_corners_px(self):
        """
        Returns the 4 corners of the rotated hitbox in full-res overhead px.
        Used by the app for rendering. Returns None if no bbox.
        """
        if self.robot_bbox_px is None:
            return None
        cx, cy, hw, hh, yaw = self.robot_bbox_px
        cos_a = np.cos(yaw)
        sin_a = np.sin(yaw)
        local = np.array([
            [ hw,  hh],
            [ hw, -hh],
            [-hw, -hh],
            [-hw,  hh],
        ], dtype=np.float32)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        pts = (local @ rot.T) + np.array([cx, cy], dtype=np.float32)
        return pts.astype(np.int32)

    @staticmethod
    def _preprocess_for_aruco(gray: np.ndarray) -> np.ndarray:
        """CLAHE + unsharp mask para recuperar esquinas borrosas."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        return gray

    def _upscale(self, mask: np.ndarray, full_h: int, full_w: int) -> np.ndarray:
        if mask.shape[:2] == (full_h, full_w):
            return mask
        return cv2.resize(mask, (full_w, full_h), interpolation=cv2.INTER_NEAREST)

    def extract_semantic_layers(self, warped_frame: np.ndarray
                                 ) -> Dict[str, np.ndarray]:
        full_h, full_w = warped_frame.shape[:2]

        proc_w = int(full_w * PROC_SCALE)
        proc_h = int(full_h * PROC_SCALE)
        small  = cv2.resize(warped_frame, (proc_w, proc_h),
                            interpolation=cv2.INTER_AREA)

        hsv  = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape

        self._costmap_frame += 1
        if (self._layers_cache is not None and
                self._costmap_frame % COSTMAP_SKIP_FRAMES != 0):
            return self._layers_cache

        layers: Dict[str, np.ndarray] = {}

        # ── Rangos de color ───────────────────────────────────────────────────
        color_ranges = {
            'cube_red': [
                (np.array([0,   60,  40]), np.array([20,  255, 255])),
                (np.array([155, 60,  40]), np.array([180, 255, 255])),
            ],
            'cube_green': [
                (np.array([25,  60,  50]), np.array([95,  255, 255])),
            ],
            'cube_blue': [
                (np.array([90,  60,  40]), np.array([130, 255, 255])),
            ],
        }

        all_colors_mask = np.zeros((fh, fw), dtype=np.uint8)
        raw_color_masks: Dict[str, np.ndarray] = {}

        k_ex = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for name, ranges in color_ranges.items():
            combined = np.zeros((fh, fw), dtype=np.uint8)
            for lo, hi in ranges:
                combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lo, hi))
            dilated_color = cv2.dilate(combined, k_ex)
            all_colors_mask = cv2.bitwise_or(all_colors_mask, dilated_color)
            raw_color_masks[name] = combined

        # ── Obstáculos negros ─────────────────────────────────────────────────
        robot_excl = self._robot_exclusion_mask(gray.shape)
        # Exclude robot vicinity from color detections
        if robot_excl is not None:
            for name in raw_color_masks:
                raw_color_masks[name] = cv2.bitwise_and(
                    raw_color_masks[name], cv2.bitwise_not(robot_excl))

        _, thresh_black = cv2.threshold(gray, BLACK_THRESHOLD, 255,
                                        cv2.THRESH_BINARY_INV)
        thresh_black = cv2.bitwise_and(thresh_black,
                                       cv2.bitwise_not(all_colors_mask))
        if robot_excl is not None:
            thresh_black = cv2.bitwise_and(thresh_black,
                                           cv2.bitwise_not(robot_excl))

        k_clean       = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned_black = cv2.morphologyEx(thresh_black, cv2.MORPH_OPEN, k_clean)

        cnts_blk, _ = cv2.findContours(cleaned_black, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        raw_obs   = np.zeros_like(cleaned_black)
        walls_msk = np.zeros_like(cleaned_black)
        border    = max(1, int(BORDER_MARGIN * PROC_SCALE))

        for cnt in cnts_blk:
            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0 or h == 0: continue
            on_border = (x <= border or y <= border
                         or x+w >= fw-border or y+h >= fh-border)
            if on_border:
                cv2.drawContours(walls_msk, [cnt], -1, 255, -1)
                continue
            if not self.sim_mode:
                ar = float(w) / h
                if not (0.3 < ar < 3.0): continue
            cv2.drawContours(raw_obs, [cnt], -1, 255, -1)

        layers['walls']         = self._upscale(walls_msk, full_h, full_w)
        layers['obstacles_raw'] = self._upscale(raw_obs,   full_h, full_w)

        dilated = cv2.dilate(raw_obs, self._dilate_kernel)
        if robot_excl is not None:
            dilated = cv2.bitwise_and(dilated, cv2.bitwise_not(robot_excl))
        layers['obstacles'] = self._upscale(dilated, full_h, full_w)

        # ── Cubos y zonas dropoff ─────────────────────────────────────────────
        k_morph  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k_fuse   = cv2.getStructuringElement(cv2.MORPH_RECT,    (15, 15))

        target_area_px = (15.0 * self.px_per_cm * PROC_SCALE) ** 2
        min_area       = target_area_px * 0.10
        max_area       = target_area_px * 4.00

        for name, combined in raw_color_masks.items():
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k_morph)
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_morph)

            fused = cv2.dilate(combined, k_fuse)
            fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, k_fuse)

            solid = np.zeros_like(combined)
            zone  = np.zeros_like(combined)

            cnts_fused, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)

            for cnt_f in cnts_fused:
                x, y, w, h = cv2.boundingRect(cnt_f)
                bbox_area = float(w * h)
                if bbox_area < min_area or bbox_area > max_area:
                    continue

                roi = combined[y:y+h, x:x+w]
                ys_roi, xs_roi = np.where(roi > 0)
                if len(xs_roi) == 0:
                    continue
                color_px = float(len(xs_roi))

                tight_w = int(xs_roi.max() - xs_roi.min() + 1)
                tight_h = int(ys_roi.max() - ys_roi.min() + 1)
                tight_bbox_area = float(tight_w * tight_h)
                if tight_bbox_area == 0:
                    continue

                fill = color_px / tight_bbox_area

                if fill > 0.65:
                    cv2.rectangle(solid, (x, y), (x+w, y+h), 255, -1)
                elif 0.10 <= fill <= 0.55:
                    cv2.rectangle(zone, (x, y), (x+w, y+h), 255, -1)

            layers[f'{name}_solid'] = self._upscale(solid, full_h, full_w)
            layers[f'{name}_zone']  = self._upscale(zone,  full_h, full_w)

        self._layers_cache = layers
        return layers