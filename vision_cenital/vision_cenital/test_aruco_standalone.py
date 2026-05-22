#!/usr/bin/env python3
"""
test_aruco_standalone.py
========================
Debug ArUco detection in isolation.
Uses CargaCam (same pipeline as cargabot_vision_app.py) or V4L2 fallback.
Press Q to quit, S to save snapshot.
"""

import os
os.environ['WAYLAND_DISPLAY'] = ''
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['DISPLAY'] = os.environ.get('DISPLAY', ':0')

import sys
import argparse
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import yaml

CARGACAM_AVAILABLE = False
try:
    from vision_cenital.camera import CargaCam
    CARGACAM_AVAILABLE = True
except Exception as e:
    print(f"[WARN] CargaCam no disponible: {e}")


def load_intrinsics(path):
    with open(path) as f:
        d = yaml.safe_load(f)
    K = np.array(d['camera_matrix']['data']).reshape((3, 3))
    D = np.array(d['dist_coeffs']['data'])
    return K, D, d['image_width'], d['image_height']


def load_homography(path):
    with open(path) as f:
        d = yaml.safe_load(f)
    H = np.array(d['H']).reshape((3, 3))
    w = float(d.get('pista_w_cm', 408.0))
    h = float(d.get('pista_h_cm', 206.0))
    px_cm = int(float(d.get('px_per_cm', 5)))
    return H, w, h, px_cm


def preprocess_for_aruco(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    return gray


def detect_all_markers(gray, aruco_dict, aruco_params, aruco_detector=None):
    if aruco_detector is not None:
        corners, ids, rejected = aruco_detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=aruco_params)
    return corners, ids, rejected


def draw_detection(img, corners, ids, color=(0, 255, 0)):
    if ids is None or len(ids) == 0:
        return img
    vis = img.copy()
    vis = cv2.aruco.drawDetectedMarkers(vis, corners, ids, borderColor=color)
    for i, cid in enumerate(ids.flatten()):
        c = corners[i][0]
        cx, cy = int(c[:, 0].mean()), int(c[:, 1].mean())
        cv2.putText(vis, f"ID:{cid}", (cx - 20, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return vis


def label(img, text, color=(0, 0, 255)):
    cv2.putText(img, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', default='resource/camera_params.yaml')
    parser.add_argument('--homography', default='resource/homography_retry.yaml')
    parser.add_argument('--camera', type=int, default=4,
                        help='Camera index (CargaCam usa 4 por defecto)')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    args = parser.parse_args()

    # Load calibration
    K, D, img_w, img_h = load_intrinsics(args.params)
    H, pista_w_cm, pista_h_cm, px_per_cm = load_homography(args.homography)
    H_scaled = H.copy()
    H_scaled[0] *= px_per_cm
    H_scaled[1] *= px_per_cm
    warped_w = int(pista_w_cm * px_per_cm)
    warped_h = int(pista_h_cm * px_per_cm)

    mapx, mapy = cv2.initUndistortRectifyMap(
        K, D, None, K, (img_w, img_h), cv2.CV_32FC1)

    # ArUco setup (same as perception.py)
    if hasattr(cv2.aruco, 'Dictionary_get'):
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters_create()
        aruco_detector = None
    else:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_detector = None
        if hasattr(cv2.aruco, 'ArucoDetector'):
            aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    if hasattr(aruco_params, 'adaptiveThreshWinSizeMin'):
        aruco_params.adaptiveThreshWinSizeMin = 3
        aruco_params.adaptiveThreshWinSizeMax = 53
        aruco_params.adaptiveThreshWinSizeStep = 10
        aruco_params.minMarkerPerimeterRate = 0.02
        aruco_params.maxMarkerPerimeterRate = 0.5
        aruco_params.polygonalApproxAccuracyRate = 0.10
        aruco_params.cornerRefinementMethod = getattr(
            cv2.aruco, 'CORNER_REFINE_SUBPIX',
            getattr(cv2.aruco, 'CORNER_REFINE_NONE', 0))
    if hasattr(aruco_params, 'perspectiveRemovePixelPerCell'):
        aruco_params.perspectiveRemovePixelPerCell = 3
    if hasattr(aruco_params, 'errorCorrectionRate'):
        aruco_params.errorCorrectionRate = 1.0

    # ── Camera initialization (match cargabot_vision_app.py) ──
    cap = None
    cam = None

    if CARGACAM_AVAILABLE:
        try:
            cam = CargaCam(cam_id=args.camera, width=args.width, height=args.height)
            if cam.start():
                print(f"[Camera] CargaCam /dev/video{args.camera} OK ({args.width}x{args.height})")
            else:
                print("[Camera] CargaCam start() failed, probando V4L2...")
                cam = None
        except Exception as e:
            print(f"[Camera] CargaCam error: {e}")
            cam = None

    if cam is None:
        # Fallback: V4L2 directo (evita GStreamer por defecto)
        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not cap.isOpened() or actual_w == 0:
            print("[ERROR] No se pudo abrir la camara con V4L2.")
            return
        print(f"[Camera] V4L2 /dev/video{args.camera} OK ({actual_w}x{actual_h})")

    print("[Controls] Q = quit | S = save snapshot")
    print("-" * 50)

    # ── Window init (same ritual as cargabot_vision_app.py) ──
    WIN = "ArUco Standalone Test"
    _dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(_dummy, "Iniciando...", (20, 240),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 220, 255), 2)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.imshow(WIN, _dummy)
    for _ in range(20):
        cv2.waitKey(50)

    while True:
        if cam is not None:
            ret, frame = cam.read()
        else:
            ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            time.sleep(0.01)
            continue

        # 1) RAW
        raw_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        c_raw, ids_raw, _ = detect_all_markers(
            raw_gray, aruco_dict, aruco_params, aruco_detector)
        vis_raw = draw_detection(frame, c_raw, ids_raw, (0, 255, 0))

        # 2) UNDISTORTED
        undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        und_gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        c_und, ids_und, _ = detect_all_markers(
            und_gray, aruco_dict, aruco_params, aruco_detector)
        vis_und = draw_detection(undistorted, c_und, ids_und, (255, 0, 0))

        # 3) WARPED (overhead)
        warped = cv2.warpPerspective(
            undistorted, H_scaled, (warped_w, warped_h))
        warp_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        c_warp, ids_warp, _ = detect_all_markers(
            warp_gray, aruco_dict, aruco_params, aruco_detector)
        vis_warp = draw_detection(warped, c_warp, ids_warp, (0, 255, 255))

        # 4) WARPED + SHARPEN
        warp_pp = preprocess_for_aruco(warp_gray)
        c_pp, ids_pp, _ = detect_all_markers(
            warp_pp, aruco_dict, aruco_params, aruco_detector)
        vis_pp = cv2.cvtColor(warp_pp, cv2.COLOR_GRAY2BGR)
        vis_pp = draw_detection(vis_pp, c_pp, ids_pp, (255, 0, 255))

        # Build 2x2 mosaic
        target_h, target_w = 480, 640
        vis_raw = cv2.resize(vis_raw, (target_w, target_h))
        vis_und = cv2.resize(vis_und, (target_w, target_h))
        vis_warp = cv2.resize(vis_warp, (target_w, target_h))
        vis_pp = cv2.resize(vis_pp, (target_w, target_h))

        top = np.hstack([
            label(vis_raw, "1. RAW", (0, 255, 0)),
            label(vis_und, "2. UNDISTORTED", (255, 0, 0)),
        ])
        bot = np.hstack([
            label(vis_warp, "3. WARPED", (0, 255, 255)),
            label(vis_pp, "4. WARPED+SHARP", (255, 0, 255)),
        ])
        mosaic = np.vstack([top, bot])

        # Status overlay
        status = [
            f"RAW      : {list(ids_raw.flatten())  if ids_raw  is not None else 'NONE'}",
            f"UNDIST   : {list(ids_und.flatten())  if ids_und  is not None else 'NONE'}",
            f"WARPED   : {list(ids_warp.flatten()) if ids_warp is not None else 'NONE'}",
            f"SHARPEN  : {list(ids_pp.flatten())   if ids_pp   is not None else 'NONE'}",
        ]
        y0 = mosaic.shape[0] - 110
        for i, s in enumerate(status):
            cv2.putText(mosaic, s, (12, y0 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        cv2.imshow(WIN, mosaic)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"aruco_snapshot_{int(time.time())}.png"
            cv2.imwrite(fname, mosaic)
            print(f"[Snapshot] Saved {fname}")

    if cam is not None:
        cam.release()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
