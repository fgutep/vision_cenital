#!/usr/bin/env python3
"""
record_camera.py
================
Graba la cámara cenital a .mp4 para poder desarrollar sin hardware.
Preview en vivo + toggle de grabación con la tecla R.
"""

import os
os.environ['WAYLAND_DISPLAY'] = ''
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['DISPLAY'] = os.environ.get('DISPLAY', ':0')

import sys
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np

CARGACAM_AVAILABLE = False
try:
    from vision_cenital.camera import CargaCam
    CARGACAM_AVAILABLE = True
except Exception as e:
    print(f"[WARN] CargaCam no disponible: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=4,
                        help='Índice de cámara (CargaCam usa 4 por defecto)')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--output', type=str, default='',
                        help='Nombre del archivo (default: auto con timestamp)')
    args = parser.parse_args()

    # ── Camera init (misma lógica que test_aruco_standalone.py) ──
    cap = None
    cam = None

    if CARGACAM_AVAILABLE:
        try:
            cam = CargaCam(cam_id=args.camera, width=args.width, height=args.height)
            if cam.start():
                print(f"[Camera] CargaCam /dev/video{args.camera} OK")
            else:
                print("[Camera] CargaCam falló, probando V4L2...")
                cam = None
        except Exception as e:
            print(f"[Camera] CargaCam error: {e}")
            cam = None

    if cam is None:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not cap.isOpened() or actual_w == 0:
            print("[ERROR] No se pudo abrir la cámara.")
            return
        print(f"[Camera] V4L2 /dev/video{args.camera} OK ({actual_w}x{actual_h})")

    # ── Window init ──
    WIN = "Recorder (R=toggle | Q=quit)"
    _dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(_dummy, "Iniciando...", (20, 240),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 220, 255), 2)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.imshow(WIN, _dummy)
    for _ in range(10):
        cv2.waitKey(50)

    # ── VideoWriter setup ──
    out = None
    recording = False
    start_time = 0.0
    frame_count = 0
    filename = ""

    # Intentar mp4v (más compatible), fallback a XVID
    fourcc_candidates = ['mp4v', 'XVID', 'MJPG']
    fourcc = None
    for c in fourcc_candidates:
        cc = cv2.VideoWriter_fourcc(*c)
        test_path = "/tmp/_test_recorder.mp4"
        test_writer = cv2.VideoWriter(test_path, cc, args.fps,
                                      (args.width, args.height))
        if test_writer.isOpened():
            test_writer.release()
            os.remove(test_path)
            fourcc = cc
            print(f"[Recorder] Codec seleccionado: {c}")
            break
    if fourcc is None:
        print("[ERROR] No se encontró codec de video compatible.")
        return

    print("[Controls] R = iniciar/detener grabación | Q = salir")
    print("-" * 50)

    while True:
        if cam is not None:
            ret, frame = cam.read()
        else:
            ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            time.sleep(0.01)
            continue

        display = frame.copy()
        h, w = display.shape[:2]

        # OSD
        if recording and out is not None:
            # Puntos rojos parpadeantes
            blink = int(time.time() * 2) % 2 == 0
            if blink:
                cv2.circle(display, (30, 30), 10, (0, 0, 255), -1)
                cv2.circle(display, (30, 30), 10, (0, 0, 0), 2)
            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            hrs, mins = divmod(mins, 60)
            status = f"REC {hrs:02d}:{mins:02d}:{secs:02d} | {frame_count} frames"
            cv2.putText(display, status, (55, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.putText(display, filename, (12, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        else:
            cv2.putText(display, "LISTO (R para grabar)", (12, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow(WIN, display)

        if recording and out is not None:
            # Asegurar tamaño correcto
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height))
            out.write(frame)
            frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            if not recording:
                # Iniciar grabación
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = args.output if args.output else f"cenital_{ts}.mp4"
                out = cv2.VideoWriter(filename, fourcc, args.fps,
                                      (args.width, args.height))
                if not out.isOpened():
                    print(f"[ERROR] No se pudo crear: {filename}")
                    out = None
                    continue
                recording = True
                start_time = time.time()
                frame_count = 0
                print(f"[REC] Iniciado: {filename}")
            else:
                # Detener grabación
                recording = False
                if out is not None:
                    out.release()
                    out = None
                print(f"[REC] Detenido: {filename} ({frame_count} frames)")
                filename = ""

    # Limpieza
    if recording and out is not None:
        out.release()
        print(f"[REC] Guardado: {filename} ({frame_count} frames)")
    if cam is not None:
        cam.release()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
