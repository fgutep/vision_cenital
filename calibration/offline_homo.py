"""
PASO 3 (OFFLINE): Cálculo de homografía usando fotos de archivo
Uso: uv run homography_offline.py --image ./fotos/pista_referencia.jpg --params camera_params.yaml
"""

import cv2
import numpy as np
import yaml
import argparse
import os

# ─── CONFIGURACIÓN DE LA PISTA (Ajusta a tus medidas reales en cm) ───
# Por ejemplo, si tu cuadrícula de papel mide 17.5cm x 25cm (7x10 esquinas)
REAL_W_CM = 17.5 
REAL_H_CM = 25.0

REAL_CORNERS_CM = np.float32([
    [0, 0],           # SUP-IZQ
    [REAL_W_CM, 0],   # SUP-DER
    [REAL_W_CM, REAL_H_CM], # INF-DER
    [0, REAL_H_CM],   # INF-IZQ
])
# ─────────────────────────────────────────────────────────────────────

WINDOW = 'Homografia Offline - Click en las 4 esquinas del tablero'
clicked_points = []

def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append([x, y])
        labels = ['SUP-IZQ', 'SUP-DER', 'INF-DER', 'INF-IZQ']
        print(f"  📌 [{len(clicked_points)}/4] {labels[len(clicked_points)-1]}: ({x}, {y}) px")

def main():
    global clicked_points
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Foto de la pista con el tablero')
    parser.add_argument('--params', type=str, default='camera_params.yaml')
    parser.add_argument('--output', type=str, default='homography.yaml')
    args = parser.parse_args()

    # 1. Cargar calibración
    with open(args.params, 'r') as f:
        data = yaml.safe_load(f)
    K = np.array(data['camera_matrix']['data']).reshape(3, 3)
    D = np.array(data['dist_coeffs']['data'])
    w_orig = data['image_width']
    h_orig = data['image_height']

    # 2. Cargar imagen
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"❌ No se encontró la imagen {args.image}")
        return

    # 3. Corregir distorsión (Undistort) antes de marcar puntos
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w_orig, h_orig), 1, (w_orig, h_orig))
    frame_undist = cv2.undistort(frame, K, D, None, new_K)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, mouse_callback)

    print(f"\n📐 INSTRUCCIONES:")
    print(f"   Haz click en las 4 ESQUINAS DEL TABLERO (o área conocida) en este orden:")
    print(f"   1. Sup-Izq -> 2. Sup-Der -> 3. Inf-Der -> 4. Inf-Izq")
    print(f"   Presiona SPACE para calcular | R para resetear")

    while True:
        display = frame_undist.copy()
        
        # Dibujar puntos y líneas de guía
        for i, pt in enumerate(clicked_points):
            cv2.circle(display, tuple(pt), 7, (0, 255, 0), -1)
            if i > 0:
                cv2.line(display, tuple(clicked_points[i-1]), tuple(pt), (0, 255, 0), 2)
        if len(clicked_points) == 4:
            cv2.line(display, tuple(clicked_points[3]), tuple(clicked_points[0]), (0, 255, 0), 2)

        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27: break
        if key == ord('r'): clicked_points = []

        if key == ord(' ') and len(clicked_points) == 4:
            # Calcular H
            src_pts = np.float32(clicked_points)
            dst_pts = REAL_CORNERS_CM
            H, _ = cv2.findHomography(src_pts, dst_pts)

            # Crear vista de pájaro (Warp) para validar
            # Escalamos a 5 píxeles por cm para que se vea bien
            PX_PER_CM = 10 
            out_w = int(REAL_W_CM * PX_PER_CM)
            out_h = int(REAL_H_CM * PX_PER_CM)
            
            H_scaled = H.copy()
            H_scaled[0] *= PX_PER_CM
            H_scaled[1] *= PX_PER_CM
            
            warped = cv2.warpPerspective(frame_undist, H_scaled, (out_w, out_h))
            
            cv2.imshow("VALIDACION (Vista de Pajaro)", warped)
            print("\n👀 Revisa la ventana de Validación.")
            print("   Si el tablero se ve como un rectángulo perfecto y recto: Presiona 'S' para GUARDAR.")
            
            sub_key = cv2.waitKey(0) & 0xFF
            if sub_key == ord('s'):
                save_data = {
                    'H': H.flatten().tolist(),
                    'pista_w_cm': REAL_W_CM,
                    'pista_h_cm': REAL_H_CM,
                    'px_per_cm': PX_PER_CM
                }
                with open(args.output, 'w') as f:
                    yaml.dump(save_data, f)
                print(f"💾 Homografía guardada en {args.output}")
                break
            else:
                cv2.destroyWindow("VALIDACION (Vista de Pajaro)")
                clicked_points = []

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()