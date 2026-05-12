"""
PASO 3: Cálculo de homografía pista → coordenadas reales (cm)
==============================================================
Uso: python homography.py [--camera 0] [--params camera_params.yaml] [--output homography.yaml]

Con la cámara YA MONTADA en el techo apuntando a la pista:
1. Se abre el feed en vivo (ya sin distorsión)
2. Haces click en las 4 esquinas de la pista EN ORDEN:
     esquina superior-izquierda → superior-derecha → inferior-derecha → inferior-izquierda
3. Ingresas las coordenadas reales de cada esquina en cm
4. Se calcula H y se valida con la grilla de 20x20cm

IMPORTANTE: Las 4 esquinas deben ser puntos que puedas medir físicamente.
Recomendado: pegar ArUcos pequeños (ID 10-13) en las 4 esquinas como referencia permanente.
"""

import cv2
import numpy as np
import yaml
import argparse

# ─── Coordenadas reales de las 4 esquinas en cm ────────────────────────────
# Ajusta esto según el tamaño de TU pista.
# Origen (0,0) = esquina superior-izquierda
# x crece hacia la derecha, y crece hacia abajo
REAL_CORNERS_CM = np.float32([
    [  0,   0],   # superior-izquierda
    [200,   0],   # superior-derecha
    [200, 200],   # inferior-derecha
    [  0, 200],   # inferior-izquierda
])
# ────────────────────────────────────────────────────────────────────────────

WINDOW = 'Homografia — click en las 4 esquinas de la pista'
clicked_points = []

def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append([x, y])
        labels = ['SUP-IZQ', 'SUP-DER', 'INF-DER', 'INF-IZQ']
        print(f"  [{len(clicked_points)}/4] {labels[len(clicked_points)-1]}: ({x}, {y}) px")

def load_camera_params(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    K = np.array(data['camera_matrix']['data']).reshape(3, 3)
    D = np.array(data['dist_coeffs']['data'])
    w = data['image_width']
    h = data['image_height']
    return K, D, w, h

def draw_state(frame, points):
    """Dibuja los puntos ya clickeados y las instrucciones."""
    labels = ['SUP-IZQ', 'SUP-DER', 'INF-DER', 'INF-IZQ']
    colors = [(0,255,0), (0,200,255), (255,100,0), (200,0,255)]

    for i, pt in enumerate(points):
        cv2.circle(frame, tuple(pt), 8, colors[i], -1)
        cv2.putText(frame, labels[i], (pt[0]+10, pt[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

    if len(points) < 4:
        next_label = labels[len(points)]
        cv2.putText(frame, f"Click: {next_label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(frame, "R = resetear puntos", (20, frame.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150,150,150), 1)
    else:
        cv2.putText(frame, "SPACE = confirmar y calcular H", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    return frame

def draw_validation_grid(warped_frame, H, resolution_cm=20, pista_size_cm=200):
    """Dibuja una grilla de 20x20cm sobre la imagen warped para validar H."""
    out = warped_frame.copy()
    steps = int(pista_size_cm / resolution_cm) + 1

    # La imagen warped tiene dimensiones en píxeles; necesitamos escalar cm→px
    # Asumimos que la imagen warped fue generada con 5px/cm → 1000x1000px para 200cm
    scale = out.shape[1] / pista_size_cm  # px/cm

    for i in range(steps):
        x = int(i * resolution_cm * scale)
        cv2.line(out, (x, 0), (x, out.shape[0]), (0,200,100), 1)
        cv2.putText(out, f"{i*resolution_cm}", (x+2, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,200,100), 1)
    for j in range(steps):
        y = int(j * resolution_cm * scale)
        cv2.line(out, (0, y), (out.shape[1], y), (0,200,100), 1)
        cv2.putText(out, f"{j*resolution_cm}", (2, y+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,200,100), 1)

    return out

def main():
    global clicked_points

    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--params', type=str, default='camera_params.yaml')
    parser.add_argument('--output', type=str, default='homography.yaml')
    args = parser.parse_args()

    # Cargar parámetros de calibración
    K, D, w, h = load_camera_params(args.params)
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    print(f"✅ Parámetros de cámara cargados desde {args.params}")
    print(f"   Imagen: {w}x{h} px")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    if not cap.isOpened():
        print(f"❌ No se pudo abrir cámara {args.camera}")
        return

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, mouse_callback)

    print(f"\n📐 Instrucciones:")
    print(f"   Haz click en las 4 esquinas de la pista EN ESTE ORDEN:")
    print(f"   1. Superior-Izquierda  → corresponde a (0,0) cm")
    print(f"   2. Superior-Derecha    → corresponde a ({REAL_CORNERS_CM[1][0]:.0f},0) cm")
    print(f"   3. Inferior-Derecha    → corresponde a ({REAL_CORNERS_CM[2][0]:.0f},{REAL_CORNERS_CM[2][1]:.0f}) cm")
    print(f"   4. Inferior-Izquierda  → corresponde a (0,{REAL_CORNERS_CM[3][1]:.0f}) cm")
    print(f"   Luego: SPACE = calcular | R = resetear\n")

    H = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Corregir distorsión
        frame_undist = cv2.undistort(frame, K, D, None, new_K)
        display = frame_undist.copy()
        display = draw_state(display, clicked_points)

        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        elif key == ord('r') or key == ord('R'):
            clicked_points = []
            print("🔄 Puntos reseteados")

        elif key == ord(' ') and len(clicked_points) == 4:
            print("\n⚙️  Calculando homografía...")

            src_pts = np.float32(clicked_points)
            dst_pts = REAL_CORNERS_CM

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is None:
                print("❌ No se pudo calcular H. Intenta de nuevo.")
                clicked_points = []
                continue

            print(f"✅ Homografía calculada")
            print(f"   H =\n{H}")

            # Validación visual: warp de la imagen
            OUTPUT_SIZE_PX = 1000  # 1000px para 200cm → 5px/cm
            scale = OUTPUT_SIZE_PX / 200.0

            # Escalar H para que los cm se mapeen a píxeles
            H_scaled = H.copy()
            H_scaled[0] *= scale
            H_scaled[1] *= scale

            warped = cv2.warpPerspective(frame_undist, H_scaled, (OUTPUT_SIZE_PX, OUTPUT_SIZE_PX))
            warped_grid = draw_validation_grid(warped, H_scaled)

            cv2.imshow('Validacion — grilla 20x20cm', warped_grid)
            print("\n👀 Ventana de validación abierta.")
            print("   ¿Las líneas de la grilla coinciden con la grilla real de la pista?")
            print("   Si sí → presiona S para guardar | N para resetear y repetir")

            while True:
                k2 = cv2.waitKey(0) & 0xFF
                if k2 == ord('s') or k2 == ord('S'):
                    # Guardar H + H_scaled
                    data = {
                        'output_size_px': OUTPUT_SIZE_PX,
                        'pista_size_cm': 200,
                        'scale_px_per_cm': scale,
                        'H': H.flatten().tolist(),
                        'H_scaled': H_scaled.flatten().tolist(),
                        'src_points_px': np.array(clicked_points).flatten().tolist(),
                        'dst_points_cm': REAL_CORNERS_CM.flatten().tolist(),
                    }
                    with open(args.output, 'w') as f:
                        yaml.dump(data, f, default_flow_style=False)
                    print(f"\n💾 Homografía guardada en: {args.output}")
                    print(f"   Listo para usar en overhead_vision_node.py")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                elif k2 == ord('n') or k2 == ord('N'):
                    print("🔄 Reseteando — vuelve a hacer click en las 4 esquinas")
                    clicked_points = []
                    cv2.destroyWindow('Validacion — grilla 20x20cm')
                    break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
