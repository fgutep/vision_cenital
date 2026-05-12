"""
PASO 3 (CORREGIDO): Medición Offline sobre fotos del laboratorio
"""

import cv2
import numpy as np
import yaml
import argparse
import time

# Configuración del tablero (Basado en tu éxito de 7x10)
BOARD_W = 7
BOARD_H = 10
SQUARE_SIZE_CM = 2.27

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Ruta a la foto con el tablero en el piso')
    parser.add_argument('--params', type=str, default='camera_params.yaml')
    args = parser.parse_args()

    # 1. Cargar K y dist
    with open(args.params, 'r') as f:
        data = yaml.safe_load(f)
    
    K = np.array(data['camera_matrix']['data']).reshape((3,3))
    dist = np.array(data['dist_coeffs']['data']).reshape((1,5))

    # 2. Cargar e Undistort
    img_orig = cv2.imread(args.image)
    if img_orig is None:
        print(f"❌ No se pudo cargar la imagen: {args.image}")
        return
    
    print("✨ Aplicando corrección de lente (Undistort)...")
    img = cv2.undistort(img_orig, K, dist)
    display_img = img.copy()

    # 3. Homografía
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("🔍 Buscando tablero para escala...")
    found, corners = cv2.findChessboardCornersSB(gray, (BOARD_W, BOARD_H))

    if not found:
        print("❌ No se detectó tablero. Prueba con una foto donde el papel se vea plano y nítido.")
        return

    world_pts = np.zeros((BOARD_W * BOARD_H, 2), np.float32)
    world_pts[:, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2) * SQUARE_SIZE_CM
    H, _ = cv2.findHomography(corners, world_pts)

    # 4. Configuración de ventana con Delay para evitar el Null Pointer
    win_name = "MEDIDOR OFFLINE - Clic para medir"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)
    time.sleep(0.5) # <--- CRITICO: Tiempo para que Qt/Wayland respiren

    puntos = []

    def handle_click(event, x, y, flags, param):
        nonlocal display_img
        if event == cv2.EVENT_LBUTTONDOWN:
            # Píxel -> CM
            p = np.array([x, y, 1.0]).reshape((3, 1))
            p_cm = np.dot(H, p)
            p_cm /= p_cm[2]
            
            puntos.append((x, y, p_cm[0][0], p_cm[1][0]))
            cv2.circle(display_img, (x, y), 6, (0, 0, 255), -1)
            
            if len(puntos) >= 2:
                p1, p2 = puntos[-2], puntos[-1]
                d = np.sqrt((p1[2]-p2[2])**2 + (p1[3]-p2[3])**2)
                
                cv2.line(display_img, (p1[0], p1[1]), (p2[0], p2[1]), (0, 255, 0), 2)
                cv2.putText(display_img, f"{d:.2f}cm", (x + 10, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                print(f"📏 Medida: {d:.2f} cm")
            
            cv2.imshow(win_name, display_img)

    # Solo asignamos el callback si la ventana existe
    try:
        cv2.setMouseCallback(win_name, handle_click)
    except cv2.error:
        print("❌ Error al asignar el ratón. Intenta correrlo de nuevo.")
        return

    print("\n🟢 LISTO:")
    print("  1. Haz clic en el origen de un objeto.")
    print("  2. Haz clic en el final.")
    print("  3. Presiona 'R' para limpiar puntos o 'ESC' para salir.")

    while True:
        cv2.imshow(win_name, display_img)
        key = cv2.waitKey(20) & 0xFF
        if key == 27: # ESC
            break
        elif key == ord('r') or key == ord('R'):
            display_img = img.copy()
            puntos = []
            print("🧹 Puntos limpiados.")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()