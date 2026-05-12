import cv2
import numpy as np
import yaml
import os
import argparse
import glob

# ── CONFIGURACIÓN CRÍTICA ──
# Esquinas INTERNAS (Cuadrados - 1)
BOARD_W = 7  
BOARD_H = 10 
# Tamaño real de cada cuadrado en cm (Ajusta si tu impresora escaló el papel)
SQUARE_SIZE_CM = 2.27

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  type=str, default='./fotos')
    parser.add_argument('--output', type=str, default='camera_params.yaml')
    args = parser.parse_args()

    # Preparar puntos del objeto (0,0,0), (1,0,0), (2,0,0) ... (6,9,0) en cm
    objp = np.zeros((BOARD_H * BOARD_W, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2) * SQUARE_SIZE_CM

    objpoints = [] # Puntos 3D en el mundo real
    imgpoints = [] # Puntos 2D en los píxeles de la imagen

    imagenes = sorted(glob.glob(os.path.join(args.input, '*.jpg')))
    if not imagenes:
        print(f"❌ No hay imágenes en {args.input}")
        return

    print(f"📂 Procesando {len(imagenes)} imágenes HD...")
    print(f"   Buscando tablero de {BOARD_W}x{BOARD_H} esquinas internas.\n")

    img_shape = None
    usadas = 0

    for path in imagenes:
        img = cv2.imread(path)
        if img is None: continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        # Usamos el detector SB (Sector Based), mucho más robusto para 720p/1080p
        # No necesita refinamiento manual posterior porque ya es preciso a nivel subpíxel
        found, corners = cv2.findChessboardCornersSB(
            gray, (BOARD_W, BOARD_H), 
            cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE
        )

        if found:
            objpoints.append(objp)
            imgpoints.append(corners)
            usadas += 1
            print(f"   ✅ {os.path.basename(path)} - Detectado")
        else:
            print(f"   ⚠️  {os.path.basename(path)} - No detectado (omitido)")

    if usadas < 15:
        print(f"\n❌ Solo se detectaron {usadas} imágenes. Necesitas al menos 20 de calidad para una pista grande.")
        return

    print(f"\n⚙️  Calibrando con {usadas} muestras...")
    # Calibración de cámara
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    print("\n" + "★"*40)
    print(f"✅ CALIBRACIÓN COMPLETADA")
    status = "EXCELENTE" if rms < 0.4 else "BUENO" if rms < 0.7 else "REVISAR"
    print(f"   RMS Error: {rms:.4f} px  [{status}]")
    print("★"*40)

    # Exportar a YAML
    params = {
        'rms_error': float(rms),
        'image_width':  int(img_shape[0]),
        'image_height': int(img_shape[1]),
        'camera_matrix': {
            'rows': 3, 'cols': 3,
            'data': camera_matrix.flatten().tolist()
        },
        'dist_coeffs': {
            'rows': 1, 'cols': 5,
            'data': dist_coeffs.flatten().tolist()
        }
    }

    with open(args.output, 'w') as f:
        yaml.dump(params, f, default_flow_style=False)

    print(f"\n💾 Parámetros guardados en: {args.output}")

if __name__ == '__main__':
    main()