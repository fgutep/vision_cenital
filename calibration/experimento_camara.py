"""
BANCO DE PRUEBAS DEFINITIVO: GStreamer + Decodebin (Bypass de OpenCV V4L2)
Constante asegurada: Foco = 0
"""

import cv2
import os
import time

CAM_ID = 4

def main():
    print("⚙️ Vaciando búferes y bloqueando lente en Foco=0...")
    os.system(f"v4l2-ctl -d /dev/video{CAM_ID} --set-ctrl=focus_automatic_continuous=0 >/dev/null 2>&1")
    os.system(f"v4l2-ctl -d /dev/video{CAM_ID} --set-ctrl=exposure_dynamic_framerate=0 >/dev/null 2>&1")
    os.system(f"v4l2-ctl -d /dev/video{CAM_ID} --set-ctrl=focus_absolute=0 >/dev/null 2>&1")
    time.sleep(0.5)

    # ── PIPELINE DE GSTREAMER CON AUTO-NEGOCIACIÓN (decodebin) ──
    # Extraemos explícitamente JPEG a 1280x720 y dejamos que decodebin haga la magia
    gst_pipeline = (
        f"v4l2src device=/dev/video{CAM_ID} ! "
        f"image/jpeg, width=1280, height=720 ! "
        f"decodebin ! videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1"
    )

    print("🚀 Puenteando V4L2... Inyectando pipeline de GStreamer directo al hardware:")
    print(f"   Pipeline: {gst_pipeline}\n")
    
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("❌ GStreamer abortó. El kernel sigue teniendo secuestrado el nodo de video.")
        print("   Ejecuta en tu terminal para matar el bloqueo: sudo fuser -k /dev/video4")
        return

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("="*50)
    if real_w >= 1280:
        print(f"✅ ¡POR FIN! Matriz completa capturada: {real_w}x{real_h} reales.")
    else:
        print(f"⚠️ Resolución obtenida: {real_w}x{real_h}")
    print("="*50 + "\n")

    window_name = "Stream HD Limpio (GStreamer)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("🟢 Transmitiendo pista sin recortes. Presiona 'Q' o 'ESC' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]

        # Cruz de validación de bordes
        cv2.line(frame, (w//2, 0), (w//2, h), (0, 255, 0), 1)
        cv2.line(frame, (0, h//2), (w, h//2), (0, 255, 0), 1)

        # UI en texto
        color = (0, 255, 0) if w >= 1280 else (0, 0, 255)
        cv2.putText(frame, f"GStreamer Salida: {w}x{h} | FOV Completo", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF in [27, ord('q'), ord('Q')]:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🏁 Fin de la prueba.")

if __name__ == "__main__":
    main()