"""
NÚCLEO DE VIDEO ACELERADO POR HARDWARE (C920 + AMD Radeon 780M)
Resolución: 1280x720 reales (Sin recortes) | Códec: H.264 Hardware
Constante física: Foco = 0
"""

import cv2
import os
import time

CAM_ID = 4

def preconfigurar_lente():
    """Bloquea el hardware de la C920 en su punto de nitidez absoluto"""
    print("⚙️ Preparando hardware de la cámara (Foco=0)...")
    os.system(f"v4l2-ctl -d /dev/video{CAM_ID} --set-ctrl=focus_automatic_continuous=0 >/dev/null 2>&1")
    os.system(f"v4l2-ctl -d /dev/video{CAM_ID} --set-ctrl=exposure_dynamic_framerate=0 >/dev/null 2>&1")
    os.system(f"v4l2-ctl -d /dev/video{CAM_ID} --set-ctrl=focus_absolute=0 >/dev/null 2>&1")
    time.sleep(0.3)

def main():
    preconfigurar_lente()

    # ── PIPELINE CINEMATOGRÁFICO (El código ganador) ──
    # Extraemos H.264 puro. 'decodebin' delegará automáticamente la tarea
    # de descompresión a tu GPU AMD Radeon 780M mediante VA-API.
    # appsink extrae los frames decodificados hacia OpenCV en formato BGR.
    gst_pipeline = (
        f"v4l2src device=/dev/video{CAM_ID} ! "
        f"video/x-h264, width=1280, height=720 ! "
        f"decodebin ! videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1"
    )

    print("🚀 Iniciando puente GStreamer -> GPU AMD -> OpenCV...")
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("❌ Falló la captura. El nodo de video puede seguir bloqueado.")
        print(f"   Libéralo con: sudo fuser -k /dev/video{CAM_ID}")
        return

    # Verificamos las dimensiones entrantes
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("\n" + "★"*50)
    print(f"🎬 VIDEO LISTO PARA PROCESAMIENTO: {w}x{h} píxeles reales")
    print("★"*50 + "\n")

    window_name = "Matriz de Trabajo (C920 H.264)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("🟢 Mostrando fotogramas. Presiona 'Q' o 'ESC' para cerrar.")

    # Variables para medir los FPS reales del procesamiento
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Cálculo de FPS de procesamiento en vivo
        fps_counter += 1
        if (time.time() - fps_start_time) > 1.0:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()

        # ── AQUÍ VA TU PROCESAMIENTO DE ROBÓTICA ──
        # frame_procesado = cv2.cvtColor(frame, ...)
        # ...
        
        # Overlay informativo (HUD)
        cv2.putText(frame, f"Resolucion: {w}x{h} | FPS: {fps:.1f} | Aceleracion GPU", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "FOV Panoramico 16:9 | Foco Fijo: 0", 
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF in [27, ord('q'), ord('Q')]:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🏁 Stream de trabajo cerrado limpiamente.")

if __name__ == "__main__":
    main()