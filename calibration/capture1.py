"""
PASO 1: Captura HD nativa (16:9) con GStreamer y Foco Manual en Vivo
"""

import cv2
import os
import argparse
import pygame
import numpy as np

# Configuración del tablero de calibración (esquinas internas)
BOARD_W = 8
BOARD_H = 11

def setup_camera_driver(cam_id):
    """
    Preconfigura el driver V4L2 a nivel de kernel para liberar automatismos
    antes de inyectar el pipeline de GStreamer.
    """
    print("⚙️ Preparando driver V4L2 a nivel de sistema...")
    # Apagar enfoque automático continuo
    os.system(f"v4l2-ctl -d /dev/video{cam_id} --set-ctrl=focus_automatic_continuous=0 >/dev/null 2>&1")
    # Apagar ajuste dinámico de framerate (evita caídas de FPS con poca luz)
    os.system(f"v4l2-ctl -d /dev/video{cam_id} --set-ctrl=exposure_dynamic_framerate=0 >/dev/null 2>&1")
    pygame.time.wait(200)

def set_driver_focus(cam_id, focus_val):
    """Envía el comando de foco absoluto al hardware de la cámara"""
    os.system(f"v4l2-ctl -d /dev/video{cam_id} --set-ctrl=focus_absolute={focus_val} >/dev/null 2>&1")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=4)
    parser.add_argument('--output', type=str, default='./fotos')
    parser.add_argument('--win-width', type=int, default=1280)
    parser.add_argument('--win-height', type=int, default=760) # 720p + 40px de barra inferior
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 1. Liberar controles automáticos en el hardware
    setup_camera_driver(args.camera)

    # 2. Construir el Pipeline de GStreamer
    # Forzamos la extracción en JPEG a 720p nativo, decodificamos y convertimos a BGR.
    # drop=true y max-buffers=1 evitan el lag acumulado en el buffer de video.
    gst_pipeline = (
        f"v4l2src device=/dev/video{args.camera} ! "
        f"image/jpeg, width=1280, height=720, framerate=30/1 ! "
        f"jpegdec ! videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1"
    )

    print("🚀 Inicializando captura a través de GStreamer...")
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print(f"❌ Falló la apertura con GStreamer para /dev/video{args.camera}.")
        print("Asegúrate de tener instalados los plugins base: sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-good")
        return

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ Cámara enlazada exitosamente a: {real_w}x{real_h}")

    # Configuración de foco inicial
    current_focus = 30
    set_driver_focus(args.camera, current_focus)

    # Contabilidad de fotos previas
    count = len([f for f in os.listdir(args.output) if f.endswith('.jpg')])

    # Inicialización de Interfaz Pygame
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pygame.init()
    screen = pygame.display.set_mode((args.win_width, args.win_height), pygame.RESIZABLE)
    pygame.display.set_caption("Calibración - GStreamer HD")
    font = pygame.font.SysFont("monospace", 18)
    clock = pygame.time.Clock()

    last_frame = None
    found_board = False
    flash_frames = 0
    running = True

    print("\n" + "="*40)
    print(" CONTROLES DE CAPTURA:")
    print("  • [ESPACIO] : Guardar foto de calibración")
    print("  • [▲ / ▼]   : Ajustar foco del lente en vivo")
    print("  • [ESC]     : Finalizar sesión")
    print("="*40 + "\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if found_board and last_frame is not None:
                        filename = os.path.join(args.output, f'calib_{count:03d}.jpg')
                        # Guardamos el frame crudo de alta resolución sin renderizados de UI
                        cv2.imwrite(filename, last_frame)
                        count += 1
                        flash_frames = 8
                        print(f"📸 Foto {count:03d} guardada con éxito | Foco: {current_focus}")
                
                # Ajuste manual de foco en tiempo real
                elif event.key == pygame.K_UP:
                    current_focus = min(250, current_focus + 5)
                    set_driver_focus(args.camera, current_focus)
                elif event.key == pygame.K_DOWN:
                    current_focus = max(0, current_focus - 5)
                    set_driver_focus(args.camera, current_focus)

        ret, frame = cap.read()
        if not ret:
            continue

        last_frame = frame.copy()
        display = frame.copy()

        # Detección optimizada del tablero
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Filtro de paso rápido para no congelar los fotogramas
        fast_found, _ = cv2.findChessboardCorners(gray, (BOARD_W, BOARD_H), cv2.CALIB_CB_FAST_CHECK)
        
        found_board = False
        corners = None
        
        if fast_found:
            try:
                # Búsqueda precisa basada en sectores (ideal para alta resolución y bajo ruido)
                found_board, corners = cv2.findChessboardCornersSB(gray, (BOARD_W, BOARD_H), cv2.CALIB_CB_NORMALIZE_IMAGE)
            except AttributeError:
                pass
            
            if not found_board:
                found_board, corners = cv2.findChessboardCorners(gray, (BOARD_W, BOARD_H), cv2.CALIB_CB_ADAPTIVE_THRESH)

        if found_board and corners is not None:
            cv2.drawChessboardCorners(display, (BOARD_W, BOARD_H), corners, found_board)

        # Efecto visual de destello al capturar
        if flash_frames > 0:
            display = np.clip(display.astype(int) + 80, 0, 255).astype(np.uint8)
            flash_frames -= 1

        # ── Renderizado fluido y escalado en Pygame ──
        cur_w, cur_h = screen.get_size()
        bar_h = 40
        video_area_h = cur_h - bar_h

        frame_h, frame_w = display.shape[:2]
        scale = min(cur_w / frame_w, video_area_h / frame_h)
        new_w, new_h = int(frame_w * scale), int(frame_h * scale)
        offset_x, offset_y = (cur_w - new_w) // 2, (video_area_h - new_h) // 2

        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        
        if (new_w, new_h) != (frame_w, frame_h):
            display_rgb = cv2.resize(display_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        surface = pygame.surfarray.make_surface(display_rgb.swapaxes(0, 1))
        screen.fill((15, 15, 15))
        screen.blit(surface, (offset_x, offset_y))

        # ── Barra de estado inferior ──
        pygame.draw.rect(screen, (25, 25, 25), (0, video_area_h, cur_w, bar_h))
        
        status_color = (0, 230, 100) if found_board else (255, 160, 0)
        status_txt = f"{'✓ DETECTADO' if found_board else 'Buscando patrón...'} | Capturas: {count}"
        focus_txt = f"Foco manual: {current_focus} (▲/▼)"

        screen.blit(font.render(status_txt, True, status_color), (15, video_area_h + 10))
        screen.blit(font.render(focus_txt, True, (200, 200, 200)), (cur_w - 280, video_area_h + 10))

        pygame.display.flip()
        clock.tick(30)

    cap.release()
    pygame.quit()
    print(f"\n🏁 Proceso completado. Total de imágenes listas para calibrar: {count}")

if __name__ == '__main__':
    main()