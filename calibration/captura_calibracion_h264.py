"""
PASO 1: Captura para Calibración HD nativo (1280x720) 
Aceleración GPU por GStreamer (H.264) + Pygame UI
Constante física de calibración: Foco = 0
"""

import cv2
import os
import time
import argparse
import pygame
import numpy as np

BOARD_W = 7
BOARD_H = 10

def preconfigurar_kernel(cam_id):
    """
    Limpia los búferes de Linux y bloquea los automatismos de la C920.
    Garantiza que la distancia focal (intrínsecos) sea inmutable.
    """
    print("⚙️ Preparando hardware de la cámara (Bloqueando Foco=0)...")
    os.system(f"v4l2-ctl -d /dev/video{cam_id} --set-ctrl=focus_automatic_continuous=0 >/dev/null 2>&1")
    os.system(f"v4l2-ctl -d /dev/video{cam_id} --set-ctrl=exposure_dynamic_framerate=0 >/dev/null 2>&1")
    os.system(f"v4l2-ctl -d /dev/video{cam_id} --set-ctrl=focus_absolute=0 >/dev/null 2>&1")
    time.sleep(0.3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=4)
    parser.add_argument('--output', type=str, default='./fotos')
    parser.add_argument('--win-width', type=int, default=1280)
    parser.add_argument('--win-height', type=int, default=760) # 720p video + 40px HUD
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── 1. ASEGURAR HARDWARE Y FOCO ──
    preconfigurar_kernel(args.camera)

    # ── 2. PIPELINE DE VIDEO ACELERADO POR HARDWARE (H.264) ──
    # Extrae el flujo comprimido por el chip de la C920, pasa por el silicio
    # de tu GPU AMD Radeon y entrega matrices crudas BGR perfectas a OpenCV.
    gst_pipeline = (
        f"v4l2src device=/dev/video{args.camera} ! "
        f"video/x-h264, width=1280, height=720 ! "
        f"decodebin ! videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1"
    )

    print(f"🚀 Inicializando stream HD mediante GPU interna para /dev/video{args.camera}...")
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print(f"❌ Falló el enlace con GStreamer para la cámara {args.camera}.")
        print(f"   💡 Si el puerto sigue ocupado, ejecuta: sudo fuser -k /dev/video{args.camera}")
        return

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ Cámara capturando a: {real_w}x{real_h} nativos (FOV Completo)")

    count = len([f for f in os.listdir(args.output) if f.endswith('.jpg')])
    print(f"   Fotos existentes en carpeta: {count}")

    # ── 3. INICIALIZACIÓN DE PYGAME ──
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pygame.init()
    win_w, win_h = args.win_width, args.win_height
    screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
    pygame.display.set_caption("Calibración - Captura HD Acelerada")
    font = pygame.font.SysFont("monospace", 20)
    clock = pygame.time.Clock()

    last_frame = None
    found_board = False
    flash_frames = 0
    running = True

    print("\n" + "★"*50)
    print(" INTERFAZ LISTA. CONTROLES:")
    print("  • [ESPACIO] : Capturar y guardar foto limpia")
    print("  • [R]       : Recuento de capturas")
    print("  • [+ / -]   : Escalar ventana de previsualización")
    print("  • [ESC]     : Finalizar calibración")
    print("★"*50 + "\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                win_w, win_h = event.size
                screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if found_board and last_frame is not None:
                        filename = os.path.join(args.output, f'calib_{count:03d}.jpg')
                        # Guardamos la matriz original de 1280x720 pura y sin marcas
                        cv2.imwrite(filename, last_frame)
                        count += 1
                        flash_frames = 8
                        print(f"📸 Foto {count:03d} guardada en: {filename}")
                    else:
                        print("⚠️ Tablero no detectado con suficiente claridad")
                elif event.key == pygame.K_r:
                    print(f"📊 Total capturado hasta ahora: {count}")
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    win_w, win_h = int(win_w * 1.1), int(win_h * 1.1)
                    screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
                elif event.key == pygame.K_MINUS:
                    win_w = max(640, int(win_w * 0.9))
                    win_h = max(400, int(win_h * 0.9))
                    screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)

        ret, frame = cap.read()
        if not ret:
            continue

        # Almacenamos el fotograma intacto para guardarlo si presionas ESPACIO
        last_frame = frame.copy()
        display = frame.copy()

        # ── 4. DETECCIÓN DE PATRÓN (Alto Rendimiento) ──
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found_board = False
        corners = None

        # Pre-filtro veloz para no ralentizar los FPS si no hay tablero a la vista
        fast_found, _ = cv2.findChessboardCorners(gray, (BOARD_W, BOARD_H), cv2.CALIB_CB_FAST_CHECK)

        if fast_found:
            try:
                # Búsqueda subpíxel basada en sectores (ideal para imágenes HD nítidas)
                found_board, corners = cv2.findChessboardCornersSB(
                    gray, (BOARD_W, BOARD_H),
                    flags=cv2.CALIB_CB_NORMALIZE_IMAGE  
                )
            except AttributeError:
                pass

            if not found_board:
                found_board, corners = cv2.findChessboardCorners(
                    gray, (BOARD_W, BOARD_H),
                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                )

        if found_board and corners is not None:
            cv2.drawChessboardCorners(display, (BOARD_W, BOARD_H), corners, found_board)

        # Efecto visual de obturador al tomar foto
        if flash_frames > 0:
            display = np.clip(display.astype(int) + 80, 0, 255).astype(np.uint8)
            flash_frames -= 1

        # ── 5. RENDERIZADO FLUIDO EN PYGAME ──
        cur_w, cur_h = screen.get_size()
        bar_h = 40
        video_area_h = cur_h - bar_h

        frame_h, frame_w = display.shape[:2]
        scale = min(cur_w / frame_w, video_area_h / frame_h)
        new_w = int(frame_w * scale)
        new_h = int(frame_h * scale)
        offset_x = (cur_w - new_w) // 2
        offset_y = (video_area_h - new_h) // 2

        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        
        # INTER_LINEAR evita que la UI gráfica introduzca pixelado artificial
        if (new_w, new_h) != (frame_w, frame_h):
            display_rgb = cv2.resize(display_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        surface = pygame.surfarray.make_surface(display_rgb.swapaxes(0, 1))
        screen.fill((15, 15, 15))
        screen.blit(surface, (offset_x, offset_y))

        # ── HUD INFERIOR ──
        if found_board:
            msg = f"✓ PATRÓN FIJADO — [ESPACIO] Captura ({count})"
            color = (0, 230, 100)
        else:
            msg = f"Buscando cuadrícula {BOARD_W}x{BOARD_H}... ({count})"
            color = (255, 160, 0)

        pygame.draw.rect(screen, (25, 25, 25), (0, video_area_h, cur_w, bar_h))
        screen.blit(font.render(msg, True, color), (15, video_area_h + 10))
        
        # Etiqueta de hardware persistente
        hw_msg = "GPU H.264 | Foco: 0"
        screen.blit(font.render(hw_msg, True, (120, 120, 120)), (cur_w - 250, video_area_h + 10))

        pygame.display.flip()
        clock.tick(30)

    cap.release()
    pygame.quit()
    print(f"\n🏁 Sesión finalizada exitosamente. Total de imágenes útiles: {count}")

if __name__ == '__main__':
    main()