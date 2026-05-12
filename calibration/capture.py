"""
PASO 1: Captura de fotos para calibración en Alta Resolución (HD 1280x720 nativo)
"""

import cv2
import os
import argparse
import pygame
import numpy as np

BOARD_W = 8
BOARD_H = 11

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=4)
    parser.add_argument('--output', type=str, default='./fotos')
    parser.add_argument('--win-width', type=int, default=1280)
    parser.add_argument('--win-height', type=int, default=760)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── 1. SOLUCIÓN AL PROBLEMA V4L2 (NEGOCIACIÓN DE RESOLUCIÓN) ──
    # Forzamos los parámetros directamente en la inicialización.
    # Evita que Linux abra en YUYV por defecto y luego rechace el cambio.
    params = [
        cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'),
        cv2.CAP_PROP_FRAME_WIDTH, 1280,
        cv2.CAP_PROP_FRAME_HEIGHT, 720,
        cv2.CAP_PROP_FPS, 30,
        cv2.CAP_PROP_BUFFERSIZE, 1
    ]

    print(f"🔄 Inicializando cámara {args.camera} solicitando HD nativo desde el constructor...")
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2, params)
    
    if not cap.isOpened():
        print(f"❌ No se pudo abrir cámara {args.camera}")
        return

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join([chr((fourcc_int >> 8*i) & 0xFF) for i in range(4)])
    print(f"📷 Cámara {args.camera} configurada a {real_w}x{real_h} fourcc={fourcc}")

    # Fallback de emergencia a nivel de sistema operativo si OpenCV ignoró los params
    if real_w < 1280:
        print("⚠️ ADVERTENCIA: Intentando fallback forzando v4l2-ctl desde la terminal...")
        cap.release()
        os.system(f"v4l2-ctl -d /dev/video{args.camera} --set-fmt-video=width=1280,height=720,pixelformat=MJPG")
        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📷 Fallback resultado: {real_w}x{real_h}")

    count = len([f for f in os.listdir(args.output) if f.endswith('.jpg')])
    print(f"   Fotos existentes: {count}")
    print("   SPACE=capturar | ESC=salir | R=contar | +/-=resize de ventana")

    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pygame.init()
    win_w, win_h = args.win_width, args.win_height
    screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
    pygame.display.set_caption("Calibración - Captura HD")
    font = pygame.font.SysFont("monospace", 20)
    clock = pygame.time.Clock()

    last_frame = None
    found_board = False
    flash_frames = 0
    running = True

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
                        cv2.imwrite(filename, last_frame)
                        count += 1
                        flash_frames = 8
                        print(f"✅ Foto {count} guardada: {filename}")
                    else:
                        print("⚠️ Tablero no detectado en este frame")
                elif event.key == pygame.K_r:
                    print(f"📊 Fotos capturadas: {count}")
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

        last_frame = frame.copy()
        display = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found_board = False
        corners = None

        # ── 2. SOLUCIÓN AL CONGELAMIENTO (PRE-FILTRO DE ALTO RENDIMIENTO) ──
        # FAST_CHECK evalúa en milisegundos si vale la pena buscar un tablero.
        # Evita que la interfaz colapse cuando estás moviendo la cámara.
        fast_found, _ = cv2.findChessboardCorners(gray, (BOARD_W, BOARD_H), cv2.CALIB_CB_FAST_CHECK)

        if fast_found:
            try:
                # Quitamos EXHAUSTIVE; al estar en HD nativo ya no es necesario forzarlo tanto
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

        if flash_frames > 0:
            display = np.clip(display.astype(int) + 80, 0, 255).astype(np.uint8)
            flash_frames -= 1

        # ── Renderizado fluido en Pygame ──
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
        
        if (new_w, new_h) != (frame_w, frame_h):
            display_rgb = cv2.resize(display_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        surface = pygame.surfarray.make_surface(display_rgb.swapaxes(0, 1))
        screen.fill((20, 20, 20))
        screen.blit(surface, (offset_x, offset_y))

        if found_board:
            msg = f"✓ DETECTADO — SPACE captura ({count})"
            color = (0, 230, 100)
        else:
            msg = f"Buscando {BOARD_W}x{BOARD_H}... ({count})"
            color = (255, 160, 0)

        pygame.draw.rect(screen, (30, 30, 30), (0, video_area_h, cur_w, bar_h))
        text_surf = font.render(msg, True, color)
        screen.blit(text_surf, (10, video_area_h + 10))

        pygame.display.flip()
        clock.tick(30)

    cap.release()
    pygame.quit()
    print(f"\n✅ Sesión terminada. Total fotos: {count}")

if __name__ == '__main__':
    main()