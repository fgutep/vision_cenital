"""
Librería de Captura HD Cenital Acelerada por Hardware.
Implementa pre-bloqueos en el driver de Linux y tuberías VA-API H.264.
"""

import cv2
import os
import time
import logging
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger("CargaCam")

class CargaCam:
    """
    Controlador de captura robusto optimizado para cámaras Logitech C920
    con decodificación por GPU en entornos de robótica.
    """
    def __init__(self, cam_id: int = 4, width: int = 1280, height: int = 720, focus_step: int = 0):
        self.cam_id = cam_id
        self.width = width
        self.height = height
        self.focus_step = focus_step
        self.cap: Optional[cv2.VideoCapture] = None
        
    def _preconfigure_v4l2(self) -> None:
        """Aplica bloqueos de hardware estrictos usando llamadas directas al kernel."""
        logger.info(f"Bloqueando intrínsecos de hardware para /dev/video{self.cam_id}...")
        cmds = [
            f"v4l2-ctl -d /dev/video{self.cam_id} --set-ctrl=focus_automatic_continuous=0 >/dev/null 2>&1",
            f"v4l2-ctl -d /dev/video{self.cam_id} --set-ctrl=exposure_dynamic_framerate=0 >/dev/null 2>&1",
            f"v4l2-ctl -d /dev/video{self.cam_id} --set-ctrl=focus_absolute={self.focus_step} >/dev/null 2>&1"
        ]
        for cmd in cmds:
            os.system(cmd)
        time.sleep(0.3)

    def start(self) -> bool:
        """Inicializa el stream puenteando la capa de software nativa hacia VA-API."""
        self._preconfigure_v4l2()
        
        # Tubería de consumo ultra-bajo de USB y latencia cero
        pipeline = (
            f"v4l2src device=/dev/video{self.cam_id} ! "
            f"video/x-h264, width={self.width}, height={self.height} ! "
            f"decodebin ! videoconvert ! video/x-raw, format=BGR ! "
            f"appsink drop=true max-buffers=1"
        )
        
        logger.info("Iniciando tubería GStreamer H.264 -> GPU...")
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        return self.cap.isOpened()

    def read(self) -> Tuple[bool, np.ndarray]:
        """Obtiene el fotograma más reciente disponible en el búfer."""
        if not self.cap or not self.cap.isOpened():
            return False, np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return self.cap.read()

    def release(self) -> None:
        """Libera limpiamente la memoria y los descriptores de archivo del sistema operativo."""
        if self.cap:
            self.cap.release()
            logger.info("Stream cerrado.")

    def __enter__(self):
        if not self.start():
            raise RuntimeError(f"Imposible enlazar cámara en /dev/video{self.cam_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()