"""Accélération GPU optionnelle (CUDA via ``cv2.cuda``), avec repli CPU.

Seule l'opération la plus lourde, la rectification de l'image entière
(``warpPerspective`` 1080p → canevas), passe par le GPU : pour les petits
extraits (glyphes), le transfert CPU ↔ GPU coûterait plus qu'il ne rapporte.

Prudence : ``cv2.cuda`` n'existe que si OpenCV a été compilé avec CUDA (ce
n'est *pas* le cas du paquet pip ``opencv-python``). Le GPU n'est activé
qu'après un **auto-test fonctionnel** (warp GPU comparé au warp CPU sur une
image de test) ; toute erreur ensuite le désactive définitivement et l'on
repasse sur le CPU, sans interrompre le direct.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


def cuda_device_count() -> int:
    """Nombre de GPU CUDA visibles par OpenCV (0 si OpenCV sans CUDA)."""
    try:
        return int(cv2.cuda.getCudaEnabledDeviceCount())
    except (AttributeError, cv2.error):
        return 0


class Accelerator:
    """Façade ``warp_perspective`` : GPU si disponible et vérifié, sinon CPU.

    ``mode`` : ``"off"`` (CPU), ``"on"`` (GPU exigé, avertit s'il manque) ou
    ``"auto"`` (GPU s'il passe l'auto-test, sinon CPU silencieusement).
    """

    def __init__(self, mode: str = "auto"):
        self.mode = mode
        self.enabled = False
        self.reason = "désactivé"
        if mode == "off":
            return
        count = cuda_device_count()
        if count == 0:
            self.reason = "OpenCV sans CUDA ou aucun GPU"
        elif self._self_test():
            self.enabled = True
            self.reason = f"CUDA ({count} GPU)"
        else:
            self.reason = "auto-test GPU échoué"
        if mode == "on" and not self.enabled:
            log.warning("GPU demandé mais indisponible (%s) : calcul sur CPU.", self.reason)
        else:
            log.info("Accélération : %s", "GPU " + self.reason if self.enabled
                     else "CPU (" + self.reason + ")")

    @staticmethod
    def _gpu_warp(src, M, dsize, flags, border):
        gpu = cv2.cuda_GpuMat()
        gpu.upload(src)
        out = cv2.cuda.warpPerspective(gpu, M, dsize, flags=flags, borderMode=border)
        return out.download()

    def _self_test(self) -> bool:
        """Le warp GPU doit reproduire le warp CPU (à l'arrondi près)."""
        try:
            rng = np.random.default_rng(0)
            img = cv2.GaussianBlur(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8),
                                   (0, 0), 2)
            M = np.array([[0.9, 0.05, 10], [-0.04, 0.95, 5], [1e-4, 5e-5, 1]])
            args = (M, (300, 220), cv2.INTER_LINEAR, cv2.BORDER_REPLICATE)
            gpu = self._gpu_warp(img, *args)
            cpu = cv2.warpPerspective(img, M, (300, 220), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
            # Tolérance : GPU et CPU arrondissent l'interpolation différemment.
            diff = np.abs(gpu.astype(np.int16) - cpu.astype(np.int16))
            return gpu.shape == cpu.shape and float(np.mean(diff)) < 1.0
        except Exception as exc:  # noqa: BLE001 - tout échec = pas de GPU
            log.debug("Auto-test GPU : %s", exc)
            return False

    def warp_perspective(self, src: np.ndarray, M: np.ndarray, dsize: tuple[int, int],
                         flags: int = cv2.INTER_LINEAR,
                         border: int = cv2.BORDER_CONSTANT) -> np.ndarray:
        if self.enabled:
            try:
                return self._gpu_warp(src, M, dsize, flags, border)
            except Exception as exc:  # noqa: BLE001 - repli CPU, direct préservé
                self.enabled = False
                self.reason = f"erreur GPU ({exc})"
                log.warning("Erreur GPU, bascule définitive sur CPU : %s", exc)
        return cv2.warpPerspective(src, M, dsize, flags=flags, borderMode=border)
