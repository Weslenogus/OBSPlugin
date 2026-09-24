"""Sorties : caméra virtuelle (OBS), fenêtre d'aperçu, fichier vidéo."""

from __future__ import annotations

import logging
import queue
import threading

import cv2
import numpy as np

log = logging.getLogger(__name__)

WINDOW_NAME = "livetext - apercu"


class FrameSink:
    """Interface commune d'une sortie vidéo."""

    name = "sink"

    def send(self, frame: np.ndarray) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass


class VirtualCamSink(FrameSink):
    """Envoie le flux vers OBS via ``pyvirtualcam``.

    * Windows / macOS : backend « OBS Virtual Camera » (OBS ≥ 26 installé).
    * Linux : module noyau ``v4l2loopback`` (voir README).

    Dans OBS, ajoutez une source « Périphérique de capture vidéo » pointant sur
    cette caméra virtuelle.
    """

    name = "virtualcam"

    def __init__(self, width: int, height: int, fps: int, backend: str | None = None):
        try:
            import pyvirtualcam
        except ImportError as exc:
            raise RuntimeError(
                "pyvirtualcam n'est pas installé (pip install pyvirtualcam)"
            ) from exc
        # Format BGR natif d'OpenCV : aucune conversion de couleur par image.
        self._cam = pyvirtualcam.Camera(width, height, fps,
                                        fmt=pyvirtualcam.PixelFormat.BGR,
                                        backend=backend, print_fps=False)
        log.info("Caméra virtuelle ouverte : %s (%dx%d @ %d i/s)",
                 self._cam.device, width, height, fps)

    def send(self, frame: np.ndarray) -> None:
        self._cam.send(np.ascontiguousarray(frame))

    def close(self) -> None:
        self._cam.close()


class WindowSink(FrameSink):
    """Fenêtre d'aperçu OpenCV ; capte aussi le clavier (``cv2.waitKey``)."""

    name = "window"

    def __init__(self, width: int, height: int):
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            raise RuntimeError(
                "OpenCV est compilé sans interface graphique (paquet "
                "opencv-python-headless ?) ; installez opencv-python ou "
                "retirez la sortie « window »."
            ) from exc
        # Fenêtre à taille raisonnable même en 1080p.
        scale = min(1.0, 1280 / width)
        cv2.resizeWindow(WINDOW_NAME, int(width * scale), int(height * scale))

    def send(self, frame: np.ndarray) -> None:
        cv2.imshow(WINDOW_NAME, frame)

    @staticmethod
    def poll_key() -> int:
        """Touche pressée (code) ou -1 ; non bloquant (1 ms)."""
        return cv2.waitKeyEx(1)

    def close(self) -> None:
        cv2.destroyWindow(WINDOW_NAME)


class VideoFileSink(FrameSink):
    """Enregistre le flux dans un fichier (tests sans OBS, archivage)."""

    name = "file"

    def __init__(self, path: str, width: int, height: int, fps: int):
        fourcc = cv2.VideoWriter_fourcc(*("MJPG" if path.lower().endswith(".avi") else "mp4v"))
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Impossible d'écrire la vidéo {path}")
        self.path = path
        log.info("Enregistrement vers %s", path)

    def send(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()


class ThreadedSink(FrameSink):
    """Délègue l'envoi à un thread : conversion/encodage en parallèle du calcul.

    * ``drop=True``  (caméra virtuelle) : file de 1, la dernière image gagne ;
      un envoi lent ne crée jamais de retard cumulé.
    * ``drop=False`` (fichier) : file bornée bloquante, aucune image perdue.

    Contrat : une image passée à :meth:`send` ne doit plus être modifiée.
    """

    def __init__(self, inner: FrameSink, drop: bool):
        self.inner = inner
        self.name = inner.name
        self._drop = drop
        self._queue: queue.Queue = queue.Queue(maxsize=1 if drop else 8)
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name=f"sink-{inner.name}",
                                        daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            frame = self._queue.get()
            if frame is None:
                return
            try:
                self.inner.send(frame)
            except Exception as exc:  # noqa: BLE001 - remonté au prochain send
                self.error = exc

    def send(self, frame: np.ndarray) -> None:
        if self.error is not None:
            error, self.error = self.error, None
            raise RuntimeError(f"Sortie {self.name} : {error}") from error
        if self._drop:
            try:
                self._queue.get_nowait()  # jette l'image non encore envoyée
            except queue.Empty:
                pass
            self._queue.put_nowait(frame)
        else:
            self._queue.put(frame)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5.0)
        self.inner.close()


def open_sinks(specs: list[str], width: int, height: int, fps: int,
               backend: str | None = None) -> list[FrameSink]:
    """Ouvre les sorties demandées ; une sortie en échec est signalée et ignorée.

    Lève ``RuntimeError`` si *aucune* sortie n'a pu être ouverte.
    """
    sinks: list[FrameSink] = []
    for spec in specs:
        try:
            if spec == "none":
                continue
            if spec == "virtualcam":
                sinks.append(ThreadedSink(VirtualCamSink(width, height, fps, backend),
                                          drop=True))
            elif spec == "window":
                sinks.append(WindowSink(width, height))  # GUI : thread principal
            elif spec.startswith("file:"):
                sinks.append(ThreadedSink(VideoFileSink(spec[5:], width, height, fps),
                                          drop=False))
            else:
                raise RuntimeError(f"Sortie inconnue : {spec!r}")
        except Exception as exc:  # noqa: BLE001 - on veut continuer sans cette sortie
            log.error("Sortie « %s » indisponible : %s", spec, exc)
    if not sinks and any(s != "none" for s in specs):
        raise RuntimeError("Aucune sortie vidéo n'a pu être ouverte.")
    return sinks
