"""Sources d'images : webcam USB, fichier vidéo, image fixe, scène synthétique."""

from __future__ import annotations

import logging
import math
import threading
import time

import cv2
import numpy as np

from .geometry import Quad

log = logging.getLogger(__name__)


class FrameSource:
    """Interface commune : ``read()`` renvoie une image BGR ou ``None`` (fin)."""

    #: vrai si la source impose elle-même la cadence (caméra en direct)
    live = False

    def read(self) -> np.ndarray | None:  # pragma: no cover - interface
        raise NotImplementedError

    def release(self) -> None:
        pass


class CameraSource(FrameSource):
    """Webcam USB via ``cv2.VideoCapture``, lue dans un thread dédié.

    Le thread lit en continu et ne garde que la dernière image : la boucle
    principale n'accumule donc jamais de retard (latence minimale) même si un
    traitement ponctuel dépasse la durée d'une image.
    """

    live = True

    def __init__(self, index: int, width: int, height: int, fps: int):
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la caméra {index}")
        # MJPG : en YUYV brut, beaucoup de webcams USB 2.0 plafonnent à
        # 5-10 i/s en 1080p ; le flux compressé permet 30 i/s.
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if actual != (width, height):
            log.warning("La caméra fournit %dx%d au lieu de %dx%d demandés ; "
                        "les images seront redimensionnées.", *actual, width, height)
        log.info("Caméra %d : %dx%d à %.0f i/s", index, *actual, actual_fps or fps)
        self._size = (width, height)

        self._lock = threading.Condition()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._read_seq = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="capture", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                log.error("Lecture caméra impossible (caméra débranchée ?)")
                break
            if (frame.shape[1], frame.shape[0]) != self._size:
                frame = cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)
            with self._lock:
                self._frame = frame
                self._seq += 1
                self._lock.notify_all()
        with self._lock:
            self._running = False
            self._lock.notify_all()

    def read(self) -> np.ndarray | None:
        """Attend une image *nouvelle* (jamais deux fois la même)."""
        with self._lock:
            while self._running and self._seq == self._read_seq:
                self._lock.wait(timeout=1.0)
            if self._seq == self._read_seq:
                return None
            self._read_seq = self._seq
            return self._frame

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        self._cap.release()


class VideoFileSource(FrameSource):
    """Fichier vidéo lu en boucle (pratique pour répéter un test)."""

    def __init__(self, path: str, width: int, height: int, loop: bool = True):
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la vidéo {path}")
        self._size = (width, height)
        self._loop = loop

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        if not ok and self._loop:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        if not ok:
            return None
        return cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)

    def release(self) -> None:
        self._cap.release()


class ImageSource(FrameSource):
    """Image fixe répétée indéfiniment."""

    def __init__(self, path: str, width: int, height: int):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Impossible de lire l'image {path}")
        self._frame = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    def read(self) -> np.ndarray | None:
        return self._frame.copy()


class SyntheticSource(FrameSource):
    """Feuille de papier animée (rotation, perspective, éclairage) sur un bureau.

    Permet de tester tout le pipeline sans caméra. ``true_quad`` donne les
    coins exacts de la feuille dans la dernière image produite.
    """

    def __init__(self, width: int, height: int, seed: int = 0,
                 text: str = "Texte d'origine"):
        self._size = (width, height)
        rng = np.random.default_rng(seed)
        s = height / 720.0

        # Papier : blanc cassé, fibres, léger dégradé, texte d'origine.
        self._pw, self._ph = int(640 * s), int(450 * s)
        paper = np.full((self._ph, self._pw, 3), (222, 230, 236), np.float32)
        fibers = cv2.GaussianBlur(rng.normal(0, 6, (self._ph, self._pw)).astype(np.float32),
                                  (0, 0), 1.2)
        paper += fibers[..., None]
        paper = np.clip(paper, 0, 255).astype(np.uint8)
        font, thick = cv2.FONT_HERSHEY_SIMPLEX, max(2, int(5 * s))
        cv2.putText(paper, text, (int(40 * s), int(130 * s)), font, 1.9 * s,
                    (35, 35, 40), thick, cv2.LINE_AA)
        cv2.putText(paper, "Seconde ligne 123", (int(40 * s), int(250 * s)), font,
                    1.3 * s, (45, 45, 50), max(2, thick - 2), cv2.LINE_AA)
        self._paper = paper

        # Bureau texturé (le suivi a besoin de texture hors feuille aussi).
        desk = rng.normal(0, 1, (height // 8 + 1, width // 8 + 1)).astype(np.float32)
        desk = cv2.resize(desk, (width, height), interpolation=cv2.INTER_CUBIC)
        base = np.stack([60 + 12 * desk, 75 + 12 * desk, 95 + 14 * desk], axis=-1)
        self._desk = np.clip(base, 0, 255).astype(np.uint8)
        for _ in range(40):
            c = tuple(int(v) for v in rng.integers(20, 140, 3))
            p = (int(rng.integers(0, width)), int(rng.integers(0, height)))
            cv2.circle(self._desk, p, int(rng.integers(4, 30 * s + 5)), c, -1, cv2.LINE_AA)

        self._paper_mask = np.full((self._ph, self._pw), 255, np.uint8)
        self._noise = np.empty((height, width, 3), np.float32)
        self._seed = seed
        self._frame = 0   # compteur d'images (bruit), distinct du temps de pose _t
        self._xs = np.linspace(0, 1, width, dtype=np.float32)
        self._t = 0
        self.true_quad: Quad | None = None

    def quad_at(self, t: int) -> Quad:
        """Coins de la feuille à l'instant ``t`` (mouvement lent et continu)."""
        w, h = self._size
        phase = t / 30.0
        cx = w / 2 + 0.08 * w * math.sin(phase * 0.7)
        cy = h / 2 + 0.05 * h * math.sin(phase * 0.9 + 1.0)
        angle = math.radians(8 * math.sin(phase * 0.5))
        scale = 1.0 + 0.08 * math.sin(phase * 0.6 + 0.5)
        hw, hh = self._pw / 2 * scale, self._ph / 2 * scale
        base = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], np.float32)
        # Perspective : le haut de la feuille s'éloigne légèrement.
        tilt = 0.06 * (1 + math.sin(phase * 0.4))
        base[:2, 0] *= 1 - tilt
        rot = np.array([[math.cos(angle), -math.sin(angle)],
                        [math.sin(angle), math.cos(angle)]], np.float32)
        return (base @ rot.T + np.array([cx, cy], np.float32)).astype(np.float32)

    def read(self) -> np.ndarray | None:
        w, h = self._size
        quad = self.quad_at(self._t)
        src = np.array([[0, 0], [self._pw - 1, 0], [self._pw - 1, self._ph - 1],
                        [0, self._ph - 1]], np.float32)
        H = cv2.getPerspectiveTransform(src, quad)
        # Composition par masque alpha déformé : bords de feuille anticrénelés,
        # comme une vraie caméra (la détection sous-pixel en dépend).
        paper = cv2.warpPerspective(self._paper, H, (w, h))
        alpha = cv2.warpPerspective(self._paper_mask, H, (w, h)).astype(np.float32) / 255.0
        frame = cv2.blendLinear(paper, self._desk, alpha, 1.0 - alpha)

        # Éclairage : ombre douce qui balaie la scène (ne varie qu'en x).
        light = 0.85 + 0.15 * np.cos(2 * math.pi * (self._xs - 0.1 * math.sin(self._t / 45)))
        out = frame.astype(np.float32)
        out *= light[None, :, None]
        # Bruit capteur *indépendant* à chaque image et par canal. Ne pas tirer
        # des fenêtres d'une texture fixe : d'une image à l'autre ce serait un
        # motif rigide translaté, que le flux optique suit et qui fausse le
        # suivi (erreur moyenne ×3 mesurée). Écart-type par canal : un scalaire
        # seul ne bruiterait que le premier canal.
        # Graine dérivée de (seed, t) : image reproductible quel que soit le
        # reste du programme (le générateur d'OpenCV est global).
        cv2.setRNGSeed(self._seed * 1_000_003 + self._frame)
        cv2.randn(self._noise, (0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
        out += self._noise
        self.true_quad = quad
        self._t += 1
        self._frame += 1
        return np.clip(out, 0, 255, out=out).astype(np.uint8)


def open_source(spec: str, width: int, height: int, fps: int,
                seed: int = 0) -> FrameSource:
    """Ouvre une source d'après ``--source`` : index, chemin ou ``synthetic``."""
    if spec == "synthetic":
        return SyntheticSource(width, height, seed=seed)
    if spec.isdigit():
        return CameraSource(int(spec), width, height, fps)
    lowered = spec.lower()
    if lowered.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")):
        return ImageSource(spec, width, height)
    return VideoFileSource(spec, width, height)


class FramePacer:
    """Cadence une boucle à ``fps`` images/s pour les sources non temps réel."""

    def __init__(self, fps: int):
        self._period = 1.0 / max(1, fps)
        self._next = time.perf_counter()

    def wait(self) -> None:
        self._next += self._period
        delay = self._next - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:  # en retard : on se recale au lieu d'accumuler la dette
            self._next = time.perf_counter()
