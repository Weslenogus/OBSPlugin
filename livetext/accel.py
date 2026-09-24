"""Accélération GPU optionnelle (``cv2.cuda``), avec repli CPU garanti.

Le GPU ne gagne qu'à deux conditions : des opérations volumineuses, et une
image envoyée **une seule fois** (un envoi 1080p coûte à peu près autant que
la déformation CPU qu'il remplace). D'où une *session par image* :

* ``CpuFrame`` — chaîne de référence (code CPU d'origine) ;
* ``GpuFrame`` — même chaîne sur GPU : l'image est envoyée une fois, puis
  rectification, déformation des extraits, effacement (mélange) et encre
  (flou, fusion Produit, grain) s'enchaînent sur le GPU ; seul le rectangle
  modifié revient vers le CPU.

Restent sur CPU, faute d'équivalent CUDA dans OpenCV : ``cv2.inpaint``,
détection du texte, suivi LK. Le GPU n'est activé qu'après un **auto-test**
comparant ``GpuFrame`` à ``CpuFrame`` ; toute erreur ensuite le désactive et
l'image est recalculée sur CPU, sans couper le direct.

``cv2.cuda`` exige un OpenCV compilé avec CUDA : ``scripts/build_opencv_cuda``.
Diagnostic : ``python -m livetext.accel``.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .compositor import NoiseBank, composite_ink
from .config import PhotometryConfig
from .geometry import patch_homography, patch_roi, warp_canvas_patch

log = logging.getLogger(__name__)


def cuda_device_count() -> int:
    """Nombre de GPU CUDA visibles par OpenCV (0 si OpenCV sans CUDA)."""
    try:
        return int(cv2.cuda.getCudaEnabledDeviceCount())
    except (AttributeError, cv2.error):
        return 0


def _gaussian_ksize(sigma: float) -> int:
    """Taille de noyau qu'OpenCV choisit pour un flou float (``ksize=(0, 0)``)."""
    return max(3, int(round(sigma * 8 + 1)) | 1)


# ---------------------------------------------------------------------------
# Sessions : une par image
# ---------------------------------------------------------------------------

class CpuFrame:
    """Chaîne de composition sur CPU : la référence."""

    def __init__(self, frame: np.ndarray, noise: NoiseBank):
        self.frame = frame
        self.noise = noise

    def warp(self, M, dsize, flags=cv2.INTER_LINEAR, border=cv2.BORDER_CONSTANT):
        return cv2.warpPerspective(self.frame, M, dsize, flags=flags, borderMode=border)

    def compose(self, H, erase, ink, factors, cfg: PhotometryConfig,
                noise_sigma: float) -> np.ndarray:
        """Efface puis encre ; ``erase`` = (plaque, masque, origine) ou None,
        ``ink`` = (couverture, origine) ou None. Renvoie l'image finale."""
        out = self.frame.copy()
        if erase is not None:
            clean, soft, origin = erase
            c = warp_canvas_patch(clean, origin, H, out.shape, border=cv2.BORDER_REPLICATE)
            m = warp_canvas_patch(soft, origin, H, out.shape)
            if c is not None and m is not None:
                (clean_img, (x0, y0, x1, y1)), (mask, _) = c, m
                out[y0:y1, x0:x1] = cv2.blendLinear(out[y0:y1, x0:x1], clean_img,
                                                    1.0 - mask, mask)
        if ink is not None:
            coverage, origin = ink
            w = warp_canvas_patch(coverage, origin, H, out.shape)
            if w is not None:
                cov_img, (x0, y0, x1, y1) = w
                out[y0:y1, x0:x1] = composite_ink(out[y0:y1, x0:x1], cov_img, factors,
                                                  cfg, self.noise, noise_sigma)
        return out


class GpuFrame:
    """Même chaîne que :class:`CpuFrame` sur GPU, image envoyée une seule fois.

    Toutes les couches sont déformées dans le rectangle *union* des zones
    effacée et encrée, puis combinées par fonctions pures (mélange, Produit,
    grain) : aucune écriture en place dans une vue GPU, un seul transfert
    retour. Mathématiquement identique au CPU : hors d'une couche, son poids
    vaut 0 et son facteur Produit 1.
    """

    def __init__(self, frame: np.ndarray, accel: Accelerator):
        self.frame = frame
        self.accel = accel
        self.g = cv2.cuda_GpuMat()
        self.g.upload(frame)

    def warp(self, M, dsize, flags=cv2.INTER_LINEAR, border=cv2.BORDER_CONSTANT):
        return cv2.cuda.warpPerspective(self.g, M, dsize, flags=flags,
                                        borderMode=border).download()

    def _layer(self, patch, origin, H, union, border):
        """Extrait du canevas déformé directement dans le rectangle union."""
        x0, y0, x1, y1 = union
        src = cv2.cuda_GpuMat()
        src.upload(np.ascontiguousarray(patch))
        return cv2.cuda.warpPerspective(src, patch_homography(H, origin, union),
                                        (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                        borderMode=border)

    def compose(self, H, erase, ink, factors, cfg: PhotometryConfig,
                noise_sigma: float) -> np.ndarray:
        shape = self.frame.shape
        roi_e = patch_roi(erase[0].shape, erase[2], H, shape) if erase is not None else None
        roi_i = patch_roi(ink[0].shape, ink[1], H, shape) if ink is not None else None
        rois = [r for r in (roi_e, roi_i) if r is not None]
        out = self.frame.copy()
        if not rois:
            return out
        union = (min(r[0] for r in rois), min(r[1] for r in rois),
                 max(r[2] for r in rois), max(r[3] for r in rois))
        x0, y0, x1, y1 = union
        base = cv2.cuda_GpuMat(self.g, (x0, y0, x1 - x0, y1 - y0))   # lecture seule

        if roi_e is not None:
            clean, soft, origin = erase
            g_clean = self._layer(clean, origin, H, union, cv2.BORDER_REPLICATE)
            g_mask = self._layer(soft, origin, H, union, cv2.BORDER_CONSTANT)
            inverse = cv2.cuda.addWeighted(g_mask, -1.0, g_mask, 0.0, 1.0)   # 1 - m
            base = cv2.cuda.blendLinear(base, g_clean, inverse, g_mask)

        if roi_i is not None:
            coverage, origin = ink
            g_cov = self._layer(coverage, origin, H, union, cv2.BORDER_CONSTANT)
            if cfg.blur_sigma > 0:
                g_cov = self.accel.gaussian(cfg.blur_sigma).apply(g_cov)
            # Calque Produit par canal : 1 - cov·(1 - f) = cov·(f - 1) + 1.
            # ``dst`` explicite : sans lui, la première surcharge de merge()
            # renvoie un tableau numpy (téléchargé sur le CPU) — piège des
            # liaisons Python, vérifié sur les signatures réelles.
            layer = cv2.cuda.merge([cv2.cuda.addWeighted(g_cov, float(f) - 1.0, g_cov, 0.0, 1.0)
                                    for f in factors], cv2.cuda_GpuMat())
            result = cv2.cuda.multiply(base.convertTo(cv2.CV_32FC3), layer)
            if noise_sigma > 0:
                grain = cv2.cuda.multiply(self.accel.noise(y1 - y0, x1 - x0), g_cov,
                                          scale=float(noise_sigma))
                result = cv2.cuda.add(result, cv2.cuda.merge([grain, grain, grain],
                                                             cv2.cuda_GpuMat()))
            base = result.convertTo(cv2.CV_8UC3)   # arrondi + saturation 0-255

        out[y0:y1, x0:x1] = base.download()
        return out


# ---------------------------------------------------------------------------
# Accélérateur
# ---------------------------------------------------------------------------

class Accelerator:
    """Choisit la session (GPU vérifié ou CPU) pour chaque image.

    ``mode`` : ``"off"`` (CPU), ``"on"`` (GPU exigé, avertit s'il manque) ou
    ``"auto"`` (GPU s'il passe l'auto-test, sinon CPU silencieusement).
    """

    def __init__(self, mode: str = "auto", seed: int | None = None):
        self.mode = mode
        self.enabled = False
        self.reason = "désactivé"
        self._rng = np.random.default_rng(seed)
        self._filters: dict[float, object] = {}
        self._noise_gpu = None
        if mode == "off":
            return
        count = cuda_device_count()
        if count == 0:
            self.reason = "OpenCV sans CUDA ou aucun GPU"
        else:
            ok, detail = self.self_test()
            self.enabled = ok
            self.reason = f"CUDA ({count} GPU)" if ok else f"auto-test GPU échoué ({detail})"
        if mode == "on" and not self.enabled:
            log.warning("GPU demandé mais indisponible (%s) : calcul sur CPU.", self.reason)
        else:
            log.info("Accélération : %s", "GPU " + self.reason if self.enabled
                     else "CPU (" + self.reason + ")")

    # -- ressources GPU (créées une fois) ------------------------------------

    def gaussian(self, sigma: float):
        key = round(float(sigma), 3)
        if key not in self._filters:
            k = min(_gaussian_ksize(key), 31)  # les filtres CUDA sont limités à 32
            self._filters[key] = cv2.cuda.createGaussianFilter(
                cv2.CV_32FC1, cv2.CV_32FC1, (k, k), key)
        return self._filters[key]

    def noise(self, height: int, width: int):
        """Fenêtre aléatoire d'une réserve de bruit N(0, 1) résidant sur le GPU."""
        bank = self._noise_gpu
        if bank is None or bank.size()[1] < height * 2 or bank.size()[0] < width * 2:
            h, w = max(height * 3, 64), max(width * 3, 64)
            if bank is not None:
                h, w = max(h, bank.size()[1]), max(w, bank.size()[0])
            self._noise_gpu = cv2.cuda_GpuMat()
            self._noise_gpu.upload(self._rng.standard_normal((h, w), dtype=np.float32))
            bank = self._noise_gpu
        bw, bh = bank.size()
        y = int(self._rng.integers(0, bh - height + 1))
        x = int(self._rng.integers(0, bw - width + 1))
        return cv2.cuda_GpuMat(bank, (x, y, width, height))

    # -- sessions ------------------------------------------------------------

    def frame(self, frame: np.ndarray, noise: NoiseBank):
        """Session pour une image : GPU si actif, sinon CPU."""
        if self.enabled:
            return GpuFrame(frame, self)
        return CpuFrame(frame, noise)

    def disable(self, reason: str) -> None:
        if self.enabled:
            log.warning("GPU désactivé, bascule définitive sur CPU : %s", reason)
        self.enabled = False
        self.reason = reason

    def self_test(self) -> tuple[bool, str]:
        """Compare ``GpuFrame`` à ``CpuFrame`` sur une scène de test (sans grain)."""
        try:
            rng = np.random.default_rng(0)
            frame = cv2.GaussianBlur(rng.integers(40, 230, (240, 320, 3), dtype=np.uint8),
                                     (0, 0), 2)
            H = np.array([[0.9, 0.05, 30], [-0.04, 0.95, 20], [1e-4, 5e-5, 1]])
            clean = np.full((60, 90, 3), (200, 210, 220), np.uint8)
            soft = np.zeros((60, 90), np.float32)
            soft[10:50, 10:80] = 1.0
            cov = np.zeros((60, 90), np.float32)
            cov[20:40, 15:75] = 1.0
            cfg = PhotometryConfig(blur_sigma=0.8)
            factors = np.array([0.6, 0.3, 0.2], np.float32)
            outs = []
            for session in (CpuFrame(frame, NoiseBank(0)), GpuFrame(frame, self)):
                canvas = session.warp(H, (200, 150), cv2.INTER_LINEAR, cv2.BORDER_REPLICATE)
                out = session.compose(H, (clean, soft, (40.0, 30.0)), (cov, (40.5, 30.25)),
                                      factors, cfg, 0.0)
                outs.append((canvas, out))
            (cpu_canvas, cpu_out), (gpu_canvas, gpu_out) = outs
            for name, cpu, gpu in (("rectification", cpu_canvas, gpu_canvas),
                                   ("composition", cpu_out, gpu_out)):
                diff = np.abs(cpu.astype(np.int16) - gpu.astype(np.int16))
                if cpu.shape != gpu.shape or diff.mean() > 1.0:
                    return False, f"{name} : écart moyen {diff.mean():.2f}"
            return True, "conforme"
        except Exception as exc:  # noqa: BLE001 - tout échec = pas de GPU
            return False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Diagnostic : python -m livetext.accel
# ---------------------------------------------------------------------------

def main() -> int:
    import time

    from .config import AppConfig
    from .pipeline import TextReplacementPipeline
    from .sources import SyntheticSource
    from .tracker import PlanarTracker, detect_document_quad

    logging.basicConfig(level=logging.WARNING)
    info = cv2.getBuildInformation()
    cuda_line = next((line.strip() for line in info.splitlines()
                      if line.strip().startswith("NVIDIA CUDA")), "NVIDIA CUDA: NO")
    print(f"OpenCV {cv2.__version__} | {cuda_line}")
    count = cuda_device_count()
    print(f"GPU CUDA visibles : {count}")
    if count:
        cv2.cuda.printShortCudaDeviceInfo(cv2.cuda.getDevice())
    accel = Accelerator("auto")
    print(f"Auto-test : {'GPU ACTIF' if accel.enabled else 'CPU'} ({accel.reason})")

    frames = SyntheticSource(1920, 1080, seed=3)
    frames = [frames.read() for _ in range(61)]
    quad = detect_document_quad(frames[0])
    for mode in (["off", "auto"] if accel.enabled else ["off"]):
        cfg = AppConfig(gpu=mode)
        cfg.text.initial_text = "Texte remplacé\nen direct"
        tracker = PlanarTracker(cfg.tracker)
        tracker.initialize(frames[0], quad)
        pipe = TextReplacementPipeline(cfg, seed=1, accel=accel if mode != "off" else None)
        pipe.set_target(quad)
        times = []
        for f in frames[1:]:
            t = time.perf_counter()
            pipe.process(f, tracker.update(f))
            times.append((time.perf_counter() - t) * 1e3)
        t = np.array(times[5:])
        print(f"1080p {'GPU' if mode != 'off' else 'CPU'} : {t.mean():5.1f} ms/image "
              f"(p95 {np.percentile(t, 95):5.1f} ms, budget 33,3 ms à 30 i/s)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
