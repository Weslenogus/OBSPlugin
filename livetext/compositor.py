"""Intégration photométrique du texte : fusion Produit, luminosité, grain.

Principe de la fusion « Produit » (Multiply) : ``résultat = base × calque``
avec un calque blanc (1.0) hors de l'encre et gris foncé sur l'encre. Le
papier réel (plis, ombres, grain) module donc directement l'encre
numérique, exactement comme une encre physique qui absorbe la lumière.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import PhotometryConfig


class NoiseBank:
    """Réserve de bruit gaussien pré-calculé, découpée aléatoirement.

    Générer ~2 millions d'échantillons normaux par image (1080p) coûterait
    plusieurs millisecondes ; on tire plutôt une fenêtre aléatoire dans une
    texture de bruit calculée une fois.
    """

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)
        self._bank: np.ndarray | None = None

    def sample(self, height: int, width: int) -> np.ndarray:
        """Bruit N(0, 1) float32 de forme (height, width)."""
        bh, bw = (0, 0) if self._bank is None else self._bank.shape
        if bh < height * 2 or bw < width * 2:
            # Croissance monotone : plusieurs appelants aux tailles différentes
            # ne doivent pas se forcer mutuellement à régénérer la réserve.
            shape = (max(bh, height * 2, 64), max(bw, width * 2, 64))
            self._bank = self._rng.standard_normal(shape, dtype=np.float32)
        bh, bw = self._bank.shape
        y = int(self._rng.integers(0, bh - height + 1))
        x = int(self._rng.integers(0, bw - width + 1))
        return self._bank[y:y + height, x:x + width]


def luminance(bgr: np.ndarray) -> np.ndarray:
    """Luminance Rec. 601 (float32, 0..255)."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    return 0.114 * b + 0.587 * g + 0.299 * r


def measure_paper_luminance(canvas: np.ndarray,
                            text_mask: np.ndarray | None,
                            box_px: tuple[int, int, int, int],
                            ring: int) -> float:
    """Luminosité moyenne du papier autour de la zone de texte.

    La mesure porte sur le canevas rectifié, dans la boîte de texte élargie
    de ``ring`` pixels, en excluant les pixels détectés comme de l'encre
    d'origine (il ne reste donc que du papier). On utilise une moyenne
    tronquée (percentiles 10-90) pour ignorer reflets et taches.
    """
    h, w = canvas.shape[:2]
    x0, y0, x1, y1 = box_px
    x0, y0 = max(0, x0 - ring), max(0, y0 - ring)
    x1, y1 = min(w, x1 + ring), min(h, y1 + ring)
    if x1 <= x0 or y1 <= y0:
        x0, y0, x1, y1 = 0, 0, w, h
    # Sous-échantillonnage 1/2 : largement suffisant pour une moyenne.
    lum = luminance(canvas[y0:y1:2, x0:x1:2])
    if text_mask is not None:
        keep = text_mask[y0:y1:2, x0:x1:2] == 0
        if keep.sum() > 50:
            lum = lum[keep]
    lum = lum.ravel()
    if lum.size == 0:
        return 200.0
    lo, hi = np.percentile(lum, (10, 90))
    core = lum[(lum >= lo) & (lum <= hi)]
    return float(core.mean() if core.size else lum.mean())


def ink_multiply_factor(paper_luminance: float, cfg: PhotometryConfig) -> float:
    """Facteur Produit de l'encre pleine, adapté à la luminosité du papier.

    La luminance visée de l'encre est ``L_papier × ink_ratio``, bornée par
    ``ink_min`` (encre jamais plus claire qu'un gris minimal) et par la
    luminance du papier lui-même. Sur un papier sombre (mal éclairé), le
    rapport est donc automatiquement adouci pour que l'encre ne devienne pas
    un noir pur « collé ».
    """
    paper = max(float(paper_luminance), 1.0)
    target = float(np.clip(paper * cfg.ink_ratio, min(cfg.ink_min, paper), paper))
    return target / paper


def composite_ink(roi_bgr: np.ndarray, coverage: np.ndarray,
                  paper_luminance: float, cfg: PhotometryConfig,
                  noise: NoiseBank) -> np.ndarray:
    """Applique l'encre sur ``roi_bgr`` (uint8) et renvoie une nouvelle ROI.

    ``coverage`` est la couverture de l'encre (float32 0..1) déjà déformée
    dans l'espace image de la ROI.
    """
    if cfg.blur_sigma > 0:
        # Flou gaussien : l'encre subit la même défocalisation que la scène.
        coverage = cv2.GaussianBlur(coverage, (0, 0), cfg.blur_sigma)
    if not np.any(coverage > 1e-3):
        return roi_bgr

    m = ink_multiply_factor(paper_luminance, cfg)
    # Calque Produit : 1 hors encre, m × teinte sur l'encre pleine.
    layer = cv2.merge([1.0 - coverage * (1.0 - float(np.clip(m * t, 0.0, 1.0)))
                       for t in cfg.ink_tint])
    out = cv2.multiply(roi_bgr, layer, dtype=cv2.CV_32F)

    if cfg.noise_sigma > 0:
        # Grain capteur : la multiplication a atténué le grain d'origine sous
        # l'encre ; on réinjecte un bruit léger (luminance) à cet endroit.
        grain = noise.sample(*coverage.shape) * (cfg.noise_sigma * coverage)
        out += grain[..., None]

    return np.clip(out, 0, 255, out=out).astype(np.uint8)
