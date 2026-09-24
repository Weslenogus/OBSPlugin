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
            # Croissance monotone *avec marge* (×1,5, comme un tableau
            # dynamique) : des appelants de tailles différentes, ou une zone de
            # texte qui grandit de quelques pixels par image quand la feuille
            # approche, ne doivent pas régénérer la réserve à chaque image.
            shape = (max(bh, int(height * 3), 64), max(bw, int(width * 3), 64))
            self._bank = self._rng.standard_normal(shape, dtype=np.float32)
        bh, bw = self._bank.shape
        y = int(self._rng.integers(0, bh - height + 1))
        x = int(self._rng.integers(0, bw - width + 1))
        return self._bank[y:y + height, x:x + width]


# Poids de luminance Rec. 601 dans l'ordre BGR d'OpenCV.
LUMA_BGR = np.array([0.114, 0.587, 0.299], np.float32)


def luminance(bgr: np.ndarray) -> np.ndarray:
    """Luminance Rec. 601 (float32, 0..255)."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    return 0.114 * b + 0.587 * g + 0.299 * r


def estimate_noise_sigma(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Écart-type du bruit capteur visible sur le papier (hors encre).

    Résidu haute fréquence (image - flou) puis estimateur MAD, robuste aux
    quelques traits d'encre restants. Sert à calibrer le grain ISO ajouté sur
    l'encre, et le grain réinjecté dans le papier effacé, sur la vraie caméra.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    residual = gray - cv2.GaussianBlur(gray, (0, 0), 1.5)
    values = residual[mask == 0] if mask is not None else residual.ravel()
    if values.size < 100:
        return 0.0
    values = values[:: max(1, values.size // 20000)]  # sous-échantillonnage
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def _ring_box(shape: tuple[int, ...], box_px: tuple[int, int, int, int],
              ring: int) -> tuple[int, int, int, int]:
    """Boîte de texte élargie de ``ring`` px et bornée au canevas."""
    h, w = shape[:2]
    x0, y0, x1, y1 = box_px
    x0, y0 = max(0, x0 - ring), max(0, y0 - ring)
    x1, y1 = min(w, x1 + ring), min(h, y1 + ring)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, w, h
    return x0, y0, x1, y1


def measure_paper_color(canvas: np.ndarray, text_mask: np.ndarray | None,
                        box_px: tuple[int, int, int, int], ring: int) -> np.ndarray:
    """Couleur moyenne (BGR, float32) du papier autour de la zone de texte.

    La mesure porte sur le canevas rectifié, dans la boîte de texte élargie
    de ``ring`` pixels, en excluant l'encre d'origine détectée (il ne reste
    que du papier). Moyenne tronquée (luminance entre les percentiles 10 et
    90) pour ignorer reflets et taches.
    """
    x0, y0, x1, y1 = _ring_box(canvas.shape, box_px, ring)
    # Sous-échantillonnage 1/4 : des milliers d'échantillons restent largement
    # suffisants pour une moyenne, pour une fraction du coût.
    pixels = canvas[y0:y1:4, x0:x1:4].reshape(-1, 3).astype(np.float32)
    if text_mask is not None:
        keep = text_mask[y0:y1:4, x0:x1:4].ravel() == 0
        if keep.sum() > 50:
            pixels = pixels[keep]
    if pixels.size == 0:
        return np.full(3, 200.0, np.float32)
    lum = pixels @ LUMA_BGR
    lo, hi = np.percentile(lum, (10, 90))
    core = pixels[(lum >= lo) & (lum <= hi)]
    return (core if len(core) else pixels).mean(axis=0).astype(np.float32)


def measure_paper_luminance(canvas: np.ndarray, text_mask: np.ndarray | None,
                            box_px: tuple[int, int, int, int], ring: int) -> float:
    """Luminance moyenne du papier autour de la zone (voir ``measure_paper_color``)."""
    return float(measure_paper_color(canvas, text_mask, box_px, ring) @ LUMA_BGR)


def measure_ink_color(canvas: np.ndarray, text_mask: np.ndarray | None,
                      paper_bgr: np.ndarray, box_px: tuple[int, int, int, int],
                      ring: int, min_pixels: int = 30) -> np.ndarray | None:
    """Couleur réelle (BGR) de l'encre imprimée, échantillonnée autour de la zone.

    Le masque de texte est dilaté : il contient aussi le halo anticrénelé,
    plus clair, autour des traits. On ne garde donc que les pixels nettement
    plus sombres que le papier, puis les 30 % les plus sombres d'entre eux
    (le cœur des traits), dont on prend la médiane. On cherche d'abord
    autour de la zone du nouveau texte, puis sur toute la feuille. Renvoie
    ``None`` s'il n'y a pas d'encre visible (feuille vierge).
    """
    if text_mask is None or not text_mask.any():
        return None
    paper_lum = float(paper_bgr @ LUMA_BGR)
    regions = [_ring_box(canvas.shape, box_px, ring), (0, 0, canvas.shape[1], canvas.shape[0])]
    for x0, y0, x1, y1 in regions:
        # Sous-échantillonnage 1/2 : les traits font plusieurs pixels de large.
        pixels = canvas[y0:y1:2, x0:x1:2][text_mask[y0:y1:2, x0:x1:2] > 0]
        pixels = pixels.astype(np.float32)
        if len(pixels) < min_pixels:
            continue
        lum = pixels @ LUMA_BGR
        dark = lum < 0.8 * paper_lum
        if dark.sum() < min_pixels:
            continue
        pixels, lum = pixels[dark], lum[dark]
        core = pixels[lum <= np.percentile(lum, 30)]
        return np.median(core, axis=0).astype(np.float32)
    return None


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


def ink_multiply_factors(paper_bgr: np.ndarray, ink_bgr: np.ndarray | None,
                         cfg: PhotometryConfig) -> np.ndarray:
    """Facteurs Produit par canal (B, G, R) de l'encre pleine.

    Avec l'encre échantillonnée : ``facteur = encre / papier`` par canal. À
    couverture pleine, ``papier × facteur`` redonne *exactement* la teinte de
    l'encre réelle, tout en laissant plis, ombres et grain du papier moduler
    le résultat (fusion Produit). Sans encre visible, repli sur un gris
    ``L_papier × ink_ratio`` teinté par ``ink_tint``.
    """
    paper = np.maximum(np.asarray(paper_bgr, np.float32), 1.0)
    if ink_bgr is not None:
        factors = np.asarray(ink_bgr, np.float32) / paper
    else:
        m = ink_multiply_factor(float(paper @ LUMA_BGR), cfg)
        factors = m * np.asarray(cfg.ink_tint, np.float32)
    return np.clip(factors, 0.0, 1.0).astype(np.float32)


def composite_ink(roi_bgr: np.ndarray, coverage: np.ndarray,
                  factors: np.ndarray, cfg: PhotometryConfig,
                  noise: NoiseBank, noise_sigma: float | None = None) -> np.ndarray:
    """Applique l'encre sur ``roi_bgr`` (uint8) et renvoie une nouvelle ROI.

    ``coverage`` est la couverture de l'encre (float32 0..1) déjà déformée
    dans l'espace image de la ROI ; ``factors`` les facteurs Produit par
    canal de l'encre pleine (:func:`ink_multiply_factors`). ``noise_sigma``
    : écart-type du grain ISO (calibré sur la caméra) ; défaut ``cfg``.
    """
    if cfg.blur_sigma > 0:
        # Flou gaussien : l'encre subit la même défocalisation que la scène.
        coverage = cv2.GaussianBlur(coverage, (0, 0), cfg.blur_sigma)
    if not np.any(coverage > 1e-3):
        return roi_bgr

    # Calque Produit : 1 hors encre, facteur de l'encre sur l'encre pleine.
    layer = cv2.merge([1.0 - coverage * (1.0 - float(f)) for f in factors])
    out = cv2.multiply(roi_bgr, layer, dtype=cv2.CV_32F)

    sigma = cfg.noise_sigma if noise_sigma is None else noise_sigma
    if sigma > 0:
        # Grain capteur : la multiplication a atténué le grain d'origine sous
        # l'encre ; on réinjecte un bruit léger (luminance) à cet endroit.
        grain = noise.sample(*coverage.shape) * (sigma * coverage)
        out += grain[..., None]

    return np.clip(out, 0, 255, out=out).astype(np.uint8)
