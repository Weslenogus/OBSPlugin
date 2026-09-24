"""Pipeline de remplacement du texte pour une image et une cible données.

Étapes pour chaque image (``process``) :

1. rectification de la feuille vers le canevas (``quad → canevas``) ;
2. effacement du texte d'origine dans le canevas ;
3. rendu (mis en cache) du nouveau texte dans le canevas ;
4. déformation ``cv2.warpPerspective`` (canevas → image) de la plaque
   nettoyée, du masque d'effacement et de la couverture d'encre ;
5. intégration photométrique (fusion Produit, luminosité, flou, grain).

Seuls les pixels effacés ou encrés sont modifiés : le reste de la feuille
garde sa résolution native, sans double rééchantillonnage. Chaque calque
n'est déformé que dans son propre rectangle englobant, si bien que le coût
suit la surface du texte et non celle de la feuille.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .compositor import NoiseBank, composite_ink, measure_paper_luminance
from .config import AppConfig
from .geometry import (
    Quad,
    canvas_size_for_quad,
    canvas_to_quad,
    quad_size,
    quad_to_canvas,
    warp_canvas_patch,
)
from .inpaint import EraseResult, TextEraser
from .textrender import TextRenderer

# Lissage temporel de la luminosité mesurée (évite le « pompage » de l'encre).
_LUMINANCE_EMA = 0.2


@dataclass
class FrameDebug:
    """Informations de diagnostic de la dernière image traitée."""

    paper_luminance: float = 0.0
    erased_pixels: int = 0
    canvas: np.ndarray | None = None
    erase: EraseResult | None = None


class TextReplacementPipeline:
    """Efface le texte d'une feuille suivie et y incruste un nouveau texte."""

    def __init__(self, config: AppConfig, seed: int | None = None):
        self.config = config
        self.noise = NoiseBank(seed)
        self.eraser = TextEraser(config.erase, self.noise)
        self.renderer = TextRenderer(config.text)
        self.canvas_size: tuple[int, int] | None = None
        self.text = config.text.initial_text
        self._coverage_key: tuple | None = None
        self._coverage: np.ndarray | None = None
        self._coverage_box = (0, 0, 0, 0)
        self._luminance: float | None = None
        self.debug = FrameDebug()

    # -- configuration de la cible -------------------------------------------

    def set_target(self, quad: Quad) -> None:
        """Fixe la taille du canevas d'après la feuille à l'initialisation."""
        self.canvas_size = canvas_size_for_quad(quad, self.config.canvas_max_side)
        self.eraser.reset()
        self._luminance = None
        self._coverage_key = None

    def set_text(self, text: str) -> None:
        """Change le texte affiché (le rendu est refait à la prochaine image)."""
        self.text = text

    def invalidate_text(self) -> None:
        """Force un nouveau rendu (police, alignement… modifiés à chaud)."""
        self._coverage_key = None

    # -- rendu du texte (mis en cache) ---------------------------------------

    def coverage(self) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Couverture d'encre dans le canevas et sa boîte englobante (px)."""
        assert self.canvas_size is not None
        w, h = self.canvas_size
        key = (self.text, w, h)
        if key != self._coverage_key:
            cov = self.renderer.render(self.text, w, h)
            points = cv2.findNonZero((cov > 0.02).astype(np.uint8))
            if points is not None:
                bx, by, bw, bh = cv2.boundingRect(points)
                box = (bx, by, bx + bw, by + bh)
            else:
                x0, y0, x1, y1 = self.config.text.box
                box = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
            self._coverage, self._coverage_box, self._coverage_key = cov, box, key
        return self._coverage, self._coverage_box

    # -- traitement d'une image ----------------------------------------------

    def process(self, frame: np.ndarray, quad: Quad | None) -> np.ndarray:
        """Renvoie une copie de ``frame`` avec le texte remplacé."""
        if quad is None or self.canvas_size is None:
            return frame

        cw, ch = self.canvas_size
        H_rect = quad_to_canvas(cw, ch, quad)
        H_back = canvas_to_quad(cw, ch, quad)

        # 1. Rectification : la feuille « à plat ».
        canvas = cv2.warpPerspective(frame, H_rect, (cw, ch), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)

        # 2. Effacement : on ne ramène dans l'image que l'extrait nettoyé.
        erased = self.eraser.erase(canvas)
        out = frame.copy()
        if not erased.empty:
            clean = warp_canvas_patch(erased.clean, erased.origin, H_back, frame.shape,
                                      border=cv2.BORDER_REPLICATE)
            soft = warp_canvas_patch(erased.soft_mask, erased.origin, H_back, frame.shape)
            if clean is not None and soft is not None:
                (clean_img, (x0, y0, x1, y1)), (m, _) = clean, soft
                out[y0:y1, x0:x1] = cv2.blendLinear(out[y0:y1, x0:x1], clean_img,
                                                    1.0 - m, m)

        # 3. Nouveau texte et luminosité du papier autour de la zone.
        cov, (bx0, by0, bx1, by1) = self.coverage()
        lum = measure_paper_luminance(canvas, erased.text_mask, (bx0, by0, bx1, by1),
                                      self.config.photometry.luminance_ring)
        self._luminance = lum if self._luminance is None else (
            (1 - _LUMINANCE_EMA) * self._luminance + _LUMINANCE_EMA * lum
        )

        # Extrait de couverture avec marge pour le flou (sans rognage du halo).
        pad = int(np.ceil(4 * self.config.photometry.blur_sigma)) + 4
        cx0, cy0 = max(0, bx0 - pad), max(0, by0 - pad)
        cx1, cy1 = min(cw, bx1 + pad), min(ch, by1 + pad)
        cov_patch = cov[cy0:cy1, cx0:cx1]

        # Préfiltre anti-crénelage : si la feuille est plus petite à l'écran
        # que le canevas, on floute la couverture avant de la réduire.
        shrink = cw / max(quad_size(quad)[0], 1.0)
        if shrink > 1.5:
            cov_patch = cv2.GaussianBlur(cov_patch, (0, 0), 0.4 * shrink)

        # 4. Déformation géométrique du calque de texte sur la feuille.
        warped = warp_canvas_patch(cov_patch, (cx0, cy0), H_back, frame.shape)

        # 5. Intégration photométrique (fusion Produit, flou, grain).
        if warped is not None and cov_patch.size:
            cov_img, (x0, y0, x1, y1) = warped
            out[y0:y1, x0:x1] = composite_ink(out[y0:y1, x0:x1], cov_img,
                                              self._luminance,
                                              self.config.photometry, self.noise)

        self.debug = FrameDebug(
            paper_luminance=float(self._luminance),
            erased_pixels=int(np.count_nonzero(erased.text_mask)),
            canvas=canvas,
            erase=erased,
        )
        return out
