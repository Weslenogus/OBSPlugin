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

from .compositor import (
    LUMA_BGR,
    NoiseBank,
    composite_ink,
    ink_multiply_factors,
    measure_ink_color,
    measure_paper_color,
)
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

# Lissage temporel des couleurs mesurées (évite le « pompage » de l'encre).
_COLOR_EMA = 0.2


def _ema(previous: np.ndarray | None, value: np.ndarray) -> np.ndarray:
    return value if previous is None else (1 - _COLOR_EMA) * previous + _COLOR_EMA * value


@dataclass
class FrameDebug:
    """Informations de diagnostic de la dernière image traitée."""

    paper_luminance: float = 0.0
    paper_bgr: np.ndarray | None = None
    ink_bgr: np.ndarray | None = None      # None : pas d'encre visible (repli gris)
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
        self._paper_bgr: np.ndarray | None = None
        self._ink_bgr: np.ndarray | None = None
        # Décalage fin du texte, en pixels du *canevas* : il reste attaché à la
        # feuille quand elle bouge ; sous-pixel grâce au warp (origine réelle).
        self.text_offset = np.zeros(2, np.float64)
        self._canvas_per_screen = 1.0
        self.debug = FrameDebug()

    # -- configuration de la cible -------------------------------------------

    def set_target(self, quad: Quad) -> None:
        """Fixe la taille du canevas d'après la feuille à l'initialisation."""
        self.canvas_size = canvas_size_for_quad(quad, self.config.canvas_max_side)
        self.eraser.reset()
        self._paper_bgr = self._ink_bgr = None
        self._coverage_key = None

    def set_text(self, text: str) -> None:
        """Change le texte affiché (le rendu est refait à la prochaine image)."""
        self.text = text

    def invalidate_text(self) -> None:
        """Force un nouveau rendu (police, alignement… modifiés à chaud)."""
        self._coverage_key = None

    # -- réglages fins en direct ------------------------------------------------

    def nudge_text(self, dx: float, dy: float) -> None:
        """Décale le texte de (dx, dy) pixels *à l'écran* (demi-pixel possible).

        Converti en unités du canevas à l'échelle courante de la feuille, puis
        mémorisé dans le canevas : le décalage suit ensuite la feuille.
        """
        self.text_offset += np.array([dx, dy]) * self._canvas_per_screen

    def reset_offset(self) -> None:
        self.text_offset[:] = 0.0

    def adjust_font_size(self, delta: int) -> int:
        """Change la taille de ``delta`` points ; renvoie la nouvelle taille.

        Part de la taille réellement affichée (même en mode automatique) et
        désactive l'ajustement automatique : la taille choisie est respectée.
        """
        text_cfg = self.config.text
        current = text_cfg.font_size or self.renderer.last_size or 12
        text_cfg.font_size = max(4, int(current) + int(delta))
        text_cfg.auto_fit = False
        return text_cfg.font_size

    # -- rendu du texte (mis en cache) ---------------------------------------

    def coverage(self) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Couverture d'encre dans le canevas et sa boîte englobante (px)."""
        assert self.canvas_size is not None
        w, h = self.canvas_size
        t = self.config.text
        # Tous les réglages de mise en page font partie de la clé : un réglage
        # modifié à chaud relance automatiquement le rendu.
        key = (self.text, w, h, t.font_size, t.tracking, t.weight, t.align,
               t.valign, t.box, t.auto_fit, t.line_spacing, t.relative_font_size)
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

        # 3. Nouveau texte ; couleurs réelles du papier et de l'encre autour.
        cov, box = self.coverage()
        bx0, by0, bx1, by1 = box
        photometry = self.config.photometry
        ring = photometry.luminance_ring
        self._paper_bgr = _ema(self._paper_bgr,
                               measure_paper_color(canvas, erased.text_mask, box, ring))
        if photometry.sample_ink:
            ink_mask = erased.text_mask
            if not ink_mask.any() and self.config.erase.method == "none":
                ink_mask = self.eraser.detect_text(canvas)  # effacement désactivé
            ink = measure_ink_color(canvas, ink_mask, self._paper_bgr, box, ring)
            if ink is not None:  # sinon : dernière teinte connue conservée
                self._ink_bgr = _ema(self._ink_bgr, ink)
        factors = ink_multiply_factors(
            self._paper_bgr, self._ink_bgr if photometry.sample_ink else None, photometry)

        # Extrait de couverture avec marge pour le flou (sans rognage du halo).
        pad = int(np.ceil(4 * self.config.photometry.blur_sigma)) + 4
        cx0, cy0 = max(0, bx0 - pad), max(0, by0 - pad)
        cx1, cy1 = min(cw, bx1 + pad), min(ch, by1 + pad)
        cov_patch = cov[cy0:cy1, cx0:cx1]

        # Préfiltre anti-crénelage : si la feuille est plus petite à l'écran
        # que le canevas, on floute la couverture avant de la réduire.
        shrink = cw / max(quad_size(quad)[0], 1.0)
        self._canvas_per_screen = shrink
        if shrink > 1.5:
            cov_patch = cv2.GaussianBlur(cov_patch, (0, 0), 0.4 * shrink)

        # 4. Déformation géométrique du calque de texte sur la feuille ; le
        # décalage fin est une origine *réelle* : précision sous-pixel.
        ox, oy = self.text_offset
        warped = warp_canvas_patch(cov_patch, (cx0 + ox, cy0 + oy), H_back, frame.shape)

        # 5. Intégration photométrique (fusion Produit, flou, grain).
        if warped is not None and cov_patch.size:
            cov_img, (x0, y0, x1, y1) = warped
            out[y0:y1, x0:x1] = composite_ink(out[y0:y1, x0:x1], cov_img, factors,
                                              photometry, self.noise)

        self.debug = FrameDebug(
            paper_luminance=float(self._paper_bgr @ LUMA_BGR),
            paper_bgr=self._paper_bgr,
            ink_bgr=self._ink_bgr,
            erased_pixels=int(np.count_nonzero(erased.text_mask)),
            canvas=canvas,
            erase=erased,
        )
        return out
