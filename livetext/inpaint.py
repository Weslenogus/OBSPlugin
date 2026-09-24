"""Effacement du texte d'origine sans laisser de rectangle uni.

Tout se passe dans le canevas rectifié (le papier « à plat ») :

1. détection des traits sombres par seuillage adaptatif → masque du texte ;
2. reconstruction du papier sous ce masque selon la méthode choisie :

   * ``plate``   — plaque de fond : ``cv2.inpaint`` (Telea) à basse
     résolution puis agrandissement. Très rapide et conserve les variations
     lentes d'éclairage (ombres, plis larges) ;
   * ``inpaint`` — ``cv2.inpaint`` (Telea) en pleine résolution, plus fin mais
     plus coûteux ;
   * ``median``  — remplissage par la teinte médiane du papier, avec un masque
     flouté ;
   * ``none``    — aucun effacement ;

3. réinjection d'un grain de même écart-type que celui du papier mesuré, pour
   que la zone reconstruite ne paraisse pas lisse ;
4. masque adouci (bords progressifs) pour le mélange final.

La reconstruction ne porte que sur le rectangle englobant du masque (plus une
marge de contexte) : le coût suit la surface du texte, pas celle de la
feuille.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .compositor import NoiseBank
from .config import EraseConfig

_PLATE_DOWNSCALE = 4


@dataclass
class EraseResult:
    """Résultat de l'effacement, exprimé dans l'espace du canevas."""

    text_mask: np.ndarray                 # uint8 0/255, canevas entier
    clean: np.ndarray | None = None       # extrait BGR uint8 sans texte
    soft_mask: np.ndarray | None = None   # float32 0..1, taille de ``clean``
    origin: tuple[int, int] = (0, 0)      # coin haut-gauche de l'extrait

    @property
    def empty(self) -> bool:
        return self.clean is None


def _odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 else value + 1


def _disk(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)


class TextEraser:
    """Efface le texte existant dans une région du canevas rectifié."""

    def __init__(self, config: EraseConfig, noise: NoiseBank | None = None):
        self.config = config
        self.noise = noise or NoiseBank()
        self._region_cache: tuple[tuple[int, int], np.ndarray] | None = None
        self._history: np.ndarray | None = None

    def reset(self) -> None:
        """Oublie la mémoire temporelle (nouvelle cible ou nouveau canevas)."""
        self._history = None

    def region_mask(self, canvas_w: int, canvas_h: int) -> np.ndarray:
        """Masque uint8 de la zone à effacer (en retrait des bords du papier)."""
        key = (canvas_w, canvas_h)
        if self._region_cache and self._region_cache[0] == key:
            return self._region_cache[1]
        x0, y0, x1, y1 = self.config.region
        mask = np.zeros((canvas_h, canvas_w), np.uint8)
        # Retrait de ~1,5 % : le bord exact du papier est interpolé avec le fond
        # lors de la rectification, on ne doit pas le prendre pour du texte.
        inset = max(2, int(0.015 * min(canvas_w, canvas_h)))
        px0 = max(inset, int(x0 * canvas_w))
        py0 = max(inset, int(y0 * canvas_h))
        px1 = min(canvas_w - inset, int(x1 * canvas_w))
        py1 = min(canvas_h - inset, int(y1 * canvas_h))
        if px1 > px0 and py1 > py0:
            mask[py0:py1, px0:px1] = 255
        self._region_cache = (key, mask)
        return mask

    def detect_text(self, canvas_bgr: np.ndarray) -> np.ndarray:
        """Masque uint8 (0/255) des traits sombres dans la zone à effacer."""
        cfg = self.config
        h, w = canvas_bgr.shape[:2]
        gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)
        ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, _odd(cfg.block_size),
                                    cfg.threshold_c)
        ink &= self.region_mask(w, h)

        # Élimine les mouchetures isolées (bruit du capteur, fibres du papier).
        n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        small = stats[:, cv2.CC_STAT_AREA] < 6
        small[0] = False
        if small.any():
            keep = np.where(small, 0, 255).astype(np.uint8)
            ink &= np.take(keep, labels)

        if cfg.stroke_dilate > 0:
            # La dilatation ne doit pas déborder de la zone demandée.
            ink = cv2.dilate(ink, _disk(cfg.stroke_dilate)) & self.region_mask(w, h)
        return ink

    def _stabilize(self, mask: np.ndarray) -> np.ndarray:
        """Cumule le masque dans le temps pour éviter le scintillement.

        Historique en uint8 (0-255) : ``max(masque, historique × décroissance)``
        puis seuil à 50 %, entièrement en opérations OpenCV vectorisées.
        """
        decay = self.config.temporal_decay
        if decay <= 0:
            return mask
        if self._history is None or self._history.shape != mask.shape:
            self._history = mask.copy()
        else:
            self._history = cv2.max(mask, cv2.multiply(self._history, decay))
        return cv2.threshold(self._history, 127, 255, cv2.THRESH_BINARY)[1]

    # -- reconstructions (sur l'extrait) -------------------------------------

    @staticmethod
    def _plate(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        sw, sh = max(8, w // _PLATE_DOWNSCALE), max(8, h // _PLATE_DOWNSCALE)
        small = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_AREA)
        small_mask = cv2.dilate((small_mask > 0).astype(np.uint8) * 255,
                                np.ones((3, 3), np.uint8))
        filled = cv2.inpaint(small, small_mask, 3, cv2.INPAINT_TELEA)
        filled = cv2.GaussianBlur(filled, (0, 0), 1.0)
        return cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC)

    def _inpaint(self, crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return cv2.inpaint(crop, mask, self.config.inpaint_radius, cv2.INPAINT_TELEA)

    @staticmethod
    def _median(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
        paper = crop[mask == 0]
        if paper.size == 0:
            paper = crop.reshape(-1, 3)
        color = np.median(paper.reshape(-1, 3), axis=0).astype(np.uint8)
        return np.broadcast_to(color, crop.shape).copy()

    @staticmethod
    def _paper_grain_sigma(crop: np.ndarray, mask: np.ndarray) -> float:
        """Écart-type robuste du grain haute fréquence du papier (hors texte)."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        residual = gray - cv2.GaussianBlur(gray, (0, 0), 1.5)
        values = residual[mask == 0]
        if values.size < 100:
            return 0.0
        values = values[:: max(1, values.size // 20000)]  # sous-échantillonnage
        # MAD : estimateur robuste, insensible aux quelques traits restants.
        return float(1.4826 * np.median(np.abs(values - np.median(values))))

    # -- API ----------------------------------------------------------------

    def erase(self, canvas_bgr: np.ndarray) -> EraseResult:
        """Efface le texte détecté dans ``canvas_bgr``."""
        cfg = self.config
        h, w = canvas_bgr.shape[:2]
        if cfg.method == "none":
            return EraseResult(np.zeros((h, w), np.uint8))

        mask = self._stabilize(self.detect_text(canvas_bgr))
        points = cv2.findNonZero(mask)
        if points is None:
            return EraseResult(mask)

        # Extrait englobant le masque + marge (adoucissement et contexte).
        bx, by, bw, bh = cv2.boundingRect(points)
        margin = cfg.feather * 2 + 8
        x0, y0 = max(0, bx - margin), max(0, by - margin)
        x1, y1 = min(w, bx + bw + margin), min(h, by + bh + margin)
        crop = canvas_bgr[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]

        if cfg.method == "inpaint":
            clean = self._inpaint(crop, crop_mask)
        elif cfg.method == "median":
            clean = self._median(crop, crop_mask)
        else:
            clean = self._plate(crop, crop_mask)

        if cfg.grain_strength > 0 and cfg.method != "inpaint":
            sigma = self._paper_grain_sigma(crop, crop_mask) * cfg.grain_strength
            if sigma > 0.05:
                grain = self.noise.sample(*clean.shape[:2]) * sigma
                clean = np.clip(clean.astype(np.float32) + grain[..., None],
                                0, 255).astype(np.uint8)

        # Masque adouci : 1 sur les traits, décroissance progressive autour.
        soft = crop_mask.astype(np.float32) / 255.0
        if cfg.feather > 0:
            grown = cv2.dilate(crop_mask, _disk(cfg.feather // 2)).astype(np.float32)
            soft = np.maximum(soft, cv2.GaussianBlur(grown / 255.0, (0, 0),
                                                     cfg.feather / 2.0))
        # Rien ne doit être modifié hors de la zone demandée.
        soft *= self.region_mask(w, h)[y0:y1, x0:x1].astype(np.float32) / 255.0
        return EraseResult(mask, clean, np.clip(soft, 0.0, 1.0), (x0, y0))
