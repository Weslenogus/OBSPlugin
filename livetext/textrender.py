"""Rendu dynamique du texte avec Pillow à partir d'une police vectorielle.

Le rendu produit une *carte de couverture* (float32, 0..1) dans l'espace du
canevas rectifié (le papier « à plat »). La couleur finale de l'encre n'est
pas décidée ici : elle est échantillonnée sur l'encre réelle du papier par le
module photométrique.

La police est par défaut ``police.ttf`` (dossier courant ou dossier du
script), chargée par ``PIL.ImageFont.truetype``. Le rendu est suréchantillonné
(×4) puis réduit : graisse et interlettrage se règlent au quart de point, et
l'anticrénelage est meilleur que celui du tracé direct.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import FONT_FILE, TextConfig

log = logging.getLogger(__name__)

# Facteur de suréchantillonnage du rendu (précision de 1/4 de point).
_SUPERSAMPLE = 4

# Polices système essayées si ``police.ttf`` est absente.
_SYSTEM_FONTS = (
    "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


def local_font_path() -> str | None:
    """Chemin de ``police.ttf`` : dossier courant, puis dossier du script."""
    package_dir = Path(__file__).resolve().parent
    for folder in (Path.cwd(), package_dir.parent, package_dir):
        candidate = folder / FONT_FILE
        if candidate.is_file():
            return str(candidate)
    return None


def find_system_font() -> str | None:
    """Chemin d'une police TrueType du système (sert aussi à l'interface)."""
    for candidate in _SYSTEM_FONTS:
        try:
            # Pillow résout les noms nus (« DejaVuSans.ttf ») dans les dossiers
            # de polices du système : on renvoie le chemin réel trouvé.
            return ImageFont.truetype(candidate, 12).path
        except OSError:
            continue
    return None


def find_default_font() -> str | None:
    """Police du texte incrusté par défaut : ``police.ttf``, sinon le système."""
    return local_font_path() or find_system_font()


@lru_cache(maxsize=128)
def load_font(path: str | None, size: int) -> ImageFont.FreeTypeFont:
    """``PIL.ImageFont.truetype(path, size)`` mis en cache.

    ``path=None`` : police par défaut (:func:`find_default_font`). Lève
    ``FileNotFoundError`` si ``path`` est fourni mais introuvable, afin que
    l'utilisateur sache que *sa* police n'a pas été utilisée.
    """
    size = max(1, int(size))
    if path:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Police introuvable : {path}")
        return ImageFont.truetype(path, size)
    default = find_default_font()
    if default is None:
        raise FileNotFoundError(
            f"Aucune police trouvée : placez « {FONT_FILE} » à côté de main.py "
            "ou indiquez --font chemin/vers/police.ttf"
        )
    return ImageFont.truetype(default, size)


def normalize_text(text: str) -> str:
    """Interprète les séquences ``\\n`` tapées dans le terminal."""
    return text.replace("\\n", "\n").replace("\r\n", "\n")


def line_width(font: ImageFont.FreeTypeFont, line: str, tracking: float = 0.0) -> float:
    """Largeur d'une ligne, interlettrage compris (crénage conservé)."""
    return font.getlength(line) + tracking * max(0, len(line) - 1)


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: float,
              tracking: float = 0.0) -> list[str]:
    """Découpe le texte en lignes tenant dans ``max_width`` pixels.

    Respecte les sauts de ligne explicites ; coupe au caractère les mots plus
    longs que la largeur disponible.
    """
    def fits(s: str) -> bool:
        return line_width(font, s, tracking) <= max_width

    lines: list[str] = []
    for paragraph in normalize_text(text).split("\n"):
        current = ""
        for word in paragraph.split(" "):
            candidate = word if not current else f"{current} {word}"
            if fits(candidate):
                current = candidate
                continue
            if current:
                lines.append(current)
                current = ""
            # Mot trop long : découpe caractère par caractère.
            while word and not fits(word):
                cut = len(word)
                while cut > 1 and not fits(word[:cut]):
                    cut -= 1
                lines.append(word[:cut])
                word = word[cut:]
            current = word
        lines.append(current)
    return lines


def _draw_line(draw: ImageDraw.ImageDraw, x: float, y: float, line: str,
               font: ImageFont.FreeTypeFont, tracking: float, stroke: int) -> None:
    """Trace une ligne ; avec interlettrage, glyphe par glyphe.

    Chaque glyphe est placé à ``getlength(ligne[:i]) + i × interlettrage`` :
    la longueur du préfixe inclut les paires de crénage, qui sont donc
    conservées (contrairement à une simple somme des avances).
    """
    style = dict(font=font, fill=255, stroke_width=stroke, stroke_fill=255)
    if tracking == 0.0:
        draw.text((x, y), line, **style)
        return
    for i, char in enumerate(line):
        if not char.isspace():
            draw.text((x + font.getlength(line[:i]) + tracking * i, y), char, **style)


class TextRenderer:
    """Rend un texte en carte de couverture dans le canevas rectifié."""

    def __init__(self, config: TextConfig):
        self.config = config
        self.font_path = config.font_path or find_default_font()
        load_font(self.font_path, 16)  # vérifie tôt que la police est utilisable
        if config.font_path is None and local_font_path() is None:
            log.warning("« %s » introuvable : police système utilisée (%s).",
                        FONT_FILE, self.font_path)
        else:
            log.info("Police : %s", self.font_path)
        self.last_size = 0  # taille (points) réellement utilisée au dernier rendu

    def _base_font_size(self, canvas_h: int) -> int:
        if self.config.font_size > 0:
            return int(self.config.font_size)
        return max(6, int(round(canvas_h * self.config.relative_font_size)))

    def _layout(self, text: str, box_w: float, box_h: float, canvas_h: int,
                s: int, tracking: float, stroke: int):
        """Choisit la taille (points) et les lignes, en coordonnées ×``s``."""
        size = self._base_font_size(canvas_h)
        while True:
            font = load_font(self.font_path, size * s)
            lines = wrap_text(text, font, box_w - 2 * stroke, tracking)
            ascent, descent = font.getmetrics()
            line_h = (ascent + descent) * self.config.line_spacing
            block_h = line_h * (len(lines) - 1) + ascent + descent + 2 * stroke
            if not self.config.auto_fit or block_h <= box_h or size <= 6:
                self.last_size = size
                return font, lines, line_h, block_h
            size = max(6, int(size * 0.9))

    def render(self, text: str, canvas_w: int, canvas_h: int) -> np.ndarray:
        """Retourne la couverture de l'encre (float32 HxW, 0..1)."""
        if not text.strip():
            return np.zeros((canvas_h, canvas_w), dtype=np.float32)

        s = _SUPERSAMPLE
        W, H = canvas_w * s, canvas_h * s
        tracking = self.config.tracking * s
        weight = self.config.weight * s
        stroke = int(round(weight)) if weight > 0 else 0

        x0, y0, x1, y1 = self.config.box
        bx0, by0 = x0 * W, y0 * H
        box_w = max(1.0, (x1 - x0) * W)
        box_h = max(1.0, (y1 - y0) * H)
        font, lines, line_h, block_h = self._layout(text, box_w, box_h, canvas_h,
                                                    s, tracking, stroke)

        if self.config.valign == "middle":
            y = by0 + (box_h - block_h) / 2.0
        elif self.config.valign == "bottom":
            y = by0 + box_h - block_h
        else:
            y = by0
        y += stroke

        placements = []
        for line in lines:
            width = line_width(font, line, tracking)
            if self.config.align == "center":
                x = bx0 + (box_w - width) / 2.0
            elif self.config.align == "right":
                x = bx0 + box_w - width - stroke
            else:
                x = bx0 + stroke
            placements.append((x, y, line, width))
            y += line_h

        # On ne trace que le bloc de texte (et non tout le canevas ×4) : le
        # rendu reste rapide pendant la saisie en direct et les réglages à
        # chaud. Boîte alignée sur la grille ×s pour une réduction exacte.
        ascent, descent = font.getmetrics()
        pad = int(0.3 * (ascent + descent)) + stroke + s
        X0 = max(0, int(min(p[0] for p in placements) - pad) // s * s)
        Y0 = max(0, int(placements[0][1] - pad) // s * s)
        X1 = min(W, -(-int(max(p[0] + p[3] for p in placements) + pad) // s) * s)
        Y1 = min(H, -(-int(placements[-1][1] + ascent + descent + pad) // s) * s)
        coverage = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        if X1 <= X0 or Y1 <= Y0:
            return coverage

        image = Image.new("L", (X1 - X0, Y1 - Y0), 0)
        draw = ImageDraw.Draw(image)
        for x, y, line, _ in placements:
            _draw_line(draw, x - X0, y - Y0, line, font, tracking, stroke)

        hires = np.asarray(image)
        if weight < 0:  # graisse négative : on amincit les traits
            r = int(round(-weight))
            hires = cv2.erode(hires, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                               (2 * r + 1,) * 2))
        patch = cv2.resize(hires, ((X1 - X0) // s, (Y1 - Y0) // s),
                           interpolation=cv2.INTER_AREA)
        coverage[Y0 // s:Y0 // s + patch.shape[0],
                 X0 // s:X0 // s + patch.shape[1]] = patch.astype(np.float32) / 255.0
        return coverage
