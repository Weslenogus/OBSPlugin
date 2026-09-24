"""Rendu dynamique du texte avec Pillow à partir d'une police vectorielle.

Le rendu produit une *carte de couverture* (float32, 0..1) dans l'espace du
canevas rectifié (le papier « à plat »). La couleur finale de l'encre n'est
pas décidée ici : elle est calculée par le module photométrique en fonction
de la luminosité mesurée du papier.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import TextConfig

# Polices système essayées si aucune police .ttf n'est fournie.
_FALLBACK_FONTS = (
    "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


def find_default_font() -> str | None:
    """Retourne le chemin d'une police TrueType disponible sur le système."""
    for candidate in _FALLBACK_FONTS:
        try:
            ImageFont.truetype(candidate, 12)
            return candidate
        except OSError:
            continue
    return None


@lru_cache(maxsize=64)
def load_font(path: str | None, size: int) -> ImageFont.FreeTypeFont:
    """Charge une police vectorielle à la taille demandée (mise en cache).

    Lève ``FileNotFoundError`` si ``path`` est fourni mais introuvable, afin
    que l'utilisateur sache que *sa* police n'a pas été utilisée.
    """
    size = max(1, int(size))
    if path:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Police introuvable : {path}")
        return ImageFont.truetype(path, size)
    default = find_default_font()
    if default is None:
        raise FileNotFoundError(
            "Aucune police .ttf trouvée sur le système ; "
            "fournissez-en une avec --font chemin/vers/police.ttf"
        )
    return ImageFont.truetype(default, size)


def normalize_text(text: str) -> str:
    """Interprète les séquences ``\\n`` tapées dans le terminal."""
    return text.replace("\\n", "\n").replace("\r\n", "\n")


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: float) -> list[str]:
    """Découpe le texte en lignes tenant dans ``max_width`` pixels.

    Respecte les sauts de ligne explicites ; coupe au caractère les mots plus
    longs que la largeur disponible.
    """
    lines: list[str] = []
    for paragraph in normalize_text(text).split("\n"):
        words = paragraph.split(" ")
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if font.getlength(candidate) <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
                current = ""
            # Mot trop long : découpe caractère par caractère.
            while word and font.getlength(word) > max_width:
                cut = len(word)
                while cut > 1 and font.getlength(word[:cut]) > max_width:
                    cut -= 1
                lines.append(word[:cut])
                word = word[cut:]
            current = word
        lines.append(current)
    return lines


class TextRenderer:
    """Rend un texte en carte de couverture dans le canevas rectifié."""

    def __init__(self, config: TextConfig):
        self.config = config
        # Vérifie tôt que la police est utilisable.
        load_font(config.font_path, 16)

    def _base_font_size(self, canvas_h: int) -> int:
        if self.config.font_size > 0:
            return int(self.config.font_size)
        return max(6, int(round(canvas_h * self.config.relative_font_size)))

    def _layout(self, text: str, box_w: float, box_h: float, canvas_h: int):
        """Choisit la taille de police et les lignes (ajustement automatique)."""
        size = self._base_font_size(canvas_h)
        while True:
            font = load_font(self.config.font_path, size)
            lines = wrap_text(text, font, box_w)
            ascent, descent = font.getmetrics()
            line_h = (ascent + descent) * self.config.line_spacing
            block_h = line_h * (len(lines) - 1) + ascent + descent
            if not self.config.auto_fit or block_h <= box_h or size <= 6:
                return font, lines, line_h, block_h
            size = max(6, int(size * 0.9))

    def render(self, text: str, canvas_w: int, canvas_h: int) -> np.ndarray:
        """Retourne la couverture de l'encre (float32 HxW, 0..1)."""
        image = Image.new("L", (canvas_w, canvas_h), 0)
        if not text.strip():
            return np.zeros((canvas_h, canvas_w), dtype=np.float32)

        x0, y0, x1, y1 = self.config.box
        bx0, by0 = x0 * canvas_w, y0 * canvas_h
        box_w = max(1.0, (x1 - x0) * canvas_w)
        box_h = max(1.0, (y1 - y0) * canvas_h)

        font, lines, line_h, block_h = self._layout(text, box_w, box_h, canvas_h)

        if self.config.valign == "middle":
            y = by0 + (box_h - block_h) / 2.0
        elif self.config.valign == "bottom":
            y = by0 + box_h - block_h
        else:
            y = by0

        draw = ImageDraw.Draw(image)
        for line in lines:
            width = font.getlength(line)
            if self.config.align == "center":
                x = bx0 + (box_w - width) / 2.0
            elif self.config.align == "right":
                x = bx0 + box_w - width
            else:
                x = bx0
            draw.text((x, y), line, font=font, fill=255)
            y += line_h

        return np.asarray(image, dtype=np.float32) / 255.0


def resolve_font_path(path: str | None) -> str | None:
    """Rend absolu un chemin de police relatif au répertoire courant."""
    if not path:
        return None
    return str(Path(path).expanduser().resolve())
