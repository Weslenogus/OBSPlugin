"""Fixtures partagées : scène synthétique (feuille + texte) sans caméra."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PAPER_BGR = (225, 232, 238)


def make_scene(width: int = 960, height: int = 540, quad=None, seed: int = 0,
               with_text: bool = True):
    """Feuille (avec texte d'origine) déformée en perspective sur un fond sombre.

    Renvoie ``(image, quad_exact, masque_feuille)``.
    """
    rng = np.random.default_rng(seed)
    pw, ph = 600, 420
    paper = np.full((ph, pw, 3), PAPER_BGR, np.float32)
    paper *= np.linspace(0.85, 1.03, pw, dtype=np.float32)[None, :, None]
    paper += rng.normal(0, 2.5, paper.shape)
    paper = np.clip(paper, 0, 255).astype(np.uint8)
    if with_text:
        cv2.putText(paper, "ORIGINAL", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.4,
                    (30, 30, 30), 7, cv2.LINE_AA)
        cv2.putText(paper, "line two 42", (50, 290), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                    (40, 40, 40), 4, cv2.LINE_AA)
    bg = cv2.GaussianBlur(rng.normal(70, 20, (height, width, 3)).astype(np.float32), (0, 0), 2)
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    if quad is None:
        quad = np.float32([[230, 70], [760, 105], [730, 480], [200, 455]])
    quad = np.asarray(quad, np.float32)
    src = np.float32([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]])
    M = cv2.getPerspectiveTransform(src, quad)
    warped = cv2.warpPerspective(paper, M, (width, height))
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), M, (width, height))
    frame = np.where(mask[..., None] > 0, warped, bg)
    return frame, quad, mask


@pytest.fixture
def scene():
    return make_scene()
