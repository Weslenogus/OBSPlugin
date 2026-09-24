"""Outils géométriques partagés : ordre des coins, homographies, tailles."""

from __future__ import annotations

import cv2
import numpy as np

Quad = np.ndarray  # tableau float32 de forme (4, 2), ordre TL, TR, BR, BL


def order_quad(pts: np.ndarray) -> Quad:
    """Ordonne 4 points en haut-gauche, haut-droite, bas-droite, bas-gauche.

    Utilise la somme et la différence des coordonnées : le coin haut-gauche a
    la plus petite somme x+y, le bas-droite la plus grande ; le haut-droite a
    la plus petite différence y-x, le bas-gauche la plus grande.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    ordered[0] = pts[np.argmin(s)]  # TL
    ordered[2] = pts[np.argmax(s)]  # BR
    ordered[1] = pts[np.argmin(d)]  # TR
    ordered[3] = pts[np.argmax(d)]  # BL
    return ordered


def quad_size(quad: Quad) -> tuple[float, float]:
    """Largeur et hauteur moyennes (en pixels) d'un quadrilatère ordonné."""
    tl, tr, br, bl = quad
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    return float(width), float(height)


def canvas_corners(width: int, height: int) -> Quad:
    """Coins d'un canevas rectifié ``width`` x ``height`` (ordre TL,TR,BR,BL)."""
    return np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )


def canvas_to_quad(width: int, height: int, quad: Quad) -> np.ndarray:
    """Homographie qui envoie le canevas rectifié sur le quadrilatère image."""
    return cv2.getPerspectiveTransform(canvas_corners(width, height), quad)


def quad_to_canvas(width: int, height: int, quad: Quad) -> np.ndarray:
    """Homographie qui rectifie le quadrilatère image vers le canevas."""
    return cv2.getPerspectiveTransform(quad, canvas_corners(width, height))


def canvas_size_for_quad(quad: Quad, max_side: int = 960) -> tuple[int, int]:
    """Taille de canevas conservant le ratio du quadrilatère, bornée en pixels.

    Le canevas est l'espace « papier à plat » dans lequel on détecte le texte
    et on rend le nouveau texte ; il reste fixe pendant tout le suivi.
    """
    w, h = quad_size(quad)
    w, h = max(w, 8.0), max(h, 8.0)
    scale = min(1.0, max_side / max(w, h))
    return max(8, int(round(w * scale))), max(8, int(round(h * scale)))


def is_valid_quad(quad: Quad | None, frame_shape: tuple[int, ...],
                  min_area_ratio: float = 0.002) -> bool:
    """Vérifie qu'un quadrilatère est convexe, non dégénéré et dans l'image."""
    if quad is None:
        return False
    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    if not np.all(np.isfinite(quad)):
        return False
    h, w = frame_shape[:2]
    area = abs(cv2.contourArea(quad))
    if area < min_area_ratio * w * h:
        return False
    if not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
        return False
    # Tolère un léger débordement (papier partiellement hors champ).
    margin = 0.5 * max(w, h)
    if np.any(quad < -margin) or np.any(quad[:, 0] > w + margin) or np.any(
        quad[:, 1] > h + margin
    ):
        return False
    return True


def quad_mask(frame_shape: tuple[int, ...], quad: Quad) -> np.ndarray:
    """Masque binaire uint8 (0/255) de l'intérieur du quadrilatère."""
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
    return mask


def quad_roi(quad: Quad, frame_shape: tuple[int, ...],
             pad: int = 4) -> tuple[int, int, int, int] | None:
    """Rectangle englobant (x0, y0, x1, y1) du quadrilatère, borné à l'image.

    Retourne ``None`` si le quadrilatère est entièrement hors champ. Travailler
    dans ce rectangle plutôt que sur l'image entière divise le coût des
    ``warpPerspective`` par le rapport de surfaces.
    """
    h, w = frame_shape[:2]
    x0 = int(np.floor(quad[:, 0].min())) - pad
    y0 = int(np.floor(quad[:, 1].min())) - pad
    x1 = int(np.ceil(quad[:, 0].max())) + pad + 1
    y1 = int(np.ceil(quad[:, 1].max())) + pad + 1
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def translate_homography(H: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Compose ``H`` avec une translation (dx, dy) appliquée après ``H``."""
    T = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)
    return T @ H


def transform_quad(quad: Quad, H: np.ndarray) -> Quad:
    """Applique une homographie aux 4 coins d'un quadrilatère."""
    pts = cv2.perspectiveTransform(quad.reshape(-1, 1, 2).astype(np.float32), H)
    return pts.reshape(4, 2).astype(np.float32)


def patch_roi(patch_shape: tuple[int, ...], origin: tuple[float, float],
              H_canvas_to_frame: np.ndarray, frame_shape: tuple[int, ...]
              ) -> tuple[int, int, int, int] | None:
    """Rectangle image (x0, y0, x1, y1) couvert par un extrait du canevas.

    ``origin`` est le coin haut-gauche de l'extrait dans le canevas (réel :
    décalage sous-pixel possible). ``None`` si l'extrait sort de l'image.
    """
    ox, oy = origin
    ph, pw = patch_shape[:2]
    corners = np.array([[ox, oy], [ox + pw, oy], [ox + pw, oy + ph], [ox, oy + ph]],
                       dtype=np.float32)
    return quad_roi(transform_quad(corners, H_canvas_to_frame), frame_shape, pad=2)


def patch_homography(H_canvas_to_frame: np.ndarray, origin: tuple[float, float],
                     roi: tuple[int, int, int, int]) -> np.ndarray:
    """Homographie extrait → rectangle ``roi`` (pour ``warpPerspective``)."""
    x0, y0 = roi[:2]
    ox, oy = origin
    return translate_homography(H_canvas_to_frame, -x0, -y0) @ np.array(
        [[1, 0, ox], [0, 1, oy], [0, 0, 1]], dtype=np.float64)


def warp_canvas_patch(patch: np.ndarray, origin: tuple[float, float],
                      H_canvas_to_frame: np.ndarray, frame_shape: tuple[int, ...],
                      border: int = cv2.BORDER_CONSTANT
                      ) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Déforme un morceau du canevas vers l'image, dans sa seule boîte utile.

    ``patch`` est un extrait du canevas dont le coin haut-gauche se trouve en
    ``origin``. On projette ses 4 coins pour obtenir le rectangle image
    concerné, puis on ne calcule ``warpPerspective`` que sur ce rectangle :
    le coût devient proportionnel à la surface du texte, pas à celle de la
    feuille. Renvoie ``(patch_déformé, (x0, y0, x1, y1))`` ou ``None``.
    """
    roi = patch_roi(patch.shape, origin, H_canvas_to_frame, frame_shape)
    if roi is None:
        return None
    x0, y0, x1, y1 = roi
    warped = cv2.warpPerspective(patch, patch_homography(H_canvas_to_frame, origin, roi),
                                 (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                 borderMode=border, borderValue=0)
    return warped, roi
