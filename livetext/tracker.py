"""Détection et suivi planaire robuste de la zone cible (la feuille).

Trois briques, combinables :

* :func:`detect_document_quad` — détection d'un polygone 4 points (contour
  du papier) ; sert à l'initialisation automatique et au mode ``contour``.
* Flux optique pyramidal de Lucas-Kanade (``cv2.calcOpticalFlowPyrLK``) —
  chaque point suivi garde sa position dans l'image de *référence* ; on
  recalcule donc à chaque image l'homographie référence → image courante au
  lieu de composer des homographies image à image, ce qui évite la dérive.
* Appariement ORB sur la référence — relocalisation robuste (rotation,
  échelle, perspective) quand le flux optique perd des points ou dérive.
* Score d'alignement photométrique (ZNCC passe-haut entre la référence
  rectifiée et l'image courante recalée) — arbitre entre flux optique et
  ORB : on garde l'estimation qui explique le mieux l'image, jamais celle
  d'ORB par principe (ORB est moins précis que LK tant que LK tient).

Toutes les coordonnées internes sont exprimées à l'échelle de suivi (image
réduite à ~960 px de côté pour la vitesse) ; l'API publique travaille en
pixels de l'image d'origine.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .config import TrackerConfig
from .geometry import (
    Quad,
    canvas_corners,
    canvas_size_for_quad,
    is_valid_quad,
    order_quad,
    quad_mask,
    transform_quad,
)

_TRACK_MAX_SIDE = 960
_TEMPLATE_SIDE = 240     # côté max du gabarit de référence (score ZNCC)
_TEMPLATE_INSET = 0.04   # retrait du gabarit : que du papier, pas de fond
_UNIT_SQUARE = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])

_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


# ---------------------------------------------------------------------------
# Détection du polygone 4 points
# ---------------------------------------------------------------------------

def _inner_mask(shape: tuple[int, ...], quad: Quad) -> np.ndarray:
    """Masque de la feuille en retrait d'une demi-fenêtre LK.

    Un point proche du bord a une fenêtre de suivi à cheval sur la feuille
    (mobile) et le fond (immobile) : il « glisse » et biaise l'homographie.
    Or ce sont justement les coins les plus contrastés, donc ceux que
    ``goodFeaturesToTrack`` et ORB choisiraient en premier.
    """
    inset = _LK_PARAMS["winSize"][0] // 2 + 2
    mask = quad_mask(shape, quad)
    return cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                     (2 * inset + 1,) * 2))


def _normalized_highpass(img: np.ndarray) -> np.ndarray:
    """Passe-haut, centré, norme 1 : prêt pour une corrélation ZNCC.

    Le passe-haut retire l'éclairage (basses fréquences) : le score dépend de
    la structure du texte, et chute vers 0 en cas de mauvais alignement.
    """
    f = img.astype(np.float32)
    f -= cv2.GaussianBlur(f, (0, 0), 3.0)
    f -= f.mean()
    norm = float(np.linalg.norm(f))
    return f / norm if norm > 1e-6 else np.zeros_like(f)


def _approx_quad(contour: np.ndarray) -> np.ndarray | None:
    """Approxime un contour par un quadrilatère convexe, si possible."""
    perimeter = cv2.arcLength(contour, True)
    hull = cv2.convexHull(contour)
    for eps in (0.01, 0.02, 0.03, 0.04, 0.05, 0.07):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    return None


def _intersect(l1: np.ndarray, l2: np.ndarray) -> np.ndarray | None:
    """Intersection de deux droites (vx, vy, x0, y0) ; None si parallèles."""
    (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = l1, l2
    det = vx1 * (-vy2) - vy1 * (-vx2)
    if abs(det) < 1e-6:
        return None
    t = ((x2 - x1) * (-vy2) - (y2 - y1) * (-vx2)) / det
    return np.array([x1 + t * vx1, y1 + t * vy1], np.float32)


def refine_quad(gray: np.ndarray, quad: Quad, search: int = 10,
                samples: int = 48, max_residual: float = 1.5) -> Quad | None:
    """Affine les côtés du quadrilatère au sous-pixel sur les vrais bords.

    Pour chaque côté, on cherche le long de sa normale (± ``search`` px) le
    maximum du gradient, puis on ajuste une droite robuste (Huber) sur ces
    points ; les coins sont les intersections des côtés adjacents. Corrige
    le biais « vers l'extérieur » de la détection (dilatation des bords).

    Sert aussi de *vérification* : renvoie ``None`` si un côté n'est pas un
    vrai bord rectiligne (résidu de l'ajustement > ``max_residual`` px). Un
    bord de feuille réel s'ajuste à ~0,3 px près ; un côté d'enveloppe
    convexe qui coupe à travers le décor, non.
    """
    # Gradients calculés sur la seule boîte englobante (+ marge de recherche).
    h, w = gray.shape
    x0 = max(0, int(quad[:, 0].min()) - search - 2)
    y0 = max(0, int(quad[:, 1].min()) - search - 2)
    x1 = min(w, int(np.ceil(quad[:, 0].max())) + search + 3)
    y1 = min(h, int(np.ceil(quad[:, 1].max())) + search + 3)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    roi = gray[y0:y1, x0:x1]
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    local = quad - np.array([x0, y0], np.float32)
    rh, rw = roi.shape

    offsets = np.arange(-search, search + 1, dtype=np.float32)
    lines = []
    for i in range(4):
        p0, p1 = local[i], local[(i + 1) % 4]
        d = p1 - p0
        length = float(np.linalg.norm(d))
        if length < 4 * search:
            return None
        normal = np.array([-d[1], d[0]], np.float32) / length
        # Échantillons sur 10 %-90 % du côté (les coins sont ambigus).
        base = p0 + np.linspace(0.1, 0.9, samples, dtype=np.float32)[:, None] * d
        cand = base[:, None, :] + offsets[None, :, None] * normal  # (S, O, 2)
        xs = np.clip(np.round(cand[..., 0]).astype(int), 0, rw - 1)
        ys = np.clip(np.round(cand[..., 1]).astype(int), 0, rh - 1)
        strength = np.abs(gx[ys, xs] * normal[0] + gy[ys, xs] * normal[1])
        best = strength.argmax(axis=1)
        peak = strength[np.arange(samples), best]
        keep = peak >= 0.5 * np.median(peak)
        if keep.sum() < samples * 0.5:
            return None
        pts = cand[np.arange(samples), best][keep]
        line = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        vx, vy, lx, ly = line
        residual = np.abs((pts[:, 0] - lx) * vy - (pts[:, 1] - ly) * vx)
        if float(np.median(residual)) > max_residual:
            return None
        lines.append(line)
    corners = [_intersect(lines[i - 1], lines[i]) for i in range(4)]
    if any(c is None for c in corners):
        return None
    refined = np.array(corners, np.float32) + np.array([x0, y0], np.float32)
    if np.linalg.norm(refined - quad, axis=1).max() > 2 * search:
        return None
    return order_quad(refined)


def find_document_quads(frame_bgr: np.ndarray, min_area_ratio: float = 0.05,
                        per_strategy: int = 5) -> list[Quad]:
    """Toutes les feuilles candidates *vérifiées*, de la plus grande à la plus petite.

    Deux stratégies complémentaires : contours de Canny (papier sur fond
    texturé) et seuillage d'Otsu (papier clair sur fond plus sombre). Chaque
    candidat est affiné et vérifié au sous-pixel sur l'image pleine
    résolution (:func:`refine_quad`) ; les candidats qui échouent (contour
    fusionné avec un objet voisin, etc.) sont écartés.
    """
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, 640.0 / max(h, w))
    small = cv2.resize(frame_bgr, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA) if scale < 1 else frame_bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    min_area = min_area_ratio * gray.shape[0] * gray.shape[1]

    median = float(np.median(gray))
    edges = cv2.Canny(gray, int(max(0, 0.66 * median)), int(min(255, 1.33 * median)))
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    full_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    search = int(np.ceil(4 / scale)) + 2
    found: list[tuple[float, Quad]] = []
    for binary in (edges, bright):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:per_strategy]:
            if cv2.contourArea(contour) < min_area:
                break
            approx = _approx_quad(contour)
            if approx is None:
                continue
            quad = order_quad(approx / scale)
            if not is_valid_quad(quad, frame_bgr.shape):
                continue
            refined = refine_quad(full_gray, quad, search=search)
            if refined is None or not is_valid_quad(refined, frame_bgr.shape):
                continue
            # Les deux stratégies trouvent souvent la même feuille : dédoublonne.
            if any(np.abs(refined - q).max() < 3.0 for _, q in found):
                continue
            found.append((abs(cv2.contourArea(refined)), refined))
    found.sort(key=lambda item: item[0], reverse=True)
    return [quad for _, quad in found]


def detect_document_quad(frame_bgr: np.ndarray,
                         min_area_ratio: float = 0.05) -> Quad | None:
    """Plus grande feuille vérifiée (coins TL, TR, BR, BL en pixels) ou ``None``."""
    quads = find_document_quads(frame_bgr, min_area_ratio)
    return quads[0] if quads else None


# ---------------------------------------------------------------------------
# Suivi planaire
# ---------------------------------------------------------------------------

class PlanarTracker:
    """Suit un plan (la feuille) et renvoie ses 4 coins à chaque image."""

    def __init__(self, config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        self._orb = cv2.ORB_create(nfeatures=self.config.orb_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.reset()

    # -- état ---------------------------------------------------------------

    def reset(self) -> None:
        """Oublie la cible : il faudra rappeler :meth:`initialize`."""
        self._scale = 1.0
        self._ref_quad: Quad | None = None      # à l'échelle de suivi
        self._ref_kp = None
        self._ref_des = None
        self._prev_gray: np.ndarray | None = None
        self._ref_pts = np.empty((0, 2), np.float32)  # coords référence
        self._cur_pts = np.empty((0, 2), np.float32)  # coords courantes
        self._quad: Quad | None = None           # lissé, échelle de suivi
        self._T: np.ndarray | None = None        # gabarit -> référence
        self._template: np.ndarray | None = None
        self._frame_index = 0
        self.lost_frames = 0
        self.alignment = 0.0                     # dernier score ZNCC mesuré
        self.status = "non initialisé"

    @property
    def initialized(self) -> bool:
        return self._ref_quad is not None

    @property
    def quad(self) -> Quad | None:
        """Coins actuels en pixels de l'image d'origine (ou ``None``)."""
        if self._quad is None or self.lost_frames > 0:
            return None
        return self._quad / self._scale

    # -- initialisation -----------------------------------------------------

    def _to_gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._scale != 1.0:
            gray = cv2.resize(gray, None, fx=self._scale, fy=self._scale,
                              interpolation=cv2.INTER_AREA)
        return gray

    def initialize(self, frame_bgr: np.ndarray, quad: Quad) -> None:
        """Mémorise la référence : image, coins et descripteurs ORB."""
        h, w = frame_bgr.shape[:2]
        self.reset()
        self._scale = min(1.0, _TRACK_MAX_SIDE / max(h, w))
        gray = self._to_gray(frame_bgr)
        ref_quad = order_quad(np.asarray(quad, np.float32)) * self._scale

        self._ref_quad = ref_quad
        mask = _inner_mask(gray.shape, ref_quad)
        self._ref_kp, self._ref_des = self._orb.detectAndCompute(gray, mask)

        # Gabarit : intérieur de la feuille de référence, rectifié et réduit.
        a = _TEMPLATE_INSET
        inner = transform_quad(
            np.float32([[a, a], [1 - a, a], [1 - a, 1 - a], [a, 1 - a]]),
            cv2.getPerspectiveTransform(_UNIT_SQUARE, ref_quad))
        tw, th = canvas_size_for_quad(inner, _TEMPLATE_SIDE)
        self._T = cv2.getPerspectiveTransform(canvas_corners(tw, th), inner)
        self._template = _normalized_highpass(cv2.warpPerspective(
            gray, self._T, (tw, th), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP))

        self._prev_gray = gray
        self._quad = ref_quad.copy()
        self._seed_points(gray, ref_quad, np.eye(3))
        self.alignment = 1.0
        self.status = "suivi"

    def alignment_score(self, gray: np.ndarray, H: np.ndarray) -> float:
        """ZNCC passe-haut entre la référence et l'image recalée par ``H``.

        ~1 : alignement parfait ; ~0,75 : ~4 px d'erreur ; → 0 : mauvaise
        position. Coût ~0,3 ms (gabarit de 240 px).
        """
        th, tw = self._template.shape
        patch = cv2.warpPerspective(gray, H @ self._T, (tw, th),
                                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_REPLICATE)
        return float((self._template * _normalized_highpass(patch)).sum())

    def _seed_points(self, gray: np.ndarray, quad: Quad, H: np.ndarray) -> None:
        """(Ré)ensemence des coins de Shi-Tomasi dans le quadrilatère courant.

        Les nouveaux points sont reprojetés dans la référence via ``H⁻¹`` pour
        conserver des correspondances référence ↔ image courante.
        """
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=self.config.max_corners,
                                      qualityLevel=0.01, minDistance=7,
                                      mask=_inner_mask(gray.shape, quad), blockSize=7)
        if pts is None:
            return
        cur = pts.reshape(-1, 2).astype(np.float32)
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return
        ref = cv2.perspectiveTransform(cur.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
        self._cur_pts = cur.astype(np.float32)
        self._ref_pts = ref.astype(np.float32)

    # -- estimations --------------------------------------------------------

    def _track_flow(self, gray: np.ndarray) -> np.ndarray | None:
        """Flux optique LK avec contrôle aller-retour ; renvoie H réf → cour."""
        if self._prev_gray is None or len(self._cur_pts) < 4:
            return None
        p0 = self._cur_pts.reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, p0, None,
                                             **_LK_PARAMS)
        if p1 is None:
            return None
        p0r, st_back, _ = cv2.calcOpticalFlowPyrLK(gray, self._prev_gray, p1, None,
                                                   **_LK_PARAMS)
        fb_err = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
        good = (st.ravel() == 1) & (st_back.ravel() == 1) & (fb_err < 1.0)
        ref = self._ref_pts[good]
        cur = p1.reshape(-1, 2)[good]
        if len(cur) < 4:
            return None
        H, inliers = cv2.findHomography(ref, cur, cv2.RANSAC,
                                        self.config.ransac_threshold)
        if H is None:
            return None
        keep = inliers.ravel().astype(bool)
        self._ref_pts, self._cur_pts = ref[keep], cur[keep]
        return H

    def _track_orb(self, gray: np.ndarray, search_quad: Quad | None) -> np.ndarray | None:
        """Relocalisation par appariement ORB sur la référence."""
        if self._ref_des is None or len(self._ref_des) < 8:
            return None
        mask = None
        if search_quad is not None:
            # Cherche autour de la dernière position connue (fenêtre élargie).
            center = search_quad.mean(axis=0)
            grown = center + (search_quad - center) * 1.6
            mask = quad_mask(gray.shape, grown)
        kp, des = self._orb.detectAndCompute(gray, mask)
        if des is None or len(des) < 8:
            return None
        pairs = self._matcher.knnMatch(self._ref_des, des, k=2)
        good = [m for m, *rest in (p for p in pairs if p)
                if not rest or m.distance < 0.75 * rest[0].distance]
        if len(good) < 12:
            return None
        src = np.float32([self._ref_kp[m.queryIdx].pt for m in good])
        dst = np.float32([kp[m.trainIdx].pt for m in good])
        H, inliers = cv2.findHomography(src, dst, cv2.USAC_MAGSAC,
                                        self.config.ransac_threshold)
        if H is None or int(inliers.sum()) < 10:
            return None
        return H

    def _track_contour(self, frame_bgr: np.ndarray) -> Quad | None:
        """Redétection du polygone : garde le candidat le plus proche du précédent."""
        candidates = [q * self._scale for q in find_document_quads(frame_bgr)]
        if not candidates:
            return None
        if self._quad is None or self.lost_frames > 0:
            return candidates[0]
        dists = [float(np.linalg.norm(q - self._quad, axis=1).max()) for q in candidates]
        best = int(np.argmin(dists))
        # Une feuille ne se déplace pas d'un quart de sa taille en 1/30 s.
        diag = float(np.linalg.norm(self._quad[2] - self._quad[0]))
        return candidates[best] if dists[best] < 0.25 * diag else None

    def _plausible(self, quad: Quad, gray_shape: tuple[int, ...]) -> bool:
        """Rejette les homographies aberrantes (sauts, dégénérescences)."""
        if not is_valid_quad(quad, gray_shape):
            return False
        ref_area = abs(cv2.contourArea(self._ref_quad))
        area = abs(cv2.contourArea(quad))
        return 0.1 < area / max(ref_area, 1e-6) < 10.0

    def _verify(self, gray: np.ndarray, H: np.ndarray | None,
                search: Quad | None) -> np.ndarray | None:
        """Contrôle l'alignement du flux optique ; relocalise par ORB si besoin.

        ORB n'est lancé que si LK a échoué, manque de points, ou s'aligne mal.
        Son estimation n'est adoptée que si elle explique *mieux* l'image
        (score ZNCC supérieur d'au moins ``orb_margin``) : c'est le score, et
        non un seuil de distance, qui arbitre.
        """
        cfg = self.config
        weak = H is None or len(self._cur_pts) < cfg.min_tracked_points
        if not weak and self._frame_index % cfg.check_interval:
            return H
        score = self.alignment_score(gray, H) if H is not None else -1.0
        if weak or score < cfg.min_alignment:
            H_orb = self._track_orb(gray, search)
            if H_orb is not None and self._plausible(
                    transform_quad(self._ref_quad, H_orb), gray.shape):
                orb_score = self.alignment_score(gray, H_orb)
                floor = cfg.recover_alignment if H is None else score + cfg.orb_margin
                if orb_score > floor:
                    H, score = H_orb, orb_score
                    self._seed_points(gray, transform_quad(self._ref_quad, H), H)
        self.alignment = score
        return H

    # -- mise à jour --------------------------------------------------------

    def update(self, frame_bgr: np.ndarray) -> Quad | None:
        """Met à jour le suivi ; renvoie les coins (pixels d'origine) ou None."""
        if not self.initialized:
            return None
        gray = self._to_gray(frame_bgr)
        mode = self.config.mode
        self._frame_index += 1
        new_quad: Quad | None = None
        H: np.ndarray | None = None

        if mode == "contour":
            new_quad = self._track_contour(frame_bgr)
        else:
            search = self._quad if self.lost_frames == 0 else None
            if mode == "orb":
                H = self._track_orb(gray, search)
                if H is not None:
                    self.alignment = self.alignment_score(gray, H)
                    if self.alignment < self.config.recover_alignment:
                        H = None  # appariement ORB erroné
            else:
                H = self._track_flow(gray)
                if mode == "hybrid":
                    H = self._verify(gray, H, search)
            if H is not None:
                new_quad = transform_quad(self._ref_quad, H)

        self._prev_gray = gray

        if new_quad is None or not self._plausible(new_quad, gray.shape):
            self.lost_frames += 1
            self.status = f"perdu ({self.lost_frames})"
            return None

        if H is not None and mode in ("flow", "hybrid") and (
            len(self._cur_pts) < self.config.min_tracked_points
        ):
            self._seed_points(gray, new_quad, H)

        self._quad = self._smooth(new_quad)
        self.lost_frames = 0
        self.status = ("suivi (contour)" if mode == "contour" else
                       f"suivi ({len(self._cur_pts)} pts, corr {self.alignment:.2f})")
        return self._quad / self._scale

    def _smooth(self, quad: Quad) -> Quad:
        """Lissage exponentiel adaptatif : fort à l'arrêt, nul en mouvement."""
        alpha = self.config.smoothing
        if self._quad is None or alpha <= 0 or self.lost_frames > 0:
            return quad
        motion = float(np.linalg.norm(quad - self._quad, axis=1).mean())
        alpha_eff = alpha * math.exp(-motion / 3.0)
        return (alpha_eff * self._quad + (1.0 - alpha_eff) * quad).astype(np.float32)
