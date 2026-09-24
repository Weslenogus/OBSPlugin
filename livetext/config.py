"""Configuration de l'application (valeurs par défaut + surcharge CLI)."""

from __future__ import annotations

from dataclasses import dataclass, field

# Résolutions supportées par le cahier des charges.
RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1080p": (1920, 1080),
    "720p": (1280, 720),
}


@dataclass
class TrackerConfig:
    """Paramètres du suivi planaire."""

    # "hybrid" : flux optique LK image à image + recalage ORB sur la référence.
    # "flow"   : flux optique LK seul (rapide, peut dériver).
    # "orb"    : appariement ORB sur la référence à chaque image (robuste).
    # "contour": redétection du polygone 4 points à chaque image.
    mode: str = "hybrid"
    max_corners: int = 300          # points suivis par le flux optique
    min_tracked_points: int = 25    # en dessous : réensemencement / recalage
    orb_features: int = 1000
    # Contrôle d'alignement (hybrid) toutes les N images : corrélation (ZNCC)
    # entre la référence rectifiée et l'image courante recalée. Coût ~0,3 ms.
    check_interval: int = 10
    # Score sous lequel le flux optique est jugé douteux → relocalisation ORB.
    min_alignment: float = 0.6
    # Score minimal pour accepter une relocalisation quand le suivi est perdu.
    recover_alignment: float = 0.45
    # ORB ne remplace le flux optique que s'il s'aligne nettement mieux :
    # LK est plus précis qu'ORB tant qu'il n'a pas décroché.
    orb_margin: float = 0.03
    ransac_threshold: float = 3.0
    smoothing: float = 0.5          # lissage exponentiel des coins (0 = aucun)


@dataclass
class EraseConfig:
    """Paramètres d'effacement du texte d'origine."""

    # "plate"   : plaque de fond estimée par fermeture morphologique
    #             (conserve ombres et plis) — défaut recommandé.
    # "inpaint" : cv2.inpaint (Telea) sur les pixels de texte détectés.
    # "none"    : pas d'effacement.
    method: str = "plate"
    # Zone à effacer dans le canevas, en coordonnées normalisées
    # (x0, y0, x1, y1) ; (0, 0, 1, 1) = toute la feuille.
    region: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    block_size: int = 31            # fenêtre du seuillage adaptatif
    threshold_c: int = 12           # sensibilité du seuillage adaptatif
    stroke_dilate: int = 3          # dilatation du masque de texte (px)
    inpaint_radius: int = 5
    feather: int = 7                # adoucissement des bords du masque (px)
    grain_strength: float = 1.0     # réinjection du grain du papier
    # Mémoire temporelle du masque (le canevas est recalé sur le papier, donc
    # l'ancien texte reste au même endroit d'une image à l'autre) : supprime
    # le scintillement des bords de traits. 0 = aucune mémoire.
    temporal_decay: float = 0.85


@dataclass
class TextConfig:
    """Paramètres du rendu du nouveau texte."""

    initial_text: str = "Bonjour OBS !"
    font_path: str | None = None    # .ttf fourni ; None = police système
    # Taille relative à la hauteur du canevas (0.12 = 12 %) si font_size == 0.
    font_size: int = 0
    relative_font_size: float = 0.12
    # Boîte de texte dans le canevas (coordonnées normalisées x0, y0, x1, y1).
    box: tuple[float, float, float, float] = (0.08, 0.08, 0.92, 0.92)
    align: str = "left"             # left | center | right
    valign: str = "top"             # top | middle | bottom
    line_spacing: float = 1.15
    auto_fit: bool = True           # réduit la police si le texte déborde


@dataclass
class PhotometryConfig:
    """Paramètres d'intégration photométrique."""

    # Rapport luminance encre / luminance papier (fusion Produit).
    ink_ratio: float = 0.22
    ink_min: float = 18.0           # luminance minimale de l'encre (0-255)
    ink_tint: tuple[float, float, float] = (1.0, 1.0, 1.0)  # teinte BGR
    blur_sigma: float = 0.8         # flou gaussien de l'encre (défocalisation)
    noise_sigma: float = 2.5        # bruit gaussien (grain capteur)
    luminance_ring: int = 24        # marge de mesure autour du texte (px canevas)


@dataclass
class AppConfig:
    """Configuration complète de l'application."""

    # Source : index de caméra ("0"), chemin vidéo/image, ou "synthetic".
    source: str = "0"
    resolution: str = "720p"
    fps: int = 30
    # Sorties : "virtualcam", "window", "file:<chemin.mp4>", "none".
    outputs: list[str] = field(default_factory=lambda: ["virtualcam", "window"])
    virtualcam_backend: str | None = None  # ex. "obs", "v4l2loopback"
    mirror: bool = False
    # Initialisation de la cible : "auto" (détection du papier), "manual"
    # (4 clics), ou "x1,y1,x2,y2,x3,y3,x4,y4" en pixels.
    init: str = "auto"
    canvas_max_side: int = 960
    show_debug: bool = False
    stdin_input: bool = True

    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    erase: EraseConfig = field(default_factory=EraseConfig)
    text: TextConfig = field(default_factory=TextConfig)
    photometry: PhotometryConfig = field(default_factory=PhotometryConfig)

    @property
    def frame_size(self) -> tuple[int, int]:
        """(largeur, hauteur) de la résolution demandée."""
        try:
            return RESOLUTIONS[self.resolution]
        except KeyError as exc:  # pragma: no cover - validé par la CLI
            raise ValueError(
                f"Résolution inconnue {self.resolution!r} "
                f"(choix : {', '.join(RESOLUTIONS)})"
            ) from exc
