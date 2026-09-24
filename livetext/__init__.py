"""livetext — Remplacement de texte en direct sur plan suivi pour OBS.

Ce paquet fournit un pipeline temps réel qui :

* capture un flux webcam (OpenCV) ;
* suit une zone plane cible (une feuille de papier) par homographie ;
* efface le texte d'origine sans laisser de rectangle uni ;
* rend un nouveau texte vectoriel (Pillow / .ttf) déformé selon la perspective ;
* intègre le rendu de façon photométrique (fusion « Produit », luminosité,
  flou et grain) ;
* renvoie le flux modifié vers OBS via une caméra virtuelle (pyvirtualcam).

Le point d'entrée en ligne de commande est :func:`livetext.app.main` (exposé
aussi par ``python -m livetext`` et par ``main.py``).
"""

from __future__ import annotations

__all__ = [
    "__version__",
    "AppConfig",
    "TextReplacementPipeline",
    "PlanarTracker",
    "LiveTextApp",
]

__version__ = "0.1.0"


def __getattr__(name: str):  # pragma: no cover - simple lazy re-export
    # Import paresseux : importer le paquet ne doit pas forcer le chargement
    # d'OpenCV/Pillow tant qu'on n'utilise pas réellement le pipeline.
    if name == "AppConfig":
        from .config import AppConfig

        return AppConfig
    if name == "TextReplacementPipeline":
        from .pipeline import TextReplacementPipeline

        return TextReplacementPipeline
    if name == "PlanarTracker":
        from .tracker import PlanarTracker

        return PlanarTracker
    if name == "LiveTextApp":
        from .app import LiveTextApp

        return LiveTextApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
