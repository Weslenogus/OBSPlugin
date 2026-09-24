"""Filtre passe-bas adaptatif « One Euro » (Casiez, Roussel & Vogel, CHI 2012).

Référence pour lisser des coordonnées suivies en temps réel : la fréquence
de coupure augmente avec la vitesse du signal. À l'arrêt, le lissage est fort
(le texte ne tremble plus) ; en mouvement, la coupure monte et le retard reste
de l'ordre du pixel (le texte ne « traîne » pas derrière la feuille).

Ici on filtre les 4 coins de l'homographie ensemble, avec une coupure
*commune* calculée sur la vitesse moyenne des coins : tous les coins ont le
même retard, donc le quadrilatère se déplace sans se cisailler.
"""

from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff_hz: float, dt: float) -> float:
    """Coefficient d'un passe-bas du premier ordre de coupure ``cutoff_hz``."""
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Filtre One Euro sur un tableau de coordonnées (ex. coins (4, 2)).

    * ``min_cutoff`` (Hz) : coupure à l'arrêt ; plus bas = plus lisse.
    * ``beta`` : gain de la coupure avec la vitesse (px/s) ; plus haut =
      moins de retard en mouvement.
    * ``d_cutoff`` (Hz) : lissage de la vitesse estimée.
    """

    def __init__(self, rate_hz: float = 30.0, min_cutoff: float = 1.0,
                 beta: float = 0.05, d_cutoff: float = 1.0):
        self.dt = 1.0 / max(rate_hz, 1e-3)
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        """Oublie l'historique (à appeler après une discontinuité du suivi)."""
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self.min_cutoff <= 0:
            self._x, self._dx = x.copy(), np.zeros_like(x)
            return x.astype(np.float32)
        # Vitesse estimée puis lissée (sa propre coupure, fixe).
        dx = (x - self._x) / self.dt
        self._dx += _alpha(self.d_cutoff, self.dt) * (dx - self._dx)
        # Vitesse commune : norme moyenne des déplacements des points (px/s).
        pts = self._dx.reshape(-1, 2) if self._dx.size % 2 == 0 else self._dx.reshape(-1, 1)
        speed = float(np.linalg.norm(pts, axis=1).mean())
        cutoff = self.min_cutoff + self.beta * speed
        self._x += _alpha(cutoff, self.dt) * (x - self._x)
        return self._x.astype(np.float32)
