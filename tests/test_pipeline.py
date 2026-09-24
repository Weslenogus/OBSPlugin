import cv2
import numpy as np
import pytest

from livetext.config import AppConfig
from livetext.pipeline import TextReplacementPipeline


def _pipeline(text="Nouveau", **erase):
    cfg = AppConfig()
    cfg.text.initial_text = text
    for key, value in erase.items():
        setattr(cfg.erase, key, value)
    return TextReplacementPipeline(cfg, seed=0)


def _dark(img, mask):
    return int(((img.max(axis=2) < 90) & (mask > 0)).sum())


def test_process_replaces_text_only_inside_the_sheet(scene):
    frame, quad, mask = scene
    pipe = _pipeline("")  # effacement seul
    pipe.set_target(quad)
    out = pipe.process(frame, quad)

    assert out.shape == frame.shape and out.dtype == np.uint8
    # L'ancien texte a disparu de la feuille (hors liseré anticrénelé du bord,
    # où le papier se mélange au fond sombre : ce n'est pas du texte)…
    inner = cv2.erode(mask, np.ones((7, 7), np.uint8))
    assert _dark(frame, inner) > 2000
    assert _dark(out, inner) == 0
    # …et rien n'a bougé hors de la feuille.
    outside = mask == 0
    np.testing.assert_array_equal(out[outside], frame[outside])


def test_process_draws_new_text_in_perspective(scene):
    frame, quad, mask = scene
    pipe = _pipeline("NOUVEAU TEXTE")
    pipe.set_target(quad)
    out = pipe.process(frame, quad)
    ink = (out.max(axis=2) < 110) & (mask > 0)
    assert ink.sum() > 1500
    # L'encre suit la luminosité mesurée : gris foncé, pas noir pur.
    assert 20 < out[ink].mean() < 110
    assert pipe.debug.paper_luminance == pytest.approx(210, abs=20)


def test_text_change_is_applied_on_next_frame(scene):
    frame, quad, _ = scene
    pipe = _pipeline("A")
    pipe.set_target(quad)
    first = pipe.process(frame, quad)
    pipe.set_text("B C D E F G")
    second = pipe.process(frame, quad)
    assert np.abs(first.astype(int) - second.astype(int)).sum() > 0


def test_process_without_target_is_passthrough(scene):
    frame, quad, _ = scene
    pipe = _pipeline()
    assert pipe.process(frame, quad) is frame        # pas de set_target
    pipe.set_target(quad)
    assert pipe.process(frame, None) is frame        # suivi perdu


def test_sheet_partially_out_of_frame(scene):
    frame, quad, _ = scene
    shifted = quad + np.float32([-300, 0])            # déborde à gauche
    pipe = _pipeline("Bord")
    pipe.set_target(quad)
    out = pipe.process(frame, shifted)
    assert out.shape == frame.shape
