import cv2
import numpy as np
import pytest

from livetext.config import AppConfig
from livetext.pipeline import TextReplacementPipeline

from conftest import make_scene


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


# -- Couleur de l'encre réelle et décalage sous-pixel --------------------------

BLUE_INK = (150, 60, 40)  # BGR : stylo bille bleu


def _new_text_pixels(out, mask, threshold=150):
    inner = cv2.erode(mask, np.ones((7, 7), np.uint8)) > 0
    return out[(out.max(axis=2) < threshold) & inner]


def test_new_text_takes_the_tint_of_the_printed_ink():
    """Réalisme 1:1 : pas de noir ni de gris neutre, la teinte de l'encre réelle."""
    frame, quad, mask = make_scene(ink=BLUE_INK)
    pipe = _pipeline("Nouveau texte")
    pipe.set_target(quad)
    for _ in range(3):
        out = pipe.process(frame, quad)
    np.testing.assert_allclose(pipe.debug.ink_bgr, BLUE_INK, atol=6)
    b, g, r = _new_text_pixels(out, mask).mean(axis=0)
    assert b > r + 60 and b > g + 50            # nettement bleu, comme l'original
    assert min(b, g, r) > 20                    # jamais un noir pur


def test_without_ink_sampling_the_text_is_neutral_gray(scene):
    frame, quad, mask = make_scene(ink=BLUE_INK)
    pipe = _pipeline("Nouveau texte")
    pipe.config.photometry.sample_ink = False
    pipe.set_target(quad)
    out = pipe.process(frame, quad)
    b, g, r = _new_text_pixels(out, mask).mean(axis=0)
    assert abs(b - r) < 15


def _text_centroid_x(pipe, frame, quad, mask):
    """Barycentre (x) de l'encre, mesuré uniquement *dans* la feuille : le bureau
    sombre autour, immobile, écraserait sinon le déplacement du texte."""
    out = pipe.process(frame, quad).astype(np.float32).mean(axis=2)
    inner = cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0
    ink = np.where(inner, np.clip(180.0 - out, 0, None), 0)
    ys, xs = np.nonzero(ink)
    return float((xs * ink[ys, xs]).sum() / ink[ys, xs].sum())


def test_nudge_moves_text_by_half_a_pixel_on_screen(scene):
    frame, quad, mask = scene
    # Texte d'origine effacé (sinon, immobile, il domine le barycentre) ; sans
    # grain aléatoire pour une mesure déterministe.
    pipe = _pipeline("DECALAGE", grain_strength=0.0)
    pipe.config.photometry.noise_sigma = 0.0
    pipe.set_target(quad)
    before = _text_centroid_x(pipe, frame, quad, mask)
    pipe.nudge_text(0.5, 0.0)
    after = _text_centroid_x(pipe, frame, quad, mask)
    assert after - before == pytest.approx(0.5, abs=0.12)   # mesuré : +0,455 px
    pipe.nudge_text(0.5, 0.0)
    assert _text_centroid_x(pipe, frame, quad, mask) - before == pytest.approx(1.0, abs=0.12)
    pipe.reset_offset()
    assert _text_centroid_x(pipe, frame, quad, mask) == pytest.approx(before, abs=0.02)


def test_font_size_change_is_applied_live(scene):
    frame, quad, _ = scene
    pipe = _pipeline("Taille")
    pipe.set_target(quad)
    pipe.process(frame, quad)
    auto = pipe.renderer.last_size
    assert pipe.adjust_font_size(+3) == auto + 3
    pipe.process(frame, quad)
    assert pipe.renderer.last_size == auto + 3  # re-rendu sans relancer


def _noisy(base, sigma, seed):
    n = np.empty(base.shape, np.float32)
    cv2.setRNGSeed(seed)
    cv2.randn(n, (0, 0, 0), (sigma,) * 3)
    return np.clip(base.astype(np.float32) + n, 0, 255).astype(np.uint8)


def test_iso_grain_is_calibrated_on_the_camera_noise():
    """Plus la caméra est bruitée, plus le grain ajouté sur l'encre l'est."""
    base, quad, _ = make_scene()
    measured = []
    for sigma in (1.0, 4.0, 8.0):
        pipe = _pipeline("Calibrage")
        pipe.set_target(quad)
        for k in range(8):
            pipe.process(_noisy(base, sigma, k), quad)
        measured.append(pipe.debug.noise_sigma)
    assert measured[0] < measured[1] < measured[2]
    # Bruit de luminance : ~0,67 × σ par canal (canaux indépendants).
    assert measured[2] == pytest.approx(0.67 * 8.0, rel=0.2)


def test_fixed_noise_when_calibration_is_off(scene):
    frame, quad, _ = scene
    pipe = _pipeline("Fixe")
    pipe.config.photometry.noise_auto = False
    pipe.config.photometry.noise_sigma = 3.5
    pipe.set_target(quad)
    pipe.process(frame, quad)
    assert pipe.debug.noise_sigma == 3.5
