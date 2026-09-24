import numpy as np
import pytest

from livetext.compositor import (
    NoiseBank,
    composite_ink,
    estimate_noise_sigma,
    ink_multiply_factor,
    ink_multiply_factors,
    measure_ink_color,
    measure_paper_color,
    measure_paper_luminance,
)
from livetext.config import PhotometryConfig


def _flat(value, h=40, w=60):
    return np.full((h, w, 3), value, np.uint8)


def test_multiply_leaves_paper_untouched_without_ink():
    cfg = PhotometryConfig(blur_sigma=0, noise_sigma=0)
    roi = _flat(200)
    out = composite_ink(roi, np.zeros((40, 60), np.float32), np.full(3, 0.2), cfg,
                        NoiseBank(0))
    np.testing.assert_array_equal(out, roi)


def test_gray_fallback_reaches_paper_luminance_times_ratio():
    """Sans encre visible : gris = luminance du papier × ink_ratio."""
    cfg = PhotometryConfig(ink_ratio=0.25, blur_sigma=0, noise_sigma=0)
    factors = ink_multiply_factors(np.full(3, 200.0), None, cfg)
    out = composite_ink(_flat(200), np.ones((40, 60), np.float32), factors, cfg, NoiseBank(0))
    assert out.mean() == pytest.approx(50, abs=1)


def test_multiply_preserves_paper_texture_under_ink():
    """Plis et ombres doivent rester visibles sous l'encre (fusion Produit)."""
    cfg = PhotometryConfig(ink_ratio=0.3, blur_sigma=0, noise_sigma=0)
    roi = _flat(200)
    roi[:, 30:] = 120  # une ombre sur la moitié droite
    factors = ink_multiply_factors(np.full(3, 160.0), None, cfg)
    out = composite_ink(roi, np.ones((40, 60), np.float32), factors, cfg, NoiseBank(0))
    left, right = out[:, :30].mean(), out[:, 30:].mean()
    assert right < left
    assert left / right == pytest.approx(200 / 120, rel=0.05)


def test_ink_factor_is_softened_on_dark_paper():
    cfg = PhotometryConfig(ink_ratio=0.2, ink_min=30)
    assert ink_multiply_factor(220, cfg) == pytest.approx(0.2)
    # 60 × 0,2 = 12 < ink_min : l'encre reste à 30, soit un facteur de 0,5.
    assert ink_multiply_factor(60, cfg) == pytest.approx(0.5)
    assert ink_multiply_factor(10, cfg) == pytest.approx(1.0)


def test_noise_is_confined_to_ink():
    cfg = PhotometryConfig(blur_sigma=0, noise_sigma=5)
    cov = np.zeros((40, 60), np.float32)
    cov[:, :20] = 1.0
    out = composite_ink(_flat(200), cov, np.full(3, 0.2), cfg, NoiseBank(0))
    assert out[:, 25:].std() == 0
    assert out[:, :20].std() > 1


def test_measure_paper_luminance_ignores_text_pixels():
    canvas = _flat(180, 100, 100)
    mask = np.zeros((100, 100), np.uint8)
    canvas[40:60, 40:60] = 20
    mask[40:60, 40:60] = 255
    assert measure_paper_luminance(canvas, mask, (30, 30, 70, 70), 10) == pytest.approx(180, abs=1)


def test_noise_bank_does_not_regenerate_for_alternating_shapes():
    """Régression : deux appelants de tailles différentes se forçaient
    mutuellement à régénérer la réserve (≈ 12 ms perdues par image)."""
    bank = NoiseBank(0)
    bank.sample(300, 100)
    bank.sample(100, 300)
    reserve = bank._bank
    for _ in range(5):
        assert bank.sample(300, 100).shape == (300, 100)
        assert bank.sample(100, 300).shape == (100, 300)
    assert bank._bank is reserve


def test_noise_bank_does_not_regenerate_while_the_request_creeps_up():
    """Régression : quand la feuille approche, la zone de texte grandit de
    quelques pixels par image ; la réserve était régénérée à chaque image
    (≈ 4 ms perdues par image en 1080p)."""
    bank = NoiseBank(0)
    regenerations, last = 0, None
    for step in range(200):
        bank.sample(300 + step, 500 + 2 * step)
        if bank._bank is not last:
            regenerations, last = regenerations + 1, bank._bank
    assert regenerations <= 3


# -- Couleur de l'encre réelle (réalisme 1:1) ---------------------------------

BLUE_INK = np.array([150.0, 60.0, 40.0])   # BGR : encre de stylo bleue
PAPER = np.array([205.0, 222.0, 230.0])    # BGR : papier légèrement crème


def _printed_canvas():
    """Papier crème, traits d'encre bleue épais + halo anticrénelé autour."""
    canvas = np.tile(PAPER, (120, 200, 1)).astype(np.uint8)
    mask = np.zeros((120, 200), np.uint8)
    canvas[40:60, 30:170] = BLUE_INK                      # cœur des traits
    canvas[37:40, 30:170] = (PAPER + BLUE_INK) / 2        # halo (plus clair)
    canvas[60:63, 30:170] = (PAPER + BLUE_INK) / 2
    mask[35:65, 28:172] = 255                             # masque dilaté
    return canvas, mask


def test_ink_colour_is_sampled_from_stroke_cores_not_halo():
    canvas, mask = _printed_canvas()
    paper = measure_paper_color(canvas, mask, (30, 40, 170, 60), 10)
    np.testing.assert_allclose(paper, PAPER, atol=1)
    ink = measure_ink_color(canvas, mask, paper, (30, 40, 170, 60), 10)
    np.testing.assert_allclose(ink, BLUE_INK, atol=1)


def test_no_visible_ink_returns_none():
    canvas = np.tile(PAPER, (120, 200, 1)).astype(np.uint8)
    assert measure_ink_color(canvas, np.zeros((120, 200), np.uint8),
                             PAPER, (0, 0, 200, 120), 10) is None


def test_sampled_ink_tint_is_reproduced_exactly_by_multiply():
    """Couverture pleine sur ce papier = exactement la teinte de l'encre."""
    cfg = PhotometryConfig(blur_sigma=0, noise_sigma=0)
    factors = ink_multiply_factors(PAPER, BLUE_INK, cfg)
    roi = np.tile(PAPER, (40, 60, 1)).astype(np.uint8)
    out = composite_ink(roi, np.ones((40, 60), np.float32), factors, cfg, NoiseBank(0))
    np.testing.assert_allclose(out.reshape(-1, 3).mean(axis=0), BLUE_INK, atol=1)


def test_sampled_ink_is_never_pure_black():
    cfg = PhotometryConfig()
    factors = ink_multiply_factors(PAPER, BLUE_INK, cfg)
    assert (factors > 0.1).all()


@pytest.mark.parametrize("sigma", [1.0, 3.0, 8.0])
def test_noise_estimator_measures_sensor_noise(sigma):
    rng = np.random.default_rng(0)
    img = np.clip(128 + rng.normal(0, sigma, (200, 300)), 0, 255).astype(np.uint8)
    # Le passe-haut (image - flou σ=1,5) retire ~5 % de l'énergie du bruit blanc.
    assert estimate_noise_sigma(img) == pytest.approx(0.945 * sigma, rel=0.1)


def test_noise_estimator_ignores_masked_ink():
    rng = np.random.default_rng(0)
    img = np.clip(200 + rng.normal(0, 2.0, (200, 300)), 0, 255).astype(np.uint8)
    mask = np.zeros((200, 300), np.uint8)
    img[50:150:4, :] = 20            # traits d'encre très contrastés
    mask[50:150, :] = 255
    assert estimate_noise_sigma(img, mask) == pytest.approx(0.945 * 2.0, rel=0.15)


def test_explicit_noise_sigma_overrides_config():
    cfg = PhotometryConfig(blur_sigma=0, noise_sigma=0.0)
    out = composite_ink(_flat(200), np.ones((40, 60), np.float32), np.full(3, 0.5), cfg,
                        NoiseBank(0), noise_sigma=6.0)
    assert out.std() > 3


def test_black_point_follows_a_shadow_proportionally():
    """Black point matching : une ombre sur la feuille assombrit l'encre
    virtuelle dans la même proportion (jamais un noir figé)."""
    cfg = PhotometryConfig(blur_sigma=0, noise_sigma=0)
    factors = ink_multiply_factors(PAPER, BLUE_INK, cfg)
    cov = np.ones((40, 60), np.float32)
    lit = composite_ink(np.tile(PAPER, (40, 60, 1)).astype(np.uint8), cov, factors, cfg,
                        NoiseBank(0))
    shadow = composite_ink((np.tile(PAPER, (40, 60, 1)) * 0.6).astype(np.uint8), cov,
                           factors, cfg, NoiseBank(0))
    ratio = shadow.reshape(-1, 3).mean(axis=0) / lit.reshape(-1, 3).mean(axis=0)
    np.testing.assert_allclose(ratio, 0.6, atol=0.03)
