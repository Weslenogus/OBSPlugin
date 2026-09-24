import numpy as np
import pytest

from livetext.compositor import (
    NoiseBank,
    composite_ink,
    ink_multiply_factor,
    measure_paper_luminance,
)
from livetext.config import PhotometryConfig


def _flat(value, h=40, w=60):
    return np.full((h, w, 3), value, np.uint8)


def test_multiply_leaves_paper_untouched_without_ink():
    cfg = PhotometryConfig(blur_sigma=0, noise_sigma=0)
    roi = _flat(200)
    out = composite_ink(roi, np.zeros((40, 60), np.float32), 200.0, cfg, NoiseBank(0))
    np.testing.assert_array_equal(out, roi)


def test_full_ink_reaches_paper_luminance_times_ratio():
    cfg = PhotometryConfig(ink_ratio=0.25, blur_sigma=0, noise_sigma=0)
    out = composite_ink(_flat(200), np.ones((40, 60), np.float32), 200.0, cfg, NoiseBank(0))
    assert out.mean() == pytest.approx(50, abs=1)


def test_multiply_preserves_paper_texture_under_ink():
    """Plis et ombres doivent rester visibles sous l'encre (fusion Produit)."""
    cfg = PhotometryConfig(ink_ratio=0.3, blur_sigma=0, noise_sigma=0)
    roi = _flat(200)
    roi[:, 30:] = 120  # une ombre sur la moitié droite
    out = composite_ink(roi, np.ones((40, 60), np.float32), 160.0, cfg, NoiseBank(0))
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
    out = composite_ink(_flat(200), cov, 200.0, cfg, NoiseBank(0))
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
