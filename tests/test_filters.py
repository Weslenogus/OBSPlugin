import numpy as np
import pytest

from livetext.config import TrackerConfig
from livetext.filters import OneEuroFilter

BASE = np.float32([[300, 150], [900, 160], [880, 600], [290, 590]])


def _default_filter():
    cfg = TrackerConfig()
    return OneEuroFilter(30, cfg.smooth_min_cutoff, cfg.smooth_beta, cfg.smooth_d_cutoff)


def _jitter(seq):
    return float(np.linalg.norm(np.diff(np.asarray(seq), axis=0), axis=2).mean())


@pytest.mark.parametrize("sigma,kept", [(0.02, 0.15), (0.1, 0.25), (0.3, 0.40)])
def test_jitter_at_rest_is_strongly_reduced(sigma, kept):
    """Mesuré : 10 %, 18 %, 32 % du tremblement conservé."""
    rng = np.random.default_rng(0)
    noisy = [BASE + rng.normal(0, sigma, BASE.shape).astype(np.float32) for _ in range(200)]
    f = _default_filter()
    out = [f(q) for q in noisy]
    assert _jitter(out[30:]) < kept * _jitter(noisy[30:])


@pytest.mark.parametrize("speed", [1.0, 2.0, 6.0, 15.0])
def test_sub_pixel_lag_when_motion_starts(speed):
    """Le texte ne « traîne » pas derrière la feuille (mesuré : ≤ 0,67 px)."""
    truth = [BASE + (0 if t < 30 else speed * (t - 30)) * np.float32([1, 0.3])
             for t in range(80)]
    f = _default_filter()
    out = np.array([f(q) for q in truth])
    assert np.linalg.norm(out - np.array(truth), axis=2).max() < 1.0


def test_shared_cutoff_keeps_the_quad_rigid():
    """Tous les coins ont le même retard : la feuille ne se cisaille pas."""
    f = _default_filter()
    f(BASE)
    shifts = f(BASE + np.float32([10, 0])) - BASE
    np.testing.assert_allclose(shifts, np.broadcast_to(shifts[0], shifts.shape), atol=1e-4)


def test_reset_and_disabled_filter_pass_input_through():
    f = _default_filter()
    f(BASE)
    f.reset()
    np.testing.assert_array_equal(f(BASE + 50), BASE + 50)
    off = OneEuroFilter(30, 0.0)
    off(BASE)
    np.testing.assert_array_equal(off(BASE + 7), BASE + 7)
