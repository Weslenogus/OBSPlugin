"""Accélération GPU : impossible à exécuter sans CUDA, mais sa logique de
sélection et de repli est testée avec un ``cv2.cuda`` simulé."""

import cv2
import numpy as np
import pytest

from livetext import accel
from livetext.accel import Accelerator

IMG = cv2.GaussianBlur(np.random.default_rng(0).integers(0, 255, (120, 160, 3),
                                                         dtype=np.uint8), (0, 0), 2)
M = np.array([[1.0, 0.02, 3.0], [0.01, 1.0, 2.0], [0.0, 0.0, 1.0]])


def _cpu(img, m, dsize, flags, border):
    return cv2.warpPerspective(img, m, dsize, flags=flags, borderMode=border)


def test_off_mode_never_uses_gpu():
    a = Accelerator("off")
    assert not a.enabled
    np.testing.assert_array_equal(a.warp_perspective(IMG, M, (160, 120)),
                                  cv2.warpPerspective(IMG, M, (160, 120)))


def test_auto_without_cuda_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(accel, "cuda_device_count", lambda: 0)
    a = Accelerator("auto")
    assert not a.enabled and "CUDA" in a.reason


def test_gpu_enabled_only_after_a_passing_self_test(monkeypatch):
    monkeypatch.setattr(accel, "cuda_device_count", lambda: 1)
    monkeypatch.setattr(Accelerator, "_gpu_warp", staticmethod(_cpu))  # GPU simulé exact
    a = Accelerator("auto")
    assert a.enabled


def test_gpu_rejected_when_self_test_diverges(monkeypatch):
    monkeypatch.setattr(accel, "cuda_device_count", lambda: 1)
    monkeypatch.setattr(Accelerator, "_gpu_warp",
                        staticmethod(lambda img, *a: np.zeros_like(_cpu(img, *a))))
    a = Accelerator("on")
    assert not a.enabled and "auto-test" in a.reason


def test_runtime_gpu_error_falls_back_to_cpu_for_good(monkeypatch):
    monkeypatch.setattr(accel, "cuda_device_count", lambda: 1)
    monkeypatch.setattr(Accelerator, "_gpu_warp", staticmethod(_cpu))
    a = Accelerator("auto")

    def broken(*args):
        raise cv2.error("CUDA out of memory")

    monkeypatch.setattr(Accelerator, "_gpu_warp", staticmethod(broken))
    out = a.warp_perspective(IMG, M, (160, 120), cv2.INTER_LINEAR, cv2.BORDER_REPLICATE)
    np.testing.assert_array_equal(out, _cpu(IMG, M, (160, 120), cv2.INTER_LINEAR,
                                            cv2.BORDER_REPLICATE))
    assert not a.enabled  # le direct continue, définitivement sur CPU


@pytest.mark.parametrize("mode", ["auto", "on", "off"])
def test_pipeline_output_is_identical_whatever_the_gpu_mode(mode, monkeypatch):
    """Sans CUDA ici, les trois modes doivent donner exactement le même rendu."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from conftest import make_scene

    from livetext.config import AppConfig
    from livetext.pipeline import TextReplacementPipeline
    frame, quad, _ = make_scene()
    outs = []
    for m in ("off", mode):
        cfg = AppConfig(gpu=m)
        cfg.photometry.noise_auto = False
        cfg.photometry.noise_sigma = 0.0
        cfg.erase.grain_strength = 0.0
        p = TextReplacementPipeline(cfg, seed=0)
        p.set_target(quad)
        outs.append(p.process(frame, quad))
    np.testing.assert_array_equal(outs[0], outs[1])
