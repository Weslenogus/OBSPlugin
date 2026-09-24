"""Accélération GPU : testée sans GPU grâce à un ``cv2.cuda`` simulé et strict
(``fake_cuda``), qui impose les contraintes du vrai OpenCV CUDA."""

import cv2
import numpy as np
import pytest

import fake_cuda
from conftest import make_scene
from livetext.accel import Accelerator
from livetext.config import AppConfig
from livetext.pipeline import TextReplacementPipeline


def _pipeline(accelerator, grain=False):
    cfg = AppConfig()
    cfg.text.initial_text = "Texte GPU"
    if not grain:  # grains aléatoires : non comparables pixel à pixel
        cfg.photometry.noise_auto = False
        cfg.photometry.noise_sigma = 0.0
        cfg.erase.grain_strength = 0.0
    return TextReplacementPipeline(cfg, seed=0, accel=accelerator)


def _run(accelerator, frame, quad, grain=False):
    pipe = _pipeline(accelerator, grain)
    pipe.set_target(quad)
    return pipe.process(frame, quad)


def test_off_mode_never_uses_gpu(monkeypatch):
    fake_cuda.install(monkeypatch)
    assert not Accelerator("off").enabled


def test_auto_without_cuda_stays_on_cpu(monkeypatch):
    fake_cuda.install(monkeypatch, devices=0)
    a = Accelerator("auto")
    assert not a.enabled and "CUDA" in a.reason


def test_outdated_driver_gives_an_actionable_reason(monkeypatch):
    """-1 : OpenCV compilé avec CUDA mais pilote absent / trop ancien."""
    fake_cuda.install(monkeypatch, devices=-1)
    a = Accelerator("on")
    assert not a.enabled and "pilote" in a.reason and "580" in a.reason


def test_real_opencv_here_has_no_cuda():
    """Ce conteneur (pip opencv) n'a pas CUDA : repli CPU sans erreur."""
    a = Accelerator("auto")
    assert not a.enabled


def test_gpu_enabled_after_passing_self_test(monkeypatch):
    fake_cuda.install(monkeypatch)
    a = Accelerator("auto")
    assert a.enabled, a.reason


def test_gpu_chain_matches_cpu_chain(monkeypatch):
    """Toute la chaîne GPU (rectification, effacement, encre) = référence CPU."""
    frame, quad, _ = make_scene()
    cpu = _run(Accelerator("off"), frame, quad)
    fake_cuda.install(monkeypatch)
    gpu = _run(Accelerator("auto"), frame, quad)
    diff = np.abs(cpu.astype(np.int16) - gpu.astype(np.int16))
    assert diff.mean() < 0.05 and diff.max() <= 1   # arrondi final uniquement


def test_frame_is_uploaded_to_gpu_exactly_once(monkeypatch):
    """Un envoi 1080p coûte autant que le calcul : il ne doit avoir lieu qu'une fois."""
    frame, quad, _ = make_scene()
    fake_cuda.install(monkeypatch)
    pipe = _pipeline(Accelerator("auto"))
    pipe.set_target(quad)
    uploads = []
    original = fake_cuda.GpuMat.upload
    monkeypatch.setattr(fake_cuda.GpuMat, "upload",
                        lambda self, a: (uploads.append(np.shape(a)), original(self, a)))
    before = fake_cuda.GpuMat.downloads
    pipe.process(frame, quad)
    assert uploads.count(frame.shape) == 1
    assert fake_cuda.GpuMat.downloads - before == 2   # canevas + rectangle modifié


def test_gpu_grain_is_applied_and_calibrated(monkeypatch):
    frame, quad, mask = make_scene()
    fake_cuda.install(monkeypatch)
    pipe = _pipeline(Accelerator("auto"), grain=True)
    pipe.set_target(quad)
    out = pipe.process(frame, quad)
    assert pipe.debug.noise_sigma > 0.5
    assert out.shape == frame.shape


def test_runtime_gpu_error_recomputes_frame_on_cpu(monkeypatch):
    frame, quad, _ = make_scene()
    expected = _run(Accelerator("off"), frame, quad)
    fake = fake_cuda.install(monkeypatch)
    a = Accelerator("auto")
    assert a.enabled

    def out_of_memory(*args, **kwargs):
        raise cv2.error("CUDA out of memory")

    monkeypatch.setattr(fake, "blendLinear", out_of_memory)
    out = _run(a, frame, quad)
    np.testing.assert_array_equal(out, expected)   # image correcte malgré l'erreur
    assert not a.enabled and "out of memory" in a.reason


def test_diverging_gpu_is_rejected_by_self_test(monkeypatch):
    fake = fake_cuda.install(monkeypatch)
    monkeypatch.setattr(fake, "blendLinear", lambda i1, i2, w1, w2: i1)   # ignore l'effacement
    a = Accelerator("on")
    assert not a.enabled and "composition" in a.reason


def test_gaussian_filters_are_created_once(monkeypatch):
    fake = fake_cuda.install(monkeypatch)
    created = []
    original = fake.createGaussianFilter
    monkeypatch.setattr(fake, "createGaussianFilter",
                        lambda *a: (created.append(a), original(*a))[1])
    a = Accelerator("auto")
    for _ in range(5):
        a.gaussian(0.8)
    assert len(created) == 1


@pytest.mark.parametrize("sigma,expected", [(0.4, 5), (0.8, 7), (1.5, 13), (10.0, 31)])
def test_gaussian_kernel_size_matches_opencv_cpu(sigma, expected, monkeypatch):
    """Même noyau que GaussianBlur(ksize=(0,0)) en float, plafonné à 31 (CUDA)."""
    fake_cuda.install(monkeypatch)
    assert Accelerator("auto").gaussian(sigma).ksize == (expected, expected)


def test_windows_cuda_dlls_are_registered_before_cv2(tmp_path, monkeypatch):
    """Windows ≥ Python 3.8 : sans add_dll_directory, « DLL load failed »."""
    import livetext
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "x64").mkdir()          # disposition CUDA 13
    registered = []
    monkeypatch.setattr(livetext.sys, "platform", "win32")
    monkeypatch.setenv("CUDA_PATH", str(tmp_path))
    monkeypatch.setattr(livetext.os, "add_dll_directory", registered.append, raising=False)
    livetext._register_cuda_dlls()
    assert registered == [str(tmp_path / "bin"), str(tmp_path / "bin" / "x64")]


def test_no_dll_registration_outside_windows(monkeypatch):
    import livetext
    calls = []
    monkeypatch.setattr(livetext.os, "add_dll_directory", calls.append, raising=False)
    monkeypatch.setenv("CUDA_PATH", "/usr/local/cuda")
    livetext._register_cuda_dlls()   # plateforme réelle : Linux
    assert calls == []
