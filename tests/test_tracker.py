import numpy as np
import pytest

from livetext.config import TrackerConfig
from livetext.sources import SyntheticSource
from livetext.tracker import PlanarTracker, detect_document_quad

from conftest import make_scene


def _corner_error(a, b):
    return float(np.linalg.norm(a - b, axis=1).max())


def test_detect_document_quad_is_subpixel_accurate(scene):
    frame, quad, _ = scene
    detected = detect_document_quad(frame)
    assert detected is not None
    # Régression : sans affinage des côtés, la dilatation des bords
    # décalait les coins de ~6 px vers l'extérieur.
    assert _corner_error(detected, quad) < 1.5


def test_detect_document_quad_returns_none_without_paper():
    rng = np.random.default_rng(0)
    frame = np.clip(rng.normal(80, 10, (360, 640, 3)), 0, 255).astype(np.uint8)
    assert detect_document_quad(frame) is None


def _track(mode, frames=90, size=(960, 540)):
    src = SyntheticSource(*size, seed=3)
    first = src.read()
    tracker = PlanarTracker(TrackerConfig(mode=mode))
    tracker.initialize(first, detect_document_quad(first))
    errors = []
    for _ in range(frames):
        frame = src.read()
        quad = tracker.update(frame)
        errors.append(np.inf if quad is None else _corner_error(quad, src.true_quad))
    return np.array(errors)


@pytest.mark.parametrize("mode,mean_tol", [("hybrid", 3.0), ("flow", 3.0),
                                           ("orb", 4.0), ("contour", 3.0)])
def test_tracker_follows_moving_rotating_sheet(mode, mean_tol):
    errors = _track(mode)
    assert np.isfinite(errors).all(), "suivi perdu"
    assert errors.mean() < mean_tol


def test_orb_does_not_override_healthy_optical_flow():
    """Régression : un contrôle ORB périodique remplaçait un flux optique
    précis (~1,5 px) par sa propre estimation, fausse de ~11 px. C'est
    désormais le score d'alignement ZNCC qui arbitre."""
    errors = _track("hybrid", frames=200)
    assert errors.max() < 6.0


def test_alignment_score_discriminates_misalignment():
    src = SyntheticSource(960, 540, seed=3)
    first = src.read()
    tracker = PlanarTracker()
    tracker.initialize(first, detect_document_quad(first))
    gray = tracker._to_gray(first)
    good = tracker.alignment_score(gray, np.eye(3))
    shifted = tracker.alignment_score(gray, np.array([[1, 0, 12], [0, 1, 0], [0, 0, 1.0]]))
    assert good > 0.95
    assert shifted < 0.5 * good


def test_tracker_relocalizes_after_occlusion():
    src = SyntheticSource(960, 540, seed=3)
    first = src.read()
    tracker = PlanarTracker(TrackerConfig(mode="hybrid"))
    tracker.initialize(first, detect_document_quad(first))
    for _ in range(5):
        tracker.update(src.read())
    black = np.zeros_like(first)
    for _ in range(3):  # la feuille disparaît (main devant la caméra…)
        assert tracker.update(black) is None
    assert tracker.lost_frames == 3 and tracker.quad is None
    for _ in range(5):
        src.read()  # la feuille a bougé pendant l'occultation
    quad = tracker.update(src.read())
    assert quad is not None, "ORB doit relocaliser la feuille"
    assert _corner_error(quad, src.true_quad) < 8.0


def test_update_before_initialize_returns_none(scene):
    assert PlanarTracker().update(scene[0]) is None


def test_initialize_accepts_static_scene(scene):
    frame, quad, _ = scene
    tracker = PlanarTracker()
    tracker.initialize(frame, quad)
    for _ in range(5):
        out = tracker.update(frame)
    assert _corner_error(out, quad) < 0.5


def test_contour_mode_on_textless_sheet():
    frame, quad, _ = make_scene(with_text=False)
    tracker = PlanarTracker(TrackerConfig(mode="contour"))
    tracker.initialize(frame, detect_document_quad(frame))
    assert _corner_error(tracker.update(frame), quad) < 1.5
