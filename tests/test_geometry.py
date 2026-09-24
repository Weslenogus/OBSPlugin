import numpy as np
import pytest

from livetext.geometry import (
    canvas_size_for_quad,
    canvas_to_quad,
    is_valid_quad,
    order_quad,
    quad_roi,
    transform_quad,
    warp_canvas_patch,
)

QUAD = np.float32([[100, 50], [500, 80], [480, 400], [90, 380]])


def test_order_quad_is_invariant_to_input_order():
    rng = np.random.default_rng(0)
    for _ in range(10):
        shuffled = QUAD[rng.permutation(4)]
        np.testing.assert_array_equal(order_quad(shuffled), QUAD)


def test_canvas_to_quad_maps_canvas_corners_onto_quad():
    H = canvas_to_quad(300, 200, QUAD)
    corners = np.float32([[0, 0], [299, 0], [299, 199], [0, 199]])
    np.testing.assert_allclose(transform_quad(corners, H), QUAD, atol=1e-3)


def test_canvas_size_keeps_aspect_ratio_and_bound():
    w, h = canvas_size_for_quad(QUAD, max_side=200)
    assert max(w, h) == 200
    assert w / h == pytest.approx(395 / 325, rel=0.05)


@pytest.mark.parametrize("quad", [
    None,
    np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]),                 # trop petit
    np.float32([[100, 50], [480, 400], [500, 80], [90, 380]]),     # croisé
    np.float32([[np.nan, 0], [500, 80], [480, 400], [90, 380]]),   # non fini
])
def test_is_valid_quad_rejects_bad_quads(quad):
    assert not is_valid_quad(quad, (480, 640, 3))


def test_is_valid_quad_accepts_normal_quad():
    assert is_valid_quad(QUAD, (480, 640, 3))


def test_quad_roi_clamps_and_rejects_offscreen():
    assert quad_roi(QUAD, (480, 640, 3), pad=0) == (90, 50, 501, 401)
    assert quad_roi(QUAD + 5000, (480, 640, 3)) is None


def test_warp_canvas_patch_places_patch_in_the_right_frame_box():
    cw, ch = 400, 300
    H = canvas_to_quad(cw, ch, QUAD)
    patch = np.ones((50, 80), np.float32)
    warped, (x0, y0, x1, y1) = warp_canvas_patch(patch, (100, 120), H, (480, 640))
    # Le centre du patch dans le canevas doit tomber sur un pixel non nul.
    center = transform_quad(np.float32([[140, 145]] * 4), H)[0]
    cx, cy = int(center[0]) - x0, int(center[1]) - y0
    assert warped[cy, cx] == pytest.approx(1.0)
    assert warped.shape == (y1 - y0, x1 - x0)
