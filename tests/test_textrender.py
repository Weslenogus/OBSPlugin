import numpy as np
import pytest

from livetext.config import TextConfig
from livetext.textrender import TextRenderer, load_font, normalize_text, wrap_text


def test_normalize_text_interprets_escaped_newlines():
    assert normalize_text(r"a\nb") == "a\nb"


def test_wrap_text_respects_max_width():
    font = load_font(None, 32)
    lines = wrap_text("le renard brun rapide saute par-dessus le chien", font, 200)
    assert len(lines) > 1
    assert all(font.getlength(line) <= 200 for line in lines)


def test_wrap_text_breaks_overlong_words():
    font = load_font(None, 32)
    lines = wrap_text("x" * 80, font, 150)
    assert "".join(lines) == "x" * 80
    assert all(font.getlength(line) <= 150 for line in lines)


def test_missing_font_file_is_reported():
    with pytest.raises(FileNotFoundError):
        load_font("/nonexistent/font.ttf", 12)


def test_render_places_ink_inside_the_box_only():
    cfg = TextConfig(box=(0.25, 0.25, 0.75, 0.75))
    cov = TextRenderer(cfg).render("Bonjour", 400, 300)
    assert cov.dtype == np.float32 and cov.shape == (300, 400)
    assert 0.0 <= cov.min() and cov.max() <= 1.0
    ys, xs = np.nonzero(cov > 0.05)
    assert xs.size > 0
    assert xs.min() >= 0.25 * 400 - 2 and xs.max() <= 0.75 * 400 + 2
    assert ys.min() >= 0.25 * 300 - 2 and ys.max() <= 0.75 * 300 + 2


def test_auto_fit_shrinks_long_text_into_the_box():
    cfg = TextConfig(box=(0.0, 0.0, 1.0, 0.3), relative_font_size=0.3)
    cov = TextRenderer(cfg).render("un texte beaucoup trop long pour une seule ligne", 400, 300)
    ys, _ = np.nonzero(cov > 0.05)
    assert ys.max() <= 0.3 * 300 + 2


def test_center_alignment_centers_ink():
    cfg = TextConfig(box=(0.0, 0.0, 1.0, 1.0), align="center", valign="middle")
    cov = TextRenderer(cfg).render("I", 400, 300)
    ys, xs = np.nonzero(cov > 0.5)
    assert xs.mean() == pytest.approx(200, abs=15)
    assert ys.mean() == pytest.approx(150, abs=25)


def test_empty_text_renders_nothing():
    assert not TextRenderer(TextConfig()).render("   ", 100, 80).any()
