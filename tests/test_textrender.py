import numpy as np
import pytest

import shutil

from livetext import textrender
from livetext.config import FONT_FILE, TextConfig
from livetext.textrender import (
    TextRenderer,
    find_system_font,
    line_width,
    load_font,
    normalize_text,
    wrap_text,
)


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


def test_local_police_ttf_is_used_by_default(tmp_path, monkeypatch):
    """« police.ttf » dans le dossier courant l'emporte sur la police système."""
    local = tmp_path / FONT_FILE
    shutil.copy(find_system_font(), local)
    monkeypatch.chdir(tmp_path)
    assert textrender.local_font_path() == str(local)
    assert TextRenderer(TextConfig()).font_path == str(local)


def test_explicit_font_overrides_police_ttf(tmp_path, monkeypatch):
    shutil.copy(find_system_font(), tmp_path / FONT_FILE)
    monkeypatch.chdir(tmp_path)
    other = str(tmp_path / "autre.ttf")
    shutil.copy(find_system_font(), other)
    assert TextRenderer(TextConfig(font_path=other)).font_path == other


def _ink_extent(cov):
    xs = np.nonzero(cov.max(axis=0) > 0.05)[0]
    return xs.max() - xs.min()


def test_tracking_spaces_letters_without_changing_glyphs():
    base = TextRenderer(TextConfig(auto_fit=False)).render("AVATAR", 600, 200)
    wide = TextRenderer(TextConfig(auto_fit=False, tracking=6.0)).render("AVATAR", 600, 200)
    assert _ink_extent(wide) == pytest.approx(_ink_extent(base) + 5 * 6.0, abs=3)
    assert wide.sum() == pytest.approx(base.sum(), rel=0.02)  # mêmes glyphes


def test_tracking_is_counted_when_wrapping():
    font = load_font(None, 32)
    assert line_width(font, "abcd", 5.0) == pytest.approx(font.getlength("abcd") + 15.0)
    lines = wrap_text("mot mot mot mot mot mot", font, 200, tracking=8.0)
    assert all(line_width(font, line, 8.0) <= 200 for line in lines)


@pytest.mark.parametrize("weight", [-0.75, 0.5, 1.5])
def test_weight_thickens_or_thins_strokes(weight):
    base = TextRenderer(TextConfig(auto_fit=False)).render("Graisse", 600, 200).sum()
    other = TextRenderer(TextConfig(auto_fit=False, weight=weight)).render("Graisse", 600,
                                                                            200).sum()
    assert (other > base * 1.05) if weight > 0 else (other < base * 0.95)


def test_last_size_reports_the_size_actually_used():
    r = TextRenderer(TextConfig(font_size=40, auto_fit=False))
    r.render("Taille", 600, 200)
    assert r.last_size == 40
