import cv2
import numpy as np
import pytest

from livetext.config import EraseConfig
from livetext.inpaint import TextEraser

PAPER = 215


def _canvas_with_text(seed=0):
    rng = np.random.default_rng(seed)
    canvas = np.clip(rng.normal(PAPER, 2.5, (300, 400, 3)), 0, 255).astype(np.uint8)
    text = canvas.copy()
    cv2.putText(text, "Texte", (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 3, (30, 30, 30), 8,
                cv2.LINE_AA)
    strokes = cv2.absdiff(text, canvas).max(axis=2) > 40
    return text, strokes


def _apply(result, canvas):
    """Recompose le canevas nettoyé comme le fait le pipeline."""
    out = canvas.astype(np.float32).copy()
    x0, y0 = result.origin
    h, w = result.clean.shape[:2]
    m = result.soft_mask[..., None]
    out[y0:y0 + h, x0:x0 + w] = out[y0:y0 + h, x0:x0 + w] * (1 - m) + result.clean * m
    return out


@pytest.mark.parametrize("method,algo", [("plate", "telea"), ("plate", "ns"),
                                         ("inpaint", "telea"), ("inpaint", "ns"),
                                         ("median", "telea")])
def test_erase_removes_strokes_without_flat_patch(method, algo):
    canvas, strokes = _canvas_with_text()
    result = TextEraser(EraseConfig(method=method, inpaint_algo=algo,
                                    temporal_decay=0)).erase(canvas)
    assert not result.empty
    assert result.text_mask[strokes].mean() > 250  # tous les traits détectés
    out = _apply(result, canvas)
    # Anciennes positions de l'encre : revenues à la teinte du papier.
    assert out[strokes].mean() == pytest.approx(PAPER, abs=6)
    assert out.min() > PAPER - 40
    if method != "inpaint":
        # Grain réinjecté : pas de zone parfaitement lisse (« rectangle uni »).
        assert out[strokes].std() > 1.0


def test_erase_on_blank_paper_is_a_noop():
    rng = np.random.default_rng(1)
    canvas = np.clip(rng.normal(PAPER, 2.5, (200, 300, 3)), 0, 255).astype(np.uint8)
    assert TextEraser(EraseConfig(temporal_decay=0)).erase(canvas).empty


def test_erase_method_none_returns_empty():
    canvas, _ = _canvas_with_text()
    assert TextEraser(EraseConfig(method="none")).erase(canvas).empty


@pytest.mark.parametrize("feather", [7, 0])
def test_erase_region_limits_detection(feather):
    """Régression : la dilatation des traits débordait de la zone demandée,
    et avec ``feather=0`` le masque adouci n'était plus rogné du tout."""
    canvas, strokes = _canvas_with_text()
    cfg = EraseConfig(region=(0.0, 0.0, 0.3, 1.0), feather=feather, temporal_decay=0)
    result = TextEraser(cfg).erase(canvas)
    limit = int(0.3 * 400)
    assert not result.text_mask[:, limit:].any()
    assert result.text_mask[:, :limit].any()
    # Le masque de mélange (ce qui modifie réellement l'image) est rogné aussi.
    x0 = result.origin[0]
    assert not result.soft_mask[:, max(0, limit - x0):].any()


def test_temporal_memory_bridges_a_missed_detection():
    """Le masque persiste une image si la détection « clignote »."""
    canvas, strokes = _canvas_with_text()
    blank = np.full_like(canvas, PAPER)
    eraser = TextEraser(EraseConfig(temporal_decay=0.85))
    eraser.erase(canvas)
    assert eraser.erase(blank).text_mask[strokes].mean() > 250
    eraser.reset()
    assert eraser.erase(blank).empty
