import numpy as np
import pytest

from livetext.app import LiveTextApp, build_parser, config_from_args, main, parse_quad
from livetext.config import AppConfig
from livetext.controls import Command
from livetext.sinks import FrameSink
from livetext.sources import SyntheticSource


class CaptureSink(FrameSink):
    name = "capture"

    def __init__(self):
        self.frames = []

    def send(self, frame):
        self.frames.append(frame)


def _app(**overrides):
    cfg = AppConfig(stdin_input=False, **overrides)
    sink = CaptureSink()
    app = LiveTextApp(cfg, source=SyntheticSource(960, 540, seed=3), sinks=[sink], seed=0)
    return app, sink


def test_headless_run_tracks_every_frame_and_outputs_clean_frames():
    app, sink = _app()
    stats = app.run(max_frames=30)
    assert stats.frames == 30 and len(sink.frames) == 30
    assert stats.tracked == 30
    assert all(f.shape == (540, 960, 3) and f.dtype == np.uint8 for f in sink.frames)


def test_text_command_updates_overlay_without_interrupting_stream():
    app, sink = _app()
    app.controller.set_text("Alpha")
    frame = app.source.read()
    app.step(frame)
    assert app.pipeline.text == "Alpha"
    app.controller.handle_line("Beta\n")
    app.step(app.source.read())
    assert app.pipeline.text == "Beta"


def test_quit_command_stops_the_loop():
    app, sink = _app()
    app.controller.push(Command("quit"))
    stats = app.run(max_frames=100)
    assert stats.frames == 1


def test_runtime_commands_change_settings():
    app, _ = _app()
    frame = app.source.read()
    for name, arg in [("erase", "median"), ("align", "center"), ("size", "40"), ("debug", "")]:
        app.execute(Command(name, arg), frame)
    assert app.config.erase.method == "median"
    assert app.config.text.align == "center"
    assert app.config.text.font_size == 40
    assert app.config.show_debug


def test_fixed_coordinates_initialisation():
    src = SyntheticSource(960, 540, seed=3)
    src.read()
    quad = src.quad_at(1)
    spec = ",".join(f"{v:.1f}" for v in quad.ravel())
    app, _ = _app(init=spec)
    app.run(max_frames=5)
    assert app.tracker.initialized


def test_mirror_is_applied_once_at_read_time():
    """Régression : en miroir, les commandes clavier de la fenêtre recevaient
    l'image non retournée alors que le suivi travaillait sur l'image
    retournée (sélection manuelle décalée)."""
    app, sink = _app(mirror=True)
    reference = SyntheticSource(960, 540, seed=3)
    np.testing.assert_array_equal(app._read(), reference.read()[:, ::-1])
    stats = app.run(max_frames=10)
    assert stats.tracked == 10  # le suivi fonctionne sur l'image retournée


def test_window_keys_go_through_the_command_queue():
    app, _ = _app()
    command = app.controller.handle_key(ord("d"))
    app.controller.push(command)
    assert not app.config.show_debug
    app.step(app._read())
    assert app.config.show_debug


class FailingSink(CaptureSink):
    name = "virtualcam"

    def send(self, frame):
        raise RuntimeError("caméra virtuelle arrêtée")


def test_failing_sink_is_dropped_and_stream_continues():
    cfg = AppConfig(stdin_input=False)
    good, bad = CaptureSink(), FailingSink()
    app = LiveTextApp(cfg, source=SyntheticSource(960, 540, seed=3),
                      sinks=[bad, good], seed=0)
    stats = app.run(max_frames=5)
    assert stats.frames == 5 and len(good.frames) == 5
    assert bad not in app.sinks


def test_parse_quad_orders_points():
    q = parse_quad("500,80, 100,50, 90,380, 480,400")
    np.testing.assert_array_equal(q, [[100, 50], [500, 80], [480, 400], [90, 380]])
    with pytest.raises(ValueError):
        parse_quad("1,2,3")


def test_cli_builds_config():
    args = build_parser().parse_args([
        "--source", "synthetic", "--resolution", "1080p", "--output", "none",
        "--text", "Salut", "--erase", "inpaint", "--tracker", "orb",
        "--text-box", "0.1,0.2,0.9,0.8", "--ink-ratio", "0.3", "--no-stdin",
    ])
    cfg = config_from_args(args)
    assert cfg.frame_size == (1920, 1080)
    assert cfg.outputs == ["none"] and cfg.text.initial_text == "Salut"
    assert cfg.erase.method == "inpaint" and cfg.tracker.mode == "orb"
    assert cfg.text.box == (0.1, 0.2, 0.9, 0.8)
    assert cfg.photometry.ink_ratio == 0.3 and not cfg.stdin_input


def test_cli_end_to_end_writes_video(tmp_path):
    out = tmp_path / "out.avi"
    code = main(["--source", "synthetic", "--output", f"file:{out}",
                 "--max-frames", "10", "--no-stdin", "--seed", "1"])
    assert code == 0 and out.stat().st_size > 0


def test_cli_rejects_unavailable_outputs():
    # opencv-headless : pas de fenêtre ; pyvirtualcam absent → aucune sortie.
    code = main(["--source", "synthetic", "--output", "file:/nonexistent/dir/x.avi",
                 "--no-stdin", "--max-frames", "1"])
    assert code == 1


def test_live_layout_commands_through_the_app():
    app, _ = _app()
    app.step(app._read())
    auto = app.pipeline.renderer.last_size
    for line in ("/tracking 2", "/weight 0.75", "/nudge 1 0"):
        app.controller.handle_line(line)
    app.controller.push(app.controller.handle_key(ord("+")))
    app.step(app._read())
    t = app.config.text
    assert (t.tracking, t.weight, t.font_size) == (2.0, 0.75, auto + 1)
    assert app.pipeline.text_offset[0] > 0
    app.controller.push(app.controller.handle_key(ord("0")))
    app.step(app._read())
    assert not app.pipeline.text_offset.any()


def test_cli_font_settings_default_to_the_globals():
    from livetext import config as globals_
    cfg = config_from_args(build_parser().parse_args(["--output", "none"]))
    assert (cfg.text.font_size, cfg.text.tracking, cfg.text.weight) == (
        globals_.FONT_SIZE_PT, globals_.TRACKING, globals_.WEIGHT)
    cfg = config_from_args(build_parser().parse_args(
        ["--output", "none", "--font-size", "30", "--tracking", "-0.5", "--weight", "1",
         "--no-ink-sampling"]))
    assert (cfg.text.font_size, cfg.text.tracking, cfg.text.weight) == (30, -0.5, 1.0)
    assert not cfg.photometry.sample_ink
