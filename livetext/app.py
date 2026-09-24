"""Application temps réel et interface en ligne de commande.

Boucle principale : capture → suivi → effacement / rendu / déformation /
intégration → caméra virtuelle (OBS) + aperçu. Le texte change à chaud via
le terminal ou le bandeau de la fenêtre d'aperçu, sans jamais bloquer le flux.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import __version__
from .config import RESOLUTIONS, AppConfig
from .controls import HELP, Command, StdinReader, TextController
from .geometry import Quad, is_valid_quad, order_quad
from .pipeline import TextReplacementPipeline
from .sinks import WINDOW_NAME, FrameSink, WindowSink, open_sinks
from .sources import FramePacer, FrameSource, open_source
from .textrender import find_default_font, find_system_font, load_font
from .tracker import PlanarTracker, detect_document_quad

log = logging.getLogger("livetext")

# Après ~1,5 s de suivi perdu en mode auto, on redétecte la feuille.
_REACQUIRE_AFTER = 45
# En attente de cible, on tente la détection toutes les N images.
_DETECT_EVERY = 5


@dataclass
class RunStats:
    """Statistiques d'exécution (affichées à la fin)."""

    frames: int = 0
    tracked: int = 0
    process_ms: list[float] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)

    @property
    def fps(self) -> float:
        elapsed = time.perf_counter() - self.started
        return self.frames / elapsed if elapsed > 0 else 0.0

    def summary(self) -> str:
        ms = np.array(self.process_ms or [0.0])
        return (f"{self.frames} images, {self.fps:.1f} i/s, traitement moyen "
                f"{ms.mean():.1f} ms (p95 {np.percentile(ms, 95):.1f} ms), "
                f"feuille suivie sur {100 * self.tracked / max(1, self.frames):.0f} % "
                f"des images")


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def parse_quad(spec: str) -> Quad:
    """``"x1,y1,x2,y2,x3,y3,x4,y4"`` → quadrilatère ordonné."""
    values = [float(v) for v in spec.replace(";", ",").split(",")]
    if len(values) != 8:
        raise ValueError("--init attend 8 nombres : x1,y1,x2,y2,x3,y3,x4,y4")
    return order_quad(np.array(values, np.float32).reshape(4, 2))


def put_text(img: np.ndarray, text: str, org: tuple[int, int], size: int,
             color=(255, 255, 255), background=(0, 0, 0), alpha: float = 0.6,
             font_path: str | None = None) -> None:
    """Écrit du texte Unicode (accents) sur ``img`` sur un fond translucide.

    ``cv2.putText`` ne gère que l'ASCII ; on rend donc un petit patch avec
    Pillow puis on le fusionne, sans convertir l'image entière.
    """
    font = load_font(font_path, size)
    x0, y0, x1, y1 = font.getbbox(text or " ")
    pad = size // 3
    w, h = x1 + 2 * pad, y1 + 2 * pad
    x, y = org
    ih, iw = img.shape[:2]
    w, h = min(w, iw - x), min(h, ih - y)
    if w <= 0 or h <= 0:
        return
    patch = Image.new("L", (w, h), 0)
    ImageDraw.Draw(patch).text((pad, pad), text, font=font, fill=255)
    mask = np.asarray(patch, np.float32)[..., None] / 255.0
    region = img[y:y + h, x:x + w].astype(np.float32)
    region = region * (1 - alpha) + np.asarray(background, np.float32) * alpha
    region = region * (1 - mask) + np.asarray(color, np.float32) * mask
    img[y:y + h, x:x + w] = region.astype(np.uint8)


class LiveTextApp:
    """Assemble source, suivi, pipeline, sorties et contrôles."""

    def __init__(self, config: AppConfig, source: FrameSource | None = None,
                 sinks: list[FrameSink] | None = None, seed: int | None = None):
        self.config = config
        width, height = config.frame_size
        self.source = source or open_source(config.source, width, height,
                                            config.fps, seed=seed or 0)
        self.sinks = sinks if sinks is not None else open_sinks(
            config.outputs, width, height, config.fps, config.virtualcam_backend)
        self.window = next((s for s in self.sinks if isinstance(s, WindowSink)), None)
        self.tracker = PlanarTracker(config.tracker, fps=config.fps)
        self.pipeline = TextReplacementPipeline(config, seed=seed)
        self.controller = TextController(config.text.initial_text)
        self.stats = RunStats()
        self._fixed_quad = None if config.init in ("auto", "manual") else parse_quad(config.init)
        self._running = False
        self._ms_avg = 0.0
        # Police de l'interface : police système lisible (pas la police.ttf du
        # texte incrusté, parfois décorative), sinon celle du texte.
        self._ui_font = (find_system_font() or config.text.font_path
                         or find_default_font())

    def _read(self) -> np.ndarray | None:
        """Lit une image, en miroir si demandé : *tous* les traitements (suivi,
        sélection manuelle, sorties) voient ainsi exactement la même image."""
        frame = self.source.read()
        if frame is not None and self.config.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def _put_text(self, img: np.ndarray, text: str, org: tuple[int, int], size: int,
                  **kwargs) -> None:
        put_text(img, text, org, size, font_path=self._ui_font, **kwargs)

    # -- cible -----------------------------------------------------------------

    def set_target(self, frame: np.ndarray, quad: Quad) -> None:
        self.tracker.initialize(frame, quad)
        self.pipeline.set_target(quad)
        log.info("Feuille verrouillée : %s", np.round(quad).astype(int).tolist())

    def _try_acquire(self, frame: np.ndarray) -> Quad | None:
        """Initialise le suivi (auto ou coordonnées fixes) si possible."""
        if self._fixed_quad is not None:
            quad = self._fixed_quad
        elif self.config.init == "auto":
            quad = detect_document_quad(frame)
        else:
            return None  # mode manuel : attend la sélection à la souris
        if quad is None or not is_valid_quad(quad, frame.shape):
            return None
        self.set_target(frame, quad)
        return quad

    def select_manual(self, frame: np.ndarray) -> Quad | None:
        """Sélection des 4 coins à la souris sur une image figée.

        Pendant la sélection, les autres sorties (OBS) continuent de recevoir
        le flux brut : le direct ne se fige jamais.
        """
        if self.window is None:
            log.error("La sélection manuelle nécessite la sortie « window ».")
            return None
        points: list[tuple[int, int]] = []

        def on_mouse(event, x, y, *_):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((x, y))

        cv2.setMouseCallback(WINDOW_NAME, on_mouse)
        others = [s for s in self.sinks if s is not self.window]
        try:
            while True:
                view = frame.copy()
                for p in points:
                    cv2.circle(view, p, 6, (0, 255, 0), -1, cv2.LINE_AA)
                if len(points) >= 2:
                    cv2.polylines(view, [np.array(points, np.int32)], len(points) == 4,
                                  (0, 255, 0), 2, cv2.LINE_AA)
                msg = ("Cliquez les 4 coins de la feuille" if len(points) < 4
                       else "Entrée = valider, c = recommencer")
                self._put_text(view, f"{msg} ({len(points)}/4) - Échap = annuler",
                               (10, 10), 22)
                self.window.send(view)
                live = self._read()
                if live is None:
                    return None
                for sink in others:
                    sink.send(live)
                key = WindowSink.poll_key()
                if key in (27,):
                    return None
                if key in (ord("c"), ord("C")):
                    points.clear()
                if key in (10, 13) and len(points) == 4:
                    quad = order_quad(np.array(points, np.float32))
                    if is_valid_quad(quad, frame.shape):
                        return quad
                    log.warning("Quadrilatère invalide (non convexe ?), recommencez.")
                    points.clear()
        finally:
            cv2.setMouseCallback(WINDOW_NAME, lambda *a: None)

    # -- commandes -------------------------------------------------------------

    def execute(self, command: Command, frame: np.ndarray) -> None:
        name, arg = command.name, command.arg
        if name in ("quit", "q", "exit"):
            self._running = False
        elif name == "reset":
            self.tracker.reset()
            if self.config.init == "manual":
                self.config.init = "auto"
            log.info("Redétection de la feuille…")
        elif name == "select":
            quad = self.select_manual(frame)
            if quad is not None:
                self.set_target(frame, quad)
        elif name == "debug":
            self.config.show_debug = not self.config.show_debug
        elif name == "erase" and arg in ("plate", "inpaint", "median", "none"):
            self.config.erase.method = arg
            self.pipeline.eraser.reset()
        elif name == "align" and arg in ("left", "center", "right"):
            self.config.text.align = arg
            self.pipeline.invalidate_text()
        elif name == "size" and arg.isdigit():
            # Taille explicite : respectée telle quelle ; 0 = retour à l'auto.
            self.config.text.font_size = int(arg)
            self.config.text.auto_fit = int(arg) == 0
        elif name == "fontsize" and _is_number(arg):
            size = self.pipeline.adjust_font_size(int(float(arg)))
            log.info("Taille de police : %d pt", size)
        elif name == "nudge" and len(arg.split()) == 2 and all(map(_is_number, arg.split())):
            self.pipeline.nudge_text(*(float(v) for v in arg.split()))
        elif name == "recenter":
            self.pipeline.reset_offset()
        elif name in ("tracking", "weight") and _is_number(arg):
            setattr(self.config.text, name, float(arg))
            log.info("%s : %s pt", "Interlettrage" if name == "tracking" else "Graisse", arg)
        else:
            log.warning("Commande inconnue ou argument invalide : /%s %s (voir /help)",
                        name, arg)

    # -- aperçu ----------------------------------------------------------------

    def _preview(self, out: np.ndarray, quad: Quad | None) -> np.ndarray:
        """Image de la fenêtre : sortie + diagnostic + bandeau de saisie."""
        view = out.copy()
        h = view.shape[0]
        size = max(16, h // 40)
        if self.config.show_debug:
            if quad is not None:
                cv2.polylines(view, [np.round(quad).astype(np.int32)], True,
                              (0, 255, 0), 2, cv2.LINE_AA)
            dbg, t = self.pipeline.debug, self.config.text
            ink = ("gris (pas d'encre visible)" if dbg.ink_bgr is None else
                   "BGR " + ",".join(f"{v:.0f}" for v in dbg.ink_bgr))
            ox, oy = self.pipeline.text_offset
            self._put_text(view, f"{self.stats.fps:4.1f} i/s | {self._ms_avg:4.1f} ms | "
                                 f"suivi : {self.tracker.status} | L papier "
                                 f"{dbg.paper_luminance:.0f} | encre {ink}", (10, 10), size)
            self._put_text(view, f"taille {self.pipeline.renderer.last_size} pt | "
                                 f"interlettrage {t.tracking:+.2f} | graisse {t.weight:+.2f} | "
                                 f"décalage ({ox:+.1f}, {oy:+.1f}) | effacement : "
                                 f"{self.config.erase.method}", (10, 10 + 2 * size), size)
        if quad is None:
            hint = ("Recherche de la feuille…  (s = sélection manuelle)"
                    if self.config.init != "manual" or self.tracker.initialized
                    else "Appuyez sur s pour sélectionner la feuille")
            self._put_text(view, hint, (10, 10 + 4 * size if self.config.show_debug else 10),
                           size, background=(0, 0, 160))
        if self.controller.editing:
            self._put_text(view, f"Texte : {self.controller.buffer}▌   (Entrée = valider, "
                                 f"Échap = annuler)", (10, h - 2 * size - 10), size,
                           background=(40, 40, 40), alpha=0.8)
        return view

    # -- boucle ----------------------------------------------------------------

    def step(self, frame: np.ndarray) -> tuple[np.ndarray, Quad | None]:
        """Traite une image : commandes en attente, suivi, remplacement du texte."""
        for command in self.controller.pop_commands():
            self.execute(command, frame)

        quad = None
        if self.tracker.initialized:
            quad = self.tracker.update(frame)
            if (quad is None and self.config.init == "auto"
                    and self.tracker.lost_frames >= _REACQUIRE_AFTER):
                self.tracker.reset()
        if not self.tracker.initialized and self.stats.frames % _DETECT_EVERY == 0:
            quad = self._try_acquire(frame)

        self.pipeline.set_text(self.controller.display_text)
        return self.pipeline.process(frame, quad), quad

    def run(self, max_frames: int | None = None) -> RunStats:
        """Boucle temps réel jusqu'à ``/quit``, ``q``, Ctrl-C ou fin de source."""
        pacer = None if self.source.live or not any(
            s.name in ("virtualcam", "window") for s in self.sinks) else FramePacer(self.config.fps)
        if self.config.stdin_input:
            StdinReader(self.controller).start()
            print(HELP, flush=True)
        if self.config.init == "manual" and not self.tracker.initialized:
            self.controller.push(Command("select"))
        self._running = True
        self.stats = RunStats()
        try:
            while self._running:
                frame = self._read()
                if frame is None:
                    break
                t0 = time.perf_counter()
                out, quad = self.step(frame)
                elapsed = (time.perf_counter() - t0) * 1000
                self.stats.process_ms.append(elapsed)
                self._ms_avg = 0.9 * self._ms_avg + 0.1 * elapsed if self._ms_avg else elapsed
                self.stats.frames += 1
                self.stats.tracked += quad is not None

                for sink in list(self.sinks):
                    try:
                        sink.send(self._preview(out, quad) if sink is self.window else out)
                    except RuntimeError as exc:
                        # Ex. caméra virtuelle arrêtée côté OBS : on continue le
                        # direct sur les autres sorties plutôt que de tout couper.
                        log.error("%s — sortie retirée.", exc)
                        self.sinks.remove(sink)
                        self._close_sink(sink)
                if not self.sinks:
                    log.error("Plus aucune sortie active.")
                    break
                if self.window is not None:
                    command = self.controller.handle_key(WindowSink.poll_key())
                    if command is not None:
                        # Même file que le terminal : exécutée par step() sur
                        # l'image suivante, préparée comme toutes les autres.
                        self.controller.push(command)
                if max_frames is not None and self.stats.frames >= max_frames:
                    break
                if pacer is not None:
                    pacer.wait()
        except KeyboardInterrupt:
            log.info("Interrompu (Ctrl-C).")
        finally:
            self.close()
        log.info("Terminé : %s", self.stats.summary())
        return self.stats

    def close(self) -> None:
        self.source.release()
        for sink in self.sinks:
            self._close_sink(sink)

    @staticmethod
    def _close_sink(sink: FrameSink) -> None:
        try:
            sink.close()
        except Exception:  # noqa: BLE001 - fermeture au mieux
            log.exception("Fermeture de la sortie %s", sink.name)


# ---------------------------------------------------------------------------
# Ligne de commande
# ---------------------------------------------------------------------------

def _box(value: str) -> tuple[float, float, float, float]:
    parts = [float(v) for v in value.split(",")]
    if len(parts) != 4 or not all(0.0 <= p <= 1.0 for p in parts):
        raise argparse.ArgumentTypeError("attendu : x0,y0,x1,y1 entre 0 et 1")
    return tuple(parts)  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="livetext",
        description="Remplacement de texte en direct sur une feuille suivie, "
                    "envoyé vers OBS par caméra virtuelle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    io = p.add_argument_group("entrée / sortie")
    io.add_argument("--source", default="0",
                    help="index de webcam, fichier vidéo/image, ou 'synthetic'")
    io.add_argument("--resolution", choices=list(RESOLUTIONS), default="720p")
    io.add_argument("--fps", type=int, default=30)
    io.add_argument("--output", action="append", dest="outputs",
                    help="virtualcam | window | file:<chemin> | none (répétable ; "
                         "défaut : virtualcam + window)")
    io.add_argument("--backend", dest="virtualcam_backend",
                    help="backend pyvirtualcam (obs, v4l2loopback, unitycapture)")
    io.add_argument("--mirror", action="store_true", help="effet miroir horizontal")
    io.add_argument("--max-frames", type=int, help="arrête après N images (tests)")

    tgt = p.add_argument_group("cible et suivi")
    tgt.add_argument("--init", default="auto",
                     help="auto | manual | x1,y1,x2,y2,x3,y3,x4,y4 (pixels)")
    tgt.add_argument("--tracker", choices=("hybrid", "flow", "orb", "contour"),
                     default="hybrid")

    txt = p.add_argument_group("texte")
    txt.add_argument("--text", help="texte initial (\\n = saut de ligne)")
    txt.add_argument("--font", dest="font", help="police vectorielle (défaut : police.ttf)")
    txt.add_argument("--font-size", type=int, default=None,
                     help="taille en points (0 = auto ; défaut : FONT_SIZE_PT)")
    txt.add_argument("--tracking", type=float, default=None,
                     help="interlettrage en points (défaut : TRACKING)")
    txt.add_argument("--weight", type=float, default=None,
                     help="graisse en points, négatif = plus maigre (défaut : WEIGHT)")
    txt.add_argument("--text-box", type=_box, help="boîte du texte x0,y0,x1,y1 (0-1)")
    txt.add_argument("--align", choices=("left", "center", "right"), default="left")
    txt.add_argument("--valign", choices=("top", "middle", "bottom"), default="top")

    fx = p.add_argument_group("effacement et photométrie")
    fx.add_argument("--erase", choices=("plate", "inpaint", "median", "none"),
                    default="plate")
    fx.add_argument("--erase-region", type=_box, help="zone à effacer x0,y0,x1,y1 (0-1)")
    fx.add_argument("--no-ink-sampling", action="store_true",
                    help="ne pas échantillonner la couleur de l'encre réelle")
    fx.add_argument("--ink-ratio", type=float, default=0.22,
                    help="repli sans encre visible : luminance encre / papier")
    fx.add_argument("--blur", type=float, default=0.8, help="sigma du flou de l'encre")
    fx.add_argument("--noise", type=float, default=2.5, help="sigma du grain ajouté")

    misc = p.add_argument_group("divers")
    misc.add_argument("--debug", action="store_true", help="diagnostic dans l'aperçu")
    misc.add_argument("--no-stdin", action="store_true",
                      help="ne pas lire le texte depuis le terminal")
    misc.add_argument("--seed", type=int, help="graine aléatoire (reproductibilité)")
    misc.add_argument("-v", "--verbose", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> AppConfig:
    cfg = AppConfig(source=args.source, resolution=args.resolution, fps=args.fps,
                    virtualcam_backend=args.virtualcam_backend, mirror=args.mirror,
                    init=args.init, show_debug=args.debug,
                    stdin_input=not args.no_stdin)
    if args.outputs:
        cfg.outputs = args.outputs
    cfg.tracker.mode = args.tracker
    if args.text is not None:
        cfg.text.initial_text = args.text
    cfg.text.font_path = args.font
    # None = on garde les variables globales de config.py.
    if args.font_size is not None:
        cfg.text.font_size = args.font_size
    if args.tracking is not None:
        cfg.text.tracking = args.tracking
    if args.weight is not None:
        cfg.text.weight = args.weight
    cfg.text.align, cfg.text.valign = args.align, args.valign
    if args.text_box:
        cfg.text.box = args.text_box
    cfg.erase.method = args.erase
    if args.erase_region:
        cfg.erase.region = args.erase_region
    cfg.photometry.sample_ink = not args.no_ink_sampling
    cfg.photometry.ink_ratio = args.ink_ratio
    cfg.photometry.blur_sigma = args.blur
    cfg.photometry.noise_sigma = args.noise
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    try:
        config = config_from_args(args)
        if config.init not in ("auto", "manual"):
            parse_quad(config.init)  # valide tôt
        app = LiveTextApp(config, seed=args.seed)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 1
    app.run(max_frames=args.max_frames)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
