"""Contrôle en direct : saisie du texte sans interrompre le flux vidéo.

Deux canaux, utilisables en même temps :

* **Terminal** — un thread démon lit ``stdin`` ligne par ligne. Chaque ligne
  devient le nouveau texte (Entrée), ou une commande si elle commence par
  ``/``. La lecture bloquante vit dans son propre thread : la boucle vidéo
  n'attend jamais le clavier.
* **Bandeau d'entrée** dans la fenêtre d'aperçu — touches captées par
  ``cv2.waitKeyEx(1)`` (non bloquant). Le texte incrusté suit la frappe
  caractère par caractère ; Entrée valide, Échap annule.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
from dataclasses import dataclass

from .config import FONT_STEP_PT, NUDGE_STEP_PX, TRACK_STEP_PT

log = logging.getLogger(__name__)

HELP = """\
Commandes (terminal) :
  <texte> + Entrée   remplace le texte incrusté (\\n = saut de ligne)
  /clear             efface sans rien écrire
  /reset             relance la détection automatique de la feuille
  /select            sélection manuelle des 4 coins (fenêtre)
  /erase <méthode>   plate | inpaint | median | none
  /align <valeur>    left | center | right
  /size <pt>         taille de police (0 = automatique)
  /tracking <pt>     interlettrage (négatif = resserré)
  /weight <pt>       graisse (négatif = plus maigre)
  /nudge <dx> <dy>   décale le texte (pixels écran, ex. /nudge 0.5 0)
  /recenter          annule le décalage
  /debug             affiche/masque le diagnostic
  /quit              quitte
  //texte            texte commençant par « / »
Fenêtre d'aperçu : t ou Entrée = saisir, Échap = annuler, r = redétecter,
                   s = sélection manuelle, d = diagnostic, q = quitter,
                   flèches = décaler le texte d'un demi-pixel,
                   + / - = taille de police, [ / ] = interlettrage,
                   0 = recentrer."""

# Codes renvoyés par cv2.waitKeyEx selon les plateformes.
_ENTER = {10, 13, 65421}
_ESC = {27}
_BACKSPACE = {8, 127, 65288}
# Flèches → direction (dx, dy). Codes selon l'interface d'OpenCV :
# GTK/X11, Windows, macOS (Cocoa) et Qt.
_ARROWS = {
    65361: (-1, 0), 65362: (0, -1), 65363: (1, 0), 65364: (0, 1),
    2424832: (-1, 0), 2490368: (0, -1), 2555904: (1, 0), 2621440: (0, 1),
    63234: (-1, 0), 63232: (0, -1), 63235: (1, 0), 63233: (0, 1),
    16777234: (-1, 0), 16777235: (0, -1), 16777236: (1, 0), 16777237: (0, 1),
}
# « = » compte comme « + » (même touche sans Maj) ; pavé numérique (GTK).
_PLUS = {ord("+"), ord("="), 65451}
_MINUS = {ord("-"), 65453}


@dataclass(frozen=True)
class Command:
    """Commande à exécuter par la boucle principale."""

    name: str
    arg: str = ""


class TextController:
    """État partagé (thread-safe) : texte courant, saisie en cours, commandes."""

    def __init__(self, initial_text: str):
        self._lock = threading.Lock()
        self._text = initial_text
        self._commands: queue.SimpleQueue[Command] = queue.SimpleQueue()
        self.editing = False
        self.buffer = ""

    # -- texte ----------------------------------------------------------------

    @property
    def text(self) -> str:
        with self._lock:
            return self._text

    def set_text(self, text: str) -> None:
        with self._lock:
            self._text = text

    @property
    def display_text(self) -> str:
        """Texte à incruster : la saisie en cours, sinon le texte validé."""
        return self.buffer if self.editing else self.text

    # -- commandes -------------------------------------------------------------

    def push(self, command: Command) -> None:
        self._commands.put(command)

    def pop_commands(self) -> list[Command]:
        commands = []
        while True:
            try:
                commands.append(self._commands.get_nowait())
            except queue.Empty:
                return commands

    def handle_line(self, line: str) -> None:
        """Interprète une ligne tapée dans le terminal."""
        line = line.rstrip("\r\n")
        if line.startswith("//"):
            self.set_text(line[1:])
        elif line.startswith("/"):
            name, _, arg = line[1:].partition(" ")
            name = name.strip().lower()
            if name == "clear":
                self.set_text("")
            elif name in ("help", "h", "?"):
                print(HELP, flush=True)
            elif name:
                self.push(Command(name, arg.strip()))
            return
        elif line.strip():
            self.set_text(line)
        else:
            return  # ligne vide ignorée (évite d'effacer par mégarde)
        print(f"✓ Texte incrusté : {self.text!r}", flush=True)

    # -- bandeau (fenêtre) -----------------------------------------------------

    def handle_key(self, key: int) -> Command | None:
        """Traite une touche de la fenêtre ; renvoie une commande éventuelle."""
        if key < 0:
            return None
        if key in _ARROWS:  # actif même pendant la saisie (non imprimable)
            dx, dy = _ARROWS[key]
            return Command("nudge", f"{dx * NUDGE_STEP_PX} {dy * NUDGE_STEP_PX}")
        if self.editing:
            if key in _ENTER:
                self.set_text(self.buffer)
                self.editing = False
            elif key in _ESC:
                self.editing = False  # retour au texte validé
            elif key in _BACKSPACE:
                self.buffer = self.buffer[:-1]
            else:
                char = _printable(key)
                if char:
                    self.buffer += char
            return None

        char = _printable(key)
        if key in _ENTER or char == "t":
            self.editing = True
            self.buffer = self.text
        elif char == "q":
            # Échap ne quitte volontairement pas : un double Échap pour annuler
            # une saisie ne doit pas couper un direct.
            return Command("quit")
        elif char == "r":
            return Command("reset")
        elif char == "s":
            return Command("select")
        elif char == "d":
            return Command("debug")
        elif key in _PLUS:
            return Command("fontsize", f"+{FONT_STEP_PT}")
        elif key in _MINUS:
            return Command("fontsize", f"-{FONT_STEP_PT}")
        elif char == "[":
            return Command("tracking_delta", f"-{TRACK_STEP_PT}")
        elif char == "]":
            return Command("tracking_delta", f"+{TRACK_STEP_PT}")
        elif char == "0":
            return Command("recenter")
        return None


def _printable(key: int) -> str:
    """Caractère imprimable correspondant au code, sinon chaîne vide."""
    code = key & 0xFFFF if key > 0xFFFF else key
    if 32 <= code <= 126 or 160 <= code <= 0x2FFF:
        return chr(code)
    return ""


class StdinReader:
    """Lit le terminal dans un thread démon et alimente le contrôleur."""

    def __init__(self, controller: TextController, stream=None):
        self._controller = controller
        self._stream = stream or sys.stdin
        self._thread = threading.Thread(target=self._run, name="stdin", daemon=True)

    def start(self) -> StdinReader:
        self._thread.start()
        return self

    def _run(self) -> None:
        while True:
            try:
                line = self._stream.readline()
            except (OSError, ValueError):  # flux fermé
                return
            if not line:  # EOF (entrée redirigée épuisée, Ctrl-D)
                return
            try:
                self._controller.handle_line(line)
            except Exception:  # noqa: BLE001 - ne jamais tuer le lecteur
                log.exception("Erreur de traitement de la ligne %r", line)
