# livetext — texte remplacé en direct sur une feuille, pour OBS

`livetext` capture une webcam USB, suit une feuille de papier dans l'image,
**efface son texte d'origine** et y **incruste un nouveau texte** qui épouse la
perspective, l'éclairage et le grain de la vidéo. Le flux modifié est envoyé à
**OBS** par une caméra virtuelle (`pyvirtualcam`), et le texte se change à
chaud depuis le terminal ou la fenêtre d'aperçu, sans jamais couper le flux.

```
webcam ─► suivi du plan ─► effacement ─► rendu Pillow (.ttf) ─► warpPerspective ─► fusion Produit ─► OBS
 (thread)   LK + ORB + ZNCC   inpaint      texte en direct       homographie          luminosité, flou, grain
```

## Installation

Python ≥ 3.10.

```bash
pip install -r requirements.txt        # ou : pip install -e .[dev]
```

### Caméra virtuelle (sortie vers OBS)

| Système | Préparation |
|---|---|
| **Windows** | Installer OBS ≥ 26. `pyvirtualcam` écrit dans le périphérique « OBS Virtual Camera » : **ne cliquez pas** sur « Démarrer la caméra virtuelle » dans OBS. |
| **macOS** | Installer OBS ≥ 30, cliquer une fois sur « Démarrer la caméra virtuelle » (installe l'extension système), puis l'arrêter. |
| **Linux** | `sudo apt install v4l2loopback-dkms` puis `sudo modprobe v4l2loopback devices=1 exclusive_caps=1 card_label=livetext` |

Dans OBS : **Sources → + → Périphérique de capture vidéo** et choisissez la
caméra virtuelle (« OBS Virtual Camera » ou « livetext » sous Linux).

## Utilisation

```bash
# Webcam 0 en 720p/30 i/s → OBS + fenêtre d'aperçu, avec votre police
python main.py --font polices/MaPolice.ttf --text "Bonjour le direct !"

# 1080p, texte centré, diagnostic affiché
python main.py --resolution 1080p --align center --valign middle --debug

# Sélection manuelle des 4 coins à la souris (feuille peu contrastée)
python main.py --init manual

# Démo sans caméra ni OBS : scène synthétique animée, enregistrée dans un fichier
python main.py --source synthetic --output file:demo.mp4 --max-frames 300 --no-stdin
```

`python main.py --help` liste toutes les options (aussi : `python -m livetext`,
ou la commande `livetext` après `pip install -e .`).

### Contrôle en direct

Deux canaux, utilisables en même temps, qui n'interrompent jamais la vidéo :

**Terminal** (lu par un thread dédié) — tapez une ligne puis Entrée :

| Saisie | Effet |
|---|---|
| `Nouveau texte` | remplace le texte incrusté (`\n` = saut de ligne) |
| `/clear` | efface sans rien écrire |
| `/reset` | relance la détection automatique de la feuille |
| `/select` | sélection manuelle des 4 coins (fenêtre) |
| `/erase plate\|inpaint\|median\|none` | méthode d'effacement |
| `/align left\|center\|right` · `/size <px>` | mise en page à chaud |
| `/debug` · `/help` · `/quit` | diagnostic, aide, quitter |
| `//texte` | texte commençant par `/` |

**Fenêtre d'aperçu** (bandeau d'entrée) — `t` ou Entrée ouvre la saisie ; le
texte incrusté **suit chaque frappe** ; Entrée valide, Échap annule.
Hors saisie : `r` redétecter, `s` sélection manuelle, `d` diagnostic, `q`
quitter (Échap ne quitte volontairement pas, pour ne pas couper un direct).
Les caractères non latins se saisissent plus sûrement par le terminal.

## Fonctionnement

### 1. Entrée et sortie vidéo — `sources.py`, `sinks.py`
- `cv2.VideoCapture` en 1080p ou 720p à 30 i/s, format **MJPG** (en YUYV brut,
  beaucoup de webcams USB 2.0 plafonnent à 5-10 i/s en 1080p). La capture tourne
  dans un **thread** qui ne garde que la dernière image : latence minimale.
- Sortie `pyvirtualcam` au format BGR natif (aucune conversion par image), dans
  un thread : la conversion interne vers OBS se fait en parallèle du calcul.
  Si la caméra virtuelle tombe, elle est retirée et le reste continue.

### 2. Traitement du texte — `tracker.py`, `inpaint.py`, `textrender.py`, `compositor.py`

**Suivi de plan** (`--tracker`, défaut `hybrid`) :
- *Initialisation* : détection du polygone 4 points (Canny + Otsu), puis
  **affinage sous-pixel** de chaque côté (droite ajustée sur le maximum du
  gradient) qui sert aussi de vérification (un faux candidat est rejeté).
- *Suivi* : `cv2.calcOpticalFlowPyrLK` avec contrôle aller-retour. Chaque point
  garde sa position dans l'image de **référence** : l'homographie est recalculée
  (RANSAC) référence → image courante à chaque image, sans composer d'erreurs.
  Les points sont semés en retrait du bord (une fenêtre LK à cheval sur la
  feuille et le fond « glisse »).
- *Relocalisation* : appariement **ORB** + `USAC_MAGSAC`. Il n'est lancé que si le
  flux optique échoue ou s'aligne mal, et **arbitré par un score ZNCC**
  (corrélation passe-haut avec la référence rectifiée) : l'estimation qui
  explique le mieux l'image l'emporte.
- Modes : `hybrid`, `flow` (LK seul), `orb` (ORB à chaque image),
  `contour` (redétection du polygone à chaque image, utile sur feuille vierge).

**Effacement** (`--erase`) dans le canevas rectifié (feuille « à plat ») :
seuillage adaptatif → masque des traits (mémoire temporelle anti-scintillement),
puis reconstruction : `plate` (`cv2.inpaint` à basse résolution, conserve
ombres et plis), `inpaint` (pleine résolution), `median` (teinte médiane du
papier, masque flouté). Le grain du papier est mesuré et **réinjecté** : pas de
rectangle uni.

**Rendu** : Pillow `ImageFont.truetype` sur votre `.ttf`, retour à la ligne et
ajustement automatique de la taille dans la boîte `--text-box`.

**Déformation** : `cv2.getPerspectiveTransform` canevas → coins suivis puis
`cv2.warpPerspective`, limité au rectangle englobant du texte (le coût suit la
surface du texte, pas celle de la feuille), avec préfiltre anti-crénelage quand
la feuille s'éloigne.

**Photométrie** :
- fusion **Produit** : `résultat = papier × calque` — plis, ombres et grain restent
  visibles sous l'encre ;
- luminosité moyenne du papier mesurée **autour** de la zone (hors encre
  d'origine, moyenne tronquée, lissée dans le temps) → luminance de l'encre
  `= L_papier × --ink-ratio`, bornée pour ne jamais devenir un noir « collé » ;
- léger flou gaussien (`--blur`) et bruit gaussien (`--noise`) pour égaler la
  défocalisation et le grain du capteur.

### 3. Contrôle en direct — `controls.py`
Thread démon de lecture du terminal + bandeau `cv2.waitKeyEx(1)` non bloquant ;
les deux alimentent la même file de commandes, exécutée par la boucle vidéo.

## Performances et précision

Mesurées dans un conteneur Linux à 4 vCPU, sans GPU (scène synthétique :
feuille en rotation ±8°, échelle ±8 %, perspective, éclairage mobile) :

| | 720p | 1080p |
|---|---|---|
| Suivi (`hybrid`) | ~4,5 ms | ~4 ms |
| Effacement + rendu + intégration | ~15 ms | ~24 ms |
| Erreur de coin (moy. / max, 200 images) | 1,3 / 1,7 px | 1,8 / 2,4 px |

Détection initiale : 0,6-0,7 px d'erreur moyenne sur 30 poses, sans échec.

## Tests

```bash
pip install pytest
python -m pytest
```

71 tests, sans caméra ni OBS (scène synthétique) : géométrie, rendu, fusion
Produit, effacement, précision du suivi, contrôles, application de bout en bout.

## Limites et conseils

- Éclairage stable et feuille bien contrastée sur le fond : la détection
  automatique n'en est que plus fiable (sinon `--init manual`).
- Une feuille **vierge** n'offre pas de texture à suivre : préférez
  `--tracker contour`.
- Les pliures franches peuvent être prises pour du texte et lissées ; réduisez
  la zone avec `--erase-region`.
- La feuille est supposée plane (homographie) : une page très courbée ne sera
  suivie qu'approximativement.

## Usage responsable

Cet outil est destiné à la création vidéo, au direct et à l'enseignement de la
vision par ordinateur. N'utilisez pas l'incrustation pour faire passer un
document modifié pour authentique (vérification d'identité, preuve, contrat…) ;
signalez à votre public que l'image est retouchée.
