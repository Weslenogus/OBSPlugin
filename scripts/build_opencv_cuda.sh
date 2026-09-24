#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Compile OpenCV + contrib AVEC CUDA depuis les sources officielles (PyPI),
# sous forme d'une vraie « wheel » pip installée dans l'environnement Python
# courant (venv conseillé). Linux x86_64.
#
# Prérequis : pilote NVIDIA, CUDA Toolkit (nvcc + NPP), gcc/g++, Python ≥ 3.10.
#   Validé : OpenCV 5.0.0.93 + CUDA 13.4 (pilote ≥ 580) ; CUDA 12.9 si pilote plus ancien.
#   Ubuntu 24.04 : sudo apt install cuda-nvcc-13-4 cuda-cudart-dev-13-4 libnpp-dev-13-4
#                  (dépôt NVIDIA), + libgtk-3-dev pour la fenêtre d'aperçu.
#
# Usage : scripts/build_opencv_cuda.sh
# Variables (optionnelles) :
#   CUDA_ARCH=8.9          capacité de calcul du GPU (auto via nvidia-smi)
#   OPENCV_VERSION=...     version PyPI de opencv-contrib-python
#   CUDA_HOME=/usr/local/cuda
#   HEADLESS=1             sans interface graphique (serveur, conteneur)
#   JOBS=N                 compilation parallèle (défaut : nb de cœurs)
#   EXTRA_CMAKE_ARGS="..." options CMake supplémentaires
# ---------------------------------------------------------------------------
set -euo pipefail

OPENCV_VERSION="${OPENCV_VERSION:-5.0.0.93}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
HEADLESS="${HEADLESS:-0}"
JOBS="${JOBS:-$(nproc)}"
WORKDIR="${WORKDIR:-$PWD/build-opencv-cuda}"
EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS:-}"
PYTHON="${PYTHON:-python3}"

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERREUR : %s\033[0m\n' "$*" >&2; exit 1; }

# 1. Vérifications ----------------------------------------------------------
say "Vérifications"
[ -x "$CUDA_HOME/bin/nvcc" ] || die "nvcc introuvable dans $CUDA_HOME/bin (installez le CUDA Toolkit ou fixez CUDA_HOME)."
"$CUDA_HOME/bin/nvcc" --version | tail -1
[ -e "$CUDA_HOME/include/nppcore.h" ] || die "NPP absent ($CUDA_HOME/include/nppcore.h) : installez libnpp-dev (requis par les modules CUDA d'OpenCV)."
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
  || die "Python ≥ 3.10 requis."
if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_PREFIX:-}" ]; then
  echo "Attention : aucun environnement virtuel actif ; la wheel ira dans le Python système."
fi

if [ -z "${CUDA_ARCH:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
  fi
  [ -n "${CUDA_ARCH:-}" ] || die "GPU non détecté : indiquez CUDA_ARCH (RTX 20=7.5, 30=8.6, 40=8.9, 50=12.0)."
fi
echo "Architecture GPU ciblée : $CUDA_ARCH | parallélisme : $JOBS | sans GUI : $HEADLESS"

# 2. Sources officielles (PyPI) --------------------------------------------
say "Téléchargement des sources opencv-contrib-python $OPENCV_VERSION"
mkdir -p "$WORKDIR" && cd "$WORKDIR"
SDIST="opencv_contrib_python-$OPENCV_VERSION.tar.gz"
if [ ! -f "$SDIST" ]; then
  URL="$("$PYTHON" - "$OPENCV_VERSION" <<'PY'
import json, sys, urllib.request
meta = json.load(urllib.request.urlopen(
    f"https://pypi.org/pypi/opencv-contrib-python/{sys.argv[1]}/json"))
print(next(u["url"] for u in meta["urls"] if u["packagetype"] == "sdist"))
PY
)"
  "$PYTHON" -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$URL" "$SDIST"
fi

# 3. Compilation -------------------------------------------------------------
# Modules limités à ce que livetext utilise (+ CUDA) : compilation bien plus
# courte qu'un OpenCV complet. cuBLAS/cuFFT/cuDNN/décodeurs vidéo inutiles ici.
say "Compilation (~15 min sur 4 cœurs, davantage par architecture GPU ajoutée)"
# OpenCV 5 a renommé des modules (features2d → features, calib3d éclaté en
# geometry/calib…). CMake IGNORE SANS ERREUR un nom inconnu : on liste donc
# les noms 4.x ET 5.x, sinon ORB / findHomography manqueraient en silence.
# objdetect/calib/stereo : inutiles à livetext, mais le générateur des
# annotations Python d'OpenCV les exige (sinon pas de py.typed et
# l'empaquetage échoue après une compilation complète).
MODULES="core,imgproc,imgcodecs,videoio,highgui,video,photo,flann,objdetect,python3"
MODULES="$MODULES,features2d,calib3d"          # noms OpenCV 4.x
MODULES="$MODULES,features,geometry,calib,stereo"   # noms OpenCV 5.x
MODULES="$MODULES,cudev,cudaarithm,cudawarping,cudaimgproc,cudafilters"
export CMAKE_ARGS="-DWITH_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME \
  -DCUDA_ARCH_BIN=$CUDA_ARCH -DCUDA_ARCH_PTX= \
  -DWITH_CUBLAS=OFF -DWITH_CUFFT=OFF -DWITH_CUDNN=OFF -DOPENCV_DNN_CUDA=OFF \
  -DWITH_NVCUVID=OFF -DWITH_NVCUVENC=OFF \
  -DBUILD_LIST=$MODULES $EXTRA_CMAKE_ARGS"
export ENABLE_CONTRIB=1              # sans cela : pas de modules CUDA, sans erreur !
export ENABLE_HEADLESS="$HEADLESS"
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" MAKEFLAGS="-j$JOBS"
export PATH="$CUDA_HOME/bin:$PATH"
rm -rf wheelhouse
"$PYTHON" -m pip wheel --no-deps -w wheelhouse "$SDIST" -v 2>&1 | tee build.log | \
  grep --line-buffered -E "^\s*\[[0-9]+/[0-9]+\]|^\s*\[ *[0-9]+%\]|NVIDIA CUDA:|CUDA_ARCH_BIN|error:|Error " || true
WHEEL="$(ls wheelhouse/opencv_contrib_python*.whl 2>/dev/null | head -1 || true)"
[ -n "$WHEEL" ] || die "Échec de compilation : voir $WORKDIR/build.log"

# 4. Installation (remplace tout OpenCV pip existant) -----------------------
say "Installation de $WHEEL"
"$PYTHON" -m pip uninstall -y opencv-python opencv-python-headless \
  opencv-contrib-python opencv-contrib-python-headless >/dev/null 2>&1 || true
"$PYTHON" -m pip install --force-reinstall --no-deps "$WHEEL"

# 5. Vérification ------------------------------------------------------------
say "Vérification"
"$PYTHON" - <<'PY'
import sys
import cv2
info = cv2.getBuildInformation()
cuda_line = next((l.strip() for l in info.splitlines() if l.strip().startswith("NVIDIA CUDA")), "?")
print("OpenCV", cv2.__version__, "|", cuda_line)
# Toutes les fonctions OpenCV dont livetext a besoin (un module manquant
# ne provoque aucune erreur de compilation : on le détecte ici).
needed = ["ORB_create", "BFMatcher", "findHomography", "USAC_MAGSAC", "calcOpticalFlowPyrLK",
          "goodFeaturesToTrack", "inpaint", "warpPerspective", "getPerspectiveTransform",
          "adaptiveThreshold", "connectedComponentsWithStats", "blendLinear", "VideoCapture",
          "VideoWriter", "imshow", "waitKeyEx"]
missing = [n for n in needed if not hasattr(cv2, n)]
cuda_needed = ["warpPerspective", "blendLinear", "addWeighted", "merge", "multiply", "add",
               "createGaussianFilter", "getCudaEnabledDeviceCount"]
missing += ["cuda." + n for n in cuda_needed if not hasattr(getattr(cv2, "cuda", None), n)]
if missing:
    sys.exit("Fonctions OpenCV manquantes : " + ", ".join(missing))
print("Fonctions requises par livetext : toutes présentes")
print("GPU CUDA visibles :", cv2.cuda.getCudaEnabledDeviceCount())
PY
echo "Terminé. Testez ensuite :  python -m livetext.accel"
