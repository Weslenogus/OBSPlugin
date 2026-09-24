"""Garde-fous sur les scripts de compilation d'OpenCV avec CUDA.

CMake ignore *sans erreur* un module inconnu de ``BUILD_LIST`` : une liste
erronée donne un OpenCV qui compile mais auquel il manque ORB ou
findHomography (arrivé avec OpenCV 5, qui a renommé features2d → features
et éclaté calib3d). Ces tests figent la liste des modules requis.
"""

import re
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

# Modules dont livetext a besoin, sous leurs noms OpenCV 4.x et 5.x.
REQUIRED = {
    "core", "imgproc", "imgcodecs", "videoio", "highgui", "video", "photo", "flann",
    "python3",
    "features2d", "calib3d",      # OpenCV 4.x : ORB, findHomography
    "features", "geometry",       # OpenCV 5.x : mêmes fonctions, modules renommés
    # Exigés par le générateur d'annotations Python d'OpenCV (py.typed) :
    # sans eux, l'empaquetage échoue après une compilation complète.
    "objdetect", "calib", "stereo",
    "cudev", "cudaarithm", "cudawarping", "cudaimgproc", "cudafilters",
}


def _modules_sh() -> set[str]:
    text = (SCRIPTS / "build_opencv_cuda.sh").read_text()
    parts = re.findall(r'^MODULES="(?:\$MODULES,)?([^"]+)"', text, re.M)
    return {m for part in parts for m in part.split(",")}


def _modules_ps1() -> set[str]:
    text = (SCRIPTS / "build_opencv_cuda.ps1").read_text()
    block = text[text.index("$modules ="):text.index("$cudaRoot")]
    parts = re.findall(r'"([^"]+)"', block)
    return {m for part in parts for m in part.strip(",").split(",") if m}


@pytest.mark.parametrize("reader", [_modules_sh, _modules_ps1])
def test_build_list_contains_every_required_module(reader):
    missing = REQUIRED - reader()
    assert not missing, f"modules manquants : {sorted(missing)}"


def test_linux_and_windows_build_the_same_modules():
    assert _modules_sh() == _modules_ps1()


@pytest.mark.parametrize("name", ["build_opencv_cuda.sh", "build_opencv_cuda.ps1"])
def test_scripts_enable_contrib_and_verify_livetext_apis(name):
    text = (SCRIPTS / name).read_text()
    assert re.search(r"ENABLE_CONTRIB\s*=\s*\"?1", text), "sans contrib : aucun module CUDA"
    for api in ("ORB_create", "findHomography", "calcOpticalFlowPyrLK", "inpaint",
                "createGaussianFilter"):
        assert api in text, f"la vérification finale doit contrôler {api}"
