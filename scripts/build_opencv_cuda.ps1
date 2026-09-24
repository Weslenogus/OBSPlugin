# ---------------------------------------------------------------------------
# Compile OpenCV + contrib AVEC CUDA depuis les sources officielles (PyPI),
# sous forme d'une « wheel » pip installée dans l'environnement Python courant.
# Windows 10/11 x64, PowerShell.
#
# Prérequis (dans cet ordre) :
#   1. Pilote NVIDIA récent.
#   2. Visual Studio 2022 Build Tools, charge de travail « Développement
#      Desktop en C++ » (MSVC + SDK Windows).
#   3. CUDA Toolkit 13.4 (pilote ≥ 580 ; sinon 12.9) installé APRÈS Visual
#      Studio, pour que son intégration MSBuild soit ajoutée.
#   4. Python ≥ 3.10 (venv conseillé), CMake est installé automatiquement.
#
# Usage (depuis le dossier du projet, venv activé) :
#   powershell -ExecutionPolicy Bypass -File scripts\build_opencv_cuda.ps1
# Options : -CudaArch 8.9  -OpenCvVersion 5.0.0.93  -Jobs 8  -ExtraCmakeArgs "..."
# ---------------------------------------------------------------------------
param(
    [string]$CudaArch = "",
    [string]$OpenCvVersion = "5.0.0.93",
    [int]$Jobs = [Environment]::ProcessorCount,
    [string]$WorkDir = "$PWD\build-opencv-cuda",
    [string]$ExtraCmakeArgs = "",
    [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
function Say($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Die($m) { Write-Host "`nERREUR : $m" -ForegroundColor Red; exit 1 }

# 1. Vérifications ----------------------------------------------------------
Say "Vérifications"
if (-not $env:CUDA_PATH) { Die "CUDA_PATH non défini : installez le CUDA Toolkit (13.4 conseillé)." }
$nvcc = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
if (-not (Test-Path $nvcc)) { Die "nvcc introuvable : $nvcc" }
& $nvcc --version | Select-Object -Last 1
if (-not (Test-Path (Join-Path $env:CUDA_PATH "include\nppcore.h"))) {
    Die "NPP absent du CUDA Toolkit (composant « npp ») : requis par les modules CUDA d'OpenCV."
}
& $Python -c "import sys; assert sys.version_info >= (3, 10), sys.version"
if ($LASTEXITCODE -ne 0) { Die "Python >= 3.10 requis." }
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere) -or -not (& $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)) {
    Die "Visual Studio 2022 (Build Tools) avec « Développement Desktop en C++ » introuvable."
}
if (-not $env:VIRTUAL_ENV -and -not $env:CONDA_PREFIX) {
    Write-Host "Attention : aucun environnement virtuel actif." -ForegroundColor Yellow
}
if (-not $CudaArch) {
    try { $CudaArch = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader | Select-Object -First 1).Trim() } catch {}
    if (-not $CudaArch) { Die "GPU non détecté : passez -CudaArch (RTX 20=7.5, 30=8.6, 40=8.9, 50=12.0)." }
}
Write-Host "Architecture GPU : $CudaArch | parallélisme : $Jobs"

# 2. Sources officielles (PyPI) --------------------------------------------
Say "Téléchargement des sources opencv-contrib-python $OpenCvVersion"
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
Set-Location $WorkDir
$sdist = "opencv_contrib_python-$OpenCvVersion.tar.gz"
if (-not (Test-Path $sdist)) {
    $meta = Invoke-RestMethod "https://pypi.org/pypi/opencv-contrib-python/$OpenCvVersion/json"
    $url = ($meta.urls | Where-Object { $_.packagetype -eq "sdist" } | Select-Object -First 1).url
    Invoke-WebRequest -Uri $url -OutFile $sdist
}

# 3. Compilation -------------------------------------------------------------
Say "Compilation (15 à 45 min selon la machine)"
# OpenCV 5 a renommé des modules (features2d -> features, calib3d éclaté en
# geometry/calib...). CMake IGNORE SANS ERREUR un nom inconnu : on liste les
# noms 4.x ET 5.x, sinon ORB / findHomography manqueraient en silence.
# objdetect/calib/stereo : exigés par le générateur des annotations Python
# (sinon pas de py.typed et l'empaquetage échoue après la compilation).
$modules = "core,imgproc,imgcodecs,videoio,highgui,video,photo,flann,objdetect,python3," +
           "features2d,calib3d," +                 # noms OpenCV 4.x
           "features,geometry,calib,stereo," +     # noms OpenCV 5.x
           "cudev,cudaarithm,cudawarping,cudaimgproc,cudafilters"
$cudaRoot = $env:CUDA_PATH -replace '\\', '/'
$env:CMAKE_ARGS = "-DWITH_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=`"$cudaRoot`" " +
    "-DCUDA_ARCH_BIN=$CudaArch -DCUDA_ARCH_PTX= " +
    "-DWITH_CUBLAS=OFF -DWITH_CUFFT=OFF -DWITH_CUDNN=OFF -DOPENCV_DNN_CUDA=OFF " +
    "-DWITH_NVCUVID=OFF -DWITH_NVCUVENC=OFF -DBUILD_LIST=$modules $ExtraCmakeArgs"
$env:ENABLE_CONTRIB = "1"      # sans cela : pas de modules CUDA, sans erreur !
$env:ENABLE_HEADLESS = "0"     # garde la fenêtre d'aperçu (Win32)
$env:CMAKE_BUILD_PARALLEL_LEVEL = "$Jobs"
if (Test-Path wheelhouse) { Remove-Item -Recurse -Force wheelhouse }
& $Python -m pip wheel --no-deps -w wheelhouse $sdist -v *>&1 | Tee-Object -FilePath build.log |
    Select-String -Pattern '^\s*\[\d+/\d+\]|^\s*\[ *\d+%\]|NVIDIA CUDA:|error' | ForEach-Object { $_.Line }
$wheel = Get-ChildItem wheelhouse\opencv_contrib_python*.whl -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $wheel) { Die "Échec de compilation : voir $WorkDir\build.log" }

# 4. Installation (remplace tout OpenCV pip existant) -----------------------
Say "Installation de $($wheel.Name)"
& $Python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless 2>$null
& $Python -m pip install --force-reinstall --no-deps $wheel.FullName

# 5. Vérification ------------------------------------------------------------
# Depuis Python 3.8, Windows ne cherche plus les DLL dans le PATH : les DLL
# CUDA (NPP) doivent être déclarées, sinon « DLL load failed » à l'import.
# livetext le fait automatiquement ; on le fait ici aussi pour la vérification.
Say "Vérification"
$check = @'
import os
for sub in ("bin", "bin/x64"):
    d = os.path.join(os.environ["CUDA_PATH"], sub)
    if os.path.isdir(d):
        os.add_dll_directory(d)
import cv2
info = cv2.getBuildInformation()
print("OpenCV", cv2.__version__, "|", next((l.strip() for l in info.splitlines()
      if l.strip().startswith("NVIDIA CUDA")), "?"))
needed = ["ORB_create", "BFMatcher", "findHomography", "USAC_MAGSAC", "calcOpticalFlowPyrLK",
          "goodFeaturesToTrack", "inpaint", "warpPerspective", "getPerspectiveTransform",
          "adaptiveThreshold", "connectedComponentsWithStats", "blendLinear", "VideoCapture",
          "VideoWriter", "imshow", "waitKeyEx"]
missing = [n for n in needed if not hasattr(cv2, n)]
cuda_needed = ["warpPerspective", "blendLinear", "addWeighted", "merge", "multiply", "add",
               "createGaussianFilter", "getCudaEnabledDeviceCount"]
missing += ["cuda." + n for n in cuda_needed if not hasattr(getattr(cv2, "cuda", None), n)]
if missing:
    raise SystemExit("Fonctions OpenCV manquantes : " + ", ".join(missing))
print("Fonctions requises par livetext : toutes présentes")
print("GPU CUDA visibles :", cv2.cuda.getCudaEnabledDeviceCount())
'@
& $Python -c $check
if ($LASTEXITCODE -ne 0) { Die "cv2 compilé mais non utilisable : voir le message ci-dessus." }
Write-Host "`nTerminé. Testez ensuite :  python -m livetext.accel" -ForegroundColor Green
