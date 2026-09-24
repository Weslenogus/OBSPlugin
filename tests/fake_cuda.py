"""``cv2.cuda`` simulé sur CPU, pour tester ``GpuFrame`` sans GPU.

Volontairement **strict** : il impose les contraintes du vrai OpenCV CUDA
que le CPU tolère (mêmes types pour l'arithmétique, poids float32 mono-canal
pour ``blendLinear``, filtre gaussien au type exact et noyau ≤ 32…). Un appel
qui passerait ici mais échouerait sur GPU serait donc détecté par les tests.
Les signatures reproduisent celles des liaisons Python réelles (vérifiées sur
un OpenCV compilé avec CUDA 12.9).
"""

from __future__ import annotations

import types

import cv2
import numpy as np

_TYPES = {cv2.CV_8UC1: (np.uint8, 1), cv2.CV_8UC3: (np.uint8, 3),
          cv2.CV_32FC1: (np.float32, 1), cv2.CV_32FC3: (np.float32, 3)}


def _kind(a: np.ndarray) -> tuple:
    return a.dtype, (1 if a.ndim == 2 else a.shape[2])


class GpuMat:
    uploads = 0      # compteurs : vérifient qu'une image n'est envoyée qu'une fois
    downloads = 0

    def __init__(self, *args):
        self._a = None
        self._readonly = False
        if len(args) == 2 and isinstance(args[0], GpuMat):   # vue ROI (x, y, w, h)
            parent, (x, y, w, h) = args
            assert x >= 0 and y >= 0 and x + w <= parent._a.shape[1] \
                and y + h <= parent._a.shape[0], "ROI hors de la matrice"
            self._a = parent._a[y:y + h, x:x + w]
            self._readonly = True
        elif args:
            raise TypeError(f"constructeur GpuMat non simulé : {args!r}")

    @classmethod
    def _wrap(cls, array: np.ndarray) -> GpuMat:
        g = cls()
        g._a = array
        return g

    def upload(self, array):
        GpuMat.uploads += 1
        self._a = np.array(array, copy=True)

    def download(self):
        GpuMat.downloads += 1
        return self._a.copy()

    def size(self):
        h, w = self._a.shape[:2]
        return (w, h)

    def convertTo(self, rtype):
        dtype, channels = _TYPES[rtype]
        assert _kind(self._a)[1] == channels, "convertTo ne change pas le nb de canaux"
        a = self._a
        if dtype == np.uint8:
            a = np.clip(np.rint(a), 0, 255)          # saturate_cast (arrondi)
        return GpuMat._wrap(a.astype(dtype))


def _check_same(*mats):
    for m in mats:
        if not isinstance(m, GpuMat):   # comme le vrai : aucune surcharge ne correspond
            raise cv2.error(f"Overload resolution failed: {type(m).__name__} n'est pas un GpuMat")
    kinds = {(_kind(m._a), m._a.shape[:2]) for m in mats}
    assert len(kinds) == 1, f"CUDA : types/tailles différents {kinds}"


def warpPerspective(src, M, dsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT):
    assert isinstance(dsize, tuple) and len(dsize) == 2
    return GpuMat._wrap(cv2.warpPerspective(src._a, M, dsize, flags=flags,
                                            borderMode=borderMode))


def blendLinear(img1, img2, weights1, weights2):
    _check_same(img1, img2)
    _check_same(weights1, weights2)
    assert _kind(weights1._a) == (np.dtype(np.float32), 1), "poids float32 mono-canal"
    assert weights1._a.shape == img1._a.shape[:2]
    return GpuMat._wrap(cv2.blendLinear(img1._a, img2._a, weights1._a, weights2._a))


def addWeighted(src1, alpha, src2, beta, gamma):
    _check_same(src1, src2)
    return GpuMat._wrap(cv2.addWeighted(src1._a, alpha, src2._a, beta, gamma))


def merge(mats, dst=None):
    """Piège réel des liaisons Python : les 3 surcharges de ``cv2.cuda.merge``
    ne diffèrent que par ``dst``. Sans ``dst``, la *première* (sortie
    ``MatLike``) l'emporte : le résultat revient en **tableau numpy** sur le
    CPU. Il faut passer ``dst=cv2.cuda_GpuMat()`` pour rester sur le GPU."""
    _check_same(*mats)
    assert _kind(mats[0]._a)[1] == 1
    merged = cv2.merge([m._a for m in mats])
    return merged if dst is None else GpuMat._wrap(merged)


def multiply(src1, src2, scale=1.0):
    _check_same(src1, src2)
    return GpuMat._wrap(cv2.multiply(src1._a, src2._a, scale=scale))


def add(src1, src2):
    _check_same(src1, src2)
    return GpuMat._wrap(cv2.add(src1._a, src2._a))


class _GaussianFilter:
    def __init__(self, src_type, ksize, sigma):
        self.src_type, self.ksize, self.sigma = src_type, ksize, sigma

    def apply(self, src):
        dtype, channels = _TYPES[self.src_type]
        assert _kind(src._a) == (np.dtype(dtype), channels), "filtre créé pour un autre type"
        return GpuMat._wrap(cv2.GaussianBlur(src._a, self.ksize, self.sigma))


def createGaussianFilter(srcType, dstType, ksize, sigma1):
    assert srcType == dstType
    assert max(ksize) <= 32 and all(k % 2 for k in ksize), "noyau CUDA : impair, ≤ 32"
    return _GaussianFilter(srcType, ksize, sigma1)


def install(monkeypatch, devices: int = 1):
    """Remplace ``cv2.cuda`` / ``cv2.cuda_GpuMat`` par ce simulateur."""
    fake = types.SimpleNamespace(
        getCudaEnabledDeviceCount=lambda: devices, getDevice=lambda: 0,
        printShortCudaDeviceInfo=lambda dev: None, warpPerspective=warpPerspective,
        blendLinear=blendLinear, addWeighted=addWeighted, merge=merge,
        multiply=multiply, add=add, createGaussianFilter=createGaussianFilter)
    monkeypatch.setattr(cv2, "cuda", fake, raising=False)
    monkeypatch.setattr(cv2, "cuda_GpuMat", GpuMat, raising=False)
    GpuMat.uploads = GpuMat.downloads = 0
    return fake
