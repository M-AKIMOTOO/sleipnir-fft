"""NumPy interface for the Sleipnir FFT shared library.

The FFT operates in-place on a contiguous NumPy array.  The array is not
copied by the transform call, so callers can use the result without an
additional allocation.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np


class SleipnirFFTError(RuntimeError):
    """Raised when the native FFT API rejects an operation."""


def _default_library_path() -> Path:
    return Path(__file__).resolve().parents[1] / "build" / "libsleipnir_fft.so"


class C2CPlan:
    """Reusable in-place complex-to-complex FFT plan.

    Parameters
    ----------
    n:
        Transform length.  It must be a positive power of two supported by
        the native library.
    dtype:
        ``numpy.complex64`` or ``numpy.complex128``.
    library:
        Optional path to ``libsleipnir_fft.so``.
    """

    _FUNCTIONS = {
        np.dtype(np.complex64): (
            "sleipnir_fft_f32_plan_create",
            "sleipnir_fft_f32_plan_destroy",
            "sleipnir_fft_f32_forward",
            "sleipnir_fft_f32_inverse",
        ),
        np.dtype(np.complex128): (
            "sleipnir_fft_f64_plan_create",
            "sleipnir_fft_f64_plan_destroy",
            "sleipnir_fft_f64_forward",
            "sleipnir_fft_f64_inverse",
        ),
    }

    def __init__(self, n: int, dtype=np.complex128, library=None):
        self.n = int(n)
        self.dtype = np.dtype(dtype)
        if self.n <= 0:
            raise ValueError("n must be positive")
        if self.dtype not in self._FUNCTIONS:
            raise TypeError("dtype must be numpy.complex64 or numpy.complex128")

        self._library = ctypes.CDLL(os.fspath(library or _default_library_path()))
        create_name, destroy_name, forward_name, inverse_name = self._FUNCTIONS[
            self.dtype
        ]
        self._destroy = getattr(self._library, destroy_name)
        self._destroy.argtypes = [ctypes.c_void_p]
        self._destroy.restype = None
        self._forward = getattr(self._library, forward_name)
        self._inverse = getattr(self._library, inverse_name)
        for function in (self._forward, self._inverse):
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
            function.restype = ctypes.c_int32

        create = getattr(self._library, create_name)
        create.argtypes = [ctypes.c_int64]
        create.restype = ctypes.c_void_p
        self._handle = create(self.n)
        if not self._handle:
            raise SleipnirFFTError(f"native plan creation failed for n={self.n}")

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "C2CPlan":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self):
        self.close()

    def _transform(self, data: np.ndarray, function) -> np.ndarray:
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy.ndarray")
        if data.dtype != self.dtype:
            raise TypeError(f"data dtype must be {self.dtype}")
        if data.ndim != 1 or data.shape[0] != self.n:
            raise ValueError(f"data must be a 1-D array with length {self.n}")
        if not data.flags.c_contiguous:
            raise ValueError("data must be C-contiguous; use np.ascontiguousarray")
        if not self._handle:
            raise SleipnirFFTError("FFT plan is closed")
        error = function(self._handle, data.ctypes.data_as(ctypes.c_void_p), self.n)
        if error:
            raise SleipnirFFTError(f"native FFT failed with error code {error}")
        return data

    def forward(self, data: np.ndarray) -> np.ndarray:
        """Perform an unnormalized forward transform in-place."""

        return self._transform(data, self._forward)

    def inverse(self, data: np.ndarray) -> np.ndarray:
        """Perform the normalized inverse transform in-place."""

        return self._transform(data, self._inverse)


def fft(data: np.ndarray, dtype=None, library=None) -> np.ndarray:
    """Return a forward FFT, copying the input once into native layout."""

    array = np.array(data, dtype=dtype, copy=True, order="C")
    if array.ndim != 1:
        raise ValueError("data must be one-dimensional")
    array = np.ascontiguousarray(array)
    with C2CPlan(array.size, array.dtype, library=library) as plan:
        return plan.forward(array)


def ifft(data: np.ndarray, dtype=None, library=None) -> np.ndarray:
    """Return an inverse FFT, copying the input once into native layout."""

    array = np.array(data, dtype=dtype, copy=True, order="C")
    if array.ndim != 1:
        raise ValueError("data must be one-dimensional")
    array = np.ascontiguousarray(array)
    with C2CPlan(array.size, array.dtype, library=library) as plan:
        return plan.inverse(array)
