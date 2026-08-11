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


class R2CPlan:
    """Reusable real-to-complex / complex-to-real FFT plan.

    ``forward`` writes ``n // 2 + 1`` positive-frequency bins and ``inverse``
    reconstructs a real signal of length ``n``. Supplying the output array
    avoids an allocation on each call.
    """

    _FUNCTIONS = {
        np.dtype(np.float32): (
            "sleipnir_fft_f32_r2c_plan_create",
            "sleipnir_fft_f32_r2c_plan_destroy",
            "sleipnir_fft_f32_r2c_forward",
            "sleipnir_fft_f32_c2r_inverse",
            np.dtype(np.complex64),
        ),
        np.dtype(np.float64): (
            "sleipnir_fft_f64_r2c_plan_create",
            "sleipnir_fft_f64_r2c_plan_destroy",
            "sleipnir_fft_f64_r2c_forward",
            "sleipnir_fft_f64_c2r_inverse",
            np.dtype(np.complex128),
        ),
    }

    def __init__(self, n: int, dtype=np.float64, library=None):
        self.n = int(n)
        self.dtype = np.dtype(dtype)
        if self.n < 2:
            raise ValueError("n must be at least 2")
        if self.dtype not in self._FUNCTIONS:
            raise TypeError("dtype must be numpy.float32 or numpy.float64")
        create_name, destroy_name, forward_name, inverse_name, self.output_dtype = self._FUNCTIONS[self.dtype]
        self.output_size = self.n // 2 + 1
        self._library = ctypes.CDLL(os.fspath(library or _default_library_path()))
        self._destroy = getattr(self._library, destroy_name)
        self._destroy.argtypes = [ctypes.c_void_p]
        self._destroy.restype = None
        self._forward = getattr(self._library, forward_name)
        self._inverse = getattr(self._library, inverse_name)
        for function in (self._forward, self._inverse):
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
            function.restype = ctypes.c_int32
        create = getattr(self._library, create_name)
        create.argtypes = [ctypes.c_int64]
        create.restype = ctypes.c_void_p
        self._handle = create(self.n)
        if not self._handle:
            raise SleipnirFFTError(f"native R2C plan creation failed for n={self.n}")

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "R2CPlan":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self):
        self.close()

    def _check_real(self, data: np.ndarray) -> np.ndarray:
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy.ndarray")
        if data.dtype != self.dtype:
            raise TypeError(f"data dtype must be {self.dtype}")
        if data.ndim != 1 or data.shape[0] != self.n:
            raise ValueError(f"data must be a 1-D array with length {self.n}")
        if not data.flags.c_contiguous:
            raise ValueError("data must be C-contiguous; use np.ascontiguousarray")
        return data

    def _check_complex(self, data: np.ndarray) -> np.ndarray:
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy.ndarray")
        if data.dtype != self.output_dtype:
            raise TypeError(f"data dtype must be {self.output_dtype}")
        if data.ndim != 1 or data.shape[0] != self.output_size:
            raise ValueError(f"data must be a 1-D array with length {self.output_size}")
        if not data.flags.c_contiguous:
            raise ValueError("data must be C-contiguous; use np.ascontiguousarray")
        return data


    @staticmethod
    def _check_output(output: np.ndarray, dtype: np.dtype, size: int) -> np.ndarray:
        if not isinstance(output, np.ndarray):
            raise TypeError("output must be a numpy.ndarray")
        if output.dtype != dtype:
            raise TypeError(f"output dtype must be {dtype}")
        if output.ndim != 1 or output.shape[0] != size:
            raise ValueError(f"output must be a 1-D array with length {size}")
        if not output.flags.c_contiguous:
            raise ValueError("output must be C-contiguous; use np.ascontiguousarray")
        return output

    def forward(self, data: np.ndarray, output: np.ndarray | None = None) -> np.ndarray:
        """Compute the positive-frequency spectrum of a real signal."""
        data = self._check_real(data)
        if output is None:
            output = np.empty(self.output_size, dtype=self.output_dtype)
        else:
            self._check_output(output, self.output_dtype, self.output_size)
        if not self._handle:
            raise SleipnirFFTError("R2C plan is closed")
        error = self._forward(self._handle, data.ctypes.data_as(ctypes.c_void_p), output.ctypes.data_as(ctypes.c_void_p), self.n)
        if error:
            raise SleipnirFFTError(f"native R2C failed with error code {error}")
        return output

    def inverse(self, data: np.ndarray, output: np.ndarray | None = None) -> np.ndarray:
        """Reconstruct a normalized real signal from a positive spectrum."""
        data = self._check_complex(data)
        if output is None:
            output = np.empty(self.n, dtype=self.dtype)
        else:
            self._check_output(output, self.dtype, self.n)
        if not self._handle:
            raise SleipnirFFTError("R2C plan is closed")
        error = self._inverse(self._handle, data.ctypes.data_as(ctypes.c_void_p), output.ctypes.data_as(ctypes.c_void_p), self.n)
        if error:
            raise SleipnirFFTError(f"native C2R failed with error code {error}")
        return output


def rfft(data: np.ndarray, dtype=None, library=None) -> np.ndarray:
    """Return the positive-frequency real-input FFT."""
    array = np.ascontiguousarray(np.asarray(data, dtype=dtype))
    if array.ndim != 1:
        raise ValueError("data must be one-dimensional")
    with R2CPlan(array.size, array.dtype, library=library) as plan:
        return plan.forward(array)


def irfft(data: np.ndarray, n=None, dtype=None, library=None) -> np.ndarray:
    """Return a real signal reconstructed from a positive spectrum."""
    if dtype is not None:
        real_dtype = np.dtype(dtype)
        if real_dtype not in R2CPlan._FUNCTIONS:
            raise TypeError("dtype must be numpy.float32 or numpy.float64")
        complex_dtype = R2CPlan._FUNCTIONS[real_dtype][4]
        array = np.ascontiguousarray(np.asarray(data, dtype=complex_dtype))
    else:
        array = np.ascontiguousarray(np.asarray(data))
        complex_dtype = array.dtype
        if complex_dtype == np.dtype(np.complex64):
            real_dtype = np.dtype(np.float32)
        elif complex_dtype == np.dtype(np.complex128):
            real_dtype = np.dtype(np.float64)
        else:
            raise TypeError("data dtype must be numpy.complex64 or numpy.complex128")
    if array.ndim != 1:
        raise ValueError("data must be one-dimensional")
    length = 2 * (array.size - 1) if n is None else int(n)
    with R2CPlan(length, real_dtype, library=library) as plan:
        return plan.inverse(array)
