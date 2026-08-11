"""Compare NumPy and Sleipnir FFT speed and save a spectrum plot.

Run from the repository root after building the shared library::

    python python/benchmark_spectrum.py --dtype complex64

The benchmark measures the transform itself. Sleipnir's reusable plan is
created once and its plan creation time is reported separately. The input
restore before each Sleipnir call is outside the timed region.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sleipnirfft import C2CPlan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=65536, help="FFT length")
    parser.add_argument("--repeats", type=int, default=30, help="timed iterations")
    parser.add_argument("--warmup", type=int, default=5, help="untimed iterations")
    parser.add_argument(
        "--dtype",
        choices=("complex64", "complex128"),
        default="complex128",
        help="complex precision used by Sleipnir",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/fft_spectrum_comparison.png"),
        help="output PNG path",
    )
    parser.add_argument("--show", action="store_true", help="also display the plot")
    return parser.parse_args()


def make_signal(n: int, dtype: np.dtype) -> np.ndarray:
    sample_rate = 48000.0
    time_axis = np.arange(n, dtype=np.float64) / sample_rate
    frequencies = sample_rate * np.array((n // 64, n // 16, n // 4 - 1)) / n
    amplitudes = (1.0, 0.45, 0.2)
    signal = sum(
        amplitude * np.sin(2.0 * np.pi * frequency * time_axis)
        for amplitude, frequency in zip(amplitudes, frequencies)
    )
    noise = np.random.default_rng(1234).normal(0.0, 0.01, n)
    return (signal + noise).astype(dtype)


def measure_numpy(
    signal: np.ndarray, warmup: int, repeats: int
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(warmup):
        np.fft.fft(signal)

    times = np.empty(repeats, dtype=np.float64)
    result = np.empty(signal.size, dtype=np.complex128)
    for i in range(repeats):
        start = time.perf_counter_ns()
        result = np.fft.fft(signal)
        times[i] = (time.perf_counter_ns() - start) * 1e-6
    return result, times


def measure_sleipnir(
    signal: np.ndarray, plan: C2CPlan, warmup: int, repeats: int
) -> tuple[np.ndarray, np.ndarray]:
    work = np.empty_like(signal)
    for _ in range(warmup):
        work[...] = signal
        plan.forward(work)

    times = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        work[...] = signal
        start = time.perf_counter_ns()
        plan.forward(work)
        times[i] = (time.perf_counter_ns() - start) * 1e-6
    return work.copy(), times


def print_results(
    signal: np.ndarray,
    numpy_result: np.ndarray,
    sleipnir_result: np.ndarray,
    numpy_times: np.ndarray,
    sleipnir_times: np.ndarray,
    plan_ms: float,
) -> None:
    error = np.abs(numpy_result - sleipnir_result)
    numpy_median = float(np.median(numpy_times))
    sleipnir_median = float(np.median(sleipnir_times))
    print(f"N={signal.size:,}, input dtype={signal.dtype}")
    print(f"NumPy output dtype={numpy_result.dtype}, Sleipnir output dtype={sleipnir_result.dtype}")
    print(f"Sleipnir plan creation: {plan_ms:.3f} ms (not included in FFT timings)")
    print(f"NumPy FFT median:       {numpy_median:.3f} ms")
    print(f"Sleipnir FFT median:    {sleipnir_median:.3f} ms")
    print(f"Sleipnir speedup:        {numpy_median / sleipnir_median:.2f}x")
    print(f"max |NumPy - Sleipnir|:  {np.max(error):.6e}")


def save_spectrum(
    signal: np.ndarray,
    numpy_result: np.ndarray,
    sleipnir_result: np.ndarray,
    numpy_times: np.ndarray,
    sleipnir_times: np.ndarray,
    output: Path,
    show: bool,
) -> None:
    sample_rate = 48000.0
    frequency = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    numpy_magnitude = np.abs(numpy_result[: frequency.size]) / signal.size
    sleipnir_magnitude = np.abs(sleipnir_result[: frequency.size]) / signal.size

    figure, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    axis.plot(frequency, numpy_magnitude, label="NumPy", linewidth=1.0, alpha=0.8)
    axis.plot(frequency, sleipnir_magnitude, "--", label="Sleipnir", linewidth=1.0)
    axis.set_title(
        f"FFT spectrum (N={signal.size:,}, {signal.dtype}; "
        f"NumPy {np.median(numpy_times):.3f} ms / "
        f"Sleipnir {np.median(sleipnir_times):.3f} ms)"
    )
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel("Magnitude")
    axis.set_xlim(0.0, sample_rate / 2.0)
    axis.grid(True, alpha=0.25)
    axis.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    print(f"Spectrum plot:           {output}")
    if show:
        plt.show()
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.n <= 0 or args.repeats <= 0 or args.warmup < 0:
        raise SystemExit("--n and --repeats must be positive; --warmup cannot be negative")
    dtype = np.dtype(args.dtype)
    signal = make_signal(args.n, dtype)

    numpy_result, numpy_times = measure_numpy(signal, args.warmup, args.repeats)
    start = time.perf_counter_ns()
    plan = C2CPlan(args.n, dtype)
    plan_ms = (time.perf_counter_ns() - start) * 1e-6
    try:
        sleipnir_result, sleipnir_times = measure_sleipnir(
            signal, plan, args.warmup, args.repeats
        )
    finally:
        plan.close()

    print_results(
        signal,
        numpy_result,
        sleipnir_result,
        numpy_times,
        sleipnir_times,
        plan_ms,
    )
    save_spectrum(
        signal,
        numpy_result,
        sleipnir_result,
        numpy_times,
        sleipnir_times,
        args.output,
        args.show,
    )


if __name__ == "__main__":
    main()
