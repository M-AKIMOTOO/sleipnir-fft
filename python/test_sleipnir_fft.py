import numpy as np

from sleipnirfft import C2CPlan, fft, ifft


def test_one_shot_matches_numpy_for_both_dtypes():
    for dtype in (np.complex64, np.complex128):
        source = (np.arange(256, dtype=np.float64) * 0.25).astype(dtype)
        source += (np.sin(np.arange(256)) * 0.5).astype(dtype) * 1j
        result = fft(source)
        expected = np.fft.fft(source)
        tolerance = 2e-5 if dtype == np.complex64 else 1e-12
        np.testing.assert_allclose(result, expected, rtol=tolerance, atol=tolerance)


def test_reusable_plan_is_in_place_and_round_trips():
    for dtype in (np.complex64, np.complex128):
        source = (np.random.default_rng(7).normal(size=1024) +
                  1j * np.random.default_rng(8).normal(size=1024)).astype(dtype)
        work = source.copy()
        with C2CPlan(work.size, dtype) as plan:
            assert plan.forward(work) is work
            assert plan.inverse(work) is work
        tolerance = 3e-5 if dtype == np.complex64 else 1e-12
        np.testing.assert_allclose(work, source, rtol=tolerance, atol=tolerance)


def test_ifft_matches_numpy():
    for dtype in (np.complex64, np.complex128):
        source = np.fft.fft(np.arange(128, dtype=dtype))
        result = ifft(source)
        np.testing.assert_allclose(result, np.fft.ifft(source), rtol=1e-5, atol=1e-5)
