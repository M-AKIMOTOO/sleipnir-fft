# Python / NumPy interface

Build the shared library first:

```sh
make build ODIN=/path/to/odin ODIN_OPT=speed MICROARCH=native LIB_MODE=shared
```

Then use the zero-copy reusable plan API:

```python
import numpy as np
from sleipnirfft import C2CPlan

x = np.ascontiguousarray(np.random.randn(4096) + 1j * np.random.randn(4096), dtype=np.complex64)
with C2CPlan(x.size, x.dtype) as plan:
    spectrum = plan.forward(x)  # x is modified in-place; spectrum is x
    plan.inverse(spectrum)       # normalized inverse, also in-place
```

For a one-shot call, `fft(x)` and `ifft(x)` return a separate contiguous array
and leave `x` unchanged.  Add this directory to `PYTHONPATH`, or copy the
module into an application package.


## Speed comparison and spectrum plot

The example below compares `np.fft.fft` with a reusable Sleipnir plan and
writes both spectra to a PNG:

```sh
PYTHONPATH=python python python/benchmark_spectrum.py --dtype complex64
```

Useful options are `--n`, `--repeats`, `--warmup`, and `--output`. NumPy's
standard FFT may promote `complex64` input to `complex128`; the script prints
both output dtypes so that this precision difference is visible.


### Real-to-complex (R2C)

Use `R2CPlan` for real input. It supports `float32` to `complex64` and
`float64` to `complex128`. The forward spectrum has `n // 2 + 1` bins.
Supplying both output buffers lets repeated calls avoid per-call allocations:

```python
import numpy as np
from sleipnirfft import R2CPlan

n = 4096
real = np.random.default_rng(7).normal(size=n).astype(np.float32)

with R2CPlan(n, np.float32) as plan:
    spectrum = np.empty(plan.output_size, dtype=plan.output_dtype)
    restored = np.empty(n, dtype=np.float32)
    plan.forward(real, spectrum)
    plan.inverse(spectrum, restored)
```

`forward` does not modify the real input. `inverse` writes the normalized real
signal to its output buffer. If the output argument is omitted, the method
allocates and returns an appropriately typed NumPy array.

For one-shot use, `rfft` and `irfft` are also available:

```python
from sleipnirfft import rfft, irfft

spectrum = rfft(real)
restored = irfft(spectrum, n=real.size)
```

For odd-length signals, pass `n` to `irfft`, because the positive-frequency
spectrum alone does not encode whether the original signal length was even or
odd.
