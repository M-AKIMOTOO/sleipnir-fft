# Python / NumPy interface

Build the shared library first:

```sh
make build ODIN=/path/to/odin ODIN_OPT=speed MICROARCH=native LIB_MODE=shared
```

Then use the zero-copy reusable plan API:

```python
import numpy as np
from sleipnir_fft import C2CPlan

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
